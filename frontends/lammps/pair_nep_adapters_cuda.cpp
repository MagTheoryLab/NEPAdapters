#include "pair_nep_adapters_cuda.h"

#include "nep_adapters/api.h"
#include "nep_adapters/engines/cuda.hpp"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "lammps.h"
#include "neigh_request.h"
#include "neighbor.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <type_traits>

#ifdef LMP_KOKKOS
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#endif

using namespace LAMMPS_NS;

namespace {

#ifdef LMP_KOKKOS
template <class RawView, class VatomView, class CvatomView>
struct PackLammpsPerAtomVirial {
  RawView raw9;
  VatomView vatom;
  CvatomView cvatom;
  int write_vatom;
  int write_cvatom;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int atom_index) const {
    const std::size_t offset = 9 * static_cast<std::size_t>(atom_index);
    if (write_vatom) {
      vatom(atom_index, 0) = raw9(offset + 0);
      vatom(atom_index, 1) = raw9(offset + 1);
      vatom(atom_index, 2) = raw9(offset + 2);
      vatom(atom_index, 3) = 0.5 * (raw9(offset + 3) + raw9(offset + 6));
      vatom(atom_index, 4) = 0.5 * (raw9(offset + 4) + raw9(offset + 7));
      vatom(atom_index, 5) = 0.5 * (raw9(offset + 5) + raw9(offset + 8));
    }
    if (write_cvatom) {
      for (int component = 0; component < 9; ++component) {
        cvatom(atom_index, component) = raw9(offset + component);
      }
    }
  }
};

struct ReplayHeader {
  char magic[8];
  std::int32_t version;
  std::int32_t nlocal;
  std::int32_t nall;
  std::int32_t inum;
  std::int32_t max_neighbors;
  std::int32_t neighbor_rows;
  std::int32_t numneigh_length;
  std::int32_t neighbor_atom_stride;
  std::int32_t neighbor_slot_stride;
  std::int32_t type_map_length;
  std::int32_t position_atom_stride;
  std::int32_t position_component_stride;
  std::int32_t force_atom_stride;
  std::int32_t force_component_stride;
  std::int64_t ilist_count;
  std::int64_t numneigh_count;
  std::int64_t neighbor_count;
  std::int64_t type_count;
  std::int64_t type_map_count;
  std::int64_t position_count;
  std::int64_t force_count;
};

int env_int(const char* name, int fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const long value = std::strtol(raw, &end, 10);
  return end == raw ? fallback : static_cast<int>(value);
}

class HostPairProfiler {
 public:
  HostPairProfiler(int rank, int nlocal) : nlocal_(nlocal) {
    const char* raw = std::getenv("NEP_ADAPTERS_PROFILE_PAIR");
    if (raw == nullptr || raw[0] == '\0' ||
        rank != env_int("NEP_ADAPTERS_PROFILE_PAIR_RANK", 0)) {
      return;
    }
    static int samples = 0;
    const int max_samples = env_int("NEP_ADAPTERS_PROFILE_PAIR", 5);
    if (max_samples > 0 && samples >= max_samples) {
      return;
    }
    sample_ = ++samples;
    enabled_ = true;
    mark_ = Clock::now();
  }

  void split(double& target_ms) {
    if (!enabled_) {
      return;
    }
    const auto now = Clock::now();
    target_ms = std::chrono::duration<double, std::milli>(now - mark_).count();
    mark_ = now;
  }

  void print(
      double engine_ms,
      double pack_ms,
      double mark_ms,
      double reverse_ms,
      double tally_ms) const {
    if (!enabled_) {
      return;
    }
    std::fprintf(
        stderr,
        "NEPA_PAIR_HOST_PROFILE sample=%d nlocal=%d engine_ms=%.3f "
        "pack_ms=%.3f mark_ms=%.3f reverse_ms=%.3f tally_ms=%.3f "
        "total_ms=%.3f\n",
        sample_,
        nlocal_,
        engine_ms,
        pack_ms,
        mark_ms,
        reverse_ms,
        tally_ms,
        engine_ms + pack_ms + mark_ms + reverse_ms + tally_ms);
  }

 private:
  using Clock = std::chrono::steady_clock;
  bool enabled_ = false;
  int sample_ = 0;
  int nlocal_ = 0;
  Clock::time_point mark_;
};

template <typename T>
void write_binary(std::ofstream& out, const T* data, std::size_t count) {
  if (count == 0) {
    return;
  }
  out.write(
      reinterpret_cast<const char*>(data),
      static_cast<std::streamsize>(count * sizeof(T)));
}

template <class View>
std::array<int, 2> checked_strides2(
    const View& view,
    Error* error,
    const std::string& label) {
  std::size_t raw[2] = {};
  view.stride(raw);
  const std::size_t max_int =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (raw[0] > max_int || raw[1] > max_int) {
    error->all(
        FLERR,
        (label + ": Kokkos view stride exceeds int range").c_str());
  }
  return {static_cast<int>(raw[0]), static_cast<int>(raw[1])};
}

template <
    class IlistView,
    class NumneighView,
    class NeighborView,
    class TypeView,
    class TypeMapView,
    class PositionView,
    class ForceView>
void dump_kokkos_replay_once(
    Error* error,
    Comm* comm,
    const std::string& label,
    int nlocal,
    int nall,
    int inum,
    int max_neighbors,
    int neighbor_rows,
    int numneigh_length,
    const std::array<int, 2>& neighbor_stride,
    const std::array<int, 2>& position_stride,
    const std::array<int, 2>& force_stride,
    const IlistView& ilist,
    const NumneighView& numneigh,
    const NeighborView& neighbors,
    const TypeView& types,
    const TypeMapView& type_map,
    int type_map_length,
    const PositionView& positions,
    const ForceView& forces) {
  const char* path = std::getenv("NEP_ADAPTERS_DUMP_DEVICE_REPLAY");
  if (path == nullptr || path[0] == '\0') {
    return;
  }
  static bool dumped = false;
  if (dumped || comm->me != 0) {
    return;
  }
  dumped = true;
  Kokkos::fence();

  const auto h_ilist = Kokkos::create_mirror_view_and_copy(LMPHostType(), ilist);
  const auto h_numneigh =
      Kokkos::create_mirror_view_and_copy(LMPHostType(), numneigh);
  const auto h_neighbors =
      Kokkos::create_mirror_view_and_copy(LMPHostType(), neighbors);
  const auto h_types = Kokkos::create_mirror_view_and_copy(LMPHostType(), types);
  const auto h_type_map =
      Kokkos::create_mirror_view_and_copy(LMPHostType(), type_map);
  const auto h_positions =
      Kokkos::create_mirror_view_and_copy(LMPHostType(), positions);

  ReplayHeader header{};
  std::memcpy(header.magic, "NEPAKKR1", sizeof(header.magic));
  header.version = 1;
  header.nlocal = nlocal;
  header.nall = nall;
  header.inum = inum;
  header.max_neighbors = max_neighbors;
  header.neighbor_rows = neighbor_rows;
  header.numneigh_length = numneigh_length;
  header.neighbor_atom_stride = neighbor_stride[0];
  header.neighbor_slot_stride = neighbor_stride[1];
  header.type_map_length = type_map_length;
  header.position_atom_stride = position_stride[0];
  header.position_component_stride = position_stride[1];
  header.force_atom_stride = force_stride[0];
  header.force_component_stride = force_stride[1];
  header.ilist_count = static_cast<std::int64_t>(h_ilist.span());
  header.numneigh_count = static_cast<std::int64_t>(h_numneigh.span());
  header.neighbor_count = static_cast<std::int64_t>(h_neighbors.span());
  header.type_count = static_cast<std::int64_t>(h_types.span());
  header.type_map_count = static_cast<std::int64_t>(h_type_map.span());
  header.position_count = static_cast<std::int64_t>(h_positions.span());
  header.force_count = static_cast<std::int64_t>(forces.span());

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    error->all(
        FLERR,
        (label + ": cannot open NEP_ADAPTERS_DUMP_DEVICE_REPLAY path").c_str());
  }
  write_binary(out, &header, 1);
  write_binary(out, h_ilist.data(), h_ilist.span());
  write_binary(out, h_numneigh.data(), h_numneigh.span());
  write_binary(out, h_neighbors.data(), h_neighbors.span());
  write_binary(out, h_types.data(), h_types.span());
  write_binary(out, h_type_map.data(), h_type_map.span());
  write_binary(out, h_positions.data(), h_positions.span());
  if (!out) {
    error->all(FLERR, (label + ": failed to write replay dump").c_str());
  }
}
#endif

}  // namespace

PairNEPAdaptersCUDA::PairNEPAdaptersCUDA(LAMMPS* lmp)
    : PairNEPAdaptersCommon(
          lmp,
          "nep/gpu",
          "cuda",
          "NEPAdapters GPU",
          &nep_adapters::register_cuda_engine) {
#ifdef LMP_KOKKOS
  kokkosable = 1;
  execution_space = ExecutionSpaceFromDevice<LMPDeviceType>::space;
  datamask_read = X_MASK | TYPE_MASK;
  datamask_modify = F_MASK;
  comm_reverse_off = 3;
  reverse_comm_device = 1;
#endif
}

PairNEPAdaptersCUDA::~PairNEPAdaptersCUDA() {
#ifdef LMP_KOKKOS
  if (copymode || memoryKK == nullptr) {
    return;
  }
  memoryKK->destroy_kokkos(k_eatom_, eatom);
  memoryKK->destroy_kokkos(k_vatom_, vatom);
  memoryKK->destroy_kokkos(k_cvatom_, cvatom);
#endif
}

void PairNEPAdaptersCUDA::coeff(int narg, char** arg) {
  PairNEPAdaptersCommon::coeff(narg, arg);
  NepaModelInfo info{};
  const NepaStatus status = nepa_model_info(model_, &info);
  if (status != NEPA_STATUS_OK) {
    error->all(FLERR, nepa_status_message(status));
  }
  charge_model_ = (info.capabilities & NEPA_CAPABILITY_CHARGE) != 0;
#ifdef LMP_KOKKOS
  datamask_read = X_MASK | TYPE_MASK | (spin_model_ ? SP_MASK : EMPTY_MASK);
  datamask_modify = F_MASK | (spin_model_ ? FM_MASK : EMPTY_MASK);
  comm_reverse_off = spin_model_ ? 6 : 3;
  d_type_map_ = Kokkos::View<int*, LMPDeviceType>();
  d_type_map_length_ = 0;
#endif
}

void PairNEPAdaptersCUDA::init_style() {
  PairNEPAdaptersCommon::init_style();
#ifdef LMP_KOKKOS
  if (lmp->kokkos != nullptr) {
    if (execution_space != Device) {
      return;
    }
    auto* request = neighbor->find_request(this);
    request->set_kokkos_device(1);
    request->enable_full();
  }
#endif
}

void PairNEPAdaptersCUDA::compute(int eflag_in, int vflag_in) {
#ifndef LMP_KOKKOS
  (void)eflag_in;
  (void)vflag_in;
  error->all(
      FLERR,
      "NEPAdapters GPU pair requires a CUDA-enabled LAMMPS Kokkos build");
#else
  if (lmp->kokkos == nullptr || lmp->atomKK == nullptr) {
    error->all(
        FLERR,
        "NEPAdapters GPU pair requires Kokkos device atom storage");
  }
  if (model_ == nullptr) {
    const std::string message =
        label_ + ": model not initialized; check pair_coeff";
    error->all(FLERR, message.c_str());
  }

  static_assert(
      std::is_same<X_FLOAT, double>::value,
      "NEPAdapters CUDA Kokkos path requires double precision positions");
  static_assert(
      std::is_same<F_FLOAT, double>::value,
      "NEPAdapters CUDA Kokkos path requires double precision forces");
  static_assert(
      std::is_same<E_FLOAT, double>::value,
      "NEPAdapters CUDA Kokkos path requires double precision energy");

  if (execution_space != Device) {
    const std::string message =
        label_ + ": Kokkos CUDA pair requires a GPU execution space";
    error->all(FLERR, message.c_str());
  }
  if (charge_model_) {
    const std::string message =
        label_ + ": qNEP is currently supported by the CUDA batch API only; "
        "the LAMMPS Kokkos path needs a separate ghost/charge data flow";
    error->all(FLERR, message.c_str());
  }

  ev_init(eflag_in, vflag_in, 0);
  const bool want_global_tally = eflag_in || vflag_in;

  const int nlocal = atom->nlocal;
  const int nall = atom->nlocal + atom->nghost;
  if (nlocal <= 0) {
    return;
  }
  if (list == nullptr) {
    const std::string message = label_ + ": missing LAMMPS neighbor list";
    error->all(FLERR, message.c_str());
  }
  if (list->inum != nlocal) {
    const std::string message =
        label_ + ": Kokkos full neighbor list must cover exactly nlocal atoms";
    error->all(FLERR, message.c_str());
  }
  auto* atom_kk = lmp->atomKK;
  ensure_kokkos_buffers();
  ensure_kokkos_type_map();
  ensure_kokkos_tally_buffers();

  atom_kk->sync(execution_space, datamask_read);
  auto x = atom_kk->k_x.template view<LMPDeviceType>();
  auto f = atom_kk->k_f.template view<LMPDeviceType>();
  auto type = atom_kk->k_type.template view<LMPDeviceType>();
  auto sp = atom_kk->k_sp.template view<LMPDeviceType>();
  auto fm = atom_kk->k_fm.template view<LMPDeviceType>();

  auto* k_list = static_cast<NeighListKokkos<LMPDeviceType>*>(list);
  auto neighbors = k_list->d_neighbors;
  auto ilist = k_list->d_ilist;
  auto numneigh = k_list->d_numneigh;

  const std::array<int, 2> f_stride =
      checked_strides2(f, error, label_);
  const std::array<int, 2> position_stride =
      checked_strides2(x, error, label_);
  const std::array<int, 2> spin_stride =
      spin_model_ ? checked_strides2(sp, error, label_) : std::array<int, 2>{0, 0};
  const std::array<int, 2> mforce_stride =
      spin_model_ ? checked_strides2(fm, error, label_) : std::array<int, 2>{0, 0};
  const std::array<int, 2> neighbor_stride =
      checked_strides2(neighbors, error, label_);
  if (neighbors.extent(1) >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    const std::string message =
        label_ + ": Kokkos neighbor capacity exceeds int range";
    error->all(FLERR, message.c_str());
  }
  if (neighbors.extent(0) >
          static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      numneigh.extent(0) >
          static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    const std::string message =
        label_ + ": Kokkos neighbor row count exceeds int range";
    error->all(FLERR, message.c_str());
  }

  const int neighbor_capacity = static_cast<int>(neighbors.extent(1));
  NepaLammpsDeviceNeighborInput input{};
  input.nlocal = nlocal;
  input.nall = nall;
  input.inum = list->inum;
  input.max_neighbors = neighbor_capacity;
  input.neighbor_rows = static_cast<int>(neighbors.extent(0));
  input.numneigh_length = static_cast<int>(numneigh.extent(0));
  input.ilist = ilist.data();
  input.numneigh = numneigh.data();
  input.neighbors = neighbors.data();
  input.neighbor_owner = nullptr;
  input.neighbor_atom_stride = neighbor_stride[0];
  input.neighbor_slot_stride = neighbor_stride[1];
  input.types = type.data();
  input.type_map = d_type_map_.data();
  input.type_map_length = d_type_map_length_;
  input.positions = x.data();
  input.position_atom_stride = position_stride[0];
  input.position_component_stride = position_stride[1];
  input.spins = spin_model_ ? sp.data() : nullptr;
  input.spin_atom_stride = spin_stride[0];
  input.spin_component_stride = spin_stride[1];

  NepaLammpsDeviceNeighborResult result{};
  result.total_potential = want_global_tally ? d_total_potential_.data() : nullptr;
  result.total_virial6 = want_global_tally ? d_total_virial6_.data() : nullptr;
  result.potential_per_atom = eflag_atom ? d_eatom_.data() : nullptr;
  result.forces = f.data();
  result.force_atom_stride = f_stride[0];
  result.force_component_stride = f_stride[1];
  result.mforces = spin_model_ ? fm.data() : nullptr;
  result.mforce_atom_stride = mforce_stride[0];
  result.mforce_component_stride = mforce_stride[1];
  if (vflag_atom || cvflag_atom) {
    result.virials_per_atom9 = d_lammps_raw9_.data();
    result.virial_atom_stride = 9;
    result.virial_component_stride = 1;
  }

  dump_kokkos_replay_once(
      error,
      comm,
      label_,
      nlocal,
      nall,
      list->inum,
      neighbor_capacity,
      input.neighbor_rows,
      input.numneigh_length,
      neighbor_stride,
      position_stride,
      f_stride,
      ilist,
      numneigh,
      neighbors,
      type,
      d_type_map_,
      d_type_map_length_,
      x,
      f);

  HostPairProfiler profiler(comm->me, nlocal);
  double engine_ms = 0.0;
  double pack_ms = 0.0;
  double mark_ms = 0.0;
  double reverse_ms = 0.0;
  double tally_ms = 0.0;
  const NepaStatus status =
      nepa_find_force_lammps_device_neighbors(model_, &input, &result);
  profiler.split(engine_ms);
  if (status != NEPA_STATUS_OK) {
    std::string message = nepa_status_message(status);
    const char* detail = nepa_last_error_message();
    if (detail != nullptr && detail[0] != '\0') {
      message += ": ";
      message += detail;
    }
    message += " nlocal=" + std::to_string(nlocal);
    message += " nall=" + std::to_string(nall);
    message += " tag_enable=" + std::to_string(atom->tag_enable ? 1 : 0);
    error->all(FLERR, message.c_str());
  }

  atom_kk->modified(execution_space, datamask_modify);
  profiler.split(mark_ms);
  // The n2 virial decomposition assigns some terms to ghost atoms.  In the
  // normal Kokkos newton-off configuration LAMMPS does not reverse vatom, so
  // carry raw9 with the existing device force/mforce reverse communication.
  reverse_per_atom_virial_ =
      !force->newton && (vflag_atom || cvflag_atom);
  if (nall > nlocal) {
    const bool reverse_force_comm_on_device =
        lmp->kokkos != nullptr && lmp->kokkos->reverse_comm_classic == 0 &&
        lmp->kokkos->reverse_comm_on_host == 0 &&
        lmp->kokkos->reverse_pair_comm_classic == 0;
    if (reverse_force_comm_on_device) {
      const int reverse_width =
          comm_reverse_off + (reverse_per_atom_virial_ ? 9 : 0);
      comm->reverse_comm(this, reverse_width);
    } else {
      if (reverse_per_atom_virial_) {
        error->all(
            FLERR,
            (label_ +
             ": per-atom virial with Kokkos newton off requires comm device")
                .c_str());
      }
      comm->reverse_comm();
      atom_kk->sync(execution_space, datamask_modify);
      Kokkos::fence();
    }
    atom_kk->modified(execution_space, datamask_modify);
  }
  reverse_per_atom_virial_ = false;
  profiler.split(reverse_ms);

  pack_kokkos_per_atom_tallies();
  profiler.split(pack_ms);

  if (eflag_in) {
    const auto total_potential =
        Kokkos::create_mirror_view_and_copy(LMPHostType(), d_total_potential_);
    eng_vdwl += total_potential(0);
  }
  if (vflag_in) {
    const auto total_virial =
        Kokkos::create_mirror_view_and_copy(LMPHostType(), d_total_virial6_);
    for (int component = 0; component < 6; ++component) {
      virial[component] += total_virial(component);
    }
  }
  profiler.split(tally_ms);
  profiler.print(engine_ms, pack_ms, mark_ms, reverse_ms, tally_ms);
#endif
}

#ifdef LMP_KOKKOS
int PairNEPAdaptersCUDA::pack_reverse_comm_kokkos(
    int n,
    int first,
    DAT::tdual_xfloat_1d& buf) {
  auto f = lmp->atomKK->k_f.template view<LMPDeviceType>();
  auto fm = lmp->atomKK->k_fm.template view<LMPDeviceType>();
  const auto raw9 = d_lammps_raw9_;
  auto out = buf.template view<LMPDeviceType>();
  const int communicate_mforce = spin_model_ ? 1 : 0;
  const int force_width = communicate_mforce ? 6 : 3;
  const int communicate_virial = reverse_per_atom_virial_ ? 1 : 0;
  const int width = force_width + (communicate_virial ? 9 : 0);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<LMPDeviceType>(0, n),
      KOKKOS_LAMBDA(const int i) {
        const int atom = first + i;
        const int offset = width * i;
        out(offset) = f(atom, 0);
        out(offset + 1) = f(atom, 1);
        out(offset + 2) = f(atom, 2);
        if (communicate_mforce) {
          out(offset + 3) = fm(atom, 0);
          out(offset + 4) = fm(atom, 1);
          out(offset + 5) = fm(atom, 2);
        }
        if (communicate_virial) {
          const int virial_offset = 9 * atom;
          for (int component = 0; component < 9; ++component) {
            out(offset + force_width + component) =
                raw9(virial_offset + component);
          }
        }
      });
  return width * n;
}

void PairNEPAdaptersCUDA::unpack_reverse_comm_kokkos(
    int n,
    DAT::tdual_int_1d list,
    DAT::tdual_xfloat_1d& buf) {
  auto f = lmp->atomKK->k_f.template view<LMPDeviceType>();
  auto fm = lmp->atomKK->k_fm.template view<LMPDeviceType>();
  const auto raw9 = d_lammps_raw9_;
  auto in = buf.template view<LMPDeviceType>();
  auto sendlist = list.template view<LMPDeviceType>();
  const int communicate_mforce = spin_model_ ? 1 : 0;
  const int force_width = communicate_mforce ? 6 : 3;
  const int communicate_virial = reverse_per_atom_virial_ ? 1 : 0;
  const int width = force_width + (communicate_virial ? 9 : 0);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<LMPDeviceType>(0, n),
      KOKKOS_LAMBDA(const int i) {
        const int atom = sendlist(i);
        const int offset = width * i;
        Kokkos::atomic_add(&f(atom, 0), in(offset));
        Kokkos::atomic_add(&f(atom, 1), in(offset + 1));
        Kokkos::atomic_add(&f(atom, 2), in(offset + 2));
        if (communicate_mforce) {
          Kokkos::atomic_add(&fm(atom, 0), in(offset + 3));
          Kokkos::atomic_add(&fm(atom, 1), in(offset + 4));
          Kokkos::atomic_add(&fm(atom, 2), in(offset + 5));
        }
        if (communicate_virial) {
          const int virial_offset = 9 * atom;
          for (int component = 0; component < 9; ++component) {
            Kokkos::atomic_add(
                &raw9(virial_offset + component),
                in(offset + force_width + component));
          }
        }
      });
}

void PairNEPAdaptersCUDA::ensure_kokkos_buffers() {
  if (d_total_potential_.extent(0) < 1) {
    d_total_potential_ =
        Kokkos::View<double*, LMPDeviceType>("nepa:total_potential", 1);
  }
  if (d_total_virial6_.extent(0) < 6) {
    d_total_virial6_ =
        Kokkos::View<double*, LMPDeviceType>("nepa:total_virial6", 6);
  }
}

void PairNEPAdaptersCUDA::ensure_kokkos_type_map() {
  const int length = atom->ntypes + 1;
  if (d_type_map_length_ == length &&
      d_type_map_.extent(0) >= static_cast<std::size_t>(length)) {
    return;
  }

  d_type_map_ = Kokkos::View<int*, LMPDeviceType>("nepa:type_map", length);
  auto host_map = Kokkos::create_mirror_view(d_type_map_);
  for (int type = 0; type < length; ++type) {
    host_map(type) = type_map_[type];
  }
  Kokkos::deep_copy(d_type_map_, host_map);
  d_type_map_length_ = length;
}

void PairNEPAdaptersCUDA::ensure_kokkos_tally_buffers() {
  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom_, eatom);
    memoryKK->create_kokkos(k_eatom_, eatom, maxeatom, "nepa:eatom");
    d_eatom_ = k_eatom_.template view<LMPDeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom_, vatom);
    memoryKK->create_kokkos(k_vatom_, vatom, maxvatom, "nepa:vatom");
    d_vatom_ = k_vatom_.template view<LMPDeviceType>();
  }
  if (cvflag_atom) {
    memoryKK->destroy_kokkos(k_cvatom_, cvatom);
    memoryKK->create_kokkos(k_cvatom_, cvatom, maxcvatom, "nepa:cvatom");
    d_cvatom_ = k_cvatom_.template view<LMPDeviceType>();
  }
  if ((vflag_atom || cvflag_atom) &&
      d_lammps_raw9_.extent(0) < 9 * static_cast<std::size_t>(atom->nmax)) {
    d_lammps_raw9_ =
        Kokkos::View<double*, LMPDeviceType>("nepa:lammps_raw9", 9 * atom->nmax);
  }
}

void PairNEPAdaptersCUDA::pack_kokkos_per_atom_tallies() {
  if (eflag_atom) {
    k_eatom_.template modify<LMPDeviceType>();
    k_eatom_.template sync<LMPHostType>();
  }
  if (!vflag_atom && !cvflag_atom) {
    return;
  }

  const int nvirial = force->newton ? atom->nlocal + atom->nghost : atom->nlocal;
  const int write_vatom = vflag_atom ? 1 : 0;
  const int write_cvatom = cvflag_atom ? 1 : 0;
  const auto raw9 = d_lammps_raw9_;
  const auto vatom_view = d_vatom_;
  const auto cvatom_view = d_cvatom_;
  Kokkos::parallel_for(
      "nepa:pack_per_atom_virial",
      Kokkos::RangePolicy<LMPDeviceType>(0, nvirial),
      PackLammpsPerAtomVirial<
          decltype(raw9),
          decltype(vatom_view),
          decltype(cvatom_view)>{
          raw9, vatom_view, cvatom_view, write_vatom, write_cvatom});
  if (vflag_atom) {
    k_vatom_.template modify<LMPDeviceType>();
    k_vatom_.template sync<LMPHostType>();
  }
  if (cvflag_atom) {
    k_cvatom_.template modify<LMPDeviceType>();
    k_cvatom_.template sync<LMPHostType>();
  }
}
#endif
