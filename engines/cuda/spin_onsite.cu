#include "device_operations.hpp"
#include "simulation_box_device.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace nep_adapters::cuda_backend {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kMaxSpinCompress = 4;
constexpr int kMaxSpinBasis = 8;
constexpr int kSpinDeg2Count = 6;
constexpr int kSpinDeg3Count = 10;
constexpr int kSpinDeg4Count = 15;
constexpr int kSpinChiralOReducedCount = 7;
constexpr int kSpinChiralHReducedCount = 9;

enum class SpinVirialMode : int {
  disabled,
  center_owned,
  cpu_atom_decomposition,
  center_and_neighbor_float_sink,
};

template <typename Launch>
void dispatch_spin_virial_mode(
    SpinVirialMode virial_mode,
    const Launch& launch) {
  switch (virial_mode) {
    case SpinVirialMode::disabled:
      launch(std::integral_constant<SpinVirialMode, SpinVirialMode::disabled>{});
      break;
    case SpinVirialMode::center_owned:
      launch(std::integral_constant<SpinVirialMode, SpinVirialMode::center_owned>{});
      break;
    case SpinVirialMode::cpu_atom_decomposition:
      launch(std::integral_constant<
             SpinVirialMode,
             SpinVirialMode::cpu_atom_decomposition>{});
      break;
    case SpinVirialMode::center_and_neighbor_float_sink:
      launch(std::integral_constant<
             SpinVirialMode,
             SpinVirialMode::center_and_neighbor_float_sink>{});
      break;
  }
}

template <typename Launch>
void dispatch_spin_channels(int channels, const Launch& launch) {
  switch (channels) {
    case 1:
      launch(std::integral_constant<int, 1>{});
      break;
    case 2:
      launch(std::integral_constant<int, 2>{});
      break;
    case 3:
      launch(std::integral_constant<int, 3>{});
      break;
    case 4:
      launch(std::integral_constant<int, 4>{});
      break;
    default:
      throw std::runtime_error("CUDA spin core supports 1 to 4 channels");
  }
}

template <typename Launch>
void dispatch_spin_lmax(int l_max, const Launch& launch) {
  switch (l_max) {
    case 0:
      launch(std::integral_constant<int, 0>{});
      break;
    case 1:
      launch(std::integral_constant<int, 1>{});
      break;
    case 2:
      launch(std::integral_constant<int, 2>{});
      break;
    case 3:
      launch(std::integral_constant<int, 3>{});
      break;
    case 4:
      launch(std::integral_constant<int, 4>{});
      break;
    default:
      throw std::runtime_error("CUDA spin core supports l_max from 0 to 4");
  }
}

template <int C, int LMax>
struct SpinStaticLayout {
  static constexpr int Rho0Offset = 2 + 4 * C;
  static constexpr int L1RdotOffset = Rho0Offset + C;
  static constexpr int L1CrossOffset = L1RdotOffset + C;
  static constexpr int L1StfOffset = L1CrossOffset + C;
  static constexpr int Angular2Offset =
      Rho0Offset + C + (LMax >= 1 ? 3 * C : 0);
  static constexpr int Angular3Offset =
      Angular2Offset + (LMax >= 2 ? C : 0);
  static constexpr int Angular4Offset =
      Angular3Offset + (LMax >= 3 ? C : 0);
  static constexpr int GeomOffset =
      Angular4Offset + (LMax >= 4 ? C : 0);
  static constexpr int Rho0DotOffset = GeomOffset + C;
  static constexpr int Raw1DotOffset = Rho0DotOffset + C;
};

// Exact Tucker reduction of the STF-projected pseudoscalar
// epsilon_abc Q_ad O_bef H_cdef.  The symmetric raw inputs have effective
// dimensions 5, 7, and 9, reducing the hot trilinear core from 192 to 50
// nonzero terms without changing the represented polynomial.
constexpr int kSpinChiralQohReducedCount = 50;
__device__ __constant__ unsigned short
kSpinChiralQohReducedPacked[kSpinChiralQohReducedCount] = {
    0, 32, 34, 49, 68, 70, 87, 100, 257, 259,
    272, 274, 289, 291, 304, 306, 325, 327, 328, 340,
    342, 357, 517, 519, 532, 534, 549, 552, 564, 577,
    579, 592, 594, 611, 774, 791, 804, 824, 832, 849,
    851, 866, 1041, 1056, 1073, 1075, 1094, 1109, 1111, 1124,
};
__device__ __constant__ float
kSpinChiralQohReducedCoeff[kSpinChiralQohReducedCount] = {
    1.0f, -2.0f, 1.0f, 2.0f, 1.0f, -1.0f, 2.0f, 0.5f, 1.0f, 1.0f,
    1.0f, 1.0f, -4.0f, -3.0f, -4.0f, -3.0f, -0.5f, -2.0f, -1.0f, -1.0f,
    -4.0f, -0.5f, 1.0f, -2.0f, 1.0f, 2.0f, -1.0f, -2.0f, 1.0f, -2.0f,
    6.0f, -4.0f, -8.0f, 2.0f, 1.0f, 1.0f, -0.5f, 1.0f, 1.0f, -2.0f,
    -4.0f, 1.0f, 1.0f, 2.0f, -2.0f, 1.0f, 1.0f, 1.0f, -2.0f, -0.5f,
};


void check_cuda(cudaError_t status, const char* message) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(message) + ": " + cudaGetErrorString(status));
  }
}

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool all_active(const std::vector<int>& mask, int num_types) {
  if (mask.empty()) {
    return true;
  }
  if (static_cast<int>(mask.size()) != num_types) {
    return false;
  }
  for (int value : mask) {
    if (value == 0) {
      return false;
    }
  }
  return true;
}

class PhaseTimer {
 public:
  explicit PhaseTimer(bool enabled) : enabled_(enabled) {
    if (enabled_) {
      check_cuda(cudaEventCreate(&mark_), "create spin phase timer mark");
      check_cuda(cudaEventCreate(&now_), "create spin phase timer event");
      check_cuda(cudaEventRecord(mark_), "record spin phase timer mark");
    }
  }

  ~PhaseTimer() {
    if (mark_ != nullptr) {
      cudaEventDestroy(mark_);
    }
    if (now_ != nullptr) {
      cudaEventDestroy(now_);
    }
  }

  void split(float& target_ms) {
    if (!enabled_) {
      return;
    }
    check_cuda(cudaEventRecord(now_), "record spin phase timer event");
    check_cuda(cudaEventSynchronize(now_), "synchronize spin phase timer event");
    check_cuda(
        cudaEventElapsedTime(&target_ms, mark_, now_),
        "measure spin phase time");
    check_cuda(cudaEventRecord(mark_), "advance spin phase timer mark");
  }

 private:
  bool enabled_ = false;
  cudaEvent_t mark_ = nullptr;
  cudaEvent_t now_ = nullptr;
};

// Keep both implementation fragments in this CUDA translation unit to preserve
// shared device constants, inlining, and kernel code generation.
#include "spin_onsite_descriptors.cuh"
#include "spin_onsite_forces.cuh"

void launch_spin_density_forces(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModelView& model_view,
    const DeviceWorkspaceView& view,
    SpinVirialMode virial_mode) {
  const auto launch_channels = [&](auto channel_tag) {
    constexpr int C = decltype(channel_tag)::value;
    const auto launch_lmax = [&](auto lmax_tag) {
      constexpr int LMax = decltype(lmax_tag)::value;
      const auto launch_virial = [&](auto virial_tag) {
        constexpr int AtomsPerWarp = 8;
        constexpr int EdgesPerAtomBatch = 4;
        constexpr SpinVirialMode VirialMode = decltype(virial_tag)::value;
        const int tile_blocks =
            (atom_count + AtomsPerWarp - 1) / AtomsPerWarp;
        accumulate_spin_density_forces_tile_f32<
            C,
            LMax,
            VirialMode,
            AtomsPerWarp,
            EdgesPerAtomBatch><<<tile_blocks, 32>>>(
                atom_count,
                static_cast<int>(view.atom_capacity),
                protocol.struct_descriptor_dim,
                protocol.num_types,
                protocol.spin_basis_size,
                static_cast<float>(protocol.spin_cutoff_radial),
                box,
                view.types,
                view.positions_soa3,
                view.spins_soa3,
                view.nn_radial,
                view.nl_radial_slot_major,
                view.fp,
                model_view.descriptor_coefficients,
                static_cast<int>(protocol.ordinary_descriptor_parameter_count),
                view.spin_density_rho0,
                view.spin_density_angular2,
                view.spin_density_angular3,
                view.spin_density_angular4,
                view.spin_density_geom,
                view.spin_density_rho0_dot,
                view.spin_density_raw1,
                view.spin_density_raw1_dot,
                view.force_soa3,
                view.mforce_soa3,
                view.virial_soa9,
                view.per_atom_virial_float_soa9);
      };
      dispatch_spin_virial_mode(virial_mode, launch_virial);
    };
    dispatch_spin_lmax(protocol.spin_l_max, launch_lmax);
  };
  dispatch_spin_channels(protocol.spin_compress, launch_channels);
}

void launch_spin_chiral_forces(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModelView& model_view,
    const DeviceWorkspaceView& view,
    SpinVirialMode virial_mode) {
  const SpinCoreLayout layout = make_spin_core_layout(protocol);
  const auto launch_channels = [&](auto channel_tag) {
    constexpr int C = decltype(channel_tag)::value;
    const auto launch_virial = [&](auto virial_tag) {
      constexpr int AtomsPerWarp = 8;
      constexpr int EdgesPerAtomBatch = 4;
      constexpr SpinVirialMode VirialMode = decltype(virial_tag)::value;
      const int tile_blocks =
          (atom_count + AtomsPerWarp - 1) / AtomsPerWarp;
      accumulate_spin_chiral_forces_tile_f32<
          C,
          VirialMode,
          AtomsPerWarp,
          EdgesPerAtomBatch><<<tile_blocks, 32>>>(
              atom_count,
              static_cast<int>(view.atom_capacity),
              protocol.struct_descriptor_dim,
              protocol.num_types,
              protocol.spin_basis_size,
              layout,
              static_cast<float>(protocol.spin_cutoff_radial),
              box,
              view.types,
              view.positions_soa3,
              view.spins_soa3,
              view.nn_radial,
              view.nl_radial_slot_major,
              view.fp,
              model_view.descriptor_coefficients,
              static_cast<int>(protocol.ordinary_descriptor_parameter_count),
              view.spin_density_geom,
              view.spin_density_raw1,
              view.spin_chiral_polar,
              view.spin_chiral_octupoles_raw,
              view.spin_chiral_hexadecapoles_raw,
              view.spin_chiral_chirals,
              view.force_soa3,
              view.mforce_soa3,
              view.virial_soa9,
              view.per_atom_virial_float_soa9);
    };
    dispatch_spin_virial_mode(virial_mode, launch_virial);
  };
  dispatch_spin_channels(protocol.spin_compress, launch_channels);
}

}  // namespace

void build_spin_descriptors_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace) {
  require(protocol.spin_mode != 0, "spin descriptor requires spin model");
  require(
      supports_cuda_spin_shape(protocol),
      "CUDA spin core requires 1 <= spin_compress <= 4, "
      "spin_compress <= spin_basis_size + 1 <= 8, and spin_l_max <= 4");
  require(protocol.spin_descriptor_dim > 0, "spin descriptor dimension must be positive");
  require(protocol.spin_descriptor_dim <= 96,
          "spin descriptor kernel supports spin descriptor dim <= 96");
  require(protocol.spin_compress > 0 &&
              protocol.spin_compress <= kMaxSpinCompress,
          "spin descriptor kernel supports spin_compress <= 4");
  require(protocol.spin_basis_size >= 0 &&
              protocol.spin_basis_size + 1 <= kMaxSpinBasis,
          "spin descriptor kernel supports spin_basis_size + 1 <= 8");
  require(protocol.spin_l_max >= 0 && protocol.spin_l_max <= 4,
          "spin descriptor kernel supports spin_l_max <= 4");
  require(all_active(protocol.spin_dof_type_active, protocol.num_types) &&
              all_active(protocol.spin_env_type_active, protocol.num_types),
          "CUDA spin path currently requires all types active for spin dof/env");
  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView view = workspace.view();
  const SpinCoreLayout layout = make_spin_core_layout(protocol);
  require(layout.descriptor_dim == protocol.spin_descriptor_dim,
          "spin descriptor layout does not match model protocol");
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(view.types != nullptr, "workspace missing types");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  require(view.spins_soa3 != nullptr, "workspace missing spins");
  require(view.nn_radial != nullptr, "workspace missing radial neighbor counts");
  require(view.nl_radial_slot_major != nullptr, "workspace missing radial neighbors");
  require(view.descriptors != nullptr, "workspace missing descriptors");
  require(view.spin_density_rho0 != nullptr, "workspace missing spin density rho0");
  require(view.spin_density_raw1 != nullptr, "workspace missing spin density raw1");
  require(protocol.spin_l_max < 2 || view.spin_density_angular2 != nullptr,
          "workspace missing spin density angular2");
  require(protocol.spin_l_max < 3 || view.spin_density_angular3 != nullptr,
          "workspace missing spin density angular3");
  require(protocol.spin_l_max < 4 || view.spin_density_angular4 != nullptr,
          "workspace missing spin density angular4");
  require(view.spin_density_geom != nullptr, "workspace missing spin density geom");
  require(view.spin_density_rho0_dot != nullptr,
          "workspace missing spin density rho0 dot");
  require(view.spin_density_raw1_dot != nullptr,
          "workspace missing spin density raw1 dot");
  require(protocol.spin_chiral == 0 || view.spin_chiral_polar != nullptr,
          "workspace missing spin chiral polar");
  require(protocol.spin_chiral == 0 || view.spin_chiral_octupoles_raw != nullptr,
          "workspace missing spin chiral octupoles raw");
  require(protocol.spin_chiral == 0 || view.spin_chiral_hexadecapoles_raw != nullptr,
          "workspace missing spin chiral hexadecapoles raw");
  require(protocol.spin_chiral == 0 || view.spin_chiral_chirals != nullptr,
          "workspace missing spin chiral chirals");
  require(model_view.descriptor_coefficients != nullptr,
          "model missing descriptor coefficients");
  require(model_view.descriptor_coefficients_count >= protocol.descriptor_parameter_count,
          "model descriptor coefficient buffer is too small");
  if (atom_count > 0) {
    const auto launch_channels = [&](auto channel_tag) {
      constexpr int C = decltype(channel_tag)::value;
      const auto launch_lmax = [&](auto lmax_tag) {
        constexpr int LMax = decltype(lmax_tag)::value;
        const auto launch_core = [&](auto chiral_tag) {
          constexpr bool Chiral = decltype(chiral_tag)::value;
          build_spin_descriptor_core_streaming<
              C,
              LMax,
              Chiral><<<atom_count, 128>>>(
                  atom_count,
                  static_cast<int>(view.atom_capacity),
                  protocol.struct_descriptor_dim,
                  protocol.num_types,
                  protocol.spin_basis_size,
                  static_cast<float>(protocol.spin_cutoff_radial),
                  box,
                  view.types,
                  view.positions_soa3,
                  view.spins_soa3,
                  view.nn_radial,
                  view.nl_radial_slot_major,
                  model_view.descriptor_coefficients,
                  static_cast<int>(protocol.ordinary_descriptor_parameter_count),
                  view.spin_density_rho0,
                  view.spin_density_raw1,
                  view.spin_density_angular2,
                  view.spin_density_angular3,
                  view.spin_density_angular4,
                  view.spin_density_geom,
                  view.spin_density_rho0_dot,
                  view.spin_density_raw1_dot,
                  view.spin_chiral_polar,
                  view.spin_chiral_octupoles_raw,
                  view.spin_chiral_hexadecapoles_raw,
                  view.descriptors);
        };
        if (protocol.spin_chiral != 0) {
          launch_core(std::true_type{});
          const int threads = 128;
          const int work_items = atom_count * C;
          const int blocks = (work_items + threads - 1) / threads;
          build_spin_chiral_descriptors_f32<C><<<blocks, threads>>>(
              atom_count,
              static_cast<int>(view.atom_capacity),
              protocol.struct_descriptor_dim,
              layout,
              view.spins_soa3,
              view.spin_density_geom,
              view.spin_density_raw1,
              view.spin_chiral_polar,
              view.spin_chiral_octupoles_raw,
              view.spin_chiral_hexadecapoles_raw,
              view.spin_chiral_chirals,
              view.descriptors);
        } else {
          launch_core(std::false_type{});
        }
      };
      dispatch_spin_lmax(protocol.spin_l_max, launch_lmax);
    };
    dispatch_spin_channels(protocol.spin_compress, launch_channels);
  }
  check_cuda(cudaGetLastError(), "build spin descriptors kernel launch failed");
}

static void accumulate_spin_onsite_mforces_impl(
    const ModelProtocol& protocol,
    int atom_count,
    DeviceWorkspace& workspace) {
  require(protocol.spin_mode != 0, "spin onsite mforce requires spin model");
  require(protocol.spin_descriptor_dim >= 2, "spin descriptor is missing onsite terms");
  const DeviceWorkspaceView view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(view.spins_soa3 != nullptr, "workspace missing spins");
  require(view.fp != nullptr, "workspace missing descriptor derivatives");
  require(view.mforce_soa3 != nullptr, "workspace missing mforce output");
  const int threads = 128;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0) {
    accumulate_spin_onsite_mforces<<<blocks, threads>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        protocol.struct_descriptor_dim,
        view.spins_soa3,
        view.fp,
        view.mforce_soa3);
  }
  check_cuda(cudaGetLastError(), "accumulate spin onsite mforces");
}

static void accumulate_spin_density_forces_impl(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    SpinVirialMode virial_mode) {
  require(protocol.spin_mode != 0, "spin density force requires spin model");
  require(protocol.spin_compress > 0 &&
              protocol.spin_compress <= kMaxSpinCompress,
          "spin density force kernel supports spin_compress <= 4");
  require(protocol.spin_basis_size >= 0 &&
              protocol.spin_basis_size + 1 <= kMaxSpinBasis,
          "spin density force kernel supports spin_basis_size + 1 <= 8");
  require(protocol.spin_l_max >= 0 && protocol.spin_l_max <= 4,
          "spin density force kernel supports spin_l_max <= 4");
  require(all_active(protocol.spin_dof_type_active, protocol.num_types) &&
              all_active(protocol.spin_env_type_active, protocol.num_types),
          "CUDA spin path currently requires all types active for spin dof/env");
  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(view.types != nullptr, "workspace missing types");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  require(view.spins_soa3 != nullptr, "workspace missing spins");
  require(view.nn_radial != nullptr, "workspace missing radial neighbor counts");
  require(view.nl_radial_slot_major != nullptr, "workspace missing radial neighbors");
  require(view.fp != nullptr, "workspace missing descriptor derivatives");
  require(view.force_soa3 != nullptr, "workspace missing forces");
  require(view.mforce_soa3 != nullptr, "workspace missing mforces");
  require(view.virial_soa9 != nullptr, "workspace missing virials");
  require(
      virial_mode != SpinVirialMode::center_and_neighbor_float_sink ||
          view.per_atom_virial_float_soa9 != nullptr,
      "workspace missing spin per-atom virial sink");
  require(view.spin_density_rho0 != nullptr, "workspace missing spin density rho0");
  require(view.spin_density_raw1 != nullptr, "workspace missing spin density raw1");
  require(protocol.spin_l_max < 2 || view.spin_density_angular2 != nullptr,
          "workspace missing spin density angular2");
  require(protocol.spin_l_max < 3 || view.spin_density_angular3 != nullptr,
          "workspace missing spin density angular3");
  require(protocol.spin_l_max < 4 || view.spin_density_angular4 != nullptr,
          "workspace missing spin density angular4");
  require(view.spin_density_geom != nullptr, "workspace missing spin density geom");
  require(view.spin_density_rho0_dot != nullptr,
          "workspace missing spin density rho0 dot");
  require(view.spin_density_raw1_dot != nullptr,
          "workspace missing spin density raw1 dot");
  require(model_view.descriptor_coefficients != nullptr,
          "model missing descriptor coefficients");
  require(model_view.descriptor_coefficients_count >= protocol.descriptor_parameter_count,
          "model descriptor coefficient buffer is too small");
  if (atom_count > 0) {
    launch_spin_density_forces(
        protocol, atom_count, box, model_view, view, virial_mode);
  }
  check_cuda(cudaGetLastError(), "accumulate spin density forces");
}

static void accumulate_spin_chiral_forces_impl(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    SpinVirialMode virial_mode) {
  require(protocol.spin_mode != 0, "spin chiral polar force requires spin model");
  require(protocol.spin_chiral != 0, "spin chiral polar force requires chiral model");
  require(
      supports_cuda_spin_shape(protocol),
      "CUDA chiral spin force received an unsupported spin shape");
  require(all_active(protocol.spin_dof_type_active, protocol.num_types) &&
              all_active(protocol.spin_env_type_active, protocol.num_types),
          "CUDA spin path currently requires all types active for spin dof/env");
  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(view.types != nullptr, "workspace missing types");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  require(view.spins_soa3 != nullptr, "workspace missing spins");
  require(view.nn_radial != nullptr, "workspace missing radial neighbor counts");
  require(view.nl_radial_slot_major != nullptr, "workspace missing radial neighbors");
  require(view.fp != nullptr, "workspace missing descriptor derivatives");
  require(view.force_soa3 != nullptr, "workspace missing forces");
  require(view.mforce_soa3 != nullptr, "workspace missing mforces");
  require(view.virial_soa9 != nullptr, "workspace missing virials");
  require(
      virial_mode != SpinVirialMode::center_and_neighbor_float_sink ||
          view.per_atom_virial_float_soa9 != nullptr,
      "workspace missing spin per-atom virial sink");
  require(view.spin_density_geom != nullptr, "workspace missing spin density geom");
  require(view.spin_density_raw1 != nullptr,
          "workspace missing spin density raw1");
  require(view.spin_chiral_polar != nullptr, "workspace missing spin chiral polar");
  require(view.spin_chiral_octupoles_raw != nullptr,
          "workspace missing spin chiral octupoles raw");
  require(view.spin_chiral_hexadecapoles_raw != nullptr,
          "workspace missing spin chiral hexadecapoles raw");
  require(view.spin_chiral_chirals != nullptr, "workspace missing spin chiral chirals");
  require(model_view.descriptor_coefficients != nullptr,
          "model missing descriptor coefficients");
  require(model_view.descriptor_coefficients_count >= protocol.descriptor_parameter_count,
          "model descriptor coefficient buffer is too small");
  if (atom_count > 0) {
    launch_spin_chiral_forces(
        protocol, atom_count, box, model_view, view, virial_mode);
  }
  check_cuda(cudaGetLastError(), "accumulate spin chiral polar forces");
}

void accumulate_spin_forces_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    VirialTarget virial_target,
    SpinForceTimings* timings) {
  require(protocol.spin_mode != 0, "spin force pipeline requires spin model");
  require(
      supports_cuda_spin_shape(protocol),
      "CUDA spin core requires 1 <= spin_compress <= 4, "
      "spin_compress <= spin_basis_size + 1 <= 8, and spin_l_max <= 4");
  SpinForceTimings ignored_timings;
  SpinForceTimings& measured = timings == nullptr ? ignored_timings : *timings;
  PhaseTimer timer(timings != nullptr);
  SpinVirialMode virial_mode = SpinVirialMode::disabled;
  switch (virial_target) {
    case VirialTarget::none:
      break;
    case VirialTarget::center_atom:
      virial_mode = SpinVirialMode::center_owned;
      break;
    case VirialTarget::neighbor_atom:
      virial_mode = SpinVirialMode::cpu_atom_decomposition;
      break;
    case VirialTarget::center_and_neighbor_float_sink:
      virial_mode = SpinVirialMode::center_and_neighbor_float_sink;
      break;
  }

  accumulate_spin_onsite_mforces_impl(protocol, atom_count, workspace);
  timer.split(measured.onsite_ms);

  accumulate_spin_density_forces_impl(
      protocol, atom_count, box, model, workspace, virial_mode);
  timer.split(measured.density_ms);

  if (protocol.spin_chiral != 0) {
    accumulate_spin_chiral_forces_impl(
        protocol, atom_count, box, model, workspace, virial_mode);
  }
  timer.split(measured.chiral_ms);
}

}  // namespace nep_adapters::cuda_backend
