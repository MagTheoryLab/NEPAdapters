#include "pair_nep_adapters_common.h"

#include "nep_adapters/api.h"
#include "nep_adapters/detail/model_file.hpp"
#include "nep_adapters/virial_order.hpp"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "lmptype.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "utils.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace LAMMPS_NS;

namespace {

std::vector<std::string> read_nep_elements(const std::string& model_path) {
  std::ifstream input = nep_adapters::detail::open_model_input(model_path);
  if (!input.is_open()) {
    return {};
  }

  std::string line;
  std::getline(input, line);
  std::istringstream stream(line);
  std::string tag;
  int num_types = 0;
  stream >> tag >> num_types;
  if (!stream || num_types <= 0) {
    return {};
  }

  std::vector<std::string> elements(static_cast<std::size_t>(num_types));
  for (int i = 0; i < num_types; ++i) {
    if (!(stream >> elements[static_cast<std::size_t>(i)])) {
      return {};
    }
  }
  return elements;
}

}  // namespace

PairNEPAdaptersCommon::PairNEPAdaptersCommon(
    LAMMPS* lmp,
    const char* style_name,
    const char* engine_name,
    const char* label,
    PairNEPAdaptersRegisterEngine register_engine)
    : Pair(lmp),
      style_name_(style_name),
      engine_name_(engine_name),
      label_(label),
      register_engine_(register_engine) {
  centroidstressflag = CENTROID_AVAIL;
  restartinfo = 0;
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;
  one_coeff = 1;
  single_enable = 0;
}

PairNEPAdaptersCommon::~PairNEPAdaptersCommon() {
  if (copymode) {
    return;
  }

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    delete[] type_map_;
    type_map_ = nullptr;
  }

  nepa_free_model(model_);
  model_ = nullptr;
}

void PairNEPAdaptersCommon::allocate() {
  const int ntypes = atom->ntypes;
  memory->create(setflag, ntypes + 1, ntypes + 1, "pair:setflag");
  memory->create(cutsq, ntypes + 1, ntypes + 1, "pair:cutsq");
  type_map_ = new int[ntypes + 1];
  type_map_[0] = -1;

  for (int i = 1; i <= ntypes; ++i) {
    type_map_[i] = -1;
    for (int j = 1; j <= ntypes; ++j) {
      setflag[i][j] = 1;
      cutsq[i][j] = 0.0;
    }
  }
  allocated = 1;
}

void PairNEPAdaptersCommon::settings(int narg, char**) {
  if (narg != 0) {
    const std::string message =
        "Illegal pair_style command; usage: pair_style " + style_name_;
    error->all(FLERR, message.c_str());
  }
}

void PairNEPAdaptersCommon::coeff(int narg, char** arg) {
  if (!allocated) {
    allocate();
  }
  if (narg != 3 + atom->ntypes) {
    error->all(
        FLERR,
        "Incorrect args for pair_coeff; usage: pair_coeff * * model.nep Element ...");
  }
  if (std::string(arg[0]) != "*" || std::string(arg[1]) != "*") {
    const std::string message =
        "pair_style " + style_name_ + " requires pair_coeff * *";
    error->all(FLERR, message.c_str());
  }

  model_filename_ = arg[2];
  const std::string model_path = utils::get_potential_file_path(model_filename_);
  read_type_map(model_path, narg, arg);
  load_model(model_path);
  log_loaded_model(model_path);

  for (int i = 1; i <= atom->ntypes; ++i) {
    for (int j = 1; j <= atom->ntypes; ++j) {
      cutsq[i][j] = cutoff_ * cutoff_;
    }
  }
}

void PairNEPAdaptersCommon::read_type_map(
    const std::string& model_path,
    int,
    char** arg) {
  const std::vector<std::string> elements = read_nep_elements(model_path);
  if (elements.empty()) {
    const std::string message =
        label_ + ": failed to read element symbols from NEP file: " + model_path;
    error->all(FLERR, message.c_str());
  }
  model_elements_ = elements;

  for (int type = 1; type <= atom->ntypes; ++type) {
    const char* symbol = arg[2 + type];
    const auto found = std::find(elements.begin(), elements.end(), symbol);
    if (found == elements.end()) {
      const std::string message =
          label_ + ": pair_coeff element is absent from NEP file";
      error->all(FLERR, message.c_str());
    }
    type_map_[type] = static_cast<int>(std::distance(elements.begin(), found));
  }
}

void PairNEPAdaptersCommon::load_model(const std::string& model_path) {
  if (register_engine_ == nullptr || !register_engine_()) {
    const std::string message = label_ + ": failed to register engine";
    error->all(FLERR, message.c_str());
  }

  nepa_free_model(model_);
  model_ = nullptr;
  NepaStatus status = nepa_load_model(engine_name_.c_str(), model_path.c_str(), &model_);
  if (status != NEPA_STATUS_OK || model_ == nullptr) {
    error->all(FLERR, nepa_status_message(status));
  }

  NepaModelInfo info{};
  status = nepa_model_info(model_, &info);
  if (status != NEPA_STATUS_OK || info.cutoff_max <= 0.0) {
    const std::string message = label_ + ": failed to query model cutoff";
    error->all(FLERR, message.c_str());
  }
  cutoff_ = info.cutoff_max;
  spin_model_ = (info.capabilities & NEPA_CAPABILITY_SPIN) != 0;
  // Full NEP neighbor lists accumulate force (and, for spin models, magnetic
  // force) contributions on periodic/MPI ghost atoms even with Newton off.
  // Reserve enough classic reverse-communication storage for f, fm, and raw9.
  comm_reverse_off = spin_model_ ? 15 : 12;
}

void PairNEPAdaptersCommon::log_loaded_model(
    const std::string& model_path) const {
  if (comm->me != 0) {
    return;
  }

  std::ostringstream elements;
  for (std::size_t index = 0; index < model_elements_.size(); ++index) {
    if (index != 0) {
      elements << ' ';
    }
    elements << model_elements_[index];
  }

  std::ostringstream type_map;
  for (int type = 1; type <= atom->ntypes; ++type) {
    if (type != 1) {
      type_map << ", ";
    }
    const int model_type = type_map_[type];
    type_map << type << "->" << model_elements_[model_type]
             << "(model " << model_type + 1 << ')';
  }

  utils::logmesg(
      lmp,
      "NEPAdapters {}: loaded model {}\n"
      "  pair_style: {}, backend: {}\n"
      "  model elements: {}\n"
      "  LAMMPS type map: {}\n",
      NEP_ADAPTERS_VERSION_STRING,
      model_path,
      style_name_,
      engine_name_,
      elements.str(),
      type_map.str());
}

void PairNEPAdaptersCommon::init_style() {
  if (!domain->xperiodic || !domain->yperiodic || !domain->zperiodic) {
    const std::string message =
        label_ + ": NEP requires periodic boundaries in x, y, and z";
    error->all(FLERR, message.c_str());
  }
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

double PairNEPAdaptersCommon::init_one(int, int) {
  return cutoff_;
}

void PairNEPAdaptersCommon::compute(int eflag, int vflag) {
  if (model_ == nullptr) {
    const std::string message =
        label_ + ": model not initialized; check pair_coeff";
    error->all(FLERR, message.c_str());
  }

  ev_init(eflag, vflag);

  const int nlocal = atom->nlocal;
  const int nall = atom->nlocal + atom->nghost;
  if (spin_model_ &&
      (!atom->sp_flag || atom->sp == nullptr || atom->fm == nullptr)) {
    const std::string message =
        label_ + ": spin model requires atom_style spin";
    error->all(FLERR, message.c_str());
  }

  if (!force->newton) {
    // Verlet only clears owned force rows when Newton is off. This pair style
    // deliberately accumulates center-based neighbor contributions on ghost
    // rows and reverse-communicates them, so every ghost row must start from
    // zero. This is especially important on ranks with no owned atoms, where
    // LAMMPS clears no force storage at all.
    for (int i = nlocal; i < nall; ++i) {
      atom->f[i][0] = 0.0;
      atom->f[i][1] = 0.0;
      atom->f[i][2] = 0.0;
      if (spin_model_) {
        atom->fm[i][0] = 0.0;
        atom->fm[i][1] = 0.0;
        atom->fm[i][2] = 0.0;
      }
    }
  }
  if (nlocal <= 0) {
    // A rank with no owned atoms still has to enter the Newton-off reverse
    // exchange.  Other ranks may send contributions through its ghost layers;
    // returning here would leave their point-to-point communication pending
    // and can hang MPI finalization (for example, four ranks for four atoms).
    const bool want_atom_virial = (vatom != nullptr || cvatom != nullptr);
    if (want_atom_virial) {
      virials_per_atom_.assign(static_cast<std::size_t>(nall) * 9, 0.0);
    }
    classic_reverse_per_atom_virial_ =
        !force->newton && want_atom_virial;
    if (!force->newton && nall > 0) {
      const int force_width = spin_model_ ? 6 : 3;
      const int reverse_width =
          force_width + (classic_reverse_per_atom_virial_ ? 9 : 0);
      comm->reverse_comm(this, reverse_width);
    }
    classic_reverse_per_atom_virial_ = false;
    return;
  }
  if (list == nullptr) {
    const std::string message = label_ + ": missing LAMMPS neighbor list";
    error->all(FLERR, message.c_str());
  }

  for (int i = 0; i < nall; ++i) {
    const int lammps_type = atom->type[i];
    if (lammps_type < 1 || lammps_type > atom->ntypes ||
        type_map_[lammps_type] < 0) {
      const std::string message = label_ + ": invalid atom type mapping";
      error->one(FLERR, message.c_str());
    }
  }

  // Keep each atom's row capacity across timesteps. Clearing the outer vector
  // would destroy every row and repeat thousands of small allocations even
  // when LAMMPS reuses an unchanged neighbor-list shape.
  sanitized_neighbors_.resize(static_cast<std::size_t>(nall));
  sanitized_numneigh_.assign(static_cast<std::size_t>(nall), 0);
  sanitized_firstneigh_.assign(static_cast<std::size_t>(nall), nullptr);

  for (int ii = 0; ii < list->inum; ++ii) {
    const int i = list->ilist[ii];
    if (i < 0 || i >= nlocal) {
      const std::string message =
          label_ + ": expected neighbor ilist in [0,nlocal)";
      error->one(FLERR, message.c_str());
    }

    std::vector<int>& neighbors = sanitized_neighbors_[static_cast<std::size_t>(i)];
    neighbors.clear();
    neighbors.reserve(static_cast<std::size_t>(list->numneigh[i]));
    const int* jlist = list->firstneigh[i];
    for (int jj = 0; jj < list->numneigh[i]; ++jj) {
      const int j = jlist[jj] & NEIGHMASK;
      if (j < 0 || j >= nall) {
        const std::string message = label_ + ": invalid neighbor atom index";
        error->one(FLERR, message.c_str());
      }
      neighbors.push_back(j);
    }
    sanitized_numneigh_[static_cast<std::size_t>(i)] =
        static_cast<int>(neighbors.size());
    sanitized_firstneigh_[static_cast<std::size_t>(i)] =
        neighbors.empty() ? nullptr : neighbors.data();
  }

  if (eflag_atom) {
    potential_.assign(static_cast<std::size_t>(nlocal), 0.0);
  }

  const bool want_atom_virial = (vatom != nullptr || cvatom != nullptr);
  if (want_atom_virial) {
    virials_per_atom_.assign(static_cast<std::size_t>(nall) * 9, 0.0);
    virial_rows_.resize(static_cast<std::size_t>(nall));
    for (int i = 0; i < nall; ++i) {
      virial_rows_[static_cast<std::size_t>(i)] =
          virials_per_atom_.data() + 9 * static_cast<std::size_t>(i);
    }
  }

  double total_potential = 0.0;
  double total_virial[6] = {};
  NepaLammpsNeighborInput input{};
  input.nlocal = nlocal;
  input.inum = list->inum;
  input.ilist = list->ilist;
  input.numneigh = sanitized_numneigh_.data();
  input.firstneigh = sanitized_firstneigh_.data();
  input.types = atom->type;
  input.type_map = type_map_;
  input.positions = atom->x;
  input.spins = spin_model_ ? atom->sp : nullptr;

  NepaLammpsNeighborResult result{};
  result.total_potential = &total_potential;
  result.total_virial6 = total_virial;
  result.potential_per_atom = eflag_atom ? potential_.data() : nullptr;
  result.forces = atom->f;
  result.mforces = spin_model_ ? atom->fm : nullptr;
  result.virials_per_atom9 = want_atom_virial ? virial_rows_.data() : nullptr;

  const NepaStatus status =
      nepa_find_force_lammps_neighbors(model_, &input, &result);
  if (status != NEPA_STATUS_OK) {
    error->all(FLERR, nepa_status_message(status));
  }

  // LAMMPS skips its normal atom reverse communication when Newton is off,
  // but the center-based NEP force decomposition still writes neighbor rows,
  // including ghost rows.  Pull those contributions back to their owners.
  // Per-atom virial uses the same neighbor ownership and must travel with f/fm.
  classic_reverse_per_atom_virial_ =
      !force->newton && want_atom_virial;
  if (!force->newton && nall > nlocal) {
    const int force_width = spin_model_ ? 6 : 3;
    const int reverse_width =
        force_width + (classic_reverse_per_atom_virial_ ? 9 : 0);
    comm->reverse_comm(this, reverse_width);
  }
  classic_reverse_per_atom_virial_ = false;

  if (eflag) {
    eng_vdwl += total_potential;
  }
  if (vflag) {
    for (int component = 0; component < 6; ++component) {
      virial[component] += total_virial[component];
    }
  }
  if (eflag_atom) {
    for (int i = 0; i < nlocal; ++i) {
      eatom[i] += potential_[static_cast<std::size_t>(i)];
    }
  }
  if (vflag_atom && !vatom && !cvatom) {
    const std::string message =
        label_ + ": per-atom virial requested but no per-atom buffer exists";
    error->one(FLERR, message.c_str());
  }
  if (cvflag_atom && !cvatom && !vatom) {
    const std::string message =
        label_ + ": centroid virial requested but no centroid buffer exists";
    error->one(FLERR, message.c_str());
  }
  const int nvirial = force->newton ? nall : nlocal;
  if (vatom) {
    for (int i = 0; i < nvirial; ++i) {
      const double* raw9 = virials_per_atom_.data() + 9 * static_cast<std::size_t>(i);
      for (int component = 0; component < 6; ++component) {
        vatom[i][component] +=
            nep_adapters::lammps_voigt6_from_lammps_raw9(raw9, component);
      }
    }
  }
  if (cvatom) {
    for (int i = 0; i < nvirial; ++i) {
      const double* raw9 = virials_per_atom_.data() + 9 * static_cast<std::size_t>(i);
      for (int component = 0; component < 9; ++component) {
        cvatom[i][component] += raw9[component];
      }
    }
  }
}

int PairNEPAdaptersCommon::pack_reverse_comm(
    int n,
    int first,
    double* buffer) {
  const int force_width = spin_model_ ? 6 : 3;
  const int width = force_width + (classic_reverse_per_atom_virial_ ? 9 : 0);
  int offset = 0;
  for (int i = first; i < first + n; ++i) {
    buffer[offset++] = atom->f[i][0];
    buffer[offset++] = atom->f[i][1];
    buffer[offset++] = atom->f[i][2];
    if (spin_model_) {
      buffer[offset++] = atom->fm[i][0];
      buffer[offset++] = atom->fm[i][1];
      buffer[offset++] = atom->fm[i][2];
    }
    if (classic_reverse_per_atom_virial_) {
      const double* raw9 = virials_per_atom_.data() + 9 * static_cast<std::size_t>(i);
      for (int component = 0; component < 9; ++component) {
        buffer[offset++] = raw9[component];
      }
    }
  }
  return width * n;
}

void PairNEPAdaptersCommon::unpack_reverse_comm(
    int n,
    int* list,
    double* buffer) {
  int offset = 0;
  for (int i = 0; i < n; ++i) {
    const int atom_index = list[i];
    atom->f[atom_index][0] += buffer[offset++];
    atom->f[atom_index][1] += buffer[offset++];
    atom->f[atom_index][2] += buffer[offset++];
    if (spin_model_) {
      atom->fm[atom_index][0] += buffer[offset++];
      atom->fm[atom_index][1] += buffer[offset++];
      atom->fm[atom_index][2] += buffer[offset++];
    }
    if (classic_reverse_per_atom_virial_) {
      double* raw9 = virials_per_atom_.data() +
          9 * static_cast<std::size_t>(atom_index);
      for (int component = 0; component < 9; ++component) {
        raw9[component] += buffer[offset++];
      }
    }
  }
}
