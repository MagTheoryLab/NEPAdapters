#include "pair_nep_adapters_cpu.h"

#include "nep_adapters/api.h"
#include "nep_adapters/engines/cpu_nep3.hpp"
#include "nep_adapters/virial_order.hpp"

#include "atom.h"
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
  std::ifstream input(model_path.c_str());
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

PairNEPAdaptersCPU::PairNEPAdaptersCPU(LAMMPS* lmp) : Pair(lmp) {
  centroidstressflag = CENTROID_AVAIL;
  restartinfo = 0;
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;
  one_coeff = 1;
  single_enable = 0;
}

PairNEPAdaptersCPU::~PairNEPAdaptersCPU() {
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

void PairNEPAdaptersCPU::allocate() {
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

void PairNEPAdaptersCPU::settings(int narg, char**) {
  if (narg != 0) {
    error->all(FLERR, "Illegal pair_style command; usage: pair_style nep/cpu");
  }
}

void PairNEPAdaptersCPU::coeff(int narg, char** arg) {
  if (!allocated) {
    allocate();
  }
  if (narg != 3 + atom->ntypes) {
    error->all(
        FLERR,
        "Incorrect args for pair_coeff; usage: pair_coeff * * model.nep Element ...");
  }
  if (std::string(arg[0]) != "*" || std::string(arg[1]) != "*") {
    error->all(FLERR, "pair_style nep/cpu requires pair_coeff * *");
  }

  model_filename_ = arg[2];
  const std::string model_path = utils::get_potential_file_path(model_filename_);
  read_type_map(model_path, narg, arg);
  load_model(model_path);

  for (int i = 1; i <= atom->ntypes; ++i) {
    for (int j = 1; j <= atom->ntypes; ++j) {
      cutsq[i][j] = cutoff_ * cutoff_;
    }
  }
}

void PairNEPAdaptersCPU::read_type_map(
    const std::string& model_path,
    int,
    char** arg) {
  const std::vector<std::string> elements = read_nep_elements(model_path);
  if (elements.empty()) {
    error->all(FLERR, "NEPAdapters CPU: failed to read element symbols from NEP file");
  }

  for (int type = 1; type <= atom->ntypes; ++type) {
    const char* symbol = arg[2 + type];
    const auto found = std::find(elements.begin(), elements.end(), symbol);
    if (found == elements.end()) {
      error->all(FLERR, "NEPAdapters CPU: pair_coeff element is absent from NEP file");
    }
    type_map_[type] = static_cast<int>(std::distance(elements.begin(), found));
  }
}

void PairNEPAdaptersCPU::load_model(const std::string& model_path) {
  if (!nep_adapters::register_cpu_nep3_engine()) {
    error->all(FLERR, "NEPAdapters CPU: failed to register cpu_nep3 engine");
  }

  nepa_free_model(model_);
  model_ = nullptr;
  NepaStatus status = nepa_load_model("cpu_nep3", model_path.c_str(), &model_);
  if (status != NEPA_STATUS_OK || model_ == nullptr) {
    error->all(FLERR, nepa_status_message(status));
  }

  NepaModelInfo info{};
  status = nepa_model_info(model_, &info);
  if (status != NEPA_STATUS_OK || info.cutoff_max <= 0.0) {
    error->all(FLERR, "NEPAdapters CPU: failed to query model cutoff");
  }
  cutoff_ = info.cutoff_max;
}

void PairNEPAdaptersCPU::init_style() {
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

double PairNEPAdaptersCPU::init_one(int, int) {
  return cutoff_;
}

void PairNEPAdaptersCPU::compute(int eflag, int vflag) {
  if (model_ == nullptr) {
    error->all(FLERR, "NEPAdapters CPU: model not initialized; check pair_coeff");
  }

  ev_init(eflag, vflag);

  const int nlocal = atom->nlocal;
  if (nlocal <= 0) {
    return;
  }
  if (list == nullptr) {
    error->all(FLERR, "NEPAdapters CPU: missing LAMMPS neighbor list");
  }

  const int nall = atom->nlocal + atom->nghost;
  for (int i = 0; i < nall; ++i) {
    const int lammps_type = atom->type[i];
    if (lammps_type < 1 || lammps_type > atom->ntypes ||
        type_map_[lammps_type] < 0) {
      error->one(FLERR, "NEPAdapters CPU: invalid atom type mapping");
    }
  }

  sanitized_neighbors_.clear();
  sanitized_neighbors_.resize(static_cast<std::size_t>(nall));
  sanitized_numneigh_.assign(static_cast<std::size_t>(nall), 0);
  sanitized_firstneigh_.assign(static_cast<std::size_t>(nall), nullptr);

  for (int ii = 0; ii < list->inum; ++ii) {
    const int i = list->ilist[ii];
    if (i < 0 || i >= nlocal) {
      error->one(FLERR, "NEPAdapters CPU: expected neighbor ilist in [0,nlocal)");
    }

    std::vector<int>& neighbors = sanitized_neighbors_[static_cast<std::size_t>(i)];
    neighbors.reserve(static_cast<std::size_t>(list->numneigh[i]));
    const int* jlist = list->firstneigh[i];
    for (int jj = 0; jj < list->numneigh[i]; ++jj) {
      const int j = jlist[jj] & NEIGHMASK;
      if (j < 0 || j >= nall) {
        error->one(FLERR, "NEPAdapters CPU: invalid neighbor atom index");
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

  NepaLammpsNeighborResult result{};
  result.total_potential = &total_potential;
  result.total_virial6 = total_virial;
  result.potential_per_atom = eflag_atom ? potential_.data() : nullptr;
  result.forces = atom->f;
  result.virials_per_atom9 = want_atom_virial ? virial_rows_.data() : nullptr;

  const NepaStatus status =
      nepa_find_force_lammps_neighbors(model_, &input, &result);
  if (status != NEPA_STATUS_OK) {
    error->all(FLERR, nepa_status_message(status));
  }

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
    error->one(
        FLERR,
        "NEPAdapters CPU: per-atom virial requested but no per-atom buffer exists");
  }
  if (cvflag_atom && !cvatom && !vatom) {
    error->one(
        FLERR,
        "NEPAdapters CPU: centroid virial requested but no centroid buffer exists");
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
