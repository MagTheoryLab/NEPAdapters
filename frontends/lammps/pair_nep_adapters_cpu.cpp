#include "pair_nep_adapters_cpu.h"

#include "nep_adapters/api.h"
#include "nep_adapters/engines/cpu_nep3.hpp"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace LAMMPS_NS;

namespace {

double lammps_virial_component(const double raw9[9], int component) {
  switch (component) {
    case 0:
      return raw9[0];
    case 1:
      return raw9[4];
    case 2:
      return raw9[8];
    case 3:
      return 0.5 * (raw9[1] + raw9[3]);
    case 4:
      return 0.5 * (raw9[2] + raw9[6]);
    case 5:
      return 0.5 * (raw9[5] + raw9[7]);
  }
  return 0.0;
}

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
    error->all(FLERR, "Illegal pair_style command; usage: pair_style nep/adapters/cpu");
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
    error->all(FLERR, "pair_style nep/adapters/cpu requires pair_coeff * *");
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
  if (comm->nprocs != 1) {
    error->all(
        FLERR,
        "NEPAdapters CPU LAMMPS frontend currently supports one MPI rank; "
        "multi-rank needs the external-neighbor engine path");
  }

  ev_init(eflag, vflag);

  const int nlocal = atom->nlocal;
  if (nlocal <= 0) {
    return;
  }
  if (nlocal > std::numeric_limits<std::int32_t>::max()) {
    error->all(FLERR, "NEPAdapters CPU: too many atoms for current C ABI");
  }

  types_.resize(static_cast<std::size_t>(nlocal));
  positions_.resize(static_cast<std::size_t>(nlocal) * 3);
  forces_.assign(static_cast<std::size_t>(nlocal) * 3, 0.0);
  potential_.assign(static_cast<std::size_t>(nlocal), 0.0);
  virials_per_atom_.assign(static_cast<std::size_t>(nlocal) * 9, 0.0);

  for (int i = 0; i < nlocal; ++i) {
    const int lammps_type = atom->type[i];
    if (lammps_type < 1 || lammps_type > atom->ntypes ||
        type_map_[lammps_type] < 0) {
      error->one(FLERR, "NEPAdapters CPU: invalid atom type mapping");
    }
    types_[static_cast<std::size_t>(i)] = type_map_[lammps_type];
    positions_[3 * static_cast<std::size_t>(i) + 0] = atom->x[i][0];
    positions_[3 * static_cast<std::size_t>(i) + 1] = atom->x[i][1];
    positions_[3 * static_cast<std::size_t>(i) + 2] = atom->x[i][2];
  }

  box_[0] = domain->h[0];
  box_[3] = 0.0;
  box_[6] = 0.0;
  box_[1] = domain->h[5];
  box_[4] = domain->h[1];
  box_[7] = 0.0;
  box_[2] = domain->h[4];
  box_[5] = domain->h[3];
  box_[8] = domain->h[2];

  pbc_[0] = domain->xperiodic ? 1 : 0;
  pbc_[1] = domain->yperiodic ? 1 : 0;
  pbc_[2] = domain->zperiodic ? 1 : 0;

  std::int32_t atom_counts[] = {static_cast<std::int32_t>(nlocal)};
  std::int32_t atom_offsets[] = {0};
  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = static_cast<std::int32_t>(nlocal);
  batch.atom_counts = atom_counts;
  batch.atom_offsets = atom_offsets;
  batch.types = types_.data();
  batch.positions_aos3 = positions_.data();
  batch.boxes_row_major9 = box_;
  batch.pbc_flags3 = pbc_;

  double energy = 0.0;
  double structure_virial[9] = {};
  NepaFindForceResult result{};
  result.energy_per_structure = &energy;
  result.potential_per_atom = potential_.data();
  result.forces_aos3 = forces_.data();
  result.virials_row_major9 = structure_virial;
  result.virials_per_atom_row_major9 = virials_per_atom_.data();

  const NepaStatus status = nepa_find_force_batch(model_, &batch, &result);
  if (status != NEPA_STATUS_OK) {
    error->all(FLERR, nepa_status_message(status));
  }

  for (int i = 0; i < nlocal; ++i) {
    atom->f[i][0] += forces_[3 * static_cast<std::size_t>(i) + 0];
    atom->f[i][1] += forces_[3 * static_cast<std::size_t>(i) + 1];
    atom->f[i][2] += forces_[3 * static_cast<std::size_t>(i) + 2];
  }

  if (eflag) {
    eng_vdwl += energy;
  }
  if (vflag) {
    for (int component = 0; component < 6; ++component) {
      virial[component] += lammps_virial_component(structure_virial, component);
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
  if (vatom) {
    for (int i = 0; i < nlocal; ++i) {
      const double* raw9 = virials_per_atom_.data() + 9 * static_cast<std::size_t>(i);
      for (int component = 0; component < 6; ++component) {
        vatom[i][component] += lammps_virial_component(raw9, component);
      }
    }
  }
  if (cvatom) {
    for (int i = 0; i < nlocal; ++i) {
      const double* raw9 = virials_per_atom_.data() + 9 * static_cast<std::size_t>(i);
      for (int component = 0; component < 9; ++component) {
        cvatom[i][component] += raw9[component];
      }
    }
  }
}
