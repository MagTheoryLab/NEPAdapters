#include "nep_adapters/api.h"
#if defined(NEP_ADAPTERS_PYTHON_GPU_MODULE)
#include "nep_adapters/engines/cuda.hpp"
#else
#include "nep_adapters/engines/cpu.hpp"
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

void check_status(NepaStatus status) {
  if (status != NEPA_STATUS_OK) {
    const char* detail = nepa_last_error_message();
    const std::string message =
        detail != nullptr && detail[0] != '\0'
            ? std::string(nepa_status_message(status)) + ": " + detail
            : nepa_status_message(status);
    if (status == NEPA_STATUS_INVALID_ARGUMENT) {
      throw std::invalid_argument(message);
    }
    throw std::runtime_error(message);
  }
}

struct BackendInfo {
  std::string name;
  std::string version;
  std::uint64_t capabilities = 0;
};

struct PreparedBatch {
  std::int32_t structure_count = 0;
  std::int32_t total_atoms = 0;
  std::vector<std::int32_t> offsets;
  std::vector<double> boxes_for_batch;
  std::vector<std::int32_t> default_pbc;
  py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> pbc_array;
  const std::int32_t* atom_counts = nullptr;
  const std::int32_t* types = nullptr;
  const double* positions = nullptr;
  const double* boxes = nullptr;
  const std::int32_t* pbc = nullptr;
};

void require_fully_periodic(const std::int32_t* pbc, std::size_t count) {
  if (pbc == nullptr ||
      !std::all_of(pbc, pbc + count, [](std::int32_t value) { return value == 1; })) {
    throw std::invalid_argument(
        "NEPAdapters supports fully periodic structures only (pbc=[1,1,1])");
  }
}

PreparedBatch prepare_batch(
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>& types,
    py::array_t<double, py::array::c_style | py::array::forcecast>& boxes,
    py::array_t<double, py::array::c_style | py::array::forcecast>& positions,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>& atom_counts,
    py::object pbc_object) {
  py::buffer_info type_info = types.request();
  py::buffer_info box_info = boxes.request();
  py::buffer_info position_info = positions.request();
  py::buffer_info atom_count_info = atom_counts.request();

  if (type_info.ndim != 1) {
    throw std::invalid_argument("types must be a 1D int32 array");
  }
  if (position_info.ndim != 2 || position_info.shape[1] != 3) {
    throw std::invalid_argument("positions must have shape (natoms, 3)");
  }
  if (atom_count_info.ndim != 1) {
    throw std::invalid_argument("atom_counts must be a 1D int32 array");
  }
  if (position_info.shape[0] != type_info.shape[0]) {
    throw std::invalid_argument("types and positions atom counts differ");
  }

  PreparedBatch prepared;
  prepared.structure_count = static_cast<std::int32_t>(atom_count_info.shape[0]);
  prepared.total_atoms = static_cast<std::int32_t>(type_info.shape[0]);
  if (prepared.structure_count <= 0 || prepared.total_atoms <= 0) {
    throw std::invalid_argument("empty batches are not supported");
  }

  prepared.atom_counts = static_cast<const std::int32_t*>(atom_count_info.ptr);
  prepared.offsets.resize(static_cast<std::size_t>(prepared.structure_count), 0);
  std::int32_t cursor = 0;
  for (std::int32_t i = 0; i < prepared.structure_count; ++i) {
    if (prepared.atom_counts[i] <= 0) {
      throw std::invalid_argument("atom_counts must be positive");
    }
    prepared.offsets[static_cast<std::size_t>(i)] = cursor;
    cursor += prepared.atom_counts[i];
  }
  if (cursor != prepared.total_atoms) {
    throw std::invalid_argument("sum(atom_counts) must equal len(types)");
  }

  if (!((box_info.ndim == 1 && box_info.size == 9) ||
        (box_info.ndim == 2 && box_info.shape[0] == prepared.structure_count &&
         box_info.shape[1] == 9))) {
    throw std::invalid_argument("boxes must have shape (9,) or (nstructures, 9)");
  }

  prepared.types = static_cast<const std::int32_t*>(type_info.ptr);
  prepared.positions = static_cast<const double*>(position_info.ptr);
  prepared.boxes = static_cast<const double*>(box_info.ptr);
  if (box_info.ndim == 1 && prepared.structure_count > 1) {
    prepared.boxes_for_batch.resize(
        static_cast<std::size_t>(prepared.structure_count) * 9);
    for (std::int32_t i = 0; i < prepared.structure_count; ++i) {
      std::copy(
          prepared.boxes,
          prepared.boxes + 9,
          prepared.boxes_for_batch.data() + 9 * i);
    }
    prepared.boxes = prepared.boxes_for_batch.data();
  }

  if (pbc_object.is_none()) {
    throw std::invalid_argument(
        "pbc defaults to (1,1,1); omit it instead of passing None");
  }
  prepared.pbc_array = py::cast<
      py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>>(
      pbc_object);
  py::buffer_info pbc_info = prepared.pbc_array.request();
  if (!((pbc_info.ndim == 1 && pbc_info.size == 3) ||
        (pbc_info.ndim == 2 && pbc_info.shape[0] == prepared.structure_count &&
         pbc_info.shape[1] == 3))) {
    throw std::invalid_argument("pbc must have shape (3,) or (nstructures, 3)");
  }
  if (pbc_info.ndim == 1 && prepared.structure_count > 1) {
    prepared.default_pbc.resize(
        static_cast<std::size_t>(prepared.structure_count) * 3);
    const auto* one_pbc = static_cast<const std::int32_t*>(pbc_info.ptr);
    for (std::int32_t i = 0; i < prepared.structure_count; ++i) {
      std::copy(one_pbc, one_pbc + 3, prepared.default_pbc.data() + 3 * i);
    }
    prepared.pbc = prepared.default_pbc.data();
  } else {
    prepared.pbc = static_cast<const std::int32_t*>(pbc_info.ptr);
  }
  require_fully_periodic(
      prepared.pbc,
      static_cast<std::size_t>(prepared.structure_count) * 3);

  return prepared;
}

const double* prepare_spins(
    py::array_t<double, py::array::c_style | py::array::forcecast>& spins,
    std::int32_t total_atoms) {
  const py::buffer_info spin_info = spins.request();
  if (spin_info.ndim != 2 || spin_info.shape[0] != total_atoms ||
      spin_info.shape[1] != 3) {
    throw std::invalid_argument("spins must have shape (natoms, 3)");
  }
  return static_cast<const double*>(spin_info.ptr);
}

void require_spin_model(NepaModel* model) {
  NepaModelInfo info{};
  check_status(nepa_model_info(model, &info));
  if ((info.capabilities & NEPA_CAPABILITY_SPIN) == 0u) {
    throw std::invalid_argument("spin input requires a spin NEP model");
  }
}

class PyModel {
 public:
  explicit PyModel(NepaModel* model) : model_(model, nepa_free_model) {}

  py::dict model_info() const {
    if (!model_) {
      throw std::runtime_error("model is closed");
    }

    NepaModelInfo info{};
    check_status(nepa_model_info(model_.get(), &info));
    py::dict out;
    out["cutoff_radial"] = info.cutoff_radial;
    out["cutoff_angular"] = info.cutoff_angular;
    out["cutoff_max"] = info.cutoff_max;
    out["capabilities"] = info.capabilities;
    out["num_types"] = info.num_types;
    out["descriptor_dim"] = info.descriptor_dim;
    return out;
  }

  py::tuple calculate(
      py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> types,
      py::array_t<double, py::array::c_style | py::array::forcecast> boxes,
      py::array_t<double, py::array::c_style | py::array::forcecast> positions,
      py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> atom_counts,
      py::object pbc_object) {
    if (!model_) {
      throw std::runtime_error("model is closed");
    }

    PreparedBatch input =
        prepare_batch(types, boxes, positions, atom_counts, pbc_object);

    py::array_t<double> potentials(static_cast<py::ssize_t>(input.total_atoms));
    py::array_t<double> forces({input.total_atoms, static_cast<std::int32_t>(3)});
    py::array_t<double> virials({input.total_atoms, static_cast<std::int32_t>(9)});
    std::vector<double> energies(static_cast<std::size_t>(input.structure_count), 0.0);
    std::vector<double> structure_virials(
        static_cast<std::size_t>(input.structure_count) * 9,
        0.0);

    NepaStructureBatch batch{};
    batch.num_structures = input.structure_count;
    batch.total_atoms = input.total_atoms;
    batch.atom_counts = input.atom_counts;
    batch.atom_offsets = input.offsets.data();
    batch.types = input.types;
    batch.positions_aos3 = input.positions;
    batch.boxes_row_major9 = input.boxes;
    batch.pbc_flags3 = input.pbc;

    NepaFindForceResult result{};
    result.energy_per_structure = energies.data();
    result.potential_per_atom = static_cast<double*>(potentials.request().ptr);
    result.forces_aos3 = static_cast<double*>(forces.request().ptr);
    result.virials_row_major9 = structure_virials.data();
    result.virials_per_atom_row_major9 = static_cast<double*>(virials.request().ptr);

    {
      py::gil_scoped_release release;
      check_status(nepa_find_force_batch(model_.get(), &batch, &result));
    }
    return py::make_tuple(potentials, forces, virials);
  }

  py::tuple calculate_spin(
      py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> types,
      py::array_t<double, py::array::c_style | py::array::forcecast> boxes,
      py::array_t<double, py::array::c_style | py::array::forcecast> positions,
      py::array_t<double, py::array::c_style | py::array::forcecast> spins,
      py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> atom_counts,
      py::object pbc_object) {
    if (!model_) {
      throw std::runtime_error("model is closed");
    }
    require_spin_model(model_.get());

    PreparedBatch input =
        prepare_batch(types, boxes, positions, atom_counts, pbc_object);
    const double* spin_data = prepare_spins(spins, input.total_atoms);

    py::array_t<double> potentials(static_cast<py::ssize_t>(input.total_atoms));
    py::array_t<double> forces({input.total_atoms, static_cast<std::int32_t>(3)});
    py::array_t<double> virials({input.total_atoms, static_cast<std::int32_t>(9)});
    py::array_t<double> mforces({input.total_atoms, static_cast<std::int32_t>(3)});
    py::array_t<double> tau({input.total_atoms, static_cast<std::int32_t>(3)});
    std::vector<double> energies(static_cast<std::size_t>(input.structure_count), 0.0);
    std::vector<double> structure_virials(
        static_cast<std::size_t>(input.structure_count) * 9,
        0.0);

    NepaStructureBatch batch{};
    batch.num_structures = input.structure_count;
    batch.total_atoms = input.total_atoms;
    batch.atom_counts = input.atom_counts;
    batch.atom_offsets = input.offsets.data();
    batch.types = input.types;
    batch.positions_aos3 = input.positions;
    batch.spins_aos3 = spin_data;
    batch.boxes_row_major9 = input.boxes;
    batch.pbc_flags3 = input.pbc;

    NepaFindForceResult result{};
    result.energy_per_structure = energies.data();
    result.potential_per_atom = static_cast<double*>(potentials.request().ptr);
    result.forces_aos3 = static_cast<double*>(forces.request().ptr);
    result.virials_row_major9 = structure_virials.data();
    result.virials_per_atom_row_major9 = static_cast<double*>(virials.request().ptr);
    result.mforces_aos3 = static_cast<double*>(mforces.request().ptr);
    result.tau_aos3 = static_cast<double*>(tau.request().ptr);

    {
      py::gil_scoped_release release;
      check_status(nepa_find_force_batch(model_.get(), &batch, &result));
    }
    return py::make_tuple(potentials, forces, virials, mforces, tau);
  }

  py::array_t<double> descriptors(
      py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> types,
      py::array_t<double, py::array::c_style | py::array::forcecast> boxes,
      py::array_t<double, py::array::c_style | py::array::forcecast> positions,
      py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> atom_counts,
      py::object pbc_object) {
    if (!model_) {
      throw std::runtime_error("model is closed");
    }

    NepaModelInfo info{};
    check_status(nepa_model_info(model_.get(), &info));
    if ((info.capabilities & NEPA_CAPABILITY_DESCRIPTORS) == 0u ||
        info.descriptor_dim <= 0) {
      throw std::runtime_error("descriptors are unsupported by this model");
    }

    PreparedBatch input =
        prepare_batch(types, boxes, positions, atom_counts, pbc_object);

    py::array_t<double> descriptors({
        input.total_atoms,
        info.descriptor_dim,
    });

    NepaStructureBatch batch{};
    batch.num_structures = input.structure_count;
    batch.total_atoms = input.total_atoms;
    batch.atom_counts = input.atom_counts;
    batch.atom_offsets = input.offsets.data();
    batch.types = input.types;
    batch.positions_aos3 = input.positions;
    batch.boxes_row_major9 = input.boxes;
    batch.pbc_flags3 = input.pbc;

    NepaFindDescriptorResult result{};
    result.descriptors = static_cast<double*>(descriptors.request().ptr);

    {
      py::gil_scoped_release release;
      check_status(nepa_find_descriptors(model_.get(), &batch, &result));
    }

    return descriptors;
  }

  py::array_t<double> descriptors_spin(
      py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> types,
      py::array_t<double, py::array::c_style | py::array::forcecast> boxes,
      py::array_t<double, py::array::c_style | py::array::forcecast> positions,
      py::array_t<double, py::array::c_style | py::array::forcecast> spins,
      py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> atom_counts,
      py::object pbc_object) {
    if (!model_) {
      throw std::runtime_error("model is closed");
    }
    require_spin_model(model_.get());

    NepaModelInfo info{};
    check_status(nepa_model_info(model_.get(), &info));
    if ((info.capabilities & NEPA_CAPABILITY_DESCRIPTORS) == 0u ||
        info.descriptor_dim <= 0) {
      throw std::runtime_error("descriptors are unsupported by this model");
    }

    PreparedBatch input =
        prepare_batch(types, boxes, positions, atom_counts, pbc_object);
    const double* spin_data = prepare_spins(spins, input.total_atoms);
    py::array_t<double> descriptors({input.total_atoms, info.descriptor_dim});

    NepaStructureBatch batch{};
    batch.num_structures = input.structure_count;
    batch.total_atoms = input.total_atoms;
    batch.atom_counts = input.atom_counts;
    batch.atom_offsets = input.offsets.data();
    batch.types = input.types;
    batch.positions_aos3 = input.positions;
    batch.spins_aos3 = spin_data;
    batch.boxes_row_major9 = input.boxes;
    batch.pbc_flags3 = input.pbc;

    NepaFindDescriptorResult result{};
    result.descriptors = static_cast<double*>(descriptors.request().ptr);
    {
      py::gil_scoped_release release;
      check_status(nepa_find_descriptors(model_.get(), &batch, &result));
    }
    return descriptors;
  }

  py::tuple find_force(
      py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> types,
      py::array_t<double, py::array::c_style | py::array::forcecast> positions,
      py::array_t<double, py::array::c_style | py::array::forcecast> box,
      py::object pbc_object) {
    if (!model_) {
      throw std::runtime_error("model is closed");
    }

    py::buffer_info type_info = types.request();
    py::buffer_info position_info = positions.request();
    py::buffer_info box_info = box.request();
    if (pbc_object.is_none()) {
      throw std::invalid_argument(
          "pbc defaults to (1,1,1); omit it instead of passing None");
    }
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> pbc_array;
    pbc_array = py::cast<
        py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>>(
        pbc_object);
    py::buffer_info pbc_info = pbc_array.request();
    if (pbc_info.size != 3) {
      throw std::invalid_argument("pbc must contain 3 values");
    }
    const auto* pbc_ptr = static_cast<const std::int32_t*>(pbc_info.ptr);
    require_fully_periodic(pbc_ptr, 3);

    if (type_info.ndim != 1) {
      throw std::invalid_argument("types must be a 1D int32 array");
    }
    if (position_info.ndim != 2 || position_info.shape[1] != 3) {
      throw std::invalid_argument("positions must have shape (natoms, 3)");
    }
    if (box_info.size != 9) {
      throw std::invalid_argument("box must contain 9 values");
    }
    if (position_info.shape[0] != type_info.shape[0]) {
      throw std::invalid_argument("types and positions atom counts differ");
    }

    const auto atom_count = static_cast<std::int32_t>(type_info.shape[0]);
    std::int32_t atom_counts[] = {atom_count};
    std::int32_t atom_offsets[] = {0};

    NepaStructureBatch batch{};
    batch.num_structures = 1;
    batch.total_atoms = atom_count;
    batch.atom_counts = atom_counts;
    batch.atom_offsets = atom_offsets;
    batch.types = static_cast<const std::int32_t*>(type_info.ptr);
    batch.positions_aos3 = static_cast<const double*>(position_info.ptr);
    batch.boxes_row_major9 = static_cast<const double*>(box_info.ptr);
    batch.pbc_flags3 = pbc_ptr;

    py::array_t<double> forces({atom_count, static_cast<std::int32_t>(3)});
    py::array_t<double> virial(static_cast<py::ssize_t>(9));
    double energy = 0.0;

    NepaFindForceResult result{};
    result.energy_per_structure = &energy;
    result.potential_per_atom = nullptr;
    result.forces_aos3 = static_cast<double*>(forces.request().ptr);
    result.virials_row_major9 = static_cast<double*>(virial.request().ptr);
    result.virials_per_atom_row_major9 = nullptr;

    {
      py::gil_scoped_release release;
      check_status(nepa_find_force_batch(model_.get(), &batch, &result));
    }
    return py::make_tuple(energy, forces, virial);
  }

  void close() { model_.reset(); }

 private:
  std::unique_ptr<NepaModel, void (*)(NepaModel*)> model_;
};

}  // namespace

#if defined(NEP_ADAPTERS_PYTHON_GPU_MODULE)
#define NEP_ADAPTERS_PYTHON_MODULE_NAME nep_gpu
#else
#define NEP_ADAPTERS_PYTHON_MODULE_NAME nep_cpu
#endif

PYBIND11_MODULE(NEP_ADAPTERS_PYTHON_MODULE_NAME, module) {
#if defined(NEP_ADAPTERS_PYTHON_GPU_MODULE)
  module.doc() = "NEPAdapters CUDA Python backend";
#else
  module.doc() = "NEPAdapters cpu Python backend";
#endif

  py::class_<BackendInfo>(module, "BackendInfo", py::module_local())
      .def_readonly("name", &BackendInfo::name)
      .def_readonly("version", &BackendInfo::version)
      .def_readonly("capabilities", &BackendInfo::capabilities);

  py::class_<PyModel>(module, "Model", py::module_local())
      .def(
          "calculate",
          &PyModel::calculate,
          py::arg("types"),
          py::arg("boxes"),
          py::arg("positions"),
          py::arg("atom_counts"),
          py::arg("pbc") = py::make_tuple(1, 1, 1))
      .def(
          "find_force",
          &PyModel::find_force,
          py::arg("types"),
          py::arg("positions"),
          py::arg("box"),
          py::arg("pbc") = py::make_tuple(1, 1, 1))
      .def(
          "calculate_spin",
          &PyModel::calculate_spin,
          py::arg("types"),
          py::arg("boxes"),
          py::arg("positions"),
          py::arg("spins"),
          py::arg("atom_counts"),
          py::arg("pbc") = py::make_tuple(1, 1, 1))
      .def(
          "descriptors",
          &PyModel::descriptors,
          py::arg("types"),
          py::arg("boxes"),
          py::arg("positions"),
          py::arg("atom_counts"),
          py::arg("pbc") = py::make_tuple(1, 1, 1))
      .def(
          "descriptors_spin",
          &PyModel::descriptors_spin,
          py::arg("types"),
          py::arg("boxes"),
          py::arg("positions"),
          py::arg("spins"),
          py::arg("atom_counts"),
          py::arg("pbc") = py::make_tuple(1, 1, 1))
      .def("close", &PyModel::close)
      .def(
          "__enter__",
          [](PyModel& model) -> PyModel& { return model; },
          py::return_value_policy::reference_internal)
      .def("__exit__", [](PyModel& model, py::object, py::object, py::object) {
        model.close();
      })
      .def("model_info", &PyModel::model_info);

#if defined(NEP_ADAPTERS_PYTHON_GPU_MODULE)
  module.def("register_cuda", []() {
    if (nepa_register_cuda_engine() != 1) {
      throw std::runtime_error("failed to register cuda engine");
    }
  });
#else
  module.def("register_cpu", []() {
    if (nepa_register_cpu_engine() != 1) {
      throw std::runtime_error("failed to register cpu engine");
    }
  });
#endif

  module.def("backend_count", []() { return nepa_backend_count(); });

  module.def("backend_info", [](int index) {
    NepaBackendInfo info{};
    check_status(nepa_backend_info(index, &info));
    return BackendInfo{
        info.name == nullptr ? "" : info.name,
        info.version == nullptr ? "" : info.version,
        info.capabilities};
  });

  module.def("load_model", [](const std::string& backend_name, const std::string& model_path) {
#if defined(NEP_ADAPTERS_PYTHON_GPU_MODULE)
    if (backend_name != "cuda") {
      throw std::invalid_argument("nep_gpu only supports backend='cuda'");
    }
    if (nepa_register_cuda_engine() != 1) {
      throw std::runtime_error("failed to register cuda engine");
    }
#else
    if (backend_name != "cpu") {
      throw std::invalid_argument("nep_cpu only supports backend='cpu'");
    }
    if (backend_name == "cpu") {
      if (nepa_register_cpu_engine() != 1) {
        throw std::runtime_error("failed to register cpu engine");
      }
    }
#endif

    NepaModel* model = nullptr;
    check_status(nepa_load_model(backend_name.c_str(), model_path.c_str(), &model));
    return PyModel(model);
  });
}
