#include "nep_adapters/api.h"
#include "nep_adapters/engines/cpu_nep3.hpp"

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
    throw std::runtime_error(nepa_status_message(status));
  }
}

struct BackendInfo {
  std::string name;
  std::string version;
  std::uint64_t capabilities = 0;
};

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

    const auto structure_count = static_cast<std::int32_t>(atom_count_info.shape[0]);
    const auto total_atoms = static_cast<std::int32_t>(type_info.shape[0]);
    if (structure_count <= 0 || total_atoms <= 0) {
      throw std::invalid_argument("empty batches are not supported");
    }

    const std::int32_t* atom_count_ptr =
        static_cast<const std::int32_t*>(atom_count_info.ptr);
    std::vector<std::int32_t> offsets(static_cast<std::size_t>(structure_count), 0);
    std::int32_t cursor = 0;
    for (std::int32_t i = 0; i < structure_count; ++i) {
      if (atom_count_ptr[i] <= 0) {
        throw std::invalid_argument("atom_counts must be positive");
      }
      offsets[static_cast<std::size_t>(i)] = cursor;
      cursor += atom_count_ptr[i];
    }
    if (cursor != total_atoms) {
      throw std::invalid_argument("sum(atom_counts) must equal len(types)");
    }

    if (!((box_info.ndim == 1 && box_info.size == 9) ||
          (box_info.ndim == 2 && box_info.shape[0] == structure_count &&
           box_info.shape[1] == 9))) {
      throw std::invalid_argument("boxes must have shape (9,) or (nstructures, 9)");
    }

    std::vector<double> boxes_for_batch;
    const double* boxes_ptr = static_cast<const double*>(box_info.ptr);
    if (box_info.ndim == 1 && structure_count > 1) {
      boxes_for_batch.resize(static_cast<std::size_t>(structure_count) * 9);
      for (std::int32_t i = 0; i < structure_count; ++i) {
        std::copy(boxes_ptr, boxes_ptr + 9, boxes_for_batch.data() + 9 * i);
      }
      boxes_ptr = boxes_for_batch.data();
    }

    std::vector<std::int32_t> default_pbc(static_cast<std::size_t>(structure_count) * 3, 1);
    const std::int32_t* pbc_ptr = default_pbc.data();
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> pbc_array;
    if (!pbc_object.is_none()) {
      pbc_array = py::cast<
          py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>>(
          pbc_object);
      py::buffer_info pbc_info = pbc_array.request();
      if (!((pbc_info.ndim == 1 && pbc_info.size == 3) ||
            (pbc_info.ndim == 2 && pbc_info.shape[0] == structure_count &&
             pbc_info.shape[1] == 3))) {
        throw std::invalid_argument("pbc must have shape (3,) or (nstructures, 3)");
      }
      if (pbc_info.ndim == 1 && structure_count > 1) {
        default_pbc.clear();
        default_pbc.resize(static_cast<std::size_t>(structure_count) * 3);
        const auto* one_pbc = static_cast<const std::int32_t*>(pbc_info.ptr);
        for (std::int32_t i = 0; i < structure_count; ++i) {
          std::copy(one_pbc, one_pbc + 3, default_pbc.data() + 3 * i);
        }
        pbc_ptr = default_pbc.data();
      } else {
        pbc_ptr = static_cast<const std::int32_t*>(pbc_info.ptr);
      }
    }

    py::array_t<double> potentials(static_cast<py::ssize_t>(total_atoms));
    py::array_t<double> forces({total_atoms, static_cast<std::int32_t>(3)});
    py::array_t<double> virials({total_atoms, static_cast<std::int32_t>(9)});
    std::vector<double> energies(static_cast<std::size_t>(structure_count), 0.0);
    std::vector<double> structure_virials(static_cast<std::size_t>(structure_count) * 9, 0.0);

    NepaStructureBatch batch{};
    batch.num_structures = structure_count;
    batch.total_atoms = total_atoms;
    batch.atom_counts = atom_count_ptr;
    batch.atom_offsets = offsets.data();
    batch.types = static_cast<const std::int32_t*>(type_info.ptr);
    batch.positions_aos3 = static_cast<const double*>(position_info.ptr);
    batch.boxes_row_major9 = boxes_ptr;
    batch.pbc_flags3 = pbc_ptr;

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
    std::int32_t default_pbc[] = {1, 1, 1};
    const std::int32_t* pbc_ptr = default_pbc;

    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> pbc_array;
    if (!pbc_object.is_none()) {
      pbc_array = py::cast<
          py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>>(
          pbc_object);
      py::buffer_info pbc_info = pbc_array.request();
      if (pbc_info.size != 3) {
        throw std::invalid_argument("pbc must contain 3 values");
      }
      pbc_ptr = static_cast<const std::int32_t*>(pbc_info.ptr);
    }

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

PYBIND11_MODULE(_native, module) {
  module.doc() = "pybind11 frontend for NEPAdapters";

  py::class_<BackendInfo>(module, "BackendInfo")
      .def_readonly("name", &BackendInfo::name)
      .def_readonly("version", &BackendInfo::version)
      .def_readonly("capabilities", &BackendInfo::capabilities);

  py::class_<PyModel>(module, "Model")
      .def(
          "calculate",
          &PyModel::calculate,
          py::arg("types"),
          py::arg("boxes"),
          py::arg("positions"),
          py::arg("atom_counts"),
          py::arg("pbc") = py::none())
      .def(
          "find_force",
          &PyModel::find_force,
          py::arg("types"),
          py::arg("positions"),
          py::arg("box"),
          py::arg("pbc") = py::none())
      .def("close", &PyModel::close)
      .def(
          "__enter__",
          [](PyModel& model) -> PyModel& { return model; },
          py::return_value_policy::reference_internal)
      .def("__exit__", [](PyModel& model, py::object, py::object, py::object) {
        model.close();
      })
      .def("model_info", &PyModel::model_info);

  module.def("register_cpu_nep3", []() {
    if (nepa_register_cpu_nep3_engine() != 1) {
      throw std::runtime_error("failed to register cpu_nep3 engine");
    }
  });

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
    if (backend_name == "cpu_nep3") {
      if (nepa_register_cpu_nep3_engine() != 1) {
        throw std::runtime_error("failed to register cpu_nep3 engine");
      }
    }

    NepaModel* model = nullptr;
    check_status(nepa_load_model(backend_name.c_str(), model_path.c_str(), &model));
    return PyModel(model);
  });
}
