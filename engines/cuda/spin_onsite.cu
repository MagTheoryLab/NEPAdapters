#include "device_operations.hpp"
#include "simulation_box_device.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>
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
constexpr int kSpinPrimitiveCount =
    61 + kSpinChiralOReducedCount + kSpinChiralHReducedCount;
constexpr int kSpinPrimitiveSlots = 88;
constexpr int kSpinChiralQohCount = 300;

__device__ __constant__ unsigned short kSpinChiralQohPacked[kSpinChiralQohCount] = {
    20, 23, 29, 35, 37, 46, 52, 55, 61, 67, 69, 78,
    86, 88, 92, 99, 101, 110, 118, 120, 124, 132, 135, 141,
    145, 146, 153, 154, 278, 280, 284, 288, 289, 290, 297, 298,
    299, 310, 312, 316, 320, 321, 322, 329, 330, 331, 340, 343,
    349, 352, 353, 354, 361, 362, 363, 372, 375, 381, 390, 392,
    396, 403, 405, 414, 528, 529, 530, 537, 538, 539, 550, 552,
    556, 560, 561, 562, 569, 570, 571, 582, 584, 588, 595, 597,
    606, 614, 616, 620, 627, 629, 638, 640, 641, 642, 649, 650,
    651, 660, 663, 669, 772, 775, 781, 800, 801, 802, 809, 810,
    811, 822, 824, 828, 832, 833, 834, 841, 842, 843, 852, 855,
    861, 864, 865, 866, 873, 874, 875, 884, 887, 893, 902, 904,
    908, 915, 917, 926, 1030, 1032, 1036, 1059, 1061, 1070, 1076, 1079,
    1085, 1091, 1093, 1102, 1110, 1112, 1116, 1123, 1125, 1134, 1142, 1144,
    1148, 1156, 1159, 1165, 1168, 1170, 1177, 1179, 1280, 1281, 1282, 1289,
    1290, 1291, 1316, 1319, 1325, 1331, 1333, 1342, 1348, 1351, 1357, 1360,
    1361, 1362, 1369, 1370, 1371, 1380, 1383, 1389, 1392, 1393, 1394, 1401,
    1402, 1403, 1411, 1413, 1422, 1430, 1432, 1436, 1539, 1541, 1550, 1552,
    1553, 1554, 1561, 1562, 1563, 1584, 1585, 1586, 1593, 1594, 1595, 1606,
    1608, 1612, 1619, 1621, 1630, 1638, 1640, 1644, 1651, 1653, 1662, 1664,
    1665, 1666, 1673, 1674, 1675, 1684, 1687, 1693, 1792, 1793, 1794, 1801,
    1802, 1803, 1811, 1813, 1822, 1843, 1845, 1854, 1860, 1863, 1869, 1872,
    1873, 1874, 1881, 1882, 1883, 1892, 1895, 1901, 1904, 1905, 1906, 1913,
    1914, 1915, 1923, 1925, 1934, 1942, 1944, 1948, 2054, 2056, 2060, 2068,
    2071, 2077, 2100, 2103, 2109, 2115, 2117, 2126, 2134, 2136, 2140, 2147,
    2149, 2158, 2166, 2168, 2172, 2180, 2183, 2189, 2192, 2193, 2202, 2203,
};

__device__ __constant__ double kSpinChiralQohCoeff[kSpinChiralQohCount] = {
    -1.0 / 7.0, -1.0 / 7.0, 6.0 / 7.0, 1.0 / 7.0, 1.0 / 7.0, -6.0 / 7.0,
    4.0 / 7.0, -3.0 / 7.0, -3.0 / 7.0, -4.0 / 7.0, 3.0 / 7.0, 3.0 / 7.0,
    -2.0 / 7.0, -2.0 / 7.0, 12.0 / 7.0, 1.0 / 7.0, -6.0 / 7.0, 15.0 / 7.0,
    2.0 / 7.0, 2.0 / 7.0, -12.0 / 7.0, -1.0 / 7.0, 6.0 / 7.0, -15.0 / 7.0,
    2.0 / 7.0, -2.0 / 7.0, -12.0 / 7.0, 12.0 / 7.0, 4.0 / 7.0, -3.0 / 7.0,
    -3.0 / 7.0, -1.0 / 35.0, 4.0 / 35.0, 4.0 / 35.0, 3.0 / 35.0, 3.0 / 35.0,
    -27.0 / 35.0, -1.0 / 7.0, -1.0 / 7.0, 6.0 / 7.0, 4.0 / 35.0, 4.0 / 35.0,
    -1.0 / 35.0, -27.0 / 35.0, 3.0 / 35.0, 3.0 / 35.0, -2.0 / 7.0, -2.0 / 7.0,
    12.0 / 7.0, -1.0 / 35.0, -16.0 / 35.0, -11.0 / 35.0, 18.0 / 35.0, -12.0 / 35.0,
    78.0 / 35.0, 2.0 / 7.0, 2.0 / 7.0, -12.0 / 7.0, -11.0 / 7.0, 10.0 / 7.0,
    3.0 / 7.0, 4.0 / 7.0, -10.0 / 7.0, 18.0 / 7.0, 1.0 / 35.0, -4.0 / 35.0,
    -4.0 / 35.0, -3.0 / 35.0, -3.0 / 35.0, 27.0 / 35.0, 3.0 / 7.0, -4.0 / 7.0,
    3.0 / 7.0, -4.0 / 35.0, 1.0 / 35.0, -4.0 / 35.0, -3.0 / 35.0, 27.0 / 35.0,
    -3.0 / 35.0, 1.0 / 7.0, 1.0 / 7.0, -6.0 / 7.0, -2.0 / 7.0, -2.0 / 7.0,
    12.0 / 7.0, -10.0 / 7.0, 11.0 / 7.0, -3.0 / 7.0, 2.0 / 7.0, 2.0 / 7.0,
    -12.0 / 7.0, 1.0 / 35.0, 11.0 / 35.0, 16.0 / 35.0, 12.0 / 35.0, -18.0 / 35.0,
    -78.0 / 35.0, -4.0 / 7.0, 10.0 / 7.0, -18.0 / 7.0, -4.0 / 7.0, 3.0 / 7.0,
    3.0 / 7.0, -4.0 / 35.0, 1.0 / 35.0, -4.0 / 35.0, -3.0 / 35.0, 27.0 / 35.0,
    -3.0 / 35.0, 2.0 / 7.0, 2.0 / 7.0, -12.0 / 7.0, 16.0 / 35.0, 1.0 / 35.0,
    11.0 / 35.0, -18.0 / 35.0, -78.0 / 35.0, 12.0 / 35.0, 1.0 / 7.0, 1.0 / 7.0,
    -6.0 / 7.0, -4.0 / 35.0, -4.0 / 35.0, 1.0 / 35.0, 27.0 / 35.0, -3.0 / 35.0,
    -3.0 / 35.0, 11.0 / 7.0, -10.0 / 7.0, -3.0 / 7.0, -2.0 / 7.0, -2.0 / 7.0,
    12.0 / 7.0, 10.0 / 7.0, -4.0 / 7.0, -18.0 / 7.0, 1.0 / 7.0, 1.0 / 7.0,
    -6.0 / 7.0, -1.0 / 7.0, -1.0 / 7.0, 6.0 / 7.0, 2.0 / 7.0, 2.0 / 7.0,
    -12.0 / 7.0, 6.0 / 7.0, -1.0 / 7.0, -15.0 / 7.0, -4.0 / 7.0, 3.0 / 7.0,
    3.0 / 7.0, -3.0 / 7.0, 4.0 / 7.0, -3.0 / 7.0, 1.0 / 7.0, -6.0 / 7.0,
    15.0 / 7.0, -2.0 / 7.0, -2.0 / 7.0, 12.0 / 7.0, -2.0 / 7.0, 2.0 / 7.0,
    12.0 / 7.0, -12.0 / 7.0, 4.0 / 35.0, -1.0 / 35.0, 4.0 / 35.0, 3.0 / 35.0,
    -27.0 / 35.0, 3.0 / 35.0, -3.0 / 7.0, 4.0 / 7.0, -3.0 / 7.0, 2.0 / 7.0,
    2.0 / 7.0, -12.0 / 7.0, 10.0 / 7.0, -11.0 / 7.0, 3.0 / 7.0, -1.0 / 35.0,
    4.0 / 35.0, 4.0 / 35.0, 3.0 / 35.0, 3.0 / 35.0, -27.0 / 35.0, -1.0 / 7.0,
    -1.0 / 7.0, 6.0 / 7.0, -11.0 / 35.0, -1.0 / 35.0, -16.0 / 35.0, -12.0 / 35.0,
    78.0 / 35.0, 18.0 / 35.0, -2.0 / 7.0, -2.0 / 7.0, 12.0 / 7.0, 4.0 / 7.0,
    -10.0 / 7.0, 18.0 / 7.0, 4.0 / 7.0, -3.0 / 7.0, -3.0 / 7.0, 4.0 / 35.0,
    4.0 / 35.0, -1.0 / 35.0, -27.0 / 35.0, 3.0 / 35.0, 3.0 / 35.0, -16.0 / 35.0,
    -11.0 / 35.0, -1.0 / 35.0, 78.0 / 35.0, 18.0 / 35.0, -12.0 / 35.0, -2.0 / 7.0,
    -2.0 / 7.0, 12.0 / 7.0, -11.0 / 7.0, 10.0 / 7.0, 3.0 / 7.0, 2.0 / 7.0,
    2.0 / 7.0, -12.0 / 7.0, -1.0 / 7.0, -1.0 / 7.0, 6.0 / 7.0, 4.0 / 35.0,
    -1.0 / 35.0, 4.0 / 35.0, 3.0 / 35.0, -27.0 / 35.0, 3.0 / 35.0, -10.0 / 7.0,
    4.0 / 7.0, 18.0 / 7.0, -4.0 / 35.0, -4.0 / 35.0, 1.0 / 35.0, 27.0 / 35.0,
    -3.0 / 35.0, -3.0 / 35.0, 3.0 / 7.0, -4.0 / 7.0, 3.0 / 7.0, -10.0 / 7.0,
    11.0 / 7.0, -3.0 / 7.0, -2.0 / 7.0, -2.0 / 7.0, 12.0 / 7.0, 11.0 / 35.0,
    16.0 / 35.0, 1.0 / 35.0, -78.0 / 35.0, 12.0 / 35.0, -18.0 / 35.0, 2.0 / 7.0,
    2.0 / 7.0, -12.0 / 7.0, 1.0 / 35.0, -4.0 / 35.0, -4.0 / 35.0, -3.0 / 35.0,
    -3.0 / 35.0, 27.0 / 35.0, 1.0 / 7.0, 1.0 / 7.0, -6.0 / 7.0, 10.0 / 7.0,
    -4.0 / 7.0, -18.0 / 7.0, -1.0 / 7.0, -1.0 / 7.0, 6.0 / 7.0, 1.0 / 7.0,
    1.0 / 7.0, -6.0 / 7.0, -6.0 / 7.0, 1.0 / 7.0, 15.0 / 7.0, -2.0 / 7.0,
    -2.0 / 7.0, 12.0 / 7.0, 6.0 / 7.0, -1.0 / 7.0, -15.0 / 7.0, 2.0 / 7.0,
    2.0 / 7.0, -12.0 / 7.0, -3.0 / 7.0, 4.0 / 7.0, -3.0 / 7.0, 3.0 / 7.0,
    -4.0 / 7.0, 3.0 / 7.0, 2.0 / 7.0, -2.0 / 7.0, -12.0 / 7.0, 12.0 / 7.0,
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

template <bool AtomMajor>
void launch_spin_density_forces_c4_l4(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceWorkspaceView& view,
    bool accumulate_virial,
    int blocks,
    int threads) {
  if constexpr (AtomMajor) {
    if (accumulate_virial) {
      prepare_spin_density_pulls_c4_l4<true><<<blocks, threads>>>(
          atom_count,
          static_cast<int>(view.atom_capacity),
          protocol.struct_descriptor_dim,
          view.spins_soa3,
          view.fp,
          view.spin_density_rho0,
          view.spin_density_l1_rdot,
          view.spin_density_l1_cross,
          view.spin_density_l1_stf,
          view.spin_density_angular2,
          view.spin_density_angular3,
          view.spin_density_angular4,
          view.spin_density_geom,
          view.spin_density_rho0_dot,
          view.spin_density_raw1,
          view.spin_density_raw1_dot,
          view.mforce_soa3);
    }
  } else {
    prepare_spin_density_pulls_c4_l4<AtomMajor><<<blocks, threads>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        protocol.struct_descriptor_dim,
        view.spins_soa3,
        view.fp,
        view.spin_density_rho0,
        view.spin_density_l1_rdot,
        view.spin_density_l1_cross,
        view.spin_density_l1_stf,
        view.spin_density_angular2,
        view.spin_density_angular3,
        view.spin_density_angular4,
        view.spin_density_geom,
        view.spin_density_rho0_dot,
        view.spin_density_raw1,
        view.spin_density_raw1_dot,
        view.mforce_soa3);
  }
  if (accumulate_virial) {
    accumulate_spin_density_forces_c4_l4_pull<AtomMajor><<<blocks, threads>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        protocol.struct_descriptor_dim,
        static_cast<float>(protocol.spin_cutoff_radial),
        view.spins_soa3,
        view.nn_radial,
        view.nl_radial_slot_major,
        view.fp,
        view.spin_edge_dx,
        view.spin_edge_dy,
        view.spin_edge_dz,
        view.spin_edge_dist,
        view.spin_edge_weights,
        view.spin_edge_weight_derivatives,
        view.spin_density_rho0,
        view.spin_density_l1_stf,
        view.spin_density_angular2,
        view.spin_density_angular3,
        view.spin_density_angular4,
        view.spin_density_raw1,
        view.spin_density_rho0_dot,
        view.spin_density_raw1_dot,
        view.force_soa3,
        view.mforce_soa3,
        accumulate_virial,
        view.virial_soa9);
  } else {
    accumulate_spin_density_forces_c4_l4_block_f32<AtomMajor>
        <<<atom_count, threads>>>(
            atom_count,
            static_cast<int>(view.atom_capacity),
            protocol.struct_descriptor_dim,
            static_cast<float>(protocol.spin_cutoff_radial),
            view.spins_soa3,
            view.nn_radial,
            view.nl_radial_slot_major,
            view.fp,
            view.spin_edge_dx,
            view.spin_edge_dy,
            view.spin_edge_dz,
            view.spin_edge_dist,
            view.spin_edge_weights,
            view.spin_edge_weight_derivatives,
            view.spin_density_rho0,
            view.spin_density_l1_rdot,
            view.spin_density_l1_cross,
            view.spin_density_l1_stf,
            view.spin_density_angular2,
            view.spin_density_angular3,
            view.spin_density_angular4,
            view.spin_density_geom,
            view.spin_density_rho0_dot,
            view.spin_density_raw1,
            view.spin_density_raw1_dot,
            view.force_soa3,
            view.mforce_soa3);
  }
}

template <bool AtomMajor>
void launch_spin_chiral_forces_c4_l4(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceWorkspaceView& view,
    bool accumulate_virial,
    int blocks,
    int threads) {
  accumulate_spin_chiral_forces_c4_l4_cached_f32<AtomMajor><<<blocks, threads>>>(
      atom_count,
      static_cast<int>(view.atom_capacity),
      protocol.struct_descriptor_dim,
      static_cast<float>(protocol.spin_cutoff_radial),
      view.spins_soa3,
      view.nn_radial,
      view.nl_radial_slot_major,
      view.fp,
      view.spin_edge_dx,
      view.spin_edge_dy,
      view.spin_edge_dz,
      view.spin_edge_dist,
      view.spin_edge_weights,
      view.spin_edge_weight_derivatives,
      view.spin_density_geom,
      view.spin_chiral_polar,
      view.spin_chiral_octupoles_raw,
      view.spin_chiral_hexadecapoles_raw,
      view.spin_chiral_chirals,
      view.spin_chiral_pseudodevs,
      view.force_soa3,
      view.mforce_soa3,
      accumulate_virial,
      view.virial_soa9);
}

}  // namespace

void build_spin_descriptors_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace) {
  require(protocol.spin_mode != 0, "spin descriptor requires spin model");
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
  require(view.spin_density_l1_rdot != nullptr,
          "workspace missing spin density l1 rdot");
  require(view.spin_density_l1_cross != nullptr,
          "workspace missing spin density l1 cross");
  require(view.spin_density_l1_stf != nullptr,
          "workspace missing spin density l1 stf");
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
  require(view.spin_edge_dx != nullptr, "workspace missing spin edge dx");
  require(view.spin_edge_dy != nullptr, "workspace missing spin edge dy");
  require(view.spin_edge_dz != nullptr, "workspace missing spin edge dz");
  require(view.spin_edge_dist != nullptr, "workspace missing spin edge dist");
  require(view.spin_edge_weights != nullptr, "workspace missing spin edge weights");
  require(view.spin_edge_weight_derivatives != nullptr,
          "workspace missing spin edge weight derivatives");
  require(protocol.spin_chiral == 0 || view.spin_chiral_polar != nullptr,
          "workspace missing spin chiral polar");
  require(protocol.spin_chiral == 0 || view.spin_chiral_octupoles_raw != nullptr,
          "workspace missing spin chiral octupoles raw");
  require(protocol.spin_chiral == 0 || view.spin_chiral_hexadecapoles_raw != nullptr,
          "workspace missing spin chiral hexadecapoles raw");
  require(protocol.spin_chiral == 0 || view.spin_chiral_chirals != nullptr,
          "workspace missing spin chiral chirals");
  require(protocol.spin_chiral == 0 || view.spin_chiral_pseudodevs != nullptr,
          "workspace missing spin chiral pseudodevs");
  require(model_view.descriptor_coefficients != nullptr,
          "model missing descriptor coefficients");
  require(model_view.descriptor_coefficients_count >= protocol.descriptor_parameter_count,
          "model descriptor coefficient buffer is too small");
  const bool use_c4_l4_chiral =
      protocol.spin_compress == 4 && protocol.spin_basis_size == 3 &&
      protocol.spin_l_max == 4 && protocol.spin_chiral != 0;
  if (use_c4_l4_chiral) {
    const int edge_threads = 256;
    const int edge_items = atom_count * protocol.neighbor_capacity_radial;
    const int edge_blocks = (edge_items + edge_threads - 1) / edge_threads;
    if (edge_blocks > 0) {
      precompute_spin_edge_weights_c4<<<edge_blocks, edge_threads>>>(
          atom_count,
          static_cast<int>(view.atom_capacity),
          protocol.neighbor_capacity_radial,
          protocol.num_types,
          static_cast<float>(protocol.spin_cutoff_radial),
          box,
          view.types,
          view.positions_soa3,
          view.nn_radial,
          view.nl_radial_slot_major,
          model_view.descriptor_coefficients,
          static_cast<int>(protocol.ordinary_descriptor_parameter_count),
          view.spin_edge_dx,
          view.spin_edge_dy,
          view.spin_edge_dz,
          view.spin_edge_dist,
          view.spin_edge_weights,
          view.spin_edge_weight_derivatives);
    }
    const int threads = 128;
    const int work_items = atom_count * 4;
    const int blocks = (work_items + threads - 1) / threads;
    if (blocks > 0) {
      if (protocol.neighbor_capacity_radial <= 32) {
        build_spin_primitive_cache_c4_l4_warp<32, true><<<atom_count, 128>>>(
            atom_count,
            static_cast<int>(view.atom_capacity),
            protocol.struct_descriptor_dim,
            static_cast<float>(protocol.spin_cutoff_radial),
            view.spins_soa3,
            view.nn_radial,
            view.nl_radial_slot_major,
            view.spin_edge_dx,
            view.spin_edge_dy,
            view.spin_edge_dz,
            view.spin_edge_dist,
            view.spin_edge_weights,
            view.spin_density_rho0,
            view.spin_density_raw1,
            view.spin_density_l1_rdot,
            view.spin_density_l1_cross,
            view.spin_density_l1_stf,
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
      } else if (protocol.neighbor_capacity_radial <= kSpinPrimitiveSlots) {
        build_spin_primitive_cache_c4_l4_warp<kSpinPrimitiveSlots, false>
            <<<atom_count, 128>>>(
            atom_count,
            static_cast<int>(view.atom_capacity),
            protocol.struct_descriptor_dim,
            static_cast<float>(protocol.spin_cutoff_radial),
            view.spins_soa3,
            view.nn_radial,
            view.nl_radial_slot_major,
            view.spin_edge_dx,
            view.spin_edge_dy,
            view.spin_edge_dz,
            view.spin_edge_dist,
            view.spin_edge_weights,
            view.spin_density_rho0,
            view.spin_density_raw1,
            view.spin_density_l1_rdot,
            view.spin_density_l1_cross,
            view.spin_density_l1_stf,
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
      } else {
        build_spin_descriptors_c4_l4_basic<<<blocks, threads>>>(
            atom_count,
            static_cast<int>(view.atom_capacity),
            protocol.struct_descriptor_dim,
            static_cast<float>(protocol.spin_cutoff_radial),
            view.spins_soa3,
            view.nn_radial,
            view.nl_radial_slot_major,
            view.spin_edge_dx,
            view.spin_edge_dy,
            view.spin_edge_dz,
            view.spin_edge_dist,
            view.spin_edge_weights,
            view.spin_density_rho0,
            view.spin_density_raw1,
            view.spin_density_l1_rdot,
            view.spin_density_l1_cross,
            view.spin_density_l1_stf,
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
      }
      if (protocol.neighbor_capacity_radial <= 32) {
        build_spin_chiral_finalize_c4_l4_f32<true><<<blocks, threads>>>(
            atom_count,
            static_cast<int>(view.atom_capacity),
            protocol.struct_descriptor_dim,
            static_cast<float>(protocol.spin_cutoff_radial),
            view.spins_soa3,
            view.nn_radial,
            view.nl_radial_slot_major,
            view.spin_edge_dx,
            view.spin_edge_dy,
            view.spin_edge_dz,
            view.spin_edge_dist,
            view.spin_edge_weights,
            view.spin_density_geom,
            view.spin_chiral_polar,
            view.spin_chiral_octupoles_raw,
            view.spin_chiral_hexadecapoles_raw,
            view.spin_chiral_chirals,
            view.spin_chiral_pseudodevs,
            view.descriptors);
      } else {
        build_spin_chiral_finalize_c4_l4_f32<false><<<blocks, threads>>>(
            atom_count,
            static_cast<int>(view.atom_capacity),
            protocol.struct_descriptor_dim,
            static_cast<float>(protocol.spin_cutoff_radial),
            view.spins_soa3,
            view.nn_radial,
            view.nl_radial_slot_major,
            view.spin_edge_dx,
            view.spin_edge_dy,
            view.spin_edge_dz,
            view.spin_edge_dist,
            view.spin_edge_weights,
            view.spin_density_geom,
            view.spin_chiral_polar,
            view.spin_chiral_octupoles_raw,
            view.spin_chiral_hexadecapoles_raw,
            view.spin_chiral_chirals,
            view.spin_chiral_pseudodevs,
            view.descriptors);
      }
    }
  } else {
    const int threads = 32;
    const int blocks = (atom_count + threads - 1) / threads;
    if (blocks > 0) {
    build_spin_descriptors<<<blocks, threads>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        protocol.struct_descriptor_dim,
        protocol.spin_descriptor_dim,
        protocol.num_types,
        protocol.spin_compress,
        protocol.spin_basis_size,
        protocol.spin_l_max,
        protocol.spin_chiral,
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
        view.spin_density_l1_rdot,
        view.spin_density_l1_cross,
        view.spin_density_l1_stf,
        view.spin_density_angular2,
        view.spin_density_angular3,
        view.spin_density_angular4,
        view.spin_density_geom,
        view.spin_density_rho0_dot,
        view.spin_density_raw1_dot,
        view.spin_chiral_polar,
        view.spin_chiral_octupoles_raw,
        view.spin_chiral_hexadecapoles_raw,
        view.spin_chiral_chirals,
        view.spin_chiral_pseudodevs,
        view.descriptors);
    }
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

static void accumulate_spin_scalar_forces_impl(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool accumulate_virial) {
  require(protocol.spin_mode != 0, "spin scalar force requires spin model");
  require(protocol.spin_compress > 0 &&
              protocol.spin_compress <= kMaxSpinCompress,
          "spin scalar force kernel supports spin_compress <= 4");
  require(protocol.spin_basis_size >= 0 &&
              protocol.spin_basis_size + 1 <= kMaxSpinBasis,
          "spin scalar force kernel supports spin_basis_size + 1 <= 8");
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
  require(model_view.descriptor_coefficients != nullptr,
          "model missing descriptor coefficients");
  require(model_view.descriptor_coefficients_count >= protocol.descriptor_parameter_count,
          "model descriptor coefficient buffer is too small");
  const bool use_cached_geometry =
      protocol.spin_compress == 4 && protocol.spin_basis_size == 3 &&
      protocol.spin_l_max == 4 && protocol.spin_chiral != 0;
  require(!use_cached_geometry || view.spin_edge_dx != nullptr,
          "workspace missing spin edge dx");
  require(!use_cached_geometry || view.spin_edge_dy != nullptr,
          "workspace missing spin edge dy");
  require(!use_cached_geometry || view.spin_edge_dz != nullptr,
          "workspace missing spin edge dz");
  require(!use_cached_geometry || view.spin_edge_dist != nullptr,
          "workspace missing spin edge dist");
  if (use_cached_geometry && !accumulate_virial) {
    return;
  }
  const int threads = 32;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0) {
    accumulate_spin_scalar_forces<<<blocks, threads>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        protocol.struct_descriptor_dim,
        protocol.num_types,
        protocol.spin_compress,
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
        view.spin_edge_dx,
        view.spin_edge_dy,
        view.spin_edge_dz,
        view.spin_edge_dist,
        view.spin_edge_weights,
        view.spin_edge_weight_derivatives,
        use_cached_geometry,
        view.force_soa3,
        view.mforce_soa3,
        accumulate_virial,
        view.virial_soa9);
  }
  check_cuda(cudaGetLastError(), "accumulate spin scalar forces");
}

static void accumulate_spin_density_forces_impl(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool accumulate_virial) {
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
  require(view.spin_density_rho0 != nullptr, "workspace missing spin density rho0");
  require(view.spin_density_raw1 != nullptr, "workspace missing spin density raw1");
  require(view.spin_density_l1_rdot != nullptr,
          "workspace missing spin density l1 rdot");
  require(view.spin_density_l1_cross != nullptr,
          "workspace missing spin density l1 cross");
  require(view.spin_density_l1_stf != nullptr,
          "workspace missing spin density l1 stf");
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
  require(view.spin_edge_weights != nullptr, "workspace missing spin edge weights");
  require(view.spin_edge_weight_derivatives != nullptr,
          "workspace missing spin edge weight derivatives");
  require(model_view.descriptor_coefficients != nullptr,
          "model missing descriptor coefficients");
  require(model_view.descriptor_coefficients_count >= protocol.descriptor_parameter_count,
          "model descriptor coefficient buffer is too small");
  const bool use_cached_geometry =
      protocol.spin_compress == 4 && protocol.spin_basis_size == 3 &&
      protocol.spin_l_max == 4 && protocol.spin_chiral != 0;
  require(!use_cached_geometry || view.spin_edge_dx != nullptr,
          "workspace missing spin edge dx");
  require(!use_cached_geometry || view.spin_edge_dy != nullptr,
          "workspace missing spin edge dy");
  require(!use_cached_geometry || view.spin_edge_dz != nullptr,
          "workspace missing spin edge dz");
  require(!use_cached_geometry || view.spin_edge_dist != nullptr,
          "workspace missing spin edge dist");
  const int threads = 32;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0 && use_cached_geometry) {
    if (protocol.neighbor_capacity_radial <= 32) {
      launch_spin_density_forces_c4_l4<true>(
          protocol, atom_count, view, accumulate_virial, blocks, threads);
    } else {
      launch_spin_density_forces_c4_l4<false>(
          protocol, atom_count, view, accumulate_virial, blocks, threads);
    }
  } else if (blocks > 0) {
    accumulate_spin_density_forces<<<blocks, threads>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        protocol.struct_descriptor_dim,
        protocol.num_types,
        protocol.spin_compress,
        protocol.spin_basis_size,
        protocol.spin_l_max,
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
        view.spin_edge_dx,
        view.spin_edge_dy,
        view.spin_edge_dz,
        view.spin_edge_dist,
        view.spin_edge_weights,
        view.spin_edge_weight_derivatives,
        use_cached_geometry,
        view.spin_density_rho0,
        view.spin_density_raw1,
        view.spin_density_l1_rdot,
        view.spin_density_l1_cross,
        view.spin_density_l1_stf,
        view.spin_density_angular2,
        view.spin_density_angular3,
        view.spin_density_angular4,
        view.spin_density_geom,
        view.spin_density_rho0_dot,
        view.spin_density_raw1_dot,
        view.force_soa3,
        view.mforce_soa3,
        accumulate_virial,
        view.virial_soa9);
  }
  check_cuda(cudaGetLastError(), "accumulate spin density forces");
}

static void accumulate_spin_chiral_polar_forces_impl(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool accumulate_virial) {
  require(protocol.spin_mode != 0, "spin chiral polar force requires spin model");
  require(protocol.spin_chiral != 0, "spin chiral polar force requires chiral model");
  require(protocol.spin_compress > 0 &&
              protocol.spin_compress <= kMaxSpinCompress,
          "spin chiral polar kernel supports spin_compress <= 4");
  require(protocol.spin_basis_size >= 0 &&
              protocol.spin_basis_size + 1 <= kMaxSpinBasis,
          "spin chiral polar kernel supports spin_basis_size + 1 <= 8");
  require(protocol.spin_l_max >= 0 && protocol.spin_l_max <= 4,
          "spin chiral polar kernel supports spin_l_max <= 4");
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
  const bool use_cached_geometry =
      protocol.spin_compress == 4 && protocol.spin_basis_size == 3 &&
      protocol.spin_l_max == 4 && protocol.spin_chiral != 0;
  require(view.spin_density_geom != nullptr, "workspace missing spin density geom");
  require(view.spin_chiral_polar != nullptr, "workspace missing spin chiral polar");
  require(view.spin_chiral_octupoles_raw != nullptr,
          "workspace missing spin chiral octupoles raw");
  require(view.spin_chiral_hexadecapoles_raw != nullptr,
          "workspace missing spin chiral hexadecapoles raw");
  require(view.spin_chiral_chirals != nullptr, "workspace missing spin chiral chirals");
  require(view.spin_chiral_pseudodevs != nullptr,
          "workspace missing spin chiral pseudodevs");
  require(view.spin_edge_weights != nullptr, "workspace missing spin edge weights");
  require(view.spin_edge_weight_derivatives != nullptr,
          "workspace missing spin edge weight derivatives");
  require(model_view.descriptor_coefficients != nullptr,
          "model missing descriptor coefficients");
  require(model_view.descriptor_coefficients_count >= protocol.descriptor_parameter_count,
          "model descriptor coefficient buffer is too small");
  require(!use_cached_geometry || view.spin_edge_dx != nullptr,
          "workspace missing spin edge dx");
  require(!use_cached_geometry || view.spin_edge_dy != nullptr,
          "workspace missing spin edge dy");
  require(!use_cached_geometry || view.spin_edge_dz != nullptr,
          "workspace missing spin edge dz");
  require(!use_cached_geometry || view.spin_edge_dist != nullptr,
          "workspace missing spin edge dist");
  const int threads = 32;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0 && use_cached_geometry) {
    if (protocol.neighbor_capacity_radial <= 32) {
      launch_spin_chiral_forces_c4_l4<true>(
          protocol, atom_count, view, accumulate_virial, blocks, threads);
    } else {
      launch_spin_chiral_forces_c4_l4<false>(
          protocol, atom_count, view, accumulate_virial, blocks, threads);
    }
  } else if (blocks > 0) {
    accumulate_spin_chiral_forces<<<blocks, threads>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        protocol.struct_descriptor_dim,
        protocol.num_types,
        protocol.spin_compress,
        protocol.spin_basis_size,
        protocol.spin_l_max,
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
        view.spin_edge_weights,
        view.spin_edge_weight_derivatives,
        view.spin_density_geom,
        view.spin_chiral_polar,
        view.spin_chiral_octupoles_raw,
        view.spin_chiral_hexadecapoles_raw,
        view.spin_chiral_chirals,
        view.spin_chiral_pseudodevs,
        view.force_soa3,
        view.mforce_soa3,
        accumulate_virial,
        view.virial_soa9);
  }
  check_cuda(cudaGetLastError(), "accumulate spin chiral polar forces");
}

void accumulate_spin_forces_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool accumulate_virial,
    SpinForceTimings* timings) {
  require(protocol.spin_mode != 0, "spin force pipeline requires spin model");
  SpinForceTimings ignored_timings;
  SpinForceTimings& measured = timings == nullptr ? ignored_timings : *timings;
  PhaseTimer timer(timings != nullptr);

  accumulate_spin_onsite_mforces_impl(protocol, atom_count, workspace);
  timer.split(measured.onsite_ms);

  accumulate_spin_scalar_forces_impl(
      protocol, atom_count, box, model, workspace, accumulate_virial);
  timer.split(measured.scalar_ms);

  accumulate_spin_density_forces_impl(
      protocol, atom_count, box, model, workspace, accumulate_virial);
  timer.split(measured.density_ms);

  if (protocol.spin_chiral != 0) {
    accumulate_spin_chiral_polar_forces_impl(
        protocol, atom_count, box, model, workspace, accumulate_virial);
  }
  timer.split(measured.chiral_ms);
}

}  // namespace nep_adapters::cuda_backend
