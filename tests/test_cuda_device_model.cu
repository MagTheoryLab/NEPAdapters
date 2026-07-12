#include "device_operations.hpp"
#include "device_model.hpp"
#include "device_workspace.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

bool copy_matches(const float* device, const std::vector<float>& expected) {
  std::vector<float> actual(expected.size(), 0.0f);
  const cudaError_t status = cudaMemcpy(
      actual.data(),
      device,
      actual.size() * sizeof(float),
      cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    return false;
  }
  return actual == expected;
}

std::vector<int> copy_ints(const int* device, std::size_t count) {
  std::vector<int> actual(count, 0);
  const cudaError_t status = cudaMemcpy(
      actual.data(),
      device,
      actual.size() * sizeof(int),
      cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    return {};
  }
  return actual;
}

std::vector<float> copy_floats(const float* device, std::size_t count) {
  std::vector<float> actual(count, 0.0f);
  const cudaError_t status = cudaMemcpy(
      actual.data(),
      device,
      actual.size() * sizeof(float),
      cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    return {};
  }
  return actual;
}

std::vector<double> copy_doubles(const double* device, std::size_t count) {
  std::vector<double> actual(count, 0.0);
  const cudaError_t status = cudaMemcpy(
      actual.data(),
      device,
      actual.size() * sizeof(double),
      cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    return {};
  }
  return actual;
}

double minimum_image_delta(double delta, double length, int pbc) {
  if (!pbc) {
    return delta;
  }
  return delta - std::nearbyint(delta / length) * length;
}

bool invert_row_major3(const double* matrix, double* inverse) {
  const double det =
      matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
      matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
      matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
  if (std::abs(det) <= 1.0e-12) {
    return false;
  }
  const double inv_det = 1.0 / det;
  inverse[0] = (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * inv_det;
  inverse[1] = (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * inv_det;
  inverse[2] = (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * inv_det;
  inverse[3] = (matrix[5] * matrix[6] - matrix[3] * matrix[8]) * inv_det;
  inverse[4] = (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * inv_det;
  inverse[5] = (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * inv_det;
  inverse[6] = (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * inv_det;
  inverse[7] = (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * inv_det;
  inverse[8] = (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * inv_det;
  return true;
}

nep_adapters::cuda_backend::SimulationBox make_simulation_box(
    const double* cell_row_major9,
    const int* pbc_flags3) {
  nep_adapters::cuda_backend::SimulationBox box;
  for (int component = 0; component < 9; ++component) {
    box.frac_to_cart[component] = cell_row_major9[component];
  }
  if (!invert_row_major3(box.frac_to_cart, box.cart_to_frac)) {
    throw std::runtime_error("singular test box");
  }
  box.pbc[0] = pbc_flags3[0];
  box.pbc[1] = pbc_flags3[1];
  box.pbc[2] = pbc_flags3[2];
  return box;
}

std::vector<double> minimum_image_delta_aos3(
    const std::vector<double>& positions_aos3,
    int center,
    int neighbor,
    const nep_adapters::cuda_backend::SimulationBox& box) {
  const double dx = positions_aos3[3 * static_cast<std::size_t>(neighbor)] -
                    positions_aos3[3 * static_cast<std::size_t>(center)];
  const double dy = positions_aos3[3 * static_cast<std::size_t>(neighbor) + 1] -
                    positions_aos3[3 * static_cast<std::size_t>(center) + 1];
  const double dz = positions_aos3[3 * static_cast<std::size_t>(neighbor) + 2] -
                    positions_aos3[3 * static_cast<std::size_t>(center) + 2];
  double sx = box.cart_to_frac[0] * dx + box.cart_to_frac[1] * dy +
              box.cart_to_frac[2] * dz;
  double sy = box.cart_to_frac[3] * dx + box.cart_to_frac[4] * dy +
              box.cart_to_frac[5] * dz;
  double sz = box.cart_to_frac[6] * dx + box.cart_to_frac[7] * dy +
              box.cart_to_frac[8] * dz;
  if (box.pbc[0]) {
    sx -= std::nearbyint(sx);
  }
  if (box.pbc[1]) {
    sy -= std::nearbyint(sy);
  }
  if (box.pbc[2]) {
    sz -= std::nearbyint(sz);
  }
  return {
      box.frac_to_cart[0] * sx + box.frac_to_cart[1] * sy +
          box.frac_to_cart[2] * sz,
      box.frac_to_cart[3] * sx + box.frac_to_cart[4] * sy +
          box.frac_to_cart[5] * sz,
      box.frac_to_cart[6] * sx + box.frac_to_cart[7] * sy +
          box.frac_to_cart[8] * sz};
}

std::vector<std::vector<int>> brute_force_neighbors(
    const std::vector<double>& positions_aos3,
    double cutoff,
    const nep_adapters::cuda_backend::SimulationBox& box) {
  const int atom_count = static_cast<int>(positions_aos3.size() / 3);
  std::vector<std::vector<int>> neighbors(static_cast<std::size_t>(atom_count));
  const double cutoff_sq = cutoff * cutoff;
  for (int i = 0; i < atom_count; ++i) {
    for (int j = 0; j < atom_count; ++j) {
      if (i == j) {
        continue;
      }
      const std::vector<double> delta =
          minimum_image_delta_aos3(positions_aos3, i, j, box);
      if (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2] <
          cutoff_sq) {
        neighbors[static_cast<std::size_t>(i)].push_back(j);
      }
    }
    std::sort(
        neighbors[static_cast<std::size_t>(i)].begin(),
        neighbors[static_cast<std::size_t>(i)].end());
  }
  return neighbors;
}

std::vector<std::vector<int>> brute_force_neighbors(
    const std::vector<double>& positions_aos3,
    double cutoff,
    double lx,
    double ly,
    double lz,
    int pbc_x,
    int pbc_y,
    int pbc_z) {
  const int atom_count = static_cast<int>(positions_aos3.size() / 3);
  std::vector<std::vector<int>> neighbors(static_cast<std::size_t>(atom_count));
  const double cutoff_sq = cutoff * cutoff;
  for (int i = 0; i < atom_count; ++i) {
    for (int j = 0; j < atom_count; ++j) {
      if (i == j) {
        continue;
      }
      double dx = positions_aos3[3 * static_cast<std::size_t>(i)] -
                  positions_aos3[3 * static_cast<std::size_t>(j)];
      double dy = positions_aos3[3 * static_cast<std::size_t>(i) + 1] -
                  positions_aos3[3 * static_cast<std::size_t>(j) + 1];
      double dz = positions_aos3[3 * static_cast<std::size_t>(i) + 2] -
                  positions_aos3[3 * static_cast<std::size_t>(j) + 2];
      dx = minimum_image_delta(dx, lx, pbc_x);
      dy = minimum_image_delta(dy, ly, pbc_y);
      dz = minimum_image_delta(dz, lz, pbc_z);
      if (dx * dx + dy * dy + dz * dz < cutoff_sq) {
        neighbors[static_cast<std::size_t>(i)].push_back(j);
      }
    }
    std::sort(
        neighbors[static_cast<std::size_t>(i)].begin(),
        neighbors[static_cast<std::size_t>(i)].end());
  }
  return neighbors;
}

std::vector<double> minimum_image_delta_aos3(
    const std::vector<double>& positions_aos3,
    int center,
    int neighbor,
    double lx,
    double ly,
    double lz,
    int pbc_x,
    int pbc_y,
    int pbc_z) {
  double dx = positions_aos3[3 * static_cast<std::size_t>(neighbor)] -
              positions_aos3[3 * static_cast<std::size_t>(center)];
  double dy = positions_aos3[3 * static_cast<std::size_t>(neighbor) + 1] -
              positions_aos3[3 * static_cast<std::size_t>(center) + 1];
  double dz = positions_aos3[3 * static_cast<std::size_t>(neighbor) + 2] -
              positions_aos3[3 * static_cast<std::size_t>(center) + 2];
  dx = minimum_image_delta(dx, lx, pbc_x);
  dy = minimum_image_delta(dy, ly, pbc_y);
  dz = minimum_image_delta(dz, lz, pbc_z);
  return {dx, dy, dz};
}

bool close_float(float lhs, double rhs) {
  const double scale = std::max(1.0, std::abs(rhs));
  return std::abs(static_cast<double>(lhs) - rhs) < 1.0e-5 * scale;
}

float radial_cutoff(float rc, float r) {
  if (r >= rc) {
    return 0.0f;
  }
  return 0.5f * std::cos(3.1415927f * r / rc) + 0.5f;
}

std::vector<float> radial_basis_values(float rc, int basis_size, float r) {
  std::vector<float> values(static_cast<std::size_t>(basis_size) + 1, 0.0f);
  const float fc = radial_cutoff(rc, r);
  const float rcinv = 1.0f / rc;
  const float x = 2.0f * (r * rcinv - 1.0f) * (r * rcinv - 1.0f) - 1.0f;
  values[0] = fc;
  if (basis_size >= 1) {
    values[1] = (x + 1.0f) * 0.5f * fc;
  }
  float t_minus_2 = 1.0f;
  float t_minus_1 = x;
  for (int k = 2; k <= basis_size; ++k) {
    const float t = 2.0f * x * t_minus_1 - t_minus_2;
    t_minus_2 = t_minus_1;
    t_minus_1 = t;
    values[static_cast<std::size_t>(k)] = (t + 1.0f) * 0.5f * fc;
  }
  return values;
}

bool neighbor_sets_match(
    const std::vector<int>& counts,
    const std::vector<int>& slot_major_neighbors,
    int atom_stride,
    const std::vector<std::vector<int>>& expected) {
  if (counts.size() < expected.size()) {
    return false;
  }
  for (std::size_t atom = 0; atom < expected.size(); ++atom) {
    if (counts[atom] != static_cast<int>(expected[atom].size())) {
      return false;
    }
    std::vector<int> actual;
    actual.reserve(expected[atom].size());
    for (int slot = 0; slot < counts[atom]; ++slot) {
      actual.push_back(
          slot_major_neighbors[atom + static_cast<std::size_t>(atom_stride * slot)]);
    }
    std::sort(actual.begin(), actual.end());
    if (actual != expected[atom]) {
      return false;
    }
  }
  return true;
}

bool geometry_cache_matches(
    const std::vector<double>& positions_aos3,
    const std::vector<int>& radial_counts,
    const std::vector<int>& radial_neighbors,
    const std::vector<float>& r12_radial,
    const std::vector<int>& angular_counts,
    const std::vector<int>& angular_neighbors,
    const std::vector<float>& f12x,
    const std::vector<float>& f12y,
    const std::vector<float>& f12z,
    int atom_stride,
    double lx,
    double ly,
    double lz,
    int pbc_x,
    int pbc_y,
    int pbc_z) {
  const int atom_count = static_cast<int>(positions_aos3.size() / 3);
  for (int atom = 0; atom < atom_count; ++atom) {
    for (int slot = 0; slot < radial_counts[static_cast<std::size_t>(atom)]; ++slot) {
      const int neighbor =
          radial_neighbors[static_cast<std::size_t>(atom + atom_stride * slot)];
      const std::vector<double> delta = minimum_image_delta_aos3(
          positions_aos3,
          atom,
          neighbor,
          lx,
          ly,
          lz,
          pbc_x,
          pbc_y,
          pbc_z);
      const double r = std::sqrt(
          delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
      if (!close_float(r12_radial[static_cast<std::size_t>(atom + atom_stride * slot)], r)) {
        return false;
      }
    }
    for (int slot = 0; slot < angular_counts[static_cast<std::size_t>(atom)]; ++slot) {
      const int neighbor =
          angular_neighbors[static_cast<std::size_t>(atom + atom_stride * slot)];
      const std::vector<double> delta = minimum_image_delta_aos3(
          positions_aos3,
          atom,
          neighbor,
          lx,
          ly,
          lz,
          pbc_x,
          pbc_y,
          pbc_z);
      const std::size_t offset = static_cast<std::size_t>(atom + atom_stride * slot);
      if (!close_float(f12x[offset], delta[0]) ||
          !close_float(f12y[offset], delta[1]) ||
          !close_float(f12z[offset], delta[2])) {
        return false;
      }
    }
  }
  return true;
}

bool geometry_cache_matches(
    const std::vector<double>& positions_aos3,
    const std::vector<int>& radial_counts,
    const std::vector<int>& radial_neighbors,
    const std::vector<float>& r12_radial,
    const std::vector<int>& angular_counts,
    const std::vector<int>& angular_neighbors,
    const std::vector<float>& f12x,
    const std::vector<float>& f12y,
    const std::vector<float>& f12z,
    int atom_stride,
    const nep_adapters::cuda_backend::SimulationBox& box) {
  const int atom_count = static_cast<int>(positions_aos3.size() / 3);
  for (int atom = 0; atom < atom_count; ++atom) {
    for (int slot = 0; slot < radial_counts[static_cast<std::size_t>(atom)]; ++slot) {
      const int neighbor =
          radial_neighbors[static_cast<std::size_t>(atom + atom_stride * slot)];
      const std::vector<double> delta =
          minimum_image_delta_aos3(positions_aos3, atom, neighbor, box);
      const double r = std::sqrt(
          delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
      if (!close_float(r12_radial[static_cast<std::size_t>(atom + atom_stride * slot)], r)) {
        return false;
      }
    }
    for (int slot = 0; slot < angular_counts[static_cast<std::size_t>(atom)]; ++slot) {
      const int neighbor =
          angular_neighbors[static_cast<std::size_t>(atom + atom_stride * slot)];
      const std::vector<double> delta =
          minimum_image_delta_aos3(positions_aos3, atom, neighbor, box);
      const std::size_t offset = static_cast<std::size_t>(atom + atom_stride * slot);
      if (!close_float(f12x[offset], delta[0]) ||
          !close_float(f12y[offset], delta[1]) ||
          !close_float(f12z[offset], delta[2])) {
        return false;
      }
    }
  }
  return true;
}

bool radial_basis_cache_matches(
    const std::vector<int>& radial_counts,
    const std::vector<int>& radial_neighbors,
    const std::vector<float>& r12_radial,
    const std::vector<float>& fc_radial,
    const std::vector<float>& fn_radial,
    int atom_count,
    int atom_stride,
    int radial_capacity,
    int basis_size,
    float cutoff) {
  for (int atom = 0; atom < atom_count; ++atom) {
    for (int slot = 0; slot < radial_capacity; ++slot) {
      const std::size_t slot_offset =
          static_cast<std::size_t>(atom + atom_stride * slot);
      if (slot >= radial_counts[static_cast<std::size_t>(atom)]) {
        if (fc_radial[slot_offset] != 0.0f) {
          return false;
        }
        for (int k = 0; k <= basis_size; ++k) {
          const std::size_t basis_offset = static_cast<std::size_t>(
              atom + atom_stride * (slot + radial_capacity * k));
          if (fn_radial[basis_offset] != 0.0f) {
            return false;
          }
        }
        continue;
      }

      if (radial_neighbors[slot_offset] < 0) {
        return false;
      }
      const float r = r12_radial[slot_offset];
      const float fc = radial_cutoff(cutoff, r);
      if (!close_float(fc_radial[slot_offset], fc)) {
        return false;
      }
      const std::vector<float> fn = radial_basis_values(cutoff, basis_size, r);
      for (int k = 0; k <= basis_size; ++k) {
        const std::size_t basis_offset = static_cast<std::size_t>(
            atom + atom_stride * (slot + radial_capacity * k));
        if (!close_float(fn_radial[basis_offset], fn[static_cast<std::size_t>(k)])) {
          return false;
        }
      }
    }
  }
  return true;
}

bool angular_basis_cache_matches(
    const std::vector<int>& angular_counts,
    const std::vector<float>& f12x,
    const std::vector<float>& f12y,
    const std::vector<float>& f12z,
    const std::vector<float>& r12_angular,
    const std::vector<float>& fc_angular,
    const std::vector<float>& fn_angular,
    int atom_count,
    int atom_stride,
    int angular_capacity,
    int basis_size,
    float cutoff) {
  for (int atom = 0; atom < atom_count; ++atom) {
    for (int slot = 0; slot < angular_capacity; ++slot) {
      const std::size_t slot_offset =
          static_cast<std::size_t>(atom + atom_stride * slot);
      if (slot >= angular_counts[static_cast<std::size_t>(atom)]) {
        if (r12_angular[slot_offset] != 0.0f || fc_angular[slot_offset] != 0.0f) {
          std::fprintf(stderr, "unused angular basis mismatch atom=%d slot=%d\n", atom, slot);
          return false;
        }
        for (int k = 0; k <= basis_size; ++k) {
          const std::size_t basis_offset = static_cast<std::size_t>(
              atom + atom_stride * (slot + angular_capacity * k));
          if (fn_angular[basis_offset] != 0.0f) {
            std::fprintf(stderr, "unused angular fn mismatch atom=%d slot=%d k=%d\n", atom, slot, k);
            return false;
          }
        }
        continue;
      }

      const float r = std::sqrt(
          f12x[slot_offset] * f12x[slot_offset] +
          f12y[slot_offset] * f12y[slot_offset] +
          f12z[slot_offset] * f12z[slot_offset]);
      const float fc = radial_cutoff(cutoff, r);
      if (!close_float(r12_angular[slot_offset], r) ||
          !close_float(fc_angular[slot_offset], fc)) {
        std::fprintf(
            stderr,
            "angular basis scalar mismatch atom=%d slot=%d r=%g actual_r=%g fc=%g actual_fc=%g\n",
            atom,
            slot,
            static_cast<double>(r),
            static_cast<double>(r12_angular[slot_offset]),
            static_cast<double>(fc),
            static_cast<double>(fc_angular[slot_offset]));
        return false;
      }
      const std::vector<float> fn = radial_basis_values(cutoff, basis_size, r);
      for (int k = 0; k <= basis_size; ++k) {
        const std::size_t basis_offset = static_cast<std::size_t>(
            atom + atom_stride * (slot + angular_capacity * k));
        if (!close_float(fn_angular[basis_offset], fn[static_cast<std::size_t>(k)])) {
          std::fprintf(
              stderr,
              "angular fn mismatch atom=%d slot=%d k=%d expected=%g actual=%g\n",
              atom,
              slot,
              k,
              static_cast<double>(fn[static_cast<std::size_t>(k)]),
              static_cast<double>(fn_angular[basis_offset]));
          return false;
        }
      }
    }
  }
  return true;
}

bool radial_descriptors_match(
    const nep_adapters::cuda_backend::HostModelParameters& host,
    const std::vector<int>& types,
    const std::vector<int>& radial_counts,
    const std::vector<int>& radial_neighbors,
    const std::vector<float>& fn_radial,
    const std::vector<float>& descriptors,
    int atom_count,
    int atom_stride) {
  const auto& protocol = host.protocol;
  const int radial_capacity = protocol.neighbor_capacity_radial;
  const int type_pairs = protocol.num_types * protocol.num_types;
  for (int atom = 0; atom < atom_count; ++atom) {
    const int type1 = types[static_cast<std::size_t>(atom)];
    for (int n = 0; n <= protocol.n_max_radial; ++n) {
      float expected = 0.0f;
      for (int slot = 0; slot < radial_counts[static_cast<std::size_t>(atom)]; ++slot) {
        const std::size_t neighbor_offset =
            static_cast<std::size_t>(atom + atom_stride * slot);
        const int neighbor = radial_neighbors[neighbor_offset];
        const int type2 = types[static_cast<std::size_t>(neighbor)];
        const int type_pair = type1 * protocol.num_types + type2;
        for (int k = 0; k <= protocol.basis_size_radial; ++k) {
          const std::size_t coefficient_index = static_cast<std::size_t>(
              (n * (protocol.basis_size_radial + 1) + k) * type_pairs + type_pair);
          const std::size_t fn_index = static_cast<std::size_t>(
              atom + atom_stride * (slot + radial_capacity * k));
          expected += fn_radial[fn_index] *
                      host.descriptor_coefficients[coefficient_index];
        }
      }
      const std::size_t descriptor_offset =
          static_cast<std::size_t>(atom + atom_stride * n);
      if (!close_float(descriptors[descriptor_offset], expected)) {
        return false;
      }
    }
    for (int d = protocol.n_max_radial + 1; d < protocol.descriptor_dim; ++d) {
      const std::size_t descriptor_offset =
          static_cast<std::size_t>(atom + atom_stride * d);
      if (descriptors[descriptor_offset] != 0.0f) {
        return false;
      }
    }
  }
  return true;
}

void accumulate_s_l1(float x, float y, float z, float gn, float* s) {
  s[0] += z * gn;
  s[1] += x * gn;
  s[2] += y * gn;
}

void accumulate_s_l2(float x, float y, float z, float gn, float* s) {
  s[3] += (-1.0f + 3.0f * z * z) * gn;
  s[4] += z * x * gn;
  s[5] += z * y * gn;
  s[6] += (x * x - y * y) * gn;
  s[7] += (2.0f * x * y) * gn;
}

void accumulate_s_l3(float x, float y, float z, float gn, float* s) {
  const float x2_minus_y2 = x * x - y * y;
  const float two_xy = 2.0f * x * y;
  const float x3_minus_3xy2 = x * x2_minus_y2 - y * two_xy;
  const float three_x2y_minus_y3 = x * two_xy + y * x2_minus_y2;
  s[8] += (-3.0f * z + 5.0f * z * z * z) * gn;
  s[9] += (-1.0f + 5.0f * z * z) * x * gn;
  s[10] += (-1.0f + 5.0f * z * z) * y * gn;
  s[11] += z * x2_minus_y2 * gn;
  s[12] += z * two_xy * gn;
  s[13] += x3_minus_3xy2 * gn;
  s[14] += three_x2y_minus_y3 * gn;
}

void accumulate_s_l4(float x, float y, float z, float gn, float* s) {
  const float z2 = z * z;
  const float x2_minus_y2 = x * x - y * y;
  const float two_xy = 2.0f * x * y;
  const float x3_minus_3xy2 = x * x2_minus_y2 - y * two_xy;
  const float three_x2y_minus_y3 = x * two_xy + y * x2_minus_y2;
  const float x4_minus_6x2y2_plus_y4 =
      x * x3_minus_3xy2 - y * three_x2y_minus_y3;
  const float four_x3y_minus_4xy3 =
      x * three_x2y_minus_y3 + y * x3_minus_3xy2;
  s[15] += (3.0f - 30.0f * z2 + 35.0f * z2 * z2) * gn;
  s[16] += (-3.0f * z + 7.0f * z * z2) * x * gn;
  s[17] += (-3.0f * z + 7.0f * z * z2) * y * gn;
  s[18] += (-1.0f + 7.0f * z2) * x2_minus_y2 * gn;
  s[19] += (-1.0f + 7.0f * z2) * two_xy * gn;
  s[20] += z * x3_minus_3xy2 * gn;
  s[21] += z * three_x2y_minus_y3 * gn;
  s[22] += x4_minus_6x2y2_plus_y4 * gn;
  s[23] += four_x3y_minus_4xy3 * gn;
}

float find_q_l1(const float* s) {
  return 0.238732414637843f * s[0] * s[0] +
         2.0f *
             (0.119366207318922f * s[1] * s[1] +
              0.119366207318922f * s[2] * s[2]);
}

float find_q_l2(const float* s) {
  return 0.099471839432435f * s[3] * s[3] +
         2.0f *
             (0.596831036594608f * s[4] * s[4] +
              0.596831036594608f * s[5] * s[5] +
              0.149207759148652f * s[6] * s[6] +
              0.149207759148652f * s[7] * s[7]);
}

float find_q_l3(const float* s) {
  return 0.139260575205408f * s[8] * s[8] +
         2.0f *
             (0.104445431404056f * s[9] * s[9] +
              0.104445431404056f * s[10] * s[10] +
              1.044454314040563f * s[11] * s[11] +
              1.044454314040563f * s[12] * s[12] +
              0.174075719006761f * s[13] * s[13] +
              0.174075719006761f * s[14] * s[14]);
}

float find_q_l4(const float* s) {
  return 0.011190581936149f * s[15] * s[15] +
         2.0f *
             (0.223811638722978f * s[16] * s[16] +
              0.223811638722978f * s[17] * s[17] +
              0.111905819361489f * s[18] * s[18] +
              0.111905819361489f * s[19] * s[19] +
              1.566681471060845f * s[20] * s[20] +
              1.566681471060845f * s[21] * s[21] +
              0.195835183882606f * s[22] * s[22] +
              0.195835183882606f * s[23] * s[23]);
}

float find_q_222(const float* s) {
  return -0.007499480826664f * s[3] * s[3] * s[3] +
         -0.134990654879954f * s[3] * (s[4] * s[4] + s[5] * s[5]) +
         0.067495327439977f * s[3] * (s[6] * s[6] + s[7] * s[7]) +
         0.404971964639861f * s[6] * (s[5] * s[5] - s[4] * s[4]) +
         -0.809943929279723f * s[4] * s[5] * s[7];
}

float find_q_1111(const float* s) {
  const float s0_sq = s[0] * s[0];
  const float s12_sq = s[1] * s[1] + s[2] * s[2];
  return 0.026596810706114f * s0_sq * s0_sq +
         0.053193621412227f * s0_sq * s12_sq +
         0.026596810706114f * s12_sq * s12_sq;
}

float find_q_112(const float* s) {
  return 0.027493550848847f * s[0] * s[0] * s[3] +
         0.164961305093080f * s[0] * (s[1] * s[4] + s[2] * s[5]) +
         -0.013746775424423f * s[3] * (s[1] * s[1] + s[2] * s[2]) +
         0.041240326273270f * s[6] * (s[1] * s[1] - s[2] * s[2]) +
         0.082480652546540f * s[1] * s[2] * s[7];
}

float find_q_123(const float* s) {
  float value = 0.0f;
  value += -0.168362926992344f *
           (s[12] * s[2] * s[4] - s[11] * s[2] * s[5] +
            s[1] * s[11] * s[4] + s[1] * s[12] * s[5]);
  value += -0.084181463496172f * (s[0] * s[11] * s[6] + s[0] * s[12] * s[7]);
  value += -0.042090731748086f *
           (s[14] * s[2] * s[6] - s[13] * s[2] * s[7] +
            s[1] * s[13] * s[6] + s[1] * s[14] * s[7]);
  value += -0.067345170796937f * (s[10] * s[0] * s[5] + s[0] * s[4] * s[9]);
  value += -0.016836292699234f *
           (s[10] * s[2] * s[3] + s[0] * s[3] * s[8] +
            s[1] * s[3] * s[9]);
  value += -0.008418146349617f *
           (s[10] * s[2] * s[6] - s[10] * s[1] * s[7] -
            s[2] * s[7] * s[9] - s[1] * s[6] * s[9]);
  value += -0.033672585398469f * (-s[2] * s[5] * s[8] - s[1] * s[4] * s[8]);
  return value;
}

float find_q_233(const float* s) {
  float value = 0.0f;
  value += 0.008572620635186f * (s[3] * s[8] * s[8]);
  value += 0.009644198214584f * (s[10] * s[10] * s[3] + s[3] * s[9] * s[9]);
  value += 0.019288396429168f * (-s[10] * s[10] * s[6] + s[6] * s[9] * s[9]);
  value += 0.025717861905558f * (s[4] * s[8] * s[9] + s[10] * s[5] * s[8]);
  value += 0.026789439484956f * (-s[13] * s[13] * s[3] - s[14] * s[14] * s[3]);
  value += 0.032147327381947f *
           (-s[14] * s[7] * s[9] - s[13] * s[6] * s[9] -
            s[10] * s[14] * s[6] + s[10] * s[13] * s[7]);
  value += 0.038576792858337f * (s[10] * s[7] * s[9]);
  value += 0.128589309527790f * (-s[11] * s[6] * s[8] - s[12] * s[7] * s[8]);
  value += 0.192883964291685f *
           (s[11] * s[4] * s[9] + s[12] * s[5] * s[9] +
            s[10] * s[12] * s[4] - s[10] * s[11] * s[5]);
  value += 0.321473273819474f *
           (s[12] * s[14] * s[4] + s[11] * s[14] * s[5] +
            s[13] * s[11] * s[4] - s[13] * s[12] * s[5]);
  return value;
}

float find_q_134(const float* s) {
  return 0.003645164295772f * (-s[10] * s[15] * s[2] - s[1] * s[15] * s[9]) +
         0.004860219061029f * (s[0] * s[15] * s[8]) +
         0.006075273826286f *
             (-s[1] * s[13] * s[18] - s[1] * s[14] * s[19] -
              s[2] * s[14] * s[18] + s[2] * s[13] * s[19]) +
         0.018225821478859f *
             (-s[10] * s[18] * s[2] + s[1] * s[10] * s[19] +
              s[1] * s[18] * s[9] + s[2] * s[19] * s[9]) +
         0.024301095305146f * (s[1] * s[16] * s[8] + s[2] * s[17] * s[8]) +
         0.036451642957719f *
             (s[0] * s[10] * s[17] + s[0] * s[16] * s[9] -
              s[1] * s[11] * s[16] - s[1] * s[12] * s[17] -
              s[2] * s[12] * s[16] + s[2] * s[11] * s[17]) +
         0.042526916784005f *
             (s[1] * s[13] * s[22] + s[1] * s[14] * s[23] -
              s[2] * s[14] * s[22] + s[2] * s[13] * s[23]) +
         0.072903285915437f * (s[0] * s[11] * s[18] + s[0] * s[12] * s[19]) +
         0.085053833568010f * (s[0] * s[13] * s[20] + s[0] * s[14] * s[21]) +
         0.255161500704030f *
             (s[1] * s[11] * s[20] + s[1] * s[12] * s[21] -
              s[2] * s[12] * s[20] + s[2] * s[11] * s[21]);
}

bool angular_descriptors_match(
    const nep_adapters::cuda_backend::HostModelParameters& host,
    const std::vector<int>& types,
    const std::vector<int>& angular_counts,
    const std::vector<int>& angular_neighbors,
    const std::vector<float>& f12x,
    const std::vector<float>& f12y,
    const std::vector<float>& f12z,
    const std::vector<float>& r12_angular,
    const std::vector<float>& fn_angular,
    const std::vector<float>& sum_fxyz,
    const std::vector<float>& descriptors,
    int atom_count,
    int atom_stride) {
  const auto& protocol = host.protocol;
  const int angular_capacity = protocol.neighbor_capacity_angular;
  const int type_pairs = protocol.num_types * protocol.num_types;
  const int radial_dim = protocol.n_max_radial + 1;
  const int abc_count = protocol.body_channels.abc_count();
  const std::size_t angular_offset = host.descriptor_layout.angular_offset;
  for (int atom = 0; atom < atom_count; ++atom) {
    const int type1 = types[static_cast<std::size_t>(atom)];
    for (int n = 0; n <= protocol.n_max_angular; ++n) {
      float s[24] = {0.0f};
      for (int slot = 0; slot < angular_counts[static_cast<std::size_t>(atom)]; ++slot) {
        const std::size_t slot_offset =
            static_cast<std::size_t>(atom + atom_stride * slot);
        const int neighbor = angular_neighbors[slot_offset];
        const int type2 = types[static_cast<std::size_t>(neighbor)];
        const int type_pair = type1 * protocol.num_types + type2;
        float gn = 0.0f;
        for (int k = 0; k <= protocol.basis_size_angular; ++k) {
          const std::size_t coefficient_index =
              angular_offset +
              static_cast<std::size_t>(
                  (n * (protocol.basis_size_angular + 1) + k) * type_pairs +
                  type_pair);
          const std::size_t fn_index = static_cast<std::size_t>(
              atom + atom_stride * (slot + angular_capacity * k));
          gn += fn_angular[fn_index] *
                host.descriptor_coefficients[coefficient_index];
        }
        const float rinv = 1.0f / r12_angular[slot_offset];
        const float x = f12x[slot_offset] * rinv;
        const float y = f12y[slot_offset] * rinv;
        const float z = f12z[slot_offset] * rinv;
        accumulate_s_l1(x, y, z, gn, s);
        accumulate_s_l2(x, y, z, gn, s);
        accumulate_s_l3(x, y, z, gn, s);
        accumulate_s_l4(x, y, z, gn, s);
      }
      for (int abc = 0; abc < abc_count; ++abc) {
        const std::size_t offset =
            static_cast<std::size_t>(atom + atom_stride * (n * abc_count + abc));
        if (!close_float(sum_fxyz[offset], s[abc])) {
          std::fprintf(
              stderr,
              "sum_fxyz mismatch atom=%d n=%d abc=%d expected=%g actual=%g\n",
              atom,
              n,
              abc,
              static_cast<double>(s[abc]),
              static_cast<double>(sum_fxyz[offset]));
          return false;
        }
      }
      const std::size_t l1_offset =
          static_cast<std::size_t>(atom + atom_stride * (radial_dim + n));
      const std::size_t l2_offset = static_cast<std::size_t>(
          atom + atom_stride * (radial_dim + (protocol.n_max_angular + 1) + n));
      if (!close_float(descriptors[l1_offset], find_q_l1(s)) ||
          !close_float(descriptors[l2_offset], find_q_l2(s))) {
        std::fprintf(
            stderr,
            "angular descriptor mismatch atom=%d n=%d l1 expected=%g actual=%g l2 expected=%g actual=%g\n",
            atom,
            n,
            static_cast<double>(find_q_l1(s)),
            static_cast<double>(descriptors[l1_offset]),
            static_cast<double>(find_q_l2(s)),
            static_cast<double>(descriptors[l2_offset]));
        return false;
      }
      const std::size_t l3_offset = static_cast<std::size_t>(
          atom + atom_stride * (radial_dim + 2 * (protocol.n_max_angular + 1) + n));
      const std::size_t l4_offset = static_cast<std::size_t>(
          atom + atom_stride * (radial_dim + 3 * (protocol.n_max_angular + 1) + n));
      if (!close_float(descriptors[l3_offset], find_q_l3(s)) ||
          !close_float(descriptors[l4_offset], find_q_l4(s))) {
        std::fprintf(
            stderr,
            "angular L3/L4 mismatch atom=%d n=%d l3 expected=%g actual=%g l4 expected=%g actual=%g\n",
            atom,
            n,
            static_cast<double>(find_q_l3(s)),
            static_cast<double>(descriptors[l3_offset]),
            static_cast<double>(find_q_l4(s)),
            static_cast<double>(descriptors[l4_offset]));
        return false;
      }
      int channel = protocol.body_channels.l_max_3body;
      if (protocol.body_channels.has_q_222) {
        const std::size_t offset = static_cast<std::size_t>(
            atom + atom_stride *
                       (radial_dim + channel * (protocol.n_max_angular + 1) + n));
        if (!close_float(descriptors[offset], find_q_222(s))) {
          std::fprintf(
              stderr,
              "q222 mismatch atom=%d n=%d expected=%g actual=%g\n",
              atom,
              n,
              static_cast<double>(find_q_222(s)),
              static_cast<double>(descriptors[offset]));
          return false;
        }
        ++channel;
      }
      if (protocol.body_channels.has_q_1111) {
        const std::size_t offset = static_cast<std::size_t>(
            atom + atom_stride *
                       (radial_dim + channel * (protocol.n_max_angular + 1) + n));
        if (!close_float(descriptors[offset], find_q_1111(s))) {
          std::fprintf(
              stderr,
              "q1111 mismatch atom=%d n=%d expected=%g actual=%g\n",
              atom,
              n,
              static_cast<double>(find_q_1111(s)),
              static_cast<double>(descriptors[offset]));
          return false;
        }
        ++channel;
      }
      if (protocol.body_channels.has_q_112) {
        const std::size_t offset = static_cast<std::size_t>(
            atom + atom_stride *
                       (radial_dim + channel * (protocol.n_max_angular + 1) + n));
        if (!close_float(descriptors[offset], find_q_112(s))) {
          std::fprintf(stderr, "q112 mismatch atom=%d n=%d\n", atom, n);
          return false;
        }
        ++channel;
      }
      if (protocol.body_channels.has_q_123) {
        const std::size_t offset = static_cast<std::size_t>(
            atom + atom_stride *
                       (radial_dim + channel * (protocol.n_max_angular + 1) + n));
        if (!close_float(descriptors[offset], find_q_123(s))) {
          std::fprintf(stderr, "q123 mismatch atom=%d n=%d\n", atom, n);
          return false;
        }
        ++channel;
      }
      if (protocol.body_channels.has_q_233) {
        const std::size_t offset = static_cast<std::size_t>(
            atom + atom_stride *
                       (radial_dim + channel * (protocol.n_max_angular + 1) + n));
        if (!close_float(descriptors[offset], find_q_233(s))) {
          std::fprintf(stderr, "q233 mismatch atom=%d n=%d\n", atom, n);
          return false;
        }
        ++channel;
      }
      if (protocol.body_channels.has_q_134) {
        const std::size_t offset = static_cast<std::size_t>(
            atom + atom_stride *
                       (radial_dim + channel * (protocol.n_max_angular + 1) + n));
        if (!close_float(descriptors[offset], find_q_134(s))) {
          std::fprintf(stderr, "q134 mismatch atom=%d n=%d\n", atom, n);
          return false;
        }
      }
    }
  }
  return true;
}

bool ann_energy_matches(
    const nep_adapters::cuda_backend::HostModelParameters& host,
    const std::vector<int>& types,
    const std::vector<float>& descriptors,
    const std::vector<double>& potential,
    const std::vector<float>& fp,
    int atom_count,
    int atom_stride) {
  const auto& protocol = host.protocol;
  const int descriptor_dim = protocol.descriptor_dim;
  const int hidden = protocol.hidden_neurons;
  const int type_block_size =
      hidden * descriptor_dim + hidden + hidden;
  const float* b1 = host.ann_type_major.data() +
                    protocol.num_types * static_cast<std::size_t>(type_block_size);

  for (int atom = 0; atom < atom_count; ++atom) {
    const int type = types[static_cast<std::size_t>(atom)];
    const float* w0 = host.ann_type_major.data() +
                      type * static_cast<std::size_t>(type_block_size);
    const float* b0 = w0 + hidden * descriptor_dim;
    const float* w1 = b0 + hidden;
    float expected_energy = 0.0f;
    std::vector<float> expected_fp(static_cast<std::size_t>(descriptor_dim), 0.0f);

    for (int neuron = 0; neuron < hidden; ++neuron) {
      float w0_times_q = 0.0f;
      for (int d = 0; d < descriptor_dim; ++d) {
        const float q = descriptors[static_cast<std::size_t>(atom + atom_stride * d)] *
                        host.q_scaler[static_cast<std::size_t>(d)];
        w0_times_q += w0[neuron * descriptor_dim + d] * q;
      }
      const float x1 = std::tanh(w0_times_q - b0[neuron]);
      const float tanh_derivative = 1.0f - x1 * x1;
      expected_energy += w1[neuron] * x1;
      for (int d = 0; d < descriptor_dim; ++d) {
        expected_fp[static_cast<std::size_t>(d)] +=
            w1[neuron] * tanh_derivative * w0[neuron * descriptor_dim + d] *
            host.q_scaler[static_cast<std::size_t>(d)];
      }
    }
    expected_energy -= b1[0];

    if (!close_float(static_cast<float>(potential[static_cast<std::size_t>(atom)]),
                     expected_energy)) {
      std::fprintf(
          stderr,
          "ANN energy mismatch atom=%d expected=%g actual=%g\n",
          atom,
          static_cast<double>(expected_energy),
          potential[static_cast<std::size_t>(atom)]);
      return false;
    }
    for (int d = 0; d < descriptor_dim; ++d) {
      const std::size_t offset =
          static_cast<std::size_t>(atom + atom_stride * d);
      if (!close_float(fp[offset], expected_fp[static_cast<std::size_t>(d)])) {
        std::fprintf(
            stderr,
            "ANN fp mismatch atom=%d d=%d expected=%g actual=%g\n",
            atom,
            d,
            static_cast<double>(expected_fp[static_cast<std::size_t>(d)]),
            static_cast<double>(fp[offset]));
        return false;
      }
    }
  }
  return true;
}

__global__ void parameter_smoke_kernel(
    nep_adapters::cuda_backend::DeviceModelView model,
    float* output) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float value = 0.0f;
    if (model.ann_type_major_count > 0) {
      value += model.ann_type_major[0];
    }
    if (model.descriptor_coefficients_count > 1) {
      value += 2.0f * model.descriptor_coefficients[1];
    }
    if (model.q_scaler_count > 0) {
      value += 3.0f * model.q_scaler[0];
    }
    output[0] = value;
  }
}

__global__ void internal_workspace_smoke_kernel(
    nep_adapters::cuda_backend::DeviceModelView model,
    nep_adapters::cuda_backend::DeviceWorkspaceView workspace,
    float* output) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const int slot_one_neighbor =
        workspace.nl_radial_slot_major[workspace.atom_capacity];
    const float value =
        model.ann_type_major[0] +
        static_cast<float>(workspace.types[1]) +
        static_cast<float>(10 * workspace.atom_to_structure[2]) +
        static_cast<float>(workspace.positions_soa3[1]) +
        static_cast<float>(workspace.boxes_row_major9[0]) +
        static_cast<float>(workspace.pbc_flags3[2]) +
        static_cast<float>(slot_one_neighbor);
    workspace.fp[0] = value;
    output[0] = value;
  }
}

__global__ void external_workspace_smoke_kernel(
    nep_adapters::cuda_backend::DeviceModelView model,
    nep_adapters::cuda_backend::DeviceWorkspaceView workspace,
    float* output) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const int active_atom = workspace.active_atom_indices[1];
    const int slot_one_neighbor =
        workspace.nl_angular_slot_major[workspace.atom_capacity + active_atom];
    const float value =
        model.q_scaler[0] +
        static_cast<float>(workspace.types[active_atom]) +
        static_cast<float>(workspace.nn_angular[active_atom]) +
        static_cast<float>(slot_one_neighbor) +
        static_cast<float>(workspace.neighbor_source);
    workspace.fp[1] = value;
    output[0] = value;
  }
}

}  // namespace

int main() {
  const std::string model_path =
      (std::filesystem::temp_directory_path() / "cuda_device_model.nep").string();
  {
    std::ofstream out(model_path);
    out << "nep4 1 C\n"
        << "cutoff 5 4 8 6\n"
        << "n_max 0 0\n"
        << "basis_size 0 0\n"
        << "l_max 1 0 0\n"
        << "ANN 2 0\n";
    for (int value = 1; value <= 13; ++value) {
      out << value << "\n";
    }
  }

  const nep_adapters::cuda_backend::HostModelParameters host =
      nep_adapters::cuda_backend::load_host_model_parameters(model_path);
  nep_adapters::cuda_backend::DeviceModel device(host);
  const nep_adapters::cuda_backend::DeviceModelUploadSummary summary =
      device.upload_summary();

  if (summary.ann_type_major_bytes != host.ann_type_major.size() * sizeof(float) ||
      summary.ann_type_major_qscaled_bytes !=
          host.ann_type_major_qscaled.size() * sizeof(float) ||
      summary.descriptor_coefficients_bytes !=
          host.descriptor_coefficients.size() * sizeof(float) ||
      summary.descriptor_coefficients_type_pair_major_bytes !=
          host.descriptor_coefficients_type_pair_major.size() * sizeof(float) ||
      summary.q_scaler_bytes != host.q_scaler.size() * sizeof(float) ||
      summary.atomic_numbers_bytes != host.atomic_numbers.size() * sizeof(int) ||
      summary.total_bytes !=
          (host.ann_type_major.size() + host.ann_type_major_qscaled.size() +
           host.descriptor_coefficients.size() +
           host.descriptor_coefficients_type_pair_major.size() + host.q_scaler.size()) *
                  sizeof(float) +
              host.atomic_numbers.size() * sizeof(int)) {
    std::fprintf(stderr, "device model upload summary contract failed\n");
    return EXIT_FAILURE;
  }

  if (!copy_matches(device.ann_type_major_device(), host.ann_type_major) ||
      !copy_matches(
          device.view().ann_type_major_qscaled,
          host.ann_type_major_qscaled) ||
      !copy_matches(
          device.descriptor_coefficients_device(),
          host.descriptor_coefficients) ||
      !copy_matches(
          device.view().descriptor_coefficients_type_pair_major,
          host.descriptor_coefficients_type_pair_major) ||
      !copy_matches(device.q_scaler_device(), host.q_scaler) ||
      copy_ints(device.view().atomic_numbers, host.atomic_numbers.size()) !=
          host.atomic_numbers) {
    std::fprintf(stderr, "device model upload copy contract failed\n");
    return EXIT_FAILURE;
  }

  float* device_output = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(float)) !=
      cudaSuccess) {
    std::fprintf(stderr, "device output allocation failed\n");
    return EXIT_FAILURE;
  }
  parameter_smoke_kernel<<<1, 32>>>(device.view(), device_output);
  if (cudaGetLastError() != cudaSuccess) {
    std::fprintf(stderr, "parameter smoke kernel launch failed\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }
  float smoke_value = 0.0f;
  if (cudaMemcpy(
          &smoke_value,
          device_output,
          sizeof(float),
          cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::fprintf(stderr, "parameter smoke copyback failed\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }
  if (smoke_value != host.ann_type_major[0] +
                         2.0f * host.descriptor_coefficients[1] +
                         3.0f * host.q_scaler[0]) {
    std::fprintf(stderr, "parameter smoke value mismatch\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }

  const nep_adapters::cuda_backend::WorkspacePlan internal_plan =
      nep_adapters::cuda_backend::make_internal_neighbor_workspace_plan(
          host.protocol, 4, 2);
  nep_adapters::cuda_backend::DeviceWorkspace internal_workspace(internal_plan);
  if (internal_workspace.summary().array_count != internal_plan.arrays.size() ||
      internal_workspace.summary().total_bytes != internal_plan.total_bytes() ||
      internal_workspace.view().boxes_row_major9 == nullptr ||
      internal_workspace.view().active_atom_indices != nullptr) {
    std::fprintf(stderr, "internal workspace contract failed\n");
    return EXIT_FAILURE;
  }
  int batch_counts[] = {2, 2};
  int batch_offsets[] = {0, 2};
  int batch_types[] = {0, 1, 0, 1};
  double batch_positions[] = {
      0.0, 10.0, 20.0,
      1.0, 11.0, 21.0,
      2.0, 12.0, 22.0,
      3.0, 13.0, 23.0,
  };
  double batch_boxes[] = {
      1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
      2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0};
  int batch_pbc[] = {1, 1, 1, 0, 0, 0};
  NepaStructureBatch batch{};
  batch.num_structures = 2;
  batch.total_atoms = 4;
  batch.atom_counts = batch_counts;
  batch.atom_offsets = batch_offsets;
  batch.types = batch_types;
  batch.positions_aos3 = batch_positions;
  batch.boxes_row_major9 = batch_boxes;
  batch.pbc_flags3 = batch_pbc;
  nep_adapters::cuda_backend::stage_batch_on_device(batch, internal_workspace);
  internal_workspace.copy_int_to_device("nn_radial", {2, 1, 0, 0});
  internal_workspace.copy_int_to_device(
      "nl_radial_slot_major",
      {1, 0, 0, 0, 2, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0});
  std::vector<float> internal_fp(
      internal_plan.find_array("fp")->element_count,
      0.0f);
  internal_workspace.copy_float_to_device("fp", internal_fp);

  internal_workspace_smoke_kernel<<<1, 32>>>(
      device.view(),
      internal_workspace.view(),
      device_output);
  if (cudaGetLastError() != cudaSuccess) {
    std::fprintf(stderr, "internal workspace smoke kernel launch failed\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }
  if (cudaMemcpy(
          &smoke_value,
          device_output,
          sizeof(float),
          cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::fprintf(stderr, "internal workspace smoke copyback failed\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }
  const float expected_internal_value =
      host.ann_type_major[0] + 1.0f + 10.0f + 1.0f + 1.0f + 1.0f + 2.0f;
  if (smoke_value != expected_internal_value) {
    std::fprintf(stderr, "internal workspace smoke value mismatch\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }

  const nep_adapters::cuda_backend::WorkspacePlan external_plan =
      nep_adapters::cuda_backend::make_external_neighbor_workspace_plan(
          host.protocol, 4, 2);
  nep_adapters::cuda_backend::DeviceWorkspace external_workspace(external_plan);
  if (external_workspace.summary().array_count != external_plan.arrays.size() ||
      external_workspace.summary().total_bytes != external_plan.total_bytes() ||
      external_workspace.view().boxes_row_major9 != nullptr ||
      external_workspace.view().active_atom_indices == nullptr ||
      external_workspace.view().neighbor_source != 1) {
    std::fprintf(stderr, "external workspace contract failed\n");
    return EXIT_FAILURE;
  }
  int lmp_ilist[] = {0, 2};
  int lmp_numneigh[] = {1, 0, 2, 0};
  int lmp_neigh0[] = {1};
  int lmp_neigh2[] = {3, 1};
  int* lmp_firstneigh[] = {lmp_neigh0, nullptr, lmp_neigh2, nullptr};
  int lmp_types[] = {1, 2, 1, 2};
  int lmp_type_map[] = {-1, 0, 1};
  double lmp_x0[] = {0.0, 10.0, 20.0};
  double lmp_x1[] = {1.0, 11.0, 21.0};
  double lmp_x2[] = {2.0, 12.0, 22.0};
  double lmp_x3[] = {3.0, 13.0, 23.0};
  double* lmp_positions[] = {lmp_x0, lmp_x1, lmp_x2, lmp_x3};
  NepaLammpsNeighborInput lmp_input{};
  lmp_input.nlocal = 4;
  lmp_input.inum = 2;
  lmp_input.ilist = lmp_ilist;
  lmp_input.numneigh = lmp_numneigh;
  lmp_input.firstneigh = lmp_firstneigh;
  lmp_input.types = lmp_types;
  lmp_input.type_map = lmp_type_map;
  lmp_input.positions = lmp_positions;
  nep_adapters::cuda_backend::stage_lammps_external_neighbors_on_device(
      lmp_input,
      host.protocol,
      external_workspace);
  std::vector<float> external_fp(
      external_plan.find_array("fp")->element_count,
      0.0f);
  external_workspace.copy_float_to_device("fp", external_fp);

  external_workspace_smoke_kernel<<<1, 32>>>(
      device.view(),
      external_workspace.view(),
      device_output);
  if (cudaGetLastError() != cudaSuccess) {
    std::fprintf(stderr, "external workspace smoke kernel launch failed\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }
  if (cudaMemcpy(
          &smoke_value,
          device_output,
          sizeof(float),
          cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::fprintf(stderr, "external workspace smoke copyback failed\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }
  const float expected_external_value = host.q_scaler[0] + 0.0f + 2.0f + 1.0f + 1.0f;
  if (smoke_value != expected_external_value) {
    std::fprintf(stderr, "external workspace smoke value mismatch\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }

  const std::string radial_model_path =
      (std::filesystem::temp_directory_path() / "cuda_radial_descriptor.nep").string();
  {
    std::ofstream out(radial_model_path);
    out << "nep4 1 C\n"
        << "cutoff 5 4 8 6\n"
        << "n_max 1 1\n"
        << "basis_size 2 2\n"
        << "l_max 4 1 1 1 1 1 1\n"
        << "ANN 1 0\n";
    for (int value = 1; value <= 59; ++value) {
      out << value << "\n";
    }
  }
  const nep_adapters::cuda_backend::HostModelParameters radial_host =
      nep_adapters::cuda_backend::load_host_model_parameters(radial_model_path);
  nep_adapters::cuda_backend::DeviceModel radial_device(radial_host);
  const nep_adapters::cuda_backend::ModelProtocol& radial_protocol =
      radial_host.protocol;
  const nep_adapters::cuda_backend::WorkspacePlan neighbor_plan =
      nep_adapters::cuda_backend::make_internal_neighbor_workspace_plan(
          radial_protocol, 8, 1);
  nep_adapters::cuda_backend::DeviceWorkspace neighbor_workspace(neighbor_plan);
  int neighbor_batch_count[] = {4};
  int neighbor_batch_offset[] = {0};
  int neighbor_types[] = {0, 0, 0, 0};
  double neighbor_positions[] = {
      0.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      6.0, 0.0, 0.0,
      11.0, 0.0, 0.0,
  };
  double neighbor_box[] = {
      12.0, 0.0, 0.0,
      0.0, 12.0, 0.0,
      0.0, 0.0, 12.0,
  };
  int neighbor_pbc[] = {1, 1, 1};
  NepaStructureBatch neighbor_batch{};
  neighbor_batch.num_structures = 1;
  neighbor_batch.total_atoms = 4;
  neighbor_batch.atom_counts = neighbor_batch_count;
  neighbor_batch.atom_offsets = neighbor_batch_offset;
  neighbor_batch.types = neighbor_types;
  neighbor_batch.positions_aos3 = neighbor_positions;
  neighbor_batch.boxes_row_major9 = neighbor_box;
  neighbor_batch.pbc_flags3 = neighbor_pbc;
  nep_adapters::cuda_backend::stage_batch_on_device(
      neighbor_batch,
      neighbor_workspace);
  const nep_adapters::cuda_backend::SimulationBox neighbor_simulation_box =
      make_simulation_box(neighbor_box, neighbor_pbc);
  nep_adapters::cuda_backend::build_internal_neighbors_on_device(
      radial_protocol,
      4,
      neighbor_simulation_box,
      neighbor_workspace);

  const nep_adapters::cuda_backend::DeviceWorkspaceView neighbor_view =
      neighbor_workspace.view();
  const std::vector<int> cell_dims = copy_ints(neighbor_view.cell_dims, 4);
  const std::vector<int> radial_counts = copy_ints(neighbor_view.nn_radial, 4);
  const std::vector<int> angular_counts = copy_ints(neighbor_view.nn_angular, 4);
  const std::vector<int> radial_neighbors = copy_ints(
      neighbor_view.nl_radial_slot_major,
      8 * static_cast<std::size_t>(radial_protocol.neighbor_capacity_radial));
  const std::vector<int> angular_neighbors = copy_ints(
      neighbor_view.nl_angular_slot_major,
      8 * static_cast<std::size_t>(radial_protocol.neighbor_capacity_angular));
  const std::vector<double> neighbor_positions_vec(
      neighbor_positions,
      neighbor_positions + 12);
  const std::vector<std::vector<int>> expected_radial =
      brute_force_neighbors(
          neighbor_positions_vec,
          radial_protocol.cutoff_radial,
          12.0,
          12.0,
          12.0,
          1,
          1,
          1);
  const std::vector<std::vector<int>> expected_angular =
      brute_force_neighbors(
          neighbor_positions_vec,
          radial_protocol.cutoff_angular,
          12.0,
          12.0,
          12.0,
          1,
          1,
          1);
  if (cell_dims.size() != 4 || cell_dims[0] != 2 || cell_dims[1] != 2 ||
      cell_dims[2] != 2 || cell_dims[3] != 8 ||
      !neighbor_sets_match(radial_counts, radial_neighbors, 8, expected_radial) ||
      !neighbor_sets_match(angular_counts, angular_neighbors, 8, expected_angular)) {
    std::fprintf(stderr, "orthorhombic neighbor contract failed\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }
  nep_adapters::cuda_backend::build_pair_geometry_cache_on_device(
      radial_protocol,
      4,
      neighbor_simulation_box,
      neighbor_workspace);
  const std::vector<float> r12_radial = copy_floats(
      neighbor_view.r12_radial,
      8 * static_cast<std::size_t>(radial_protocol.neighbor_capacity_radial));
  const std::vector<float> f12x = copy_floats(
      neighbor_view.f12x,
      8 * static_cast<std::size_t>(radial_protocol.neighbor_capacity_angular));
  const std::vector<float> f12y = copy_floats(
      neighbor_view.f12y,
      8 * static_cast<std::size_t>(radial_protocol.neighbor_capacity_angular));
  const std::vector<float> f12z = copy_floats(
      neighbor_view.f12z,
      8 * static_cast<std::size_t>(radial_protocol.neighbor_capacity_angular));
  if (!geometry_cache_matches(
          neighbor_positions_vec,
          radial_counts,
          radial_neighbors,
          r12_radial,
          angular_counts,
          angular_neighbors,
          f12x,
          f12y,
          f12z,
          8,
          12.0,
          12.0,
          12.0,
          1,
          1,
          1)) {
    std::fprintf(stderr, "orthorhombic geometry cache contract failed\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }

  const double triclinic_box[] = {
      12.0, 2.0, 1.0,
      0.0, 11.0, 1.5,
      0.0, 0.0, 10.0,
  };
  const double triclinic_positions[] = {
      0.2, 0.2, 0.2,
      11.4, 1.9, 1.0,
      2.4, 0.5, 0.3,
      7.0, 7.0, 6.0,
  };
  const nep_adapters::cuda_backend::SimulationBox triclinic_simulation_box =
      make_simulation_box(triclinic_box, neighbor_pbc);
  neighbor_batch.positions_aos3 = triclinic_positions;
  neighbor_batch.boxes_row_major9 = triclinic_box;
  nep_adapters::cuda_backend::stage_batch_on_device(
      neighbor_batch,
      neighbor_workspace);
  nep_adapters::cuda_backend::build_internal_neighbors_on_device(
      radial_protocol,
      4,
      triclinic_simulation_box,
      neighbor_workspace);
  const std::vector<int> triclinic_radial_counts =
      copy_ints(neighbor_view.nn_radial, 4);
  const std::vector<int> triclinic_angular_counts =
      copy_ints(neighbor_view.nn_angular, 4);
  const std::vector<int> triclinic_radial_neighbors = copy_ints(
      neighbor_view.nl_radial_slot_major,
      8 * static_cast<std::size_t>(radial_protocol.neighbor_capacity_radial));
  const std::vector<int> triclinic_angular_neighbors = copy_ints(
      neighbor_view.nl_angular_slot_major,
      8 * static_cast<std::size_t>(radial_protocol.neighbor_capacity_angular));
  const std::vector<double> triclinic_positions_vec(
      triclinic_positions,
      triclinic_positions + 12);
  const std::vector<std::vector<int>> expected_triclinic_radial =
      brute_force_neighbors(
          triclinic_positions_vec,
          radial_protocol.cutoff_radial,
          triclinic_simulation_box);
  const std::vector<std::vector<int>> expected_triclinic_angular =
      brute_force_neighbors(
          triclinic_positions_vec,
          radial_protocol.cutoff_angular,
          triclinic_simulation_box);
  if (!neighbor_sets_match(
          triclinic_radial_counts,
          triclinic_radial_neighbors,
          8,
          expected_triclinic_radial) ||
      !neighbor_sets_match(
          triclinic_angular_counts,
          triclinic_angular_neighbors,
          8,
          expected_triclinic_angular)) {
    std::fprintf(stderr, "triclinic neighbor contract failed\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }
  nep_adapters::cuda_backend::build_pair_geometry_cache_on_device(
      radial_protocol,
      4,
      triclinic_simulation_box,
      neighbor_workspace);
  const std::vector<float> triclinic_r12_radial = copy_floats(
      neighbor_view.r12_radial,
      8 * static_cast<std::size_t>(radial_protocol.neighbor_capacity_radial));
  const std::vector<float> triclinic_f12x = copy_floats(
      neighbor_view.f12x,
      8 * static_cast<std::size_t>(radial_protocol.neighbor_capacity_angular));
  const std::vector<float> triclinic_f12y = copy_floats(
      neighbor_view.f12y,
      8 * static_cast<std::size_t>(radial_protocol.neighbor_capacity_angular));
  const std::vector<float> triclinic_f12z = copy_floats(
      neighbor_view.f12z,
      8 * static_cast<std::size_t>(radial_protocol.neighbor_capacity_angular));
  if (!geometry_cache_matches(
          triclinic_positions_vec,
          triclinic_radial_counts,
          triclinic_radial_neighbors,
          triclinic_r12_radial,
          triclinic_angular_counts,
          triclinic_angular_neighbors,
          triclinic_f12x,
          triclinic_f12y,
          triclinic_f12z,
          8,
          triclinic_simulation_box)) {
    std::fprintf(stderr, "triclinic geometry cache contract failed\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }
  neighbor_batch.positions_aos3 = neighbor_positions;
  neighbor_batch.boxes_row_major9 = neighbor_box;
  nep_adapters::cuda_backend::stage_batch_on_device(
      neighbor_batch,
      neighbor_workspace);
  nep_adapters::cuda_backend::build_internal_neighbors_on_device(
      radial_protocol,
      4,
      neighbor_simulation_box,
      neighbor_workspace);
  nep_adapters::cuda_backend::build_pair_geometry_cache_on_device(
      radial_protocol,
      4,
      neighbor_simulation_box,
      neighbor_workspace);
  nep_adapters::cuda_backend::build_angular_basis_cache_on_device(
      radial_protocol,
      4,
      neighbor_workspace);
  nep_adapters::cuda_backend::build_radial_basis_cache_on_device(
      radial_protocol,
      4,
      neighbor_workspace);
  const std::vector<float> fc_radial = copy_floats(
      neighbor_view.fc_radial,
      8 * static_cast<std::size_t>(radial_protocol.neighbor_capacity_radial));
  const std::vector<float> fn_radial = copy_floats(
      neighbor_view.fn_radial,
      8 * static_cast<std::size_t>(radial_protocol.neighbor_capacity_radial) *
          (static_cast<std::size_t>(radial_protocol.basis_size_radial) + 1));
  if (!radial_basis_cache_matches(
          radial_counts,
          radial_neighbors,
          r12_radial,
          fc_radial,
          fn_radial,
          4,
          8,
          radial_protocol.neighbor_capacity_radial,
          radial_protocol.basis_size_radial,
          static_cast<float>(radial_protocol.cutoff_radial))) {
    std::fprintf(stderr, "radial basis cache contract failed\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }
  const std::vector<float> r12_angular = copy_floats(
      neighbor_view.r12_angular,
      8 * static_cast<std::size_t>(radial_protocol.neighbor_capacity_angular));
  const std::vector<float> fc_angular = copy_floats(
      neighbor_view.fc_angular,
      8 * static_cast<std::size_t>(radial_protocol.neighbor_capacity_angular));
  const std::vector<float> fn_angular = copy_floats(
      neighbor_view.fn_angular,
      8 * static_cast<std::size_t>(radial_protocol.neighbor_capacity_angular) *
          (static_cast<std::size_t>(radial_protocol.basis_size_angular) + 1));
  if (!angular_basis_cache_matches(
          angular_counts,
          f12x,
          f12y,
          f12z,
          r12_angular,
          fc_angular,
          fn_angular,
          4,
          8,
          radial_protocol.neighbor_capacity_angular,
          radial_protocol.basis_size_angular,
          static_cast<float>(radial_protocol.cutoff_angular))) {
    std::fprintf(stderr, "angular basis cache contract failed\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }
  nep_adapters::cuda_backend::build_radial_descriptors_on_device(
      radial_protocol,
      4,
      radial_device,
      neighbor_workspace);
  const std::vector<int> radial_types = copy_ints(neighbor_view.types, 4);
  const std::vector<float> descriptors = copy_floats(
      neighbor_view.descriptors,
      8 * static_cast<std::size_t>(radial_protocol.descriptor_dim));
  if (!radial_descriptors_match(
          radial_host,
          radial_types,
          radial_counts,
          radial_neighbors,
          fn_radial,
          descriptors,
          4,
          8)) {
    std::fprintf(stderr, "radial descriptor contract failed\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }
  nep_adapters::cuda_backend::build_angular_descriptors_on_device(
      radial_protocol,
      4,
      radial_device,
      neighbor_workspace);
  const std::vector<float> descriptors_with_angular = copy_floats(
      neighbor_view.descriptors,
      8 * static_cast<std::size_t>(radial_protocol.descriptor_dim));
  const std::vector<float> sum_fxyz = copy_floats(
      neighbor_view.sum_fxyz,
      8 * (static_cast<std::size_t>(radial_protocol.n_max_angular) + 1) *
          static_cast<std::size_t>(radial_protocol.body_channels.abc_count()));
  if (!angular_descriptors_match(
          radial_host,
          radial_types,
          angular_counts,
          angular_neighbors,
          f12x,
          f12y,
          f12z,
          r12_angular,
          fn_angular,
          sum_fxyz,
          descriptors_with_angular,
          4,
          8)) {
    std::fprintf(stderr, "angular descriptor contract failed\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }
  nep_adapters::cuda_backend::evaluate_ann_energy_on_device(
      radial_protocol,
      4,
      radial_device,
      neighbor_workspace);
  const std::vector<double> potential = copy_doubles(neighbor_view.potential, 4);
  const std::vector<float> fp = copy_floats(
      neighbor_view.fp,
      8 * static_cast<std::size_t>(radial_protocol.descriptor_dim));
  if (!ann_energy_matches(
          radial_host,
          radial_types,
          descriptors_with_angular,
          potential,
          fp,
          4,
          8)) {
    std::fprintf(stderr, "ANN energy contract failed\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }
  cudaFree(device_output);

  nep_adapters::cuda_backend::DeviceModel moved(std::move(device));
  if (moved.ann_type_major_device() == nullptr ||
      moved.descriptor_coefficients_device() == nullptr ||
      moved.q_scaler_device() == nullptr) {
    std::fprintf(stderr, "moved device model lost pointers\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
