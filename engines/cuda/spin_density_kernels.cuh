/* Adapted from GPUMD c516 cd6ed6b9. SPDX-License-Identifier: LGPL-3.0-or-later */
#pragma once
#include "../common/spin_density_layout.hpp"
#include "simulation_box_device.cuh"

namespace nep_adapters::spin_density {
__device__ __forceinline__ void find_fc_and_fcp(
    float rc,
    float rcinv,
    float r,
    float& fc,
    float& fcp) {
  if (r < rc) {
    const float x = r * rcinv;
    fc = 0.5f * cosf(3.14159265358979323846f * x) + 0.5f;
    fcp = -1.5707963f * sinf(3.14159265358979323846f * x) * rcinv;
  } else {
    fc = 0.0f;
    fcp = 0.0f;
  }
}

__device__ __forceinline__ void find_fn_and_fnp(
    int n,
    float rcinv,
    float r,
    float fc,
    float fcp,
    float& fn,
    float& fnp) {
  if (n == 0) {
    fn = fc;
    fnp = fcp;
    return;
  }

  const float r_scaled = r * rcinv;
  const float x = 2.0f * (r_scaled - 1.0f) * (r_scaled - 1.0f) - 1.0f;
  if (n == 1) {
    fn = (x + 1.0f) * 0.5f;
    fnp = 2.0f * (r_scaled - 1.0f) * rcinv * fc + fn * fcp;
    fn *= fc;
    return;
  }

  float t0 = 1.0f;
  float t1 = x;
  float t2 = x;
  float u0 = 1.0f;
  float u1 = 2.0f * x;
  for (int m = 2; m <= n; ++m) {
    t2 = 2.0f * x * t1 - t0;
    t0 = t1;
    t1 = t2;
    const float u2 = 2.0f * x * u1 - u0;
    u0 = u1;
    u1 = u2;
  }
  fn = (t2 + 1.0f) * 0.5f;
  fnp = n * u0 * 2.0f * (r_scaled - 1.0f) * rcinv;
  fnp = fnp * fc + fn * fcp;
  fn *= fc;
}

__device__ inline void radial_basis(int basis, float inverse, float distance, float fc, float fcp, float* fn, float* fnp) {
  for (int k = 0; k <= basis; ++k) find_fn_and_fnp(k, inverse, distance, fc, fcp, fn[k], fnp[k]);
}

// The generic and compiled layouts use the same arithmetic and output ordering.
template<int Capacity, bool Contracted = false>
struct RuntimeLayout : View {
  explicit RuntimeLayout(View v) : View(v) {}
  enum : int { component_capacity = Capacity, spin_capacity = 35, harmonic_capacity = 25, contracted = Contracted, tile = 1, edge_cached = 0 };
  const float* edge_radial = nullptr;
  __device__ inline Primitive primitive(int i) const { return primitives[i]; }
  __device__ inline Sector sector(int i) const { return sectors[i]; }
  __device__ inline Term term(int i) const { return terms[i]; }
  __device__ inline Center center(int i) const { return centers[i]; }
  __device__ inline Harmonic harmonic(int i) const { return harmonics[i]; }
};
using DynamicLayout = RuntimeLayout<100>;
using WideLayout = RuntimeLayout<625>; // Maximum (L+1)^2 * (min(L,S)+1)^2.
// Values and Cartesian derivatives are evaluated together, including at s=0.
struct Value { float value, derivative[3]; };
__device__ inline float power(float x, int n) {
  float result = 1;
  for (int i = 0; i < n; ++i) result *= x;
  return result;
}
template<class Layout>
__device__ __forceinline__ void spherical(Layout v, const float* x, Value* y) {
  #pragma unroll
  for (int i = 0; i < Layout::harmonic_capacity; ++i) y[i] = {0, {0, 0, 0}};
  #pragma unroll
  for (int i = 0; i < v.harmonic_count; ++i) {
    const Harmonic h = v.harmonic(i);
    if (h.component >= (v.lmax + 1) * (v.lmax + 1)) continue;
    const int p[3] = {h.x, h.y, h.z};
    float val = h.value;
    for (int d = 0; d < 3; ++d) val *= power(x[d], p[d]);
    y[h.component].value += val;
    for (int d = 0; d < 3; ++d) {
      if (!p[d]) continue;
      float grad = h.value * p[d];
      for (int k = 0; k < 3; ++k) grad *= power(x[k], p[k] - (d == k));
      y[h.component].derivative[d] += grad;
    }
  }
}
template<class Layout>
__device__ __forceinline__ void primitives_at(Layout v, const float* s, Value* phi) {
  Value y[Layout::harmonic_capacity];
  spherical(v, s, y);
  const float s2 = s[0]*s[0] + s[1]*s[1] + s[2]*s[2];
  #pragma unroll
  for (int p = 0; p < v.primitive_count; ++p) {
    const Primitive label = v.primitive(p);
    const int exponent = (label.degree - label.l) / 2;
    const float factor = power(s2, exponent);
    const float grad_factor = exponent ? 2 * exponent * power(s2, exponent-1) : 0;
    for (int m = 0; m < 2 * label.l + 1; ++m) {
      const Value harmonic = y[label.l*label.l+m];
      Value& value = phi[label.offset+m];
      value.value = factor * harmonic.value;
      for (int d = 0; d < 3; ++d)
        value.derivative[d] = factor * harmonic.derivative[d] + grad_factor*s[d]*harmonic.value;
    }
  }
}
__device__ inline void read_spin(int n, int i, const int* type, const int* active,
                                 const double* spins, float* s) {
  for (int d = 0; d < 3; ++d) s[d] = active[type[i]] ? spins[d*n+i] : 0;
}
struct Geometry {
  cuda_backend::SimulationBox box;
  int neighbor_source;
  const int* atom_to_structure;
  const double* boxes;
  const double* inverse;
  const int* pbc;
};
__device__ inline void displacement(int n, int i, int j, int, Geometry g,
                                    const double* positions, const double*, int, float* r) {
  double d[3];
  for (int k = 0; k < 3; ++k) d[k] = positions[k*n+j] - positions[k*n+i];
  if (g.neighbor_source != 0) {
    for (int k = 0; k < 3; ++k) r[k] = static_cast<float>(d[k]);
    return;
  }
  auto box = g.box;
  if (g.atom_to_structure && g.boxes)
    box = cuda_backend::load_structure_box(g.atom_to_structure[i], g.boxes, g.inverse, g.pbc);
  cuda_backend::minimum_image_delta(box, d[0], d[1], d[2], r[0], r[1], r[2]);
}
template<class Layout>
__device__ inline void radial(Layout v, int slot, int plane, int c, int channels, int na, int basis, int pair, float cutoff,
                              float distance, const float* angular, const float* mix,
                              float& weight, float& derivative) {
  if (Layout::edge_cached) {
    weight = v.edge_radial[(2*c)*plane+slot];
    derivative = v.edge_radial[(2*c+1)*plane+slot];
    return;
  }
  const float inverse = 1/cutoff;
  float fc, fcp;
  find_fc_and_fcp(cutoff, inverse, distance, fc, fcp);
  weight = derivative = 0;
  if constexpr (Layout::contracted) {
    // Consume the basis immediately. Restarting T/U for each order performs
    // quadratic work and forces the dynamically indexed basis arrays to spill.
    const float* coefficients = angular + (pair*channels+c)*(basis+1);
    weight += fc * coefficients[0];
    derivative += fcp * coefficients[0];
    if (basis == 0) return;
    const float offset = distance * inverse - 1.0f;
    const float x = 2.0f * offset * offset - 1.0f;
    float fn = (x + 1.0f) * 0.5f;
    float fnp = 2.0f * offset * inverse * fc + fn * fcp;
    fn *= fc;
    weight += fn * coefficients[1];
    derivative += fnp * coefficients[1];
    float t0 = 1.0f, t1 = x;
    float u0 = 1.0f, u1 = 2.0f * x;
    for (int k = 2; k <= basis; ++k) {
      const float t2 = 2.0f * x * t1 - t0;
      fn = (t2 + 1.0f) * 0.5f;
      fnp = k * u1 * 2.0f * offset * inverse;
      fnp = fnp * fc + fn * fcp;
      fn *= fc;
      weight += fn * coefficients[k];
      derivative += fnp * coefficients[k];
      t0 = t1;
      t1 = t2;
      const float u2 = 2.0f * x * u1 - u0;
      u0 = u1;
      u1 = u2;
    }
  } else {
    float fn[17], fnp[17];
    radial_basis(basis, inverse, distance, fc, fcp, fn, fnp);
    for (int a = 0; a <= na; ++a) {
      float value = 0, grad = 0;
      for (int k = 0; k <= basis; ++k) {
        const float coefficient = angular[(pair*(na+1)+a)*(basis+1)+k];
        value += fn[k]*coefficient;
        grad += fnp[k]*coefficient;
      }
      weight += value * mix[a*channels+c];
      derivative += grad * mix[a*channels+c];
    }
  }
}

template<class Layout, bool FactorCenter = false>
static __global__ void descriptors(
  Layout v, int atom_count, int n, int structural_dim, int types, int na, int basis, Geometry box,
  const int* type, const int* active, const double* positions, const double* spins,
  const int* nn, const int* nl, const double* edges, int plane, const float* cutoffs,
  const float* angular, const float* parameters, float* moments, float* q) {
  const int work = (blockIdx.x*blockDim.x+threadIdx.x) / Layout::tile;
  const int lane = threadIdx.x % Layout::tile;
  const int i = work / v.channels, c = work % v.channels;
  // Capture participation before divergent neighbor loops. An activemask taken
  // at a reduction can omit lanes still processing their last neighbor.
  const unsigned mask = Layout::tile > 1 ? __ballot_sync(0xffffffff, i < atom_count) : 0;
  if (i >= atom_count) return;
  float s[3], h[Layout::component_capacity] = {};
  // Center-spin terms factor through just the geometric moments. Their path
  // weights and spin primitives are independent of the neighbor.
  constexpr int geometric_components = FactorCenter ? Layout::harmonic_capacity : 1;
  float geometric[geometric_components] = {};
  Value phi_i[Layout::spin_capacity];
  read_spin(n, i, type, active, spins, s);
  primitives_at(v, s, phi_i);
  const float* weights = parameters + (na+1)*v.channels;
  const float* center = weights + v.path_count*v.channels;
  for (int neighbor = lane; neighbor < nn[i]; neighbor += Layout::tile) {
    int slot = neighbor*n+i, j = nl[slot], pair = type[i]*types+type[j];
    float r[3], u[3];
    displacement(n, i, j, slot, box, positions, edges, plane, r);
    float distance = sqrtf(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
    if (!(distance > 1.0e-12f && distance < cutoffs[pair])) continue;
    for (int d = 0; d < 3; ++d) u[d] = r[d]/distance;
    Value y[Layout::harmonic_capacity], phi_j[Layout::spin_capacity];
    spherical(v, u, y);
    read_spin(n, j, type, active, spins, s);
    primitives_at(v, s, phi_j);
    float weight, derivative;
    radial(v, slot, plane, c, v.channels, na, basis, pair, cutoffs[pair], distance, angular, parameters, weight, derivative);
    if constexpr (FactorCenter) {
      #pragma unroll
      for (int g = 0; g < Layout::harmonic_capacity; ++g)
        geometric[g] += weight * y[g].value;
    }
    #pragma unroll
    for (int t = 0; t < v.term_count; ++t) {
      const Term term = v.term(t);
      if constexpr (FactorCenter)
        if (!term.neighbor) continue;
      float spin = (term.neighbor ? phi_j : phi_i)[term.spin].value;
      h[term.output] += weight*weights[term.path*v.channels+c]*term.coefficient*y[term.geometry].value*spin;
    }
  }
  #pragma unroll
  for (int m = 0; m < v.components; ++m) {
    #pragma unroll
    for (int offset = Layout::tile/2; offset > 0; offset /= 2)
      h[m] += __shfl_down_sync(mask, h[m], offset, Layout::tile);
  }
  if constexpr (FactorCenter) {
    #pragma unroll
    for (int g = 0; g < Layout::harmonic_capacity; ++g) {
      #pragma unroll
      for (int offset = Layout::tile/2; offset > 0; offset /= 2)
        geometric[g] += __shfl_down_sync(mask, geometric[g], offset, Layout::tile);
    }
  }
  if (lane != 0) return;
  if constexpr (FactorCenter) {
    #pragma unroll
    for (int t = 0; t < v.term_count; ++t) {
      const Term term = v.term(t);
      if (term.neighbor) continue;
      h[term.output] += geometric[term.geometry] * weights[term.path*v.channels+c] *
        term.coefficient * phi_i[term.spin].value;
    }
  }
  #pragma unroll
  for (int t = 0; t < v.center_count; ++t) {
    const Center term = v.center(t);
    h[term.output] += center[term.weight*v.channels+c]*phi_i[term.primitive].value;
  }
  #pragma unroll
  for (int m = 0; m < v.components; ++m) moments[(c*v.components+m)*n+i] = h[m];
  if (c == 0) {
    int column = structural_dim;
    #pragma unroll
    for (int p = 1; p < v.primitive_count; ++p)
      if (v.primitive(p).l == 0) q[(column++)*n+i] = phi_i[v.primitive(p).offset].value;
  }
  #pragma unroll
  for (int k = 0; k < v.sector_count; ++k) {
    Sector sector = v.sector(k);
    float norm = 0;
    for (int m = 0; m < sector.size; ++m) norm += h[sector.offset+m]*h[sector.offset+m];
    q[(structural_dim+sector.norm_column+c)*n+i] = norm/(sector.size);
    if (sector.scalar_column >= 0)
      q[(structural_dim+sector.scalar_column+c)*n+i] = h[sector.offset];
  }
}

template<class Layout>
static __global__ void forces(
  Layout v, int atom_count, int n, int structural_dim, int types, int na, int basis, Geometry box,
  const int* type, const int* active, const double* positions, const double* spins,
  const int* nn, const int* nl, const double* edges, int plane, const float* cutoffs,
  const float* angular, const float* parameters, const float* moments, const float* fp,
  double* force, double* mforce, double* virial, int virial_mode,
  float* virial_float, float* transfer) {
  const int work = (blockIdx.x*blockDim.x+threadIdx.x) / Layout::tile;
  const int lane = threadIdx.x % Layout::tile;
  const int i = work / v.channels, c = work % v.channels;
  // Capture participation before divergent neighbor loops. An activemask taken
  // at a reduction can omit lanes still processing their last neighbor.
  const unsigned mask = Layout::tile > 1 ? __ballot_sync(0xffffffff, i < atom_count) : 0;
  if (i >= atom_count) return;
  float s[3], pull[Layout::component_capacity] = {}, center_pull[3] = {};
  double center_force[3] = {};
  Value phi_i[Layout::spin_capacity];
  read_spin(n, i, type, active, spins, s);
  primitives_at(v, s, phi_i);
  #pragma unroll
  for (int k = 0; k < v.sector_count; ++k) {
    Sector sector = v.sector(k);
    float derivative = 2*fp[(structural_dim+sector.norm_column+c)*n+i]/(sector.size);
    for (int m = 0; m < sector.size; ++m)
      pull[sector.offset+m] = derivative*moments[(c*v.components+sector.offset+m)*n+i];
    if (sector.scalar_column >= 0) pull[sector.offset] += fp[(structural_dim+sector.scalar_column+c)*n+i];
  }
  const float* weights = parameters + (na+1)*v.channels;
  const float* center = weights + v.path_count*v.channels;
  if (lane == 0) {
    #pragma unroll
    for (int t = 0; t < v.center_count; ++t) {
      Center term = v.center(t);
      for (int d = 0; d < 3; ++d)
        center_pull[d] += pull[term.output]*center[term.weight*v.channels+c]*phi_i[term.primitive].derivative[d];
    }
    if (c == 0) {
      int column = structural_dim;
      #pragma unroll
      for (int p = 1; p < v.primitive_count; ++p)
        if (v.primitive(p).l == 0) {
          for (int d = 0; d < 3; ++d)
            center_pull[d] += fp[column*n+i]*phi_i[v.primitive(p).offset].derivative[d];
          ++column;
        }
    }
  }
  // The compiled MD tile shares a center's adjoint in harmonic/spin space.
  // Center-spin values and derivatives do not depend on the neighbor. Fold
  // them once; combine repeated neighbor-spin coefficients before edge work.
  constexpr int harmonics = Layout::harmonic_capacity;
  constexpr int spin_components = Layout::spin_capacity;
  constexpr int groups_per_block = 128 / Layout::tile;
  constexpr int coefficients = 4 * harmonics + harmonics * (spin_components - 1);
  constexpr int cache_size = Layout::tile > 1 ? coefficients * groups_per_block : 1;
  __shared__ volatile float edge_pull[cache_size];
  const int group = threadIdx.x / Layout::tile;
  if constexpr (Layout::tile > 1) {
    for (int k = lane; k < coefficients; k += Layout::tile)
      edge_pull[k * groups_per_block + group] = 0.0f;
    __syncwarp(mask);
    if (lane == 0) {
      #pragma unroll
      for (int t = 0; t < v.term_count; ++t) {
        const Term term = v.term(t);
        const float adjoint = pull[term.output] *
          weights[term.path * v.channels + c] * term.coefficient;
        if (term.neighbor) {
          // The degree-zero primitive has only a center path.
          const int k = 4 * harmonics +
            term.geometry * (spin_components - 1) + term.spin - 1;
          edge_pull[k * groups_per_block + group] += adjoint;
        } else {
          edge_pull[term.geometry * groups_per_block + group] +=
            adjoint * phi_i[term.spin].value;
          #pragma unroll
          for (int d = 0; d < 3; ++d)
            edge_pull[((d + 1) * harmonics + term.geometry) *
                      groups_per_block + group] +=
              adjoint * phi_i[term.spin].derivative[d];
        }
      }
    }
    __syncwarp(mask);
  }
  for (int neighbor = lane; neighbor < nn[i]; neighbor += Layout::tile) {
    int slot = neighbor*n+i, j = nl[slot], pair = type[i]*types+type[j];
    float r[3], u[3];
    displacement(n, i, j, slot, box, positions, edges, plane, r);
    float distance = sqrtf(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
    if (!(distance > 1.0e-12f && distance < cutoffs[pair])) continue;
    for (int d = 0; d < 3; ++d) u[d] = r[d]/distance;
    Value y[Layout::harmonic_capacity], phi_j[Layout::spin_capacity];
    spherical(v, u, y);
    read_spin(n, j, type, active, spins, s);
    primitives_at(v, s, phi_j);
    float weight, derivative;
    radial(v, slot, plane, c, v.channels, na, basis, pair, cutoffs[pair], distance, angular, parameters, weight, derivative);
    float radial_pull = 0, angular_pull[3] = {}, neighbor_pull[3] = {};
    if constexpr (Layout::tile > 1) {
      #pragma unroll
      for (int g = 0; g < harmonics; ++g) {
        float value = edge_pull[g * groups_per_block + group];
        float spin_gradient[3] = {};
        #pragma unroll
        for (int s = 1; s < spin_components; ++s) {
          const int k = 4 * harmonics + g * (spin_components - 1) + s - 1;
          const float adjoint = edge_pull[k * groups_per_block + group];
          value += adjoint * phi_j[s].value;
          #pragma unroll
          for (int d = 0; d < 3; ++d)
            spin_gradient[d] += adjoint * phi_j[s].derivative[d];
        }
        radial_pull += value * y[g].value;
        #pragma unroll
        for (int d = 0; d < 3; ++d) {
          angular_pull[d] += weight * value * y[g].derivative[d];
          neighbor_pull[d] += weight * y[g].value * spin_gradient[d];
          center_pull[d] += weight * y[g].value *
            edge_pull[((d + 1) * harmonics + g) * groups_per_block + group];
        }
      }
    } else {
      #pragma unroll
      for (int t = 0; t < v.term_count; ++t) {
        Term term = v.term(t);
        const Value spin = (term.neighbor ? phi_j : phi_i)[term.spin];
        const Value harmonic = y[term.geometry];
        float adjoint = pull[term.output]*weights[term.path*v.channels+c]*term.coefficient;
        radial_pull += adjoint*harmonic.value*spin.value;
        for (int d = 0; d < 3; ++d) {
          angular_pull[d] += adjoint*weight*harmonic.derivative[d]*spin.value;
          (term.neighbor ? neighbor_pull : center_pull)[d] += adjoint*weight*harmonic.value*spin.derivative[d];
        }
      }
    }
    float longitudinal = 0;
    for (int d = 0; d < 3; ++d) longitudinal += angular_pull[d]*u[d];
    float grad[3];
    for (int d = 0; d < 3; ++d) {
      grad[d] = derivative*radial_pull*u[d] + (angular_pull[d]-longitudinal*u[d])/distance;
      if (Layout::tile > 1) center_force[d] += static_cast<double>(grad[d]);
      else atomicAdd(force+d*n+i, static_cast<double>(grad[d]));
      atomicAdd(force+d*n+j, -static_cast<double>(grad[d]));
      if (active[type[j]]) atomicAdd(mforce+d*n+j, -static_cast<double>(neighbor_pull[d]));
    }
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        const int raw_to_internal[9] = {0,3,4,6,1,5,7,8,2};
        const int k = raw_to_internal[3*a+b];
        const double value = -static_cast<double>(r[a]) * grad[b];
        if (virial_mode == 1 || virial_mode == 3) atomicAdd(virial+k*n+i, value);
        if (virial_mode == 2) atomicAdd(virial+k*n+j, value);
        if (virial_mode == 3) atomicAdd(virial_float+k*n+j, static_cast<float>(value));
        if (transfer && active[type[j]])
          atomicAdd(transfer+(3*a+b)*n+j, -r[a]*neighbor_pull[b]);
      }
  }
  #pragma unroll
  for (int d = 0; d < 3; ++d) {
    #pragma unroll
    for (int offset = Layout::tile/2; offset > 0; offset /= 2) {
      center_pull[d] += __shfl_down_sync(mask, center_pull[d], offset, Layout::tile);
      center_force[d] += __shfl_down_sync(mask, center_force[d], offset, Layout::tile);
    }
  }
  if (lane != 0) return;
  if (Layout::tile > 1)
    for (int d = 0; d < 3; ++d) atomicAdd(force+d*n+i, center_force[d]);
  if (active[type[i]])
    for (int d = 0; d < 3; ++d) atomicAdd(mforce+d*n+i, -static_cast<double>(center_pull[d]));
}
} // namespace nep_adapters::spin_density
