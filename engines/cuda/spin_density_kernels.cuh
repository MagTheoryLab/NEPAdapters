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
template<int Capacity, bool Contracted = false, int Spins = 35, int Harmonics = 25>
struct RuntimeLayout : View {
  explicit RuntimeLayout(View v) : View(v) {}
  enum : int { component_capacity = Capacity, spin_capacity = Spins, harmonic_capacity = Harmonics, contracted = Contracted, tile = 1, edge_cached = 0 };
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
  float density[Layout::tile == 1 ? Layout::harmonic_capacity : 1]
               [Layout::tile == 1 ? Layout::spin_capacity : 1] = {};
  constexpr int geometric_components = FactorCenter ? Layout::harmonic_capacity : 1;
  float geometric[geometric_components] = {};
  const int geometry_count = (v.lmax + 1) * (v.lmax + 1);
  int spin_components = Layout::spin_capacity;
  if constexpr (Layout::tile == 1) spin_components = v.spin_components;
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
    // Keep the compact C3L2 accumulation instead of another live density tensor.
    if constexpr (Layout::tile > 1) {
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
    } else {
      #pragma unroll
      for (int g = 0; g < geometry_count; ++g) {
        const float weighted_y = weight * y[g].value;
        density[g][0] += weighted_y;
        #pragma unroll
        for (int a = 1; a < spin_components; ++a)
          density[g][a] += weighted_y * phi_j[a].value;
      }
    }
  }
  if constexpr (Layout::tile == 1) {
    // Couple once per center/channel after accumulating the neighbor density.
    #pragma unroll
    for (int t = 0; t < v.term_count; ++t) {
      const Term term = v.term(t);
      const float product = term.neighbor ? density[term.geometry][term.spin] :
        density[term.geometry][0] * phi_i[term.spin].value;
      h[term.output] += weights[term.path*v.channels+c] * term.coefficient * product;
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
    // Keep the reduced geometric moments after the descriptor moments. The
    // reverse pass can contract center-spin derivatives once per center.
    #pragma unroll
    for (int g = 0; g < Layout::harmonic_capacity; ++g)
      moments[(v.channels*v.components+c*Layout::harmonic_capacity+g)*n+i] = geometric[g];
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

template<class Layout, bool CenterVirial = false, bool FactorCenter = false>
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
  double center_virial[CenterVirial ? 9 : 1] = {};
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
  constexpr int center_planes = FactorCenter ? 1 : 4;
  constexpr int coefficients = center_planes * harmonics + harmonics * (spin_components - 1);
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
          const int k = center_planes * harmonics +
            term.geometry * (spin_components - 1) + term.spin - 1;
          edge_pull[k * groups_per_block + group] += adjoint;
        } else {
          edge_pull[term.geometry * groups_per_block + group] +=
            adjoint * phi_i[term.spin].value;
          #pragma unroll
          for (int d = 0; d < 3; ++d) {
            if constexpr (FactorCenter) {
              const float geometric = moments[
                (v.channels*v.components+c*harmonics+term.geometry)*n+i];
              center_pull[d] += adjoint * phi_i[term.spin].derivative[d] * geometric;
            } else {
              edge_pull[((d + 1) * harmonics + term.geometry) *
                        groups_per_block + group] +=
                adjoint * phi_i[term.spin].derivative[d];
            }
          }
        }
      }
    }
    __syncwarp(mask);
    // The compiled spin primitives span the Cartesian polynomials of degree
    // one and two. Contract their adjoints once, so each edge evaluates the
    // polynomial and its gradient directly instead of rebuilding primitives.
    static_assert(spin_components == 10 && harmonics == 9);
    for (int g = lane; g < harmonics; g += Layout::tile) {
      const int base = center_planes * harmonics + g * (spin_components - 1);
      float a[9];
      #pragma unroll
      for (int k = 0; k < 9; ++k)
        a[k] = edge_pull[(base+k) * groups_per_block + group];
      const float linear = v.harmonic(3).value;
      const float diagonal = v.harmonic(4).value;
      const float difference = v.harmonic(5).value;
      const float cross = v.harmonic(6).value;
      // a indexes phi[1..9]: x,y,z,s2,xz,xy,Qyy,yz,Qzz-xx.
      // The rank-two primitive starts at phi[5], harmonic component four.
      // Store coefficients of x,y,z,xx,yy,zz,xy,xz,yz respectively.
      const float cartesian[9] = {
        linear*a[0], linear*a[1], linear*a[2],
        a[3]+diagonal*a[6]-difference*a[8],
        a[3]-2*diagonal*a[6],
        a[3]+diagonal*a[6]+difference*a[8],
        cross*a[5], cross*a[4], cross*a[7]};
      #pragma unroll
      for (int k = 0; k < 9; ++k)
        edge_pull[(base+k) * groups_per_block + group] = cartesian[k];
    }
    __syncwarp(mask);
  }
  float density_pull[Layout::tile == 1 ? Layout::harmonic_capacity : 1]
                    [Layout::tile == 1 ? Layout::spin_capacity : 1] = {};
  float center_gradient[Layout::tile == 1 ? Layout::harmonic_capacity : 1][3] = {};
  float geometry_sum[Layout::tile == 1 ? Layout::harmonic_capacity : 1] = {};
  const int geometry_count = (v.lmax + 1) * (v.lmax + 1);
  if constexpr (Layout::tile == 1) {
    for (int t = 0; t < v.term_count; ++t) {
      const Term term = v.term(t);
      const float adjoint = pull[term.output] *
        weights[term.path*v.channels+c] * term.coefficient;
      if (term.neighbor) density_pull[term.geometry][term.spin] += adjoint;
      else {
        density_pull[term.geometry][0] += adjoint * phi_i[term.spin].value;
        for (int d = 0; d < 3; ++d)
          center_gradient[term.geometry][d] += adjoint * phi_i[term.spin].derivative[d];
      }
    }
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
    if constexpr (Layout::tile == 1) primitives_at(v, s, phi_j);
    float weight, derivative;
    radial(v, slot, plane, c, v.channels, na, basis, pair, cutoffs[pair], distance, angular, parameters, weight, derivative);
    float radial_pull = 0, angular_pull[3] = {}, neighbor_pull[3] = {};
    if constexpr (Layout::tile > 1) {
      #pragma unroll
      for (int g = 0; g < harmonics; ++g) {
        float value = edge_pull[g * groups_per_block + group];
        float spin_gradient[3] = {};
        const int base = center_planes * harmonics + g * (spin_components - 1);
        float a[9];
        #pragma unroll
        for (int k = 0; k < 9; ++k)
          a[k] = edge_pull[(base+k) * groups_per_block + group];
        value += s[0]*(a[0]+a[3]*s[0]+a[6]*s[1]+a[7]*s[2]) +
                 s[1]*(a[1]+a[4]*s[1]+a[8]*s[2]) +
                 s[2]*(a[2]+a[5]*s[2]);
        spin_gradient[0] = a[0]+2*a[3]*s[0]+a[6]*s[1]+a[7]*s[2];
        spin_gradient[1] = a[1]+2*a[4]*s[1]+a[6]*s[0]+a[8]*s[2];
        spin_gradient[2] = a[2]+2*a[5]*s[2]+a[7]*s[0]+a[8]*s[1];
        radial_pull += value * y[g].value;
        #pragma unroll
        for (int d = 0; d < 3; ++d) {
          angular_pull[d] += weight * value * y[g].derivative[d];
          neighbor_pull[d] += weight * y[g].value * spin_gradient[d];
          if constexpr (!FactorCenter)
            center_pull[d] += weight * y[g].value *
              edge_pull[((d + 1) * harmonics + g) * groups_per_block + group];
        }
      }
    } else {
      float spin_pull[Layout::spin_capacity] = {};
      for (int g = 0; g < geometry_count; ++g) {
        float y_pull = density_pull[g][0];
        const float weighted_y = weight * y[g].value;
        geometry_sum[g] += weighted_y;
        for (int a = 1; a < v.spin_components; ++a) {
          y_pull += density_pull[g][a] * phi_j[a].value;
          spin_pull[a] += weighted_y * density_pull[g][a];
        }
        radial_pull += y_pull * y[g].value;
        for (int d = 0; d < 3; ++d)
          angular_pull[d] += weight * y_pull * y[g].derivative[d];
      }
      for (int a = 1; a < v.spin_components; ++a)
        for (int d = 0; d < 3; ++d)
          neighbor_pull[d] += spin_pull[a] * phi_j[a].derivative[d];
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
    #pragma unroll
    for (int a = 0; a < 3; ++a)
      #pragma unroll
      for (int b = 0; b < 3; ++b) {
        const int raw_to_internal[9] = {0,3,4,6,1,5,7,8,2};
        const int k = raw_to_internal[3*a+b];
        const double value = -static_cast<double>(r[a]) * grad[b];
        if constexpr (CenterVirial) center_virial[k] += value;
        if (virial_mode == 2) atomicAdd(virial+k*n+j, value);
        if (virial_mode == 3) atomicAdd(virial_float+k*n+j, static_cast<float>(value));
        if (transfer && active[type[j]])
          atomicAdd(transfer+(3*a+b)*n+j, -r[a]*neighbor_pull[b]);
      }
  }
  if constexpr (Layout::tile == 1) {
    for (int g = 0; g < geometry_count; ++g)
      for (int d = 0; d < 3; ++d)
        center_pull[d] += geometry_sum[g] * center_gradient[g][d];
  }
  #pragma unroll
  for (int d = 0; d < 3; ++d) {
    #pragma unroll
    for (int offset = Layout::tile/2; offset > 0; offset /= 2) {
      center_pull[d] += __shfl_down_sync(mask, center_pull[d], offset, Layout::tile);
      center_force[d] += __shfl_down_sync(mask, center_force[d], offset, Layout::tile);
    }
  }
  // Keep the center tensor in FP64, including its antisymmetric components.
  // Only center-output instantiations carry these registers through the loop.
  if constexpr (CenterVirial) {
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
      #pragma unroll
      for (int offset = Layout::tile/2; offset > 0; offset /= 2)
        center_virial[k] += __shfl_down_sync(mask, center_virial[k], offset, Layout::tile);
      if (lane == 0) atomicAdd(virial+k*n+i, center_virial[k]);
    }
  }
  if (lane != 0) return;
  if (Layout::tile > 1)
    for (int d = 0; d < 3; ++d) atomicAdd(force+d*n+i, center_force[d]);
  if (active[type[i]])
    for (int d = 0; d < 3; ++d) atomicAdd(mforce+d*n+i, -static_cast<double>(center_pull[d]));
}

// Stable per-component monomial lists allow all lanes to evaluate a harmonic.
template<class Layout>
__device__ inline Value component_harmonic(Layout v, const float* x, int component,
                                         const int* offsets, const int* order) {
  Value value = {};
  for (int k = offsets[component]; k < offsets[component+1]; ++k) {
    const Harmonic h = v.harmonic(order[k]);
    const int p[3] = {h.x, h.y, h.z};
    float val = h.value;
    for (int d = 0; d < 3; ++d) val *= power(x[d], p[d]);
    value.value += val;
    for (int d = 0; d < 3; ++d) {
      if (!p[d]) continue;
      float grad = h.value * p[d];
      for (int j = 0; j < 3; ++j) grad *= power(x[j], p[j] - (d == j));
      value.derivative[d] += grad;
    }
  }
  return value;
}
template<class Layout>
__device__ inline Value component_primitive(Layout v, const float* s, Value harmonic, int a) {
  int component = 0, exponent = 0;
  for (int p = 0; p < v.primitive_count; ++p) {
    const Primitive label = v.primitive(p);
    if (a >= label.offset && a < label.offset + 2*label.l + 1) {
      component = label.l*label.l + a - label.offset;
      exponent = (label.degree - label.l) / 2;
    }
  }
  Value y;
  y.value = __shfl_sync(0xffffffff, harmonic.value, component);
  for (int d = 0; d < 3; ++d)
    y.derivative[d] = __shfl_sync(0xffffffff, harmonic.derivative[d], component);
  const float s2 = s[0]*s[0] + s[1]*s[1] + s[2]*s[2];
  const float factor = power(s2, exponent);
  const float grad_factor = exponent ? 2*exponent*power(s2, exponent-1) : 0;
  Value result;
  result.value = factor * y.value;
  for (int d = 0; d < 3; ++d)
    result.derivative[d] = factor*y.derivative[d] + grad_factor*s[d]*y.value;
  return result;
}
// A warp owns one center/channel; spin rows are shared in a-major order so
// adjacent lanes read different banks while traversing the same harmonic.
template<class Layout, bool CenterVirial = false>
static __global__ void forces_components(
  Layout v, int atom_count, int n, int structural_dim, int types, int na, int basis, Geometry box,
  const int* type, const int* active, const double* positions, const double* spins,
  const int* nn, const int* nl, const double* edges, int plane, const float* cutoffs,
  const float* angular, const float* parameters, const float* moments, const float* fp,
  double* force, double* mforce, double* virial, int virial_mode,
  float* virial_float, float* transfer, const int* reverse_index) {
  constexpr int width = 32, groups = 128 / width;
  constexpr int harmonics = Layout::harmonic_capacity, spin_capacity = Layout::spin_capacity;
  struct State {
    Value center[spin_capacity], y[harmonics];
    float pull[Layout::component_capacity], density[spin_capacity][harmonics];
  };
  __shared__ State states[groups];
  const int lane = threadIdx.x % width;
  const int work = (blockIdx.x * blockDim.x + threadIdx.x) / width;
  const int i = work / v.channels, c = work % v.channels;
  if (i >= atom_count) return;
  constexpr unsigned mask = 0xffffffff;
  State& state = states[threadIdx.x / width];
  const int geometry_count = (v.lmax + 1) * (v.lmax + 1);
  const int* order = reverse_index + geometry_count + v.spin_components + 1;
  const int* harmonic_offsets = order + v.term_count;
  const int* harmonic_order = harmonic_offsets + 26;
  constexpr int spin_rounds = (spin_capacity + width - 1) / width;
  const float* weights = parameters + (na+1)*v.channels;
  const float* center = weights + v.path_count*v.channels;
  float si[3];
  read_spin(n, i, type, active, spins, si);
  const Value center_harmonic = lane < geometry_count ?
    component_harmonic(v, si, lane, harmonic_offsets, harmonic_order) : Value{};
  #pragma unroll
  for (int round = 0; round < spin_rounds; ++round) {
    const int a = lane + round*width;
    const Value phi = component_primitive(v, si, center_harmonic, a);
    if (a < v.spin_components) state.center[a] = phi;
  }
  for (int k = 0; k < v.sector_count; ++k) {
    const Sector sector = v.sector(k);
    const float derivative = 2*fp[(structural_dim+sector.norm_column+c)*n+i]/sector.size;
    for (int m = lane; m < sector.size; m += width)
      state.pull[sector.offset+m] = derivative*moments[(c*v.components+sector.offset+m)*n+i];
    if (lane == 0 && sector.scalar_column >= 0)
      state.pull[sector.offset] += fp[(structural_dim+sector.scalar_column+c)*n+i];
  }
  __syncwarp(mask);
  float center_value = 0, center_gradient[3] = {}, geometry_sum = 0;
  if (lane < geometry_count) {
    for (int k = reverse_index[lane]; k < reverse_index[lane+1]; ++k) {
      const Term term = v.term(order[k]);
      const float adjoint = state.pull[term.output] *
        weights[term.path*v.channels+c] * term.coefficient;
      center_value += adjoint * state.center[term.spin].value;
      for (int d = 0; d < 3; ++d)
        center_gradient[d] += adjoint * state.center[term.spin].derivative[d];
    }
  }
  #pragma unroll
  for (int round = 0; round < spin_rounds; ++round) {
    const int a = lane + round*width;
    if (a > 0 && a < v.spin_components) {
      for (int g = 0; g < geometry_count; ++g) state.density[a][g] = 0;
      for (int k = reverse_index[geometry_count+a]; k < reverse_index[geometry_count+a+1]; ++k) {
        const Term term = v.term(order[k]);
        state.density[a][term.geometry] += state.pull[term.output] *
          weights[term.path*v.channels+c] * term.coefficient;
      }
    }
  }
  __syncwarp(mask);
  float center_pull[3] = {};
  double center_force[3] = {}, center_virial[CenterVirial ? 9 : 1] = {};
  if (lane == 0) {
    for (int k = 0; k < v.center_count; ++k) {
      const Center term = v.center(k);
      for (int d = 0; d < 3; ++d)
        center_pull[d] += state.pull[term.output] * center[term.weight*v.channels+c] *
          state.center[term.primitive].derivative[d];
    }
    if (c == 0) {
      int column = structural_dim;
      for (int p = 1; p < v.primitive_count; ++p)
        if (v.primitive(p).l == 0) {
          for (int d = 0; d < 3; ++d)
            center_pull[d] += fp[column*n+i] * state.center[v.primitive(p).offset].derivative[d];
          ++column;
        }
    }
  }
  for (int neighbor = 0; neighbor < nn[i]; ++neighbor) {
    const int slot = neighbor*n+i, j = nl[slot], pair = type[i]*types+type[j];
    float r[3], unit[3], sj[3];
    displacement(n, i, j, slot, box, positions, edges, plane, r);
    const float distance = sqrtf(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
    if (!(distance > 1.0e-12f && distance < cutoffs[pair])) continue;
    for (int d = 0; d < 3; ++d) unit[d] = r[d] / distance;
    const Value y = lane < geometry_count ?
      component_harmonic(v, unit, lane, harmonic_offsets, harmonic_order) : Value{};
    if (lane < geometry_count) state.y[lane] = y;
    read_spin(n, j, type, active, spins, sj);
    const Value spin_harmonic = lane < geometry_count ?
      component_harmonic(v, sj, lane, harmonic_offsets, harmonic_order) : Value{};
    Value neighbor_spin[spin_rounds];
    #pragma unroll
    for (int round = 0; round < spin_rounds; ++round)
      neighbor_spin[round] = component_primitive(v, sj, spin_harmonic, lane + round*width);
    float weight, derivative;
    radial(v, slot, plane, c, v.channels, na, basis, pair, cutoffs[pair], distance,
           angular, parameters, weight, derivative);
    __syncwarp(mask);
    float radial_pull = 0, angular_pull[3] = {}, neighbor_pull[3] = {};
    if (lane < geometry_count) {
      geometry_sum += weight * y.value;
      radial_pull += center_value * y.value;
      for (int d = 0; d < 3; ++d)
        angular_pull[d] += weight * center_value * y.derivative[d];
    }
    #pragma unroll
    for (int round = 0; round < spin_rounds; ++round) {
      const int a = lane + round*width;
      if (a > 0 && a < v.spin_components) {
        const Value spin = neighbor_spin[round];
        float value_pull = 0, direction_pull[3] = {};
        for (int g = 0; g < geometry_count; ++g) {
          const float coefficient = state.density[a][g];
          value_pull += coefficient * state.y[g].value;
          for (int d = 0; d < 3; ++d)
            direction_pull[d] += coefficient * state.y[g].derivative[d];
        }
        radial_pull += value_pull * spin.value;
        for (int d = 0; d < 3; ++d) {
          angular_pull[d] += weight * spin.value * direction_pull[d];
          neighbor_pull[d] += weight * value_pull * spin.derivative[d];
        }
      }
    }
    #pragma unroll
    for (int offset = width/2; offset > 0; offset /= 2) {
      radial_pull += __shfl_down_sync(mask, radial_pull, offset);
      for (int d = 0; d < 3; ++d) {
        angular_pull[d] += __shfl_down_sync(mask, angular_pull[d], offset);
        neighbor_pull[d] += __shfl_down_sync(mask, neighbor_pull[d], offset);
      }
    }
    if (lane == 0) {
      float longitudinal = 0, grad[3];
      for (int d = 0; d < 3; ++d) longitudinal += angular_pull[d] * unit[d];
      for (int d = 0; d < 3; ++d) {
        grad[d] = derivative*radial_pull*unit[d] +
          (angular_pull[d]-longitudinal*unit[d]) / distance;
        center_force[d] += static_cast<double>(grad[d]);
        atomicAdd(force+d*n+j, -static_cast<double>(grad[d]));
        if (active[type[j]]) atomicAdd(mforce+d*n+j, -static_cast<double>(neighbor_pull[d]));
      }
      #pragma unroll
      for (int a = 0; a < 3; ++a)
        #pragma unroll
        for (int b = 0; b < 3; ++b) {
          const int raw_to_internal[9] = {0,3,4,6,1,5,7,8,2};
          const int k = raw_to_internal[3*a+b];
          const double value = -static_cast<double>(r[a]) * grad[b];
          if constexpr (CenterVirial) center_virial[k] += value;
          if (virial_mode == 2) atomicAdd(virial+k*n+j, value);
          if (virial_mode == 3) atomicAdd(virial_float+k*n+j, static_cast<float>(value));
          if (transfer && active[type[j]])
            atomicAdd(transfer+(3*a+b)*n+j, -r[a]*neighbor_pull[b]);
        }
    }
    __syncwarp(mask);
  }
  float geometry_pull[3];
  for (int d = 0; d < 3; ++d) geometry_pull[d] = geometry_sum * center_gradient[d];
  #pragma unroll
  for (int offset = width/2; offset > 0; offset /= 2)
    for (int d = 0; d < 3; ++d)
      geometry_pull[d] += __shfl_down_sync(mask, geometry_pull[d], offset);
  if (lane != 0) return;
  for (int d = 0; d < 3; ++d) {
    atomicAdd(force+d*n+i, center_force[d]);
    if (active[type[i]])
      atomicAdd(mforce+d*n+i, -static_cast<double>(center_pull[d] + geometry_pull[d]));
  }
  if constexpr (CenterVirial)
    for (int k = 0; k < 9; ++k) atomicAdd(virial+k*n+i, center_virial[k]);
}
} // namespace nep_adapters::spin_density
