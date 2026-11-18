/* Adapted from GPUMD 73f6b09. SPDX-License-Identifier: LGPL-3.0-or-later */
#pragma once
#include "simulation_box_device.cuh"
#include "spin_md_projection.hpp"
#include <type_traits>

namespace nep_adapters::spin_density::md {
struct Geometry {
  cuda_backend::SimulationBox box;
  int neighbor_source;
  const int* atom_to_structure;
  const double* boxes;
  const double* inverse;
  const int* pbc;
};
struct GhostCoordinates {};

using WarpMask = unsigned int;
__device__ __forceinline__ WarpMask tile_mask(int width)
{
  const int first = (threadIdx.x % warpSize) / width * width;
  return (~WarpMask{0} >> (sizeof(WarpMask) * 8 - width)) << first;
}

__device__ __forceinline__ WarpMask warp_ballot(int predicate)
{
  return __ballot_sync(~WarpMask{0}, predicate);
}
template <class Real>
struct alignas(2 * sizeof(Real)) RadialEdge {
  Real value, derivative;
};

// Each atom/channel owns a tile of consecutive neighbors. Adjacent work groups own adjacent tiles.
template <int Tile>
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
constexpr size_t radial_edge_index(int atoms, int channels, int atom, int channel, int neighbor)
{
  return ((size_t(neighbor / Tile) * atoms + atom) * channels + channel) * Tile + neighbor % Tile;
}

struct HarmonicDerivative {
  int output, axis, input;
  double coefficient;
};
static constexpr HarmonicDerivative harmonic_derivatives[] = {
  {1, 0, 0, 1.732050807568879},      {2, 1, 0, 1.7320508075688781},
  {3, 2, 0, 1.7320508075688774},     {4, 0, 3, 2.2360679774997902},
  {4, 2, 1, 2.236067977499788},      {5, 0, 2, 2.2360679774997894},
  {5, 1, 1, 2.2360679774997885},     {6, 0, 1, -1.2909944487358058},
  {6, 1, 2, 2.5819888974716108},     {6, 2, 3, -1.2909944487358065},
  {7, 1, 3, 2.2360679774997925},     {7, 2, 2, 2.2360679774997911},
  {8, 0, 1, -2.2360679774997894},    {8, 2, 3, 2.2360679774997898},
  {9, 0, 8, 3.2403703492039297},     {9, 2, 4, 3.2403703492039315},
  {10, 0, 7, 2.6457513110645863},    {10, 1, 4, 2.6457513110645885},
  {10, 2, 5, 2.6457513110645885},    {11, 0, 6, 2.8982753492378874},
  {11, 0, 8, 0.83666002653407379},   {11, 1, 5, 3.3466401061363009},
  {11, 2, 4, -0.83666002653407578},  {12, 0, 5, -2.0493901531919207},
  {12, 1, 6, 3.5496478698597742},    {12, 2, 7, -2.0493901531919194},
  {13, 0, 4, -0.83666002653407456},  {13, 1, 7, 3.3466401061362983},
  {13, 2, 6, 2.8982753492378874},    {13, 2, 8, -0.83666002653407623},
  {14, 0, 5, -2.6457513110645925},   {14, 1, 8, 2.6457513110645903},
  {14, 2, 7, 2.6457513110645876},    {15, 0, 4, -3.2403703492039306},
  {15, 2, 8, 3.2403703492039297},    {16, 0, 15, 4.2426406871192937},
  {16, 2, 9, 4.2426406871192874},    {17, 0, 14, 3.6742346141747642},
  {17, 1, 9, 2.9999999999999973},    {17, 2, 10, 3.674234614174765},
  {18, 0, 13, 3.1052950170405951},   {18, 0, 15, 0.80178372573727519},
  {18, 1, 10, 3.9279220242478634},   {18, 2, 9, -0.80178372573727175},
  {18, 2, 11, 3.1052950170405942},   {19, 0, 12, 3.5856858280031796},
  {19, 0, 14, 1.3887301496588282},   {19, 1, 11, 4.3915503282684014},
  {19, 2, 10, -1.3887301496588274},  {20, 0, 11, -2.7774602993176565},
  {20, 1, 12, 4.5355736761107295},   {20, 2, 13, -2.7774602993176578},
  {21, 0, 10, -1.3887301496588287},  {21, 1, 13, 4.3915503282684059},
  {21, 2, 12, 3.5856858280031836},   {21, 2, 14, -1.3887301496588267},
  {22, 0, 9, -0.80178372573727374},  {22, 0, 11, -3.1052950170405955},
  {22, 1, 14, 3.9279220242478599},   {22, 2, 13, 3.1052950170405937},
  {22, 2, 15, -0.80178372573727208}, {23, 0, 10, -3.6742346141747726},
  {23, 1, 15, 3.0000000000000027},   {23, 2, 14, 3.6742346141747677},
  {24, 0, 9, -4.2426406871192874},   {24, 2, 15, 4.2426406871192892},
};
template <class Real>
struct SpinValue {
  Real value, derivative[3];
};
template <class Real>
__device__ inline Real power(Real x, int n)
{
  Real result = 1;
  for (int i = 0; i < n; ++i)
    result *= x;
  return result;
}
// Harmonic monomials are shared by every model. Literal exponents let CUDA
// fold their powers and derivatives in both the forward and force kernels.
template <size_t I, class Real>
__device__ __forceinline__ void add_harmonic(int lmax, const Real* x, SpinValue<Real>* y)
{
  constexpr auto h = harmonics[I];
  if (h.x + h.y + h.z > lmax)
    return;
  constexpr int p[3] = {h.x, h.y, h.z};
  Real val = h.value;
#pragma unroll
  for (int d = 0; d < 3; ++d)
    val *= power(x[d], p[d]);
  y[h.component].value += val;
#pragma unroll
  for (int d = 0; d < 3; ++d) {
    if (p[d]) {
      Real grad = h.value * p[d];
#pragma unroll
      for (int k = 0; k < 3; ++k)
        grad *= power(x[k], p[k] - (d == k));
      y[h.component].derivative[d] += grad;
    }
  }
}

template <class Real, size_t... I>
__device__ __forceinline__ void spherical_terms(
  int lmax, const Real* x, SpinValue<Real>* y, std::index_sequence<I...>)
{
  int ordered[] = {0, (add_harmonic<I>(lmax, x, y), 0)...};
  (void)ordered;
}

template <class Layout, class Real>
__device__ __forceinline__ void spherical(Layout v, const Real* x, SpinValue<Real>* y)
{
#pragma unroll
  for (int i = 0; i < Layout::harmonic_capacity; ++i)
    y[i] = {0, {0, 0, 0}};
  spherical_terms(v.lmax, x, y, std::make_index_sequence<sizeof(harmonics) / sizeof(Harmonic)>{});
}
template <size_t I, int H, class Real>
__device__ __forceinline__ void apply_harmonic_gradient(
  const Real* coeff, const Real* y, Real* grad)
{
  constexpr auto t = harmonic_derivatives[I];
  if (t.output < H)
    grad[t.axis] += (coeff[t.output] * Real(t.coefficient)) * y[t.input];
}

template <int H, class Real, size_t... I>
__device__ __forceinline__ void harmonic_pull(
  const Real* coeff, const Real* y, Real* grad, std::index_sequence<I...>)
{
  int ordered[] = {0, (apply_harmonic_gradient<I, H>(coeff, y, grad), 0)...};
  (void)ordered;
}
using HarmonicDerivativeIndices =
  std::make_index_sequence<sizeof(harmonic_derivatives) / sizeof(HarmonicDerivative)>;

template <class Layout, class Real>
__device__ __forceinline__ void primitives_at(Layout v, const Real* s, SpinValue<Real>* phi)
{
  SpinValue<Real> y[Layout::harmonic_capacity];
  spherical(v, s, y);
  const Real s2 = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
#pragma unroll
  for (int p = 0; p < v.primitive_count; ++p) {
    const Primitive label = v.primitive(p);
    const int exponent = (label.degree - label.l) / 2;
    const Real factor = power(s2, exponent);
    const Real grad_factor = exponent ? 2 * exponent * power(s2, exponent - 1) : 0;
    for (int m = 0; m < 2 * label.l + 1; ++m) {
      const SpinValue<Real> harmonic = y[label.l * label.l + m];
      SpinValue<Real>& value = phi[label.offset + m];
      value.value = factor * harmonic.value;
      for (int d = 0; d < 3; ++d)
        value.derivative[d] = factor * harmonic.derivative[d] + grad_factor * s[d] * harmonic.value;
    }
  }
}
// Reverse the radial spin powers before applying the solid-harmonic derivative.
template <class Layout, class Real>
__device__ __forceinline__ void primitive_pull(
  Layout v, const Real* s, const SpinValue<Real>* phi, const Real* beta, Real* grad)
{
  Real coeff[Layout::harmonic_capacity] = {}, value[Layout::harmonic_capacity] = {};
  const Real s2 = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
  Real radial = 0;
#pragma unroll
  for (int p = 0; p < v.primitive_count; ++p) {
    const Primitive label = v.primitive(p);
    const int exponent = (label.degree - label.l) / 2;
    const Real scale = power(s2, exponent);
    const Real slope = exponent ? 2 * exponent * power(s2, exponent - 1) : 0;
    for (int m = 0; m < 2 * label.l + 1; ++m) {
      const int h = label.l * label.l + m;
      // Primitives are ordered by degree: the bare harmonic precedes its radial powers.
      if (exponent == 0)
        value[h] = phi[label.offset + m].value;
      coeff[h] += scale * beta[label.offset + m];
      if (exponent)
        radial += slope * beta[label.offset + m] * value[h];
    }
  }
  for (int k = 0; k < 3; ++k)
    grad[k] = radial * s[k];
  harmonic_pull<Layout::harmonic_capacity>(coeff, value, grad, HarmonicDerivativeIndices{});
}
template <class Real>
__device__ inline void
read_spin(int n, int i, const int* type, const int* active, const double* spins, Real* s)
{
  for (int d = 0; d < 3; ++d)
    s[d] = active[type[i]] ? spins[d * n + i] : 0;
}
template <class Real>
__device__ inline void displacement(
  int n,
  int i,
  int j,
  int slot,
  Geometry geometry,
  const double* positions,
  const double*,
  int,
  Real* r)
{
  double dx = positions[j] - positions[i];
  double dy = positions[n+j] - positions[n+i];
  double dz = positions[2*n+j] - positions[2*n+i];
  if (geometry.neighbor_source == 0) {
    auto box = geometry.box;
    if (geometry.atom_to_structure && geometry.boxes)
      box = cuda_backend::load_structure_box(
        geometry.atom_to_structure[i], geometry.boxes, geometry.inverse, geometry.pbc);
    cuda_backend::minimum_image_delta(box, dx, dy, dz, r[0], r[1], r[2]);
  } else {
    r[0] = dx;
    r[1] = dy;
    r[2] = dz;
  }
}

template <class Real>
__device__ __forceinline__ void displacement(
  int n, int i, int j, int, GhostCoordinates, const double* positions,
  const double*, int, Real* r)
{
  r[0] = static_cast<Real>(positions[j] - positions[i]);
  r[1] = static_cast<Real>(positions[n + j] - positions[n + i]);
  r[2] = static_cast<Real>(positions[2 * n + j] - positions[2 * n + i]);
}
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

static __device__ __host__ __forceinline__ void
find_fc_and_fcp(double rc, double rcinv, double d12, double& fc, double& fcp)
{
  if (d12 < rc) {
    const double pi = 3.14159265358979323846;
    const double phase = pi * d12 * rcinv;
    fc = 0.5 * (1.0 + cos(phase));
    fcp = -0.5 * pi * rcinv * sin(phase);
  } else {
    fc = fcp = 0.0;
  }
}

template <typename Real, class Consume>
static __device__ __host__ __forceinline__ void for_each_radial_basis(
  const int n_max,
  const Real rcinv,
  const Real d12,
  const Real fc12,
  const Real fcp12,
  Consume consume)
{
  Real d12_mul_rcinv = d12 * rcinv;
  Real x = 2.0f * (d12_mul_rcinv - 1.0f) * (d12_mul_rcinv - 1.0f) - 1.0f;
  consume(0, fc12, fcp12);
  if (n_max == 0)
    return;
  consume(1, (x + 1.0f) * 0.5f * fc12,
          2.0f * (d12_mul_rcinv - 1.0f) * rcinv * fc12 + (x + 1.0f) * 0.5f * fcp12);
  Real u0 = 1.0f;
  Real u1 = 2.0f * x;
  Real u2;
  Real fn_m_minus_2 = 1.0f;
  Real fn_m_minus_1 = x;
  for (int m = 2; m <= n_max; ++m) {
    Real fn_tmp1 = 2.0f * x * fn_m_minus_1 - fn_m_minus_2;
    fn_m_minus_2 = fn_m_minus_1;
    fn_m_minus_1 = fn_tmp1;
    Real fnp_tmp = m * u1;
    u2 = 2.0f * x * u1 - u0;
    u0 = u1;
    u1 = u2;

    Real fn_tmp2 = (fn_tmp1 + 1.0f) * 0.5f;
    consume(m, fn_tmp2 * fc12,
            (fnp_tmp * 2.0f * (d12 * rcinv - 1.0f) * rcinv) * fc12 + fn_tmp2 * fcp12);
  }
}

template <typename Real>
static __device__ __host__ __forceinline__ void find_fn_and_fnp(
  const int n_max,
  const Real rcinv,
  const Real d12,
  const Real fc12,
  const Real fcp12,
  Real* fn,
  Real* fnp)
{
  for_each_radial_basis(n_max, rcinv, d12, fc12, fcp12, [&](int k, Real f, Real df) {
    fn[k] = f;
    fnp[k] = df;
  });
}

template <bool StageBasis = false, class Real>
__device__ inline void radial(
  int c,
  int channels,
  int basis,
  int pair,
  float cutoff,
  Real distance,
  const Real* coefficients,
  Real& weight,
  Real& derivative)
{
  Real fc, fcp;
  const Real inv = Real(1) / cutoff;
  find_fc_and_fcp(Real(cutoff), inv, distance, fc, fcp);
  weight = derivative = 0;
  const Real* mix = coefficients + (pair * channels + c) * (basis + 1);
  auto accumulate = [&](int k, Real f, Real df) {
    weight += f * mix[k];
    derivative += df * mix[k];
  };
  if (StageBasis) {
    Real fn[17], fnp[17];
    find_fn_and_fnp(basis, inv, distance, fc, fcp, fn, fnp);
    for (int k = 0; k <= basis; ++k)
      accumulate(k, fn[k], fnp[k]);
  } else {
    for_each_radial_basis(basis, inv, distance, fc, fcp, accumulate);
  }
}

template <class Layout> struct CenterMomentRows : std::false_type {};

template <class Layout>
static __device__ __forceinline__ size_t moment_index(Layout const& v, int n, int i, int c, int m)
{
  return CenterMomentRows<Layout>::value ? (size_t(i) * v.channels + c) * v.components + m
                                         : size_t(c * v.components + m) * n + i;
}

template <bool WarpDensity = false, bool StoreRadial = false, class Layout,
          class Coordinates, class Real = float, class Output = float>
static __global__ void descriptors(
  Layout v,
  int count_local,
  int n,
  int structural_dim,
  int types,
  int na,
  int basis,
  Coordinates box,
  const int* type,
  const int* active,
  const double* positions,
  const double* spins,
  const int* nn,
  const int* nl,
  const double* edges,
  int plane,
  const float* cutoffs,
  const Real* coefficients,
  const float* parameters,
  float* moments,
  Output* q,
  ProjectionView projection = {},
  float* geometry_moments = nullptr,
  RadialEdge<Real>* radial_edges = nullptr)
{
  constexpr int lanes = WarpDensity ? int(SpinProjection::width) : int(Layout::tile);
  const int work = (blockIdx.x * blockDim.x + threadIdx.x) / lanes;
  const int lane = threadIdx.x % lanes;
  const int i = work / v.channels, c = work % v.channels;
  // Capture participation before divergent neighbor loops. An activemask taken
  // at a reduction can omit lanes still processing their last neighbor.
  const WarpMask mask = WarpDensity ? tile_mask(lanes) : (lanes > 1 ? warp_ballot(i < count_local) : 0);
  if (i >= count_local)
    return;
  constexpr int geometry = Layout::harmonic_capacity, spin = Layout::spin_capacity;
  constexpr int elements = geometry * spin;
  // Padding lets lanes read different components of the same edge without bank conflicts.
  constexpr int edge_stride = WarpDensity ? lanes + 1 : 1;
  union Scratch {
    struct {
      Real y[WarpDensity ? geometry : 1][edge_stride];
      Real phi[WarpDensity ? spin : 1][edge_stride];
    } edge;
    struct {
      Real density[WarpDensity ? elements : 1];
      Real moments[WarpDensity ? Layout::component_capacity : 1];
    } result;
  };
  __shared__ Scratch scratch[WarpDensity ? 2 : 1]; // 64 threads per block.
  Scratch& tile = scratch[WarpDensity ? threadIdx.x / lanes : 0];
  Real private_h[WarpDensity ? 1 : Layout::component_capacity] = {};
  Real* h = WarpDensity ? tile.result.moments : private_h;
  Real s[3];
  SpinValue<Real> phi_i[Layout::spin_capacity];
  read_spin(n, i, type, active, spins, s);
  primitives_at(v, s, phi_i);
  const float* weights = parameters + (na + 1) * v.channels;
  const float* center = weights + v.path_count * v.channels;
  Real density[WarpDensity ? 1 : geometry][WarpDensity ? 1 : spin] = {};
  Real sums[WarpDensity ? (elements + lanes - 1) / lanes : 1] = {};
  const int geometry_count = (v.lmax + 1) * (v.lmax + 1);
  auto edge_basis = [&](int neighbor, SpinValue<Real>* y, SpinValue<Real>* phi_j, Real& weight) {
    int slot = neighbor * n + i, j = nl[slot], pair = type[i] * types + type[j];
    Real r[3], u[3];
    displacement(n, i, j, slot, box, positions, edges, plane, r);
    Real distance = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    if (!(distance > Real(1e-12) && distance < cutoffs[pair]))
      return false;
    for (int d = 0; d < 3; ++d)
      u[d] = r[d] / distance;
    spherical(v, u, y);
    read_spin(n, j, type, active, spins, s);
    primitives_at(v, s, phi_j);
    Real derivative;
    radial(
      c,
      v.channels,
      basis,
      pair,
      cutoffs[pair],
      distance,
      coefficients,
      weight,
      derivative);
    if (StoreRadial)
      radial_edges[radial_edge_index<Layout::tile>(n, v.channels, i, c, neighbor)] = {weight, derivative};
    return true;
  };
  const int count = nn[i];
  for (int base = 0; base < (WarpDensity ? count : count - lane); base += lanes) {
    SpinValue<Real> y[geometry], phi_j[spin];
    Real weight;
    const int neighbor = base + lane;
    const bool inside = (!WarpDensity || neighbor < count) && edge_basis(neighbor, y, phi_j, weight);
    if (WarpDensity) {
      // Load one edge per lane, then distribute the density matrix over registers.
#pragma unroll
      for (int g = 0; g < geometry; ++g)
        tile.edge.y[g][lane] = inside ? weight * y[g].value : 0;
#pragma unroll
      for (int a = 0; a < spin; ++a)
        tile.edge.phi[a][lane] = inside ? (a == 0 ? Real(1) : phi_j[a].value) : 0;
      __syncwarp(mask);
      const int batch = count - base < lanes ? count - base : lanes;
      for (int e = 0; e < batch; ++e) {
#pragma unroll
        for (int k = 0; k < (elements + lanes - 1) / lanes; ++k) {
          const int element = k * lanes + lane;
          if (element < elements)
            sums[k] += tile.edge.y[element / spin][e] * tile.edge.phi[element % spin][e];
        }
      }
      __syncwarp(mask);
    } else if (inside) {
      for (int g = 0; g < geometry_count; ++g) {
        const Real weighted_y = weight * y[g].value;
        density[g][0] += weighted_y;
#pragma unroll
        for (int a = 1; a < v.spin_components; ++a)
          density[g][a] += weighted_y * phi_j[a].value;
      }
    }
  }
  if (WarpDensity) {
#pragma unroll
    for (int k = 0; k < (elements + lanes - 1) / lanes; ++k)
      if (k * lanes + lane < elements)
        tile.result.density[k * lanes + lane] = sums[k];
    __syncwarp(mask);
    if (geometry_moments) {
      for (int g = lane; g < geometry; g += lanes)
        geometry_moments[(size_t(i) * v.channels + c) * geometry + g] = tile.result.density[g * spin];
    }
    for (int m = lane; m < v.components; m += lanes) {
      const ProjectionRow row = projection.rows[m];
      Real value = 0;
      for (int j = 0; j < row.count; ++j) {
        const ProjectionTerm term = projection.terms[row.begin + (j * v.channels + c) * lanes];
        Real product = tile.result.density[term.input];
        if (term.center_spin >= 0)
          product *= phi_i[term.center_spin].value;
        value += term.weight * product;
      }
      h[row.output] = value;
    }
    __syncwarp(mask);
  } else {
    if (geometry_moments) {
      for (int g = 0; g < geometry_count; ++g) {
        Real sum = density[g][0];
#pragma unroll
        for (int offset = Layout::tile / 2; offset; offset /= 2)
          sum += __shfl_down_sync(mask, sum, offset, Layout::tile);
        if (lane == 0)
          geometry_moments[(size_t(i) * v.channels + c) * geometry + g] = sum;
      }
    }
    for (int t = 0; t < v.term_count; ++t) {
      const Term term = v.term(t);
      const Real* sums = density[term.geometry];
      const Real product = term.neighbor ? sums[term.spin] : sums[0] * phi_i[term.spin].value;
      h[term.output] += weights[term.path * v.channels + c] * term.coefficient * product;
    }
#pragma unroll
    for (int m = 0; m < v.components; ++m) {
#pragma unroll
      for (int offset = Layout::tile / 2; offset > 0; offset /= 2)
        h[m] += __shfl_down_sync(mask, h[m], offset, Layout::tile);
    }
    if (lane != 0)
      return;
  }
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < v.center_count; ++t) {
      const Center term = v.center(t);
      h[term.output] += center[term.weight * v.channels + c] * phi_i[term.primitive].value;
    }
  }
  if (WarpDensity)
    __syncwarp(mask);
#pragma unroll
  for (int m = lane; m < v.components; m += WarpDensity ? lanes : 1)
    moments[moment_index(v, n, i, c, m)] = h[m];
  if (lane == 0 && c == 0) {
    int column = structural_dim;
#pragma unroll
    for (int p = 1; p < v.primitive_count; ++p)
      if (v.primitive(p).l == 0)
        q[(column++) * n + i] = phi_i[v.primitive(p).offset].value;
  }
#pragma unroll
  for (int k = lane; k < v.sector_count; k += WarpDensity ? lanes : 1) {
    Sector sector = v.sector(k);
    Real norm = 0;
    for (int m = 0; m < sector.size; ++m)
      norm += h[sector.offset + m] * h[sector.offset + m];
    q[(structural_dim + sector.norm_column + c) * n + i] = norm / (sector.size);
    if (sector.scalar_column >= 0)
      q[(structural_dim + sector.scalar_column + c) * n + i] = h[sector.offset];
  }
}

template <class Layout>
__device__ __forceinline__ Term force_term(Layout v, int i, std::false_type)
{
  return v.term(i);
}
template <class Layout>
__device__ __forceinline__ Term force_term(Layout v, int i, std::true_type)
{
  return v.quadratic_term(i);
}

template <class Real>
__device__ __forceinline__ void quadratic_gradient(const Real* a, const Real* s, Real* grad)
{
  grad[0] = a[0] + 2 * a[3] * s[0] + a[6] * s[1] + a[7] * s[2];
  grad[1] = a[1] + 2 * a[4] * s[1] + a[6] * s[0] + a[8] * s[2];
  grad[2] = a[2] + 2 * a[5] * s[2] + a[7] * s[0] + a[8] * s[1];
}

template <
  class Layout,
  bool CenterVirial = false,
  bool Quadratic = false,
  bool FuseChannels = false,
  bool CacheRows = false,
  bool ReadRadial = false,
  bool NeighborOutputs = true,
  class Coordinates,
  class Real = float>
static __global__ void forces(
  Layout v,
  int count_local,
  int n,
  int structural_dim,
  int types,
  int na,
  int basis,
  Coordinates box,
  const int* type,
  const int* active,
  const double* positions,
  const double* spins,
  const int* nn,
  const int* nl,
  const double* edges,
  int plane,
  const float* cutoffs,
  const Real* coefficients,
  const float* parameters,
  const float* moments,
  const float* fp,
  double* force,
  double* mforce,
  double* virial,
  int virial_mode,
  float* virial_float,
  float* transfer,
  const int* adjoint_order = nullptr,
  ProjectionView transpose = {},
  const RadialEdge<Real>* radial_edges = nullptr)
{
  static_assert(
    !Quadratic ||
      (Layout::spin_capacity == 10 && Layout::tile > 1),
    "quadratic tiled layout required");
  // A single channel reuses less geometry; cap its live density matrix at 2 KiB.
  constexpr bool single_pull = Layout::tile == SpinProjection::width &&
                               Layout::harmonic_capacity * Layout::spin_capacity <= 512;
  constexpr bool direct_pull = CenterVirial && !Quadratic && std::is_same<Real, float>::value &&
                               (FuseChannels || single_pull);
  static_assert(!FuseChannels || Quadratic || direct_pull,
                "channel fusion requires a tiled adjoint");
  constexpr int channels = FuseChannels ? 3 : 1;
  const int work = (blockIdx.x * blockDim.x + threadIdx.x) / Layout::tile;
  const int lane = threadIdx.x % Layout::tile;
  const int batches = (v.channels + channels - 1) / channels;
  const int i = work / batches;
  const int first_channel = (work % batches) * channels;
  const int channel_count = FuseChannels && CenterVirial && first_channel + channels > v.channels
    ? v.channels - first_channel : channels;
  // Capture participation before divergent neighbor loops. An activemask taken
  // at a reduction can omit lanes still processing their last neighbor.
  const WarpMask mask = direct_pull ? tile_mask(Layout::tile) :
                                  (Layout::tile > 1 ? warp_ballot(i < count_local) : 0);
  if (i >= count_local)
    return;
  Real s[3], center_pull[3] = {};
  double center_force[3] = {};
  double center_virial[CenterVirial ? 9 : 1] = {};
  SpinValue<Real> phi_i[Layout::spin_capacity];
  read_spin(n, i, type, active, spins, s);
  primitives_at(v, s, phi_i);
  const int geometry_count = (v.lmax + 1) * (v.lmax + 1);
  constexpr int groups = 64 / Layout::tile;
  constexpr bool use_geometry = CacheRows;
  constexpr int row_size = Layout::spin_capacity + (use_geometry ? 0 : 3);
  constexpr int density_size = Layout::harmonic_capacity * row_size;
  // Reuse the center adjoint when a full block fits in static shared memory.
  constexpr bool shared_density = CenterVirial && !Quadratic && Layout::tile > 1 &&
                                  density_size * groups * channels * sizeof(Real) + sizeof(Real) <= 48 * 1024;
  Real private_density[Quadratic || shared_density ? 1 : density_size * channels] = {};
  __shared__ Real tile_density[shared_density ? density_size * groups * channels : 1];
  const int group = threadIdx.x / Layout::tile;
  Real* group_density = shared_density ? tile_density + group * density_size * channels : private_density;
  constexpr bool projected = shared_density && (Layout::tile == SpinProjection::width || use_geometry);
  static_assert(!use_geometry || projected, "cached geometry requires projected density rows");
  if (shared_density && !projected) {
    for (int k = lane; k < density_size * channels; k += Layout::tile)
      group_density[k] = 0;
    __syncwarp(mask);
  }
  Real geometry_sum[direct_pull || use_geometry ? 1 : channels][direct_pull || use_geometry ? 1 : Layout::harmonic_capacity] = {};
  constexpr int harmonics_per_channel = Layout::harmonic_capacity;
  // Large quadratic matrices amortize a single Cartesian pull across all harmonic rows.
  constexpr bool quadratic_pull = Quadratic && CenterVirial && harmonics_per_channel > 16;
  // Staging bounds register use when a medium quadratic matrix also needs all Cartesian derivatives.
  constexpr bool stage_radial = Quadratic && CenterVirial && harmonics_per_channel >= 16 && !quadratic_pull;
  constexpr int neighbor_offset = 4 * harmonics_per_channel;
  constexpr int adjoint_size = neighbor_offset + 9 * harmonics_per_channel;
  __shared__ volatile Real edge_pull[Quadratic ? adjoint_size * groups * channels : 1];
  const float* weights = parameters + (na + 1) * v.channels;
  const float* center = weights + v.path_count * v.channels;
  // Reuse each edge geometry across the channels in this batch.
  for (int channel = 0; channel < channel_count; ++channel) {
    Real* density = group_density + (direct_pull ? channel * density_size : 0);
    const int c = first_channel + channel;
    Real private_pull[projected ? 1 : Layout::component_capacity] = {};
    __shared__ Real tile_pull[projected ? groups * Layout::component_capacity : 1];
    Real* pull = projected ? tile_pull + group * Layout::component_capacity : private_pull;
    volatile Real* channel_pull = edge_pull + (Quadratic ? channel * adjoint_size * groups : 0);
    if (!Quadratic || lane == 0) {
#pragma unroll
      for (int k = projected ? lane : 0; k < v.sector_count; k += projected ? Layout::tile : 1) {
        Sector sector = v.sector(k);
        Real derivative = 2 * fp[(structural_dim + sector.norm_column + c) * n + i] / (sector.size);
        for (int m = 0; m < sector.size; ++m)
          pull[sector.offset + m] =
            derivative * moments[moment_index(v, n, i, c, sector.offset + m)];
        if (sector.scalar_column >= 0)
          pull[sector.offset] += fp[(structural_dim + sector.scalar_column + c) * n + i];
      }
    }
    if (projected)
      __syncwarp(mask);
    if (lane == 0) {
#pragma unroll
      for (int t = 0; t < v.center_count; ++t) {
        Center term = v.center(t);
        for (int d = 0; d < 3; ++d)
          center_pull[d] += pull[term.output] * center[term.weight * v.channels + c] *
                            phi_i[term.primitive].derivative[d];
      }
      if (c == 0) {
        int column = structural_dim;
#pragma unroll
        for (int p = 1; p < v.primitive_count; ++p)
          if (v.primitive(p).l == 0) {
            for (int d = 0; d < 3; ++d)
              center_pull[d] += fp[column * n + i] * phi_i[v.primitive(p).offset].derivative[d];
            ++column;
          }
      }
    }
    if (Quadratic) {
      for (int k = lane; k < adjoint_size; k += Layout::tile)
        channel_pull[k * groups + group] = 0;
      __syncwarp(mask);
    }
    if (projected) {
      constexpr int spin = Layout::spin_capacity;
      constexpr int elements = Layout::harmonic_capacity * spin;
      if (CacheRows) {
        __shared__ Real center_coeff[CacheRows ? groups * elements : 1];
        Real* center_row = center_coeff + group * elements;
        for (int m = lane; m < elements; m += Layout::tile) {
          int output = m;
          Real local = 0, neighbor = 0;
          auto accumulate = [&](int index) {
            const ProjectionTerm term = transpose.terms[index];
            const Real adjoint = pull[term.input];
            local += adjoint * term.weight;
            neighbor += adjoint * transpose.neighbor_weights[index];
          };
          if (transpose.rows) {
            const ProjectionRow row = transpose.rows[m];
            output = row.output;
            for (int j = 0; j < row.count; ++j)
              accumulate(row.begin + (j * v.channels + c) * SpinProjection::width);
          } else {
            constexpr int width = SpinProjection::width;
            accumulate((m / width * v.channels + c) * width + m % width);
          }
          const int g = output / spin, a = output % spin;
          center_row[output] = local;
          density[g * row_size + a] = neighbor;
        }
        __syncwarp(mask);
        // Each lane owns one harmonic row. All rows reuse the same central spin basis.
        for (int g = lane; g < harmonics_per_channel; g += Layout::tile) {
          Real value = 0, grad[3] = {};
#pragma unroll
          for (int a = 0; a < spin; ++a) {
            const Real w = center_row[g * spin + a];
            value += w * phi_i[a].value;
            for (int k = 0; k < 3; ++k)
              grad[k] += w * phi_i[a].derivative[k];
          }
          density[g * row_size] += value;
          const float* raw = moments + size_t(n) * v.channels * v.components;
          const Real sum = raw[(size_t(i) * v.channels + c) * harmonics_per_channel + g];
          for (int k = 0; k < 3; ++k)
            center_pull[k] += sum * grad[k];
        }
      } else {
        for (int m = lane; m < elements; m += Layout::tile) {
          const ProjectionRow row = transpose.rows[m];
          Real value = 0, gradient[3] = {};
          for (int j = 0; j < row.count; ++j) {
            const ProjectionTerm term =
              transpose.terms[row.begin + (j * v.channels + c) * Layout::tile];
            const Real adjoint = pull[term.input] * term.weight;
            if (term.center_spin < 0) {
              value += adjoint;
            } else {
              const SpinValue<Real> local = phi_i[term.center_spin];
              value += adjoint * local.value;
              for (int d = 0; d < 3; ++d)
                gradient[d] += adjoint * local.derivative[d];
            }
          }
          const int g = row.output / spin, a = row.output % spin;
          density[g * row_size + a] = value;
          if (a == 0)
            for (int d = 0; d < 3; ++d)
              density[g * row_size + spin + d] = gradient[d];
        }
      }
    } else if (!Quadratic || lane == 0) {
      const int first = shared_density ? adjoint_order[lane] : 0;
      const int end = shared_density ? adjoint_order[lane + 1] : v.term_count;
      constexpr int term_unroll = Quadratic && !std::is_base_of<View, Layout>::value ? 512 : 1;
#pragma unroll term_unroll
      for (int t = first; t < end; ++t) {
        const int index = shared_density ? adjoint_order[t] : t;
        const Term term = force_term(v, index, std::integral_constant<bool, Quadratic>{});
        const Real adjoint =
          pull[term.output] * weights[term.path * v.channels + c] * term.coefficient;
        if (term.neighbor) {
          if (Quadratic)
            channel_pull[(neighbor_offset + 9 * term.geometry + term.spin - 1) * groups + group] +=
              adjoint;
          else
            density[term.geometry * row_size + term.spin] += adjoint;
        } else {
          const SpinValue<Real> spin = phi_i[term.spin];
          if (Quadratic) {
            channel_pull[term.geometry * groups + group] += adjoint * spin.value;
            for (int d = 0; d < 3; ++d)
              channel_pull[((d + 1) * harmonics_per_channel + term.geometry) * groups + group] +=
                adjoint * spin.derivative[d];
          } else {
            density[term.geometry * row_size] += adjoint * spin.value;
            for (int d = 0; d < 3; ++d)
              density[term.geometry * row_size + Layout::spin_capacity + d] +=
                adjoint * spin.derivative[d];
          }
        }
      }
    }
    if (direct_pull)
      __syncwarp(mask); // All consumers finish before the next channel reuses the component derivatives.
    if (Quadratic) {
      __syncwarp(mask);
      // phi[1..9] spans x,y,z,s2,xz,xy,Qyy,yz,Qzz-xx. Fold it to Cartesian form once.
      for (int g = lane; g < harmonics_per_channel; g += Layout::tile) {
        const int base = neighbor_offset + 9 * g;
        Real a[9];
#pragma unroll
        for (int k = 0; k < 9; ++k)
          a[k] = channel_pull[(base + k) * groups + group];
        const Real linear = v.harmonic(3).value, diagonal = v.harmonic(4).value;
        const Real difference = v.harmonic(5).value, cross = v.harmonic(6).value;
        const Real cartesian[9] = {
          linear * a[0],
          linear * a[1],
          linear * a[2],
          a[3] + diagonal * a[6] - difference * a[8],
          a[3] - 2 * diagonal * a[6],
          a[3] + diagonal * a[6] + difference * a[8],
          cross * a[5],
          cross * a[4],
          cross * a[7]};
#pragma unroll
        for (int k = 0; k < 9; ++k)
          channel_pull[(base + k) * groups + group] = cartesian[k];
      }
      __syncwarp(mask);
    }
  }
  if (shared_density)
    __syncwarp(mask);
  for (int neighbor = lane; neighbor < nn[i]; neighbor += Layout::tile) {
    int slot = neighbor * n + i, j = nl[slot], pair = type[i] * types + type[j];
    Real r[3], u[3];
    displacement(n, i, j, slot, box, positions, edges, plane, r);
    Real distance = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    if (!(distance > Real(1e-12) && distance < cutoffs[pair]))
      continue;
    for (int d = 0; d < 3; ++d)
      u[d] = r[d] / distance;
    SpinValue<Real> y[Layout::harmonic_capacity], phi_j[Layout::spin_capacity];
    spherical(v, u, y);
    read_spin(n, j, type, active, spins, s);
    if (!Quadratic)
      primitives_at(v, s, phi_j);
    double edge_force[3] = {}, edge_mforce[3] = {}, edge_virial[9] = {};
    const int rows[9] = {0, 1, 2, 0, 0, 1, 1, 2, 2};
    const int cols[9] = {0, 1, 2, 1, 2, 2, 0, 0, 1};
    if (direct_pull) {
      // Accumulate the angular and spin adjoints across channels before differentiating the bases.
      Real weights[channels] = {}, derivatives[channels] = {};
      if (ReadRadial) {
        for (int c = 0; c < channel_count; ++c) {
          const auto edge = radial_edges[radial_edge_index<Layout::tile>(n, v.channels, i, first_channel + c, neighbor)];
          weights[c] = edge.value;
          derivatives[c] = edge.derivative;
        }
      } else {
        Real fc, fcp;
        const Real inv = Real(1) / cutoffs[pair];
        find_fc_and_fcp(Real(cutoffs[pair]), inv, distance, fc, fcp);
        for_each_radial_basis(basis, inv, distance, fc, fcp, [&](int k, Real f, Real df) {
          for (int c = 0; c < channel_count; ++c) {
            const Real mix = coefficients[(pair * v.channels + first_channel + c) * (basis + 1) + k];
            weights[c] += f * mix;
            derivatives[c] += df * mix;
          }
        });
      }
      Real beta[Layout::spin_capacity] = {}, alpha[Layout::harmonic_capacity] = {};
      Real values[Layout::harmonic_capacity], radial_gradient = 0;
      for (int g = 0; g < geometry_count; ++g)
        values[g] = y[g].value;
      for (int c = 0; c < channel_count; ++c) {
        const Real* density = group_density + c * density_size;
        const Real weight = weights[c];
        Real radial_sum = 0;
        for (int g = 0; g < geometry_count; ++g) {
          const Real* row = density + g * row_size;
          const Real weighted_y = weight * values[g];
          Real pull = row[0];
#pragma unroll
          for (int a = 1; a < v.spin_components; ++a) {
            pull += row[a] * phi_j[a].value;
            beta[a] += weighted_y * row[a];
          }
          alpha[g] += weight * pull;
          radial_sum += values[g] * pull;
          if (!use_geometry)
            for (int k = 0; k < 3; ++k)
              center_pull[k] += weighted_y * row[Layout::spin_capacity + k];
        }
        radial_gradient += derivatives[c] * radial_sum;
      }
      Real angular[3] = {}, magnetic[3] = {};
      harmonic_pull<Layout::harmonic_capacity>(alpha, values, angular, HarmonicDerivativeIndices{});
      primitive_pull(v, s, phi_j, beta, magnetic);
      Real longitudinal = 0;
      for (int k = 0; k < 3; ++k)
        longitudinal += angular[k] * u[k];
      for (int k = 0; k < 3; ++k) {
        edge_force[k] = radial_gradient * u[k] + (angular[k] - longitudinal * u[k]) / distance;
        edge_mforce[k] = magnetic[k];
      }
      for (int k = 0; k < 9; ++k)
        edge_virial[k] = -r[rows[k]] * edge_force[cols[k]];
    } else {
      for (int channel = 0; channel < channel_count; ++channel) {
        const Real* density = group_density;
        const int c = first_channel + channel;
        const volatile Real* channel_pull = edge_pull + (Quadratic ? channel * adjoint_size * groups : 0);
        Real weight, derivative;
        if (ReadRadial) {
          const auto edge = radial_edges[radial_edge_index<Layout::tile>(n, v.channels, i, c, neighbor)];
          weight = edge.value;
          derivative = edge.derivative;
        } else {
          radial<stage_radial>(
            c,
            v.channels,
            basis,
            pair,
            cutoffs[pair],
            distance,
            coefficients,
            weight,
            derivative);
        }
        Real radial_pull = 0, angular_pull[3] = {}, neighbor_pull[3] = {};
        if (Quadratic) {
          Real coeff[quadratic_pull ? harmonics_per_channel : 1];
          Real values[quadratic_pull ? harmonics_per_channel : 1];
          Real spin_coeff[quadratic_pull ? 9 : 1] = {};
#pragma unroll
          for (int g = 0; g < harmonics_per_channel; ++g) {
            Real a[9];
#pragma unroll
            for (int k = 0; k < 9; ++k)
              a[k] = channel_pull[(neighbor_offset + 9 * g + k) * groups + group];
            const Real value = channel_pull[g * groups + group] +
                               s[0] * (a[0] + a[3] * s[0] + a[6] * s[1] + a[7] * s[2]) +
                               s[1] * (a[1] + a[4] * s[1] + a[8] * s[2]) +
                               s[2] * (a[2] + a[5] * s[2]);
            const Real weighted_y = weight * y[g].value;
            if (!use_geometry)
              geometry_sum[channel][g] += weighted_y;
            radial_pull += value * y[g].value;
            if (quadratic_pull) {
              coeff[g] = weight * value;
              values[g] = y[g].value;
#pragma unroll
              for (int k = 0; k < 9; ++k)
                spin_coeff[k] += weighted_y * a[k];
            } else {
              Real gradient[3];
              quadratic_gradient(a, s, gradient);
              for (int d = 0; d < 3; ++d) {
                angular_pull[d] += (weight * value) * y[g].derivative[d];
                neighbor_pull[d] += weighted_y * gradient[d];
              }
            }
          }
          if (quadratic_pull) {
            harmonic_pull<harmonics_per_channel>(coeff, values, angular_pull, HarmonicDerivativeIndices{});
            quadratic_gradient(spin_coeff, s, neighbor_pull);
          }
        } else {
          Real spin_pull[Layout::spin_capacity] = {};
          for (int g = 0; g < geometry_count; ++g) {
            Real y_pull = density[g * row_size];
            const Real weighted_y = weight * y[g].value;
            if (!use_geometry)
              geometry_sum[channel][g] += weighted_y;
#pragma unroll
            for (int a = 1; a < v.spin_components; ++a) {
              y_pull += density[g * row_size + a] * phi_j[a].value;
              spin_pull[a] += weighted_y * density[g * row_size + a];
            }
            radial_pull += y_pull * y[g].value;
            for (int d = 0; d < 3; ++d)
              angular_pull[d] += (weight * y_pull) * y[g].derivative[d];
          }
          for (int a = 1; a < v.spin_components; ++a)
            for (int d = 0; d < 3; ++d)
              neighbor_pull[d] += spin_pull[a] * phi_j[a].derivative[d];
        }
        Real longitudinal = 0;
        for (int d = 0; d < 3; ++d)
          longitudinal += angular_pull[d] * u[d];
        Real grad[3];
        for (int d = 0; d < 3; ++d) {
          grad[d] =
            derivative * radial_pull * u[d] + (angular_pull[d] - longitudinal * u[d]) / distance;
          edge_force[d] += static_cast<double>(grad[d]);
          edge_mforce[d] += static_cast<double>(neighbor_pull[d]);
        }
#pragma unroll
        for (int k = 0; k < 9; ++k)
          edge_virial[k] -= static_cast<double>(r[rows[k]] * grad[cols[k]]);
      }
    }
    for (int d = 0; d < 3; ++d) {
      if (CenterVirial || Layout::tile > 1)
        center_force[d] += edge_force[d];
      else
        atomicAdd(force + d * n + i, edge_force[d]);
      atomicAdd(force + d * n + j, -edge_force[d]);
      if (active[type[j]])
        atomicAdd(mforce + d * n + j, -edge_mforce[d]);
    }
#pragma unroll
    for (int k = 0; k < 9; ++k) {
      if ((virial_mode & 1) != 0)
        center_virial[k] += edge_virial[k];
      if constexpr (NeighborOutputs) {
        if (virial_mode == 2)
          atomicAdd(virial + k*n+j, edge_virial[k]);
        if (virial_mode == 3)
          atomicAdd(virial_float + k*n+j, static_cast<float>(edge_virial[k]));
      }
    }
    if constexpr (NeighborOutputs) {
      if (transfer && active[type[j]])
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b)
            atomicAdd(transfer + (3*a+b)*n+j, static_cast<float>(-r[a]*edge_mforce[b]));
    }
  }
  for (int channel = 0; channel < (direct_pull || use_geometry ? 0 : channel_count); ++channel) {
    const Real* density = group_density;
    const volatile Real* channel_pull = edge_pull + (Quadratic ? channel * adjoint_size * groups : 0);
    for (int g = 0; g < geometry_count; ++g)
      for (int d = 0; d < 3; ++d)
        center_pull[d] +=
          geometry_sum[channel][g] *
          (Quadratic ? channel_pull[((d + 1) * harmonics_per_channel + g) * groups + group]
                     : density[g * row_size + Layout::spin_capacity + d]);
  }
#pragma unroll
  for (int d = 0; d < 3; ++d) {
#pragma unroll
    for (int offset = Layout::tile / 2; offset > 0; offset /= 2) {
      center_pull[d] += __shfl_down_sync(mask, center_pull[d], offset, Layout::tile);
      center_force[d] += __shfl_down_sync(mask, center_force[d], offset, Layout::tile);
    }
  }
  // MD assigns each edge virial to its center. Reduce before the atomic write;
  // the training path retains its neighbor-owned per-atom convention.
  if ((virial_mode & 1) != 0) {
#pragma unroll
    for (int k = 0; k < 9; ++k) {
#pragma unroll
      for (int offset = Layout::tile / 2; offset > 0; offset /= 2)
        center_virial[k] += __shfl_down_sync(mask, center_virial[k], offset, Layout::tile);
      if (lane == 0)
        atomicAdd(virial + k * n + i, center_virial[k]);
    }
  }
  if (lane != 0)
    return;
  if (CenterVirial || Layout::tile > 1)
    for (int d = 0; d < 3; ++d)
      atomicAdd(force + d * n + i, center_force[d]);
  if (active[type[i]])
    for (int d = 0; d < 3; ++d)
      atomicAdd(mforce + d * n + i, -static_cast<double>(center_pull[d]));
}
} // namespace nep_adapters::spin_density::md
