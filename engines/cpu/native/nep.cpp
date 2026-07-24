/*
    Copyright 2022 Zheyong Fan, Junjie Wang, Eric Lindgren
    This file is part of NEP_CPU.
    NEP_CPU is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    NEP_CPU is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with NEP_CPU.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
A CPU implementation of the neuroevolution potential (NEP)
Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#include "nep.h"
#include "dftd3para.h"
#include "nep_utilities.h"
#include "neighbor_nep.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(NEP_ADAPTERS_CPU_USE_CBLAS)
#if defined(NEP_ADAPTERS_CPU_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace
{

using NepPhaseClock = std::chrono::steady_clock;

struct NepPhaseTotals {
  long long calls = 0;
  long long active_atoms = 0;
  long long centers = 0;
  long long neighbors = 0;
  double setup = 0.0;
  double neighbor = 0.0;
  double cache = 0.0;
  double table = 0.0;
  double descriptor = 0.0;
  double descriptor_core = 0.0;
  double ann = 0.0;
  double scratch = 0.0;
  double radial = 0.0;
  double angular = 0.0;
  double reduce = 0.0;
  double zbl = 0.0;
  double spin_setup = 0.0;
  double spin_edges = 0.0;
  double spin_unpack = 0.0;
  double spin_merge = 0.0;
  double spin_contract = 0.0;
  double spin_chiral = 0.0;
  double spin_copy = 0.0;
  double spin_ann = 0.0;
  double spin_gradient = 0.0;
  double spin_gradient_nonchiral = 0.0;
  double spin_gradient_chiral = 0.0;
};

struct NepPhaseTimerState {
  NepPhaseTotals batch;
  NepPhaseTotals lammps;

  ~NepPhaseTimerState()
  {
    print("batch", batch);
    print("lammps", lammps);
  }

  static void print(const char* mode, const NepPhaseTotals& totals)
  {
    if (totals.calls == 0) {
      return;
    }
    const double total =
      totals.setup + totals.neighbor + totals.cache + totals.table + totals.descriptor +
      totals.scratch + totals.radial + totals.angular + totals.reduce + totals.zbl +
      totals.spin_setup + totals.spin_edges + totals.spin_unpack + totals.spin_merge +
      totals.spin_contract + totals.spin_chiral + totals.spin_copy + totals.spin_ann +
      totals.spin_gradient;
    std::cerr << "{\"phase_timer\":\"nep_cpu\","
              << "\"mode\":\"" << mode << "\","
              << "\"calls\":" << totals.calls << ','
              << "\"active_atoms\":" << totals.active_atoms << ','
              << "\"centers\":" << totals.centers << ','
              << "\"neighbors\":" << totals.neighbors << ','
              << "\"total_seconds\":" << total << ','
              << "\"phase_seconds\":{"
              << "\"setup\":" << totals.setup << ','
              << "\"neighbor\":" << totals.neighbor << ','
              << "\"cache\":" << totals.cache << ','
              << "\"table\":" << totals.table << ','
              << "\"descriptor\":" << totals.descriptor << ','
              << "\"descriptor_core\":" << totals.descriptor_core << ','
              << "\"ann\":" << totals.ann << ','
              << "\"scratch\":" << totals.scratch << ','
              << "\"radial\":" << totals.radial << ','
              << "\"angular\":" << totals.angular << ','
              << "\"reduce\":" << totals.reduce << ','
              << "\"zbl\":" << totals.zbl << ','
              << "\"spin_setup\":" << totals.spin_setup << ','
              << "\"spin_edges\":" << totals.spin_edges << ','
              << "\"spin_unpack\":" << totals.spin_unpack << ','
              << "\"spin_merge\":" << totals.spin_merge << ','
              << "\"spin_contract\":" << totals.spin_contract << ','
              << "\"spin_chiral\":" << totals.spin_chiral << ','
              << "\"spin_copy\":" << totals.spin_copy << ','
              << "\"spin_ann\":" << totals.spin_ann << ','
              << "\"spin_gradient\":" << totals.spin_gradient << ','
              << "\"spin_gradient_nonchiral\":" << totals.spin_gradient_nonchiral << ','
              << "\"spin_gradient_chiral\":" << totals.spin_gradient_chiral
              << "}}\n";
  }
};

NepPhaseTimerState& nep_phase_timer_state()
{
  static NepPhaseTimerState state;
  return state;
}

bool nep_phase_timer_enabled()
{
  const char* value = std::getenv("NEP_CPU_PHASE_TIMER");
  return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
}

int spin_openmp_threads()
{
#if defined(_OPENMP)
  const int cap = 16;
  return std::max(1, std::min(omp_get_max_threads(), cap));
#else
  return 1;
#endif
}

double nep_phase_elapsed(NepPhaseClock::time_point& mark)
{
  const auto now = NepPhaseClock::now();
  const double seconds = std::chrono::duration<double>(now - mark).count();
  mark = now;
  return seconds;
}

void cache_type_pair_constants(NEP::ParaMB& paramb, const NEP::ZBL& zbl)
{
  const std::size_t num_types = paramb.num_types;
  const std::size_t num_types_sq = paramb.num_types_sq;
  paramb.rc_radial_pair.resize(num_types_sq);
  paramb.rcinv_radial_pair.resize(num_types_sq);
  paramb.rc_angular_pair.resize(num_types_sq);
  paramb.rcinv_angular_pair.resize(num_types_sq);
  paramb.zbl_zizj_pair.resize(num_types_sq);
  paramb.zbl_a_inv_pair.resize(num_types_sq);
  paramb.zbl_rc_inner_pair.resize(num_types_sq);
  paramb.zbl_rc_outer_pair.resize(num_types_sq);

  std::vector<double> z_pow(num_types);
  for (std::size_t t = 0; t < num_types; ++t) {
    z_pow[t] = pow(double(paramb.atomic_numbers[t] + 1), 0.23);
  }

  for (std::size_t t1 = 0; t1 < num_types; ++t1) {
    const int zi = paramb.atomic_numbers[t1] + 1;
    for (std::size_t t2 = 0; t2 < num_types; ++t2) {
      const std::size_t t12 = t1 * num_types + t2;
      const int zj = paramb.atomic_numbers[t2] + 1;

      const double rc_radial = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5;
      const double rc_angular = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5;
      paramb.rc_radial_pair[t12] = rc_radial;
      paramb.rcinv_radial_pair[t12] = 1.0 / rc_radial;
      paramb.rc_angular_pair[t12] = rc_angular;
      paramb.rcinv_angular_pair[t12] = 1.0 / rc_angular;

      paramb.zbl_zizj_pair[t12] = K_C_SP * zi * zj;
      paramb.zbl_a_inv_pair[t12] = (z_pow[t1] + z_pow[t2]) * 2.134563;

      double rc_inner = zbl.rc_inner;
      double rc_outer = zbl.rc_outer;
      if (!zbl.flexibled && paramb.use_typewise_cutoff_zbl) {
        rc_outer = std::min(
          (COVALENT_RADIUS[zi - 1] + COVALENT_RADIUS[zj - 1]) * paramb.typewise_cutoff_zbl_factor,
          rc_outer);
        rc_inner = 0.0;
      }
      paramb.zbl_rc_inner_pair[t12] = rc_inner;
      paramb.zbl_rc_outer_pair[t12] = rc_outer;
    }
  }
}

void cache_descriptor_coefficients(const NEP::ParaMB& paramb, NEP::ANN& annmb)
{
  const std::size_t radial_basis = static_cast<std::size_t>(paramb.basis_size_radial + 1);
  const std::size_t radial_n = static_cast<std::size_t>(paramb.n_max_radial + 1);
  const std::size_t angular_basis = static_cast<std::size_t>(paramb.basis_size_angular + 1);
  const std::size_t angular_n = static_cast<std::size_t>(paramb.n_max_angular + 1);

  annmb.c_radial_pair.resize(paramb.num_types_sq * radial_n * radial_basis);
  annmb.c_angular_pair.resize(paramb.num_types_sq * angular_n * angular_basis);

  for (std::size_t t12 = 0; t12 < paramb.num_types_sq; ++t12) {
    double* radial_pair = annmb.c_radial_pair.data() + t12 * radial_n * radial_basis;
    for (std::size_t n = 0; n < radial_n; ++n) {
      for (std::size_t k = 0; k < radial_basis; ++k) {
        radial_pair[n * radial_basis + k] =
          annmb.c[(n * radial_basis + k) * paramb.num_types_sq + t12];
      }
    }

    double* angular_pair = annmb.c_angular_pair.data() + t12 * angular_n * angular_basis;
    for (std::size_t n = 0; n < angular_n; ++n) {
      for (std::size_t k = 0; k < angular_basis; ++k) {
        angular_pair[n * angular_basis + k] =
          annmb.c[paramb.num_c_radial + (n * angular_basis + k) * paramb.num_types_sq + t12];
      }
    }
  }
}

#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
std::size_t table_pair_slot(const NEP::ParaMB& paramb, const int t12)
{
  if (t12 < 0 || static_cast<std::size_t>(t12) >= paramb.table_pair_to_slot.size()) {
    std::cout << "Invalid table type-pair index." << std::endl;
    exit(1);
  }
  const int slot = paramb.table_pair_to_slot[static_cast<std::size_t>(t12)];
  if (slot < 0) {
    std::cout << "Missing tabulated radial function for an active type pair." << std::endl;
    exit(1);
  }
  return static_cast<std::size_t>(slot);
}

std::size_t table_lookup_index(
  const NEP::ParaMB& paramb,
  const int table_index,
  const int t12,
  const int n_count,
  const int n)
{
  return (static_cast<std::size_t>(table_index) * paramb.table_pair_count +
          table_pair_slot(paramb, t12)) *
    n_count +
    n;
}

std::size_t table_pair_slot_unchecked(const NEP::ParaMB& paramb, const int t12)
{
  return static_cast<std::size_t>(paramb.table_pair_to_slot[static_cast<std::size_t>(t12)]);
}
#endif

#if defined(__AVX2__)
#define NEP_CPU_HAS_LMAX4_EDGE_SIMD 1
using Lmax4EdgeVec = __m256d;
constexpr int lmax4_num_terms = 24;
constexpr int lmax4_edge_simd_width = 4;

inline Lmax4EdgeVec vec_set1_f64(const double value)
{
  return _mm256_set1_pd(value);
}

inline Lmax4EdgeVec vec_load_f64(const double* values)
{
  return _mm256_loadu_pd(values);
}

inline Lmax4EdgeVec vec_add_f64(const Lmax4EdgeVec a, const Lmax4EdgeVec b)
{
  return _mm256_add_pd(a, b);
}

inline Lmax4EdgeVec vec_sub_f64(const Lmax4EdgeVec a, const Lmax4EdgeVec b)
{
  return _mm256_sub_pd(a, b);
}

inline Lmax4EdgeVec vec_mul_f64(const Lmax4EdgeVec a, const Lmax4EdgeVec b)
{
  return _mm256_mul_pd(a, b);
}

inline Lmax4EdgeVec vec_div_f64(const Lmax4EdgeVec a, const Lmax4EdgeVec b)
{
  return _mm256_div_pd(a, b);
}

inline double horizontal_add_f64(const Lmax4EdgeVec v)
{
  double values[4];
  _mm256_storeu_pd(values, v);
  return values[0] + values[1] + values[2] + values[3];
}
#elif defined(__AVX512F__)
#define NEP_CPU_HAS_LMAX4_EDGE_SIMD 1
using Lmax4EdgeVec = __m512d;
constexpr int lmax4_num_terms = 24;
constexpr int lmax4_edge_simd_width = 8;

inline Lmax4EdgeVec vec_set1_f64(const double value)
{
  return _mm512_set1_pd(value);
}

inline Lmax4EdgeVec vec_load_f64(const double* values)
{
  return _mm512_loadu_pd(values);
}

inline Lmax4EdgeVec vec_add_f64(const Lmax4EdgeVec a, const Lmax4EdgeVec b)
{
  return _mm512_add_pd(a, b);
}

inline Lmax4EdgeVec vec_sub_f64(const Lmax4EdgeVec a, const Lmax4EdgeVec b)
{
  return _mm512_sub_pd(a, b);
}

inline Lmax4EdgeVec vec_mul_f64(const Lmax4EdgeVec a, const Lmax4EdgeVec b)
{
  return _mm512_mul_pd(a, b);
}

inline Lmax4EdgeVec vec_div_f64(const Lmax4EdgeVec a, const Lmax4EdgeVec b)
{
  return _mm512_div_pd(a, b);
}

inline double horizontal_add_f64(const Lmax4EdgeVec v)
{
  return _mm512_reduce_add_pd(v);
}
#endif

#if defined(NEP_CPU_HAS_LMAX4_EDGE_SIMD)
void compute_s_lmax4_edge_terms(
  const double* x12,
  const double* y12,
  const double* z12,
  const double* d12,
  Lmax4EdgeVec* s_terms)
{
  const Lmax4EdgeVec d = vec_load_f64(d12);
  const Lmax4EdgeVec inv_d = vec_div_f64(vec_set1_f64(1.0), d);
  const Lmax4EdgeVec x = vec_mul_f64(vec_load_f64(x12), inv_d);
  const Lmax4EdgeVec y = vec_mul_f64(vec_load_f64(y12), inv_d);
  const Lmax4EdgeVec z = vec_mul_f64(vec_load_f64(z12), inv_d);

  const Lmax4EdgeVec x2 = vec_mul_f64(x, x);
  const Lmax4EdgeVec y2 = vec_mul_f64(y, y);
  const Lmax4EdgeVec z2 = vec_mul_f64(z, z);
  const Lmax4EdgeVec z3 = vec_mul_f64(z2, z);
  const Lmax4EdgeVec z4 = vec_mul_f64(z2, z2);
  const Lmax4EdgeVec real2 = vec_sub_f64(x2, y2);
  const Lmax4EdgeVec imag2 = vec_mul_f64(vec_set1_f64(2.0), vec_mul_f64(x, y));
  const Lmax4EdgeVec real3 = vec_sub_f64(vec_mul_f64(x, real2), vec_mul_f64(y, imag2));
  const Lmax4EdgeVec imag3 = vec_add_f64(vec_mul_f64(x, imag2), vec_mul_f64(y, real2));
  const Lmax4EdgeVec real4 = vec_sub_f64(vec_mul_f64(x, real3), vec_mul_f64(y, imag3));
  const Lmax4EdgeVec imag4 = vec_add_f64(vec_mul_f64(x, imag3), vec_mul_f64(y, real3));

  s_terms[0] = z;
  s_terms[1] = x;
  s_terms[2] = y;

  s_terms[3] = vec_add_f64(vec_set1_f64(-1.0), vec_mul_f64(vec_set1_f64(3.0), z2));
  s_terms[4] = vec_mul_f64(z, x);
  s_terms[5] = vec_mul_f64(z, y);
  s_terms[6] = real2;
  s_terms[7] = imag2;

  const Lmax4EdgeVec l3_z0 =
    vec_add_f64(vec_mul_f64(vec_set1_f64(-3.0), z), vec_mul_f64(vec_set1_f64(5.0), z3));
  const Lmax4EdgeVec l3_z1 =
    vec_add_f64(vec_set1_f64(-1.0), vec_mul_f64(vec_set1_f64(5.0), z2));
  s_terms[8] = l3_z0;
  s_terms[9] = vec_mul_f64(l3_z1, x);
  s_terms[10] = vec_mul_f64(l3_z1, y);
  s_terms[11] = vec_mul_f64(z, real2);
  s_terms[12] = vec_mul_f64(z, imag2);
  s_terms[13] = real3;
  s_terms[14] = imag3;

  const Lmax4EdgeVec l4_z0 = vec_add_f64(
    vec_add_f64(vec_set1_f64(3.0), vec_mul_f64(vec_set1_f64(-30.0), z2)),
    vec_mul_f64(vec_set1_f64(35.0), z4));
  const Lmax4EdgeVec l4_z1 = vec_add_f64(
    vec_mul_f64(vec_set1_f64(-3.0), z), vec_mul_f64(vec_set1_f64(7.0), z3));
  const Lmax4EdgeVec l4_z2 =
    vec_add_f64(vec_set1_f64(-1.0), vec_mul_f64(vec_set1_f64(7.0), z2));
  s_terms[15] = l4_z0;
  s_terms[16] = vec_mul_f64(l4_z1, x);
  s_terms[17] = vec_mul_f64(l4_z1, y);
  s_terms[18] = vec_mul_f64(l4_z2, real2);
  s_terms[19] = vec_mul_f64(l4_z2, imag2);
  s_terms[20] = vec_mul_f64(z, real3);
  s_terms[21] = vec_mul_f64(z, imag3);
  s_terms[22] = real4;
  s_terms[23] = imag4;
}

void accumulate_s_lmax4_edge_terms(
  const Lmax4EdgeVec* s_terms,
  const double* gn12,
  double* s)
{
  const Lmax4EdgeVec fn = vec_load_f64(gn12);
  for (int abc = 0; abc < lmax4_num_terms; ++abc) {
    s[abc] += horizontal_add_f64(vec_mul_f64(fn, s_terms[abc]));
  }
}

void accumulate_s_lmax4_edge_terms_vec(
  const Lmax4EdgeVec* s_terms,
  const double* gn12,
  Lmax4EdgeVec* s)
{
  const Lmax4EdgeVec fn = vec_load_f64(gn12);
  for (int abc = 0; abc < lmax4_num_terms; ++abc) {
    s[abc] = vec_add_f64(s[abc], vec_mul_f64(fn, s_terms[abc]));
  }
}

void accumulate_s_lmax4_edges(
  const double* x12,
  const double* y12,
  const double* z12,
  const double* d12,
  const double* gn12,
  double* s)
{
  Lmax4EdgeVec s_terms[lmax4_num_terms];
  compute_s_lmax4_edge_terms(x12, y12, z12, d12, s_terms);
  accumulate_s_lmax4_edge_terms(s_terms, gn12, s);
}
#endif

#if defined(NEP_ADAPTERS_CPU_USE_CBLAS)
void apply_ann_one_layer_batched_by_type(
  const NEP::ParaMB& paramb,
  const NEP::ANN& annmb,
  const int N,
  const int* g_type,
  double* q_and_Fp,
  double* g_potential,
  std::vector<double>& q_group,
  std::vector<double>& hidden,
  std::vector<double>& coeff,
  std::vector<double>& fp_group)
{
  const int dim = annmb.dim;
  const int num_neurons = annmb.num_neurons1;
  std::vector<std::vector<int>> atoms_by_type(paramb.num_types);
  for (int n = 0; n < N; ++n) {
    atoms_by_type[g_type[n]].push_back(n);
  }

  for (std::size_t t = 0; t < paramb.num_types; ++t) {
    const std::vector<int>& atoms = atoms_by_type[t];
    const int atom_count = static_cast<int>(atoms.size());
    if (atom_count == 0) {
      continue;
    }

    if (atom_count < 16) {
      double q[MAX_DIM];
      double Fp[MAX_DIM];
      double latent_space[MAX_NEURON];
      for (int atom : atoms) {
        for (int d = 0; d < dim; ++d) {
          q[d] = q_and_Fp[static_cast<std::size_t>(atom) * dim + d];
          Fp[d] = 0.0;
        }
        for (int n = 0; n < num_neurons; ++n) {
          latent_space[n] = 0.0;
        }
        double F = 0.0;
        apply_ann_one_layer(
          dim, num_neurons, annmb.w0[t], annmb.b0[t], annmb.w1[t], annmb.b1, q, F, Fp,
          latent_space, false, nullptr);
        g_potential[atom] += F;
        for (int d = 0; d < dim; ++d) {
          q_and_Fp[static_cast<std::size_t>(atom) * dim + d] = Fp[d] * paramb.q_scaler[d];
        }
      }
      continue;
    }

    q_group.resize(static_cast<std::size_t>(atom_count) * dim);
    for (int a = 0; a < atom_count; ++a) {
      const int atom = atoms[a];
      const double* q_src = q_and_Fp + static_cast<std::size_t>(atom) * dim;
      double* q_dst = q_group.data() + static_cast<std::size_t>(a) * dim;
      for (int d = 0; d < dim; ++d) {
        q_dst[d] = q_src[d];
      }
    }

    hidden.resize(static_cast<std::size_t>(num_neurons) * atom_count);
    cblas_dgemm(
      CblasRowMajor, CblasNoTrans, CblasTrans, num_neurons, atom_count, dim, 1.0,
      annmb.w0[t], dim, q_group.data(), dim, 0.0, hidden.data(), atom_count);

    coeff.resize(static_cast<std::size_t>(num_neurons) * atom_count);
    for (int a = 0; a < atom_count; ++a) {
      g_potential[atoms[a]] -= annmb.b1[0];
    }

    for (int n = 0; n < num_neurons; ++n) {
      const double b0 = annmb.b0[t][n];
      const double w1 = annmb.w1[t][n];
      double* hidden_row = hidden.data() + static_cast<std::size_t>(n) * atom_count;
      double* coeff_row = coeff.data() + static_cast<std::size_t>(n) * atom_count;
      for (int a = 0; a < atom_count; ++a) {
        const double x1 = tanh(hidden_row[a] - b0);
        g_potential[atoms[a]] += w1 * x1;
        coeff_row[a] = w1 * (1.0 - x1 * x1);
      }
    }

    fp_group.resize(static_cast<std::size_t>(atom_count) * dim);
    cblas_dgemm(
      CblasRowMajor, CblasTrans, CblasNoTrans, atom_count, dim, num_neurons, 1.0,
      coeff.data(), atom_count, annmb.w0[t], dim, 0.0, fp_group.data(), dim);

    for (int a = 0; a < atom_count; ++a) {
      const int atom = atoms[a];
      const double* fp_src = fp_group.data() + static_cast<std::size_t>(a) * dim;
      double* fp_dst = q_and_Fp + static_cast<std::size_t>(atom) * dim;
      for (int d = 0; d < dim; ++d) {
        fp_dst[d] = fp_src[d] * paramb.q_scaler[d];
      }
    }
  }
}

void apply_ann_one_layer_batched_for_lammps(
  const NEP::ParaMB& paramb,
  const NEP::ANN& annmb,
  const int nlocal,
  const int N,
  int* g_ilist,
  int* g_type,
  int* type_map,
  double* q_and_Fp,
  double& g_total_potential,
  double* g_potential,
  std::vector<double>& q_group,
  std::vector<double>& hidden,
  std::vector<double>& coeff,
  std::vector<double>& fp_group)
{
  const int dim = annmb.dim;
  const int num_neurons = annmb.num_neurons1;
  std::vector<std::vector<int>> atoms_by_type(paramb.num_types);
  for (int ii = 0; ii < N; ++ii) {
    const int n1 = g_ilist[ii];
    atoms_by_type[type_map[g_type[n1]]].push_back(n1);
  }

  double total_potential = 0.0;
  for (std::size_t t = 0; t < atoms_by_type.size(); ++t) {
    const std::vector<int>& atoms = atoms_by_type[t];
    const int atom_count = static_cast<int>(atoms.size());
    if (atom_count == 0) {
      continue;
    }

    if (atom_count < 16) {
      double q[MAX_DIM];
      double Fp[MAX_DIM];
      double latent_space[MAX_NEURON];
      for (int atom : atoms) {
        for (int d = 0; d < dim; ++d) {
          q[d] = q_and_Fp[static_cast<std::size_t>(atom) * dim + d];
          Fp[d] = 0.0;
        }
        for (int n = 0; n < num_neurons; ++n) {
          latent_space[n] = 0.0;
        }
        double F = 0.0;
        apply_ann_one_layer(
          dim, num_neurons, annmb.w0[t], annmb.b0[t], annmb.w1[t], annmb.b1, q, F, Fp,
          latent_space, false, nullptr);
        total_potential += F;
        if (g_potential) {
          g_potential[atom] += F;
        }
        for (int d = 0; d < dim; ++d) {
          q_and_Fp[static_cast<std::size_t>(atom) * dim + d] = Fp[d] * paramb.q_scaler[d];
        }
      }
      continue;
    }

    q_group.resize(static_cast<std::size_t>(atom_count) * dim);
    for (int a = 0; a < atom_count; ++a) {
      const int atom = atoms[a];
      const double* q_src = q_and_Fp + static_cast<std::size_t>(atom) * dim;
      double* q_dst = q_group.data() + static_cast<std::size_t>(a) * dim;
      for (int d = 0; d < dim; ++d) {
        q_dst[d] = q_src[d];
      }
    }

    hidden.resize(static_cast<std::size_t>(num_neurons) * atom_count);
    cblas_dgemm(
      CblasRowMajor, CblasNoTrans, CblasTrans, num_neurons, atom_count, dim, 1.0,
      annmb.w0[t], dim, q_group.data(), dim, 0.0, hidden.data(), atom_count);

    coeff.resize(static_cast<std::size_t>(num_neurons) * atom_count);
    total_potential -= annmb.b1[0] * atom_count;
    if (g_potential) {
      for (int atom : atoms) {
        g_potential[atom] -= annmb.b1[0];
      }
    }

    for (int n = 0; n < num_neurons; ++n) {
      const double b0 = annmb.b0[t][n];
      const double w1 = annmb.w1[t][n];
      double* hidden_row = hidden.data() + static_cast<std::size_t>(n) * atom_count;
      double* coeff_row = coeff.data() + static_cast<std::size_t>(n) * atom_count;
      for (int a = 0; a < atom_count; ++a) {
        const double x1 = tanh(hidden_row[a] - b0);
        const double energy = w1 * x1;
        total_potential += energy;
        if (g_potential) {
          g_potential[atoms[a]] += energy;
        }
        coeff_row[a] = w1 * (1.0 - x1 * x1);
      }
    }

    fp_group.resize(static_cast<std::size_t>(atom_count) * dim);
    cblas_dgemm(
      CblasRowMajor, CblasTrans, CblasNoTrans, atom_count, dim, num_neurons, 1.0,
      coeff.data(), atom_count, annmb.w0[t], dim, 0.0, fp_group.data(), dim);

    for (int a = 0; a < atom_count; ++a) {
      const int atom = atoms[a];
      const double* fp_src = fp_group.data() + static_cast<std::size_t>(a) * dim;
      double* fp_dst = q_and_Fp + static_cast<std::size_t>(atom) * dim;
      for (int d = 0; d < dim; ++d) {
        fp_dst[d] = fp_src[d] * paramb.q_scaler[d];
      }
    }
  }
  g_total_potential += total_potential;
}
#endif

void find_descriptor_small_box(
  const bool calculating_potential,
  const bool calculating_descriptor,
  const bool calculating_latent_space,
  const bool calculating_polarizability,
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  const int N,
  const int* g_NN_radial,
  const int* g_NL_radial,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const double* g_x12_radial,
  const double* g_y12_radial,
  const double* g_z12_radial,
  const double* g_x12_angular,
  const double* g_y12_angular,
  const double* g_z12_angular,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  const double* g_gn_radial,
  const double* g_gnp_radial,
  const double* g_gn_angular,
  const double* g_gnp_angular,
#endif
  double* g_Fp,
  double* g_sum_fxyz,
  double* g_potential,
  double* g_descriptor,
  double* g_latent_space,
  double* g_virial,
  bool calculating_B_projection,
  double* g_B_projection,
  std::vector<double>& ann_q_group_workspace,
  std::vector<double>& ann_hidden_workspace,
  std::vector<double>& ann_coeff_workspace,
  std::vector<double>& ann_fp_group_workspace,
  const int* radial_edge_offsets = nullptr,
  double* radial_gnp_cache = nullptr,
  const int* angular_edge_offsets = nullptr,
  double* angular_gn_cache = nullptr,
  double* angular_gnp_cache = nullptr)
{
#if defined(NEP_ADAPTERS_CPU_USE_CBLAS)
  const bool use_batched_ann =
    calculating_potential && !calculating_latent_space && !calculating_polarizability &&
    !calculating_B_projection && paramb.version != 5;
#else
  const bool use_batched_ann = false;
#endif

#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int n1 = 0; n1 < N; ++n1) {
    int t1 = g_type[n1];
    std::array<double, MAX_DIM> q = {};

    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_radial[index];
      int t2 = g_type[n2];
      int t12 = t1 * paramb.num_types + t2;
      double r12[3] = {g_x12_radial[index], g_y12_radial[index], g_z12_radial[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);

#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
      int index_left, index_right;
      double weight_left, weight_right;
      double rcinv = paramb.rcinv_radial_pair[t12];
      find_index_and_weight(
        d12 * rcinv, index_left, index_right, weight_left, weight_right);
      double table_step = paramb.rc_radial_pair[t12] * table_resolution;
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        std::size_t index_left_all =
          table_lookup_index(paramb, index_left, t12, paramb.n_max_radial + 1, n);
        std::size_t index_right_all =
          table_lookup_index(paramb, index_right, t12, paramb.n_max_radial + 1, n);
        q[n] += interpolate_table_value(
          g_gn_radial, g_gnp_radial, index_left_all, index_right_all, weight_right, table_step);
      }
#else
      double rc = paramb.rc_radial_pair[t12];
      double rcinv = paramb.rcinv_radial_pair[t12];
      double fc12;
      double fcp12 = 0.0;
      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      if (radial_gnp_cache) {
        find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
        find_fn_and_fnp(
          paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      } else {
        find_fc(rc, rcinv, d12, fc12);
        find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      }
      const double* c_pair =
        annmb.c_radial_pair.data() + static_cast<std::size_t>(t12) *
                                      (paramb.n_max_radial + 1) *
                                      (paramb.basis_size_radial + 1);
      const std::size_t cache_base = radial_gnp_cache
        ? static_cast<std::size_t>(radial_edge_offsets[n1] + i1) *
            (paramb.n_max_radial + 1)
        : 0;
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gn12 = 0.0;
        double gnp12 = 0.0;
        const double* c_n = c_pair + n * (paramb.basis_size_radial + 1);
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          gn12 += fn12[k] * c_n[k];
          if (radial_gnp_cache) {
            gnp12 += fnp12[k] * c_n[k];
          }
        }
        q[n] += gn12;
        if (radial_gnp_cache) {
          radial_gnp_cache[cache_base + n] = gnp12;
        }
      }
#endif
    }

#if defined(USE_TABLE_FOR_RADIAL_FUNCTIONS) && defined(NEP_CPU_HAS_LMAX4_EDGE_SIMD)
    if (paramb.L_max == 4) {
      double s_all[MAX_NUM_N * NUM_OF_ABC];
      std::fill(
        s_all, s_all + static_cast<std::size_t>(paramb.n_max_angular + 1) * NUM_OF_ABC, 0.0);
      Lmax4EdgeVec s_vec[MAX_NUM_N * lmax4_num_terms];
      std::fill(
        s_vec, s_vec + static_cast<std::size_t>(paramb.n_max_angular + 1) * lmax4_num_terms,
        vec_set1_f64(0.0));
      int i1 = 0;
      for (; i1 + lmax4_edge_simd_width - 1 < g_NN_angular[n1];
           i1 += lmax4_edge_simd_width) {
        double x12_block[lmax4_edge_simd_width];
        double y12_block[lmax4_edge_simd_width];
        double z12_block[lmax4_edge_simd_width];
        double d12_block[lmax4_edge_simd_width];
        int t12_block[lmax4_edge_simd_width];
        int index_left_block[lmax4_edge_simd_width];
        int index_right_block[lmax4_edge_simd_width];
        double weight_right_block[lmax4_edge_simd_width];
        double table_step_block[lmax4_edge_simd_width];
        for (int lane = 0; lane < lmax4_edge_simd_width; ++lane) {
          const int index = (i1 + lane) * N + n1;
          const int n2 = g_NL_angular[index];
          const int t2 = g_type[n2];
          const int t12 = t1 * paramb.num_types + t2;
          t12_block[lane] = t12;
          x12_block[lane] = g_x12_angular[index];
          y12_block[lane] = g_y12_angular[index];
          z12_block[lane] = g_z12_angular[index];
          const double d12 = sqrt(
            x12_block[lane] * x12_block[lane] + y12_block[lane] * y12_block[lane] +
            z12_block[lane] * z12_block[lane]);
          d12_block[lane] = d12;
          double weight_left;
          find_index_and_weight(
            d12 * paramb.rcinv_angular_pair[t12], index_left_block[lane],
            index_right_block[lane], weight_left, weight_right_block[lane]);
          table_step_block[lane] = paramb.rc_angular_pair[t12] * table_resolution;
        }
        Lmax4EdgeVec s_terms[lmax4_num_terms];
        compute_s_lmax4_edge_terms(x12_block, y12_block, z12_block, d12_block, s_terms);
        for (int n = 0; n <= paramb.n_max_angular; ++n) {
          double gn12_block[lmax4_edge_simd_width];
          for (int lane = 0; lane < lmax4_edge_simd_width; ++lane) {
            std::size_t index_left_all =
              table_lookup_index(paramb, index_left_block[lane], t12_block[lane], paramb.n_max_angular + 1, n);
            std::size_t index_right_all =
              table_lookup_index(paramb, index_right_block[lane], t12_block[lane], paramb.n_max_angular + 1, n);
            gn12_block[lane] = interpolate_table_value(
              g_gn_angular, g_gnp_angular, index_left_all, index_right_all,
              weight_right_block[lane], table_step_block[lane]);
          }
          accumulate_s_lmax4_edge_terms_vec(s_terms, gn12_block, s_vec + n * lmax4_num_terms);
        }
      }
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        for (int abc = 0; abc < lmax4_num_terms; ++abc) {
          s_all[n * NUM_OF_ABC + abc] += horizontal_add_f64(s_vec[n * lmax4_num_terms + abc]);
        }
      }
      for (; i1 < g_NN_angular[n1]; ++i1) {
        int index = i1 * N + n1;
        int n2 = g_NL_angular[index];
        int t2 = g_type[n2];
        int t12 = t1 * paramb.num_types + t2;
        double r12[3] = {g_x12_angular[index], g_y12_angular[index], g_z12_angular[index]};
        double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
        int index_left, index_right;
        double weight_left, weight_right;
        double rcinv = paramb.rcinv_angular_pair[t12];
        find_index_and_weight(
          d12 * rcinv, index_left, index_right, weight_left, weight_right);
        double table_step = paramb.rc_angular_pair[t12] * table_resolution;
        for (int n = 0; n <= paramb.n_max_angular; ++n) {
          std::size_t index_left_all =
            table_lookup_index(paramb, index_left, t12, paramb.n_max_angular + 1, n);
          std::size_t index_right_all =
            table_lookup_index(paramb, index_right, t12, paramb.n_max_angular + 1, n);
          double gn12 = interpolate_table_value(
            g_gn_angular, g_gnp_angular, index_left_all, index_right_all, weight_right, table_step);
          accumulate_s(paramb.L_max, d12, r12[0], r12[1], r12[2], gn12, s_all + n * NUM_OF_ABC);
        }
      }
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        double* s = s_all + n * NUM_OF_ABC;
        find_q(
          paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123,
          paramb.has_q_233, paramb.has_q_134, paramb.n_max_angular + 1, n,
          s, q.data() + (paramb.n_max_radial + 1));
        for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
          const int d = n * NUM_OF_ABC + abc;
          g_sum_fxyz[
            static_cast<std::size_t>(n1) * (paramb.n_max_angular + 1) * NUM_OF_ABC + d] =
            s[abc];
        }
      }
    } else
#endif
#ifndef USE_TABLE_FOR_RADIAL_FUNCTIONS
    {
      double s_all[MAX_NUM_N * NUM_OF_ABC];
      std::fill(
        s_all, s_all + static_cast<std::size_t>(paramb.n_max_angular + 1) * NUM_OF_ABC, 0.0);
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        const int index = i1 * N + n1;
        const int n2 = g_NL_angular[index];
        const int t2 = g_type[n2];
        const int t12 = t1 * paramb.num_types + t2;
        const double r12[3] = {
          g_x12_angular[index], g_y12_angular[index], g_z12_angular[index]};
        const double d12 =
          sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
        const double rc = paramb.rc_angular_pair[t12];
        const double rcinv = paramb.rcinv_angular_pair[t12];
        double fc12;
        double fcp12 = 0.0;
        double fn12[MAX_NUM_N];
        double fnp12[MAX_NUM_N];
        if (angular_gnp_cache) {
          find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
          find_fn_and_fnp(
            paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
        } else {
          find_fc(rc, rcinv, d12, fc12);
          find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        }
        const double* c_pair =
          annmb.c_angular_pair.data() + static_cast<std::size_t>(t12) *
                                           (paramb.n_max_angular + 1) *
                                           (paramb.basis_size_angular + 1);
        const std::size_t cache_base = angular_gnp_cache
          ? static_cast<std::size_t>(angular_edge_offsets[n1] + i1) *
              (paramb.n_max_angular + 1)
          : 0;
        for (int n = 0; n <= paramb.n_max_angular; ++n) {
          double gn12 = 0.0;
          double gnp12 = 0.0;
          const double* c_n = c_pair + n * (paramb.basis_size_angular + 1);
          for (int k = 0; k <= paramb.basis_size_angular; ++k) {
            gn12 += fn12[k] * c_n[k];
            if (angular_gnp_cache) {
              gnp12 += fnp12[k] * c_n[k];
            }
          }
          if (angular_gnp_cache) {
            angular_gn_cache[cache_base + n] = gn12;
            angular_gnp_cache[cache_base + n] = gnp12;
          }
          accumulate_s(
            paramb.L_max, d12, r12[0], r12[1], r12[2], gn12,
            s_all + n * NUM_OF_ABC);
        }
      }
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        double* s = s_all + n * NUM_OF_ABC;
        find_q(
          paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112,
          paramb.has_q_123, paramb.has_q_233, paramb.has_q_134,
          paramb.n_max_angular + 1, n, s, q.data() + (paramb.n_max_radial + 1));
        for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
          const int d = n * NUM_OF_ABC + abc;
          g_sum_fxyz[
            static_cast<std::size_t>(n1) * (paramb.n_max_angular + 1) * NUM_OF_ABC + d] =
            s[abc];
        }
      }
    }
#else
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      double s[NUM_OF_ABC] = {0.0};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int index = i1 * N + n1;
        int n2 = g_NL_angular[index];
        int t2 = g_type[n2];
        int t12 = t1 * paramb.num_types + t2;
        double r12[3] = {g_x12_angular[index], g_y12_angular[index], g_z12_angular[index]};
        double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
        int index_left, index_right;
        double weight_left, weight_right;
        double rcinv = paramb.rcinv_angular_pair[t12];
        find_index_and_weight(
          d12 * rcinv, index_left, index_right, weight_left, weight_right);
        double table_step = paramb.rc_angular_pair[t12] * table_resolution;
        std::size_t index_left_all =
          table_lookup_index(paramb, index_left, t12, paramb.n_max_angular + 1, n);
        std::size_t index_right_all =
          table_lookup_index(paramb, index_right, t12, paramb.n_max_angular + 1, n);
        double gn12 = interpolate_table_value(
          g_gn_angular, g_gnp_angular, index_left_all, index_right_all, weight_right, table_step);
        accumulate_s(paramb.L_max, d12, r12[0], r12[1], r12[2], gn12, s);
      }
      find_q(
        paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123,
        paramb.has_q_233, paramb.has_q_134, paramb.n_max_angular + 1, n, s,
        q.data() + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        const int d = n * NUM_OF_ABC + abc;
        g_sum_fxyz[
          static_cast<std::size_t>(n1) * (paramb.n_max_angular + 1) * NUM_OF_ABC + d] =
          s[abc];
      }
    }
#endif

    if (calculating_descriptor) {
      for (int d = 0; d < annmb.dim; ++d) {
        g_descriptor[d * N + n1] = q[d] * paramb.q_scaler[d];
      }
    }

    if (
      calculating_potential || calculating_latent_space || calculating_polarizability ||
      calculating_B_projection) {
      for (int d = 0; d < annmb.dim; ++d) {
        q[d] = q[d] * paramb.q_scaler[d];
      }

      if (use_batched_ann) {
        for (int d = 0; d < annmb.dim; ++d) {
          g_Fp[static_cast<std::size_t>(n1) * annmb.dim + d] = q[d];
        }
        continue;
      }

      double F = 0.0;
      std::array<double, MAX_DIM> Fp = {};
      std::array<double, MAX_NEURON> latent_space = {};

      if (calculating_polarizability) {
        apply_ann_one_layer(
          annmb.dim, annmb.num_neurons1, annmb.w0_pol[t1], annmb.b0_pol[t1], annmb.w1_pol[t1],
          annmb.b1_pol, q.data(), F, Fp.data(), latent_space.data(), false, nullptr);
        g_virial[n1] = F;
        g_virial[n1 + N * 4] = F;
        g_virial[n1 + N * 8] = F;

        for (int d = 0; d < annmb.dim; ++d) {
          Fp[d] = 0.0;
        }
        for (int d = 0; d < annmb.num_neurons1; ++d) {
          latent_space[d] = 0.0;
        }
      }

      if (paramb.version == 5) {
        apply_ann_one_layer_nep5(
          annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1,
          q.data(), F, Fp.data(), latent_space.data());
      } else {
        apply_ann_one_layer(
          annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1,
          q.data(), F, Fp.data(), latent_space.data(), calculating_B_projection,
          g_B_projection + n1 * (annmb.num_neurons1 * (annmb.dim + 2)));
      }

      if (calculating_latent_space) {
        for (int n = 0; n < annmb.num_neurons1; ++n) {
          g_latent_space[n * N + n1] = latent_space[n];
        }
      }

      if (calculating_potential) {
        g_potential[n1] += F;
      }

      for (int d = 0; d < annmb.dim; ++d) {
        g_Fp[static_cast<std::size_t>(n1) * annmb.dim + d] = Fp[d] * paramb.q_scaler[d];
      }
    }
  }

  if (use_batched_ann) {
#if defined(NEP_ADAPTERS_CPU_USE_CBLAS)
    apply_ann_one_layer_batched_by_type(
      paramb, annmb, N, g_type, g_Fp, g_potential,
      ann_q_group_workspace, ann_hidden_workspace, ann_coeff_workspace, ann_fp_group_workspace);
#endif
  }
}

struct LammpsThreadLocalScratchView {
  int force_rows = 0;
  int num_threads = 1;
  bool dense_rows = false;
  const std::vector<int>* touched_rows = nullptr;
  double* force_private = nullptr;
  double* mforce_private = nullptr;
  double* spin_transfer_private = nullptr;
  double* total_virial_private = nullptr;
  double* virial_private = nullptr;
};

constexpr int kLammpsTotalVirialStride = 8;

inline std::size_t lammps_vector_index(const int row, const int d, const int stride)
{
  (void)stride;
  return static_cast<std::size_t>(row) * 3 + d;
}

inline std::size_t lammps_virial_index(const int row, const int d, const int stride)
{
  (void)stride;
  return static_cast<std::size_t>(row) * 9 + d;
}

inline void add_spin_transfer_row_major9(
  const std::array<double, 3>& rhat,
  const double dist,
  const std::array<double, 3>& grad_sj,
  double* spin_transfer,
  const int stride,
  const int row)
{
  if (!spin_transfer) {
    return;
  }
  for (int a = 0; a < 3; ++a) {
    const double rij_a = rhat[a] * dist;
    for (int b = 0; b < 3; ++b) {
      spin_transfer[lammps_virial_index(row, 3 * a + b, stride)] -=
        rij_a * grad_sj[b];
    }
  }
}

inline void add_spin_transfer_soa9(
  const int atom_count,
  const int atom,
  const std::array<double, 3>& rhat,
  const double dist,
  const std::array<double, 3>& grad_sj,
  double* spin_transfer)
{
  if (!spin_transfer) {
    return;
  }
  for (int a = 0; a < 3; ++a) {
    const double rij_a = rhat[a] * dist;
    for (int b = 0; b < 3; ++b) {
      spin_transfer[static_cast<std::size_t>(3 * a + b) * atom_count + atom] -=
        rij_a * grad_sj[b];
    }
  }
}

inline void add_lammps_force3(
  double* force, const int row, const int stride, const double fx, const double fy, const double fz)
{
  force[lammps_vector_index(row, 0, stride)] += fx;
  force[lammps_vector_index(row, 1, stride)] += fy;
  force[lammps_vector_index(row, 2, stride)] += fz;
}

inline void add_lammps_virial9(
  double* virial,
  const int row,
  const int stride,
  const double v00,
  const double v11,
  const double v22,
  const double v01,
  const double v02,
  const double v12,
  const double v10,
  const double v20,
  const double v21)
{
  virial[lammps_virial_index(row, 0, stride)] += v00;
  virial[lammps_virial_index(row, 1, stride)] += v11;
  virial[lammps_virial_index(row, 2, stride)] += v22;
  virial[lammps_virial_index(row, 3, stride)] += v01;
  virial[lammps_virial_index(row, 4, stride)] += v02;
  virial[lammps_virial_index(row, 5, stride)] += v12;
  virial[lammps_virial_index(row, 6, stride)] += v10;
  virial[lammps_virial_index(row, 7, stride)] += v20;
  virial[lammps_virial_index(row, 8, stride)] += v21;
}

struct LammpsAngularEdgeCacheView {
  int num_centers = 0;
  int n_max_angular_plus_1 = 0;
  int* offsets = nullptr;
  int* neighbors = nullptr;
  double* x12 = nullptr;
  double* y12 = nullptr;
  double* z12 = nullptr;
  double* d12 = nullptr;
  double* gn = nullptr;
  double* gnp = nullptr;
};

struct LammpsRadialEdgeCacheView {
  int num_centers = 0;
  int n_max_radial_plus_1 = 0;
  int* offsets = nullptr;
  int* neighbors = nullptr;
  double* x12 = nullptr;
  double* y12 = nullptr;
  double* z12 = nullptr;
  double* d12 = nullptr;
  double* gnp = nullptr;
};

bool lammps_thread_scratch_active(const LammpsThreadLocalScratchView* scratch)
{
  return scratch && scratch->num_threads >= 1 && scratch->force_rows > 0 &&
         scratch->touched_rows && scratch->force_private && scratch->total_virial_private;
}

bool lammps_spin_scratch_active(const LammpsThreadLocalScratchView* scratch)
{
  return lammps_thread_scratch_active(scratch) && scratch->mforce_private;
}

bool lammps_angular_edge_cache_active(const LammpsAngularEdgeCacheView* cache)
{
  return cache && cache->num_centers > 0 && cache->n_max_angular_plus_1 > 0 &&
         cache->offsets && cache->neighbors && cache->x12 && cache->y12 &&
         cache->z12 && cache->d12 && cache->gn && cache->gnp;
}

bool lammps_radial_edge_cache_active(const LammpsRadialEdgeCacheView* cache)
{
  return cache && cache->num_centers > 0 && cache->n_max_radial_plus_1 > 0 &&
         cache->offsets && cache->neighbors && cache->x12 && cache->y12 &&
         cache->z12 && cache->d12 && cache->gnp;
}

bool use_parallel_lammps_scratch_reduce(const LammpsThreadLocalScratchView& scratch)
{
  return scratch.num_threads >= 8 && scratch.force_rows >= 32768;
}

int angular_sum_fxyz_abc_count(const NEP::ParaMB& paramb)
{
  int count = paramb.L_max * (paramb.L_max + 2);
  if (paramb.has_q_1111) {
    count = std::max(count, 3);
  }
  if (paramb.has_q_222 || paramb.has_q_112) {
    count = std::max(count, 8);
  }
  if (paramb.has_q_123 || paramb.has_q_233) {
    count = std::max(count, 15);
  }
  if (paramb.has_q_134) {
    count = std::max(count, 24);
  }
  return std::min(count, NUM_OF_ABC);
}

#if defined(_OPENMP)
void reduce_thread_local_force_virial(
  const int N,
  const int num_threads,
  const int virial_components,
  const std::vector<double>& fx_private,
  const std::vector<double>& fy_private,
  const std::vector<double>& fz_private,
  const std::vector<double>& virial_private,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  if (g_fx) {
    for (int n = 0; n < N; ++n) {
      double sum = 0.0;
      for (int tid = 0; tid < num_threads; ++tid) {
        sum += fx_private[static_cast<size_t>(tid) * N + n];
      }
      g_fx[n] += sum;
    }
  }

  if (g_fy) {
    for (int n = 0; n < N; ++n) {
      double sum = 0.0;
      for (int tid = 0; tid < num_threads; ++tid) {
        sum += fy_private[static_cast<size_t>(tid) * N + n];
      }
      g_fy[n] += sum;
    }
  }

  if (g_fz) {
    for (int n = 0; n < N; ++n) {
      double sum = 0.0;
      for (int tid = 0; tid < num_threads; ++tid) {
        sum += fz_private[static_cast<size_t>(tid) * N + n];
      }
      g_fz[n] += sum;
    }
  }

  for (int component = 0; component < virial_components; ++component) {
    double* g_virial_component = g_virial + static_cast<size_t>(component) * N;
    for (int n = 0; n < N; ++n) {
      double sum = 0.0;
      for (int tid = 0; tid < num_threads; ++tid) {
        sum += virial_private[
          (static_cast<size_t>(tid) * virial_components + component) * N + n];
      }
      g_virial_component[n] += sum;
    }
  }
}

int infer_lammps_touched_rows(
  const int N,
  int* g_ilist,
  int* g_NN,
  int** g_NL,
  std::vector<int>& touched_rows,
  std::vector<int>& touched_marks,
  int& touched_stamp)
{
  touched_rows.clear();
  if (touched_stamp == std::numeric_limits<int>::max()) {
    std::fill(touched_marks.begin(), touched_marks.end(), 0);
    touched_stamp = 0;
  }
  ++touched_stamp;

  int force_rows = 0;
  auto touch_row = [&](const int row) {
    force_rows = std::max(force_rows, row + 1);
    if (row >= static_cast<int>(touched_marks.size())) {
      touched_marks.resize(row + 1, 0);
    }
    if (touched_marks[row] != touched_stamp) {
      touched_marks[row] = touched_stamp;
      touched_rows.push_back(row);
    }
  };

  for (int ii = 0; ii < N; ++ii) {
    int n1 = g_ilist[ii];
    touch_row(n1);
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      touch_row(g_NL[n1][i1]);
    }
  }
  return force_rows;
}

void zero_lammps_thread_local_scratch(
  const LammpsThreadLocalScratchView& scratch,
  const bool use_virial)
{
  const int force_rows = scratch.force_rows;
  const int num_threads = scratch.num_threads;
  const std::vector<int>& touched_rows = *scratch.touched_rows;

  std::fill(
    scratch.total_virial_private,
    scratch.total_virial_private + static_cast<std::size_t>(num_threads) * kLammpsTotalVirialStride,
    0.0);

  const bool parallel_rows = use_parallel_lammps_scratch_reduce(scratch);
#pragma omp parallel for schedule(static) if (parallel_rows)
  for (int tid = 0; tid < num_threads; ++tid) {
    double* local_force = scratch.force_private + static_cast<std::size_t>(tid) * 3 * force_rows;
    if (scratch.dense_rows) {
      std::fill(local_force, local_force + static_cast<std::size_t>(3) * force_rows, 0.0);
    } else {
      for (int row : touched_rows) {
        local_force[lammps_vector_index(row, 0, force_rows)] = 0.0;
        local_force[lammps_vector_index(row, 1, force_rows)] = 0.0;
        local_force[lammps_vector_index(row, 2, force_rows)] = 0.0;
      }
    }

    if (scratch.mforce_private) {
      double* local_mforce =
        scratch.mforce_private + static_cast<std::size_t>(tid) * 3 * force_rows;
      if (scratch.dense_rows) {
        std::fill(local_mforce, local_mforce + static_cast<std::size_t>(3) * force_rows, 0.0);
      } else {
        for (int row : touched_rows) {
          local_mforce[lammps_vector_index(row, 0, force_rows)] = 0.0;
          local_mforce[lammps_vector_index(row, 1, force_rows)] = 0.0;
          local_mforce[lammps_vector_index(row, 2, force_rows)] = 0.0;
        }
      }
    }

    if (scratch.spin_transfer_private) {
      double* local_spin_transfer =
        scratch.spin_transfer_private + static_cast<std::size_t>(tid) * 9 * force_rows;
      if (scratch.dense_rows) {
        std::fill(
          local_spin_transfer,
          local_spin_transfer + static_cast<std::size_t>(9) * force_rows,
          0.0);
      } else {
        for (int row : touched_rows) {
          for (int d = 0; d < 9; ++d) {
            local_spin_transfer[lammps_virial_index(row, d, force_rows)] = 0.0;
          }
        }
      }
    }

    if (use_virial && scratch.virial_private) {
      double* local_virial = scratch.virial_private + static_cast<std::size_t>(tid) * 9 * force_rows;
      if (scratch.dense_rows) {
        std::fill(local_virial, local_virial + static_cast<std::size_t>(9) * force_rows, 0.0);
      } else {
        for (int row : touched_rows) {
          for (int d = 0; d < 9; ++d) {
            local_virial[lammps_virial_index(row, d, force_rows)] = 0.0;
          }
        }
      }
    }
  }
}

void reduce_lammps_thread_local_force_virial(
  const LammpsThreadLocalScratchView& scratch,
  double** g_force,
  double g_total_virial[6],
  double** g_virial,
  double** g_mforce = nullptr,
  double** g_spin_transfer = nullptr)
{
  const int force_rows = scratch.force_rows;
  const int num_threads = scratch.num_threads;
  const std::vector<int>& touched_rows = *scratch.touched_rows;

  const int rows_to_reduce = scratch.dense_rows ? force_rows : static_cast<int>(touched_rows.size());
  const bool parallel_rows = use_parallel_lammps_scratch_reduce(scratch);
#pragma omp parallel for schedule(static) if (parallel_rows)
  for (int i = 0; i < rows_to_reduce; ++i) {
    const int n = scratch.dense_rows ? i : touched_rows[i];
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    for (int tid = 0; tid < num_threads; ++tid) {
      const std::size_t base = static_cast<std::size_t>(tid) * 3 * force_rows;
      fx += scratch.force_private[base + lammps_vector_index(n, 0, force_rows)];
      fy += scratch.force_private[base + lammps_vector_index(n, 1, force_rows)];
      fz += scratch.force_private[base + lammps_vector_index(n, 2, force_rows)];
    }
    g_force[n][0] += fx;
    g_force[n][1] += fy;
    g_force[n][2] += fz;
    if (g_mforce && scratch.mforce_private) {
      double mx = 0.0;
      double my = 0.0;
      double mz = 0.0;
      for (int tid = 0; tid < num_threads; ++tid) {
        const std::size_t base = static_cast<std::size_t>(tid) * 3 * force_rows;
        mx += scratch.mforce_private[base + lammps_vector_index(n, 0, force_rows)];
        my += scratch.mforce_private[base + lammps_vector_index(n, 1, force_rows)];
        mz += scratch.mforce_private[base + lammps_vector_index(n, 2, force_rows)];
      }
      g_mforce[n][0] -= mx;
      g_mforce[n][1] -= my;
      g_mforce[n][2] -= mz;
    }
  }

  for (int d = 0; d < 6; ++d) {
    double sum = 0.0;
    for (int tid = 0; tid < num_threads; ++tid) {
      sum += scratch.total_virial_private[
        static_cast<std::size_t>(tid) * kLammpsTotalVirialStride + d];
    }
    g_total_virial[d] += sum;
  }

  if (g_virial && scratch.virial_private) {
#pragma omp parallel for schedule(static) if (parallel_rows)
    for (int i = 0; i < rows_to_reduce; ++i) {
      const int n = scratch.dense_rows ? i : touched_rows[i];
      for (int d = 0; d < 9; ++d) {
        double sum = 0.0;
        for (int tid = 0; tid < num_threads; ++tid) {
          const std::size_t base = static_cast<std::size_t>(tid) * 9 * force_rows;
          sum += scratch.virial_private[base + lammps_virial_index(n, d, force_rows)];
        }
        g_virial[n][d] += sum;
      }
    }
  }

  if (g_spin_transfer && scratch.spin_transfer_private) {
#pragma omp parallel for schedule(static) if (parallel_rows)
    for (int i = 0; i < rows_to_reduce; ++i) {
      const int n = scratch.dense_rows ? i : touched_rows[i];
      for (int d = 0; d < 9; ++d) {
        double sum = 0.0;
        for (int tid = 0; tid < num_threads; ++tid) {
          const std::size_t base = static_cast<std::size_t>(tid) * 9 * force_rows;
          sum += scratch.spin_transfer_private[
            base + lammps_virial_index(n, d, force_rows)];
        }
        g_spin_transfer[n][d] += sum;
      }
    }
  }
}
#endif

void find_force_radial_small_box(
  const bool is_dipole,
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  const int N,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  const double* g_Fp,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  const double* g_gn_radial,
  const double* g_gnp_radial,
#endif
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  const int* edge_offsets = nullptr,
  const double* gnp_cache = nullptr)
{
  auto evaluate_n1 = [&](
                       const int n1,
                       double* local_fx,
                       double* local_fy,
                       double* local_fz,
                       double* local_virial) {
    int t1 = g_type[n1];
    const double* fp_center = g_Fp + static_cast<std::size_t>(n1) * annmb.dim;
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      int t12 = t1 * paramb.num_types + t2;
      std::array<double, 3> r12 = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double d12inv = 1.0 / d12;
      std::array<double, 3> f12 = {0.0, 0.0, 0.0};
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
      int index_left, index_right;
      double weight_left, weight_right;
      double rcinv = paramb.rcinv_radial_pair[t12];
      find_index_and_weight(
        d12 * rcinv, index_left, index_right, weight_left, weight_right);
      double table_step = paramb.rc_radial_pair[t12] * table_resolution;
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        std::size_t index_left_all =
          table_lookup_index(paramb, index_left, t12, paramb.n_max_radial + 1, n);
        std::size_t index_right_all =
          table_lookup_index(paramb, index_right, t12, paramb.n_max_radial + 1, n);
        double gnp12 = interpolate_table_derivative(
          g_gn_radial, g_gnp_radial, index_left_all, index_right_all, weight_right, table_step);
        double tmp12 = fp_center[n] * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }
#else
      const double* cached_gnp = gnp_cache
        ? gnp_cache + static_cast<std::size_t>(edge_offsets[n1] + i1) *
                        (paramb.n_max_radial + 1)
        : nullptr;
      std::array<double, MAX_NUM_N> computed_gnp;
      if (!cached_gnp) {
        double fc12, fcp12;
        double rc = paramb.rc_radial_pair[t12];
        double rcinv = paramb.rcinv_radial_pair[t12];
        find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
        std::array<double, MAX_NUM_N> fn12;
        std::array<double, MAX_NUM_N> fnp12;
        find_fn_and_fnp(
          paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12.data(), fnp12.data());
        const double* c_pair =
          annmb.c_radial_pair.data() + static_cast<std::size_t>(t12) *
                                        (paramb.n_max_radial + 1) *
                                        (paramb.basis_size_radial + 1);
        for (int n = 0; n <= paramb.n_max_radial; ++n) {
          double gnp12 = 0.0;
          const double* c_n = c_pair + n * (paramb.basis_size_radial + 1);
          for (int k = 0; k <= paramb.basis_size_radial; ++k) {
            gnp12 += fnp12[k] * c_n[k];
          }
          computed_gnp[n] = gnp12;
        }
        cached_gnp = computed_gnp.data();
      }
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double tmp12 = fp_center[n] * cached_gnp[n] * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }
#endif

      if (local_fx) {
        local_fx[n1] += f12[0];
        local_fx[n2] -= f12[0];
      }

      if (local_fy) {
        local_fy[n1] += f12[1];
        local_fy[n2] -= f12[1];
      }

      if (local_fz) {
        local_fz[n1] += f12[2];
        local_fz[n2] -= f12[2];
      }

      if (!is_dipole) {
        local_virial[n2 + 0 * N] -= r12[0] * f12[0];
        local_virial[n2 + 1 * N] -= r12[0] * f12[1];
        local_virial[n2 + 2 * N] -= r12[0] * f12[2];
        local_virial[n2 + 3 * N] -= r12[1] * f12[0];
        local_virial[n2 + 4 * N] -= r12[1] * f12[1];
        local_virial[n2 + 5 * N] -= r12[1] * f12[2];
        local_virial[n2 + 6 * N] -= r12[2] * f12[0];
        local_virial[n2 + 7 * N] -= r12[2] * f12[1];
        local_virial[n2 + 8 * N] -= r12[2] * f12[2];
      } else {
        double r12_square = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
        local_virial[n2 + 0 * N] -= r12_square * f12[0];
        local_virial[n2 + 1 * N] -= r12_square * f12[1];
        local_virial[n2 + 2 * N] -= r12_square * f12[2];
      }
    }
  };

#if defined(_OPENMP)
  const int num_threads = omp_get_max_threads();
  if (num_threads > 1 && N > 0) {
    const int virial_components = is_dipole ? 3 : 9;
    std::vector<double> fx_private(g_fx ? static_cast<size_t>(num_threads) * N : 0, 0.0);
    std::vector<double> fy_private(g_fy ? static_cast<size_t>(num_threads) * N : 0, 0.0);
    std::vector<double> fz_private(g_fz ? static_cast<size_t>(num_threads) * N : 0, 0.0);
    std::vector<double> virial_private(
      static_cast<size_t>(num_threads) * virial_components * N, 0.0);

#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      double* local_fx = g_fx ? fx_private.data() + static_cast<size_t>(tid) * N : nullptr;
      double* local_fy = g_fy ? fy_private.data() + static_cast<size_t>(tid) * N : nullptr;
      double* local_fz = g_fz ? fz_private.data() + static_cast<size_t>(tid) * N : nullptr;
      double* local_virial =
        virial_private.data() + static_cast<size_t>(tid) * virial_components * N;
#pragma omp for schedule(static)
      for (int n1 = 0; n1 < N; ++n1) {
        evaluate_n1(n1, local_fx, local_fy, local_fz, local_virial);
      }
    }

    reduce_thread_local_force_virial(
      N, num_threads, virial_components, fx_private, fy_private, fz_private, virial_private, g_fx,
      g_fy, g_fz, g_virial);
    return;
  }
#endif

  for (int n1 = 0; n1 < N; ++n1) {
    evaluate_n1(n1, g_fx, g_fy, g_fz, g_virial);
  }
}

void find_force_angular_small_box(
  const bool is_dipole,
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  const int N,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  const double* g_Fp,
  const double* g_sum_fxyz,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  const double* g_gn_angular,
  const double* g_gnp_angular,
#endif
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  const int* edge_offsets = nullptr,
  const double* gn_cache = nullptr,
  const double* gnp_cache = nullptr)
{
  const int angular_sum_stride = (paramb.n_max_angular + 1) * NUM_OF_ABC;
  auto evaluate_n1 = [&](
                       const int n1,
                       double* local_fx,
                       double* local_fy,
                       double* local_fz,
                       double* local_virial) {

    const double* fp_center = g_Fp + static_cast<std::size_t>(n1) * annmb.dim;
    const double* Fp = fp_center + paramb.n_max_radial + 1;
    const double* sum_fxyz = g_sum_fxyz + static_cast<std::size_t>(n1) * angular_sum_stride;
    std::array<double, NUM_OF_ABC * MAX_NUM_N> scaled_sum_fxyz;
    std::array<double, 5 * MAX_NUM_N> q222_derivatives;
    std::array<double, 3 * MAX_NUM_N> q1111_derivatives;
    scale_sum_fxyz_3body_all_n(
      paramb.L_max, paramb.n_max_angular + 1, Fp, sum_fxyz, scaled_sum_fxyz.data());
    scale_f12_q222_q1111_all_n(
      paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.n_max_angular + 1,
      Fp, sum_fxyz, q222_derivatives.data(), q1111_derivatives.data());

    int t1 = g_type[n1];

    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[n1 + N * i1];
      int t2 = g_type[n2];
      int t12 = t1 * paramb.num_types + t2;
      std::array<double, 3> r12 = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      std::array<double, 3> f12 = {0.0, 0.0, 0.0};
      std::array<double, MAX_NUM_N> gn12_all;
      std::array<double, MAX_NUM_N> gnp12_all;
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
      int index_left, index_right;
      double weight_left, weight_right;
      double rcinv = paramb.rcinv_angular_pair[t12];
      find_index_and_weight(
        d12 * rcinv, index_left, index_right, weight_left, weight_right);
      double table_step = paramb.rc_angular_pair[t12] * table_resolution;
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        std::size_t index_left_all =
          table_lookup_index(paramb, index_left, t12, paramb.n_max_angular + 1, n);
        std::size_t index_right_all =
          table_lookup_index(paramb, index_right, t12, paramb.n_max_angular + 1, n);
        double gn12 = interpolate_table_value(
          g_gn_angular, g_gnp_angular, index_left_all, index_right_all, weight_right, table_step);
        double gnp12 = interpolate_table_derivative(
          g_gn_angular, g_gnp_angular, index_left_all, index_right_all, weight_right, table_step);
        gn12_all[n] = gn12;
        gnp12_all[n] = gnp12;
      }
#else
      if (gn_cache) {
        const std::size_t cache_base =
          static_cast<std::size_t>(edge_offsets[n1] + i1) *
          (paramb.n_max_angular + 1);
        for (int n = 0; n <= paramb.n_max_angular; ++n) {
          gn12_all[n] = gn_cache[cache_base + n];
          gnp12_all[n] = gnp_cache[cache_base + n];
        }
      } else {
        double fc12, fcp12;
        double rc = paramb.rc_angular_pair[t12];
        double rcinv = paramb.rcinv_angular_pair[t12];
        find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

        std::array<double, MAX_NUM_N> fn12;
        std::array<double, MAX_NUM_N> fnp12;
        find_fn_and_fnp(
          paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12.data(), fnp12.data());
        const double* c_pair =
          annmb.c_angular_pair.data() + static_cast<std::size_t>(t12) *
                                         (paramb.n_max_angular + 1) *
                                         (paramb.basis_size_angular + 1);
        for (int n = 0; n <= paramb.n_max_angular; ++n) {
          double gn12 = 0.0;
          double gnp12 = 0.0;
          const double* c_n = c_pair + n * (paramb.basis_size_angular + 1);
          for (int k = 0; k <= paramb.basis_size_angular; ++k) {
            gn12 += fn12[k] * c_n[k];
            gnp12 += fnp12[k] * c_n[k];
          }
          gn12_all[n] = gn12;
          gnp12_all[n] = gnp12;
        }
      }
#endif
      accumulate_f12_contracted_all_n(
        paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112,
        paramb.has_q_123, paramb.has_q_233, paramb.has_q_134, paramb.num_L,
        paramb.n_max_angular + 1, d12, r12.data(), gn12_all.data(), gnp12_all.data(), Fp, sum_fxyz,
        scaled_sum_fxyz.data(), q222_derivatives.data(), q1111_derivatives.data(), f12.data());

      if (local_fx) {
        local_fx[n1] += f12[0];
        local_fx[n2] -= f12[0];
      }

      if (local_fy) {
        local_fy[n1] += f12[1];
        local_fy[n2] -= f12[1];
      }

      if (local_fz) {
        local_fz[n1] += f12[2];
        local_fz[n2] -= f12[2];
      }

      if (!is_dipole) {
        local_virial[n2 + 0 * N] -= r12[0] * f12[0];
        local_virial[n2 + 1 * N] -= r12[0] * f12[1];
        local_virial[n2 + 2 * N] -= r12[0] * f12[2];
        local_virial[n2 + 3 * N] -= r12[1] * f12[0];
        local_virial[n2 + 4 * N] -= r12[1] * f12[1];
        local_virial[n2 + 5 * N] -= r12[1] * f12[2];
        local_virial[n2 + 6 * N] -= r12[2] * f12[0];
        local_virial[n2 + 7 * N] -= r12[2] * f12[1];
        local_virial[n2 + 8 * N] -= r12[2] * f12[2];
      } else {
        double r12_square = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
        local_virial[n2 + 0 * N] -= r12_square * f12[0];
        local_virial[n2 + 1 * N] -= r12_square * f12[1];
        local_virial[n2 + 2 * N] -= r12_square * f12[2];
      }
    }
  };

#if defined(_OPENMP)
  const int num_threads = omp_get_max_threads();
  if (num_threads > 1 && N > 0) {
    const int virial_components = is_dipole ? 3 : 9;
    std::vector<double> fx_private(g_fx ? static_cast<size_t>(num_threads) * N : 0, 0.0);
    std::vector<double> fy_private(g_fy ? static_cast<size_t>(num_threads) * N : 0, 0.0);
    std::vector<double> fz_private(g_fz ? static_cast<size_t>(num_threads) * N : 0, 0.0);
    std::vector<double> virial_private(
      static_cast<size_t>(num_threads) * virial_components * N, 0.0);

#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      double* local_fx = g_fx ? fx_private.data() + static_cast<size_t>(tid) * N : nullptr;
      double* local_fy = g_fy ? fy_private.data() + static_cast<size_t>(tid) * N : nullptr;
      double* local_fz = g_fz ? fz_private.data() + static_cast<size_t>(tid) * N : nullptr;
      double* local_virial =
        virial_private.data() + static_cast<size_t>(tid) * virial_components * N;
#pragma omp for schedule(static)
      for (int n1 = 0; n1 < N; ++n1) {
        evaluate_n1(n1, local_fx, local_fy, local_fz, local_virial);
      }
    }

    reduce_thread_local_force_virial(
      N, num_threads, virial_components, fx_private, fy_private, fz_private, virial_private, g_fx,
      g_fy, g_fz, g_virial);
    return;
  }
#endif

  for (int n1 = 0; n1 < N; ++n1) {
    evaluate_n1(n1, g_fx, g_fy, g_fz, g_virial);
  }
}

void find_force_ZBL_small_box(
  const int N,
  NEP::ParaMB& paramb,
  const NEP::ZBL& zbl,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe)
{
  for (int n1 = 0; n1 < N; ++n1) {
    int type1 = g_type[n1];
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double d12inv = 1.0 / d12;
      double f, fp;
      int type2 = g_type[n2];
      int t12 = type1 * paramb.num_types + type2;
      double a_inv = paramb.zbl_a_inv_pair[t12];
      double zizj = paramb.zbl_zizj_pair[t12];
      if (zbl.flexibled) {
        int t1, t2;
        if (type1 < type2) {
          t1 = type1;
          t2 = type2;
        } else {
          t1 = type2;
          t2 = type1;
        }
        int zbl_index = t1 * zbl.num_types - (t1 * (t1 - 1)) / 2 + (t2 - t1);
        double ZBL_para[10];
        for (int i = 0; i < 10; ++i) {
          ZBL_para[i] = zbl.para[10 * zbl_index + i];
        }
        find_f_and_fp_zbl(ZBL_para, zizj, a_inv, d12, d12inv, f, fp);
      } else {
        double rc_inner = paramb.zbl_rc_inner_pair[t12];
        double rc_outer = paramb.zbl_rc_outer_pair[t12];
        find_f_and_fp_zbl(zizj, a_inv, rc_inner, rc_outer, d12, d12inv, f, fp);
      }
      double f2 = fp * d12inv * 0.5;
      double f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      g_fx[n1] += f12[0];
      g_fy[n1] += f12[1];
      g_fz[n1] += f12[2];
      g_fx[n2] -= f12[0];
      g_fy[n2] -= f12[1];
      g_fz[n2] -= f12[2];
      g_virial[n2 + 0 * N] -= r12[0] * f12[0];
      g_virial[n2 + 1 * N] -= r12[0] * f12[1];
      g_virial[n2 + 2 * N] -= r12[0] * f12[2];
      g_virial[n2 + 3 * N] -= r12[1] * f12[0];
      g_virial[n2 + 4 * N] -= r12[1] * f12[1];
      g_virial[n2 + 5 * N] -= r12[1] * f12[2];
      g_virial[n2 + 6 * N] -= r12[2] * f12[0];
      g_virial[n2 + 7 * N] -= r12[2] * f12[1];
      g_virial[n2 + 8 * N] -= r12[2] * f12[2];
      g_pe[n1] += f * 0.5;
    }
  }
}

void find_descriptor_small_box(
  const bool calculating_potential,
  const bool calculating_descriptor,
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  const int N,
  const int* g_NN_radial,
  const int* g_NL_radial,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const double* g_x12_radial,
  const double* g_y12_radial,
  const double* g_z12_radial,
  const double* g_x12_angular,
  const double* g_y12_angular,
  const double* g_z12_angular,
  double* g_Fp,
  double* g_sum_fxyz,
  double* g_charge,
  double* g_charge_derivative,
  double* g_potential,
  double* g_descriptor)
{
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int n1 = 0; n1 < N; ++n1) {
    int t1 = g_type[n1];
    double q[MAX_DIM] = {0.0};

    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_radial[index];
      double r12[3] = {g_x12_radial[index], g_y12_radial[index], g_z12_radial[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);

      double fc12;
      int t2 = g_type[n2];
      double rc = paramb.rc_radial_max;
      double rcinv = 1.0 / rc;
      find_fc(rc, rcinv, d12, fc12);
      double fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gn12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        q[n] += gn12;
      }
    }

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      double s[NUM_OF_ABC] = {0.0};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int index = i1 * N + n1;
        int n2 = g_NL_angular[index];
        double r12[3] = {g_x12_angular[index], g_y12_angular[index], g_z12_angular[index]};
        double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
        int t2 = g_type[n2];
        double fc12;
        double rc = paramb.rc_angular_max;
        double rcinv = 1.0 / rc;
        find_fc(rc, rcinv, d12, fc12);
        double fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        double gn12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        accumulate_s(paramb.L_max, d12, r12[0], r12[1], r12[2], gn12, s);
      }
      find_q(
        paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123,
        paramb.has_q_233, paramb.has_q_134, paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        const int d = n * NUM_OF_ABC + abc;
        g_sum_fxyz[
          static_cast<std::size_t>(n1) * (paramb.n_max_angular + 1) * NUM_OF_ABC + d] =
          s[abc];
      }
    }

    if (calculating_descriptor) {
      for (int d = 0; d < annmb.dim; ++d) {
        g_descriptor[d * N + n1] = q[d] * paramb.q_scaler[d];
      }
    }

    if (calculating_potential) {
      for (int d = 0; d < annmb.dim; ++d) {
        q[d] = q[d] * paramb.q_scaler[d];
      }

      double F = 0.0, Fp[MAX_DIM] = {0.0};
      double charge = 0.0;
      double charge_derivative[MAX_DIM] = {0.0};

      apply_ann_one_layer_charge(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0[t1],
        annmb.b0[t1],
        annmb.w1[t1],
        annmb.b1,
        q,
        F,
        Fp,
        charge,
        charge_derivative);

      if (calculating_potential) {
        g_potential[n1] += F;
        g_charge[n1] = charge;
      }

      for (int d = 0; d < annmb.dim; ++d) {
        g_Fp[static_cast<std::size_t>(n1) * annmb.dim + d] = Fp[d] * paramb.q_scaler[d];
        g_charge_derivative[static_cast<std::size_t>(n1) * annmb.dim + d] =
          charge_derivative[d] * paramb.q_scaler[d];
      }
    }
  }
}

void subtract_mean(const int N, double* values)
{
  double mean = 0.0;
  for (int n = 0; n < N; ++n) {
    mean += values[n];
  }
  mean /= N;
  for (int n = 0; n < N; ++n) {
    values[n] -= mean;
  }
}

void find_force_radial_small_box(
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  const int N,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  const double* g_Fp,
  const double* g_charge_derivative,
  const double* g_D_real, 
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  for (int n1 = 0; n1 < N; ++n1) {
    int t1 = g_type[n1];
    const double* fp_center = g_Fp + static_cast<std::size_t>(n1) * annmb.dim;
    const double* charge_derivative_center =
      g_charge_derivative + static_cast<std::size_t>(n1) * annmb.dim;
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double d12inv = 1.0 / d12;
      double f12[3] = {0.0};
      double fc12, fcp12;
      double rc = paramb.rc_radial_max;
      double rcinv = 1.0 / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gnp12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        const double fp = fp_center[n];
        const double charge_derivative = charge_derivative_center[n];
        double tmp12 = (fp + charge_derivative * g_D_real[n1]) * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }

      if (g_fx) {
        g_fx[n1] += f12[0];
        g_fx[n2] -= f12[0];
      }

      if (g_fy) {
        g_fy[n1] += f12[1];
        g_fy[n2] -= f12[1];
      }

      if (g_fz) {
        g_fz[n1] += f12[2];
        g_fz[n2] -= f12[2];
      }

      g_virial[n2 + 0 * N] -= r12[0] * f12[0];
      g_virial[n2 + 1 * N] -= r12[0] * f12[1];
      g_virial[n2 + 2 * N] -= r12[0] * f12[2];
      g_virial[n2 + 3 * N] -= r12[1] * f12[0];
      g_virial[n2 + 4 * N] -= r12[1] * f12[1];
      g_virial[n2 + 5 * N] -= r12[1] * f12[2];
      g_virial[n2 + 6 * N] -= r12[2] * f12[0];
      g_virial[n2 + 7 * N] -= r12[2] * f12[1];
      g_virial[n2 + 8 * N] -= r12[2] * f12[2];
    }
  }
}

void find_force_angular_small_box(
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  const int N,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  const double* g_Fp,
  const double* g_charge_derivative,
  const double* g_D_real, 
  const double* g_sum_fxyz,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  const int angular_sum_stride = (paramb.n_max_angular + 1) * NUM_OF_ABC;
  for (int n1 = 0; n1 < N; ++n1) {

    double Fp[MAX_DIM_ANGULAR] = {0.0};
    const double* fp_center = g_Fp + static_cast<std::size_t>(n1) * annmb.dim;
    const double* charge_derivative_center =
      g_charge_derivative + static_cast<std::size_t>(n1) * annmb.dim;
    const double* sum_fxyz = g_sum_fxyz + static_cast<std::size_t>(n1) * angular_sum_stride;
    for (int d = 0; d < paramb.dim_angular; ++d) {
      const int fp_index = paramb.n_max_radial + 1 + d;
      Fp[d] = fp_center[fp_index] + charge_derivative_center[fp_index] * g_D_real[n1];
    }

    int t1 = g_type[n1];

    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[n1 + N * i1];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double f12[3] = {0.0};
      int t2 = g_type[n2];
      double fc12, fcp12;
      double rc = paramb.rc_angular_max;
      double rcinv = 1.0 / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        double gn12 = 0.0;
        double gnp12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        accumulate_f12(
          paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123,
          paramb.has_q_233, paramb.has_q_134, paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12,
          Fp, sum_fxyz, f12);
      }

      if (g_fx) {
        g_fx[n1] += f12[0];
        g_fx[n2] -= f12[0];
      }

      if (g_fy) {
        g_fy[n1] += f12[1];
        g_fy[n2] -= f12[1];
      }

      if (g_fz) {
        g_fz[n1] += f12[2];
        g_fz[n2] -= f12[2];
      }

      g_virial[n2 + 0 * N] -= r12[0] * f12[0];
      g_virial[n2 + 1 * N] -= r12[0] * f12[1];
      g_virial[n2 + 2 * N] -= r12[0] * f12[2];
      g_virial[n2 + 3 * N] -= r12[1] * f12[0];
      g_virial[n2 + 4 * N] -= r12[1] * f12[1];
      g_virial[n2 + 5 * N] -= r12[1] * f12[2];
      g_virial[n2 + 6 * N] -= r12[2] * f12[0];
      g_virial[n2 + 7 * N] -= r12[2] * f12[1];
      g_virial[n2 + 8 * N] -= r12[2] * f12[2];
    }
  }
}

void find_bec_diagonal(const int N, const double* g_q, double* g_bec)
{
  for (int n1 = 0; n1 < N; ++n1) {
    g_bec[n1 + N * 0] = g_q[n1];
    g_bec[n1 + N * 1] = 0.0;
    g_bec[n1 + N * 2] = 0.0;
    g_bec[n1 + N * 3] = 0.0;
    g_bec[n1 + N * 4] = g_q[n1];
    g_bec[n1 + N * 5] = 0.0;
    g_bec[n1 + N * 6] = 0.0;
    g_bec[n1 + N * 7] = 0.0;
    g_bec[n1 + N * 8] = g_q[n1];
  }
}

void find_bec_radial_small_box(
  const NEP::ParaMB paramb,
  const NEP::ANN annmb,
  const int N,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  const double* g_charge_derivative,
  double* g_bec)
{
  for (int n1 = 0; n1 < N; ++n1) {
    int t1 = g_type[n1];
    const double* charge_derivative_center =
      g_charge_derivative + static_cast<std::size_t>(n1) * annmb.dim;
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double d12inv = 1.0 / d12;
      double fc12, fcp12;
      double rc = paramb.rc_radial_max;
      double rcinv = 1.0 / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      double f12[3] = {0.0};

      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gnp12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        const double charge_derivative = charge_derivative_center[n];
        const double tmp12 = charge_derivative * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }

      double bec_xx = 0.5* (r12[0] * f12[0]);
      double bec_xy = 0.5* (r12[0] * f12[1]);
      double bec_xz = 0.5* (r12[0] * f12[2]);
      double bec_yx = 0.5* (r12[1] * f12[0]);
      double bec_yy = 0.5* (r12[1] * f12[1]);
      double bec_yz = 0.5* (r12[1] * f12[2]);
      double bec_zx = 0.5* (r12[2] * f12[0]);
      double bec_zy = 0.5* (r12[2] * f12[1]);
      double bec_zz = 0.5* (r12[2] * f12[2]);

      g_bec[n1] += bec_xx;
      g_bec[n1 + N] += bec_xy;
      g_bec[n1 + N * 2] += bec_xz;
      g_bec[n1 + N * 3] += bec_yx;
      g_bec[n1 + N * 4] += bec_yy;
      g_bec[n1 + N * 5] += bec_yz;
      g_bec[n1 + N * 6] += bec_zx;
      g_bec[n1 + N * 7] += bec_zy;
      g_bec[n1 + N * 8] += bec_zz;

      g_bec[n2] -= bec_xx;
      g_bec[n2 + N] -= bec_xy;
      g_bec[n2 + N * 2] -= bec_xz;
      g_bec[n2 + N * 3] -= bec_yx;
      g_bec[n2 + N * 4] -= bec_yy;
      g_bec[n2 + N * 5] -= bec_yz;
      g_bec[n2 + N * 6] -= bec_zx;
      g_bec[n2 + N * 7] -= bec_zy;
      g_bec[n2 + N * 8] -= bec_zz;
    }
  }
}

void find_bec_angular_small_box(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  const double* g_charge_derivative,
  const double* g_sum_fxyz,
  double* g_bec)
{
  const int angular_sum_stride = (paramb.n_max_angular + 1) * NUM_OF_ABC;
  for (int n1 = 0; n1 < N; ++n1) {
    const double* charge_derivative_center =
      g_charge_derivative + static_cast<std::size_t>(n1) * annmb.dim;
    const double* Fp = charge_derivative_center + paramb.n_max_radial + 1;
    const double* sum_fxyz = g_sum_fxyz + static_cast<std::size_t>(n1) * angular_sum_stride;
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[index];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double f12[3] = {0.0};
      double fc12, fcp12;
      int t2 = g_type[n2];
      double rc = paramb.rc_angular_max;
      double rcinv = 1.0 / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        double gn12 = 0.0;
        double gnp12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        accumulate_f12(
          paramb.L_max,
          paramb.num_L,
          n,
          paramb.n_max_angular + 1,
          d12,
          r12,
          gn12,
          gnp12,
          Fp,
          sum_fxyz,
          f12);
      }

      double bec_xx = 0.5* (r12[0] * f12[0]);
      double bec_xy = 0.5* (r12[0] * f12[1]);
      double bec_xz = 0.5* (r12[0] * f12[2]);
      double bec_yx = 0.5* (r12[1] * f12[0]);
      double bec_yy = 0.5* (r12[1] * f12[1]);
      double bec_yz = 0.5* (r12[1] * f12[2]);
      double bec_zx = 0.5* (r12[2] * f12[0]);
      double bec_zy = 0.5* (r12[2] * f12[1]);
      double bec_zz = 0.5* (r12[2] * f12[2]);

      g_bec[n1] += bec_xx;
      g_bec[n1 + N] += bec_xy;
      g_bec[n1 + N * 2] += bec_xz;
      g_bec[n1 + N * 3] += bec_yx;
      g_bec[n1 + N * 4] += bec_yy;
      g_bec[n1 + N * 5] += bec_yz;
      g_bec[n1 + N * 6] += bec_zx;
      g_bec[n1 + N * 7] += bec_zy;
      g_bec[n1 + N * 8] += bec_zz;

      g_bec[n2] -= bec_xx;
      g_bec[n2 + N] -= bec_xy;
      g_bec[n2 + N * 2] -= bec_xz;
      g_bec[n2 + N * 3] -= bec_yx;
      g_bec[n2 + N * 4] -= bec_yy;
      g_bec[n2 + N * 5] -= bec_yz;
      g_bec[n2 + N * 6] -= bec_zx;
      g_bec[n2 + N * 7] -= bec_zy;
      g_bec[n2 + N * 8] -= bec_zz;
    }
  }
}

void scale_bec(const int N, const double* sqrt_epsilon_inf, double* g_bec)
{
  for (int n1 = 0; n1 < N; ++n1) {
    for (int d = 0; d < 9; ++d) {
      g_bec[n1 + N * d] *= sqrt_epsilon_inf[0];
    }
  }
}

void find_force_charge_real_space_only_small_box(
  const int N,
  const NEP::Charge_Para charge_para,
  const int* g_NN,
  const int* g_NL,
  const double* g_charge,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe,
  double* g_D_real)
{
  for (int n1 = 0; n1 < N; ++n1) {
    double s_fx = 0.0;
    double s_fy = 0.0;
    double s_fz = 0.0;
    double s_sxx = 0.0;
    double s_sxy = 0.0;
    double s_sxz = 0.0;
    double s_syx = 0.0;
    double s_syy = 0.0;
    double s_syz = 0.0;
    double s_szx = 0.0;
    double s_szy = 0.0;
    double s_szz = 0.0;
    double q1 = g_charge[n1];
    double s_pe = 0; // no self energy
    double D_real = 0; // no self energy

    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      double q2 = g_charge[n2];
      double qq = q1 * q2;
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double d12inv = 1.0 / d12;

      double erfc_r = erfc(charge_para.alpha * d12) * d12inv;
      D_real += q2 * (erfc_r + charge_para.A * d12 + charge_para.B);
      s_pe += 0.5 * qq * (erfc_r + charge_para.A * d12 + charge_para.B);
      double f2 = erfc_r + charge_para.two_alpha_over_sqrt_pi * exp(-charge_para.alpha * charge_para.alpha * d12 * d12);
      f2 = -0.5 * K_C_SP * qq * (f2 * d12inv * d12inv - charge_para.A * d12inv);
      double f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      double f21[3] = {-r12[0] * f2, -r12[1] * f2, -r12[2] * f2};

      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];
      s_sxx -= r12[0] * f12[0];
      s_sxy -= r12[0] * f12[1];
      s_sxz -= r12[0] * f12[2];
      s_syx -= r12[1] * f12[0];
      s_syy -= r12[1] * f12[1];
      s_syz -= r12[1] * f12[2];
      s_szx -= r12[2] * f12[0];
      s_szy -= r12[2] * f12[1];
      s_szz -= r12[2] * f12[2];
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    g_virial[n1 + 0 * N] += s_sxx;
    g_virial[n1 + 1 * N] += s_sxy;
    g_virial[n1 + 2 * N] += s_sxz;
    g_virial[n1 + 3 * N] += s_syx;
    g_virial[n1 + 4 * N] += s_syy;
    g_virial[n1 + 5 * N] += s_syz;
    g_virial[n1 + 6 * N] += s_szx;
    g_virial[n1 + 7 * N] += s_szy;
    g_virial[n1 + 8 * N] += s_szz;
    g_D_real[n1] = K_C_SP * D_real;
    g_pe[n1] += K_C_SP * s_pe;
  }
}

void find_force_charge_real_space_small_box(
  const int N,
  const NEP::Charge_Para charge_para,
  const int* g_NN,
  const int* g_NL,
  const double* g_charge,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe,
  double* g_D_real)
{
  for (int n1 = 0; n1 < N; ++n1) {
    double s_fx = 0.0;
    double s_fy = 0.0;
    double s_fz = 0.0;
    double s_sxx = 0.0;
    double s_sxy = 0.0;
    double s_sxz = 0.0;
    double s_syx = 0.0;
    double s_syy = 0.0;
    double s_syz = 0.0;
    double s_szx = 0.0;
    double s_szy = 0.0;
    double s_szz = 0.0;
    double q1 = g_charge[n1];
    double s_pe = -charge_para.two_alpha_over_sqrt_pi * 0.5 * q1 * q1; // self energy part
    double D_real = -q1 * charge_para.two_alpha_over_sqrt_pi; // self energy part

    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      double q2 = g_charge[n2];
      double qq = q1 * q2;
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double d12inv = 1.0 / d12;

      double erfc_r = erfc(charge_para.alpha * d12) * d12inv;
      D_real += q2 * erfc_r;
      s_pe += 0.5 * qq * erfc_r;
      double f2 = erfc_r + charge_para.two_alpha_over_sqrt_pi * exp(-charge_para.alpha * charge_para.alpha * d12 * d12);
      f2 *= -0.5 * K_C_SP * qq * d12inv * d12inv;
      double f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      double f21[3] = {-r12[0] * f2, -r12[1] * f2, -r12[2] * f2};

      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];
      s_sxx -= r12[0] * f12[0];
      s_sxy -= r12[0] * f12[1];
      s_sxz -= r12[0] * f12[2];
      s_syx -= r12[1] * f12[0];
      s_syy -= r12[1] * f12[1];
      s_syz -= r12[1] * f12[2];
      s_szx -= r12[2] * f12[0];
      s_szy -= r12[2] * f12[1];
      s_szz -= r12[2] * f12[2];
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    g_virial[n1 + 0 * N] += s_sxx;
    g_virial[n1 + 1 * N] += s_sxy;
    g_virial[n1 + 2 * N] += s_sxz;
    g_virial[n1 + 3 * N] += s_syx;
    g_virial[n1 + 4 * N] += s_syy;
    g_virial[n1 + 5 * N] += s_syz;
    g_virial[n1 + 6 * N] += s_szx;
    g_virial[n1 + 7 * N] += s_szy;
    g_virial[n1 + 8 * N] += s_szz;
    g_D_real[n1] += K_C_SP * D_real;
    g_pe[n1] += K_C_SP * s_pe;
  }
}

void find_dftd3_coordination_number(
  NEP::DFTD3& dftd3,
  const int N,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12)
{
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int n1 = 0; n1 < N; ++n1) {
    int z1 = dftd3.atomic_number[g_type[n1]];
    double R_cov_1 = dftd3para::Bohr * dftd3para::covalent_radius[z1];
    double cn_temp = 0.0;
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[index];
      int z2 = dftd3.atomic_number[g_type[n2]];
      double R_cov_2 = dftd3para::Bohr * dftd3para::covalent_radius[z2];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      cn_temp += 1.0 / (exp(-16.0 * ((R_cov_1 + R_cov_2) / d12 - 1.0)) + 1.0);
    }
    dftd3.cn[n1] = cn_temp;
  }
}

void add_dftd3_force(
  NEP::DFTD3& dftd3,
  const int N,
  const int* g_NN_radial,
  const int* g_NL_radial,
  const int* g_type,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  double* g_potential,
  double* g_force,
  double* g_virial)
{
  for (int n1 = 0; n1 < N; ++n1) {
    int z1 = dftd3.atomic_number[g_type[n1]];
    int num_cn_1 = dftd3para::num_cn[z1];
    double dc6_sum = 0.0;
    double dc8_sum = 0.0;
    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_radial[index];
      int z2 = dftd3.atomic_number[g_type[n2]];
      int z_small = z1, z_large = z2;
      if (z1 > z2) {
        z_small = z2;
        z_large = z1;
      }
      int z12 = z_small * dftd3para::max_elem - (z_small * (z_small - 1)) / 2 + (z_large - z_small);
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12_2 = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      double d12_4 = d12_2 * d12_2;
      double d12_6 = d12_4 * d12_2;
      double d12_8 = d12_6 * d12_2;
      double c6 = 0.0;
      double dc6 = 0.0;
      int num_cn_2 = dftd3para::num_cn[z2];
      if (num_cn_1 == 1 && num_cn_2 == 1) {
        c6 = dftd3para::c6_ref[z12 * dftd3para::max_cn2];
      } else {
        double W = 0.0;
        double dW = 0.0;
        double Z = 0.0;
        double dZ = 0.0;
        for (int i = 0; i < num_cn_1; ++i) {
          for (int j = 0; j < num_cn_2; ++j) {
            double diff_i = dftd3.cn[n1] - dftd3para::cn_ref[z1 * dftd3para::max_cn + i];
            double diff_j = dftd3.cn[n2] - dftd3para::cn_ref[z2 * dftd3para::max_cn + j];
            double L_ij = exp(-4.0 * (diff_i * diff_i + diff_j * diff_j));
            W += L_ij;
            dW += L_ij * (-8.0 * diff_i);
            double c6_ref_ij =
              (z1 < z2) ? dftd3para::c6_ref[z12 * dftd3para::max_cn2 + i * dftd3para::max_cn + j]
                        : dftd3para::c6_ref[z12 * dftd3para::max_cn2 + j * dftd3para::max_cn + i];
            Z += c6_ref_ij * L_ij;
            dZ += c6_ref_ij * L_ij * (-8.0 * diff_i);
          }
        }
        if (W < 1.0e-30) {
          int i = num_cn_1 - 1;
          int j = num_cn_2 - 1;
          c6 = (z1 < z2) ? dftd3para::c6_ref[z12 * dftd3para::max_cn2 + i * dftd3para::max_cn + j]
                         : dftd3para::c6_ref[z12 * dftd3para::max_cn2 + j * dftd3para::max_cn + i];
        } else {
          W = 1.0 / W;
          c6 = Z * W;
          dc6 = dZ * W - c6 * dW * W;
        }
      }

      c6 *= dftd3para::HartreeBohr6;
      dc6 *= dftd3para::HartreeBohr6;
      double c8_over_c6 = 3.0 * dftd3para::r2r4[z1] * dftd3para::r2r4[z2] * dftd3para::Bohr2;
      double c8 = c6 * c8_over_c6;
      double damp = dftd3.a1 * sqrt(c8_over_c6) + dftd3.a2;
      double damp_2 = damp * damp;
      double damp_4 = damp_2 * damp_2;
      double damp_6 = 1.0 / (d12_6 + damp_4 * damp_2);
      double damp_8 = 1.0 / (d12_8 + damp_4 * damp_4);
      g_potential[n1] -= (dftd3.s6 * c6 * damp_6 + dftd3.s8 * c8 * damp_8) * 0.5;
      double f2 = dftd3.s6 * c6 * 3.0 * d12_4 * (damp_6 * damp_6) +
                  dftd3.s8 * c8 * 4.0 * d12_6 * (damp_8 * damp_8);
      double f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      g_force[n1 + 0 * N] += f12[0];
      g_force[n1 + 1 * N] += f12[1];
      g_force[n1 + 2 * N] += f12[2];
      g_force[n2 + 0 * N] -= f12[0];
      g_force[n2 + 1 * N] -= f12[1];
      g_force[n2 + 2 * N] -= f12[2];
      g_virial[n2 + 0 * N] -= r12[0] * f12[0];
      g_virial[n2 + 1 * N] -= r12[0] * f12[1];
      g_virial[n2 + 2 * N] -= r12[0] * f12[2];
      g_virial[n2 + 3 * N] -= r12[1] * f12[0];
      g_virial[n2 + 4 * N] -= r12[1] * f12[1];
      g_virial[n2 + 5 * N] -= r12[1] * f12[2];
      g_virial[n2 + 6 * N] -= r12[2] * f12[0];
      g_virial[n2 + 7 * N] -= r12[2] * f12[1];
      g_virial[n2 + 8 * N] -= r12[2] * f12[2];
      dc6_sum += dc6 * dftd3.s6 * damp_6;
      dc8_sum += dc6 * c8_over_c6 * dftd3.s8 * damp_8;
    }
    dftd3.dc6_sum[n1] = dc6_sum;
    dftd3.dc8_sum[n1] = dc8_sum;
  }
}

void add_dftd3_force_extra(
  const NEP::DFTD3& dftd3,
  const int N,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  double* g_force,
  double* g_virial)
{
  for (int n1 = 0; n1 < N; ++n1) {
    int z1 = dftd3.atomic_number[g_type[n1]];
    double R_cov_1 = dftd3para::Bohr * dftd3para::covalent_radius[z1];
    double dc6_sum = dftd3.dc6_sum[n1];
    double dc8_sum = dftd3.dc8_sum[n1];
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[index];
      int z2 = dftd3.atomic_number[g_type[n2]];
      double R_cov_2 = dftd3para::Bohr * dftd3para::covalent_radius[z2];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12_2 = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      double d12 = sqrt(d12_2);
      double cn_exp_factor = exp(-16.0 * ((R_cov_1 + R_cov_2) / d12 - 1.0));
      double f2 = cn_exp_factor * 16.0 * (R_cov_1 + R_cov_2) * (dc6_sum + dc8_sum); // not 8.0
      f2 /= (cn_exp_factor + 1.0) * (cn_exp_factor + 1.0) * d12 * d12_2;
      double f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      g_force[n1 + 0 * N] += f12[0];
      g_force[n1 + 1 * N] += f12[1];
      g_force[n1 + 2 * N] += f12[2];
      g_force[n2 + 0 * N] -= f12[0];
      g_force[n2 + 1 * N] -= f12[1];
      g_force[n2 + 2 * N] -= f12[2];
      g_virial[n2 + 0 * N] -= r12[0] * f12[0];
      g_virial[n2 + 1 * N] -= r12[0] * f12[1];
      g_virial[n2 + 2 * N] -= r12[0] * f12[2];
      g_virial[n2 + 3 * N] -= r12[1] * f12[0];
      g_virial[n2 + 4 * N] -= r12[1] * f12[1];
      g_virial[n2 + 5 * N] -= r12[1] * f12[2];
      g_virial[n2 + 6 * N] -= r12[2] * f12[0];
      g_virial[n2 + 7 * N] -= r12[2] * f12[1];
      g_virial[n2 + 8 * N] -= r12[2] * f12[2];
    }
  }
}

void find_descriptor_for_lammps(
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  int nlocal,
  int N,
  int* g_ilist,
  int* g_NN,
  int** g_NL,
  int* g_type,
  int* type_map,
  double** g_pos,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  const double* g_gn_radial,
  const double* g_gnp_radial,
  const double* g_gn_angular,
  const double* g_gnp_angular,
#endif
  double* g_Fp,
  double* g_sum_fxyz,
  double& g_total_potential,
  double* g_potential,
  LammpsRadialEdgeCacheView* radial_cache,
  LammpsAngularEdgeCacheView* angular_cache,
  std::vector<double>& ann_q_group_workspace,
  std::vector<double>& ann_hidden_workspace,
  std::vector<double>& ann_coeff_workspace,
  std::vector<double>& ann_fp_group_workspace,
  double* g_descriptor = nullptr,
  bool skip_ann = false,
  double* g_descriptor_aos = nullptr)
{
  double total_potential = 0.0;
  const int n_max_radial_plus_1 = paramb.n_max_radial + 1;
  const int n_max_angular_plus_1 = paramb.n_max_angular + 1;
  const int angular_abc_count = angular_sum_fxyz_abc_count(paramb);
  const bool use_radial_cache =
    lammps_radial_edge_cache_active(radial_cache) &&
    radial_cache->num_centers == N &&
    radial_cache->n_max_radial_plus_1 == n_max_radial_plus_1;
  const bool use_angular_cache =
    lammps_angular_edge_cache_active(angular_cache) &&
    angular_cache->num_centers == N &&
    angular_cache->n_max_angular_plus_1 == n_max_angular_plus_1;
#if defined(NEP_ADAPTERS_CPU_USE_CBLAS)
  const bool use_batched_ann = !skip_ann && paramb.version != 5;
#else
  const bool use_batched_ann = false;
#endif
  const bool phase_timing = nep_phase_timer_enabled();
  auto phase_mark = NepPhaseClock::now();
#if defined(_OPENMP)
#pragma omp parallel for reduction(+:total_potential)
#endif
  for (int ii = 0; ii < N; ++ii) {
    int n1 = g_ilist[ii];
    int t1 = type_map[g_type[n1]]; // from LAMMPS to NEP convention
    double q[MAX_DIM] = {0.0};

    const int radial_edge_offset = use_radial_cache ? radial_cache->offsets[ii] : 0;
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      const int edge_index = radial_edge_offset + i1;
      if (use_radial_cache) {
        radial_cache->neighbors[edge_index] = -1;
      }

      int n2 = g_NL[n1][i1];
      int t2 = type_map[g_type[n2]]; // from LAMMPS to NEP convention
      int t12 = t1 * paramb.num_types + t2;
      double rc = paramb.rc_radial_pair[t12];
      double rcinv = paramb.rcinv_radial_pair[t12];
      double r12[3] = {
        g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};

      double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      if (d12sq >= rc * rc) {
        continue;
      }
      double d12 = sqrt(d12sq);
      if (use_radial_cache) {
        radial_cache->neighbors[edge_index] = n2;
        radial_cache->x12[edge_index] = r12[0];
        radial_cache->y12[edge_index] = r12[1];
        radial_cache->z12[edge_index] = r12[2];
        radial_cache->d12[edge_index] = d12;
      }

#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
      int index_left, index_right;
      double weight_left, weight_right;
      find_index_and_weight(
        d12 * rcinv, index_left, index_right, weight_left, weight_right);
      double table_step = rc * table_resolution;
      const std::size_t table_pair = table_pair_slot_unchecked(paramb, t12);
      const std::size_t index_left_base =
        (static_cast<std::size_t>(index_left) * paramb.table_pair_count + table_pair) *
        n_max_radial_plus_1;
      const std::size_t index_right_base =
        (static_cast<std::size_t>(index_right) * paramb.table_pair_count + table_pair) *
        n_max_radial_plus_1;
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        std::size_t index_left_all = index_left_base + static_cast<std::size_t>(n);
        std::size_t index_right_all = index_right_base + static_cast<std::size_t>(n);
        q[n] += interpolate_table_value(
          g_gn_radial, g_gnp_radial, index_left_all, index_right_all, weight_right, table_step);
        if (use_radial_cache) {
          radial_cache->gnp[static_cast<std::size_t>(edge_index) * n_max_radial_plus_1 + n] =
            interpolate_table_derivative(
              g_gn_radial, g_gnp_radial, index_left_all, index_right_all, weight_right, table_step);
        }
      }
#else
      double fc12;
      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      if (use_radial_cache) {
        double fcp12;
        find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
        find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      } else {
        find_fc(rc, rcinv, d12, fc12);
        find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      }
      const double* c_pair = annmb.c_radial_pair.data() + static_cast<std::size_t>(t12) *
        (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gn12 = 0.0;
        double gnp12 = 0.0;
        const double* c_n = c_pair + static_cast<std::size_t>(n) * (paramb.basis_size_radial + 1);
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          gn12 += fn12[k] * c_n[k];
          if (use_radial_cache) {
            gnp12 += fnp12[k] * c_n[k];
          }
        }
        q[n] += gn12;
        if (use_radial_cache) {
          radial_cache->gnp[static_cast<std::size_t>(edge_index) * n_max_radial_plus_1 + n] = gnp12;
        }
      }
#endif
    }

    double s_all[MAX_NUM_N * NUM_OF_ABC];
    std::fill(s_all, s_all + static_cast<std::size_t>(n_max_angular_plus_1) * NUM_OF_ABC, 0.0);
    const int edge_offset = use_angular_cache ? angular_cache->offsets[ii] : 0;
#if defined(USE_TABLE_FOR_RADIAL_FUNCTIONS) && defined(NEP_CPU_HAS_LMAX4_EDGE_SIMD)
    if (paramb.L_max == 4) {
      Lmax4EdgeVec s_vec[MAX_NUM_N * lmax4_num_terms];
      std::fill(
        s_vec, s_vec + static_cast<std::size_t>(n_max_angular_plus_1) * lmax4_num_terms,
        vec_set1_f64(0.0));

      double x12_block[lmax4_edge_simd_width];
      double y12_block[lmax4_edge_simd_width];
      double z12_block[lmax4_edge_simd_width];
      double d12_block[lmax4_edge_simd_width];
      double weight_right_block[lmax4_edge_simd_width];
      double table_step_block[lmax4_edge_simd_width];
      std::size_t index_left_base_block[lmax4_edge_simd_width];
      std::size_t index_right_base_block[lmax4_edge_simd_width];
      int edge_index_block[lmax4_edge_simd_width];
      int lanes = 0;

      auto flush_lmax4_block = [&]() {
        if (lanes == lmax4_edge_simd_width) {
          Lmax4EdgeVec s_terms[lmax4_num_terms];
          compute_s_lmax4_edge_terms(x12_block, y12_block, z12_block, d12_block, s_terms);
          for (int n = 0; n < n_max_angular_plus_1; ++n) {
            double gn12_block[lmax4_edge_simd_width];
            for (int lane = 0; lane < lmax4_edge_simd_width; ++lane) {
              const std::size_t index_left_all = index_left_base_block[lane] + static_cast<std::size_t>(n);
              const std::size_t index_right_all = index_right_base_block[lane] + static_cast<std::size_t>(n);
              gn12_block[lane] = interpolate_table_value(
                g_gn_angular, g_gnp_angular, index_left_all, index_right_all,
                weight_right_block[lane], table_step_block[lane]);
              if (use_angular_cache) {
                const std::size_t cache_index =
                  static_cast<std::size_t>(edge_index_block[lane]) * n_max_angular_plus_1 + n;
                angular_cache->gn[cache_index] = gn12_block[lane];
                angular_cache->gnp[cache_index] = interpolate_table_derivative(
                  g_gn_angular, g_gnp_angular, index_left_all, index_right_all,
                  weight_right_block[lane], table_step_block[lane]);
              }
            }
            accumulate_s_lmax4_edge_terms_vec(s_terms, gn12_block, s_vec + n * lmax4_num_terms);
          }
        } else {
          for (int lane = 0; lane < lanes; ++lane) {
            const double r12[3] = {x12_block[lane], y12_block[lane], z12_block[lane]};
            for (int n = 0; n < n_max_angular_plus_1; ++n) {
              const std::size_t index_left_all = index_left_base_block[lane] + static_cast<std::size_t>(n);
              const std::size_t index_right_all = index_right_base_block[lane] + static_cast<std::size_t>(n);
              const double gn12 = interpolate_table_value(
                g_gn_angular, g_gnp_angular, index_left_all, index_right_all,
                weight_right_block[lane], table_step_block[lane]);
              accumulate_s(4, d12_block[lane], r12[0], r12[1], r12[2], gn12, s_all + n * NUM_OF_ABC);
              if (use_angular_cache) {
                const std::size_t cache_index =
                  static_cast<std::size_t>(edge_index_block[lane]) * n_max_angular_plus_1 + n;
                angular_cache->gn[cache_index] = gn12;
                angular_cache->gnp[cache_index] = interpolate_table_derivative(
                  g_gn_angular, g_gnp_angular, index_left_all, index_right_all,
                  weight_right_block[lane], table_step_block[lane]);
              }
            }
          }
        }
        lanes = 0;
      };

      for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
        const int edge_index = edge_offset + i1;
        if (use_angular_cache) {
          angular_cache->neighbors[edge_index] = -1;
        }

        int n2 = g_NL[n1][i1];
        int t2 = type_map[g_type[n2]]; // from LAMMPS to NEP convention
        int t12 = t1 * paramb.num_types + t2;
        double rc = paramb.rc_angular_pair[t12];
        double rcinv = paramb.rcinv_angular_pair[t12];
        double r12[3] = {
          g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};

        double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
        if (d12sq >= rc * rc) {
          continue;
        }
        double d12 = sqrt(d12sq);
        if (use_angular_cache) {
          angular_cache->neighbors[edge_index] = n2;
          angular_cache->x12[edge_index] = r12[0];
          angular_cache->y12[edge_index] = r12[1];
          angular_cache->z12[edge_index] = r12[2];
          angular_cache->d12[edge_index] = d12;
        }

        int index_left, index_right;
        double weight_left, weight_right;
        find_index_and_weight(d12 * rcinv, index_left, index_right, weight_left, weight_right);
        const std::size_t table_pair = table_pair_slot_unchecked(paramb, t12);
        x12_block[lanes] = r12[0];
        y12_block[lanes] = r12[1];
        z12_block[lanes] = r12[2];
        d12_block[lanes] = d12;
        weight_right_block[lanes] = weight_right;
        table_step_block[lanes] = rc * table_resolution;
        index_left_base_block[lanes] =
          (static_cast<std::size_t>(index_left) * paramb.table_pair_count + table_pair) *
          n_max_angular_plus_1;
        index_right_base_block[lanes] =
          (static_cast<std::size_t>(index_right) * paramb.table_pair_count + table_pair) *
          n_max_angular_plus_1;
        edge_index_block[lanes] = edge_index;
        ++lanes;
        if (lanes == lmax4_edge_simd_width) {
          flush_lmax4_block();
        }
      }
      if (lanes > 0) {
        flush_lmax4_block();
      }
      for (int n = 0; n < n_max_angular_plus_1; ++n) {
        for (int abc = 0; abc < lmax4_num_terms; ++abc) {
          s_all[n * NUM_OF_ABC + abc] += horizontal_add_f64(s_vec[n * lmax4_num_terms + abc]);
        }
      }
    } else
#endif
    {
      for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
        const int edge_index = edge_offset + i1;
        if (use_angular_cache) {
          angular_cache->neighbors[edge_index] = -1;
        }

        int n2 = g_NL[n1][i1];
        int t2 = type_map[g_type[n2]]; // from LAMMPS to NEP convention
        int t12 = t1 * paramb.num_types + t2;
        double rc = paramb.rc_angular_pair[t12];
        double rcinv = paramb.rcinv_angular_pair[t12];

        double r12[3] = {
          g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};

        double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
        if (d12sq >= rc * rc) {
          continue;
        }
        double d12 = sqrt(d12sq);
        if (use_angular_cache) {
          angular_cache->neighbors[edge_index] = n2;
          angular_cache->x12[edge_index] = r12[0];
          angular_cache->y12[edge_index] = r12[1];
          angular_cache->z12[edge_index] = r12[2];
          angular_cache->d12[edge_index] = d12;
        }

#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
        int index_left, index_right;
        double weight_left, weight_right;
        find_index_and_weight(
          d12 * rcinv, index_left, index_right, weight_left, weight_right);
        double table_step = rc * table_resolution;
        const std::size_t table_pair = table_pair_slot_unchecked(paramb, t12);
        const std::size_t index_left_base =
          (static_cast<std::size_t>(index_left) * paramb.table_pair_count + table_pair) *
          n_max_angular_plus_1;
        const std::size_t index_right_base =
          (static_cast<std::size_t>(index_right) * paramb.table_pair_count + table_pair) *
          n_max_angular_plus_1;
        for (int n = 0; n < n_max_angular_plus_1; ++n) {
          std::size_t index_left_all = index_left_base + static_cast<std::size_t>(n);
          std::size_t index_right_all = index_right_base + static_cast<std::size_t>(n);
          double gn12 = interpolate_table_value(
            g_gn_angular, g_gnp_angular, index_left_all, index_right_all, weight_right, table_step);
          accumulate_s(paramb.L_max, d12, r12[0], r12[1], r12[2], gn12, s_all + n * NUM_OF_ABC);
          if (use_angular_cache) {
            angular_cache->gn[static_cast<std::size_t>(edge_index) * n_max_angular_plus_1 + n] = gn12;
            angular_cache->gnp[static_cast<std::size_t>(edge_index) * n_max_angular_plus_1 + n] =
              interpolate_table_derivative(
                g_gn_angular, g_gnp_angular, index_left_all, index_right_all, weight_right, table_step);
          }
        }
#else
        double fc12;
        double fn12[MAX_NUM_N];
        double fnp12[MAX_NUM_N];
        if (use_angular_cache) {
          double fcp12;
          find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
          find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
        } else {
          find_fc(rc, rcinv, d12, fc12);
          find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        }
        const double* c_pair = annmb.c_angular_pair.data() + static_cast<std::size_t>(t12) *
          n_max_angular_plus_1 * (paramb.basis_size_angular + 1);
        for (int n = 0; n < n_max_angular_plus_1; ++n) {
          double gn12 = 0.0;
          double gnp12 = 0.0;
          const double* c_n = c_pair + static_cast<std::size_t>(n) * (paramb.basis_size_angular + 1);
          for (int k = 0; k <= paramb.basis_size_angular; ++k) {
            gn12 += fn12[k] * c_n[k];
            if (use_angular_cache) {
              gnp12 += fnp12[k] * c_n[k];
            }
          }
          accumulate_s(paramb.L_max, d12, r12[0], r12[1], r12[2], gn12, s_all + n * NUM_OF_ABC);
          if (use_angular_cache) {
            angular_cache->gn[static_cast<std::size_t>(edge_index) * n_max_angular_plus_1 + n] = gn12;
            angular_cache->gnp[static_cast<std::size_t>(edge_index) * n_max_angular_plus_1 + n] = gnp12;
          }
        }
#endif
      }
    }

    for (int n = 0; n < n_max_angular_plus_1; ++n) {
      double* s = s_all + n * NUM_OF_ABC;
      find_q(
        paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123,
        paramb.has_q_233, paramb.has_q_134, n_max_angular_plus_1, n, s, q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < angular_abc_count; ++abc) {
        const int d = n * NUM_OF_ABC + abc;
        g_sum_fxyz[static_cast<std::size_t>(n1) * n_max_angular_plus_1 * NUM_OF_ABC + d] =
          s[abc];
      }
    }

    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * paramb.q_scaler[d];
    }
    if (g_descriptor || g_descriptor_aos) {
      for (int d = 0; d < annmb.dim; ++d) {
        if (g_descriptor) {
          g_descriptor[static_cast<std::size_t>(d) * nlocal + n1] = q[d];
        }
        if (g_descriptor_aos) {
          g_descriptor_aos[static_cast<std::size_t>(ii) * annmb.dim + d] = q[d];
        }
      }
    }
    if (skip_ann) {
      continue;
    }

    if (use_batched_ann) {
      for (int d = 0; d < annmb.dim; ++d) {
        g_Fp[static_cast<std::size_t>(n1) * annmb.dim + d] = q[d];
      }
      continue;
    }

    double F = 0.0, Fp[MAX_DIM] = {0.0}, latent_space[MAX_NEURON] = {0.0};

    if (paramb.version == 5) {
      apply_ann_one_layer_nep5(
        annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1, q, F, Fp,
        latent_space);
    } else {
      apply_ann_one_layer(
        annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1, q, F, Fp,
        latent_space, false, nullptr);
    }

    total_potential += F; // always calculate this
    if (g_potential) {      // only calculate when required
      g_potential[n1] += F;
    }

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[static_cast<std::size_t>(n1) * annmb.dim + d] = Fp[d] * paramb.q_scaler[d];
    }
  }
  if (phase_timing) {
    nep_phase_timer_state().lammps.descriptor_core += nep_phase_elapsed(phase_mark);
  }
  if (use_batched_ann) {
    phase_mark = NepPhaseClock::now();
#if defined(NEP_ADAPTERS_CPU_USE_CBLAS)
    apply_ann_one_layer_batched_for_lammps(
      paramb, annmb, nlocal, N, g_ilist, g_type, type_map,
      g_Fp, g_total_potential, g_potential,
      ann_q_group_workspace, ann_hidden_workspace, ann_coeff_workspace, ann_fp_group_workspace);
    if (phase_timing) {
      nep_phase_timer_state().lammps.ann += nep_phase_elapsed(phase_mark);
    }
    return;
#endif
  }
  g_total_potential += total_potential;
}

void find_force_radial_for_lammps(
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  int nlocal,
  int N,
  int* g_ilist,
  int* g_NN,
  int** g_NL,
  int* g_type,
  int* type_map,
  double** g_pos,
  double* g_Fp,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  const double* g_gn_radial,
  const double* g_gnp_radial,
#endif
  LammpsRadialEdgeCacheView* radial_cache,
  double** g_force,
  double g_total_virial[6],
  double** g_virial,
  LammpsThreadLocalScratchView* scratch)
{
  const int n_max_radial_plus_1 = paramb.n_max_radial + 1;
  const bool use_radial_cache =
    lammps_radial_edge_cache_active(radial_cache) &&
    radial_cache->num_centers == N &&
    radial_cache->n_max_radial_plus_1 == n_max_radial_plus_1;
#if defined(_OPENMP)
  if (lammps_thread_scratch_active(scratch) && N > 0) {
    const int force_rows = scratch->force_rows;

#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      double* local_force =
        scratch->force_private + static_cast<std::size_t>(tid) * 3 * force_rows;
      double* local_total_virial =
        scratch->total_virial_private +
        static_cast<std::size_t>(tid) * kLammpsTotalVirialStride;
      double* local_virial = scratch->virial_private
        ? scratch->virial_private + static_cast<std::size_t>(tid) * 9 * force_rows
        : nullptr;

      if (use_radial_cache) {
#pragma omp for schedule(static)
        for (int ii = 0; ii < N; ++ii) {
          int n1 = g_ilist[ii];
          const double* fp_center = g_Fp + static_cast<std::size_t>(n1) * annmb.dim;
          const int edge_begin = radial_cache->offsets[ii];
          const int edge_end = radial_cache->offsets[ii + 1];
          double center_fx = 0.0;
          double center_fy = 0.0;
          double center_fz = 0.0;
          double center_virial[6] = {0.0};
          for (int edge_index = edge_begin; edge_index < edge_end; ++edge_index) {
            int n2 = radial_cache->neighbors[edge_index];
            if (n2 < 0) {
              continue;
            }
            const double r12[3] = {
              radial_cache->x12[edge_index],
              radial_cache->y12[edge_index],
              radial_cache->z12[edge_index]};
            const double d12inv = 1.0 / radial_cache->d12[edge_index];
            const double* gnp_edge =
              radial_cache->gnp + static_cast<std::size_t>(edge_index) * n_max_radial_plus_1;
            double f12[3] = {0.0};
            for (int n = 0; n < n_max_radial_plus_1; ++n) {
              const double tmp12 = fp_center[n] * gnp_edge[n] * d12inv;
              f12[0] += tmp12 * r12[0];
              f12[1] += tmp12 * r12[1];
              f12[2] += tmp12 * r12[2];
            }

            center_fx += f12[0];
            center_fy += f12[1];
            center_fz += f12[2];
            add_lammps_force3(local_force, n2, force_rows, -f12[0], -f12[1], -f12[2]);

            center_virial[0] -= r12[0] * f12[0]; // xx
            center_virial[1] -= r12[1] * f12[1]; // yy
            center_virial[2] -= r12[2] * f12[2]; // zz
            center_virial[3] -= r12[0] * f12[1]; // xy
            center_virial[4] -= r12[0] * f12[2]; // xz
            center_virial[5] -= r12[1] * f12[2]; // yz
            if (local_virial) {
              add_lammps_virial9(
                local_virial, n2, force_rows,
                -r12[0] * f12[0], -r12[1] * f12[1], -r12[2] * f12[2],
                -r12[0] * f12[1], -r12[0] * f12[2], -r12[1] * f12[2],
                -r12[1] * f12[0], -r12[2] * f12[0], -r12[2] * f12[1]);
            }
          }
          add_lammps_force3(local_force, n1, force_rows, center_fx, center_fy, center_fz);
          for (int d = 0; d < 6; ++d) {
            local_total_virial[d] += center_virial[d];
          }
        }
      } else {
#pragma omp for schedule(static)
        for (int ii = 0; ii < N; ++ii) {
          int n1 = g_ilist[ii];
          const double* fp_center = g_Fp + static_cast<std::size_t>(n1) * annmb.dim;
          int t1 = type_map[g_type[n1]]; // from LAMMPS to NEP convention
          for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
            int n2 = g_NL[n1][i1];
            int t2 = type_map[g_type[n2]]; // from LAMMPS to NEP convention
            int t12 = t1 * paramb.num_types + t2;
            double rc = paramb.rc_radial_pair[t12];
            double rcinv = paramb.rcinv_radial_pair[t12];
            double r12[3] = {
              g_pos[n2][0] - g_pos[n1][0],
              g_pos[n2][1] - g_pos[n1][1],
              g_pos[n2][2] - g_pos[n1][2]};

            double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
            if (d12sq >= rc * rc) {
              continue;
            }
            double d12 = sqrt(d12sq);
            double d12inv = 1.0 / d12;
            double f12[3] = {0.0};
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
            int index_left, index_right;
            double weight_left, weight_right;
            find_index_and_weight(
              d12 * rcinv, index_left, index_right, weight_left, weight_right);
            double table_step = rc * table_resolution;
            for (int n = 0; n <= paramb.n_max_radial; ++n) {
              std::size_t index_left_all =
                table_lookup_index(paramb, index_left, t12, paramb.n_max_radial + 1, n);
              std::size_t index_right_all =
                table_lookup_index(paramb, index_right, t12, paramb.n_max_radial + 1, n);
              double gnp12 = interpolate_table_derivative(
                g_gn_radial, g_gnp_radial, index_left_all, index_right_all, weight_right, table_step);
              double tmp12 = fp_center[n] * gnp12 * d12inv;
              for (int d = 0; d < 3; ++d) {
                f12[d] += tmp12 * r12[d];
              }
            }
#else
            double fc12, fcp12;
            find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
            double fn12[MAX_NUM_N];
            double fnp12[MAX_NUM_N];
            find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
            const double* c_pair = annmb.c_radial_pair.data() + static_cast<std::size_t>(t12) *
              (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);
            for (int n = 0; n <= paramb.n_max_radial; ++n) {
              double gnp12 = 0.0;
              const double* c_n = c_pair + static_cast<std::size_t>(n) * (paramb.basis_size_radial + 1);
              for (int k = 0; k <= paramb.basis_size_radial; ++k) {
                gnp12 += fnp12[k] * c_n[k];
              }
              double tmp12 = fp_center[n] * gnp12 * d12inv;
              for (int d = 0; d < 3; ++d) {
                f12[d] += tmp12 * r12[d];
              }
            }
#endif

            add_lammps_force3(local_force, n1, force_rows, f12[0], f12[1], f12[2]);
            add_lammps_force3(local_force, n2, force_rows, -f12[0], -f12[1], -f12[2]);

            local_total_virial[0] -= r12[0] * f12[0]; // xx
            local_total_virial[1] -= r12[1] * f12[1]; // yy
            local_total_virial[2] -= r12[2] * f12[2]; // zz
            local_total_virial[3] -= r12[0] * f12[1]; // xy
            local_total_virial[4] -= r12[0] * f12[2]; // xz
            local_total_virial[5] -= r12[1] * f12[2]; // yz
            if (local_virial) {
              add_lammps_virial9(
                local_virial, n2, force_rows,
                -r12[0] * f12[0], -r12[1] * f12[1], -r12[2] * f12[2],
                -r12[0] * f12[1], -r12[0] * f12[2], -r12[1] * f12[2],
                -r12[1] * f12[0], -r12[2] * f12[0], -r12[2] * f12[1]);
            }
          }
        }
      }
    }
    return;
  }
#endif

  for (int ii = 0; ii < N; ++ii) {
    int n1 = g_ilist[ii];
    const double* fp_center = g_Fp + static_cast<std::size_t>(n1) * annmb.dim;
    int t1 = type_map[g_type[n1]]; // from LAMMPS to NEP convention
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1][i1];
      int t2 = type_map[g_type[n2]]; // from LAMMPS to NEP convention
      int t12 = t1 * paramb.num_types + t2;
      double rc = paramb.rc_radial_pair[t12];
      double rcinv = paramb.rcinv_radial_pair[t12];
      double r12[3] = {
        g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};

      double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      if (d12sq >= rc * rc) {
        continue;
      }
      double d12 = sqrt(d12sq);
      double d12inv = 1.0 / d12;
      double f12[3] = {0.0};
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
      int index_left, index_right;
      double weight_left, weight_right;
      find_index_and_weight(
        d12 * rcinv, index_left, index_right, weight_left, weight_right);
      double table_step = rc * table_resolution;
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        std::size_t index_left_all =
          table_lookup_index(paramb, index_left, t12, paramb.n_max_radial + 1, n);
        std::size_t index_right_all =
          table_lookup_index(paramb, index_right, t12, paramb.n_max_radial + 1, n);
        double gnp12 = interpolate_table_derivative(
          g_gn_radial, g_gnp_radial, index_left_all, index_right_all, weight_right, table_step);
        double tmp12 = fp_center[n] * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }
#else
      double fc12, fcp12;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      const double* c_pair = annmb.c_radial_pair.data() + static_cast<std::size_t>(t12) *
        (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gnp12 = 0.0;
        const double* c_n = c_pair + static_cast<std::size_t>(n) * (paramb.basis_size_radial + 1);
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          gnp12 += fnp12[k] * c_n[k];
        }
        double tmp12 = fp_center[n] * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }
#endif

      g_force[n1][0] += f12[0];
      g_force[n1][1] += f12[1];
      g_force[n1][2] += f12[2];
      g_force[n2][0] -= f12[0];
      g_force[n2][1] -= f12[1];
      g_force[n2][2] -= f12[2];

      // always calculate the total virial:
      g_total_virial[0] -= r12[0] * f12[0]; // xx
      g_total_virial[1] -= r12[1] * f12[1]; // yy
      g_total_virial[2] -= r12[2] * f12[2]; // zz
      g_total_virial[3] -= r12[0] * f12[1]; // xy
      g_total_virial[4] -= r12[0] * f12[2]; // xz
      g_total_virial[5] -= r12[1] * f12[2]; // yz
      if (g_virial) {                       // only calculate the per-atom virial when required
        g_virial[n2][0] -= r12[0] * f12[0]; // xx
        g_virial[n2][1] -= r12[1] * f12[1]; // yy
        g_virial[n2][2] -= r12[2] * f12[2]; // zz
        g_virial[n2][3] -= r12[0] * f12[1]; // xy
        g_virial[n2][4] -= r12[0] * f12[2]; // xz
        g_virial[n2][5] -= r12[1] * f12[2]; // yz
        g_virial[n2][6] -= r12[1] * f12[0]; // yx
        g_virial[n2][7] -= r12[2] * f12[0]; // zx
        g_virial[n2][8] -= r12[2] * f12[1]; // zy
      }
    }
  }
}

void find_force_angular_for_lammps(
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  int nlocal,
  int N,
  int* g_ilist,
  int* g_NN,
  int** g_NL,
  int* g_type,
  int* type_map,
  double** g_pos,
  double* g_Fp,
  double* g_sum_fxyz,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  const double* g_gn_angular,
  const double* g_gnp_angular,
#endif
  double** g_force,
  double g_total_virial[6],
  double** g_virial,
  LammpsAngularEdgeCacheView* angular_cache,
  LammpsThreadLocalScratchView* scratch)
{
  const int n_max_angular_plus_1 = paramb.n_max_angular + 1;
  const int angular_sum_stride = n_max_angular_plus_1 * NUM_OF_ABC;
  const bool use_angular_cache =
    lammps_angular_edge_cache_active(angular_cache) &&
    angular_cache->num_centers == N &&
    angular_cache->n_max_angular_plus_1 == n_max_angular_plus_1;
  const bool use_lmax4_q222_q1111_n5 =
    paramb.L_max == 4 && paramb.has_q_222 && paramb.has_q_1111 &&
    !paramb.has_q_112 && !paramb.has_q_123 && !paramb.has_q_233 &&
    !paramb.has_q_134 && paramb.num_L == 6 && n_max_angular_plus_1 == 5;
#if defined(_OPENMP)
  if (lammps_thread_scratch_active(scratch) && N > 0) {
    const int force_rows = scratch->force_rows;

#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      double* local_force =
        scratch->force_private + static_cast<std::size_t>(tid) * 3 * force_rows;
      double* local_total_virial =
        scratch->total_virial_private +
        static_cast<std::size_t>(tid) * kLammpsTotalVirialStride;
      double* local_virial = scratch->virial_private
        ? scratch->virial_private + static_cast<std::size_t>(tid) * 9 * force_rows
        : nullptr;

#pragma omp for schedule(static)
      for (int ii = 0; ii < N; ++ii) {
        int n1 = g_ilist[ii];
        const double* fp_center = g_Fp + static_cast<std::size_t>(n1) * annmb.dim;
        const double* Fp = fp_center + paramb.n_max_radial + 1;
        const double* sum_fxyz =
          g_sum_fxyz + static_cast<std::size_t>(n1) * angular_sum_stride;

        if (use_angular_cache) {
          double scaled_sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
          double q222_derivatives[5 * MAX_NUM_N];
          double q1111_derivatives[3 * MAX_NUM_N];
          scale_sum_fxyz_3body_all_n(
            paramb.L_max, n_max_angular_plus_1, Fp, sum_fxyz, scaled_sum_fxyz);
          scale_f12_q222_q1111_all_n(
            paramb.L_max, paramb.has_q_222, paramb.has_q_1111, n_max_angular_plus_1,
            Fp, sum_fxyz, q222_derivatives, q1111_derivatives);
          const int edge_begin = angular_cache->offsets[ii];
          const int edge_end = angular_cache->offsets[ii + 1];
          double center_fx = 0.0;
          double center_fy = 0.0;
          double center_fz = 0.0;
          double center_virial[6] = {0.0};
          for (int edge_index = edge_begin; edge_index < edge_end; ++edge_index) {
            int n2 = angular_cache->neighbors[edge_index];
            if (n2 < 0) {
              continue;
            }

            double r12[3] = {
              angular_cache->x12[edge_index],
              angular_cache->y12[edge_index],
              angular_cache->z12[edge_index]};
            double d12 = angular_cache->d12[edge_index];
            double f12[3] = {0.0};
            const double* gn_edge =
              angular_cache->gn + static_cast<std::size_t>(edge_index) * n_max_angular_plus_1;
            const double* gnp_edge =
              angular_cache->gnp + static_cast<std::size_t>(edge_index) * n_max_angular_plus_1;
            if (use_lmax4_q222_q1111_n5) {
              accumulate_f12_contracted_lmax4_q222_q1111_n5(
                d12, r12, gn_edge, gnp_edge, scaled_sum_fxyz,
                q222_derivatives, q1111_derivatives, f12);
            } else {
              accumulate_f12_contracted_all_n(
                paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112,
                paramb.has_q_123, paramb.has_q_233, paramb.has_q_134, paramb.num_L,
                n_max_angular_plus_1, d12, r12, gn_edge, gnp_edge, Fp, sum_fxyz,
                scaled_sum_fxyz, q222_derivatives, q1111_derivatives, f12);
            }

            center_fx += f12[0];
            center_fy += f12[1];
            center_fz += f12[2];
            add_lammps_force3(local_force, n2, force_rows, -f12[0], -f12[1], -f12[2]);

            center_virial[0] -= r12[0] * f12[0]; // xx
            center_virial[1] -= r12[1] * f12[1]; // yy
            center_virial[2] -= r12[2] * f12[2]; // zz
            center_virial[3] -= r12[0] * f12[1]; // xy
            center_virial[4] -= r12[0] * f12[2]; // xz
            center_virial[5] -= r12[1] * f12[2]; // yz
            if (local_virial) {
              add_lammps_virial9(
                local_virial, n2, force_rows,
                -r12[0] * f12[0], -r12[1] * f12[1], -r12[2] * f12[2],
                -r12[0] * f12[1], -r12[0] * f12[2], -r12[1] * f12[2],
                -r12[1] * f12[0], -r12[2] * f12[0], -r12[2] * f12[1]);
            }
          }
          add_lammps_force3(local_force, n1, force_rows, center_fx, center_fy, center_fz);
          for (int d = 0; d < 6; ++d) {
            local_total_virial[d] += center_virial[d];
          }
          continue;
        }

        int t1 = type_map[g_type[n1]]; // from LAMMPS to NEP convention
        for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
          int n2 = g_NL[n1][i1];
          int t2 = type_map[g_type[n2]]; // from LAMMPS to NEP convention
          int t12 = t1 * paramb.num_types + t2;
          double rc = paramb.rc_angular_pair[t12];
          double rcinv = paramb.rcinv_angular_pair[t12];
          double r12[3] = {
            g_pos[n2][0] - g_pos[n1][0],
            g_pos[n2][1] - g_pos[n1][1],
            g_pos[n2][2] - g_pos[n1][2]};

          double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
          if (d12sq >= rc * rc) {
            continue;
          }
          double d12 = sqrt(d12sq);
          double f12[3] = {0.0};

#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
          int index_left, index_right;
          double weight_left, weight_right;
          find_index_and_weight(
            d12 * rcinv, index_left, index_right, weight_left, weight_right);
          double table_step = rc * table_resolution;
          for (int n = 0; n <= paramb.n_max_angular; ++n) {
            std::size_t index_left_all =
              table_lookup_index(paramb, index_left, t12, paramb.n_max_angular + 1, n);
            std::size_t index_right_all =
              table_lookup_index(paramb, index_right, t12, paramb.n_max_angular + 1, n);
            double gn12 = interpolate_table_value(
              g_gn_angular, g_gnp_angular, index_left_all, index_right_all, weight_right, table_step);
            double gnp12 = interpolate_table_derivative(
              g_gn_angular, g_gnp_angular, index_left_all, index_right_all, weight_right, table_step);
            accumulate_f12(
              paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123,
              paramb.has_q_233, paramb.has_q_134, paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12,
              Fp, sum_fxyz, f12);
          }
#else
          double fc12, fcp12;
          find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
          double fn12[MAX_NUM_N];
          double fnp12[MAX_NUM_N];
          find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
          const double* c_pair = annmb.c_angular_pair.data() + static_cast<std::size_t>(t12) *
            (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1);
          for (int n = 0; n <= paramb.n_max_angular; ++n) {
            double gn12 = 0.0;
            double gnp12 = 0.0;
            const double* c_n = c_pair + static_cast<std::size_t>(n) * (paramb.basis_size_angular + 1);
            for (int k = 0; k <= paramb.basis_size_angular; ++k) {
              gn12 += fn12[k] * c_n[k];
              gnp12 += fnp12[k] * c_n[k];
            }
            accumulate_f12(
              paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123,
              paramb.has_q_233, paramb.has_q_134, paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12,
              Fp, sum_fxyz, f12);
          }
#endif

          add_lammps_force3(local_force, n1, force_rows, f12[0], f12[1], f12[2]);
          add_lammps_force3(local_force, n2, force_rows, -f12[0], -f12[1], -f12[2]);

          local_total_virial[0] -= r12[0] * f12[0]; // xx
          local_total_virial[1] -= r12[1] * f12[1]; // yy
          local_total_virial[2] -= r12[2] * f12[2]; // zz
          local_total_virial[3] -= r12[0] * f12[1]; // xy
          local_total_virial[4] -= r12[0] * f12[2]; // xz
          local_total_virial[5] -= r12[1] * f12[2]; // yz
          if (local_virial) {
            add_lammps_virial9(
              local_virial, n2, force_rows,
              -r12[0] * f12[0], -r12[1] * f12[1], -r12[2] * f12[2],
              -r12[0] * f12[1], -r12[0] * f12[2], -r12[1] * f12[2],
              -r12[1] * f12[0], -r12[2] * f12[0], -r12[2] * f12[1]);
          }
        }
      }
    }
    return;
  }
#endif

  for (int ii = 0; ii < N; ++ii) {
    int n1 = g_ilist[ii];
    const double* fp_center = g_Fp + static_cast<std::size_t>(n1) * annmb.dim;
    const double* Fp = fp_center + paramb.n_max_radial + 1;
    const double* sum_fxyz = g_sum_fxyz + static_cast<std::size_t>(n1) * angular_sum_stride;

    int t1 = type_map[g_type[n1]]; // from LAMMPS to NEP convention

    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1][i1];
      int t2 = type_map[g_type[n2]]; // from LAMMPS to NEP convention
      int t12 = t1 * paramb.num_types + t2;
      double rc = paramb.rc_angular_pair[t12];
      double rcinv = paramb.rcinv_angular_pair[t12];
      double r12[3] = {
        g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};

      double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      if (d12sq >= rc * rc) {
        continue;
      }
      double d12 = sqrt(d12sq);
      double f12[3] = {0.0};

#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
      int index_left, index_right;
      double weight_left, weight_right;
      find_index_and_weight(
        d12 * rcinv, index_left, index_right, weight_left, weight_right);
      double table_step = rc * table_resolution;
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        std::size_t index_left_all =
          table_lookup_index(paramb, index_left, t12, paramb.n_max_angular + 1, n);
        std::size_t index_right_all =
          table_lookup_index(paramb, index_right, t12, paramb.n_max_angular + 1, n);
        double gn12 = interpolate_table_value(
          g_gn_angular, g_gnp_angular, index_left_all, index_right_all, weight_right, table_step);
        double gnp12 = interpolate_table_derivative(
          g_gn_angular, g_gnp_angular, index_left_all, index_right_all, weight_right, table_step);
        accumulate_f12(
          paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123,
          paramb.has_q_233, paramb.has_q_134, paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12,
          Fp, sum_fxyz, f12);
      }
#else
      double fc12, fcp12;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      const double* c_pair = annmb.c_angular_pair.data() + static_cast<std::size_t>(t12) *
        (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        double gn12 = 0.0;
        double gnp12 = 0.0;
        const double* c_n = c_pair + static_cast<std::size_t>(n) * (paramb.basis_size_angular + 1);
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          gn12 += fn12[k] * c_n[k];
          gnp12 += fnp12[k] * c_n[k];
        }
        accumulate_f12(
          paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123,
          paramb.has_q_233, paramb.has_q_134, paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12,
          Fp, sum_fxyz, f12);
      }
#endif

      g_force[n1][0] += f12[0];
      g_force[n1][1] += f12[1];
      g_force[n1][2] += f12[2];
      g_force[n2][0] -= f12[0];
      g_force[n2][1] -= f12[1];
      g_force[n2][2] -= f12[2];
      // always calculate the total virial:
      g_total_virial[0] -= r12[0] * f12[0]; // xx
      g_total_virial[1] -= r12[1] * f12[1]; // yy
      g_total_virial[2] -= r12[2] * f12[2]; // zz
      g_total_virial[3] -= r12[0] * f12[1]; // xy
      g_total_virial[4] -= r12[0] * f12[2]; // xz
      g_total_virial[5] -= r12[1] * f12[2]; // yz
      if (g_virial) {                       // only calculate the per-atom virial when required
        g_virial[n2][0] -= r12[0] * f12[0]; // xx
        g_virial[n2][1] -= r12[1] * f12[1]; // yy
        g_virial[n2][2] -= r12[2] * f12[2]; // zz
        g_virial[n2][3] -= r12[0] * f12[1]; // xy
        g_virial[n2][4] -= r12[0] * f12[2]; // xz
        g_virial[n2][5] -= r12[1] * f12[2]; // yz
        g_virial[n2][6] -= r12[1] * f12[0]; // yx
        g_virial[n2][7] -= r12[2] * f12[0]; // zx
        g_virial[n2][8] -= r12[2] * f12[1]; // zy
      }
    }
  }
}

void find_force_ZBL_for_lammps(
  NEP::ParaMB& paramb,
  const NEP::ZBL& zbl,
  int N,
  int* g_ilist,
  int* g_NN,
  int** g_NL,
  int* g_type,
  int* type_map,
  double** g_pos,
  double** g_force,
  double g_total_virial[6],
  double** g_virial,
  double& g_total_potential,
  double* g_potential,
  LammpsThreadLocalScratchView* scratch)
{
#if defined(_OPENMP)
  if (lammps_thread_scratch_active(scratch) && N > 0) {
    const int force_rows = scratch->force_rows;
    double total_potential = 0.0;

#pragma omp parallel reduction(+:total_potential)
    {
      const int tid = omp_get_thread_num();
      double* local_force =
        scratch->force_private + static_cast<std::size_t>(tid) * 3 * force_rows;
      double* local_total_virial =
        scratch->total_virial_private +
        static_cast<std::size_t>(tid) * kLammpsTotalVirialStride;
      double* local_virial = scratch->virial_private
        ? scratch->virial_private + static_cast<std::size_t>(tid) * 9 * force_rows
        : nullptr;

#pragma omp for schedule(static)
      for (int ii = 0; ii < N; ++ii) {
        int n1 = g_ilist[ii];
        int type1 = type_map[g_type[n1]]; // from LAMMPS to NEP convention
        for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
          int n2 = g_NL[n1][i1];
          double r12[3] = {
            g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};

          double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
          double max_rc_outer = 2.5;
          if (d12sq >= max_rc_outer * max_rc_outer) {
            continue;
          }
          double d12 = sqrt(d12sq);

          double d12inv = 1.0 / d12;
          double f, fp;
          int type2 = type_map[g_type[n2]]; // from LAMMPS to NEP convention
          int t12 = type1 * paramb.num_types + type2;
          double a_inv = paramb.zbl_a_inv_pair[t12];
          double zizj = paramb.zbl_zizj_pair[t12];
          if (zbl.flexibled) {
            int t1, t2;
            if (type1 < type2) {
              t1 = type1;
              t2 = type2;
            } else {
              t1 = type2;
              t2 = type1;
            }
            int zbl_index = t1 * zbl.num_types - (t1 * (t1 - 1)) / 2 + (t2 - t1);
            double ZBL_para[10];
            for (int i = 0; i < 10; ++i) {
              ZBL_para[i] = zbl.para[10 * zbl_index + i];
            }
            find_f_and_fp_zbl(ZBL_para, zizj, a_inv, d12, d12inv, f, fp);
          } else {
            double rc_inner = paramb.zbl_rc_inner_pair[t12];
            double rc_outer = paramb.zbl_rc_outer_pair[t12];
            find_f_and_fp_zbl(zizj, a_inv, rc_inner, rc_outer, d12, d12inv, f, fp);
          }
          double f2 = fp * d12inv * 0.5;
          double f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
          add_lammps_force3(local_force, n1, force_rows, f12[0], f12[1], f12[2]);
          add_lammps_force3(local_force, n2, force_rows, -f12[0], -f12[1], -f12[2]);
          local_total_virial[0] -= r12[0] * f12[0]; // xx
          local_total_virial[1] -= r12[1] * f12[1]; // yy
          local_total_virial[2] -= r12[2] * f12[2]; // zz
          local_total_virial[3] -= r12[0] * f12[1]; // xy
          local_total_virial[4] -= r12[0] * f12[2]; // xz
          local_total_virial[5] -= r12[1] * f12[2]; // yz
          if (local_virial) {
            add_lammps_virial9(
              local_virial, n2, force_rows,
              -r12[0] * f12[0], -r12[1] * f12[1], -r12[2] * f12[2],
              -r12[0] * f12[1], -r12[0] * f12[2], -r12[1] * f12[2],
              -r12[1] * f12[0], -r12[2] * f12[0], -r12[2] * f12[1]);
          }
          total_potential += f * 0.5;
          if (g_potential) {
            g_potential[n1] += f * 0.5;
          }
        }
      }
    }
    g_total_potential += total_potential;
    return;
  }
#endif

  for (int ii = 0; ii < N; ++ii) {
    int n1 = g_ilist[ii];
    int type1 = type_map[g_type[n1]]; // from LAMMPS to NEP convention
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1][i1];
      double r12[3] = {
        g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};

      double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      double max_rc_outer = 2.5;
      if (d12sq >= max_rc_outer * max_rc_outer) {
        continue;
      }
      double d12 = sqrt(d12sq);

      double d12inv = 1.0 / d12;
      double f, fp;
      int type2 = type_map[g_type[n2]]; // from LAMMPS to NEP convention
      int t12 = type1 * paramb.num_types + type2;
      double a_inv = paramb.zbl_a_inv_pair[t12];
      double zizj = paramb.zbl_zizj_pair[t12];
      if (zbl.flexibled) {
        int t1, t2;
        if (type1 < type2) {
          t1 = type1;
          t2 = type2;
        } else {
          t1 = type2;
          t2 = type1;
        }
        int zbl_index = t1 * zbl.num_types - (t1 * (t1 - 1)) / 2 + (t2 - t1);
        double ZBL_para[10];
        for (int i = 0; i < 10; ++i) {
          ZBL_para[i] = zbl.para[10 * zbl_index + i];
        }
        find_f_and_fp_zbl(ZBL_para, zizj, a_inv, d12, d12inv, f, fp);
      } else {
        double rc_inner = paramb.zbl_rc_inner_pair[t12];
        double rc_outer = paramb.zbl_rc_outer_pair[t12];
        find_f_and_fp_zbl(zizj, a_inv, rc_inner, rc_outer, d12, d12inv, f, fp);
      }
      double f2 = fp * d12inv * 0.5;
      double f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      g_force[n1][0] += f12[0]; // accumulation here
      g_force[n1][1] += f12[1];
      g_force[n1][2] += f12[2];
      g_force[n2][0] -= f12[0];
      g_force[n2][1] -= f12[1];
      g_force[n2][2] -= f12[2];
      // always calculate the total virial:
      g_total_virial[0] -= r12[0] * f12[0]; // xx
      g_total_virial[1] -= r12[1] * f12[1]; // yy
      g_total_virial[2] -= r12[2] * f12[2]; // zz
      g_total_virial[3] -= r12[0] * f12[1]; // xy
      g_total_virial[4] -= r12[0] * f12[2]; // xz
      g_total_virial[5] -= r12[1] * f12[2]; // yz
      if (g_virial) {                       // only calculate the per-atom virial when required
        g_virial[n2][0] -= r12[0] * f12[0]; // xx
        g_virial[n2][1] -= r12[1] * f12[1]; // yy
        g_virial[n2][2] -= r12[2] * f12[2]; // zz
        g_virial[n2][3] -= r12[0] * f12[1]; // xy
        g_virial[n2][4] -= r12[0] * f12[2]; // xz
        g_virial[n2][5] -= r12[1] * f12[2]; // yz
        g_virial[n2][6] -= r12[1] * f12[0]; // yx
        g_virial[n2][7] -= r12[2] * f12[0]; // zx
        g_virial[n2][8] -= r12[2] * f12[1]; // zy
      }
      g_total_potential += f * 0.5; // always calculate this
      if (g_potential) {            // only calculate when required
        g_potential[n1] += f * 0.5;
      }
    }
  }
}

std::vector<std::string> get_tokens(std::ifstream& input)
{
  std::string line;
  std::getline(input, line);
  std::istringstream iss(line);
  std::vector<std::string> tokens{
    std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
  return tokens;
}

void print_tokens(const std::vector<std::string>& tokens)
{
  std::cout << "Line:";
  for (const auto& token : tokens) {
    std::cout << " " << token;
  }
  std::cout << std::endl;
}

int get_int_from_token(const std::string& token, const char* filename, const int line)
{
  int value = 0;
  try {
    value = std::stoi(token);
  } catch (const std::exception& e) {
    throw std::invalid_argument(
      "invalid integer token '" + token + "' at " + filename + ":" + std::to_string(line));
  }
  return value;
}

double get_double_from_token(const std::string& token, const char* filename, const int line)
{
  double value = 0;
  try {
    value = std::stod(token);
  } catch (const std::exception& e) {
    throw std::invalid_argument(
      "invalid floating-point token '" + token + "' at " + filename + ":" + std::to_string(line));
  }
  return value;
}

int spin_descriptor_dim(const int compress, const int l_max, const bool chiral)
{
  int dim = 2 + 4 * compress;
  if (l_max >= 0) {
    dim += compress;
  }
  if (l_max >= 1) {
    dim += 3 * compress;
  }
  for (int ell = 2; ell <= l_max; ++ell) {
    dim += compress;
  }
  dim += compress;
  dim += compress;
  if (l_max >= 1) {
    dim += compress;
  }
  if (chiral) {
    dim += std::min(2, compress) + 2 * compress;
  }
  return dim;
}

double dot3(const std::array<double, 3>& a, const std::array<double, 3>& b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

std::array<double, 3> cross3(
  const std::array<double, 3>& a,
  const std::array<double, 3>& b)
{
  return {
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]};
}

int levi_civita(const int a, const int b, const int c)
{
  if (a == b || b == c || a == c) {
    return 0;
  }
  return ((a == 0 && b == 1 && c == 2) ||
          (a == 1 && b == 2 && c == 0) ||
          (a == 2 && b == 0 && c == 1)) ? 1 : -1;
}

std::array<double, 9> stf_outer3(const std::array<double, 3>& a, const std::array<double, 3>& b)
{
  std::array<double, 9> out;
  double trace = 0.0;
  for (int i = 0; i < 3; ++i) {
    trace += a[i] * b[i];
  }
  trace /= 3.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out[3 * i + j] = 0.5 * (a[i] * b[j] + a[j] * b[i]);
      if (i == j) {
        out[3 * i + j] -= trace;
      }
    }
  }
  return out;
}

constexpr int kSpinDeg2Count = 6;
constexpr int kSpinDeg3Count = 10;
constexpr int kSpinDeg4Count = 15;
constexpr int kSpinChiralOReducedCount = 7;
constexpr int kSpinChiralHReducedCount = 9;
constexpr int kSpinChiralQohReducedCount = 50;

constexpr unsigned short kSpinChiralQohReducedPacked[kSpinChiralQohReducedCount] = {
  0, 32, 34, 49, 68, 70, 87, 100, 257, 259,
  272, 274, 289, 291, 304, 306, 325, 327, 328, 340,
  342, 357, 517, 519, 532, 534, 549, 552, 564, 577,
  579, 592, 594, 611, 774, 791, 804, 824, 832, 849,
  851, 866, 1041, 1056, 1073, 1075, 1094, 1109, 1111, 1124};
constexpr double kSpinChiralQohReducedCoeff[kSpinChiralQohReducedCount] = {
  1.0, -2.0, 1.0, 2.0, 1.0, -1.0, 2.0, 0.5, 1.0, 1.0,
  1.0, 1.0, -4.0, -3.0, -4.0, -3.0, -0.5, -2.0, -1.0, -1.0,
  -4.0, -0.5, 1.0, -2.0, 1.0, 2.0, -1.0, -2.0, 1.0, -2.0,
  6.0, -4.0, -8.0, 2.0, 1.0, 1.0, -0.5, 1.0, 1.0, -2.0,
  -4.0, 1.0, 1.0, 2.0, -2.0, 1.0, 1.0, 1.0, -2.0, -0.5};

constexpr int kSpinDeg2Exp[kSpinDeg2Count][3] = {
  {2, 0, 0}, {0, 2, 0}, {0, 0, 2}, {1, 1, 0}, {1, 0, 1}, {0, 1, 1}};
constexpr int kSpinDeg3Exp[kSpinDeg3Count][3] = {
  {3, 0, 0}, {0, 3, 0}, {0, 0, 3}, {2, 1, 0}, {2, 0, 1},
  {1, 2, 0}, {0, 2, 1}, {1, 0, 2}, {0, 1, 2}, {1, 1, 1}};
constexpr int kSpinDeg4Exp[kSpinDeg4Count][3] = {
  {4, 0, 0}, {0, 4, 0}, {0, 0, 4}, {3, 1, 0}, {3, 0, 1},
  {1, 3, 0}, {0, 3, 1}, {1, 0, 3}, {0, 1, 3}, {2, 2, 0},
  {2, 0, 2}, {0, 2, 2}, {2, 1, 1}, {1, 2, 1}, {1, 1, 2}};

constexpr int kSpinPerm3[6][3] = {
  {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
constexpr int kSpinPerm4[24][4] = {
  {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1},
  {0, 3, 1, 2}, {0, 3, 2, 1}, {1, 0, 2, 3}, {1, 0, 3, 2},
  {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 0, 2}, {1, 3, 2, 0},
  {2, 0, 1, 3}, {2, 0, 3, 1}, {2, 1, 0, 3}, {2, 1, 3, 0},
  {2, 3, 0, 1}, {2, 3, 1, 0}, {3, 0, 1, 2}, {3, 0, 2, 1},
  {3, 1, 0, 2}, {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 2, 1, 0}};

int spin_tensor3_index(const int a, const int b, const int c)
{
  return (a * 3 + b) * 3 + c;
}

int spin_tensor4_index(const int a, const int b, const int c, const int d)
{
  return ((a * 3 + b) * 3 + c) * 3 + d;
}

int spin_monomial_index(const int degree, const int nx, const int ny, const int nz)
{
  const int (*exps)[3] = nullptr;
  int count = 0;
  if (degree == 2) {
    exps = kSpinDeg2Exp;
    count = kSpinDeg2Count;
  } else if (degree == 3) {
    exps = kSpinDeg3Exp;
    count = kSpinDeg3Count;
  } else {
    exps = kSpinDeg4Exp;
    count = kSpinDeg4Count;
  }
  for (int k = 0; k < count; ++k) {
    if (exps[k][0] == nx && exps[k][1] == ny && exps[k][2] == nz) {
      return k;
    }
  }
  return 0;
}

int spin_factorial(const int n)
{
  return n <= 1 ? 1 : (n == 2 ? 2 : (n == 3 ? 6 : 24));
}

int spin_monomial_multiplicity(const int degree, const int nx, const int ny, const int nz)
{
  return spin_factorial(degree) / (spin_factorial(nx) * spin_factorial(ny) * spin_factorial(nz));
}

void fill_spin_monomials(const std::array<double, 3>& u, double* m2, double* m3, double* m4)
{
  const double x = u[0];
  const double y = u[1];
  const double z = u[2];
  const double x2 = x * x;
  const double y2 = y * y;
  const double z2 = z * z;
  const double xy = x * y;
  const double xz = x * z;
  const double yz = y * z;
  const double x3 = x2 * x;
  const double y3 = y2 * y;
  const double z3 = z2 * z;
  m2[0] = x2;
  m2[1] = y2;
  m2[2] = z2;
  m2[3] = xy;
  m2[4] = xz;
  m2[5] = yz;
  m3[0] = x3;
  m3[1] = y3;
  m3[2] = z3;
  m3[3] = x2 * y;
  m3[4] = x2 * z;
  m3[5] = x * y2;
  m3[6] = y2 * z;
  m3[7] = x * z2;
  m3[8] = y * z2;
  m3[9] = xy * z;
  m4[0] = x2 * x2;
  m4[1] = y2 * y2;
  m4[2] = z2 * z2;
  m4[3] = x3 * y;
  m4[4] = x3 * z;
  m4[5] = x * y3;
  m4[6] = y3 * z;
  m4[7] = x * z3;
  m4[8] = y * z3;
  m4[9] = x2 * y2;
  m4[10] = x2 * z2;
  m4[11] = y2 * z2;
  m4[12] = x2 * yz;
  m4[13] = y2 * xz;
  m4[14] = z2 * xy;
}

void fill_spin_chiral_reduced_moments(
  const std::array<double, 3>& u,
  double* o_reduced,
  double* h_reduced)
{
  const double x = u[0];
  const double y = u[1];
  const double z = u[2];
  const double x2 = x * x;
  const double y2 = y * y;
  const double z2 = z * z;
  const double xy = x * y;
  const double xz = x * z;
  const double yz = y * z;

  o_reduced[0] = y * (y2 - 3.0 * z2);
  o_reduced[1] = z * (z2 - 3.0 * y2);
  o_reduced[2] = y * (x2 - z2);
  o_reduced[3] = z * (x2 - y2);
  o_reduced[4] = x * (y2 - z2);
  o_reduced[5] = xy * z;
  o_reduced[6] = x * (3.0 * z2 - x2);

  constexpr double one_seventh = 1.0 / 7.0;
  h_reduced[0] = xz * (-x2 - z2 + 6.0 * y2) * one_seventh;
  h_reduced[1] = xy * (x2 + y2 - 6.0 * z2) * one_seventh;
  h_reduced[2] = xz * (4.0 * x2 - 3.0 * z2 - 3.0 * y2) * one_seventh;
  h_reduced[3] = xy * (-4.0 * x2 + 3.0 * y2 + 3.0 * z2) * one_seventh;
  h_reduced[4] = yz * (-2.0 * y2 - 2.0 * z2 + 12.0 * x2) * one_seventh;
  h_reduced[5] = 2.0 * (y2 - z2) * (y2 + z2 - 6.0 * x2) * one_seventh;
  h_reduced[6] = yz * (4.0 * y2 - 3.0 * z2 - 3.0 * x2) * one_seventh;
  h_reduced[7] = (y2 - x2) * (x2 + y2 - 6.0 * z2) * one_seventh;
  h_reduced[8] =
    (4.0 * x2 * x2 + y2 * y2 + 2.0 * z2 * z2 - 9.0 * x2 * y2 -
     15.0 * x2 * z2 + 3.0 * y2 * z2) * one_seventh;
}

void fill_spin_chiral_q_reduced(const double* q, double* reduced)
{
  reduced[0] = q[0] - q[8];
  reduced[1] = -q[5];
  reduced[2] = 0.5 * q[2];
  reduced[3] = q[1];
  reduced[4] = q[0] - q[4];
}

double contract_spin_chiral_qoh_reduced(
  const double* q,
  const double* o,
  const double* h)
{
  double value = 0.0;
  for (int term = 0; term < kSpinChiralQohReducedCount; ++term) {
    const unsigned short packed = kSpinChiralQohReducedPacked[term];
    const int qi = packed >> 8;
    const int oi = (packed >> 4) & 0x0f;
    const int hi = packed & 0x0f;
    value += kSpinChiralQohReducedCoeff[term] * q[qi] * o[oi] * h[hi];
  }
  return value;
}

void add_spin_chiral_qoh_reduced_pull(
  const double pull,
  const double* q,
  const double* o,
  const double* h,
  double* grad_q,
  double* grad_o,
  double* grad_h)
{
  double grad_q_reduced[5] = {0.0};
  for (int term = 0; term < kSpinChiralQohReducedCount; ++term) {
    const unsigned short packed = kSpinChiralQohReducedPacked[term];
    const int qi = packed >> 8;
    const int oi = (packed >> 4) & 0x0f;
    const int hi = packed & 0x0f;
    const double scale = pull * kSpinChiralQohReducedCoeff[term];
    grad_q_reduced[qi] += scale * o[oi] * h[hi];
    grad_o[oi] += scale * q[qi] * h[hi];
    grad_h[hi] += scale * q[qi] * o[oi];
  }
  grad_q[0] += grad_q_reduced[0] + grad_q_reduced[4];
  grad_q[1] += grad_q_reduced[3];
  grad_q[2] += 0.5 * grad_q_reduced[2];
  grad_q[4] -= grad_q_reduced[4];
  grad_q[5] -= grad_q_reduced[1];
  grad_q[8] -= grad_q_reduced[0];
}

void add_spin_chiral_o_reduced_terms(const double* grad, double* terms)
{
  terms[0] -= grad[6];
  terms[1] += grad[0];
  terms[2] += grad[1];
  terms[3] += grad[2];
  terms[4] += grad[3];
  terms[5] += grad[4];
  terms[6] -= 3.0 * grad[1] + grad[3];
  terms[7] += -grad[4] + 3.0 * grad[6];
  terms[8] -= 3.0 * grad[0] + grad[2];
  terms[9] += grad[5];
}

void add_spin_chiral_h_reduced_terms(const double* grad, double* terms)
{
  constexpr double s = 1.0 / 7.0;
  terms[0] += (-grad[7] + 4.0 * grad[8]) * s;
  terms[1] += (2.0 * grad[5] + grad[7] + grad[8]) * s;
  terms[2] += (-2.0 * grad[5] + 2.0 * grad[8]) * s;
  terms[3] += (grad[1] - 4.0 * grad[3]) * s;
  terms[4] += (-grad[0] + 4.0 * grad[2]) * s;
  terms[5] += (grad[1] + 3.0 * grad[3]) * s;
  terms[6] += (-2.0 * grad[4] + 4.0 * grad[6]) * s;
  terms[7] += (-grad[0] - 3.0 * grad[2]) * s;
  terms[8] += (-2.0 * grad[4] - 3.0 * grad[6]) * s;
  terms[9] += (-12.0 * grad[5] - 9.0 * grad[8]) * s;
  terms[10] += (12.0 * grad[5] + 6.0 * grad[7] - 15.0 * grad[8]) * s;
  terms[11] += (-6.0 * grad[7] + 3.0 * grad[8]) * s;
  terms[12] += (12.0 * grad[4] - 3.0 * grad[6]) * s;
  terms[13] += (6.0 * grad[0] - 3.0 * grad[2]) * s;
  terms[14] += (-6.0 * grad[1] + 3.0 * grad[3]) * s;
}

double dot_spin_terms(const double* a, const double* b, const int count)
{
  double sum = 0.0;
  for (int k = 0; k < count; ++k) {
    sum += a[k] * b[k];
  }
  return sum;
}

void find_spin_basis3_and_derivatives(
  const double rcinv,
  const double d12,
  const double fc12,
  const double fcp12,
  double* fn,
  double* fnp)
{
  const double a = d12 * rcinv - 1.0;
  const double x = 2.0 * a * a - 1.0;
  const double dx_half = 2.0 * a * rcinv;
  const double t2 = 2.0 * x * x - 1.0;
  const double t3 = 2.0 * x * t2 - x;
  const double base0 = 1.0;
  const double base1 = 0.5 * (x + 1.0);
  const double base2 = 0.5 * (t2 + 1.0);
  const double base3 = 0.5 * (t3 + 1.0);
  fn[0] = base0 * fc12;
  fn[1] = base1 * fc12;
  fn[2] = base2 * fc12;
  fn[3] = base3 * fc12;
  fnp[0] = fcp12;
  fnp[1] = dx_half * fc12 + base1 * fcp12;
  fnp[2] = (4.0 * x * dx_half) * fc12 + base2 * fcp12;
  fnp[3] = (3.0 * (4.0 * x * x - 1.0) * dx_half) * fc12 + base3 * fcp12;
}

double spin_packed_value(const double* packed, const int degree, const int* counts)
{
  return packed[spin_monomial_index(degree, counts[0], counts[1], counts[2])];
}

void unpack_rank3_spin_stf_slow(const double* raw, double* out)
{
  double trace[3] = {0.0, 0.0, 0.0};
  for (int c = 0; c < 3; ++c) {
    for (int e = 0; e < 3; ++e) {
      int counts[3] = {0, 0, 0};
      counts[e] += 2;
      ++counts[c];
      trace[c] += spin_packed_value(raw, 3, counts);
    }
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 3; ++c) {
        int counts[3] = {0, 0, 0};
        ++counts[a];
        ++counts[b];
        ++counts[c];
        const double traced =
          ((a == b) ? trace[c] : 0.0) +
          ((a == c) ? trace[b] : 0.0) +
          ((b == c) ? trace[a] : 0.0);
        out[spin_tensor3_index(a, b, c)] = spin_packed_value(raw, 3, counts) - traced / 5.0;
      }
    }
  }
}

void unpack_rank4_spin_stf_slow(const double* raw, double* out)
{
  double trace[9] = {0.0};
  for (int c = 0; c < 3; ++c) {
    for (int d = 0; d < 3; ++d) {
      for (int e = 0; e < 3; ++e) {
        int counts[3] = {0, 0, 0};
        counts[e] += 2;
        ++counts[c];
        ++counts[d];
        trace[3 * c + d] += spin_packed_value(raw, 4, counts);
      }
    }
  }
  double double_trace = 0.0;
  for (int e = 0; e < 3; ++e) {
    double_trace += trace[3 * e + e];
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < 3; ++d) {
          int counts[3] = {0, 0, 0};
          ++counts[a];
          ++counts[b];
          ++counts[c];
          ++counts[d];
          const double six =
            ((a == b) ? trace[3 * c + d] : 0.0) +
            ((a == c) ? trace[3 * b + d] : 0.0) +
            ((a == d) ? trace[3 * b + c] : 0.0) +
            ((b == c) ? trace[3 * a + d] : 0.0) +
            ((b == d) ? trace[3 * a + c] : 0.0) +
            ((c == d) ? trace[3 * a + b] : 0.0);
          const double three =
            ((a == b && c == d) ? double_trace : 0.0) +
            ((a == c && b == d) ? double_trace : 0.0) +
            ((a == d && b == c) ? double_trace : 0.0);
          out[spin_tensor4_index(a, b, c, d)] =
            spin_packed_value(raw, 4, counts) - six / 7.0 + three / 35.0;
        }
      }
    }
  }
}

using SpinRank3UnpackTable = std::array<std::array<double, kSpinDeg3Count>, 27>;
using SpinRank4UnpackTable = std::array<std::array<double, kSpinDeg4Count>, 81>;

const SpinRank3UnpackTable& spin_rank3_unpack_table()
{
  static const SpinRank3UnpackTable table = [] {
    SpinRank3UnpackTable out{};
    double raw[kSpinDeg3Count] = {0.0};
    double unpacked[27] = {0.0};
    for (int k = 0; k < kSpinDeg3Count; ++k) {
      std::fill(raw, raw + kSpinDeg3Count, 0.0);
      raw[k] = 1.0;
      unpack_rank3_spin_stf_slow(raw, unpacked);
      for (int i = 0; i < 27; ++i) {
        out[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] = unpacked[i];
      }
    }
    return out;
  }();
  return table;
}

const SpinRank4UnpackTable& spin_rank4_unpack_table()
{
  static const SpinRank4UnpackTable table = [] {
    SpinRank4UnpackTable out{};
    double raw[kSpinDeg4Count] = {0.0};
    double unpacked[81] = {0.0};
    for (int k = 0; k < kSpinDeg4Count; ++k) {
      std::fill(raw, raw + kSpinDeg4Count, 0.0);
      raw[k] = 1.0;
      unpack_rank4_spin_stf_slow(raw, unpacked);
      for (int i = 0; i < 81; ++i) {
        out[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] = unpacked[i];
      }
    }
    return out;
  }();
  return table;
}

void unpack_rank3_spin_stf(const double* raw, double* out)
{
  const SpinRank3UnpackTable& table = spin_rank3_unpack_table();
  for (int i = 0; i < 27; ++i) {
    const auto& row = table[static_cast<std::size_t>(i)];
    double value = 0.0;
    for (int k = 0; k < kSpinDeg3Count; ++k) {
      value += row[static_cast<std::size_t>(k)] * raw[k];
    }
    out[i] = value;
  }
}

void unpack_rank4_spin_stf(const double* raw, double* out)
{
  const SpinRank4UnpackTable& table = spin_rank4_unpack_table();
  for (int i = 0; i < 81; ++i) {
    const auto& row = table[static_cast<std::size_t>(i)];
    double value = 0.0;
    for (int k = 0; k < kSpinDeg4Count; ++k) {
      value += row[static_cast<std::size_t>(k)] * raw[k];
    }
    out[i] = value;
  }
}

void project_rank2_spin_gradient(const double* grad, double* terms)
{
  const double trace = (grad[0] + grad[4] + grad[8]) / 3.0;
  terms[0] = grad[0] - trace;
  terms[1] = grad[4] - trace;
  terms[2] = grad[8] - trace;
  terms[3] = grad[1] + grad[3];
  terms[4] = grad[2] + grad[6];
  terms[5] = grad[5] + grad[7];
}

void fill_spin_term_derivatives(
  const int degree,
  const int count,
  const double* terms,
  double* derivatives)
{
  const int lower_count = degree == 3 ? kSpinDeg2Count : kSpinDeg3Count;
  std::fill(derivatives, derivatives + 3 * lower_count, 0.0);
  const int (*exps)[3] = degree == 3 ? kSpinDeg3Exp : kSpinDeg4Exp;
  for (int k = 0; k < count; ++k) {
    for (int axis = 0; axis < 3; ++axis) {
      const int power = exps[k][axis];
      if (power == 0) {
        continue;
      }
      int lower[3] = {exps[k][0], exps[k][1], exps[k][2]};
      --lower[axis];
      const int lower_index = spin_monomial_index(degree - 1, lower[0], lower[1], lower[2]);
      derivatives[axis * lower_count + lower_index] += power * terms[k];
    }
  }
}

void project_rank3_spin_gradient_slow(const double* grad, double* terms, double* derivatives)
{
  double sym[27];
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 3; ++c) {
        const int ids[3] = {a, b, c};
        double sum = 0.0;
        for (const auto& p : kSpinPerm3) {
          sum += grad[spin_tensor3_index(ids[p[0]], ids[p[1]], ids[p[2]])];
        }
        sym[spin_tensor3_index(a, b, c)] = sum / 6.0;
      }
    }
  }
  double trace[3] = {0.0, 0.0, 0.0};
  for (int c = 0; c < 3; ++c) {
    for (int e = 0; e < 3; ++e) {
      trace[c] += sym[spin_tensor3_index(e, e, c)];
    }
  }
  for (int k = 0; k < kSpinDeg3Count; ++k) {
    int ids[3];
    int n = 0;
    for (int axis = 0; axis < 3; ++axis) {
      for (int repeat = 0; repeat < kSpinDeg3Exp[k][axis]; ++repeat) {
        ids[n++] = axis;
      }
    }
    const int a = ids[0];
    const int b = ids[1];
    const int c = ids[2];
    const double traced =
      ((a == b) ? trace[c] : 0.0) +
      ((a == c) ? trace[b] : 0.0) +
      ((b == c) ? trace[a] : 0.0);
    const double projected = sym[spin_tensor3_index(a, b, c)] - traced / 5.0;
    terms[k] = spin_monomial_multiplicity(3, kSpinDeg3Exp[k][0], kSpinDeg3Exp[k][1], kSpinDeg3Exp[k][2]) * projected;
  }
  fill_spin_term_derivatives(3, kSpinDeg3Count, terms, derivatives);
}

void project_rank4_spin_gradient_slow(const double* grad, double* terms, double* derivatives)
{
  double sym[81];
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < 3; ++d) {
          const int ids[4] = {a, b, c, d};
          double sum = 0.0;
          for (const auto& p : kSpinPerm4) {
            sum += grad[spin_tensor4_index(ids[p[0]], ids[p[1]], ids[p[2]], ids[p[3]])];
          }
          sym[spin_tensor4_index(a, b, c, d)] = sum / 24.0;
        }
      }
    }
  }
  double trace[9] = {0.0};
  for (int c = 0; c < 3; ++c) {
    for (int d = 0; d < 3; ++d) {
      for (int e = 0; e < 3; ++e) {
        trace[3 * c + d] += sym[spin_tensor4_index(e, e, c, d)];
      }
    }
  }
  double double_trace = 0.0;
  for (int e = 0; e < 3; ++e) {
    double_trace += trace[3 * e + e];
  }
  for (int k = 0; k < kSpinDeg4Count; ++k) {
    int ids[4];
    int n = 0;
    for (int axis = 0; axis < 3; ++axis) {
      for (int repeat = 0; repeat < kSpinDeg4Exp[k][axis]; ++repeat) {
        ids[n++] = axis;
      }
    }
    const int a = ids[0];
    const int b = ids[1];
    const int c = ids[2];
    const int d = ids[3];
    const double six =
      ((a == b) ? trace[3 * c + d] : 0.0) +
      ((a == c) ? trace[3 * b + d] : 0.0) +
      ((a == d) ? trace[3 * b + c] : 0.0) +
      ((b == c) ? trace[3 * a + d] : 0.0) +
      ((b == d) ? trace[3 * a + c] : 0.0) +
      ((c == d) ? trace[3 * a + b] : 0.0);
    const double three =
      ((a == b && c == d) ? double_trace : 0.0) +
      ((a == c && b == d) ? double_trace : 0.0) +
      ((a == d && b == c) ? double_trace : 0.0);
    const double projected = sym[spin_tensor4_index(a, b, c, d)] - six / 7.0 + three / 35.0;
    terms[k] = spin_monomial_multiplicity(4, kSpinDeg4Exp[k][0], kSpinDeg4Exp[k][1], kSpinDeg4Exp[k][2]) * projected;
  }
  fill_spin_term_derivatives(4, kSpinDeg4Count, terms, derivatives);
}

using SpinRank3GradientTable = std::array<std::array<double, 27>, kSpinDeg3Count>;
using SpinRank4GradientTable = std::array<std::array<double, 81>, kSpinDeg4Count>;

const SpinRank3GradientTable& spin_rank3_gradient_table()
{
  static const SpinRank3GradientTable table = [] {
    SpinRank3GradientTable out{};
    double basis[27] = {0.0};
    double terms[kSpinDeg3Count] = {0.0};
    double derivatives[3 * kSpinDeg2Count] = {0.0};
    for (int input = 0; input < 27; ++input) {
      std::fill(basis, basis + 27, 0.0);
      basis[input] = 1.0;
      project_rank3_spin_gradient_slow(basis, terms, derivatives);
      for (int k = 0; k < kSpinDeg3Count; ++k) {
        out[static_cast<std::size_t>(k)][static_cast<std::size_t>(input)] = terms[k];
      }
    }
    return out;
  }();
  return table;
}

const SpinRank4GradientTable& spin_rank4_gradient_table()
{
  static const SpinRank4GradientTable table = [] {
    SpinRank4GradientTable out{};
    double basis[81] = {0.0};
    double terms[kSpinDeg4Count] = {0.0};
    double derivatives[3 * kSpinDeg3Count] = {0.0};
    for (int input = 0; input < 81; ++input) {
      std::fill(basis, basis + 81, 0.0);
      basis[input] = 1.0;
      project_rank4_spin_gradient_slow(basis, terms, derivatives);
      for (int k = 0; k < kSpinDeg4Count; ++k) {
        out[static_cast<std::size_t>(k)][static_cast<std::size_t>(input)] = terms[k];
      }
    }
    return out;
  }();
  return table;
}

void project_rank3_spin_gradient(const double* grad, double* terms, double* derivatives)
{
  const SpinRank3GradientTable& table = spin_rank3_gradient_table();
  for (int k = 0; k < kSpinDeg3Count; ++k) {
    const auto& row = table[static_cast<std::size_t>(k)];
    double sum = 0.0;
    for (int input = 0; input < 27; ++input) {
      sum += row[static_cast<std::size_t>(input)] * grad[input];
    }
    terms[k] = sum;
  }
  fill_spin_term_derivatives(3, kSpinDeg3Count, terms, derivatives);
}

void project_rank4_spin_gradient(const double* grad, double* terms, double* derivatives)
{
  const SpinRank4GradientTable& table = spin_rank4_gradient_table();
  for (int k = 0; k < kSpinDeg4Count; ++k) {
    const auto& row = table[static_cast<std::size_t>(k)];
    double sum = 0.0;
    for (int input = 0; input < 81; ++input) {
      sum += row[static_cast<std::size_t>(input)] * grad[input];
    }
    terms[k] = sum;
  }
  fill_spin_term_derivatives(4, kSpinDeg4Count, terms, derivatives);
}

int real_spherical_harmonics_spin(
  const std::array<double, 3>& rhat,
  const int ell,
  double* out)
{
  const double x = rhat[0];
  const double y = rhat[1];
  const double z = rhat[2];
  if (ell == 2) {
    out[0] = std::sqrt(15.0 / (4.0 * PI)) * x * y;
    out[1] = std::sqrt(15.0 / (4.0 * PI)) * y * z;
    out[2] = std::sqrt(5.0 / (16.0 * PI)) * (2.0 * z * z - x * x - y * y);
    out[3] = std::sqrt(15.0 / (4.0 * PI)) * x * z;
    out[4] = std::sqrt(15.0 / (16.0 * PI)) * (x * x - y * y);
    return 5;
  }
  if (ell == 3) {
    const double rho2 = x * x + y * y;
    out[0] = std::sqrt(35.0 / (32.0 * PI)) * y * (3.0 * x * x - y * y);
    out[1] = std::sqrt(105.0 / (4.0 * PI)) * x * y * z;
    out[2] = std::sqrt(21.0 / (32.0 * PI)) * y * (4.0 * z * z - rho2);
    out[3] = std::sqrt(7.0 / (16.0 * PI)) * z * (2.0 * z * z - 3.0 * rho2);
    out[4] = std::sqrt(21.0 / (32.0 * PI)) * x * (4.0 * z * z - rho2);
    out[5] = std::sqrt(105.0 / (16.0 * PI)) * z * (x * x - y * y);
    out[6] = std::sqrt(35.0 / (32.0 * PI)) * x * (x * x - 3.0 * y * y);
    return 7;
  }
  const double x2 = x * x;
  const double y2 = y * y;
  const double z2 = z * z;
  out[0] = 0.75 * std::sqrt(35.0 / PI) * x * y * (x2 - y2);
  out[1] = 0.75 * std::sqrt(35.0 / (2.0 * PI)) * y * z * (3.0 * x2 - y2);
  out[2] = 0.75 * std::sqrt(5.0 / PI) * x * y * (7.0 * z2 - 1.0);
  out[3] = 0.75 * std::sqrt(5.0 / (2.0 * PI)) * y * z * (7.0 * z2 - 3.0);
  out[4] = (3.0 / 16.0) * std::sqrt(1.0 / PI) * (35.0 * z2 * z2 - 30.0 * z2 + 3.0);
  out[5] = 0.75 * std::sqrt(5.0 / (2.0 * PI)) * x * z * (7.0 * z2 - 3.0);
  out[6] = 0.375 * std::sqrt(5.0 / PI) * (x2 - y2) * (7.0 * z2 - 1.0);
  out[7] = 0.75 * std::sqrt(35.0 / (2.0 * PI)) * x * z * (x2 - 3.0 * y2);
  out[8] = (3.0 / 16.0) * std::sqrt(35.0 / PI) * (x2 * x2 - 6.0 * x2 * y2 + y2 * y2);
  return 9;
}

template<int Width>
void add_density_fixed(
  std::vector<double>& density,
  const int C,
  const int atom,
  const double* values,
  const double* weights,
  const double weight_scale = 1.0)
{
  if (C == 4) {
    double* out0 = density.data() + static_cast<std::size_t>(atom) * 4 * Width;
    double* out1 = out0 + Width;
    double* out2 = out1 + Width;
    double* out3 = out2 + Width;
    const double w0 = weight_scale * weights[0];
    const double w1 = weight_scale * weights[1];
    const double w2 = weight_scale * weights[2];
    const double w3 = weight_scale * weights[3];
    for (int k = 0; k < Width; ++k) {
      const double v = values[k];
      out0[k] += w0 * v;
      out1[k] += w1 * v;
      out2[k] += w2 * v;
      out3[k] += w3 * v;
    }
    return;
  }
  for (int c = 0; c < C; ++c) {
    double* out = density.data() + (static_cast<std::size_t>(atom) * C + c) * Width;
    const double weight = weight_scale * weights[c];
    for (int k = 0; k < Width; ++k) {
      out[k] += weight * values[k];
    }
  }
}

void resize_and_zero(std::vector<double>& values, const std::size_t size)
{
  values.resize(size);
#if defined(_OPENMP)
  if (size >= 262144 && omp_get_max_threads() > 1) {
    double* data = values.data();
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(size); ++i) {
      data[static_cast<std::size_t>(i)] = 0.0;
    }
    return;
  }
#endif
  std::fill(values.begin(), values.end(), 0.0);
}

void add_lammps_spin_virial(
  const std::array<double, 3>& rhat,
  const double dist,
  const std::array<double, 3>& grad_rij,
  double* total_virial,
  double* virial,
  const int stride,
  const int row)
{
  const double rx = rhat[0] * dist;
  const double ry = rhat[1] * dist;
  const double rz = rhat[2] * dist;
  const double v00 = -rx * grad_rij[0];
  const double v01 = -rx * grad_rij[1];
  const double v02 = -rx * grad_rij[2];
  const double v11 = -ry * grad_rij[1];
  const double v12 = -ry * grad_rij[2];
  const double v22 = -rz * grad_rij[2];
  const double v10 = -ry * grad_rij[0];
  const double v20 = -rz * grad_rij[0];
  const double v21 = -rz * grad_rij[1];
  total_virial[0] += v00;
  total_virial[1] += v11;
  total_virial[2] += v22;
  total_virial[3] += 0.5 * (v01 + v10);
  total_virial[4] += 0.5 * (v02 + v20);
  total_virial[5] += 0.5 * (v12 + v21);
  if (!virial) {
    return;
  }
  virial[lammps_virial_index(row, 0, stride)] += v00;
  virial[lammps_virial_index(row, 1, stride)] += v11;
  virial[lammps_virial_index(row, 2, stride)] += v22;
  virial[lammps_virial_index(row, 3, stride)] += v01;
  virial[lammps_virial_index(row, 4, stride)] += v02;
  virial[lammps_virial_index(row, 5, stride)] += v12;
  virial[lammps_virial_index(row, 6, stride)] += v10;
  virial[lammps_virial_index(row, 7, stride)] += v20;
  virial[lammps_virial_index(row, 8, stride)] += v21;
}

void add_lammps_spin_total_virial(
  const std::array<double, 3>& rhat,
  const double dist,
  const std::array<double, 3>& grad_rij,
  double* total_virial)
{
  const double rx = rhat[0] * dist;
  const double ry = rhat[1] * dist;
  const double rz = rhat[2] * dist;
  total_virial[0] -= rx * grad_rij[0];
  total_virial[1] -= ry * grad_rij[1];
  total_virial[2] -= rz * grad_rij[2];
  total_virial[3] -= 0.5 * (rx * grad_rij[1] + ry * grad_rij[0]);
  total_virial[4] -= 0.5 * (rx * grad_rij[2] + rz * grad_rij[0]);
  total_virial[5] -= 0.5 * (ry * grad_rij[2] + rz * grad_rij[1]);
}

constexpr int MAX_SPIN_COMPRESS = 4;

using SpinEdge = NEP::SpinEdge;
using SpinCache = NEP::SpinCache;

template <bool SpinsAos3>
bool add_spin_chiral_gradient_lammps_single_center_deferred(
  const NEP::ParaMB& paramb,
  const NEP::ANN& annmb,
  const int N,
  const double* spins,
  const SpinCache& cache,
  const double* Fp,
  LammpsThreadLocalScratchView* lammps_scratch,
  std::vector<double>& edge_grad_weight,
  std::vector<double>& edge_grad_rhat)
{
  if (!paramb.spin_chiral || cache.edges.empty() ||
      !lammps_spin_scratch_active(lammps_scratch) ||
      cache.edge_offsets.size() != 2) {
    return false;
  }

  const int C = paramb.spin_compress;
  const int chiC = std::min(2, C);
  const int base_offset = spin_descriptor_dim(C, paramb.spin_l_max, false);
  const int offset0 = paramb.struct_dim;
  const int force_stride = lammps_scratch->force_rows;
  const int begin = cache.edge_offsets[0];
  const int end = cache.edge_offsets[1];
  const int count = end - begin;
  double* local_grad_spin = lammps_scratch->mforce_private;

  auto fp = [&](const int atom, const int dim) {
    return Fp[static_cast<std::size_t>(atom) * annmb.dim + offset0 + dim];
  };
  auto spin = [&](const int atom, const int d) {
    if constexpr (SpinsAos3) {
      (void)N;
      return spins[static_cast<std::size_t>(atom) * 3 + d];
    } else {
      return spins[static_cast<std::size_t>(d) * N + atom];
    }
  };
  auto blockC = [&](const std::vector<double>& v, const int c, const int width) {
    return v.data() + static_cast<std::size_t>(c) * width;
  };
  auto blockChi = [&](const std::vector<double>& v, const int c, const int width) {
    return v.data() + static_cast<std::size_t>(c) * width;
  };
  auto add_lammps_spin_pull = [&](const int atom, const std::array<double, 3>& g) {
    for (int d = 0; d < 3; ++d) {
      local_grad_spin[lammps_vector_index(atom, d, force_stride)] += g[d];
    }
  };

  edge_grad_weight.resize(static_cast<std::size_t>(count) * C);
  edge_grad_rhat.resize(static_cast<std::size_t>(count) * 3);
  std::fill(edge_grad_weight.begin(), edge_grad_weight.end(), 0.0);
  std::fill(edge_grad_rhat.begin(), edge_grad_rhat.end(), 0.0);

  std::array<double, MAX_SPIN_COMPRESS> grad_chi;
  std::array<double, MAX_SPIN_COMPRESS * 3> grad_polar;
  std::array<double, MAX_SPIN_COMPRESS * 9> grad_pseudodev;
  std::array<double, MAX_SPIN_COMPRESS * 9> grad_Q;
  std::array<double, MAX_SPIN_COMPRESS * kSpinChiralOReducedCount> grad_O_reduced;
  std::array<double, MAX_SPIN_COMPRESS * kSpinChiralHReducedCount> grad_H_reduced;
  std::array<double, MAX_SPIN_COMPRESS * kSpinDeg2Count> grad_Q_terms;
  std::array<double, MAX_SPIN_COMPRESS * kSpinDeg3Count> grad_O_terms;
  std::array<double, MAX_SPIN_COMPRESS * 3 * kSpinDeg2Count> grad_O_derivatives;
  std::array<double, MAX_SPIN_COMPRESS * kSpinDeg4Count> grad_H_terms;
  std::array<double, MAX_SPIN_COMPRESS * 3 * kSpinDeg3Count> grad_H_derivatives;
  grad_chi.fill(0.0);
  grad_polar.fill(0.0);
  grad_pseudodev.fill(0.0);
  grad_Q.fill(0.0);
  grad_O_reduced.fill(0.0);
  grad_H_reduced.fill(0.0);

  auto local_gw = [&](const int edge, const int c) -> double& {
    return edge_grad_weight[static_cast<std::size_t>(edge) * C + c];
  };
  auto local_gr = [&](const int edge, const int d) -> double& {
    return edge_grad_rhat[static_cast<std::size_t>(edge) * 3 + d];
  };

  for (int e = begin; e < end; ++e) {
    const int le = e - begin;
    const SpinEdge& edge = cache.edges[static_cast<std::size_t>(e)];
    const std::array<double, 3> si = {spin(edge.i, 0), spin(edge.i, 1), spin(edge.i, 2)};
    const std::array<double, 3> sj = {spin(edge.j, 0), spin(edge.j, 1), spin(edge.j, 2)};
    const std::array<double, 3> x = cross3(si, sj);
    std::array<double, 3> gx = {0.0, 0.0, 0.0};
    std::array<double, 3> gu = {0.0, 0.0, 0.0};
    const double xu = dot3(x, edge.rhat);
    for (int c = 0; c < chiC; ++c) {
      const double alpha = fp(edge.i, base_offset + c);
      const double chi = cache.chirals[static_cast<std::size_t>(c)];
      const double aw = alpha * edge.weights[c];
      local_gw(le, c) += alpha * xu * chi;
      grad_chi[c] += aw * xu;
      for (int d = 0; d < 3; ++d) {
        gx[d] += aw * chi * edge.rhat[d];
        gu[d] += aw * chi * x[d];
      }
    }
    int chiral_offset = base_offset + chiC;
    for (int c = 0; c < C; ++c) {
      const double alpha = fp(edge.i, chiral_offset + c);
      const double aw = alpha * edge.weights[c];
      const double* p = blockC(cache.polars, c, 3);
      const std::array<double, 3> polar = {p[0], p[1], p[2]};
      const std::array<double, 3> axis = cross3(polar, edge.rhat);
      local_gw(le, c) += alpha * dot3(x, axis);
      std::array<double, 3> gaxis = {0.0, 0.0, 0.0};
      for (int d = 0; d < 3; ++d) {
        gx[d] += aw * axis[d];
        gaxis[d] = aw * x[d];
      }
      const std::array<double, 3> gp = cross3(edge.rhat, gaxis);
      const std::array<double, 3> gu_part = cross3(gaxis, polar);
      double* grad_p = grad_polar.data() + static_cast<std::size_t>(c) * 3;
      for (int d = 0; d < 3; ++d) {
        grad_p[d] += gp[d];
        gu[d] += gu_part[d];
      }
    }
    chiral_offset += C;
    for (int c = 0; c < C; ++c) {
      const double alpha = fp(edge.i, chiral_offset + c);
      const double aw = alpha * edge.weights[c];
      const double* P = blockC(cache.pseudodevs, c, 9);
      std::array<double, 3> axis = {0.0, 0.0, 0.0};
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          axis[a] += P[3 * a + b] * edge.rhat[b];
        }
      }
      local_gw(le, c) += alpha * dot3(x, axis);
      std::array<double, 3> gaxis = {0.0, 0.0, 0.0};
      for (int d = 0; d < 3; ++d) {
        gx[d] += aw * axis[d];
        gaxis[d] = aw * x[d];
      }
      double* gP = grad_pseudodev.data() + static_cast<std::size_t>(c) * 9;
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          gP[3 * a + b] += gaxis[a] * edge.rhat[b];
          gu[b] += P[3 * a + b] * gaxis[a];
        }
      }
    }
    const std::array<double, 3> gsi = cross3(sj, gx);
    const std::array<double, 3> gsj = cross3(gx, si);
    add_lammps_spin_pull(edge.i, gsi);
    add_lammps_spin_pull(edge.j, gsj);
    add_spin_transfer_row_major9(
      edge.rhat, edge.dist, gsj, lammps_scratch->spin_transfer_private,
      force_stride, edge.j);
    for (int d = 0; d < 3; ++d) {
      local_gr(le, d) += gu[d];
    }
  }

  for (int c = 0; c < chiC; ++c) {
    const double g = grad_chi[c];
    if (g == 0.0) {
      continue;
    }
    const double* Q = blockC(cache.geom, c, 9);
    const double* O = blockChi(cache.octupoles, c, kSpinChiralOReducedCount);
    const double* H = blockChi(cache.hexadecapoles, c, kSpinChiralHReducedCount);
    double* gQ = grad_Q.data() + static_cast<std::size_t>(c) * 9;
    double* gO = grad_O_reduced.data() +
      static_cast<std::size_t>(c) * kSpinChiralOReducedCount;
    double* gH = grad_H_reduced.data() +
      static_cast<std::size_t>(c) * kSpinChiralHReducedCount;
    double Q_reduced[5];
    fill_spin_chiral_q_reduced(Q, Q_reduced);
    add_spin_chiral_qoh_reduced_pull(g, Q_reduced, O, H, gQ, gO, gH);
  }

  for (int e = begin; e < end; ++e) {
    const int le = e - begin;
    const SpinEdge& edge = cache.edges[static_cast<std::size_t>(e)];
    for (int c = 0; c < C; ++c) {
      const double* gp = grad_polar.data() + static_cast<std::size_t>(c) * 3;
      local_gw(le, c) += gp[0] * edge.rhat[0] + gp[1] * edge.rhat[1] + gp[2] * edge.rhat[2];
      for (int d = 0; d < 3; ++d) {
        local_gr(le, d) += edge.weights[c] * gp[d];
      }

      const double* gPd = grad_pseudodev.data() + static_cast<std::size_t>(c) * 9;
      const double* Q = blockC(cache.geom, c, 9);
      std::array<double, 3> Qu = {0.0, 0.0, 0.0};
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          Qu[a] += Q[3 * a + b] * edge.rhat[b];
        }
      }
      const std::array<double, 3> pseudo_axis = cross3(edge.rhat, Qu);
      const double pseudo00 = pseudo_axis[0] * edge.rhat[0];
      const double pseudo01 =
        0.5 * (pseudo_axis[0] * edge.rhat[1] + pseudo_axis[1] * edge.rhat[0]);
      const double pseudo02 =
        0.5 * (pseudo_axis[0] * edge.rhat[2] + pseudo_axis[2] * edge.rhat[0]);
      const double pseudo10 = pseudo01;
      const double pseudo11 = pseudo_axis[1] * edge.rhat[1];
      const double pseudo12 =
        0.5 * (pseudo_axis[1] * edge.rhat[2] + pseudo_axis[2] * edge.rhat[1]);
      const double pseudo20 = pseudo02;
      const double pseudo21 = pseudo12;
      const double pseudo22 = pseudo_axis[2] * edge.rhat[2];
      local_gw(le, c) +=
        gPd[0] * pseudo00 + gPd[1] * pseudo01 + gPd[2] * pseudo02 +
        gPd[3] * pseudo10 + gPd[4] * pseudo11 + gPd[5] * pseudo12 +
        gPd[6] * pseudo20 + gPd[7] * pseudo21 + gPd[8] * pseudo22;
      const double s01 = 0.5 * (gPd[1] + gPd[3]);
      const double s02 = 0.5 * (gPd[2] + gPd[6]);
      const double s12 = 0.5 * (gPd[5] + gPd[7]);
      const double w = edge.weights[c];
      const std::array<double, 3> g_axis = {
        w * (gPd[0] * edge.rhat[0] + s01 * edge.rhat[1] + s02 * edge.rhat[2]),
        w * (s01 * edge.rhat[0] + gPd[4] * edge.rhat[1] + s12 * edge.rhat[2]),
        w * (s02 * edge.rhat[0] + s12 * edge.rhat[1] + gPd[8] * edge.rhat[2])};
      const std::array<double, 3> gu2 = {
        w * (gPd[0] * pseudo_axis[0] + s01 * pseudo_axis[1] + s02 * pseudo_axis[2]),
        w * (s01 * pseudo_axis[0] + gPd[4] * pseudo_axis[1] + s12 * pseudo_axis[2]),
        w * (s02 * pseudo_axis[0] + s12 * pseudo_axis[1] + gPd[8] * pseudo_axis[2])};
      const std::array<double, 3> g_u_cross = cross3(Qu, g_axis);
      const std::array<double, 3> g_Qu = cross3(g_axis, edge.rhat);
      double* gQ = grad_Q.data() + static_cast<std::size_t>(c) * 9;
      for (int a = 0; a < 3; ++a) {
        local_gr(le, a) += gu2[a] + g_u_cross[a];
        for (int b = 0; b < 3; ++b) {
          gQ[3 * a + b] += g_Qu[a] * edge.rhat[b];
          local_gr(le, b) += Q[3 * a + b] * g_Qu[a];
        }
      }
    }
  }

  for (int c = 0; c < C; ++c) {
    project_rank2_spin_gradient(
      grad_Q.data() + static_cast<std::size_t>(c) * 9,
      grad_Q_terms.data() + static_cast<std::size_t>(c) * kSpinDeg2Count);
  }
  for (int c = 0; c < chiC; ++c) {
    double* o_terms = grad_O_terms.data() + static_cast<std::size_t>(c) * kSpinDeg3Count;
    double* h_terms = grad_H_terms.data() + static_cast<std::size_t>(c) * kSpinDeg4Count;
    std::fill(o_terms, o_terms + kSpinDeg3Count, 0.0);
    std::fill(h_terms, h_terms + kSpinDeg4Count, 0.0);
    add_spin_chiral_o_reduced_terms(
      grad_O_reduced.data() + static_cast<std::size_t>(c) * kSpinChiralOReducedCount,
      o_terms);
    add_spin_chiral_h_reduced_terms(
      grad_H_reduced.data() + static_cast<std::size_t>(c) * kSpinChiralHReducedCount,
      h_terms);
    fill_spin_term_derivatives(
      3, kSpinDeg3Count, o_terms,
      grad_O_derivatives.data() + static_cast<std::size_t>(c) * 3 * kSpinDeg2Count);
    fill_spin_term_derivatives(
      4, kSpinDeg4Count, h_terms,
      grad_H_derivatives.data() + static_cast<std::size_t>(c) * 3 * kSpinDeg3Count);
  }

  for (int e = begin; e < end; ++e) {
    const int le = e - begin;
    const SpinEdge& edge = cache.edges[static_cast<std::size_t>(e)];
    double m2[kSpinDeg2Count];
    double m3[kSpinDeg3Count];
    double m4[kSpinDeg4Count];
    fill_spin_monomials(edge.rhat, m2, m3, m4);
    for (int c = 0; c < C; ++c) {
      const double* q_terms = grad_Q_terms.data() + static_cast<std::size_t>(c) * kSpinDeg2Count;
      local_gw(le, c) += dot_spin_terms(q_terms, m2, kSpinDeg2Count);
      const double w = edge.weights[c];
      local_gr(le, 0) += w * (2.0 * q_terms[0] * edge.rhat[0] + q_terms[3] * edge.rhat[1] + q_terms[4] * edge.rhat[2]);
      local_gr(le, 1) += w * (2.0 * q_terms[1] * edge.rhat[1] + q_terms[3] * edge.rhat[0] + q_terms[5] * edge.rhat[2]);
      local_gr(le, 2) += w * (2.0 * q_terms[2] * edge.rhat[2] + q_terms[4] * edge.rhat[0] + q_terms[5] * edge.rhat[1]);
    }
    for (int c = 0; c < chiC; ++c) {
      const double* o_terms = grad_O_terms.data() + static_cast<std::size_t>(c) * kSpinDeg3Count;
      const double* o_derivatives =
        grad_O_derivatives.data() + static_cast<std::size_t>(c) * 3 * kSpinDeg2Count;
      const double* h_terms = grad_H_terms.data() + static_cast<std::size_t>(c) * kSpinDeg4Count;
      const double* h_derivatives =
        grad_H_derivatives.data() + static_cast<std::size_t>(c) * 3 * kSpinDeg3Count;
      local_gw(le, c) += dot_spin_terms(o_terms, m3, kSpinDeg3Count) +
                         dot_spin_terms(h_terms, m4, kSpinDeg4Count);
      for (int d = 0; d < 3; ++d) {
        local_gr(le, d) += edge.weights[c] * (
          dot_spin_terms(o_derivatives + d * kSpinDeg2Count, m2, kSpinDeg2Count) +
          dot_spin_terms(h_derivatives + d * kSpinDeg3Count, m3, kSpinDeg3Count));
      }
    }
  }

  return true;
}

struct SpinPhaseBreakdown {
  double setup = 0.0;
  double edges = 0.0;
  double unpack = 0.0;
  double merge = 0.0;
  double contract = 0.0;
  double chiral = 0.0;
  double copy = 0.0;
  double ann = 0.0;
  double gradient_nonchiral = 0.0;
  double gradient_chiral = 0.0;
};

void add_spin_phase_breakdown(SpinPhaseBreakdown& dst, const SpinPhaseBreakdown& src)
{
  dst.setup += src.setup;
  dst.edges += src.edges;
  dst.unpack += src.unpack;
  dst.merge += src.merge;
  dst.contract += src.contract;
  dst.chiral += src.chiral;
  dst.copy += src.copy;
  dst.ann += src.ann;
  dst.gradient_nonchiral += src.gradient_nonchiral;
  dst.gradient_chiral += src.gradient_chiral;
}

void resize_spin_center_cache(
  SpinCache& cache,
  const int C,
  const int chiC,
  const bool chiral)
{
  cache.edge_offsets.resize(2);
  cache.rho0.resize(static_cast<std::size_t>(C) * 3);
  cache.raw1.resize(static_cast<std::size_t>(C) * 9);
  cache.l1_rdot.resize(static_cast<std::size_t>(C));
  cache.l1_cross.resize(static_cast<std::size_t>(C) * 3);
  cache.l1_stf.resize(static_cast<std::size_t>(C) * 9);
  cache.angular2.resize(static_cast<std::size_t>(C) * 15);
  cache.angular3.resize(static_cast<std::size_t>(C) * 21);
  cache.angular4.resize(static_cast<std::size_t>(C) * 27);
  cache.geom.resize(static_cast<std::size_t>(C) * 9);
  cache.rho0_dot.resize(static_cast<std::size_t>(C) * 3);
  cache.raw1_dot.resize(static_cast<std::size_t>(C) * 9);
  if (chiral) {
    cache.polars.resize(static_cast<std::size_t>(C) * 3);
    cache.octupoles.resize(static_cast<std::size_t>(chiC) * kSpinChiralOReducedCount);
    cache.hexadecapoles.resize(static_cast<std::size_t>(chiC) * kSpinChiralHReducedCount);
    cache.chirals.resize(static_cast<std::size_t>(chiC));
    cache.pseudodevs.resize(static_cast<std::size_t>(C) * 9);
  }
}

void clear_spin_center_cache(
  SpinCache& cache,
  const bool chiral,
  const int edge_capacity)
{
  std::fill(cache.rho0.begin(), cache.rho0.end(), 0.0);
  std::fill(cache.raw1.begin(), cache.raw1.end(), 0.0);
  std::fill(cache.l1_rdot.begin(), cache.l1_rdot.end(), 0.0);
  std::fill(cache.l1_cross.begin(), cache.l1_cross.end(), 0.0);
  std::fill(cache.l1_stf.begin(), cache.l1_stf.end(), 0.0);
  std::fill(cache.angular2.begin(), cache.angular2.end(), 0.0);
  std::fill(cache.angular3.begin(), cache.angular3.end(), 0.0);
  std::fill(cache.angular4.begin(), cache.angular4.end(), 0.0);
  std::fill(cache.geom.begin(), cache.geom.end(), 0.0);
  std::fill(cache.rho0_dot.begin(), cache.rho0_dot.end(), 0.0);
  std::fill(cache.raw1_dot.begin(), cache.raw1_dot.end(), 0.0);
  if (chiral) {
    std::fill(cache.polars.begin(), cache.polars.end(), 0.0);
    std::fill(cache.octupoles.begin(), cache.octupoles.end(), 0.0);
    std::fill(cache.hexadecapoles.begin(), cache.hexadecapoles.end(), 0.0);
    std::fill(cache.chirals.begin(), cache.chirals.end(), 0.0);
    std::fill(cache.pseudodevs.begin(), cache.pseudodevs.end(), 0.0);
  }
  cache.edges.clear();
  cache.edges.reserve(static_cast<std::size_t>(edge_capacity));
  cache.edge_offsets[0] = 0;
}

void clear_spin_cache(SpinCache& cache)
{
  cache.edge_offsets.clear();
  cache.rho0.clear();
  cache.raw1.clear();
  cache.l1_rdot.clear();
  cache.l1_cross.clear();
  cache.l1_stf.clear();
  cache.angular2.clear();
  cache.angular3.clear();
  cache.angular4.clear();
  cache.geom.clear();
  cache.polars.clear();
  cache.octupoles.clear();
  cache.hexadecapoles.clear();
  cache.chirals.clear();
  cache.pseudodevs.clear();
  cache.rho0_dot.clear();
  cache.raw1_dot.clear();
}

void fill_spin_descriptor(
  const NEP::ParaMB& paramb,
  const NEP::ANN& annmb,
  const int N,
  const int* NN,
  const int* NL,
  const int* type,
  const double* x12,
  const double* y12,
  const double* z12,
  const double* spins,
  double* descriptor_soa,
  SpinCache* cache_out = nullptr,
  SpinPhaseBreakdown* phase = nullptr,
  const int center_count = 0,
  const int* centers = nullptr,
  int** lammps_NL = nullptr,
  double** lammps_pos = nullptr,
  double* center_descriptor_aos = nullptr,
  const LammpsRadialEdgeCacheView* lammps_radial_cache = nullptr)
{
  auto phase_mark = NepPhaseClock::now();
  auto add_phase = [&](double SpinPhaseBreakdown::*slot) {
    if (phase) {
      phase->*slot += nep_phase_elapsed(phase_mark);
    }
  };
  const int C = paramb.spin_compress;
  const int B = paramb.spin_basis_size + 1;
  const int l_max = paramb.spin_l_max;
  const int offset0 = paramb.struct_dim;
  std::unique_ptr<double[]> q_storage(new double[static_cast<std::size_t>(N) * paramb.spin_dim]);
  double* q = q_storage.get();
  SpinCache local_cache;
  SpinCache& cache = cache_out ? *cache_out : local_cache;
  clear_spin_cache(cache);

  auto spin = [&](const int atom, const int component) -> double {
    return spins[static_cast<std::size_t>(component) * N + atom];
  };
  auto mask_all_active = [](const std::vector<int>& mask) -> bool {
    return mask.empty() || std::all_of(mask.begin(), mask.end(), [](const int value) {
      return value != 0;
    });
  };
  const bool spin_dof_all_active = mask_all_active(paramb.spin_dof_type_active);
  const bool spin_env_all_active = mask_all_active(paramb.spin_env_type_active);
  auto active = [&](const std::vector<int>& mask, const bool all_active, const int t) -> bool {
    return all_active || mask[static_cast<std::size_t>(t)] != 0;
  };
  const bool use_lammps_edges = centers && lammps_NL && lammps_pos;
  const int loop_count = use_lammps_edges ? center_count : N;
  auto center_atom = [&](const int idx) {
    return use_lammps_edges ? centers[idx] : idx;
  };
  const int descriptor_loop_count = use_lammps_edges ? loop_count : N;
  const int density_rows = descriptor_loop_count;
  auto descriptor_atom = [&](const int idx) {
    return use_lammps_edges ? center_atom(idx) : idx;
  };
  bool center_slots_identity = true;
  if (use_lammps_edges) {
    for (int idx = 0; idx < loop_count; ++idx) {
      if (center_atom(idx) != idx) {
        center_slots_identity = false;
        break;
      }
    }
  }
  std::vector<int> center_slots;
  if (use_lammps_edges && !center_slots_identity) {
    center_slots.assign(static_cast<std::size_t>(N), -1);
    for (int idx = 0; idx < loop_count; ++idx) {
      center_slots[static_cast<std::size_t>(center_atom(idx))] = idx;
    }
  }
  auto qref = [&](const int atom, const int dim) -> double& {
    const int row = use_lammps_edges && !center_slots_identity
      ? center_slots[static_cast<std::size_t>(atom)]
      : atom;
    return q[static_cast<std::size_t>(row) * paramb.spin_dim + dim];
  };
  bool radial_cache_covers_spin = true;
  bool radial_cache_within_spin = true;
  for (double rc : paramb.rc_radial_pair) {
    radial_cache_covers_spin = radial_cache_covers_spin &&
      rc + 1.0e-12 >= paramb.spin_cutoff_radial;
    radial_cache_within_spin = radial_cache_within_spin &&
      rc <= paramb.spin_cutoff_radial + 1.0e-12;
  }
  const bool use_lammps_radial_geometry_cache =
    use_lammps_edges && radial_cache_covers_spin &&
    lammps_radial_edge_cache_active(lammps_radial_cache) &&
    lammps_radial_cache->num_centers == loop_count;
  auto load_edge_geometry = [&](
    const int center_idx,
    const int i,
    const int n,
    int& j,
    double& dx,
    double& dy,
    double& dz,
    double& d) {
    if (use_lammps_radial_geometry_cache) {
      const int edge_index = lammps_radial_cache->offsets[center_idx] + n;
      j = lammps_radial_cache->neighbors[edge_index];
      if (j < 0) {
        return false;
      }
      dx = lammps_radial_cache->x12[edge_index];
      dy = lammps_radial_cache->y12[edge_index];
      dz = lammps_radial_cache->z12[edge_index];
      d = lammps_radial_cache->d12[edge_index];
      return true;
    }
    const int index = n * N + i;
    j = use_lammps_edges ? lammps_NL[i][n] : NL[index];
    dx = use_lammps_edges ? lammps_pos[j][0] - lammps_pos[i][0] : x12[index];
    dy = use_lammps_edges ? lammps_pos[j][1] - lammps_pos[i][1] : y12[index];
    dz = use_lammps_edges ? lammps_pos[j][2] - lammps_pos[i][2] : z12[index];
    d = std::sqrt(dx * dx + dy * dy + dz * dz);
    return true;
  };

  for (int idx = 0; idx < descriptor_loop_count; ++idx) {
    const int atom = descriptor_atom(idx);
    const bool dof = active(paramb.spin_dof_type_active, spin_dof_all_active, type[atom]);
    const double sx = spin(atom, 0);
    const double sy = spin(atom, 1);
    const double sz = spin(atom, 2);
    const double s2 = sx * sx + sy * sy + sz * sz;
    qref(atom, 0) = dof ? s2 : 0.0;
    qref(atom, 1) = dof ? s2 * s2 : 0.0;
    for (int d = 2; d < 2 + 4 * C; ++d) {
      qref(atom, d) = 0.0;
    }
    if (paramb.spin_chiral) {
      const int chiral_offset = spin_descriptor_dim(C, l_max, false);
      for (int d = chiral_offset; d < paramb.spin_dim; ++d) {
        qref(atom, d) = 0.0;
      }
    }
  }

	  cache.rho0.assign(static_cast<std::size_t>(density_rows) * C * 3, 0.0);
	  cache.raw1.assign(static_cast<std::size_t>(density_rows) * C * 9, 0.0);
	  cache.l1_rdot.assign(static_cast<std::size_t>(density_rows) * C, 0.0);
	  cache.l1_cross.assign(static_cast<std::size_t>(density_rows) * C * 3, 0.0);
	  cache.l1_stf.assign(static_cast<std::size_t>(density_rows) * C * 9, 0.0);
	  cache.angular2.assign(static_cast<std::size_t>(density_rows) * C * 15, 0.0);
	  cache.angular3.assign(static_cast<std::size_t>(density_rows) * C * 21, 0.0);
	  cache.angular4.assign(static_cast<std::size_t>(density_rows) * C * 27, 0.0);
	  cache.geom.assign(static_cast<std::size_t>(density_rows) * C * 9, 0.0);
	  cache.rho0_dot.assign(static_cast<std::size_t>(density_rows) * C * 3, 0.0);
	  cache.raw1_dot.assign(static_cast<std::size_t>(density_rows) * C * 9, 0.0);
  std::vector<double>& rho0 = cache.rho0;
  std::vector<double>& raw1 = cache.raw1;
  std::vector<double>& l1_rdot = cache.l1_rdot;
  std::vector<double>& l1_cross = cache.l1_cross;
  std::vector<double>& l1_stf = cache.l1_stf;
  std::vector<double>& angular2 = cache.angular2;
  std::vector<double>& angular3 = cache.angular3;
  std::vector<double>& angular4 = cache.angular4;
  std::vector<double>& geom = cache.geom;
  std::vector<double>& polars = cache.polars;
  std::vector<double>& octupoles = cache.octupoles;
  std::vector<double>& hexadecapoles = cache.hexadecapoles;
  std::vector<double>& chirals = cache.chirals;
  std::vector<double>& pseudodevs = cache.pseudodevs;
  std::vector<double>& rho0_dot = cache.rho0_dot;
  std::vector<double>& raw1_dot = cache.raw1_dot;
  const int chiC = std::min(2, C);
	  std::vector<double> octupoles_raw;
	  std::vector<double> hexadecapoles_raw;
	  if (paramb.spin_chiral) {
	    polars.assign(static_cast<std::size_t>(density_rows) * C * 3, 0.0);
	    octupoles.assign(static_cast<std::size_t>(density_rows) * chiC * 27, 0.0);
	    hexadecapoles.assign(static_cast<std::size_t>(density_rows) * chiC * 81, 0.0);
	    octupoles_raw.assign(static_cast<std::size_t>(density_rows) * chiC * kSpinDeg3Count, 0.0);
	    hexadecapoles_raw.assign(static_cast<std::size_t>(density_rows) * chiC * kSpinDeg4Count, 0.0);
	    chirals.assign(static_cast<std::size_t>(density_rows) * chiC, 0.0);
	    pseudodevs.assign(static_cast<std::size_t>(density_rows) * C * 9, 0.0);
	  }

#if defined(_OPENMP)
  const int num_threads = spin_openmp_threads();
#else
  const int num_threads = 1;
#endif
  const bool use_parallel_edges = num_threads > 1 && N > 8;
  const bool keep_edges = cache_out || paramb.spin_chiral;
  const double spin_rcinv = 1.0 / paramb.spin_cutoff_radial;
  const bool direct_edge_cache =
    keep_edges && use_parallel_edges && use_lammps_edges && loop_count >= 1024;
  const bool fast_direct_edge_count =
    direct_edge_cache && use_lammps_radial_geometry_cache && radial_cache_within_spin;
  std::vector<int>& edge_offsets = cache.edge_offsets;
  std::vector<std::vector<SpinEdge>> private_edges(
    keep_edges && use_parallel_edges && !direct_edge_cache ? static_cast<std::size_t>(num_threads) : 0);
  if (keep_edges) {
    std::size_t edge_capacity = 0;
    for (int idx = 0; idx < loop_count; ++idx) {
      edge_capacity += static_cast<std::size_t>(NN[center_atom(idx)]);
    }
    if (direct_edge_cache) {
      edge_offsets.assign(static_cast<std::size_t>(loop_count) + 1, 0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads)
#endif
      for (int idx = 0; idx < loop_count; ++idx) {
        const int i = center_atom(idx);
        if (!active(paramb.spin_dof_type_active, spin_dof_all_active, type[i])) {
          continue;
        }
        int count = 0;
        if (fast_direct_edge_count) {
          const int edge_offset = lammps_radial_cache->offsets[idx];
          for (int n = 0; n < NN[i]; ++n) {
            const int j = lammps_radial_cache->neighbors[edge_offset + n];
            if (j >= 0 &&
                active(paramb.spin_env_type_active, spin_env_all_active, type[j])) {
              ++count;
            }
          }
        } else {
          for (int n = 0; n < NN[i]; ++n) {
            int j = 0;
            double dx = 0.0;
            double dy = 0.0;
            double dz = 0.0;
            double d = 0.0;
            if (!load_edge_geometry(idx, i, n, j, dx, dy, dz, d)) {
              continue;
            }
            if (d > 1.0e-12 && d < paramb.spin_cutoff_radial &&
                active(paramb.spin_env_type_active, spin_env_all_active, type[j])) {
              ++count;
            }
          }
        }
        edge_offsets[static_cast<std::size_t>(idx) + 1] = count;
      }
      for (int idx = 0; idx < loop_count; ++idx) {
        edge_offsets[static_cast<std::size_t>(idx) + 1] +=
          edge_offsets[static_cast<std::size_t>(idx)];
      }
      cache.edges.resize(static_cast<std::size_t>(edge_offsets[loop_count]));
    } else if (use_parallel_edges) {
      cache.edges.clear();
      const std::size_t reserve_per_thread =
        edge_capacity / static_cast<std::size_t>(num_threads) + 64;
      for (auto& thread_edges : private_edges) {
        thread_edges.reserve(reserve_per_thread);
      }
    } else {
      cache.edges.clear();
      cache.edges.reserve(edge_capacity);
    }
  }
  add_phase(&SpinPhaseBreakdown::setup);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_edges)
#endif
  for (int idx = 0; idx < loop_count; ++idx) {
    const int i = center_atom(idx);
    if (!active(paramb.spin_dof_type_active, spin_dof_all_active, type[i])) {
      continue;
    }
    double* q_center = q +
      static_cast<std::size_t>(use_lammps_edges ? idx : i) * paramb.spin_dim;
    int direct_edge_offset = direct_edge_cache ? edge_offsets[static_cast<std::size_t>(idx)] : 0;
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    for (int n = 0; n < NN[i]; ++n) {
      int j = 0;
      double dx = 0.0;
      double dy = 0.0;
      double dz = 0.0;
      double d = 0.0;
      if (!load_edge_geometry(idx, i, n, j, dx, dy, dz, d)) {
        continue;
      }
      if (d <= 1.0e-12 || d >= paramb.spin_cutoff_radial) {
        continue;
      }
      if (!active(paramb.spin_env_type_active, spin_env_all_active, type[j])) {
        continue;
      }
      double fc = 0.0;
      double fcp = 0.0;
      double fn[MAX_NUM_N];
      double fnp[MAX_NUM_N];
      if (cache_out) {
        find_fc_and_fcp(paramb.spin_cutoff_radial, spin_rcinv, d, fc, fcp);
        if (paramb.spin_basis_size == 3) {
          find_spin_basis3_and_derivatives(spin_rcinv, d, fc, fcp, fn, fnp);
        } else {
          find_fn_and_fnp(paramb.spin_basis_size, spin_rcinv, d, fc, fcp, fn, fnp);
        }
      } else {
        find_fc(paramb.spin_cutoff_radial, spin_rcinv, d, fc);
        find_fn(paramb.spin_basis_size, spin_rcinv, d, fc, fn);
      }
      SpinEdge edge;
      edge.i = i;
      edge.j = j;
      edge.center = use_lammps_edges ? idx : i;
      edge.t12 = type[i] * paramb.num_types + type[j];
      edge.dist = d;
      edge.rhat = {dx / d, dy / d, dz / d};
      const std::array<double, 3> si = {spin(i, 0), spin(i, 1), spin(i, 2)};
      const std::array<double, 3> sj = {spin(j, 0), spin(j, 1), spin(j, 2)};
      edge.weights.fill(0.0);
      edge.weight_derivatives.fill(0.0);
      for (int c = 0; c < C; ++c) {
        double w = 0.0;
        double dw = 0.0;
        const double* coeff =
          annmb.c_spin + (static_cast<std::size_t>(c) * B * paramb.num_types_sq + edge.t12);
        for (int k = 0; k < B; ++k) {
          const double ck = coeff[static_cast<std::size_t>(k) * paramb.num_types_sq];
          w += fn[k] * ck;
          if (cache_out) {
            dw += fnp[k] * ck;
          }
        }
        edge.weights[c] = w;
        edge.weight_derivatives[c] = dw;
      }
      edge.dot = si[0] * sj[0] + si[1] * sj[1] + si[2] * sj[2];
      edge.sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
      edge.ri_dot_si =
        edge.rhat[0] * si[0] + edge.rhat[1] * si[1] + edge.rhat[2] * si[2];
      edge.ri_dot_sj =
        edge.rhat[0] * sj[0] + edge.rhat[1] * sj[1] + edge.rhat[2] * sj[2];
      edge.bond_axis = edge.ri_dot_si * edge.ri_dot_sj;

      int scalar_offset = 2;
      const double scalars[4] = {
        edge.dot, edge.dot * edge.dot, edge.sj2, edge.bond_axis};
      for (int term = 0; term < 4; ++term) {
        for (int c = 0; c < C; ++c) {
          q_center[scalar_offset + c] += edge.weights[c] * scalars[term];
        }
        scalar_offset += C;
      }

      const double sj_value[3] = {sj[0], sj[1], sj[2]};
      add_density_fixed<3>(rho0, C, edge.center, sj_value, edge.weights.data());
      double raw1_value[9];
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          raw1_value[3 * a + b] = edge.rhat[a] * sj[b];
        }
      }
      add_density_fixed<9>(raw1, C, edge.center, raw1_value, edge.weights.data());
      if (l_max >= 1) {
        const double rdot = edge.rhat[0] * sj[0] + edge.rhat[1] * sj[1] + edge.rhat[2] * sj[2];
        add_density_fixed<1>(l1_rdot, C, edge.center, &rdot, edge.weights.data());
        const double cross_value[3] = {
          edge.rhat[1] * sj[2] - edge.rhat[2] * sj[1],
          edge.rhat[2] * sj[0] - edge.rhat[0] * sj[2],
          edge.rhat[0] * sj[1] - edge.rhat[1] * sj[0]};
        add_density_fixed<3>(l1_cross, C, edge.center, cross_value, edge.weights.data());
        const auto stf = stf_outer3(edge.rhat, sj);
        add_density_fixed<9>(l1_stf, C, edge.center, stf.data(), edge.weights.data());
      }
      for (int ell = 2; ell <= l_max; ++ell) {
        double ylm[9];
        const int ylm_width = real_spherical_harmonics_spin(edge.rhat, ell, ylm);
        double value[27];
        int width = 0;
        for (int m = 0; m < ylm_width; ++m) {
          const double y = ylm[m];
          value[width++] = y * sj[0];
          value[width++] = y * sj[1];
          value[width++] = y * sj[2];
        }
        if (ell == 2) {
          add_density_fixed<15>(angular2, C, edge.center, value, edge.weights.data());
        } else if (ell == 3) {
          add_density_fixed<21>(angular3, C, edge.center, value, edge.weights.data());
        } else {
          add_density_fixed<27>(angular4, C, edge.center, value, edge.weights.data());
        }
      }
      const auto rr = stf_outer3(edge.rhat, edge.rhat);
      add_density_fixed<9>(geom, C, edge.center, rr.data(), edge.weights.data());
      if (paramb.spin_chiral) {
        add_density_fixed<3>(polars, C, edge.center, edge.rhat.data(), edge.weights.data());
        double m2[kSpinDeg2Count];
        double m3[kSpinDeg3Count];
        double m4[kSpinDeg4Count];
        fill_spin_monomials(edge.rhat, m2, m3, m4);
        add_density_fixed<kSpinDeg3Count>(octupoles_raw, chiC, edge.center, m3, edge.weights.data());
        add_density_fixed<kSpinDeg4Count>(hexadecapoles_raw, chiC, edge.center, m4, edge.weights.data());
      }
      add_density_fixed<3>(rho0_dot, C, edge.center, sj_value, edge.weights.data(), edge.dot);
      add_density_fixed<9>(raw1_dot, C, edge.center, raw1_value, edge.weights.data(), edge.dot);
      if (keep_edges) {
        if (direct_edge_cache) {
          cache.edges[static_cast<std::size_t>(direct_edge_offset++)] = std::move(edge);
        } else if (use_parallel_edges) {
          private_edges[static_cast<std::size_t>(tid)].push_back(std::move(edge));
        } else {
          cache.edges.push_back(std::move(edge));
        }
      }
    }
  }
  add_phase(&SpinPhaseBreakdown::edges);

  if (paramb.spin_chiral) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_edges)
#endif
    for (int idx = 0; idx < descriptor_loop_count; ++idx) {
      const int row = idx;
      for (int c = 0; c < chiC; ++c) {
        const std::size_t raw_base = static_cast<std::size_t>(row) * chiC + c;
        unpack_rank3_spin_stf(
          octupoles_raw.data() + raw_base * kSpinDeg3Count,
          octupoles.data() + raw_base * 27);
        unpack_rank4_spin_stf(
          hexadecapoles_raw.data() + raw_base * kSpinDeg4Count,
          hexadecapoles.data() + raw_base * 81);
      }
    }
  }
  add_phase(&SpinPhaseBreakdown::unpack);

  if (keep_edges && use_parallel_edges && !direct_edge_cache) {
    std::size_t edge_count = 0;
    for (const auto& thread_edges : private_edges) {
      edge_count += thread_edges.size();
    }
    cache.edges.reserve(edge_count);
    for (auto& thread_edges : private_edges) {
      cache.edges.insert(
        cache.edges.end(),
        std::make_move_iterator(thread_edges.begin()),
        std::make_move_iterator(thread_edges.end()));
    }
  }
  add_phase(&SpinPhaseBreakdown::merge);

	  int offset = 2 + 4 * C;
	  auto contract = [&](const std::vector<double>& a, const std::vector<double>& b, const int width) {
	    const int q_offset = offset;
	#if defined(_OPENMP)
	#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_edges)
	#endif
	    for (int idx = 0; idx < descriptor_loop_count; ++idx) {
	      const int atom = descriptor_atom(idx);
	      const int row = idx;
	      for (int c = 0; c < C; ++c) {
	        const double* av = a.data() + (static_cast<std::size_t>(row) * C + c) * width;
	        const double* bv = b.data() + (static_cast<std::size_t>(row) * C + c) * width;
	        double sum = 0.0;
	        for (int k = 0; k < width; ++k) {
	          sum += av[k] * bv[k];
	        }
	        qref(atom, q_offset + c) = sum;
	      }
	    }
	    offset += C;
	  };

	  contract(rho0, rho0, 3);
	  if (l_max >= 1) {
	    contract(l1_rdot, l1_rdot, 1);
	    contract(l1_cross, l1_cross, 3);
	    contract(l1_stf, l1_stf, 9);
	  }
	  for (int ell = 2; ell <= l_max; ++ell) {
	    contract(ell == 2 ? angular2 : ell == 3 ? angular3 : angular4, ell == 2 ? angular2 : ell == 3 ? angular3 : angular4, (2 * ell + 1) * 3);
	  }
	  const int geom_q_offset = offset;
	#if defined(_OPENMP)
	#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_edges)
	#endif
	  for (int idx = 0; idx < descriptor_loop_count; ++idx) {
	    const int atom = descriptor_atom(idx);
	    const int row = idx;
	    const std::array<double, 3> s = {spin(atom, 0), spin(atom, 1), spin(atom, 2)};
	    for (int c = 0; c < C; ++c) {
	      const double* g = geom.data() + (static_cast<std::size_t>(row) * C + c) * 9;
	      double value = 0.0;
	      for (int a = 0; a < 3; ++a) {
	        for (int b = 0; b < 3; ++b) {
	          value += s[a] * g[3 * a + b] * s[b];
	        }
	      }
	      qref(atom, geom_q_offset + c) = value;
	    }
	  }
	  offset += C;
	  contract(rho0, rho0_dot, 3);
	  if (l_max >= 1) {
	    contract(raw1, raw1_dot, 9);
	  }
	  add_phase(&SpinPhaseBreakdown::contract);

  if (paramb.spin_chiral) {
    const bool use_center_edges = !edge_offsets.empty();
    auto block = [&](const std::vector<double>& v, const int atom, const int c, const int width) {
      return v.data() + (static_cast<std::size_t>(atom) * C + c) * width;
    };
    auto chi_block = [&](const std::vector<double>& v, const int atom, const int c, const int width) {
      return v.data() + (static_cast<std::size_t>(atom) * chiC + c) * width;
    };
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_edges)
#endif
    for (int idx = 0; idx < descriptor_loop_count; ++idx) {
      const int row = idx;
      for (int c = 0; c < chiC; ++c) {
        const double* Q = block(geom, row, c, 9);
        const double* O = chi_block(octupoles, row, c, 27);
        const double* H = chi_block(hexadecapoles, row, c, 81);
        double value = 0.0;
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            for (int cc = 0; cc < 3; ++cc) {
              const int eps = levi_civita(a, b, cc);
              if (eps == 0) {
                continue;
              }
              for (int d = 0; d < 3; ++d) {
                for (int e = 0; e < 3; ++e) {
                  for (int f = 0; f < 3; ++f) {
                    value += eps * Q[3 * a + d] * O[(b * 3 + e) * 3 + f] *
                             H[((cc * 3 + d) * 3 + e) * 3 + f];
                  }
                }
              }
            }
          }
        }
        chirals[static_cast<std::size_t>(row) * chiC + c] = value;
      }
    }
    auto add_pseudodev = [&](const SpinEdge& edge) {
      for (int c = 0; c < C; ++c) {
        const double* Q = block(geom, edge.center, c, 9);
        std::array<double, 3> Qu = {0.0, 0.0, 0.0};
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            Qu[a] += Q[3 * a + b] * edge.rhat[b];
          }
        }
        const std::array<double, 3> axis = cross3(edge.rhat, Qu);
        double* out = pseudodevs.data() + (static_cast<std::size_t>(edge.center) * C + c) * 9;
        const double w = edge.weights[c];
        const double wh = 0.5 * w;
        const double pseudo01 = wh * (axis[0] * edge.rhat[1] + axis[1] * edge.rhat[0]);
        const double pseudo02 = wh * (axis[0] * edge.rhat[2] + axis[2] * edge.rhat[0]);
        const double pseudo12 = wh * (axis[1] * edge.rhat[2] + axis[2] * edge.rhat[1]);
        out[0] += w * axis[0] * edge.rhat[0];
        out[1] += pseudo01;
        out[2] += pseudo02;
        out[3] += pseudo01;
        out[4] += w * axis[1] * edge.rhat[1];
        out[5] += pseudo12;
        out[6] += pseudo02;
        out[7] += pseudo12;
        out[8] += w * axis[2] * edge.rhat[2];
      }
    };
    if (use_center_edges) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_edges)
#endif
      for (std::ptrdiff_t center_index = 0;
           center_index < static_cast<std::ptrdiff_t>(edge_offsets.size() - 1);
           ++center_index) {
        const std::size_t idx = static_cast<std::size_t>(center_index);
        for (int e = edge_offsets[idx]; e < edge_offsets[idx + 1]; ++e) {
          add_pseudodev(cache.edges[static_cast<std::size_t>(e)]);
        }
      }
    } else {
      for (const SpinEdge& edge : cache.edges) {
        add_pseudodev(edge);
      }
    }
    auto add_chiral_q = [&](const SpinEdge& edge, double* q_center) {
      const std::array<double, 3> si = {spin(edge.i, 0), spin(edge.i, 1), spin(edge.i, 2)};
      const std::array<double, 3> sj = {spin(edge.j, 0), spin(edge.j, 1), spin(edge.j, 2)};
      const std::array<double, 3> spin_cross = cross3(si, sj);
      for (int c = 0; c < chiC; ++c) {
        q_center[offset + c] += edge.weights[c] * dot3(spin_cross, edge.rhat) *
                                chirals[static_cast<std::size_t>(edge.center) * chiC + c];
      }
      int chiral_offset = offset + chiC;
      for (int c = 0; c < C; ++c) {
        const double* p = block(polars, edge.center, c, 3);
        const std::array<double, 3> polar = {p[0], p[1], p[2]};
        const std::array<double, 3> axis = cross3(polar, edge.rhat);
        q_center[chiral_offset + c] += edge.weights[c] * dot3(spin_cross, axis);
      }
      chiral_offset += C;
      for (int c = 0; c < C; ++c) {
        const double* P = block(pseudodevs, edge.center, c, 9);
        std::array<double, 3> axis = {0.0, 0.0, 0.0};
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            axis[a] += P[3 * a + b] * edge.rhat[b];
          }
        }
        q_center[chiral_offset + c] += edge.weights[c] * dot3(spin_cross, axis);
      }
    };
    if (use_center_edges) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_edges)
#endif
      for (std::ptrdiff_t center_index = 0;
           center_index < static_cast<std::ptrdiff_t>(edge_offsets.size() - 1);
           ++center_index) {
        const std::size_t idx = static_cast<std::size_t>(center_index);
        double* q_center = q + idx * paramb.spin_dim;
        for (int e = edge_offsets[idx]; e < edge_offsets[idx + 1]; ++e) {
          add_chiral_q(cache.edges[static_cast<std::size_t>(e)], q_center);
        }
      }
    } else {
      for (const SpinEdge& edge : cache.edges) {
        add_chiral_q(edge, &qref(edge.i, 0));
      }
    }
  }
  add_phase(&SpinPhaseBreakdown::chiral);

  if (center_descriptor_aos && use_lammps_edges) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_edges)
#endif
    for (int idx = 0; idx < loop_count; ++idx) {
      const int atom = center_atom(idx);
      double* dst =
        center_descriptor_aos + static_cast<std::size_t>(idx) * annmb.dim + offset0;
      for (int d = 0; d < paramb.spin_dim; ++d) {
        dst[d] = qref(atom, d) * paramb.q_scaler[offset0 + d];
      }
    }
  } else if (descriptor_soa) {
    for (int atom = 0; atom < N; ++atom) {
      for (int d = 0; d < paramb.spin_dim; ++d) {
        descriptor_soa[(offset0 + d) * N + atom] =
          qref(atom, d) * paramb.q_scaler[offset0 + d];
      }
    }
  }
  add_phase(&SpinPhaseBreakdown::copy);
}

void add_real_spherical_harmonics_gradient(
  const std::array<double, 3>& r,
  const int ell,
  const double* grad_y,
  std::array<double, 3>& grad_r)
{
  const double x = r[0];
  const double y = r[1];
  const double z = r[2];
  if (ell == 2) {
    const double a = std::sqrt(15.0 / (4.0 * PI));
    const double b = std::sqrt(5.0 / (16.0 * PI));
    const double c = std::sqrt(15.0 / (16.0 * PI));
    grad_r[0] += grad_y[0] * a * y - grad_y[2] * 2.0 * b * x + grad_y[3] * a * z + grad_y[4] * 2.0 * c * x;
    grad_r[1] += grad_y[0] * a * x + grad_y[1] * a * z - grad_y[2] * 2.0 * b * y - grad_y[4] * 2.0 * c * y;
    grad_r[2] += grad_y[1] * a * y + grad_y[2] * 4.0 * b * z + grad_y[3] * a * x;
    return;
  }
  if (ell == 3) {
    const double x2 = x * x;
    const double y2 = y * y;
    const double z2 = z * z;
    const double rho2 = x2 + y2;
    const double a = std::sqrt(35.0 / (32.0 * PI));
    const double b = std::sqrt(105.0 / (4.0 * PI));
    const double c = std::sqrt(21.0 / (32.0 * PI));
    const double d = std::sqrt(7.0 / (16.0 * PI));
    const double e = std::sqrt(105.0 / (16.0 * PI));
    grad_r[0] += grad_y[0] * 6.0 * a * x * y + grad_y[1] * b * y * z -
                 grad_y[2] * 2.0 * c * x * y + grad_y[3] * -6.0 * d * x * z +
                 grad_y[4] * c * (4.0 * z2 - 3.0 * x2 - y2) +
                 grad_y[5] * 2.0 * e * x * z + grad_y[6] * 3.0 * a * (x2 - y2);
    grad_r[1] += grad_y[0] * 3.0 * a * (x2 - y2) + grad_y[1] * b * x * z +
                 grad_y[2] * c * (4.0 * z2 - x2 - 3.0 * y2) +
                 grad_y[3] * -6.0 * d * y * z - grad_y[4] * 2.0 * c * x * y -
                 grad_y[5] * 2.0 * e * y * z - grad_y[6] * 6.0 * a * x * y;
    grad_r[2] += grad_y[1] * b * x * y + grad_y[2] * 8.0 * c * y * z +
                 grad_y[3] * d * (6.0 * z2 - 3.0 * rho2) +
                 grad_y[4] * 8.0 * c * x * z + grad_y[5] * e * (x2 - y2);
    return;
  }
  const double x2 = x * x;
  const double y2 = y * y;
  const double z2 = z * z;
  const double a = 0.75 * std::sqrt(35.0 / PI);
  const double b = 0.75 * std::sqrt(35.0 / (2.0 * PI));
  const double c = 0.75 * std::sqrt(5.0 / PI);
  const double d = 0.75 * std::sqrt(5.0 / (2.0 * PI));
  const double e = (3.0 / 16.0) * std::sqrt(1.0 / PI);
  const double f = 0.375 * std::sqrt(5.0 / PI);
  const double g = (3.0 / 16.0) * std::sqrt(35.0 / PI);
  grad_r[0] += grad_y[0] * a * y * (3.0 * x2 - y2) +
               grad_y[1] * b * 6.0 * x * y * z +
               grad_y[2] * c * y * (7.0 * z2 - 1.0) +
               grad_y[5] * d * z * (7.0 * z2 - 3.0) +
               grad_y[6] * 2.0 * f * x * (7.0 * z2 - 1.0) +
               grad_y[7] * b * z * (3.0 * x2 - 3.0 * y2) +
               grad_y[8] * g * (4.0 * x * x2 - 12.0 * x * y2);
  grad_r[1] += grad_y[0] * a * x * (x2 - 3.0 * y2) +
               grad_y[1] * b * z * (3.0 * x2 - 3.0 * y2) +
               grad_y[2] * c * x * (7.0 * z2 - 1.0) +
               grad_y[3] * d * z * (7.0 * z2 - 3.0) -
               grad_y[6] * 2.0 * f * y * (7.0 * z2 - 1.0) -
               grad_y[7] * b * 6.0 * x * y * z +
               grad_y[8] * g * (-12.0 * x2 * y + 4.0 * y * y2);
  grad_r[2] += grad_y[1] * b * y * (3.0 * x2 - y2) +
               grad_y[2] * c * 14.0 * x * y * z +
               grad_y[3] * d * y * (21.0 * z2 - 3.0) +
               grad_y[4] * e * (140.0 * z2 * z - 60.0 * z) +
               grad_y[5] * d * x * (21.0 * z2 - 3.0) +
               grad_y[6] * f * 14.0 * z * (x2 - y2) +
               grad_y[7] * b * x * (x2 - 3.0 * y2);
}

void add_spin_chiral_gradient(
  const NEP::ParaMB& paramb,
  const NEP::ANN& annmb,
  const int N,
  const double* spins,
  const SpinCache& cache,
  const double* Fp,
  std::vector<double>& grad_spin,
  double* force,
  double* virial,
  double* spin_transfer,
  const bool lammps_neighbor_virial_ownership,
  LammpsThreadLocalScratchView* lammps_scratch = nullptr,
  NEP::SpinGradientScratch* scratch = nullptr)
{
  if (!paramb.spin_chiral || cache.edges.empty()) {
    return;
  }
  const int C = paramb.spin_compress;
  const int chiC = std::min(2, C);
  const int base_offset = spin_descriptor_dim(C, paramb.spin_l_max, false);
  const int offset0 = paramb.struct_dim;
  const std::size_t edge_count = cache.edges.size();
  std::vector<double> local_grad_weight;
  std::vector<double> local_grad_rhat;
  std::vector<double> local_grad_si;
  std::vector<double> local_grad_sj;
  std::vector<double> local_grad_Q;
  std::vector<double> local_grad_O;
  std::vector<double> local_grad_H;
  std::vector<double> local_grad_chi;
  std::vector<double> local_grad_polar;
  std::vector<double> local_grad_pseudodev;
  std::vector<double>& grad_weight = scratch ? scratch->grad_weight : local_grad_weight;
  std::vector<double>& grad_rhat = scratch ? scratch->grad_rhat : local_grad_rhat;
  std::vector<double>& grad_si = scratch ? scratch->grad_si : local_grad_si;
  std::vector<double>& grad_sj = scratch ? scratch->grad_sj : local_grad_sj;
  std::vector<double>& grad_Q = scratch ? scratch->grad_Q : local_grad_Q;
  std::vector<double>& grad_O = scratch ? scratch->grad_O : local_grad_O;
  std::vector<double>& grad_H = scratch ? scratch->grad_H : local_grad_H;
  std::vector<double>& grad_chi = scratch ? scratch->grad_chi : local_grad_chi;
  std::vector<double>& grad_polar = scratch ? scratch->grad_polar : local_grad_polar;
  std::vector<double>& grad_pseudodev = scratch ? scratch->grad_pseudodev : local_grad_pseudodev;
#if defined(_OPENMP)
  const int num_threads = spin_openmp_threads();
#else
  const int num_threads = 1;
#endif
  const bool use_parallel_edges = num_threads > 1 && edge_count > 32;
  const bool use_lammps_scratch = lammps_spin_scratch_active(lammps_scratch);
  const bool direct_lammps_spin_pull = use_lammps_scratch;
  const bool use_center_edges = cache.edge_offsets.size() > 1;
  const int density_rows = use_center_edges ? static_cast<int>(cache.edge_offsets.size() - 1) : N;
  resize_and_zero(grad_weight, edge_count * C);
  resize_and_zero(grad_rhat, edge_count * 3);
  resize_and_zero(grad_si, direct_lammps_spin_pull ? 0 : edge_count * 3);
  resize_and_zero(grad_sj, direct_lammps_spin_pull ? 0 : edge_count * 3);
  resize_and_zero(grad_Q, static_cast<std::size_t>(density_rows) * C * 9);
  resize_and_zero(grad_O, static_cast<std::size_t>(density_rows) * chiC * 27);
  resize_and_zero(grad_H, static_cast<std::size_t>(density_rows) * chiC * 81);
  resize_and_zero(grad_chi, static_cast<std::size_t>(density_rows) * chiC);
  resize_and_zero(grad_polar, static_cast<std::size_t>(density_rows) * C * 3);
  resize_and_zero(grad_pseudodev, static_cast<std::size_t>(density_rows) * C * 9);

  auto fp = [&](const int atom, const int dim) {
    return Fp[static_cast<std::size_t>(atom) * annmb.dim + offset0 + dim];
  };
  auto spin = [&](const int atom, const int d) {
    return spins[static_cast<std::size_t>(d) * N + atom];
  };
  auto blockC = [&](std::vector<double>& v, const int atom, const int c, const int width) {
    return v.data() + (static_cast<std::size_t>(atom) * C + c) * width;
  };
  auto cblockC = [&](const std::vector<double>& v, const int atom, const int c, const int width) {
    return v.data() + (static_cast<std::size_t>(atom) * C + c) * width;
  };
  auto blockChi = [&](std::vector<double>& v, const int atom, const int c, const int width) {
    return v.data() + (static_cast<std::size_t>(atom) * chiC + c) * width;
  };
  auto cblockChi = [&](const std::vector<double>& v, const int atom, const int c, const int width) {
    return v.data() + (static_cast<std::size_t>(atom) * chiC + c) * width;
  };
  auto egw = [&](const std::size_t e, const int c) -> double& {
    return grad_weight[e * C + c];
  };
  auto eg3 = [&](std::vector<double>& v, const std::size_t e, const int d) -> double& {
    return v[e * 3 + d];
  };
  auto add_edge_vec = [&](std::vector<double>& v, const std::size_t e, const std::array<double, 3>& g) {
    for (int d = 0; d < 3; ++d) {
      eg3(v, e, d) += g[d];
    }
  };
  auto add_lammps_spin_pull = [&](double* local_grad_spin, const int row, const std::array<double, 3>& g) {
    const int stride = lammps_scratch->force_rows;
    for (int d = 0; d < 3; ++d) {
      local_grad_spin[lammps_vector_index(row, d, stride)] += g[d];
    }
  };
  auto reduce_private = [&](std::vector<double>& target, const std::vector<double>& source) {
    if (source.empty()) {
      return;
    }
    const std::size_t stride = target.size();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (stride > 1024)
#endif
    for (std::ptrdiff_t k = 0; k < static_cast<std::ptrdiff_t>(stride); ++k) {
      double sum = target[static_cast<std::size_t>(k)];
      for (int tid = 0; tid < num_threads; ++tid) {
        sum += source[static_cast<std::size_t>(tid) * stride + static_cast<std::size_t>(k)];
      }
      target[static_cast<std::size_t>(k)] = sum;
    }
  };

  std::vector<double> local_grad_chi_private;
  std::vector<double> local_grad_polar_private;
  std::vector<double> local_grad_pseudodev_private;
  std::vector<double>& grad_chi_private =
    scratch ? scratch->grad_chi_private : local_grad_chi_private;
  std::vector<double>& grad_polar_private =
    scratch ? scratch->grad_polar_private : local_grad_polar_private;
  std::vector<double>& grad_pseudodev_private =
    scratch ? scratch->grad_pseudodev_private : local_grad_pseudodev_private;
  resize_and_zero(
    grad_chi_private,
    use_parallel_edges && !use_center_edges
      ? static_cast<std::size_t>(num_threads) * grad_chi.size()
      : 0);
  resize_and_zero(
    grad_polar_private,
    use_parallel_edges && !use_center_edges
      ? static_cast<std::size_t>(num_threads) * grad_polar.size()
      : 0);
  resize_and_zero(
    grad_pseudodev_private,
    use_parallel_edges && !use_center_edges
      ? static_cast<std::size_t>(num_threads) * grad_pseudodev.size()
      : 0);

  auto add_chiral_edge_pull = [&](const std::size_t e,
                                  double* local_grad_chi,
                                  double* local_grad_polar,
                                  double* local_grad_pseudodev,
                                  double* local_grad_spin,
                                  double* local_spin_transfer) {
    const SpinEdge& edge = cache.edges[e];
    const std::array<double, 3> si = {spin(edge.i, 0), spin(edge.i, 1), spin(edge.i, 2)};
    const std::array<double, 3> sj = {spin(edge.j, 0), spin(edge.j, 1), spin(edge.j, 2)};
    const std::array<double, 3> x = cross3(si, sj);
    std::array<double, 3> gx = {0.0, 0.0, 0.0};
    std::array<double, 3> gu = {0.0, 0.0, 0.0};
    const double xu = dot3(x, edge.rhat);
    for (int c = 0; c < chiC; ++c) {
      const double alpha = fp(edge.i, base_offset + c);
      const double chi = cache.chirals[static_cast<std::size_t>(edge.center) * chiC + c];
      const double aw = alpha * edge.weights[c];
      egw(e, c) += alpha * xu * chi;
      local_grad_chi[static_cast<std::size_t>(edge.center) * chiC + c] +=
        aw * xu;
      for (int d = 0; d < 3; ++d) {
        gx[d] += aw * chi * edge.rhat[d];
        gu[d] += aw * chi * x[d];
      }
    }
    int chiral_offset = base_offset + chiC;
    for (int c = 0; c < C; ++c) {
      const double alpha = fp(edge.i, chiral_offset + c);
      const double aw = alpha * edge.weights[c];
      const double* p = cblockC(cache.polars, edge.center, c, 3);
      const std::array<double, 3> polar = {p[0], p[1], p[2]};
      const std::array<double, 3> axis = cross3(polar, edge.rhat);
      const double xa = dot3(x, axis);
      egw(e, c) += alpha * xa;
      std::array<double, 3> gaxis = {0.0, 0.0, 0.0};
      for (int d = 0; d < 3; ++d) {
        gx[d] += aw * axis[d];
        gaxis[d] = aw * x[d];
      }
      const std::array<double, 3> gp = cross3(edge.rhat, gaxis);
      const std::array<double, 3> gu_part = cross3(gaxis, polar);
      double* grad_p = local_grad_polar + (static_cast<std::size_t>(edge.center) * C + c) * 3;
      for (int d = 0; d < 3; ++d) {
        grad_p[d] += gp[d];
        gu[d] += gu_part[d];
      }
    }
    chiral_offset += C;
    for (int c = 0; c < C; ++c) {
      const double alpha = fp(edge.i, chiral_offset + c);
      const double aw = alpha * edge.weights[c];
      const double* P = cblockC(cache.pseudodevs, edge.center, c, 9);
      std::array<double, 3> axis = {0.0, 0.0, 0.0};
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          axis[a] += P[3 * a + b] * edge.rhat[b];
        }
      }
      const double xa = dot3(x, axis);
      egw(e, c) += alpha * xa;
      std::array<double, 3> gaxis = {0.0, 0.0, 0.0};
      for (int d = 0; d < 3; ++d) {
        gx[d] += aw * axis[d];
        gaxis[d] = aw * x[d];
      }
      double* gP = local_grad_pseudodev + (static_cast<std::size_t>(edge.center) * C + c) * 9;
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          gP[3 * a + b] += gaxis[a] * edge.rhat[b];
          gu[b] += P[3 * a + b] * gaxis[a];
        }
      }
    }
    const std::array<double, 3> gsi = cross3(sj, gx);
    const std::array<double, 3> gsj = cross3(gx, si);
    if (direct_lammps_spin_pull) {
      add_lammps_spin_pull(local_grad_spin, edge.i, gsi);
      add_lammps_spin_pull(local_grad_spin, edge.j, gsj);
      add_spin_transfer_row_major9(
        edge.rhat, edge.dist, gsj, local_spin_transfer,
        lammps_scratch->force_rows, edge.j);
    } else {
      add_edge_vec(grad_si, e, gsi);
      add_edge_vec(grad_sj, e, gsj);
    }
    add_edge_vec(grad_rhat, e, gu);
  };

  if (use_center_edges) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_edges)
#endif
    for (std::ptrdiff_t center_index = 0;
         center_index < static_cast<std::ptrdiff_t>(cache.edge_offsets.size() - 1);
         ++center_index) {
      const std::size_t idx = static_cast<std::size_t>(center_index);
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      double* local_grad_spin = direct_lammps_spin_pull
        ? lammps_scratch->mforce_private +
            static_cast<std::size_t>(tid) * 3 * lammps_scratch->force_rows
        : nullptr;
      double* local_spin_transfer =
        direct_lammps_spin_pull && lammps_scratch->spin_transfer_private
          ? lammps_scratch->spin_transfer_private +
              static_cast<std::size_t>(tid) * 9 * lammps_scratch->force_rows
          : nullptr;
      for (int e = cache.edge_offsets[idx]; e < cache.edge_offsets[idx + 1]; ++e) {
        add_chiral_edge_pull(
          static_cast<std::size_t>(e), grad_chi.data(), grad_polar.data(),
          grad_pseudodev.data(), local_grad_spin, local_spin_transfer);
      }
    }
  } else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_edges)
#endif
    for (std::ptrdiff_t edge_index = 0;
         edge_index < static_cast<std::ptrdiff_t>(edge_count);
         ++edge_index) {
      const std::size_t e = static_cast<std::size_t>(edge_index);
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      double* local_grad_chi = use_parallel_edges
        ? grad_chi_private.data() + static_cast<std::size_t>(tid) * grad_chi.size()
        : grad_chi.data();
      double* local_grad_polar = use_parallel_edges
        ? grad_polar_private.data() + static_cast<std::size_t>(tid) * grad_polar.size()
        : grad_polar.data();
      double* local_grad_pseudodev = use_parallel_edges
        ? grad_pseudodev_private.data() + static_cast<std::size_t>(tid) * grad_pseudodev.size()
        : grad_pseudodev.data();
      double* local_grad_spin = direct_lammps_spin_pull
        ? lammps_scratch->mforce_private +
            static_cast<std::size_t>(tid) * 3 * lammps_scratch->force_rows
        : nullptr;
      double* local_spin_transfer =
        direct_lammps_spin_pull && lammps_scratch->spin_transfer_private
          ? lammps_scratch->spin_transfer_private +
              static_cast<std::size_t>(tid) * 9 * lammps_scratch->force_rows
          : nullptr;
      add_chiral_edge_pull(
        e, local_grad_chi, local_grad_polar, local_grad_pseudodev,
        local_grad_spin, local_spin_transfer);
    }
    reduce_private(grad_chi, grad_chi_private);
    reduce_private(grad_polar, grad_polar_private);
    reduce_private(grad_pseudodev, grad_pseudodev_private);
  }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_edges)
#endif
  for (int atom = 0; atom < density_rows; ++atom) {
    for (int c = 0; c < chiC; ++c) {
      const double g = grad_chi[static_cast<std::size_t>(atom) * chiC + c];
      if (g == 0.0) {
        continue;
      }
      const double* Q = cblockC(cache.geom, atom, c, 9);
      const double* O = cblockChi(cache.octupoles, atom, c, 27);
      const double* H = cblockChi(cache.hexadecapoles, atom, c, 81);
      double* gQ = blockC(grad_Q, atom, c, 9);
      double* gO = blockChi(grad_O, atom, c, 27);
      double* gH = blockChi(grad_H, atom, c, 81);
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          for (int cc = 0; cc < 3; ++cc) {
            const int eps = levi_civita(a, b, cc);
            if (eps == 0) {
              continue;
            }
            const double geps = g * eps;
            for (int d = 0; d < 3; ++d) {
              for (int e = 0; e < 3; ++e) {
                for (int f = 0; f < 3; ++f) {
                  const int oidx = (b * 3 + e) * 3 + f;
                  const int hidx = ((cc * 3 + d) * 3 + e) * 3 + f;
                  gQ[3 * a + d] += geps * O[oidx] * H[hidx];
                  gO[oidx] += geps * Q[3 * a + d] * H[hidx];
                  gH[hidx] += geps * Q[3 * a + d] * O[oidx];
                }
              }
            }
          }
        }
      }
    }
  }

  std::vector<double> local_grad_Q_private;
  std::vector<double>& grad_Q_private =
    scratch ? scratch->grad_Q_private : local_grad_Q_private;
  resize_and_zero(
    grad_Q_private,
    use_parallel_edges && !use_center_edges
      ? static_cast<std::size_t>(num_threads) * grad_Q.size()
      : 0);

  auto add_chiral_pseudodev_pull = [&](const std::size_t e, double* local_grad_Q) {
    const SpinEdge& edge = cache.edges[e];
    for (int c = 0; c < C; ++c) {
      const double* gp = cblockC(grad_polar, edge.center, c, 3);
      egw(e, c) += gp[0] * edge.rhat[0] + gp[1] * edge.rhat[1] + gp[2] * edge.rhat[2];
      for (int d = 0; d < 3; ++d) {
        eg3(grad_rhat, e, d) += edge.weights[c] * gp[d];
      }

      const double* gPd = cblockC(grad_pseudodev, edge.center, c, 9);
      const double* Q = cblockC(cache.geom, edge.center, c, 9);
      std::array<double, 3> Qu = {0.0, 0.0, 0.0};
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          Qu[a] += Q[3 * a + b] * edge.rhat[b];
        }
      }
      const std::array<double, 3> pseudo_axis = cross3(edge.rhat, Qu);
      const double pseudo00 = pseudo_axis[0] * edge.rhat[0];
      const double pseudo01 =
        0.5 * (pseudo_axis[0] * edge.rhat[1] + pseudo_axis[1] * edge.rhat[0]);
      const double pseudo02 =
        0.5 * (pseudo_axis[0] * edge.rhat[2] + pseudo_axis[2] * edge.rhat[0]);
      const double pseudo10 = pseudo01;
      const double pseudo11 = pseudo_axis[1] * edge.rhat[1];
      const double pseudo12 =
        0.5 * (pseudo_axis[1] * edge.rhat[2] + pseudo_axis[2] * edge.rhat[1]);
      const double pseudo20 = pseudo02;
      const double pseudo21 = pseudo12;
      const double pseudo22 = pseudo_axis[2] * edge.rhat[2];
      const double dot =
        gPd[0] * pseudo00 + gPd[1] * pseudo01 + gPd[2] * pseudo02 +
        gPd[3] * pseudo10 + gPd[4] * pseudo11 + gPd[5] * pseudo12 +
        gPd[6] * pseudo20 + gPd[7] * pseudo21 + gPd[8] * pseudo22;
      egw(e, c) += dot;
      const double s01 = 0.5 * (gPd[1] + gPd[3]);
      const double s02 = 0.5 * (gPd[2] + gPd[6]);
      const double s12 = 0.5 * (gPd[5] + gPd[7]);
      const double w = edge.weights[c];
      const std::array<double, 3> g_axis = {
        w * (gPd[0] * edge.rhat[0] + s01 * edge.rhat[1] + s02 * edge.rhat[2]),
        w * (s01 * edge.rhat[0] + gPd[4] * edge.rhat[1] + s12 * edge.rhat[2]),
        w * (s02 * edge.rhat[0] + s12 * edge.rhat[1] + gPd[8] * edge.rhat[2])};
      const std::array<double, 3> gu2 = {
        w * (gPd[0] * pseudo_axis[0] + s01 * pseudo_axis[1] + s02 * pseudo_axis[2]),
        w * (s01 * pseudo_axis[0] + gPd[4] * pseudo_axis[1] + s12 * pseudo_axis[2]),
        w * (s02 * pseudo_axis[0] + s12 * pseudo_axis[1] + gPd[8] * pseudo_axis[2])};
      const std::array<double, 3> g_u_cross = cross3(Qu, g_axis);
      const std::array<double, 3> g_Qu = cross3(g_axis, edge.rhat);
      double* gQ = local_grad_Q + (static_cast<std::size_t>(edge.center) * C + c) * 9;
      for (int a = 0; a < 3; ++a) {
        eg3(grad_rhat, e, a) += gu2[a] + g_u_cross[a];
        for (int b = 0; b < 3; ++b) {
          gQ[3 * a + b] += g_Qu[a] * edge.rhat[b];
          eg3(grad_rhat, e, b) += Q[3 * a + b] * g_Qu[a];
        }
      }
    }
  };

  if (use_center_edges) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_edges)
#endif
    for (std::ptrdiff_t center_index = 0;
         center_index < static_cast<std::ptrdiff_t>(cache.edge_offsets.size() - 1);
         ++center_index) {
      const std::size_t idx = static_cast<std::size_t>(center_index);
      for (int e = cache.edge_offsets[idx]; e < cache.edge_offsets[idx + 1]; ++e) {
        add_chiral_pseudodev_pull(static_cast<std::size_t>(e), grad_Q.data());
      }
    }
  } else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_edges)
#endif
    for (std::ptrdiff_t edge_index = 0;
         edge_index < static_cast<std::ptrdiff_t>(edge_count);
         ++edge_index) {
      const std::size_t e = static_cast<std::size_t>(edge_index);
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      double* local_grad_Q = use_parallel_edges
        ? grad_Q_private.data() + static_cast<std::size_t>(tid) * grad_Q.size()
        : grad_Q.data();
      add_chiral_pseudodev_pull(e, local_grad_Q);
    }
    reduce_private(grad_Q, grad_Q_private);
  }

  std::vector<double> local_grad_Q_terms;
  std::vector<double> local_grad_O_terms;
  std::vector<double> local_grad_O_derivatives;
  std::vector<double> local_grad_H_terms;
  std::vector<double> local_grad_H_derivatives;
  std::vector<double>& grad_Q_terms = scratch ? scratch->grad_Q_terms : local_grad_Q_terms;
  std::vector<double>& grad_O_terms = scratch ? scratch->grad_O_terms : local_grad_O_terms;
  std::vector<double>& grad_O_derivatives =
    scratch ? scratch->grad_O_derivatives : local_grad_O_derivatives;
  std::vector<double>& grad_H_terms = scratch ? scratch->grad_H_terms : local_grad_H_terms;
  std::vector<double>& grad_H_derivatives =
    scratch ? scratch->grad_H_derivatives : local_grad_H_derivatives;
  grad_Q_terms.resize(static_cast<std::size_t>(density_rows) * C * kSpinDeg2Count);
  grad_O_terms.resize(static_cast<std::size_t>(density_rows) * chiC * kSpinDeg3Count);
  grad_O_derivatives.resize(static_cast<std::size_t>(density_rows) * chiC * 3 * kSpinDeg2Count);
  grad_H_terms.resize(static_cast<std::size_t>(density_rows) * chiC * kSpinDeg4Count);
  grad_H_derivatives.resize(static_cast<std::size_t>(density_rows) * chiC * 3 * kSpinDeg3Count);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_edges)
#endif
  for (int atom = 0; atom < density_rows; ++atom) {
    for (int c = 0; c < C; ++c) {
      project_rank2_spin_gradient(
        cblockC(grad_Q, atom, c, 9),
        blockC(grad_Q_terms, atom, c, kSpinDeg2Count));
    }
    for (int c = 0; c < chiC; ++c) {
      project_rank3_spin_gradient(
        cblockChi(grad_O, atom, c, 27),
        blockChi(grad_O_terms, atom, c, kSpinDeg3Count),
        blockChi(grad_O_derivatives, atom, c, 3 * kSpinDeg2Count));
      project_rank4_spin_gradient(
        cblockChi(grad_H, atom, c, 81),
        blockChi(grad_H_terms, atom, c, kSpinDeg4Count),
        blockChi(grad_H_derivatives, atom, c, 3 * kSpinDeg3Count));
    }
  }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_edges)
#endif
  for (std::ptrdiff_t edge_index = 0; edge_index < static_cast<std::ptrdiff_t>(edge_count); ++edge_index) {
    const std::size_t e = static_cast<std::size_t>(edge_index);
    const SpinEdge& edge = cache.edges[e];
    double m2[kSpinDeg2Count];
    double m3[kSpinDeg3Count];
    double m4[kSpinDeg4Count];
    fill_spin_monomials(edge.rhat, m2, m3, m4);
    for (int c = 0; c < C; ++c) {
      const double* q_terms = cblockC(grad_Q_terms, edge.center, c, kSpinDeg2Count);
      const double dot = dot_spin_terms(q_terms, m2, kSpinDeg2Count);
      const double w = edge.weights[c];
      egw(e, c) += dot;
      std::array<double, 3> gu = {
        w * (2.0 * q_terms[0] * edge.rhat[0] + q_terms[3] * edge.rhat[1] + q_terms[4] * edge.rhat[2]),
        w * (2.0 * q_terms[1] * edge.rhat[1] + q_terms[3] * edge.rhat[0] + q_terms[5] * edge.rhat[2]),
        w * (2.0 * q_terms[2] * edge.rhat[2] + q_terms[4] * edge.rhat[0] + q_terms[5] * edge.rhat[1])};
      add_edge_vec(grad_rhat, e, gu);
    }
    for (int c = 0; c < chiC; ++c) {
      const double* o_terms = cblockChi(grad_O_terms, edge.center, c, kSpinDeg3Count);
      const double* o_derivatives =
        cblockChi(grad_O_derivatives, edge.center, c, 3 * kSpinDeg2Count);
      const double* h_terms = cblockChi(grad_H_terms, edge.center, c, kSpinDeg4Count);
      const double* h_derivatives =
        cblockChi(grad_H_derivatives, edge.center, c, 3 * kSpinDeg3Count);
      std::array<double, 3> gu = {0.0, 0.0, 0.0};
      const double dot_o = dot_spin_terms(o_terms, m3, kSpinDeg3Count);
      const double dot_h = dot_spin_terms(h_terms, m4, kSpinDeg4Count);
      for (int d = 0; d < 3; ++d) {
        gu[d] += edge.weights[c] * (
          dot_spin_terms(o_derivatives + d * kSpinDeg2Count, m2, kSpinDeg2Count) +
          dot_spin_terms(h_derivatives + d * kSpinDeg3Count, m3, kSpinDeg3Count));
      }
      egw(e, c) += dot_o + dot_h;
      add_edge_vec(grad_rhat, e, gu);
    }
  }

  const bool use_private_edges = use_parallel_edges || use_lammps_scratch;
  const int force_stride = use_lammps_scratch ? lammps_scratch->force_rows : N;
  std::vector<double> force_private(
    use_private_edges && !use_lammps_scratch ? static_cast<std::size_t>(num_threads) * 3 * N : 0,
    0.0);
  std::vector<double> grad_spin_private(
    use_private_edges && !use_lammps_scratch ? static_cast<std::size_t>(num_threads) * 3 * N : 0,
    0.0);
  std::vector<double> spin_transfer_private(
    spin_transfer && use_private_edges && !use_lammps_scratch
      ? static_cast<std::size_t>(num_threads) * 9 * N
      : 0,
    0.0);
  std::vector<double> virial_private(
    use_private_edges && !use_lammps_scratch ? static_cast<std::size_t>(num_threads) * 9 : 0,
    0.0);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_private_edges)
#endif
  for (std::ptrdiff_t edge_index = 0; edge_index < static_cast<std::ptrdiff_t>(edge_count); ++edge_index) {
    const std::size_t e = static_cast<std::size_t>(edge_index);
    const SpinEdge& edge = cache.edges[e];
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    double* local_force = nullptr;
    double* local_grad_spin = nullptr;
    double* local_virial = nullptr;
    double* local_total_virial = nullptr;
    double* local_spin_transfer = nullptr;
    if (use_lammps_scratch) {
      local_force =
        lammps_scratch->force_private + static_cast<std::size_t>(tid) * 3 * force_stride;
      local_grad_spin =
        lammps_scratch->mforce_private + static_cast<std::size_t>(tid) * 3 * force_stride;
      local_total_virial =
        lammps_scratch->total_virial_private +
        static_cast<std::size_t>(tid) * kLammpsTotalVirialStride;
      local_virial = lammps_scratch->virial_private
        ? lammps_scratch->virial_private + static_cast<std::size_t>(tid) * 9 * force_stride
        : nullptr;
      local_spin_transfer = lammps_scratch->spin_transfer_private
        ? lammps_scratch->spin_transfer_private +
            static_cast<std::size_t>(tid) * 9 * force_stride
        : nullptr;
    } else {
      local_force = use_private_edges
        ? force_private.data() + static_cast<std::size_t>(tid) * 3 * N
        : force;
      local_grad_spin = use_private_edges
        ? grad_spin_private.data() + static_cast<std::size_t>(tid) * 3 * N
        : grad_spin.data();
      local_virial = use_private_edges
        ? virial_private.data() + static_cast<std::size_t>(tid) * 9
        : nullptr;
      local_spin_transfer = spin_transfer && use_private_edges
        ? spin_transfer_private.data() + static_cast<std::size_t>(tid) * 9 * N
        : spin_transfer;
      local_spin_transfer = spin_transfer && use_private_edges
        ? spin_transfer_private.data() + static_cast<std::size_t>(tid) * 9 * N
        : spin_transfer;
    }
    double grad_dist = 0.0;
    for (int c = 0; c < C; ++c) {
      grad_dist += grad_weight[e * C + c] * edge.weight_derivatives[c];
    }
    std::array<double, 3> gu = {
      grad_rhat[e * 3 + 0], grad_rhat[e * 3 + 1], grad_rhat[e * 3 + 2]};
    double dot_r = dot3(gu, edge.rhat);
    std::array<double, 3> grad_rij = {0.0, 0.0, 0.0};
    for (int d = 0; d < 3; ++d) {
      grad_rij[d] = grad_dist * edge.rhat[d] + (gu[d] - dot_r * edge.rhat[d]) / edge.dist;
      if (use_lammps_scratch) {
        local_force[lammps_vector_index(edge.i, d, force_stride)] += grad_rij[d];
        local_force[lammps_vector_index(edge.j, d, force_stride)] -= grad_rij[d];
        if (!direct_lammps_spin_pull) {
          local_grad_spin[lammps_vector_index(edge.i, d, force_stride)] += grad_si[e * 3 + d];
          local_grad_spin[lammps_vector_index(edge.j, d, force_stride)] += grad_sj[e * 3 + d];
        }
      } else {
        local_force[static_cast<std::size_t>(d) * force_stride + edge.i] += grad_rij[d];
        local_force[static_cast<std::size_t>(d) * force_stride + edge.j] -= grad_rij[d];
        local_grad_spin[static_cast<std::size_t>(d) * force_stride + edge.i] += grad_si[e * 3 + d];
        local_grad_spin[static_cast<std::size_t>(d) * force_stride + edge.j] += grad_sj[e * 3 + d];
      }
    }
    if (use_lammps_scratch) {
      if (local_virial) {
        add_lammps_spin_virial(
          edge.rhat, edge.dist, grad_rij, local_total_virial, local_virial,
          force_stride, edge.j);
      } else {
        add_lammps_spin_total_virial(
          edge.rhat, edge.dist, grad_rij, local_total_virial);
      }
    } else {
      for (int a = 0; a < 3; ++a) {
        const double rij_a = edge.rhat[a] * edge.dist;
        for (int b = 0; b < 3; ++b) {
          if (use_private_edges) {
            local_virial[a * 3 + b] -= rij_a * grad_rij[b];
          } else {
            const int owner = lammps_neighbor_virial_ownership ? edge.j : 0;
            virial[static_cast<std::size_t>(a * 3 + b) * N + owner] -=
              rij_a * grad_rij[b];
          }
        }
      }
    }
    if (!direct_lammps_spin_pull) {
      if (use_lammps_scratch) {
        add_spin_transfer_row_major9(
          edge.rhat, edge.dist,
          {grad_sj[e * 3 + 0], grad_sj[e * 3 + 1], grad_sj[e * 3 + 2]},
          local_spin_transfer, force_stride, edge.j);
      } else {
        add_spin_transfer_soa9(
          N, edge.j, edge.rhat, edge.dist,
          {grad_sj[e * 3 + 0], grad_sj[e * 3 + 1], grad_sj[e * 3 + 2]},
          local_spin_transfer);
      }
    }
  }

  if (use_private_edges && !use_lammps_scratch) {
    for (int tid = 0; tid < num_threads; ++tid) {
      const double* local_force = force_private.data() + static_cast<std::size_t>(tid) * 3 * N;
      const double* local_grad_spin =
        grad_spin_private.data() + static_cast<std::size_t>(tid) * 3 * N;
      const double* local_virial = virial_private.data() + static_cast<std::size_t>(tid) * 9;
      const double* local_spin_transfer = spin_transfer
        ? spin_transfer_private.data() + static_cast<std::size_t>(tid) * 9 * N
        : nullptr;
      for (int atom = 0; atom < N; ++atom) {
        for (int d = 0; d < 3; ++d) {
          const std::size_t idx = static_cast<std::size_t>(d) * N + atom;
          force[idx] += local_force[idx];
          grad_spin[idx] += local_grad_spin[idx];
        }
      }
      for (int d = 0; d < 9; ++d) {
        virial[static_cast<std::size_t>(d) * N] += local_virial[d];
        if (spin_transfer) {
          for (int atom = 0; atom < N; ++atom) {
            spin_transfer[static_cast<std::size_t>(d) * N + atom] +=
              local_spin_transfer[static_cast<std::size_t>(d) * N + atom];
          }
        }
      }
    }
  }
}

template <bool SpinsAos3>
void add_spin_gradient_lammps_single_center_nonchiral(
  const NEP::ParaMB& paramb,
  const NEP::ANN& annmb,
  const int N,
  const int* type,
  const double* spins,
  const SpinCache& cache,
  const double* Fp,
  LammpsThreadLocalScratchView* lammps_scratch,
  const int atom,
  std::vector<double>* chiral_grad_weight_work = nullptr,
  std::vector<double>* chiral_grad_rhat_work = nullptr)
{
  const int C = paramb.spin_compress;
  const int l_max = paramb.spin_l_max;
  const int offset0 = paramb.struct_dim;
  const int force_stride = lammps_scratch->force_rows;
  auto fp = [&](const int atom_index, const int dim) {
    return Fp[static_cast<std::size_t>(atom_index) * annmb.dim + offset0 + dim];
  };
  auto spin = [&](const int atom_index, const int d) {
    if constexpr (SpinsAos3) {
      (void)N;
      return spins[static_cast<std::size_t>(atom_index) * 3 + d];
    } else {
      return spins[static_cast<std::size_t>(d) * N + atom_index];
    }
  };
  auto mask_all_active = [](const std::vector<int>& mask) {
    return mask.empty() || std::all_of(mask.begin(), mask.end(), [](const int value) {
      return value != 0;
    });
  };
  const bool spin_dof_all_active = mask_all_active(paramb.spin_dof_type_active);
  auto spin_dof_active = [&](const int t) {
    return spin_dof_all_active || paramb.spin_dof_type_active[static_cast<std::size_t>(t)] != 0;
  };
  auto block = [&](const std::vector<double>& v, const int c, const int width) {
    return v.data() + static_cast<std::size_t>(c) * width;
  };
  auto local_block = [](double* v, const int c, const int width) {
    return v + c * width;
  };

  double* local_force = lammps_scratch->force_private;
  double* local_grad_spin = lammps_scratch->mforce_private;
  double* local_total_virial = lammps_scratch->total_virial_private;
  double* local_virial = lammps_scratch->virial_private;
  auto add_local_grad_spin = [&](const int atom_index, const std::array<double, 3>& g) {
    for (int d = 0; d < 3; ++d) {
      local_grad_spin[lammps_vector_index(atom_index, d, force_stride)] += g[d];
    }
  };
  std::vector<double> local_chiral_grad_weight;
  std::vector<double> local_chiral_grad_rhat;
  std::vector<double>& chiral_grad_weight =
    chiral_grad_weight_work ? *chiral_grad_weight_work : local_chiral_grad_weight;
  std::vector<double>& chiral_grad_rhat =
    chiral_grad_rhat_work ? *chiral_grad_rhat_work : local_chiral_grad_rhat;
  const bool fuse_chiral = paramb.spin_chiral &&
    add_spin_chiral_gradient_lammps_single_center_deferred<SpinsAos3>(
      paramb, annmb, N, spins, cache, Fp, lammps_scratch,
      chiral_grad_weight, chiral_grad_rhat);

  const std::array<double, 3> s = {spin(atom, 0), spin(atom, 1), spin(atom, 2)};
  const int rho0_offset = 2 + 4 * C;
  const int l1_rdot_offset = rho0_offset + C;
  const int l1_cross_offset = l1_rdot_offset + C;
  const int l1_stf_offset = l1_cross_offset + C;
  int angular_offset = rho0_offset + C;
  if (l_max >= 1) {
    angular_offset += 3 * C;
  }
  int geom_offset = angular_offset;
  for (int ell = 2; ell <= l_max; ++ell) {
    geom_offset += C;
  }
  const int rho0_dot_offset = geom_offset + C;
  const int raw1_dot_offset = rho0_dot_offset + C;

  if (spin_dof_active(type[atom])) {
    const double s2 = dot3(s, s);
    const double scale = 2.0 * fp(atom, 0) + 4.0 * fp(atom, 1) * s2;
    add_local_grad_spin(atom, {scale * s[0], scale * s[1], scale * s[2]});
    for (int c = 0; c < C; ++c) {
      const double alpha = fp(atom, geom_offset + c);
      const double* g = block(cache.geom, c, 9);
      std::array<double, 3> gs = {0.0, 0.0, 0.0};
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          gs[a] += (g[3 * a + b] + g[3 * b + a]) * s[b];
        }
        gs[a] *= alpha;
      }
      add_local_grad_spin(atom, gs);
    }
  }
  double rho0_pull[MAX_SPIN_COMPRESS * 3] = {0.0};
  double rho0_dot_pull[MAX_SPIN_COMPRESS * 3] = {0.0};
  double l1_pull[MAX_SPIN_COMPRESS * 9] = {0.0};
  double l1_dot_pull[MAX_SPIN_COMPRESS * 9] = {0.0};
  double geom_pull[MAX_SPIN_COMPRESS * 9] = {0.0};
  for (int c = 0; c < C; ++c) {
    const double alpha0 = fp(atom, rho0_offset + c);
    const double alpha0_dot = fp(atom, rho0_dot_offset + c);
    const double* rho = block(cache.rho0, c, 3);
    const double* rho_dot = block(cache.rho0_dot, c, 3);
    double* rho_out = local_block(rho0_pull, c, 3);
    double* rho_dot_out = local_block(rho0_dot_pull, c, 3);
    for (int d = 0; d < 3; ++d) {
      rho_out[d] = 2.0 * alpha0 * rho[d] + alpha0_dot * rho_dot[d];
      rho_dot_out[d] = alpha0_dot * rho[d];
    }

    const double alpha_geom = fp(atom, geom_offset + c);
    const auto ss = stf_outer3(s, s);
    double* geom_out = local_block(geom_pull, c, 9);
    for (int k = 0; k < 9; ++k) {
      geom_out[k] = alpha_geom * ss[k];
    }
  }

  if (l_max >= 1) {
    for (int c = 0; c < C; ++c) {
      double* mat = local_block(l1_pull, c, 9);
      double* mat_dot = local_block(l1_dot_pull, c, 9);
      const double alpha_rdot = fp(atom, l1_rdot_offset + c);
      const double alpha_cross = fp(atom, l1_cross_offset + c);
      const double alpha_stf = fp(atom, l1_stf_offset + c);
      const double alpha_raw1 = fp(atom, raw1_dot_offset + c);
      const double rdot = *block(cache.l1_rdot, c, 1);
      const double* cross = block(cache.l1_cross, c, 3);
      const double* stf = block(cache.l1_stf, c, 9);
      const double* raw = block(cache.raw1, c, 9);
      const double* raw_dot = block(cache.raw1_dot, c, 9);
      const double g_cross[3] = {
        2.0 * alpha_cross * cross[0],
        2.0 * alpha_cross * cross[1],
        2.0 * alpha_cross * cross[2]};
      mat[0] += 2.0 * alpha_rdot * rdot;
      mat[4] += 2.0 * alpha_rdot * rdot;
      mat[8] += 2.0 * alpha_rdot * rdot;
      mat[1] += g_cross[2];
      mat[2] -= g_cross[1];
      mat[3] -= g_cross[2];
      mat[5] += g_cross[0];
      mat[6] += g_cross[1];
      mat[7] -= g_cross[0];
      for (int k = 0; k < 9; ++k) {
        mat[k] += 2.0 * alpha_stf * stf[k] + alpha_raw1 * raw_dot[k];
        mat_dot[k] = alpha_raw1 * raw[k];
      }
    }
  }

  for (int edge_index = cache.edge_offsets[0]; edge_index < cache.edge_offsets[1]; ++edge_index) {
    const SpinEdge& edge = cache.edges[static_cast<std::size_t>(edge_index)];
    const std::array<double, 3> si = {
      spin(edge.i, 0), spin(edge.i, 1), spin(edge.i, 2)};
    const std::array<double, 3> sj = {
      spin(edge.j, 0), spin(edge.j, 1), spin(edge.j, 2)};
    std::array<double, MAX_NUM_N> grad_weight;
    grad_weight.fill(0.0);
    std::array<double, 3> grad_rhat = {0.0, 0.0, 0.0};
    std::array<double, 3> grad_si = {0.0, 0.0, 0.0};
    std::array<double, 3> grad_sj = {0.0, 0.0, 0.0};
    double grad_dot = 0.0;

    auto add_scalar = [&](const int offset, const double scalar) {
      double g = 0.0;
      for (int c = 0; c < C; ++c) {
        const double a = fp(edge.i, offset + c);
        grad_weight[c] += a * scalar;
        g += a * edge.weights[c];
      }
      return g;
    };

    int offset = 2;
    grad_dot += add_scalar(offset, edge.dot);
    offset += C;
    grad_dot += 2.0 * edge.dot * add_scalar(offset, edge.dot * edge.dot);
    offset += C;
    const double g_sj2 = add_scalar(offset, edge.sj2);
    for (int d = 0; d < 3; ++d) {
      grad_sj[d] += 2.0 * g_sj2 * sj[d];
    }
    offset += C;
    const double g_axis = add_scalar(offset, edge.bond_axis);
    for (int d = 0; d < 3; ++d) {
      grad_si[d] += g_axis * edge.ri_dot_sj * edge.rhat[d];
      grad_sj[d] += g_axis * edge.ri_dot_si * edge.rhat[d];
      grad_rhat[d] += g_axis * (edge.ri_dot_sj * si[d] + edge.ri_dot_si * sj[d]);
    }
    offset += C;

    auto add_angular_density_gradient = [&](
      const int width,
      const std::vector<double>& density,
      const double* value,
      const int q_offset,
      double* grad_value) {
      if (C == 4) {
        const double* self0 = block(density, 0, width);
        const double* self1 = block(density, 1, width);
        const double* self2 = block(density, 2, width);
        const double* self3 = block(density, 3, width);
        const double alpha0 = 2.0 * fp(edge.i, q_offset + 0);
        const double alpha1 = 2.0 * fp(edge.i, q_offset + 1);
        const double alpha2 = 2.0 * fp(edge.i, q_offset + 2);
        const double alpha3 = 2.0 * fp(edge.i, q_offset + 3);
        const double w0 = edge.weights[0];
        const double w1 = edge.weights[1];
        const double w2 = edge.weights[2];
        const double w3 = edge.weights[3];
        double gw0 = grad_weight[0];
        double gw1 = grad_weight[1];
        double gw2 = grad_weight[2];
        double gw3 = grad_weight[3];
#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp simd reduction(+:gw0, gw1, gw2, gw3)
#endif
        for (int k = 0; k < width; ++k) {
          const double v = value[k];
          const double gd0 = alpha0 * self0[k];
          const double gd1 = alpha1 * self1[k];
          const double gd2 = alpha2 * self2[k];
          const double gd3 = alpha3 * self3[k];
          gw0 += gd0 * v;
          gw1 += gd1 * v;
          gw2 += gd2 * v;
          gw3 += gd3 * v;
          grad_value[k] = gd0 * w0 + gd1 * w1 + gd2 * w2 + gd3 * w3;
        }
        grad_weight[0] = gw0;
        grad_weight[1] = gw1;
        grad_weight[2] = gw2;
        grad_weight[3] = gw3;
        return;
      }
      for (int c = 0; c < C; ++c) {
        const double* self = block(density, c, width);
        const double alpha2 = 2.0 * fp(edge.i, q_offset + c);
        const double w = edge.weights[c];
        double grad_weight_c = grad_weight[c];
        for (int k = 0; k < width; ++k) {
          const double gd = alpha2 * self[k];
          grad_weight_c += gd * value[k];
          grad_value[k] += gd * w;
        }
        grad_weight[c] = grad_weight_c;
      }
    };

    auto apply_angular = [&](const int ell, const double* ylm, const int ylm_width, const double* ge) {
      double grad_ylm[9] = {0.0};
      for (int m = 0; m < ylm_width; ++m) {
        const int base = m * 3;
        const double y = ylm[m];
        const double g0 = ge[base + 0];
        const double g1 = ge[base + 1];
        const double g2 = ge[base + 2];
        grad_sj[0] += g0 * y;
        grad_sj[1] += g1 * y;
        grad_sj[2] += g2 * y;
        grad_ylm[m] += g0 * sj[0] + g1 * sj[1] + g2 * sj[2];
      }
      add_real_spherical_harmonics_gradient(edge.rhat, ell, grad_ylm, grad_rhat);
    };

    for (int c = 0; c < C; ++c) {
      const double* b = local_block(rho0_pull, c, 3);
      const double* b_dot = local_block(rho0_dot_pull, c, 3);
      const double u0 = b[0] + edge.dot * b_dot[0];
      const double u1 = b[1] + edge.dot * b_dot[1];
      const double u2 = b[2] + edge.dot * b_dot[2];
      const double w = edge.weights[c];
      grad_weight[c] += sj[0] * u0 + sj[1] * u1 + sj[2] * u2;
      grad_sj[0] += w * u0;
      grad_sj[1] += w * u1;
      grad_sj[2] += w * u2;
      grad_dot += w * (sj[0] * b_dot[0] + sj[1] * b_dot[1] + sj[2] * b_dot[2]);
    }
    offset += C;

    for (int c = 0; c < C; ++c) {
      const double* K = local_block(geom_pull, c, 9);
      const double kr0 = K[0] * edge.rhat[0] + K[1] * edge.rhat[1] + K[2] * edge.rhat[2];
      const double kr1 = K[3] * edge.rhat[0] + K[4] * edge.rhat[1] + K[5] * edge.rhat[2];
      const double kr2 = K[6] * edge.rhat[0] + K[7] * edge.rhat[1] + K[8] * edge.rhat[2];
      double u0 = 0.0;
      double u1 = 0.0;
      double u2 = 0.0;
      double mt0 = 0.0;
      double mt1 = 0.0;
      double mt2 = 0.0;
      double dot_v2 = 0.0;
      if (l_max >= 1) {
        const double* M = local_block(l1_pull, c, 9);
        const double* Md = local_block(l1_dot_pull, c, 9);
        const double v10 = M[0] * sj[0] + M[1] * sj[1] + M[2] * sj[2];
        const double v11 = M[3] * sj[0] + M[4] * sj[1] + M[5] * sj[2];
        const double v12 = M[6] * sj[0] + M[7] * sj[1] + M[8] * sj[2];
        const double v20 = Md[0] * sj[0] + Md[1] * sj[1] + Md[2] * sj[2];
        const double v21 = Md[3] * sj[0] + Md[4] * sj[1] + Md[5] * sj[2];
        const double v22 = Md[6] * sj[0] + Md[7] * sj[1] + Md[8] * sj[2];
        u0 = v10 + edge.dot * v20;
        u1 = v11 + edge.dot * v21;
        u2 = v12 + edge.dot * v22;
        mt0 = M[0] * edge.rhat[0] + M[3] * edge.rhat[1] + M[6] * edge.rhat[2] +
              edge.dot * (Md[0] * edge.rhat[0] + Md[3] * edge.rhat[1] + Md[6] * edge.rhat[2]);
        mt1 = M[1] * edge.rhat[0] + M[4] * edge.rhat[1] + M[7] * edge.rhat[2] +
              edge.dot * (Md[1] * edge.rhat[0] + Md[4] * edge.rhat[1] + Md[7] * edge.rhat[2]);
        mt2 = M[2] * edge.rhat[0] + M[5] * edge.rhat[1] + M[8] * edge.rhat[2] +
              edge.dot * (Md[2] * edge.rhat[0] + Md[5] * edge.rhat[1] + Md[8] * edge.rhat[2]);
        dot_v2 = edge.rhat[0] * v20 + edge.rhat[1] * v21 + edge.rhat[2] * v22;
      }
      const double w = edge.weights[c];
      grad_weight[c] += edge.rhat[0] * (u0 + kr0) + edge.rhat[1] * (u1 + kr1) +
                        edge.rhat[2] * (u2 + kr2);
      grad_rhat[0] += w * (u0 + 2.0 * kr0);
      grad_rhat[1] += w * (u1 + 2.0 * kr1);
      grad_rhat[2] += w * (u2 + 2.0 * kr2);
      grad_sj[0] += w * mt0;
      grad_sj[1] += w * mt1;
      grad_sj[2] += w * mt2;
      grad_dot += w * dot_v2;
    }
    if (l_max >= 1) {
      offset += 3 * C;
    }
    for (int ell = 2; ell <= l_max; ++ell) {
      const std::vector<double>& angular = ell == 2 ? cache.angular2 : ell == 3 ? cache.angular3 : cache.angular4;
      double ylm[9];
      const int ylm_width = real_spherical_harmonics_spin(edge.rhat, ell, ylm);
      double value[27];
      int width = 0;
      for (int m = 0; m < ylm_width; ++m) {
        const double y = ylm[m];
        value[width++] = y * sj[0];
        value[width++] = y * sj[1];
        value[width++] = y * sj[2];
      }
      double ge[27];
      if (C != 4) {
        std::fill(ge, ge + width, 0.0);
      }
      add_angular_density_gradient(width, angular, value, offset, ge);
      apply_angular(ell, ylm, ylm_width, ge);
      offset += C;
    }

    offset += C;
    offset += C;
    if (l_max >= 1) {
      offset += C;
    }

    for (int d = 0; d < 3; ++d) {
      grad_si[d] += grad_dot * sj[d];
      grad_sj[d] += grad_dot * si[d];
    }
    if (fuse_chiral) {
      const int le = edge_index - cache.edge_offsets[0];
      const double* gw = chiral_grad_weight.data() + static_cast<std::size_t>(le) * C;
      const double* gr = chiral_grad_rhat.data() + static_cast<std::size_t>(le) * 3;
      for (int c = 0; c < C; ++c) {
        grad_weight[c] += gw[c];
      }
      for (int d = 0; d < 3; ++d) {
        grad_rhat[d] += gr[d];
      }
    }

    double grad_dist = 0.0;
    for (int c = 0; c < C; ++c) {
      grad_dist += grad_weight[c] * edge.weight_derivatives[c];
    }

    std::array<double, 3> grad_rij;
    double dot_r = 0.0;
    for (int d = 0; d < 3; ++d) {
      dot_r += grad_rhat[d] * edge.rhat[d];
    }
    for (int d = 0; d < 3; ++d) {
      grad_rij[d] = grad_dist * edge.rhat[d] + (grad_rhat[d] - dot_r * edge.rhat[d]) / edge.dist;
      local_force[lammps_vector_index(edge.i, d, force_stride)] += grad_rij[d];
      local_force[lammps_vector_index(edge.j, d, force_stride)] -= grad_rij[d];
    }
    if (local_virial) {
      add_lammps_spin_virial(
        edge.rhat, edge.dist, grad_rij, local_total_virial, local_virial,
        force_stride, edge.j);
    } else {
      add_lammps_spin_total_virial(edge.rhat, edge.dist, grad_rij, local_total_virial);
    }
    add_local_grad_spin(edge.i, grad_si);
    add_local_grad_spin(edge.j, grad_sj);
    add_spin_transfer_row_major9(
      edge.rhat, edge.dist, grad_sj, lammps_scratch->spin_transfer_private,
      force_stride, edge.j);
  }
}

void add_spin_gradient(
  const NEP::ParaMB& paramb,
  const NEP::ANN& annmb,
  const int N,
  const int* type,
  const double* spins,
  const SpinCache& cache,
  const double* Fp,
  double* force,
  double* virial,
  double* mforce,
  double* spin_transfer,
  SpinPhaseBreakdown* phase = nullptr,
  double** lammps_force = nullptr,
  double** lammps_mforce = nullptr,
  double* lammps_total_virial = nullptr,
  double** lammps_virial = nullptr,
  double** lammps_spin_transfer = nullptr,
  LammpsThreadLocalScratchView* lammps_scratch = nullptr,
  NEP::SpinGradientScratch* scratch = nullptr,
  int center_count = 0,
  const int* centers = nullptr)
{
  auto phase_mark = NepPhaseClock::now();
  const int C = paramb.spin_compress;
  const int l_max = paramb.spin_l_max;
  const int offset0 = paramb.struct_dim;
  const bool lammps_output = lammps_force && lammps_mforce && lammps_total_virial;
  const bool use_lammps_scratch = lammps_output && lammps_spin_scratch_active(lammps_scratch);
  const bool lammps_neighbor_virial_ownership = lammps_output && !use_lammps_scratch;
  std::vector<double> lammps_force_work;
  std::vector<double> lammps_virial_work;
  std::vector<double> lammps_spin_transfer_work;
  if (lammps_output && !use_lammps_scratch) {
    lammps_force_work.assign(static_cast<std::size_t>(N) * 3, 0.0);
    lammps_virial_work.assign(static_cast<std::size_t>(N) * 9, 0.0);
    force = lammps_force_work.data();
    virial = lammps_virial_work.data();
    if (lammps_spin_transfer) {
      lammps_spin_transfer_work.assign(static_cast<std::size_t>(N) * 9, 0.0);
      spin_transfer = lammps_spin_transfer_work.data();
    }
  }
  std::vector<double> grad_spin;
  if (!use_lammps_scratch) {
    grad_spin.assign(static_cast<std::size_t>(N) * 3, 0.0);
  }
  auto add_spin_pull = [&](const int atom, const int d, const double value) {
    if (use_lammps_scratch) {
      lammps_mforce[atom][d] -= value;
    } else {
      grad_spin[static_cast<std::size_t>(d) * N + atom] += value;
    }
  };
  auto fp = [&](const int atom, const int dim) {
    return Fp[static_cast<std::size_t>(atom) * annmb.dim + offset0 + dim];
  };
  auto spin = [&](const int atom, const int d) {
    return spins[static_cast<std::size_t>(d) * N + atom];
  };
  auto mask_all_active = [](const std::vector<int>& mask) {
    return mask.empty() || std::all_of(mask.begin(), mask.end(), [](const int value) {
      return value != 0;
    });
  };
  const bool spin_dof_all_active = mask_all_active(paramb.spin_dof_type_active);
  auto spin_dof_active = [&](const int t) {
    return spin_dof_all_active || paramb.spin_dof_type_active[static_cast<std::size_t>(t)] != 0;
  };
  auto block = [&](const std::vector<double>& v, const int atom, const int c, const int width) {
    return v.data() + (static_cast<std::size_t>(atom) * C + c) * width;
  };
  const bool use_center_atoms = centers && center_count > 0;
  const int atom_loop_count = use_center_atoms ? center_count : N;
  const int density_rows = atom_loop_count;
  auto loop_atom = [&](const int idx) {
    return use_center_atoms ? centers[idx] : idx;
  };

#if defined(_OPENMP)
  const int num_threads = spin_openmp_threads();
#else
  const int num_threads = 1;
#endif
  const bool use_parallel_atoms = num_threads > 1 && atom_loop_count > 256;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_atoms)
#endif
  for (int idx = 0; idx < atom_loop_count; ++idx) {
    const int atom = loop_atom(idx);
    if (!spin_dof_active(type[atom])) {
      continue;
    }
    const double sx = spin(atom, 0);
    const double sy = spin(atom, 1);
    const double sz = spin(atom, 2);
    const double s2 = sx * sx + sy * sy + sz * sz;
    const double scale = 2.0 * fp(atom, 0) + 4.0 * fp(atom, 1) * s2;
    add_spin_pull(atom, 0, scale * sx);
    add_spin_pull(atom, 1, scale * sy);
    add_spin_pull(atom, 2, scale * sz);
  }

  const int rho0_offset = 2 + 4 * C;
  const int l1_rdot_offset = rho0_offset + C;
  const int l1_cross_offset = l1_rdot_offset + C;
  const int l1_stf_offset = l1_cross_offset + C;
  int angular_offset = rho0_offset + C;
  if (l_max >= 1) {
    angular_offset += 3 * C;
  }
  int geom_offset = angular_offset;
  for (int ell = 2; ell <= l_max; ++ell) {
    geom_offset += C;
  }
  const int rho0_dot_offset = geom_offset + C;
  const int raw1_dot_offset = rho0_dot_offset + C;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_atoms)
#endif
  for (int idx = 0; idx < atom_loop_count; ++idx) {
    const int atom = loop_atom(idx);
    const int row = idx;
    if (!spin_dof_active(type[atom])) {
      continue;
    }
    const std::array<double, 3> s = {spin(atom, 0), spin(atom, 1), spin(atom, 2)};
    for (int c = 0; c < C; ++c) {
      const double alpha = fp(atom, geom_offset + c);
      const double* g = block(cache.geom, row, c, 9);
      for (int a = 0; a < 3; ++a) {
        double gs = 0.0;
        for (int b = 0; b < 3; ++b) {
          gs += (g[3 * a + b] + g[3 * b + a]) * s[b];
        }
        add_spin_pull(atom, a, alpha * gs);
      }
    }
  }

  std::vector<double> rho0_pull(static_cast<std::size_t>(density_rows) * C * 3, 0.0);
  std::vector<double> rho0_dot_pull(static_cast<std::size_t>(density_rows) * C * 3, 0.0);
  std::vector<double> l1_pull(l_max >= 1 ? static_cast<std::size_t>(density_rows) * C * 9 : 0, 0.0);
  std::vector<double> l1_dot_pull(l_max >= 1 ? static_cast<std::size_t>(density_rows) * C * 9 : 0, 0.0);
  std::vector<double> geom_pull(static_cast<std::size_t>(density_rows) * C * 9, 0.0);

  auto mutable_block = [&](std::vector<double>& v, const int atom, const int c, const int width) {
    return v.data() + (static_cast<std::size_t>(atom) * C + c) * width;
  };

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_atoms)
#endif
  for (int idx = 0; idx < atom_loop_count; ++idx) {
    const int atom = loop_atom(idx);
    const int row = idx;
    for (int c = 0; c < C; ++c) {
      const double alpha0 = fp(atom, rho0_offset + c);
      const double alpha0_dot = fp(atom, rho0_dot_offset + c);
      const double* rho = block(cache.rho0, row, c, 3);
      const double* rho_dot = block(cache.rho0_dot, row, c, 3);
      double* rho_out = mutable_block(rho0_pull, row, c, 3);
      double* rho_dot_out = mutable_block(rho0_dot_pull, row, c, 3);
      for (int d = 0; d < 3; ++d) {
        rho_out[d] = 2.0 * alpha0 * rho[d] + alpha0_dot * rho_dot[d];
        rho_dot_out[d] = alpha0_dot * rho[d];
      }

      const double alpha_geom = fp(atom, geom_offset + c);
      const std::array<double, 3> s = {spin(atom, 0), spin(atom, 1), spin(atom, 2)};
      const auto ss = stf_outer3(s, s);
      double* geom_out = mutable_block(geom_pull, row, c, 9);
      for (int k = 0; k < 9; ++k) {
        geom_out[k] = alpha_geom * ss[k];
      }
    }
  }

  if (l_max >= 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_atoms)
#endif
    for (int idx = 0; idx < atom_loop_count; ++idx) {
      const int atom = loop_atom(idx);
      const int row = idx;
      for (int c = 0; c < C; ++c) {
        double* mat = mutable_block(l1_pull, row, c, 9);
        double* mat_dot = mutable_block(l1_dot_pull, row, c, 9);
        const double alpha_rdot = fp(atom, l1_rdot_offset + c);
        const double alpha_cross = fp(atom, l1_cross_offset + c);
        const double alpha_stf = fp(atom, l1_stf_offset + c);
        const double alpha_raw1 = fp(atom, raw1_dot_offset + c);
        const double rdot = *block(cache.l1_rdot, row, c, 1);
        const double* cross = block(cache.l1_cross, row, c, 3);
        const double* stf = block(cache.l1_stf, row, c, 9);
        const double* raw = block(cache.raw1, row, c, 9);
        const double* raw_dot = block(cache.raw1_dot, row, c, 9);
        const double g_cross[3] = {
          2.0 * alpha_cross * cross[0],
          2.0 * alpha_cross * cross[1],
          2.0 * alpha_cross * cross[2]};
        mat[0] += 2.0 * alpha_rdot * rdot;
        mat[4] += 2.0 * alpha_rdot * rdot;
        mat[8] += 2.0 * alpha_rdot * rdot;
        mat[1] += g_cross[2];
        mat[2] -= g_cross[1];
        mat[3] -= g_cross[2];
        mat[5] += g_cross[0];
        mat[6] += g_cross[1];
        mat[7] -= g_cross[0];
        for (int k = 0; k < 9; ++k) {
          mat[k] += 2.0 * alpha_stf * stf[k] + alpha_raw1 * raw_dot[k];
          mat_dot[k] = alpha_raw1 * raw[k];
        }
      }
    }
  }

  const bool use_parallel_edges = num_threads > 1 && cache.edges.size() > 32;
  const bool use_private_edges = use_parallel_edges || use_lammps_scratch;
  const int force_stride = use_lammps_scratch ? lammps_scratch->force_rows : N;
  std::vector<double> force_private(
    use_private_edges && !use_lammps_scratch ? static_cast<std::size_t>(num_threads) * 3 * N : 0,
    0.0);
  std::vector<double> grad_spin_private(
    use_private_edges && !use_lammps_scratch ? static_cast<std::size_t>(num_threads) * 3 * N : 0,
    0.0);
  std::vector<double> spin_transfer_private(
    spin_transfer && use_private_edges && !use_lammps_scratch
      ? static_cast<std::size_t>(num_threads) * 9 * N
      : 0,
    0.0);
  std::vector<double> virial_private(
    use_private_edges && !use_lammps_scratch ? static_cast<std::size_t>(num_threads) * 9 : 0,
    0.0);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_private_edges)
#endif
  for (std::ptrdiff_t edge_index = 0; edge_index < static_cast<std::ptrdiff_t>(cache.edges.size()); ++edge_index) {
    const SpinEdge& edge = cache.edges[static_cast<std::size_t>(edge_index)];
    const std::array<double, 3> si = {
      spin(edge.i, 0), spin(edge.i, 1), spin(edge.i, 2)};
    const std::array<double, 3> sj = {
      spin(edge.j, 0), spin(edge.j, 1), spin(edge.j, 2)};
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    double* local_force = nullptr;
    double* local_grad_spin = nullptr;
    double* local_virial = nullptr;
    double* local_total_virial = nullptr;
    double* local_spin_transfer = nullptr;
    if (use_lammps_scratch) {
      local_force =
        lammps_scratch->force_private + static_cast<std::size_t>(tid) * 3 * force_stride;
      local_grad_spin =
        lammps_scratch->mforce_private + static_cast<std::size_t>(tid) * 3 * force_stride;
      local_total_virial =
        lammps_scratch->total_virial_private +
        static_cast<std::size_t>(tid) * kLammpsTotalVirialStride;
      local_virial = lammps_scratch->virial_private
        ? lammps_scratch->virial_private + static_cast<std::size_t>(tid) * 9 * force_stride
        : nullptr;
      local_spin_transfer = lammps_scratch->spin_transfer_private
        ? lammps_scratch->spin_transfer_private +
            static_cast<std::size_t>(tid) * 9 * force_stride
        : nullptr;
    } else {
      local_force = use_private_edges
        ? force_private.data() + static_cast<std::size_t>(tid) * 3 * N
        : force;
      local_grad_spin = use_private_edges
        ? grad_spin_private.data() + static_cast<std::size_t>(tid) * 3 * N
        : grad_spin.data();
      local_virial = use_private_edges
        ? virial_private.data() + static_cast<std::size_t>(tid) * 9
        : nullptr;
      local_spin_transfer = spin_transfer && use_private_edges
        ? spin_transfer_private.data() + static_cast<std::size_t>(tid) * 9 * N
        : spin_transfer;
    }
    auto add_local_grad_spin = [&](const int atom, const std::array<double, 3>& g) {
      for (int d = 0; d < 3; ++d) {
        if (use_lammps_scratch) {
          local_grad_spin[lammps_vector_index(atom, d, force_stride)] += g[d];
        } else {
          local_grad_spin[static_cast<std::size_t>(d) * force_stride + atom] += g[d];
        }
      }
    };
    std::array<double, MAX_NUM_N> grad_weight;
    grad_weight.fill(0.0);
    std::array<double, 3> grad_rhat = {0.0, 0.0, 0.0};
    std::array<double, 3> grad_si = {0.0, 0.0, 0.0};
    std::array<double, 3> grad_sj = {0.0, 0.0, 0.0};
    double grad_dot = 0.0;

    auto add_scalar = [&](const int offset, const double scalar) {
      double g = 0.0;
      for (int c = 0; c < C; ++c) {
        const double a = fp(edge.i, offset + c);
        grad_weight[c] += a * scalar;
        g += a * edge.weights[c];
      }
      return g;
    };

    int offset = 2;
    grad_dot += add_scalar(offset, edge.dot);
    offset += C;
    grad_dot += 2.0 * edge.dot * add_scalar(offset, edge.dot * edge.dot);
    offset += C;
    const double g_sj2 = add_scalar(offset, edge.sj2);
    for (int d = 0; d < 3; ++d) {
      grad_sj[d] += 2.0 * g_sj2 * sj[d];
    }
    offset += C;
    const double g_axis = add_scalar(offset, edge.bond_axis);
    for (int d = 0; d < 3; ++d) {
      grad_si[d] += g_axis * edge.ri_dot_sj * edge.rhat[d];
      grad_sj[d] += g_axis * edge.ri_dot_si * edge.rhat[d];
      grad_rhat[d] += g_axis * (edge.ri_dot_sj * si[d] + edge.ri_dot_si * sj[d]);
    }
    offset += C;

    auto add_angular_density_gradient = [&](
      const int width,
      const std::vector<double>& density,
      const double* value,
      const int q_offset,
      double* grad_value) {
      if (C == 4) {
        const double* self0 = block(density, edge.center, 0, width);
        const double* self1 = block(density, edge.center, 1, width);
        const double* self2 = block(density, edge.center, 2, width);
        const double* self3 = block(density, edge.center, 3, width);
        const double alpha0 = 2.0 * fp(edge.i, q_offset + 0);
        const double alpha1 = 2.0 * fp(edge.i, q_offset + 1);
        const double alpha2 = 2.0 * fp(edge.i, q_offset + 2);
        const double alpha3 = 2.0 * fp(edge.i, q_offset + 3);
        const double w0 = edge.weights[0];
        const double w1 = edge.weights[1];
        const double w2 = edge.weights[2];
        const double w3 = edge.weights[3];
        double gw0 = grad_weight[0];
        double gw1 = grad_weight[1];
        double gw2 = grad_weight[2];
        double gw3 = grad_weight[3];
#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp simd reduction(+:gw0, gw1, gw2, gw3)
#endif
        for (int k = 0; k < width; ++k) {
          const double v = value[k];
          const double gd0 = alpha0 * self0[k];
          const double gd1 = alpha1 * self1[k];
          const double gd2 = alpha2 * self2[k];
          const double gd3 = alpha3 * self3[k];
          gw0 += gd0 * v;
          gw1 += gd1 * v;
          gw2 += gd2 * v;
          gw3 += gd3 * v;
          grad_value[k] = gd0 * w0 + gd1 * w1 + gd2 * w2 + gd3 * w3;
        }
        grad_weight[0] = gw0;
        grad_weight[1] = gw1;
        grad_weight[2] = gw2;
        grad_weight[3] = gw3;
        return;
      }
      for (int c = 0; c < C; ++c) {
        const double* self = block(density, edge.center, c, width);
        const double alpha2 = 2.0 * fp(edge.i, q_offset + c);
        const double w = edge.weights[c];
        double grad_weight_c = grad_weight[c];
        for (int k = 0; k < width; ++k) {
          const double gd = alpha2 * self[k];
          grad_weight_c += gd * value[k];
          grad_value[k] += gd * w;
        }
        grad_weight[c] = grad_weight_c;
      }
    };

    auto apply_angular = [&](const int ell, const double* ylm, const int ylm_width, const double* ge) {
      double grad_ylm[9] = {0.0};
      for (int m = 0; m < ylm_width; ++m) {
        const int base = m * 3;
        const double y = ylm[m];
        const double g0 = ge[base + 0];
        const double g1 = ge[base + 1];
        const double g2 = ge[base + 2];
        grad_sj[0] += g0 * y;
        grad_sj[1] += g1 * y;
        grad_sj[2] += g2 * y;
        grad_ylm[m] += g0 * sj[0] + g1 * sj[1] + g2 * sj[2];
      }
      add_real_spherical_harmonics_gradient(edge.rhat, ell, grad_ylm, grad_rhat);
    };

    for (int c = 0; c < C; ++c) {
      const double* b = block(rho0_pull, edge.center, c, 3);
      const double* b_dot = block(rho0_dot_pull, edge.center, c, 3);
      const double u0 = b[0] + edge.dot * b_dot[0];
      const double u1 = b[1] + edge.dot * b_dot[1];
      const double u2 = b[2] + edge.dot * b_dot[2];
      const double w = edge.weights[c];
      grad_weight[c] += sj[0] * u0 + sj[1] * u1 + sj[2] * u2;
      grad_sj[0] += w * u0;
      grad_sj[1] += w * u1;
      grad_sj[2] += w * u2;
      grad_dot += w * (sj[0] * b_dot[0] + sj[1] * b_dot[1] + sj[2] * b_dot[2]);
    }
    offset += C;

    for (int c = 0; c < C; ++c) {
      const double* K = block(geom_pull, edge.center, c, 9);
      const double kr0 = K[0] * edge.rhat[0] + K[1] * edge.rhat[1] + K[2] * edge.rhat[2];
      const double kr1 = K[3] * edge.rhat[0] + K[4] * edge.rhat[1] + K[5] * edge.rhat[2];
      const double kr2 = K[6] * edge.rhat[0] + K[7] * edge.rhat[1] + K[8] * edge.rhat[2];
      double u0 = 0.0;
      double u1 = 0.0;
      double u2 = 0.0;
      double mt0 = 0.0;
      double mt1 = 0.0;
      double mt2 = 0.0;
      double dot_v2 = 0.0;
      if (l_max >= 1) {
        const double* M = block(l1_pull, edge.center, c, 9);
        const double* Md = block(l1_dot_pull, edge.center, c, 9);
        const double v10 = M[0] * sj[0] + M[1] * sj[1] + M[2] * sj[2];
        const double v11 = M[3] * sj[0] + M[4] * sj[1] + M[5] * sj[2];
        const double v12 = M[6] * sj[0] + M[7] * sj[1] + M[8] * sj[2];
        const double v20 = Md[0] * sj[0] + Md[1] * sj[1] + Md[2] * sj[2];
        const double v21 = Md[3] * sj[0] + Md[4] * sj[1] + Md[5] * sj[2];
        const double v22 = Md[6] * sj[0] + Md[7] * sj[1] + Md[8] * sj[2];
        u0 = v10 + edge.dot * v20;
        u1 = v11 + edge.dot * v21;
        u2 = v12 + edge.dot * v22;
        mt0 = M[0] * edge.rhat[0] + M[3] * edge.rhat[1] + M[6] * edge.rhat[2] +
              edge.dot * (Md[0] * edge.rhat[0] + Md[3] * edge.rhat[1] + Md[6] * edge.rhat[2]);
        mt1 = M[1] * edge.rhat[0] + M[4] * edge.rhat[1] + M[7] * edge.rhat[2] +
              edge.dot * (Md[1] * edge.rhat[0] + Md[4] * edge.rhat[1] + Md[7] * edge.rhat[2]);
        mt2 = M[2] * edge.rhat[0] + M[5] * edge.rhat[1] + M[8] * edge.rhat[2] +
              edge.dot * (Md[2] * edge.rhat[0] + Md[5] * edge.rhat[1] + Md[8] * edge.rhat[2]);
        dot_v2 = edge.rhat[0] * v20 + edge.rhat[1] * v21 + edge.rhat[2] * v22;
      }
      const double w = edge.weights[c];
      grad_weight[c] += edge.rhat[0] * (u0 + kr0) + edge.rhat[1] * (u1 + kr1) +
                        edge.rhat[2] * (u2 + kr2);
      grad_rhat[0] += w * (u0 + 2.0 * kr0);
      grad_rhat[1] += w * (u1 + 2.0 * kr1);
      grad_rhat[2] += w * (u2 + 2.0 * kr2);
      grad_sj[0] += w * mt0;
      grad_sj[1] += w * mt1;
      grad_sj[2] += w * mt2;
      grad_dot += w * dot_v2;
    }
    if (l_max >= 1) {
      offset += 3 * C;
    }
    for (int ell = 2; ell <= l_max; ++ell) {
      const std::vector<double>& angular = ell == 2 ? cache.angular2 : ell == 3 ? cache.angular3 : cache.angular4;
      double ylm[9];
      const int ylm_width = real_spherical_harmonics_spin(edge.rhat, ell, ylm);
      double value[27];
      int width = 0;
      for (int m = 0; m < ylm_width; ++m) {
        const double y = ylm[m];
        value[width++] = y * sj[0];
        value[width++] = y * sj[1];
        value[width++] = y * sj[2];
      }
      double ge[27];
      if (C != 4) {
        std::fill(ge, ge + width, 0.0);
      }
      add_angular_density_gradient(width, angular, value, offset, ge);
      apply_angular(ell, ylm, ylm_width, ge);
      offset += C;
    }

    offset += C;
    offset += C;
    if (l_max >= 1) {
      offset += C;
    }

    for (int d = 0; d < 3; ++d) {
      grad_si[d] += grad_dot * sj[d];
      grad_sj[d] += grad_dot * si[d];
    }

    double grad_dist = 0.0;
    for (int c = 0; c < C; ++c) {
      grad_dist += grad_weight[c] * edge.weight_derivatives[c];
    }

    std::array<double, 3> grad_rij;
    double dot_r = 0.0;
    for (int d = 0; d < 3; ++d) {
      dot_r += grad_rhat[d] * edge.rhat[d];
    }
    for (int d = 0; d < 3; ++d) {
      grad_rij[d] = grad_dist * edge.rhat[d] + (grad_rhat[d] - dot_r * edge.rhat[d]) / edge.dist;
      if (use_lammps_scratch) {
        local_force[lammps_vector_index(edge.i, d, force_stride)] += grad_rij[d];
        local_force[lammps_vector_index(edge.j, d, force_stride)] -= grad_rij[d];
      } else {
        local_force[static_cast<std::size_t>(d) * force_stride + edge.i] += grad_rij[d];
        local_force[static_cast<std::size_t>(d) * force_stride + edge.j] -= grad_rij[d];
      }
    }
    if (use_lammps_scratch) {
      if (local_virial) {
        add_lammps_spin_virial(
          edge.rhat, edge.dist, grad_rij, local_total_virial, local_virial,
          force_stride, edge.j);
      } else {
        add_lammps_spin_total_virial(
          edge.rhat, edge.dist, grad_rij, local_total_virial);
      }
    } else {
      for (int a = 0; a < 3; ++a) {
        const double rij_a = edge.rhat[a] * edge.dist;
        for (int b = 0; b < 3; ++b) {
          if (use_private_edges) {
            local_virial[a * 3 + b] -= rij_a * grad_rij[b];
          } else {
            const int owner = lammps_neighbor_virial_ownership ? edge.j : 0;
            virial[static_cast<std::size_t>(a * 3 + b) * N + owner] -=
              rij_a * grad_rij[b];
          }
        }
      }
    }
    add_local_grad_spin(edge.i, grad_si);
    add_local_grad_spin(edge.j, grad_sj);
    if (use_lammps_scratch) {
      add_spin_transfer_row_major9(
        edge.rhat, edge.dist, grad_sj, local_spin_transfer, force_stride, edge.j);
    } else {
      add_spin_transfer_soa9(
        N, edge.j, edge.rhat, edge.dist, grad_sj, local_spin_transfer);
    }
  }

  if (use_private_edges && !use_lammps_scratch) {
    for (int tid = 0; tid < num_threads; ++tid) {
      const double* local_force = force_private.data() + static_cast<std::size_t>(tid) * 3 * N;
      const double* local_grad_spin =
        grad_spin_private.data() + static_cast<std::size_t>(tid) * 3 * N;
      const double* local_virial = virial_private.data() + static_cast<std::size_t>(tid) * 9;
      const double* local_spin_transfer = spin_transfer
        ? spin_transfer_private.data() + static_cast<std::size_t>(tid) * 9 * N
        : nullptr;
      for (int atom = 0; atom < N; ++atom) {
        for (int d = 0; d < 3; ++d) {
          const std::size_t idx = static_cast<std::size_t>(d) * N + atom;
          force[idx] += local_force[idx];
          grad_spin[idx] += local_grad_spin[idx];
        }
      }
      for (int d = 0; d < 9; ++d) {
        virial[static_cast<std::size_t>(d) * N] += local_virial[d];
        if (spin_transfer) {
          for (int atom = 0; atom < N; ++atom) {
            spin_transfer[static_cast<std::size_t>(d) * N + atom] +=
              local_spin_transfer[static_cast<std::size_t>(d) * N + atom];
          }
        }
      }
    }
  }

  if (phase) {
    phase->gradient_nonchiral += nep_phase_elapsed(phase_mark);
    phase_mark = NepPhaseClock::now();
  }
  add_spin_chiral_gradient(
    paramb, annmb, N, spins, cache, Fp, grad_spin, force, virial, spin_transfer,
    lammps_neighbor_virial_ownership,
    use_lammps_scratch ? lammps_scratch : nullptr,
    scratch);
  if (phase) {
    phase->gradient_chiral += nep_phase_elapsed(phase_mark);
  }

  if (use_lammps_scratch) {
    return;
  }

  if (lammps_output) {
    for (int atom = 0; atom < N; ++atom) {
      for (int d = 0; d < 3; ++d) {
        lammps_force[atom][d] += lammps_force_work[static_cast<std::size_t>(d) * N + atom];
        lammps_mforce[atom][d] -= grad_spin[static_cast<std::size_t>(d) * N + atom];
      }
    }
    auto raw = [&](const int comp, const int atom) {
      return lammps_virial_work[static_cast<std::size_t>(comp) * N + atom];
    };
    for (int atom = 0; atom < N; ++atom) {
      lammps_total_virial[0] += raw(0, atom);
      lammps_total_virial[1] += raw(4, atom);
      lammps_total_virial[2] += raw(8, atom);
      lammps_total_virial[3] += 0.5 * (raw(1, atom) + raw(3, atom));
      lammps_total_virial[4] += 0.5 * (raw(2, atom) + raw(6, atom));
      lammps_total_virial[5] += 0.5 * (raw(5, atom) + raw(7, atom));
      if (lammps_virial) {
        lammps_virial[atom][0] += raw(0, atom);
        lammps_virial[atom][1] += raw(4, atom);
        lammps_virial[atom][2] += raw(8, atom);
        lammps_virial[atom][3] += raw(1, atom);
        lammps_virial[atom][4] += raw(2, atom);
        lammps_virial[atom][5] += raw(5, atom);
        lammps_virial[atom][6] += raw(3, atom);
        lammps_virial[atom][7] += raw(6, atom);
        lammps_virial[atom][8] += raw(7, atom);
      }
      if (lammps_spin_transfer) {
        for (int d = 0; d < 9; ++d) {
          lammps_spin_transfer[atom][d] +=
            lammps_spin_transfer_work[static_cast<std::size_t>(d) * N + atom];
        }
      }
    }
    return;
  }

  for (int atom = 0; atom < N; ++atom) {
    for (int d = 0; d < 3; ++d) {
      mforce[static_cast<std::size_t>(d) * N + atom] -= grad_spin[static_cast<std::size_t>(d) * N + atom];
    }
  }
}

void find_spin_descriptor_for_lammps(
  const NEP::ParaMB& paramb,
  const NEP::ANN& annmb,
  const int atom_capacity,
  const int inum,
  const int* ilist,
  const int* NN,
  int** NL,
  const int* spin_types,
  double** pos,
  const double* spins_soa,
  double* descriptor_aos,
  SpinCache& cache,
  SpinPhaseBreakdown* phase,
  const LammpsRadialEdgeCacheView* radial_cache)
{
  fill_spin_descriptor(
    paramb, annmb, atom_capacity, NN, nullptr, spin_types, nullptr, nullptr,
    nullptr, spins_soa, nullptr, &cache, phase, inum, ilist, NL, pos,
    descriptor_aos, radial_cache);
}

void apply_spin_ann_for_lammps(
  const NEP::ParaMB& paramb,
  const NEP::ANN& annmb,
  const int inum,
  const int* ilist,
  const int* spin_types,
  const std::vector<double>& spin_baseline,
  double* descriptor_aos,
  double* Fp,
  double& total_potential,
  double* potential,
  const bool use_parallel_atoms,
  const int num_threads)
{
  double reduced_potential = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) reduction(+:reduced_potential) if (use_parallel_atoms)
#endif
  for (int ii = 0; ii < inum; ++ii) {
    const int atom = ilist[ii];
    double F = 0.0;
    double Fp_local[MAX_DIM] = {0.0};
    double latent[MAX_NEURON] = {0.0};
    const int mapped_type = spin_types[atom];
    double* q = descriptor_aos + static_cast<std::size_t>(ii) * annmb.dim;
    apply_ann_one_layer(
      annmb.dim, annmb.num_neurons1, annmb.w0[mapped_type], annmb.b0[mapped_type],
      annmb.w1[mapped_type], annmb.b1, q, F, Fp_local, latent, false, nullptr);
    const double energy = F + spin_baseline[static_cast<std::size_t>(mapped_type)];
    reduced_potential += energy;
    if (potential) {
      potential[atom] += energy;
    }
    for (int d = 0; d < annmb.dim; ++d) {
      Fp[static_cast<std::size_t>(atom) * annmb.dim + d] =
        Fp_local[d] * paramb.q_scaler[d];
    }
  }
  total_potential += reduced_potential;
}

void find_spin_force_for_lammps(
  const NEP::ParaMB& paramb,
  const NEP::ANN& annmb,
  const int atom_capacity,
  const int inum,
  const int* ilist,
  const int* spin_types,
  const double* spins_soa,
  const SpinCache& cache,
  const double* Fp,
  double** force,
  double** mforce,
  double total_virial[6],
  double** virial,
  double** spin_transfer,
  LammpsThreadLocalScratchView* lammps_scratch,
  NEP::SpinGradientScratch* scratch,
  SpinPhaseBreakdown* phase)
{
  add_spin_gradient(
    paramb, annmb, atom_capacity, spin_types, spins_soa, cache, Fp,
    nullptr, nullptr, nullptr, nullptr, phase, force, mforce, total_virial, virial,
    spin_transfer, lammps_scratch, scratch, inum, ilist);
}

bool compute_spin_lammps_fused_center(
  const NEP::ParaMB& paramb,
  const NEP::ANN& annmb,
  const int atom_capacity,
  const int inum,
  const int* ilist,
  const int* NN,
  const int* spin_types,
  const double* spins_aos3,
  const std::vector<double>& spin_baseline,
  const LammpsRadialEdgeCacheView* radial_cache,
  double* descriptor_aos,
  double* Fp,
  double& total_potential,
  double* potential,
  LammpsThreadLocalScratchView* lammps_scratch,
  SpinPhaseBreakdown* phase)
{
  if (!lammps_radial_edge_cache_active(radial_cache) ||
      radial_cache->num_centers != inum ||
      paramb.spin_cutoff_radial > paramb.rc_radial_max + 1.0e-12 ||
      !lammps_spin_scratch_active(lammps_scratch) ||
      paramb.spin_compress > MAX_SPIN_COMPRESS) {
    return false;
  }

  const int C = paramb.spin_compress;
  const int B = paramb.spin_basis_size + 1;
  const int l_max = paramb.spin_l_max;
  const int chiC = std::min(2, C);
  const int offset0 = paramb.struct_dim;
  const double spin_rcinv = 1.0 / paramb.spin_cutoff_radial;

  auto mask_all_active = [](const std::vector<int>& mask) {
    return mask.empty() || std::all_of(mask.begin(), mask.end(), [](const int value) {
      return value != 0;
    });
  };
  const bool spin_dof_all_active = mask_all_active(paramb.spin_dof_type_active);
  const bool spin_env_all_active = mask_all_active(paramb.spin_env_type_active);
  auto type_active = [](const std::vector<int>& mask, const bool all_active, const int t) {
    return all_active || mask[static_cast<std::size_t>(t)] != 0;
  };
  auto spin = [&](const int atom, const int d) {
    return spins_aos3[static_cast<std::size_t>(atom) * 3 + d];
  };
  auto block = [&](std::vector<double>& v, const int c, const int width) {
    return v.data() + static_cast<std::size_t>(c) * width;
  };
  auto const_block = [&](const std::vector<double>& v, const int c, const int width) {
    return v.data() + static_cast<std::size_t>(c) * width;
  };
  auto dot_density = [](const double* a, const double* b, const int width) {
    double value = 0.0;
    for (int k = 0; k < width; ++k) {
      value += a[k] * b[k];
    }
    return value;
  };

#if defined(_OPENMP)
  const int num_threads = lammps_scratch->num_threads;
  const bool use_parallel = num_threads > 1 && inum > 256;
#else
  const int num_threads = 1;
  const bool use_parallel = false;
#endif

  double reduced_potential = 0.0;
#if defined(_OPENMP)
#pragma omp parallel num_threads(num_threads) if (use_parallel) reduction(+:reduced_potential)
#endif
  {
#if defined(_OPENMP)
    const int tid = use_parallel ? omp_get_thread_num() : 0;
#else
    const int tid = 0;
#endif
    SpinPhaseBreakdown thread_phase;
    auto thread_phase_mark = NepPhaseClock::now();
    auto add_thread_phase = [&](double SpinPhaseBreakdown::*slot) {
      if (phase) {
        thread_phase.*slot += nep_phase_elapsed(thread_phase_mark);
      }
    };
    SpinCache cache;
    resize_spin_center_cache(cache, C, chiC, paramb.spin_chiral);
    std::vector<double> chiral_grad_weight;
    std::vector<double> chiral_grad_rhat;
    LammpsThreadLocalScratchView local_scratch = *lammps_scratch;
    local_scratch.num_threads = lammps_scratch->num_threads;
    local_scratch.force_private =
      lammps_scratch->force_private + static_cast<std::size_t>(tid) * 3 * lammps_scratch->force_rows;
    local_scratch.total_virial_private =
      lammps_scratch->total_virial_private +
      static_cast<std::size_t>(tid) * kLammpsTotalVirialStride;
    local_scratch.virial_private = lammps_scratch->virial_private
      ? lammps_scratch->virial_private +
          static_cast<std::size_t>(tid) * 9 * lammps_scratch->force_rows
      : nullptr;
    local_scratch.mforce_private =
      lammps_scratch->mforce_private + static_cast<std::size_t>(tid) * 3 * lammps_scratch->force_rows;
    local_scratch.spin_transfer_private = lammps_scratch->spin_transfer_private
      ? lammps_scratch->spin_transfer_private +
          static_cast<std::size_t>(tid) * 9 * lammps_scratch->force_rows
      : nullptr;
    add_thread_phase(&SpinPhaseBreakdown::setup);

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
    for (int ii = 0; ii < inum; ++ii) {
      if (phase) {
        thread_phase_mark = NepPhaseClock::now();
      }
      const int atom = ilist[ii];
      double* q_full = descriptor_aos + static_cast<std::size_t>(ii) * annmb.dim;
      double q_spin[MAX_DIM] = {0.0};
      const bool dof =
        type_active(paramb.spin_dof_type_active, spin_dof_all_active, spin_types[atom]);
      const double sx = spin(atom, 0);
      const double sy = spin(atom, 1);
      const double sz = spin(atom, 2);
      const std::array<double, 3> si = {sx, sy, sz};
      const double s2 = sx * sx + sy * sy + sz * sz;
      q_spin[0] = dof ? s2 : 0.0;
      q_spin[1] = dof ? s2 * s2 : 0.0;

      clear_spin_center_cache(cache, paramb.spin_chiral, NN[atom]);
      add_thread_phase(&SpinPhaseBreakdown::setup);

      const int edge_offset = radial_cache->offsets[ii];
      for (int n = 0; n < NN[atom]; ++n) {
        const int j = radial_cache->neighbors[edge_offset + n];
        if (j < 0 || !type_active(paramb.spin_env_type_active, spin_env_all_active, spin_types[j])) {
          continue;
        }
        const double d = radial_cache->d12[edge_offset + n];
        if (d <= 1.0e-12 || d >= paramb.spin_cutoff_radial) {
          continue;
        }

        double fc = 0.0;
        double fcp = 0.0;
        double fn[MAX_NUM_N];
        double fnp[MAX_NUM_N];
        find_fc_and_fcp(paramb.spin_cutoff_radial, spin_rcinv, d, fc, fcp);
        if (paramb.spin_basis_size == 3) {
          find_spin_basis3_and_derivatives(spin_rcinv, d, fc, fcp, fn, fnp);
        } else {
          find_fn_and_fnp(paramb.spin_basis_size, spin_rcinv, d, fc, fcp, fn, fnp);
        }

        SpinEdge edge;
        edge.i = atom;
        edge.j = j;
        edge.center = 0;
        edge.t12 = spin_types[atom] * paramb.num_types + spin_types[j];
        edge.dist = d;
        edge.rhat = {
          radial_cache->x12[edge_offset + n] / d,
          radial_cache->y12[edge_offset + n] / d,
          radial_cache->z12[edge_offset + n] / d};
        const std::array<double, 3> sj = {spin(j, 0), spin(j, 1), spin(j, 2)};
        edge.weights.fill(0.0);
        edge.weight_derivatives.fill(0.0);
        for (int c = 0; c < C; ++c) {
          double w = 0.0;
          double dw = 0.0;
          const double* coeff =
            annmb.c_spin + (static_cast<std::size_t>(c) * B * paramb.num_types_sq + edge.t12);
          for (int k = 0; k < B; ++k) {
            const double ck = coeff[static_cast<std::size_t>(k) * paramb.num_types_sq];
            w += fn[k] * ck;
            dw += fnp[k] * ck;
          }
          edge.weights[c] = w;
          edge.weight_derivatives[c] = dw;
        }
        edge.dot = dot3(si, sj);
        edge.sj2 = dot3(sj, sj);
        edge.ri_dot_si = dot3(edge.rhat, si);
        edge.ri_dot_sj = dot3(edge.rhat, sj);
        edge.bond_axis = edge.ri_dot_si * edge.ri_dot_sj;

        int scalar_offset = 2;
        const double scalars[4] = {
          edge.dot, edge.dot * edge.dot, edge.sj2, edge.bond_axis};
        for (int term = 0; term < 4; ++term) {
          for (int c = 0; c < C; ++c) {
            q_spin[scalar_offset + c] += edge.weights[c] * scalars[term];
          }
          scalar_offset += C;
        }

        const double sj_value[3] = {sj[0], sj[1], sj[2]};
        add_density_fixed<3>(cache.rho0, C, 0, sj_value, edge.weights.data());
        double raw1_value[9];
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            raw1_value[3 * a + b] = edge.rhat[a] * sj[b];
          }
        }
        add_density_fixed<9>(cache.raw1, C, 0, raw1_value, edge.weights.data());
        if (l_max >= 1) {
          const double rdot = dot3(edge.rhat, sj);
          add_density_fixed<1>(cache.l1_rdot, C, 0, &rdot, edge.weights.data());
          const double cross_value[3] = {
            edge.rhat[1] * sj[2] - edge.rhat[2] * sj[1],
            edge.rhat[2] * sj[0] - edge.rhat[0] * sj[2],
            edge.rhat[0] * sj[1] - edge.rhat[1] * sj[0]};
          add_density_fixed<3>(cache.l1_cross, C, 0, cross_value, edge.weights.data());
          const auto stf = stf_outer3(edge.rhat, sj);
          add_density_fixed<9>(cache.l1_stf, C, 0, stf.data(), edge.weights.data());
        }
        for (int ell = 2; ell <= l_max; ++ell) {
          double ylm[9];
          const int ylm_width = real_spherical_harmonics_spin(edge.rhat, ell, ylm);
          double value[27];
          int width = 0;
          for (int m = 0; m < ylm_width; ++m) {
            const double y = ylm[m];
            value[width++] = y * sj[0];
            value[width++] = y * sj[1];
            value[width++] = y * sj[2];
          }
          if (ell == 2) {
            add_density_fixed<15>(cache.angular2, C, 0, value, edge.weights.data());
          } else if (ell == 3) {
            add_density_fixed<21>(cache.angular3, C, 0, value, edge.weights.data());
          } else {
            add_density_fixed<27>(cache.angular4, C, 0, value, edge.weights.data());
          }
        }
        const auto rr = stf_outer3(edge.rhat, edge.rhat);
        add_density_fixed<9>(cache.geom, C, 0, rr.data(), edge.weights.data());
        if (paramb.spin_chiral) {
          add_density_fixed<3>(cache.polars, C, 0, edge.rhat.data(), edge.weights.data());
          double o_reduced[kSpinChiralOReducedCount];
          double h_reduced[kSpinChiralHReducedCount];
          fill_spin_chiral_reduced_moments(edge.rhat, o_reduced, h_reduced);
          add_density_fixed<kSpinChiralOReducedCount>(
            cache.octupoles, chiC, 0, o_reduced, edge.weights.data());
          add_density_fixed<kSpinChiralHReducedCount>(
            cache.hexadecapoles, chiC, 0, h_reduced, edge.weights.data());
        }
        add_density_fixed<3>(cache.rho0_dot, C, 0, sj_value, edge.weights.data(), edge.dot);
        add_density_fixed<9>(cache.raw1_dot, C, 0, raw1_value, edge.weights.data(), edge.dot);
        cache.edges.push_back(std::move(edge));
      }
      cache.edge_offsets[1] = static_cast<int>(cache.edges.size());
      add_thread_phase(&SpinPhaseBreakdown::edges);

      add_thread_phase(&SpinPhaseBreakdown::unpack);

      int offset = 2 + 4 * C;
      auto contract = [&](const std::vector<double>& a, const std::vector<double>& b, const int width) {
        for (int c = 0; c < C; ++c) {
          q_spin[offset + c] = dot_density(const_block(a, c, width), const_block(b, c, width), width);
        }
        offset += C;
      };
      contract(cache.rho0, cache.rho0, 3);
      if (l_max >= 1) {
        contract(cache.l1_rdot, cache.l1_rdot, 1);
        contract(cache.l1_cross, cache.l1_cross, 3);
        contract(cache.l1_stf, cache.l1_stf, 9);
      }
      for (int ell = 2; ell <= l_max; ++ell) {
        const int width = (2 * ell + 1) * 3;
        const std::vector<double>& angular =
          ell == 2 ? cache.angular2 : ell == 3 ? cache.angular3 : cache.angular4;
        contract(angular, angular, width);
      }
      const int geom_q_offset = offset;
      for (int c = 0; c < C; ++c) {
        const double* g = const_block(cache.geom, c, 9);
        double value = 0.0;
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            value += si[a] * g[3 * a + b] * si[b];
          }
        }
        q_spin[geom_q_offset + c] = value;
      }
      offset += C;
      contract(cache.rho0, cache.rho0_dot, 3);
      if (l_max >= 1) {
        contract(cache.raw1, cache.raw1_dot, 9);
      }
      add_thread_phase(&SpinPhaseBreakdown::contract);

      if (paramb.spin_chiral) {
        const int chiral_offset0 = offset;
        for (int c = 0; c < chiC; ++c) {
          const double* Q = const_block(cache.geom, c, 9);
          const double* O = cache.octupoles.data() +
            static_cast<std::size_t>(c) * kSpinChiralOReducedCount;
          const double* H = cache.hexadecapoles.data() +
            static_cast<std::size_t>(c) * kSpinChiralHReducedCount;
          double Q_reduced[5];
          fill_spin_chiral_q_reduced(Q, Q_reduced);
          cache.chirals[static_cast<std::size_t>(c)] =
            contract_spin_chiral_qoh_reduced(Q_reduced, O, H);
        }
        for (const SpinEdge& edge : cache.edges) {
          for (int c = 0; c < C; ++c) {
            const double* Q = const_block(cache.geom, c, 9);
            std::array<double, 3> Qu = {0.0, 0.0, 0.0};
            for (int a = 0; a < 3; ++a) {
              for (int b = 0; b < 3; ++b) {
                Qu[a] += Q[3 * a + b] * edge.rhat[b];
              }
            }
            const std::array<double, 3> axis = cross3(edge.rhat, Qu);
            double* out = block(cache.pseudodevs, c, 9);
            const double w = edge.weights[c];
            const double wh = 0.5 * w;
            const double pseudo01 = wh * (axis[0] * edge.rhat[1] + axis[1] * edge.rhat[0]);
            const double pseudo02 = wh * (axis[0] * edge.rhat[2] + axis[2] * edge.rhat[0]);
            const double pseudo12 = wh * (axis[1] * edge.rhat[2] + axis[2] * edge.rhat[1]);
            out[0] += w * axis[0] * edge.rhat[0];
            out[1] += pseudo01;
            out[2] += pseudo02;
            out[3] += pseudo01;
            out[4] += w * axis[1] * edge.rhat[1];
            out[5] += pseudo12;
            out[6] += pseudo02;
            out[7] += pseudo12;
            out[8] += w * axis[2] * edge.rhat[2];
          }
        }
        for (const SpinEdge& edge : cache.edges) {
          const std::array<double, 3> sj = {spin(edge.j, 0), spin(edge.j, 1), spin(edge.j, 2)};
          const std::array<double, 3> spin_cross = cross3(si, sj);
          for (int c = 0; c < chiC; ++c) {
            q_spin[chiral_offset0 + c] += edge.weights[c] * dot3(spin_cross, edge.rhat) *
                                           cache.chirals[static_cast<std::size_t>(c)];
          }
          int chiral_offset = chiral_offset0 + chiC;
          for (int c = 0; c < C; ++c) {
            const double* p = const_block(cache.polars, c, 3);
            const std::array<double, 3> polar = {p[0], p[1], p[2]};
            const std::array<double, 3> axis = cross3(polar, edge.rhat);
            q_spin[chiral_offset + c] += edge.weights[c] * dot3(spin_cross, axis);
          }
          chiral_offset += C;
          for (int c = 0; c < C; ++c) {
            const double* P = const_block(cache.pseudodevs, c, 9);
            std::array<double, 3> axis = {0.0, 0.0, 0.0};
            for (int a = 0; a < 3; ++a) {
              for (int b = 0; b < 3; ++b) {
                axis[a] += P[3 * a + b] * edge.rhat[b];
              }
            }
            q_spin[chiral_offset + c] += edge.weights[c] * dot3(spin_cross, axis);
          }
        }
      }
      add_thread_phase(&SpinPhaseBreakdown::chiral);

      for (int d = 0; d < paramb.spin_dim; ++d) {
        q_full[offset0 + d] = q_spin[d] * paramb.q_scaler[offset0 + d];
      }
      add_thread_phase(&SpinPhaseBreakdown::copy);

      double F = 0.0;
      double Fp_local[MAX_DIM] = {0.0};
      double latent[MAX_NEURON] = {0.0};
      const int mapped_type = spin_types[atom];
      apply_ann_one_layer(
        annmb.dim, annmb.num_neurons1, annmb.w0[mapped_type], annmb.b0[mapped_type],
        annmb.w1[mapped_type], annmb.b1, q_full, F, Fp_local, latent, false, nullptr);
      const double energy = F + spin_baseline[static_cast<std::size_t>(mapped_type)];
      reduced_potential += energy;
      if (potential) {
        potential[atom] += energy;
      }
      for (int d = 0; d < annmb.dim; ++d) {
        Fp[static_cast<std::size_t>(atom) * annmb.dim + d] =
          Fp_local[d] * paramb.q_scaler[d];
      }
      add_thread_phase(&SpinPhaseBreakdown::ann);

      add_spin_gradient_lammps_single_center_nonchiral<true>(
        paramb, annmb, atom_capacity, spin_types, spins_aos3, cache, Fp,
        &local_scratch, atom, &chiral_grad_weight, &chiral_grad_rhat);
      add_thread_phase(&SpinPhaseBreakdown::gradient_nonchiral);
    }
    if (phase) {
#if defined(_OPENMP)
#pragma omp critical(nep_spin_phase)
#endif
      add_spin_phase_breakdown(*phase, thread_phase);
    }
  }
  total_potential += reduced_potential;
  return true;
}

} // namespace

NEP::NEP() {}

NEP::NEP(const std::string& potential_filename) { init_from_file(potential_filename, true); }

void NEP::init_from_file(const std::string& potential_filename, const bool is_rank_0)
{
  std::ifstream input(potential_filename);
  if (!input.is_open()) {
    throw std::runtime_error("failed to open NEP model: " + potential_filename);
  }

  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() < 3) {
    throw std::invalid_argument("the first line of a NEP model must contain at least 3 fields");
  }
  if (tokens[0] == "nep3") {
    paramb.model_type = 0;
    paramb.version = 3;
    zbl.enabled = false;
  } else if (tokens[0] == "nep3_zbl") {
    paramb.model_type = 0;
    paramb.version = 3;
    zbl.enabled = true;
  } else if (tokens[0] == "nep3_dipole") {
    paramb.model_type = 1;
    paramb.version = 3;
    zbl.enabled = false;
  } else if (tokens[0] == "nep3_polarizability") {
    paramb.model_type = 2;
    paramb.version = 3;
    zbl.enabled = false;
  } else if (tokens[0] == "nep4") {
    paramb.model_type = 0;
    paramb.version = 4;
    zbl.enabled = false;
  } else if (tokens[0] == "nep4_spin" || tokens[0] == "nep4_spin1") {
    paramb.model_type = 0;
    paramb.version = 4;
    paramb.spin_mode = 1;
    zbl.enabled = false;
  } else if (tokens[0] == "nep4_zbl") {
    paramb.model_type = 0;
    paramb.version = 4;
    zbl.enabled = true;
  } else if (tokens[0] == "nep4_dipole") {
    paramb.model_type = 1;
    paramb.version = 4;
    zbl.enabled = false;
  } else if (tokens[0] == "nep4_polarizability") {
    paramb.model_type = 2;
    paramb.version = 4;
    zbl.enabled = false;
  } else if (tokens[0] == "nep5") {
    paramb.model_type = 0;
    paramb.version = 5;
    zbl.enabled = false;
  } else if (tokens[0] == "nep5_zbl") {
    paramb.model_type = 0;
    paramb.version = 5;
    zbl.enabled = true;
  } else if (tokens[0] == "nep4_charge1") {
    paramb.model_type = 0;
    paramb.version = 4;
    zbl.enabled = false;
    paramb.charge_mode = 1;
  } else if (tokens[0] == "nep4_zbl_charge1") {
    paramb.model_type = 0;
    paramb.version = 4;
    zbl.enabled = true;
    paramb.charge_mode = 1;
  } else if (tokens[0] == "nep4_charge2") {
    paramb.model_type = 0;
    paramb.version = 4;
    zbl.enabled = false;
    paramb.charge_mode = 2;
  } else if (tokens[0] == "nep4_zbl_charge2") {
    paramb.model_type = 0;
    paramb.version = 4;
    zbl.enabled = true;
    paramb.charge_mode = 2;
  } else if (tokens[0] == "nep4_charge3") {
    paramb.model_type = 0;
    paramb.version = 4;
    zbl.enabled = false;
    paramb.charge_mode = 3;
  } else if (tokens[0] == "nep4_zbl_charge3") {
    paramb.model_type = 0;
    paramb.version = 4;
    zbl.enabled = true;
    paramb.charge_mode = 3;
  } else {
    throw std::invalid_argument(tokens[0] + " is an unsupported NEP model");
  }

  paramb.num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (tokens.size() != 2 + paramb.num_types) {
    throw std::invalid_argument(
      "the first line of a NEP model has the wrong number of atom symbols");
  }

  element_list.resize(paramb.num_types);
  for (std::size_t n = 0; n < paramb.num_types; ++n) {
    int atomic_number = 0;
    element_list[n] = tokens[2 + n];
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (tokens[2 + n] == ELEMENTS[m]) {
        atomic_number = m;
        break;
      }
    }
    paramb.atomic_numbers[n] = atomic_number;
    dftd3.atomic_number[n] = atomic_number;
  }

  tokens = get_tokens(input);
  spin_baseline.clear();
  spin_baseline.assign(paramb.num_types, 0.0);
  auto parse_spin_line = [&](const std::vector<std::string>& spin_tokens) {
    if (spin_tokens.empty()) {
      return;
    }
    if (spin_tokens[0] == "spin_baseline") {
      if (spin_tokens.size() != 1 + paramb.num_types) {
        throw std::runtime_error("spin_baseline must have one value per type");
      }
      for (std::size_t t = 0; t < paramb.num_types; ++t) {
        spin_baseline[t] = get_double_from_token(spin_tokens[1 + t], __FILE__, __LINE__);
      }
    } else if (spin_tokens[0] == "spin_chiral") {
      paramb.spin_chiral = get_int_from_token(spin_tokens[1], __FILE__, __LINE__);
      if (paramb.spin_chiral != 0 && paramb.spin_chiral != 1) {
        throw std::runtime_error("spin_chiral must be 0 or 1");
      }
    } else if (spin_tokens[0] == "spin_compress") {
      paramb.spin_compress = get_int_from_token(spin_tokens[1], __FILE__, __LINE__);
    } else if (spin_tokens[0] == "spin_basis_size") {
      paramb.spin_basis_size = get_int_from_token(spin_tokens[1], __FILE__, __LINE__);
    } else if (spin_tokens[0] == "spin_l_max") {
      paramb.spin_l_max = get_int_from_token(spin_tokens[1], __FILE__, __LINE__);
    } else if (spin_tokens[0] == "spin_cutoff") {
      paramb.spin_cutoff_radial = get_double_from_token(spin_tokens[1], __FILE__, __LINE__);
    } else if (spin_tokens[0] == "spin_scaler") {
      const int spin_scaler = get_int_from_token(spin_tokens[1], __FILE__, __LINE__);
      if (spin_scaler != 1) {
        throw std::runtime_error("only spin_scaler 1 is supported by cpu");
      }
    } else if (spin_tokens[0] == "spin_dof_type" || spin_tokens[0] == "spin_type") {
      paramb.spin_dof_type_active.assign(paramb.num_types, 0);
      for (std::size_t i = 1; i < spin_tokens.size(); ++i) {
        auto found = std::find(element_list.begin(), element_list.end(), spin_tokens[i]);
        if (found == element_list.end()) {
          throw std::runtime_error("unknown spin_dof_type in nep.txt");
        }
        paramb.spin_dof_type_active[static_cast<std::size_t>(found - element_list.begin())] = 1;
      }
    } else if (spin_tokens[0] == "spin_env_type") {
      paramb.spin_env_type_active.assign(paramb.num_types, 0);
      for (std::size_t i = 1; i < spin_tokens.size(); ++i) {
        auto found = std::find(element_list.begin(), element_list.end(), spin_tokens[i]);
        if (found == element_list.end()) {
          throw std::runtime_error("unknown spin_env_type in nep.txt");
        }
        paramb.spin_env_type_active[static_cast<std::size_t>(found - element_list.begin())] = 1;
      }
    }
  };
  if (tokens[0] == "spin_mode") {
    paramb.spin_mode = get_int_from_token(tokens[1], __FILE__, __LINE__);
    if (tokens.size() >= 3) {
      const int spin_header_lines = get_int_from_token(tokens[2], __FILE__, __LINE__);
      for (int line = 0; line < spin_header_lines; ++line) {
        parse_spin_line(get_tokens(input));
      }
      tokens = get_tokens(input);
    } else {
      tokens = get_tokens(input);
      while (!tokens.empty() && tokens[0].rfind("spin_", 0) == 0) {
        parse_spin_line(tokens);
        tokens = get_tokens(input);
      }
    }
  } else {
    while (!tokens.empty() && tokens[0].rfind("spin_", 0) == 0) {
      paramb.spin_mode = 1;
      parse_spin_line(tokens);
      tokens = get_tokens(input);
    }
  }
  if (paramb.spin_mode) {
    if (paramb.spin_compress <= 0 || paramb.spin_l_max < 0 || paramb.spin_l_max > 4) {
      throw std::runtime_error("invalid spin settings");
    }
    if (paramb.spin_basis_size + 1 < paramb.spin_compress) {
      throw std::runtime_error("spin_basis_size must cover spin_compress");
    }
    if (paramb.spin_basis_size + 1 > MAX_NUM_N ||
        paramb.spin_compress > MAX_SPIN_COMPRESS) {
      throw std::runtime_error("spin basis is too large for cpu");
    }
    if (paramb.spin_dof_type_active.empty()) {
      paramb.spin_dof_type_active.assign(paramb.num_types, 1);
    }
    if (paramb.spin_env_type_active.empty()) {
      paramb.spin_env_type_active = paramb.spin_dof_type_active;
    }
  }

  // zbl
  if (zbl.enabled) {
    if (tokens.size() != 3 && tokens.size() != 4) {
      throw std::invalid_argument("expected: zbl rc_inner rc_outer [zbl_factor]");
    }
    zbl.rc_inner = get_double_from_token(tokens[1], __FILE__, __LINE__);
    zbl.rc_outer = get_double_from_token(tokens[2], __FILE__, __LINE__);
    if (zbl.rc_inner == 0 && zbl.rc_outer == 0) {
      zbl.flexibled = true;
    } else {
      if (tokens.size() == 4) {
        paramb.typewise_cutoff_zbl_factor = get_double_from_token(tokens[3], __FILE__, __LINE__);
        paramb.use_typewise_cutoff_zbl = true;
      }
    }
    tokens = get_tokens(input);
  }

  // cutoff
  if (tokens.size() != 5 && tokens.size() != paramb.num_types * 2 + 3) {
    throw std::invalid_argument("cutoff has the wrong number of parameters");
  }
  if (tokens.size() == 5) {
    paramb.rc_radial[0] = get_double_from_token(tokens[1], __FILE__, __LINE__);
    paramb.rc_angular[0] = get_double_from_token(tokens[2], __FILE__, __LINE__);
    for (std::size_t n = 0; n < paramb.num_types; ++n) {
      paramb.rc_radial[n] = paramb.rc_radial[0];
      paramb.rc_angular[n] = paramb.rc_angular[0];
    }
  } else {
    for (std::size_t n = 0; n < paramb.num_types; ++n) {
      paramb.rc_radial[n] = get_double_from_token(tokens[1 + n * 2], __FILE__, __LINE__);
      paramb.rc_angular[n] = get_double_from_token(tokens[2 + n * 2], __FILE__, __LINE__);
    }
  }
  for (std::size_t n = 0; n < paramb.num_types; ++n) {
    if (paramb.rc_radial[n] > paramb.rc_radial_max) {
      paramb.rc_radial_max = paramb.rc_radial[n];
    }
    if (paramb.rc_angular[n] > paramb.rc_angular_max) {
      paramb.rc_angular_max = paramb.rc_angular[n];
    }
  }
  if (paramb.spin_mode) {
    if (paramb.spin_cutoff_radial <= 0.0) {
      paramb.spin_cutoff_radial = paramb.rc_radial_max;
    }
    paramb.rc_radial_max = std::max(paramb.rc_radial_max, paramb.spin_cutoff_radial);
  }

  int MN_radial = get_int_from_token(tokens[tokens.size() - 2], __FILE__, __LINE__);
  int MN_angular = get_int_from_token(tokens[tokens.size() - 1], __FILE__, __LINE__);

  // n_max 10 8
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    throw std::invalid_argument("expected: n_max n_max_radial n_max_angular");
  }
  paramb.n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);

  // basis_size 10 8
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    throw std::invalid_argument(
      "expected: basis_size basis_size_radial basis_size_angular");
  }
  paramb.basis_size_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.basis_size_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);

  // l_max
  tokens = get_tokens(input);
  if (tokens.size() < 4) {
    throw std::invalid_argument("l_max line has too few parameters");
  }

  paramb.L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.num_L = paramb.L_max;

  paramb.has_q_222 =
      get_int_from_token(tokens[2], __FILE__, __LINE__) != 0;
  paramb.has_q_1111 =
      get_int_from_token(tokens[3], __FILE__, __LINE__) != 0;
  if (tokens.size() >= 5)
    paramb.has_q_112 =
        get_int_from_token(tokens[4], __FILE__, __LINE__) != 0;
  if (tokens.size() >= 6)
    paramb.has_q_123 =
        get_int_from_token(tokens[5], __FILE__, __LINE__) != 0;
  if (tokens.size() >= 7)
    paramb.has_q_233 =
        get_int_from_token(tokens[6], __FILE__, __LINE__) != 0;
  if (tokens.size() >= 8)
    paramb.has_q_134 =
        get_int_from_token(tokens[7], __FILE__, __LINE__) != 0;
  paramb.num_L += paramb.has_q_222 + paramb.has_q_1111 + paramb.has_q_112
                + paramb.has_q_123 + paramb.has_q_233 + paramb.has_q_134;

  paramb.dim_angular = (paramb.n_max_angular + 1) * paramb.num_L;

  // ANN
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    throw std::invalid_argument("expected: ANN num_neurons 0");
  }
  annmb.num_neurons1 = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.struct_dim = (paramb.n_max_radial + 1) + paramb.dim_angular;
  paramb.spin_dim =
    paramb.spin_mode ? spin_descriptor_dim(paramb.spin_compress, paramb.spin_l_max, paramb.spin_chiral != 0) : 0;
  annmb.dim = paramb.struct_dim + paramb.spin_dim;

  // calculated parameters:
  paramb.num_types_sq = paramb.num_types * paramb.num_types;
  if (paramb.version == 3) {
    annmb.num_para_ann = (annmb.dim + 2) * annmb.num_neurons1 + 1;
  } else if (paramb.version == 4) {
    annmb.num_para_ann = (annmb.dim + 2) * annmb.num_neurons1 * paramb.num_types + 1;
  } else {
    annmb.num_para_ann = ((annmb.dim + 2) * annmb.num_neurons1 + 1) * paramb.num_types + 1;
  }
  if (paramb.model_type == 2) {
    annmb.num_para_ann *= 2;
  }
  if (paramb.charge_mode > 0) {
    annmb.num_para_ann += annmb.num_neurons1 * paramb.num_types + 1;
  }
  int num_para_descriptor =
    paramb.num_types_sq * ((paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1) +
                           (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
  if (paramb.spin_mode) {
    num_para_descriptor += static_cast<int>(
      paramb.num_types_sq * paramb.spin_compress * (paramb.spin_basis_size + 1));
  }
  annmb.num_para = annmb.num_para_ann + num_para_descriptor;

  paramb.num_c_radial =
    paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);

  // NN and descriptor parameters
  parameters.resize(annmb.num_para);
  for (int n = 0; n < annmb.num_para; ++n) {
    tokens = get_tokens(input);
    parameters[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }
  update_potential(parameters.data(), annmb);
  cache_descriptor_coefficients(paramb, annmb);
  for (int d = 0; d < annmb.dim; ++d) {
    tokens = get_tokens(input);
    paramb.q_scaler[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }
  // flexible zbl potential parameters if (zbl.flexibled)
  if (zbl.flexibled) {
    int num_type_zbl = (paramb.num_types * (paramb.num_types + 1)) / 2;
    for (int d = 0; d < 10 * num_type_zbl; ++d) {
      tokens = get_tokens(input);
      zbl.para[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    zbl.num_types = paramb.num_types;
  }
  cache_type_pair_constants(paramb, zbl);
  input.close();


  // charge related parameters and data
  if (paramb.charge_mode > 0) {
    charge_para.alpha = PI / paramb.rc_radial_max; // a good value
    ewald.initialize(charge_para.alpha);
    charge_para.two_alpha_over_sqrt_pi = 2.0 * charge_para.alpha / sqrt(PI);
    charge_para.A = erfc(PI) / (paramb.rc_radial_max * paramb.rc_radial_max);
    charge_para.A += charge_para.two_alpha_over_sqrt_pi * exp(-PI * PI) / paramb.rc_radial_max;
    charge_para.B = - erfc(PI) / paramb.rc_radial_max - charge_para.A * paramb.rc_radial_max;
  }

  // only report for rank_0
  if (is_rank_0) {

    if (paramb.charge_mode > 0) {
      if (paramb.num_types == 1) {
        std::cout << "Use the NEP4-Charge" << paramb.charge_mode << " potential with " << paramb.num_types
                  << " atom type.\n";
      } else {
        std::cout << "Use the NEP4-Charge" << paramb.charge_mode << " potential with " << paramb.num_types
                  << " atom types.\n";
      }
    } else {
      if (paramb.num_types == 1) {
        std::cout << "Use the NEP" << paramb.version << " potential with " << paramb.num_types
                  << " atom type.\n";
      } else {
        std::cout << "Use the NEP" << paramb.version << " potential with " << paramb.num_types
                  << " atom types.\n";
      }
    }

    for (std::size_t n = 0; n < paramb.num_types; ++n) {
      std::cout << "    type " << n << " (" << element_list[n]
                << " with Z = " << paramb.atomic_numbers[n] + 1 << ")"
                << " has cutoffs " << "(" << paramb.rc_radial[n] << " A, "
                << paramb.rc_angular[n] << " A).\n";
    }

    if (zbl.enabled) {
      if (zbl.flexibled) {
        std::cout << "    has flexible ZBL.\n";
      } else {
        std::cout << "    has universal ZBL with inner cutoff " << zbl.rc_inner
                  << " A and outer cutoff " << zbl.rc_outer << " A.\n";
        if (paramb.use_typewise_cutoff_zbl) {
          std::cout << "    ZBL typewise cutoff is enabled with factor "
                    << paramb.typewise_cutoff_zbl_factor << ".\n";
        }
      }
    }

    std::cout << "    n_max_radial = " << paramb.n_max_radial << ".\n";
    std::cout << "    n_max_angular = " << paramb.n_max_angular << ".\n";
    std::cout << "    basis_size_radial = " << paramb.basis_size_radial << ".\n";
    std::cout << "    basis_size_angular = " << paramb.basis_size_angular << ".\n";
    std::cout << "    l_max_3body = " << paramb.L_max << ".\n";
    std::cout << "    has_q_222 = " << paramb.has_q_222 << ".\n";
    std::cout << "    has_q_1111 = " << paramb.has_q_1111 << ".\n";
    std::cout << "    has_q_112 = " << paramb.has_q_112 << ".\n";
    std::cout << "    has_q_123 = " << paramb.has_q_123 << ".\n";
    std::cout << "    has_q_233 = " << paramb.has_q_233 << ".\n";
    std::cout << "    ANN = " << annmb.dim << "-" << annmb.num_neurons1 << "-1.\n";
    std::cout << "    number of neural network parameters = " << annmb.num_para_ann << ".\n";
    std::cout << "    number of descriptor parameters = " << num_para_descriptor << ".\n";
    std::cout << "    total number of parameters = " << annmb.num_para << ".\n";
  }
}

void NEP::update_type_map(const int ntype, int* type_map, char** elements)
{
  std::size_t n = 0;
  for (int itype = 0; itype < ntype + 1; ++itype) {
    // check if set NULL in lammps input file
    if (type_map[itype] == -1) {
      continue;
    }

    // find the same element name in potential file
    std::string element_name = elements[type_map[itype]];
    for (n = 0; n < paramb.num_types; ++n) {
      if (element_name == element_list[n]) {
        type_map[itype] = n;
        break;
      }
    }

    // check if no corresponding element
    if (n == paramb.num_types) {
      std::cout << "There is no element " << element_name << " in the potential file." << std::endl;
      exit(1);
    }
  }
}

void NEP::update_potential(double* parameters, ANN& ann)
{
  double* pointer = parameters;
  for (std::size_t t = 0; t < paramb.num_types; ++t) {
    if (t > 0 && paramb.version == 3) { // Use the same set of NN parameters for NEP3
      pointer -= (ann.dim + 2) * ann.num_neurons1;
    }
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    if (paramb.charge_mode > 0) {
      pointer += ann.num_neurons1 * 2;
    } else {
      pointer += ann.num_neurons1;
    }
    
    if (paramb.version == 5) {
      pointer += 1; // one extra bias for NEP5 stored in ann.w1[t]
    }
  }

  if (paramb.charge_mode > 0) {
    ann.sqrt_epsilon_inf = pointer;
    pointer += 1;
  }

  ann.b1 = pointer;
  pointer += 1;

  if (paramb.model_type == 2) {
    for (std::size_t t = 0; t < paramb.num_types; ++t) {
      if (t > 0 && paramb.version == 3) { // Use the same set of NN parameters for NEP3
        pointer -= (ann.dim + 2) * ann.num_neurons1;
      }
      ann.w0_pol[t] = pointer;
      pointer += ann.num_neurons1 * ann.dim;
      ann.b0_pol[t] = pointer;
      pointer += ann.num_neurons1;
      ann.w1_pol[t] = pointer;
      pointer += ann.num_neurons1;
    }
    ann.b1_pol = pointer;
    pointer += 1;
  }

  ann.c = pointer;
  const std::size_t ordinary_descriptor_count =
    paramb.num_types_sq * ((paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1) +
                           (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
  pointer += ordinary_descriptor_count;
  ann.c_spin = paramb.spin_mode ? pointer : nullptr;
}

#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
void NEP::construct_table(const std::vector<char>& active_pairs)
{
  std::size_t active_count = 0;
  for (char active : active_pairs) {
    active_count += active ? 1 : 0;
  }

  bool unchanged =
    paramb.table_pair_to_slot.size() == paramb.num_types_sq &&
    paramb.table_pair_count == active_count;
  if (unchanged) {
    for (std::size_t t12 = 0; t12 < paramb.num_types_sq; ++t12) {
      if ((paramb.table_pair_to_slot[t12] >= 0) != (active_pairs[t12] != 0)) {
        unchanged = false;
        break;
      }
    }
  }
  if (unchanged) {
    return;
  }

  std::vector<int> slot_to_pair;
  slot_to_pair.reserve(active_count);
  paramb.table_pair_to_slot.assign(paramb.num_types_sq, -1);
  for (std::size_t t12 = 0; t12 < paramb.num_types_sq; ++t12) {
    if (!active_pairs[t12]) {
      continue;
    }
    paramb.table_pair_to_slot[t12] = static_cast<int>(slot_to_pair.size());
    slot_to_pair.push_back(static_cast<int>(t12));
  }

  paramb.table_pair_count = slot_to_pair.size();
  gn_radial.resize(table_length * paramb.table_pair_count * (paramb.n_max_radial + 1));
  gnp_radial.resize(table_length * paramb.table_pair_count * (paramb.n_max_radial + 1));
  gn_angular.resize(table_length * paramb.table_pair_count * (paramb.n_max_angular + 1));
  gnp_angular.resize(table_length * paramb.table_pair_count * (paramb.n_max_angular + 1));
  if (paramb.table_pair_count == 0) {
    return;
  }
  construct_table_radial_or_angular(
    slot_to_pair.data(), paramb.table_pair_count, paramb.n_max_radial, paramb.basis_size_radial,
    paramb.rc_radial_pair.data(), paramb.rcinv_radial_pair.data(), annmb.c_radial_pair.data(),
    gn_radial.data(), gnp_radial.data());
  construct_table_radial_or_angular(
    slot_to_pair.data(), paramb.table_pair_count, paramb.n_max_angular, paramb.basis_size_angular,
    paramb.rc_angular_pair.data(), paramb.rcinv_angular_pair.data(), annmb.c_angular_pair.data(),
    gn_angular.data(), gnp_angular.data());
}

void NEP::prepare_table_small_box(
  const int N,
  const int* g_NN_radial,
  const int* g_NL_radial,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type)
{
  std::vector<char> active_pairs(paramb.num_types_sq, 0);
  for (int n1 = 0; n1 < N; ++n1) {
    const int t1 = g_type[n1];
    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      const int n2 = g_NL_radial[i1 * N + n1];
      active_pairs[static_cast<std::size_t>(t1) * paramb.num_types + g_type[n2]] = 1;
    }
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      const int n2 = g_NL_angular[i1 * N + n1];
      active_pairs[static_cast<std::size_t>(t1) * paramb.num_types + g_type[n2]] = 1;
    }
  }
  construct_table(active_pairs);
}

void NEP::prepare_table_for_lammps(
  const int N,
  const int* g_ilist,
  const int* g_NN,
  int** g_NL,
  const int* g_type,
  const int* type_map)
{
  std::vector<char> active_pairs(paramb.num_types_sq, 0);
  for (int ii = 0; ii < N; ++ii) {
    const int n1 = g_ilist[ii];
    const int t1 = type_map[g_type[n1]];
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      const int n2 = g_NL[n1][i1];
      const int t2 = type_map[g_type[n2]];
      active_pairs[static_cast<std::size_t>(t1) * paramb.num_types + t2] = 1;
    }
  }
  construct_table(active_pairs);
}
#endif

void NEP::allocate_memory(const int N)
{
  if (num_atoms < N || NN_radial.size() < static_cast<std::size_t>(N) ||
      NL_radial.size() < static_cast<std::size_t>(N) * MN ||
      NN_angular.size() < static_cast<std::size_t>(N) ||
      NL_angular.size() < static_cast<std::size_t>(N) * MN ||
      r12.size() < static_cast<std::size_t>(N) * MN * 6) {
    NN_radial.resize(N);
    NL_radial.resize(N * MN);
    NN_angular.resize(N);
    NL_angular.resize(N * MN);
    r12.resize(N * MN * 6);
    Fp.resize(N * annmb.dim);
    sum_fxyz.resize(N * (paramb.n_max_angular + 1) * NUM_OF_ABC);
    if (paramb.charge_mode > 0) {
      D_real.resize(N);
      charge_derivative.resize(N * annmb.dim);
    }
    dftd3.cn.resize(N);
    dftd3.dc6_sum.resize(N);
    dftd3.dc8_sum.resize(N);
    num_atoms = N;
  }
}

void NEP::compute(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& potential,
  std::vector<double>& force,
  std::vector<double>& virial,
  std::vector<double>* descriptor)
{
  if (paramb.model_type != 0) {
    std::cout << "Cannot compute potential using a non-potential NEP model.\n";
    exit(1);
  }

  if (paramb.charge_mode != 0) {
    std::cout << "Cannot use this compute for a qNEP model.\n";
    exit(1);
  }

  const std::size_t N = type.size();
  const int size_x12 = N * MN;

  if (N * 3 != position.size()) {
    std::cout << "Type and position sizes are inconsistent.\n";
    exit(1);
  }
  if (N != potential.size()) {
    std::cout << "Type and potential sizes are inconsistent.\n";
    exit(1);
  }
  if (N * 3 != force.size()) {
    std::cout << "Type and force sizes are inconsistent.\n";
    exit(1);
  }
  if (N * 9 != virial.size()) {
    std::cout << "Type and virial sizes are inconsistent.\n";
    exit(1);
  }
  if (descriptor != nullptr && N * annmb.dim != descriptor->size()) {
    throw std::runtime_error("type and descriptor sizes are inconsistent");
  }

  const bool phase_timing = nep_phase_timer_enabled();
  auto phase_mark = NepPhaseClock::now();
  double phase_setup = 0.0;
  double phase_neighbor = 0.0;
  double phase_table = 0.0;
  double phase_descriptor = 0.0;
  double phase_radial = 0.0;
  double phase_angular = 0.0;
  double phase_zbl = 0.0;

  allocate_memory(N);

  for (std::size_t n = 0; n < potential.size(); ++n) {
    potential[n] = 0.0;
  }
  for (std::size_t n = 0; n < force.size(); ++n) {
    force[n] = 0.0;
  }
  for (std::size_t n = 0; n < virial.size(); ++n) {
    virial[n] = 0.0;
  }
  if (phase_timing) {
    phase_setup = nep_phase_elapsed(phase_mark);
  }

  find_neighbor_list_small_box(
    paramb.rc_radial_max, paramb.rc_angular_max, N, MN, box, position, num_cells, ebox, NN_radial, NL_radial,
    NN_angular, NL_angular, r12);
  if (phase_timing) {
    phase_neighbor = nep_phase_elapsed(phase_mark);
  }
#ifndef USE_TABLE_FOR_RADIAL_FUNCTIONS
  small_box_radial_edge_offsets.resize(N + 1);
  small_box_angular_edge_offsets.resize(N + 1);
  small_box_radial_edge_offsets[0] = 0;
  small_box_angular_edge_offsets[0] = 0;
  for (std::size_t n = 0; n < N; ++n) {
    small_box_radial_edge_offsets[n + 1] =
      small_box_radial_edge_offsets[n] + NN_radial[n];
    small_box_angular_edge_offsets[n + 1] =
      small_box_angular_edge_offsets[n] + NN_angular[n];
  }
  small_box_radial_gnp.resize(
    static_cast<std::size_t>(small_box_radial_edge_offsets[N]) *
    (paramb.n_max_radial + 1));
  const std::size_t angular_cache_size =
    static_cast<std::size_t>(small_box_angular_edge_offsets[N]) *
    (paramb.n_max_angular + 1);
  small_box_angular_gn.resize(angular_cache_size);
  small_box_angular_gnp.resize(angular_cache_size);
#endif
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  prepare_table_small_box(
    N, NN_radial.data(), NL_radial.data(), NN_angular.data(), NL_angular.data(), type.data());
  if (phase_timing) {
    phase_table = nep_phase_elapsed(phase_mark);
  }
#endif

  find_descriptor_small_box(
    true, descriptor != nullptr, false, false, paramb, annmb, N, NN_radial.data(), NL_radial.data(),
    NN_angular.data(), NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12,
    r12.data() + size_x12 * 2, r12.data() + size_x12 * 3, r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gnp_radial.data(), gn_angular.data(), gnp_angular.data(),
#endif
    Fp.data(), sum_fxyz.data(), potential.data(), descriptor ? descriptor->data() : nullptr,
    nullptr, nullptr, false, nullptr,
    ann_q_group, ann_hidden, ann_coeff, ann_fp_group
#ifndef USE_TABLE_FOR_RADIAL_FUNCTIONS
    , small_box_radial_edge_offsets.data(), small_box_radial_gnp.data(),
    small_box_angular_edge_offsets.data(), small_box_angular_gn.data(),
    small_box_angular_gnp.data()
#endif
    );
  if (phase_timing) {
    phase_descriptor = nep_phase_elapsed(phase_mark);
  }

  find_force_radial_small_box(
    false, paramb, annmb, N, NN_radial.data(), NL_radial.data(), type.data(), r12.data(),
    r12.data() + size_x12, r12.data() + size_x12 * 2, Fp.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gnp_radial.data(),
#endif
    force.data(), force.data() + N, force.data() + N * 2, virial.data()
#ifndef USE_TABLE_FOR_RADIAL_FUNCTIONS
    , small_box_radial_edge_offsets.data(), small_box_radial_gnp.data()
#endif
    );
  if (phase_timing) {
    phase_radial = nep_phase_elapsed(phase_mark);
  }

  find_force_angular_small_box(
    false, paramb, annmb, N, NN_angular.data(), NL_angular.data(), type.data(),
    r12.data() + size_x12 * 3, r12.data() + size_x12 * 4, r12.data() + size_x12 * 5,
    Fp.data(), sum_fxyz.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_angular.data(), gnp_angular.data(),
#endif
    force.data(), force.data() + N, force.data() + N * 2, virial.data()
#ifndef USE_TABLE_FOR_RADIAL_FUNCTIONS
    , small_box_angular_edge_offsets.data(), small_box_angular_gn.data(),
    small_box_angular_gnp.data()
#endif
    );
  if (phase_timing) {
    phase_angular = nep_phase_elapsed(phase_mark);
  }

  if (zbl.enabled) {
    find_force_ZBL_small_box(
      N, paramb, zbl, NN_angular.data(), NL_angular.data(), type.data(), r12.data() + size_x12 * 3,
      r12.data() + size_x12 * 4, r12.data() + size_x12 * 5, force.data(), force.data() + N,
      force.data() + N * 2, virial.data(), potential.data());
  }
  if (phase_timing) {
    phase_zbl = nep_phase_elapsed(phase_mark);
    NepPhaseTotals& totals = nep_phase_timer_state().batch;
    ++totals.calls;
    totals.active_atoms += static_cast<long long>(N);
    totals.centers += static_cast<long long>(N);
    for (std::size_t n = 0; n < N; ++n) {
      totals.neighbors += NN_radial[n];
    }
    totals.setup += phase_setup;
    totals.neighbor += phase_neighbor;
    totals.table += phase_table;
    totals.descriptor += phase_descriptor;
    totals.radial += phase_radial;
    totals.angular += phase_angular;
    totals.zbl += phase_zbl;
  }
}

void NEP::find_descriptor(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  const std::vector<double>& spins,
  std::vector<double>& descriptor)
{
  const std::size_t N = type.size();
  const std::size_t size_x12 = N * MN;
  if (!paramb.spin_mode) {
    throw std::runtime_error("spin descriptor requested for a non-spin model");
  }
  if (N * 3 != position.size() || N * 3 != spins.size() ||
      N * annmb.dim != descriptor.size()) {
    throw std::runtime_error("spin input sizes are inconsistent");
  }

  allocate_memory(N);
  find_neighbor_list_small_box(
    paramb.rc_radial_max, paramb.rc_angular_max, N, MN, box, position, num_cells, ebox,
    NN_radial, NL_radial, NN_angular, NL_angular, r12);
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  prepare_table_small_box(
    N, NN_radial.data(), NL_radial.data(), NN_angular.data(), NL_angular.data(), type.data());
#endif
  find_descriptor_small_box(
    false, true, false, false, paramb, annmb, N, NN_radial.data(), NL_radial.data(),
    NN_angular.data(), NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12,
    r12.data() + size_x12 * 2, r12.data() + size_x12 * 3, r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gnp_radial.data(), gn_angular.data(), gnp_angular.data(),
#endif
    Fp.data(), sum_fxyz.data(), nullptr, descriptor.data(), nullptr, nullptr, false, nullptr,
    ann_q_group, ann_hidden, ann_coeff, ann_fp_group);
  fill_spin_descriptor(
    paramb, annmb, static_cast<int>(N), NN_radial.data(), NL_radial.data(), type.data(),
    r12.data(), r12.data() + size_x12, r12.data() + size_x12 * 2, spins.data(),
    descriptor.data());
}

void NEP::compute(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  const std::vector<double>& spins,
  std::vector<double>& potential,
  std::vector<double>& force,
  std::vector<double>& virial,
  std::vector<double>& descriptor,
  std::vector<double>& mforce,
  std::vector<double>* spin_transfer)
{
  const std::size_t N = type.size();
  if (N != potential.size() || N * 3 != force.size() || N * 9 != virial.size() ||
      N * annmb.dim != descriptor.size() || N * 3 != mforce.size()) {
    throw std::runtime_error("spin output sizes are inconsistent");
  }
  if (spin_transfer && spin_transfer->size() != N * 9) {
    throw std::runtime_error("spin-transfer output size is inconsistent");
  }

  const bool phase_timing = nep_phase_timer_enabled();
  auto phase_mark = NepPhaseClock::now();
  double phase_setup = 0.0;
  double phase_neighbor = 0.0;
  double phase_table = 0.0;
  double phase_descriptor = 0.0;
  double phase_ann = 0.0;
  double phase_radial = 0.0;
  double phase_angular = 0.0;
  double phase_zbl = 0.0;
  double phase_spin_gradient = 0.0;
  SpinPhaseBreakdown spin_phase;

  const std::size_t size_x12 = N * MN;
  allocate_memory(N);
  std::fill(force.begin(), force.end(), 0.0);
  std::fill(virial.begin(), virial.end(), 0.0);
  std::fill(mforce.begin(), mforce.end(), 0.0);
  if (spin_transfer) {
    std::fill(spin_transfer->begin(), spin_transfer->end(), 0.0);
  }
  std::fill(potential.begin(), potential.end(), 0.0);
  if (phase_timing) {
    phase_setup = nep_phase_elapsed(phase_mark);
  }

  find_neighbor_list_small_box(
    paramb.rc_radial_max, paramb.rc_angular_max, N, MN, box, position, num_cells, ebox,
    NN_radial, NL_radial, NN_angular, NL_angular, r12);
  if (phase_timing) {
    phase_neighbor = nep_phase_elapsed(phase_mark);
  }
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  prepare_table_small_box(
    N, NN_radial.data(), NL_radial.data(), NN_angular.data(), NL_angular.data(), type.data());
#endif
  if (phase_timing) {
    phase_table = nep_phase_elapsed(phase_mark);
  }

  find_descriptor_small_box(
    false, true, false, false, paramb, annmb, N, NN_radial.data(), NL_radial.data(),
    NN_angular.data(), NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12,
    r12.data() + size_x12 * 2, r12.data() + size_x12 * 3, r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gnp_radial.data(), gn_angular.data(), gnp_angular.data(),
#endif
    Fp.data(), sum_fxyz.data(), nullptr, descriptor.data(), nullptr, nullptr, false, nullptr,
    ann_q_group, ann_hidden, ann_coeff, ann_fp_group);
  if (phase_timing) {
    phase_descriptor = nep_phase_elapsed(phase_mark);
  }

  SpinCache spin_cache;
  fill_spin_descriptor(
    paramb, annmb, static_cast<int>(N), NN_radial.data(), NL_radial.data(), type.data(),
    r12.data(), r12.data() + size_x12, r12.data() + size_x12 * 2, spins.data(),
    descriptor.data(), &spin_cache, phase_timing ? &spin_phase : nullptr);
  if (phase_timing) {
    phase_mark = NepPhaseClock::now();
  }

  for (std::size_t atom = 0; atom < N; ++atom) {
    double q[MAX_DIM] = {0.0};
    double F = 0.0;
    double Fp_local[MAX_DIM] = {0.0};
    double latent[MAX_NEURON] = {0.0};
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = descriptor[static_cast<std::size_t>(d) * N + atom];
    }
    apply_ann_one_layer(
      annmb.dim, annmb.num_neurons1, annmb.w0[type[atom]], annmb.b0[type[atom]],
      annmb.w1[type[atom]], annmb.b1, q, F, Fp_local, latent, false, nullptr);
    potential[atom] = F + spin_baseline[static_cast<std::size_t>(type[atom])];
    for (int d = 0; d < annmb.dim; ++d) {
      Fp[atom * annmb.dim + d] = Fp_local[d] * paramb.q_scaler[d];
    }
  }
  if (phase_timing) {
    phase_ann = nep_phase_elapsed(phase_mark);
  }

  find_force_radial_small_box(
    false, paramb, annmb, static_cast<int>(N), NN_radial.data(), NL_radial.data(), type.data(),
    r12.data(), r12.data() + size_x12, r12.data() + size_x12 * 2, Fp.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gnp_radial.data(),
#endif
    force.data(), force.data() + N, force.data() + N * 2, virial.data());
  if (phase_timing) {
    phase_radial = nep_phase_elapsed(phase_mark);
  }
  find_force_angular_small_box(
    false, paramb, annmb, static_cast<int>(N), NN_angular.data(), NL_angular.data(), type.data(),
    r12.data() + size_x12 * 3, r12.data() + size_x12 * 4, r12.data() + size_x12 * 5,
    Fp.data(), sum_fxyz.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_angular.data(), gnp_angular.data(),
#endif
    force.data(), force.data() + N, force.data() + N * 2, virial.data());
  if (phase_timing) {
    phase_angular = nep_phase_elapsed(phase_mark);
  }
  if (zbl.enabled) {
    find_force_ZBL_small_box(
      static_cast<int>(N), paramb, zbl, NN_angular.data(), NL_angular.data(), type.data(),
      r12.data() + size_x12 * 3, r12.data() + size_x12 * 4, r12.data() + size_x12 * 5,
      force.data(), force.data() + N, force.data() + N * 2, virial.data(), potential.data());
  }
  if (phase_timing) {
    phase_zbl = nep_phase_elapsed(phase_mark);
  }
  add_spin_gradient(
    paramb, annmb, static_cast<int>(N), type.data(), spins.data(), spin_cache, Fp.data(),
    force.data(), virial.data(), mforce.data(),
    spin_transfer ? spin_transfer->data() : nullptr,
    phase_timing ? &spin_phase : nullptr);
  if (phase_timing) {
    phase_spin_gradient = nep_phase_elapsed(phase_mark);
    NepPhaseTotals& totals = nep_phase_timer_state().batch;
    ++totals.calls;
    totals.active_atoms += static_cast<long long>(N);
    totals.centers += static_cast<long long>(N);
    for (std::size_t n = 0; n < N; ++n) {
      totals.neighbors += NN_radial[n];
    }
    totals.setup += phase_setup;
    totals.neighbor += phase_neighbor;
    totals.table += phase_table;
    totals.descriptor += phase_descriptor;
    totals.ann += phase_ann;
    totals.radial += phase_radial;
    totals.angular += phase_angular;
    totals.zbl += phase_zbl;
    totals.spin_setup += spin_phase.setup;
    totals.spin_edges += spin_phase.edges;
    totals.spin_unpack += spin_phase.unpack;
    totals.spin_merge += spin_phase.merge;
    totals.spin_contract += spin_phase.contract;
    totals.spin_chiral += spin_phase.chiral;
    totals.spin_copy += spin_phase.copy;
    totals.spin_gradient += phase_spin_gradient;
    totals.spin_gradient_nonchiral += spin_phase.gradient_nonchiral;
    totals.spin_gradient_chiral += spin_phase.gradient_chiral;
  }
}

void NEP::compute(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& potential,
  std::vector<double>& force,
  std::vector<double>& virial,
  std::vector<double>& charge,
  std::vector<double>& bec)
{
  if (paramb.charge_mode == 0) {
    std::cout << "Can only use this compute for a qNEP model.\n";
    exit(1);
  }

  const std::size_t N = type.size();
  const std::size_t size_x12 = N * MN;

  if (N * 3 != position.size()) {
    std::cout << "Type and position sizes are inconsistent.\n";
    exit(1);
  }
  if (N != potential.size()) {
    std::cout << "Type and potential sizes are inconsistent.\n";
    exit(1);
  }
  if (N * 3 != force.size()) {
    std::cout << "Type and force sizes are inconsistent.\n";
    exit(1);
  }
  if (N * 9 != virial.size()) {
    std::cout << "Type and virial sizes are inconsistent.\n";
    exit(1);
  }
  if (N != charge.size()) {
    std::cout << "Type and charge sizes are inconsistent.\n";
    exit(1);
  }
  if (N * 9 != bec.size()) {
    std::cout << "Type and BEC sizes are inconsistent.\n";
    exit(1);
  }

  allocate_memory(N);

  for (std::size_t n = 0; n < potential.size(); ++n) {
    potential[n] = 0.0;
  }
  for (std::size_t n = 0; n < force.size(); ++n) {
    force[n] = 0.0;
  }
  for (std::size_t n = 0; n < virial.size(); ++n) {
    virial[n] = 0.0;
  }
  for (std::size_t n = 0; n < charge.size(); ++n) {
    charge[n] = 0.0;
  }
  for (std::size_t n = 0; n < bec.size(); ++n) {
    bec[n] = 0.0;
  }

  find_neighbor_list_small_box(
    paramb.rc_radial_max, paramb.rc_angular_max, N, MN, box, position, num_cells, ebox, NN_radial, NL_radial,
    NN_angular, NL_angular, r12);

  find_descriptor_small_box(
    true, false, paramb, annmb, N, NN_radial.data(), NL_radial.data(),
    NN_angular.data(), NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12,
    r12.data() + size_x12 * 2, r12.data() + size_x12 * 3, r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5,
    Fp.data(), sum_fxyz.data(), charge.data(), charge_derivative.data(), potential.data(), nullptr);

  subtract_mean(N, charge.data());

  find_bec_diagonal(N, charge.data(), bec.data());
  find_bec_radial_small_box(
    paramb,
    annmb,
    N,
    NN_radial.data(),
    NL_radial.data(),
    type.data(),
    r12.data(),
    r12.data() + size_x12,
    r12.data() + size_x12 * 2,
    charge_derivative.data(),
    bec.data());
  find_bec_angular_small_box(
    paramb,
    annmb,
    N,
    NN_angular.data(),
    NL_angular.data(),
    type.data(),
    r12.data() + size_x12 * 3,
    r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5,
    charge_derivative.data(),
    sum_fxyz.data(),
    bec.data());
  scale_bec(N, annmb.sqrt_epsilon_inf, bec.data());

  if (paramb.charge_mode == 1 || paramb.charge_mode == 2) {
    ewald.find_force(
      N,
      box.data(),
      charge,
      position,
      D_real,
      force,
      virial,
      potential);
  }

  if (paramb.charge_mode == 1) {
    find_force_charge_real_space_small_box(
      N,
      charge_para,
      NN_radial.data(),
      NL_radial.data(),
      charge.data(),
      r12.data(),
      r12.data() + size_x12,
      r12.data() + size_x12 * 2,
      force.data(),
      force.data() + N,
      force.data() + N * 2,
      virial.data(),
      potential.data(),
      D_real.data());
  }

  if (paramb.charge_mode == 3) {
    find_force_charge_real_space_only_small_box(
      N,
      charge_para,
      NN_radial.data(),
      NL_radial.data(),
      charge.data(),
      r12.data(),
      r12.data() + size_x12,
      r12.data() + size_x12 * 2,
      force.data(),
      force.data() + N,
      force.data() + N * 2,
      virial.data(),
      potential.data(),
      D_real.data());
  }

  // The predicted charges are projected onto the zero-sum subspace above.
  // Apply the transpose of the same projection to dE/dq before propagating it
  // through the charge network: P is symmetric and P^T D = D - mean(D).
  subtract_mean(N, D_real.data());

  find_force_radial_small_box(
    paramb, annmb, N, NN_radial.data(), NL_radial.data(), type.data(), r12.data(),
    r12.data() + size_x12, r12.data() + size_x12 * 2, Fp.data(),
    charge_derivative.data(), D_real.data(),
    force.data(), force.data() + N, force.data() + N * 2, virial.data());

  find_force_angular_small_box(
    paramb, annmb, N, NN_angular.data(), NL_angular.data(), type.data(),
    r12.data() + size_x12 * 3, r12.data() + size_x12 * 4, r12.data() + size_x12 * 5,
    Fp.data(), charge_derivative.data(), D_real.data(),
    sum_fxyz.data(), force.data(), force.data() + N, force.data() + N * 2,
    virial.data());

  if (zbl.enabled) {
    find_force_ZBL_small_box(
      N, paramb, zbl, NN_angular.data(), NL_angular.data(), type.data(), r12.data() + size_x12 * 3,
      r12.data() + size_x12 * 4, r12.data() + size_x12 * 5, force.data(), force.data() + N,
      force.data() + N * 2, virial.data(), potential.data());
  }
}

void NEP::compute_with_dftd3(
  const std::string& xc,
  const double rc_potential,
  const double rc_coordination_number,
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& potential,
  std::vector<double>& force,
  std::vector<double>& virial)
{
  compute(type, box, position, potential, force, virial);
  const std::size_t N = type.size();
  const std::size_t size_x12 = N * MN;
  set_dftd3_para_all(xc, rc_potential, rc_coordination_number);

  find_neighbor_list_small_box(
    dftd3.rc_radial, dftd3.rc_angular, N, MN, box, position, num_cells, ebox, NN_radial, NL_radial,
    NN_angular, NL_angular, r12);
  find_dftd3_coordination_number(
    dftd3, N, NN_angular.data(), NL_angular.data(), type.data(), r12.data() + size_x12 * 3,
    r12.data() + size_x12 * 4, r12.data() + size_x12 * 5);
  add_dftd3_force(
    dftd3, N, NN_radial.data(), NL_radial.data(), type.data(), r12.data() + size_x12 * 0,
    r12.data() + size_x12 * 1, r12.data() + size_x12 * 2, potential.data(), force.data(),
    virial.data());
  add_dftd3_force_extra(
    dftd3, N, NN_angular.data(), NL_angular.data(), type.data(), r12.data() + size_x12 * 3,
    r12.data() + size_x12 * 4, r12.data() + size_x12 * 5, force.data(), virial.data());
}

void NEP::compute_dftd3(
  const std::string& xc,
  const double rc_potential,
  const double rc_coordination_number,
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& potential,
  std::vector<double>& force,
  std::vector<double>& virial)
{
  if (paramb.model_type != 0) {
    std::cout << "Cannot compute potential using a non-potential NEP model.\n";
    exit(1);
  }

  const std::size_t N = type.size();
  const std::size_t size_x12 = N * MN;

  if (N * 3 != position.size()) {
    std::cout << "Type and position sizes are inconsistent.\n";
    exit(1);
  }
  if (N != potential.size()) {
    std::cout << "Type and potential sizes are inconsistent.\n";
    exit(1);
  }
  if (N * 3 != force.size()) {
    std::cout << "Type and force sizes are inconsistent.\n";
    exit(1);
  }
  if (N * 9 != virial.size()) {
    std::cout << "Type and virial sizes are inconsistent.\n";
    exit(1);
  }

  allocate_memory(N);

  for (std::size_t n = 0; n < potential.size(); ++n) {
    potential[n] = 0.0;
  }
  for (std::size_t n = 0; n < force.size(); ++n) {
    force[n] = 0.0;
  }
  for (std::size_t n = 0; n < virial.size(); ++n) {
    virial[n] = 0.0;
  }

  set_dftd3_para_all(xc, rc_potential, rc_coordination_number);

  find_neighbor_list_small_box(
    dftd3.rc_radial, dftd3.rc_angular, N, MN, box, position, num_cells, ebox, NN_radial, NL_radial,
    NN_angular, NL_angular, r12);
  find_dftd3_coordination_number(
    dftd3, N, NN_angular.data(), NL_angular.data(), type.data(), r12.data() + size_x12 * 3,
    r12.data() + size_x12 * 4, r12.data() + size_x12 * 5);
  add_dftd3_force(
    dftd3, N, NN_radial.data(), NL_radial.data(), type.data(), r12.data() + size_x12 * 0,
    r12.data() + size_x12 * 1, r12.data() + size_x12 * 2, potential.data(), force.data(),
    virial.data());
  add_dftd3_force_extra(
    dftd3, N, NN_angular.data(), NL_angular.data(), type.data(), r12.data() + size_x12 * 3,
    r12.data() + size_x12 * 4, r12.data() + size_x12 * 5, force.data(), virial.data());
}

void NEP::find_descriptor(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& descriptor)
{
  const std::size_t N = type.size();
  const std::size_t size_x12 = N * MN;

  if (N * 3 != position.size()) {
    std::cout << "Type and position sizes are inconsistent.\n";
    exit(1);
  }
  if (N * annmb.dim != descriptor.size()) {
    std::cout << "Type and descriptor sizes are inconsistent.\n";
    exit(1);
  }

  allocate_memory(N);

  find_neighbor_list_small_box(
    paramb.rc_radial_max, paramb.rc_angular_max, N, MN, box, position, num_cells, ebox, NN_radial, NL_radial,
    NN_angular, NL_angular, r12);
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  prepare_table_small_box(
    N, NN_radial.data(), NL_radial.data(), NN_angular.data(), NL_angular.data(), type.data());
#endif

  if (paramb.charge_mode > 0) {
    find_descriptor_small_box(
      false, true, paramb, annmb, N, NN_radial.data(), NL_radial.data(),
      NN_angular.data(), NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12,
      r12.data() + size_x12 * 2, r12.data() + size_x12 * 3, r12.data() + size_x12 * 4,
      r12.data() + size_x12 * 5,
      Fp.data(), sum_fxyz.data(), nullptr, nullptr, nullptr, descriptor.data());
  } else {
    find_descriptor_small_box(
      false, true, false, false, paramb, annmb, N, NN_radial.data(), NL_radial.data(),
      NN_angular.data(), NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12,
      r12.data() + size_x12 * 2, r12.data() + size_x12 * 3, r12.data() + size_x12 * 4,
      r12.data() + size_x12 * 5,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
      gn_radial.data(), gnp_radial.data(), gn_angular.data(), gnp_angular.data(),
#endif
      Fp.data(), sum_fxyz.data(), nullptr, descriptor.data(), nullptr, nullptr, false, nullptr,
      ann_q_group, ann_hidden, ann_coeff, ann_fp_group);
  }
}

void NEP::find_latent_space(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& latent_space)
{
  const std::size_t N = type.size();
  const std::size_t size_x12 = N * MN;

  if (N * 3 != position.size()) {
    std::cout << "Type and position sizes are inconsistent.\n";
    exit(1);
  }
  if (N * annmb.num_neurons1 != latent_space.size()) {
    std::cout << "Type and latent_space sizes are inconsistent.\n";
    exit(1);
  }

  allocate_memory(N);

  find_neighbor_list_small_box(
    paramb.rc_radial_max, paramb.rc_angular_max, N, MN, box, position, num_cells, ebox, NN_radial, NL_radial,
    NN_angular, NL_angular, r12);
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  prepare_table_small_box(
    N, NN_radial.data(), NL_radial.data(), NN_angular.data(), NL_angular.data(), type.data());
#endif

  find_descriptor_small_box(
    false, false, true, false, paramb, annmb, N, NN_radial.data(), NL_radial.data(),
    NN_angular.data(), NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12,
    r12.data() + size_x12 * 2, r12.data() + size_x12 * 3, r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gnp_radial.data(), gn_angular.data(), gnp_angular.data(),
#endif
    Fp.data(), sum_fxyz.data(), nullptr, nullptr, latent_space.data(), nullptr, false, nullptr,
    ann_q_group, ann_hidden, ann_coeff, ann_fp_group);
}

void NEP::find_B_projection(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& B_projection)
{
  const std::size_t N = type.size();
  const std::size_t size_x12 = N * MN;

  if (N * 3 != position.size()) {
    std::cout << "Type and position sizes are inconsistent.\n";
    exit(1);
  }
  if (N * annmb.num_neurons1 * (annmb.dim + 2) != B_projection.size()) {
    std::cout << "Type and B_projection sizes are inconsistent.\n";
    exit(1);
  }

  allocate_memory(N);
  find_neighbor_list_small_box(
    paramb.rc_radial_max, paramb.rc_angular_max, N, MN, box, position, num_cells, ebox, NN_radial, NL_radial,
    NN_angular, NL_angular, r12);
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  prepare_table_small_box(
    N, NN_radial.data(), NL_radial.data(), NN_angular.data(), NL_angular.data(), type.data());
#endif

  find_descriptor_small_box(
    false, false, false, false, paramb, annmb, N, NN_radial.data(), NL_radial.data(),
    NN_angular.data(), NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12,
    r12.data() + size_x12 * 2, r12.data() + size_x12 * 3, r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gnp_radial.data(), gn_angular.data(), gnp_angular.data(),
#endif
    Fp.data(), sum_fxyz.data(), nullptr, nullptr, nullptr, nullptr, true, B_projection.data(),
    ann_q_group, ann_hidden, ann_coeff, ann_fp_group);
}

void NEP::find_dipole(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& dipole)
{
  if (paramb.model_type != 1) {
    std::cout << "Cannot compute dipole using a non-dipole NEP model.\n";
    exit(1);
  }

  const std::size_t N = type.size();
  const std::size_t size_x12 = N * MN;

  if (N * 3 != position.size()) {
    std::cout << "Type and position sizes are inconsistent.\n";
    exit(1);
  }

  allocate_memory(N);
  std::vector<double> potential(N);  // not used but needed for find_descriptor_small_box
  std::vector<double> virial(N * 3); // need the 3 diagonal components only

  for (std::size_t n = 0; n < potential.size(); ++n) {
    potential[n] = 0.0;
  }
  for (std::size_t n = 0; n < virial.size(); ++n) {
    virial[n] = 0.0;
  }

  find_neighbor_list_small_box(
    paramb.rc_radial_max, paramb.rc_angular_max, N, MN, box, position, num_cells, ebox, NN_radial, NL_radial,
    NN_angular, NL_angular, r12);
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  prepare_table_small_box(
    N, NN_radial.data(), NL_radial.data(), NN_angular.data(), NL_angular.data(), type.data());
#endif

  find_descriptor_small_box(
    true, false, false, false, paramb, annmb, N, NN_radial.data(), NL_radial.data(),
    NN_angular.data(), NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12,
    r12.data() + size_x12 * 2, r12.data() + size_x12 * 3, r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gnp_radial.data(), gn_angular.data(), gnp_angular.data(),
#endif
    Fp.data(), sum_fxyz.data(), potential.data(), nullptr, nullptr, nullptr, false, nullptr,
    ann_q_group, ann_hidden, ann_coeff, ann_fp_group);

  find_force_radial_small_box(
    true, paramb, annmb, N, NN_radial.data(), NL_radial.data(), type.data(), r12.data(),
    r12.data() + size_x12, r12.data() + size_x12 * 2, Fp.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gnp_radial.data(),
#endif
    nullptr, nullptr, nullptr, virial.data());

  find_force_angular_small_box(
    true, paramb, annmb, N, NN_angular.data(), NL_angular.data(), type.data(),
    r12.data() + size_x12 * 3, r12.data() + size_x12 * 4, r12.data() + size_x12 * 5,
    Fp.data(), sum_fxyz.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_angular.data(), gnp_angular.data(),
#endif
    nullptr, nullptr, nullptr, virial.data());

  for (int d = 0; d < 3; ++d) {
    dipole[d] = 0.0;
    for (std::size_t n = 0; n < N; ++n) {
      dipole[d] += virial[d * N + n];
    }
  }
}

void NEP::find_polarizability(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& polarizability)
{
  if (paramb.model_type != 2) {
    std::cout << "Cannot compute polarizability using a non-polarizability NEP model.\n";
    exit(1);
  }

  const std::size_t N = type.size();
  const std::size_t size_x12 = N * MN;

  if (N * 3 != position.size()) {
    std::cout << "Type and position sizes are inconsistent.\n";
    exit(1);
  }

  allocate_memory(N);
  std::vector<double> potential(N);  // not used but needed for find_descriptor_small_box
  std::vector<double> virial(N * 9); // per-atom polarizability

  for (std::size_t n = 0; n < potential.size(); ++n) {
    potential[n] = 0.0;
  }
  for (std::size_t n = 0; n < virial.size(); ++n) {
    virial[n] = 0.0;
  }

  find_neighbor_list_small_box(
    paramb.rc_radial_max, paramb.rc_angular_max, N, MN, box, position, num_cells, ebox, NN_radial, NL_radial,
    NN_angular, NL_angular, r12);
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  prepare_table_small_box(
    N, NN_radial.data(), NL_radial.data(), NN_angular.data(), NL_angular.data(), type.data());
#endif

  find_descriptor_small_box(
    true, false, false, true, paramb, annmb, N, NN_radial.data(), NL_radial.data(),
    NN_angular.data(), NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12,
    r12.data() + size_x12 * 2, r12.data() + size_x12 * 3, r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gnp_radial.data(), gn_angular.data(), gnp_angular.data(),
#endif
    Fp.data(), sum_fxyz.data(), potential.data(), nullptr, nullptr, virial.data(), false, nullptr,
    ann_q_group, ann_hidden, ann_coeff, ann_fp_group);

  find_force_radial_small_box(
    false, paramb, annmb, N, NN_radial.data(), NL_radial.data(), type.data(), r12.data(),
    r12.data() + size_x12, r12.data() + size_x12 * 2, Fp.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gnp_radial.data(),
#endif
    nullptr, nullptr, nullptr, virial.data());

  find_force_angular_small_box(
    false, paramb, annmb, N, NN_angular.data(), NL_angular.data(), type.data(),
    r12.data() + size_x12 * 3, r12.data() + size_x12 * 4, r12.data() + size_x12 * 5,
    Fp.data(), sum_fxyz.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_angular.data(), gnp_angular.data(),
#endif
    nullptr, nullptr, nullptr, virial.data());

  for (int d = 0; d < 6; ++d) {
    polarizability[d] = 0.0;
  }
  for (std::size_t n = 0; n < N; ++n) {
    polarizability[0] += virial[0 * N + n]; // xx
    polarizability[1] += virial[4 * N + n]; // yy
    polarizability[2] += virial[8 * N + n]; // zz
    polarizability[3] += virial[1 * N + n]; // xy
    polarizability[4] += virial[5 * N + n]; // yz
    polarizability[5] += virial[6 * N + n]; // zx
  }
}

void NEP::compute_for_lammps(
  int nlocal,
  int N,
  int* ilist,
  int* NN,
  int** NL,
  int* type,
  int* type_map,
  double** pos,
  double& total_potential,
  double total_virial[6],
  double* potential,
  double** force,
  double** virial)
{
  const bool phase_timing = nep_phase_timer_enabled();
  auto phase_mark = NepPhaseClock::now();
  double phase_setup = 0.0;
  double phase_cache = 0.0;
  double phase_table = 0.0;
  double phase_descriptor = 0.0;
  double phase_scratch = 0.0;
  double phase_radial = 0.0;
  double phase_angular = 0.0;
  double phase_reduce = 0.0;
  double phase_zbl = 0.0;

  if (num_atoms < nlocal) {
    Fp.resize(nlocal * annmb.dim);
    sum_fxyz.resize(nlocal * (paramb.n_max_angular + 1) * NUM_OF_ABC);
    num_atoms = nlocal;
  }
  if (phase_timing) {
    phase_setup = nep_phase_elapsed(phase_mark);
  }

  LammpsRadialEdgeCacheView lammps_radial_cache;
  LammpsAngularEdgeCacheView lammps_angular_cache;
  const int n_max_radial_plus_1 = paramb.n_max_radial + 1;
  const int n_max_angular_plus_1 = paramb.n_max_angular + 1;
#if defined(_OPENMP)
  const bool use_lammps_radial_cache = omp_get_max_threads() > 1;
  const bool use_lammps_angular_cache = omp_get_max_threads() > 1;
#else
  const bool use_lammps_radial_cache = false;
  const bool use_lammps_angular_cache = false;
#endif
  if (use_lammps_radial_cache && N > 0 && n_max_radial_plus_1 > 0) {
    lammps_radial_edge_offsets.resize(N + 1);
    int edge_count = 0;
    for (int ii = 0; ii < N; ++ii) {
      lammps_radial_edge_offsets[ii] = edge_count;
      edge_count += NN[ilist[ii]];
    }
    lammps_radial_edge_offsets[N] = edge_count;
    if (edge_count > 0) {
      const std::size_t edge_count_size = static_cast<std::size_t>(edge_count);
      const std::size_t coefficient_count =
        edge_count_size * static_cast<std::size_t>(n_max_radial_plus_1);
      lammps_radial_edge_neighbors.resize(edge_count_size);
      lammps_radial_edge_x12.resize(edge_count_size);
      lammps_radial_edge_y12.resize(edge_count_size);
      lammps_radial_edge_z12.resize(edge_count_size);
      lammps_radial_edge_d12.resize(edge_count_size);
      lammps_radial_edge_gnp.resize(coefficient_count);

      lammps_radial_cache.num_centers = N;
      lammps_radial_cache.n_max_radial_plus_1 = n_max_radial_plus_1;
      lammps_radial_cache.offsets = lammps_radial_edge_offsets.data();
      lammps_radial_cache.neighbors = lammps_radial_edge_neighbors.data();
      lammps_radial_cache.x12 = lammps_radial_edge_x12.data();
      lammps_radial_cache.y12 = lammps_radial_edge_y12.data();
      lammps_radial_cache.z12 = lammps_radial_edge_z12.data();
      lammps_radial_cache.d12 = lammps_radial_edge_d12.data();
      lammps_radial_cache.gnp = lammps_radial_edge_gnp.data();
    }
  }
  if (use_lammps_angular_cache && N > 0 && n_max_angular_plus_1 > 0) {
    lammps_angular_edge_offsets.resize(N + 1);
    int edge_count = 0;
    for (int ii = 0; ii < N; ++ii) {
      lammps_angular_edge_offsets[ii] = edge_count;
      edge_count += NN[ilist[ii]];
    }
    lammps_angular_edge_offsets[N] = edge_count;
    if (edge_count > 0) {
      const std::size_t edge_count_size = static_cast<std::size_t>(edge_count);
      const std::size_t coefficient_count =
        edge_count_size * static_cast<std::size_t>(n_max_angular_plus_1);
      lammps_angular_edge_neighbors.resize(edge_count_size);
      lammps_angular_edge_x12.resize(edge_count_size);
      lammps_angular_edge_y12.resize(edge_count_size);
      lammps_angular_edge_z12.resize(edge_count_size);
      lammps_angular_edge_d12.resize(edge_count_size);
      lammps_angular_edge_gn.resize(coefficient_count);
      lammps_angular_edge_gnp.resize(coefficient_count);

      lammps_angular_cache.num_centers = N;
      lammps_angular_cache.n_max_angular_plus_1 = n_max_angular_plus_1;
      lammps_angular_cache.offsets = lammps_angular_edge_offsets.data();
      lammps_angular_cache.neighbors = lammps_angular_edge_neighbors.data();
      lammps_angular_cache.x12 = lammps_angular_edge_x12.data();
      lammps_angular_cache.y12 = lammps_angular_edge_y12.data();
      lammps_angular_cache.z12 = lammps_angular_edge_z12.data();
      lammps_angular_cache.d12 = lammps_angular_edge_d12.data();
      lammps_angular_cache.gn = lammps_angular_edge_gn.data();
      lammps_angular_cache.gnp = lammps_angular_edge_gnp.data();
    }
  }
  if (phase_timing) {
    phase_cache = nep_phase_elapsed(phase_mark);
  }

#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  prepare_table_for_lammps(N, ilist, NN, NL, type, type_map);
  if (phase_timing) {
    phase_table = nep_phase_elapsed(phase_mark);
  }
#endif

  find_descriptor_for_lammps(
    paramb, annmb, nlocal, N, ilist, NN, NL, type, type_map, pos,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gnp_radial.data(), gn_angular.data(), gnp_angular.data(),
#endif
    Fp.data(), sum_fxyz.data(), total_potential, potential, &lammps_radial_cache,
    &lammps_angular_cache, ann_q_group, ann_hidden, ann_coeff, ann_fp_group);
  if (phase_timing) {
    phase_descriptor = nep_phase_elapsed(phase_mark);
  }

  LammpsThreadLocalScratchView lammps_scratch;
#if defined(_OPENMP)
  const int num_threads = omp_get_max_threads();
  if (num_threads > 1 && N > 0) {
    const int force_rows =
      infer_lammps_touched_rows(N, ilist, NN, NL, lammps_touched_rows, lammps_touched_marks, lammps_touched_stamp);
    if (force_rows > 0) {
      const std::size_t force_size = static_cast<std::size_t>(num_threads) * 3 * force_rows;
      const std::size_t total_virial_size =
        static_cast<std::size_t>(num_threads) * kLammpsTotalVirialStride;
      const std::size_t virial_size = static_cast<std::size_t>(num_threads) * 9 * force_rows;
      if (lammps_force_private.size() < force_size) {
        lammps_force_private.resize(force_size);
      }
      if (lammps_total_virial_private.size() < total_virial_size) {
        lammps_total_virial_private.resize(total_virial_size);
      }
      if (virial && lammps_virial_private.size() < virial_size) {
        lammps_virial_private.resize(virial_size);
      }

      lammps_scratch.force_rows = force_rows;
      lammps_scratch.num_threads = num_threads;
      lammps_scratch.dense_rows =
        static_cast<std::size_t>(force_rows) * 3 <= lammps_touched_rows.size() * 4;
      lammps_scratch.touched_rows = &lammps_touched_rows;
      lammps_scratch.force_private = lammps_force_private.data();
      lammps_scratch.total_virial_private = lammps_total_virial_private.data();
      lammps_scratch.virial_private = virial ? lammps_virial_private.data() : nullptr;
      zero_lammps_thread_local_scratch(lammps_scratch, virial != nullptr);
    }
  }
#endif
  if (phase_timing) {
    phase_scratch = nep_phase_elapsed(phase_mark);
  }

  find_force_radial_for_lammps(
    paramb, annmb, nlocal, N, ilist, NN, NL, type, type_map, pos, Fp.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gnp_radial.data(),
#endif
    &lammps_radial_cache, force, total_virial, virial, &lammps_scratch);
  if (phase_timing) {
    phase_radial = nep_phase_elapsed(phase_mark);
  }
  find_force_angular_for_lammps(
    paramb, annmb, nlocal, N, ilist, NN, NL, type, type_map, pos, Fp.data(),
    sum_fxyz.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_angular.data(), gnp_angular.data(),
#endif
    force, total_virial, virial, &lammps_angular_cache, &lammps_scratch);
  if (phase_timing) {
    phase_angular = nep_phase_elapsed(phase_mark);
  }
  if (zbl.enabled) {
    find_force_ZBL_for_lammps(
      paramb, zbl, N, ilist, NN, NL, type, type_map, pos, force, total_virial, virial,
      total_potential, potential, &lammps_scratch);
  }
  if (phase_timing) {
    phase_zbl = nep_phase_elapsed(phase_mark);
  }
#if defined(_OPENMP)
  if (lammps_thread_scratch_active(&lammps_scratch)) {
    reduce_lammps_thread_local_force_virial(lammps_scratch, force, total_virial, virial);
  }
#endif
  if (phase_timing) {
    phase_reduce = nep_phase_elapsed(phase_mark);
    NepPhaseTotals& totals = nep_phase_timer_state().lammps;
    ++totals.calls;
    totals.active_atoms += nlocal;
    totals.centers += N;
    for (int ii = 0; ii < N; ++ii) {
      totals.neighbors += NN[ilist[ii]];
    }
    totals.setup += phase_setup;
    totals.cache += phase_cache;
    totals.table += phase_table;
    totals.descriptor += phase_descriptor;
    totals.scratch += phase_scratch;
    totals.radial += phase_radial;
    totals.angular += phase_angular;
    totals.reduce += phase_reduce;
    totals.zbl += phase_zbl;
  }
}

void NEP::compute_for_lammps(
  int nlocal,
  int inum,
  int* ilist,
  int* NN,
  int** NL,
  int* type,
  int* type_map,
  double** pos,
  double** spins,
  double& total_potential,
  double total_virial[6],
  double* potential,
  double** force,
  double** mforce,
  double** virial,
  double** spin_transfer)
{
  const bool phase_timing = nep_phase_timer_enabled();
  SpinPhaseBreakdown spin_phase;

  if (!spins) {
    throw std::runtime_error("spin LAMMPS path requires spins");
  }
  if (!mforce) {
    throw std::runtime_error("spin LAMMPS path requires magnetic-force output");
  }

  int atom_capacity = nlocal;
  for (int ii = 0; ii < inum; ++ii) {
    const int i = ilist[ii];
    atom_capacity = std::max(atom_capacity, i + 1);
    for (int jj = 0; jj < NN[i]; ++jj) {
      atom_capacity = std::max(atom_capacity, NL[i][jj] + 1);
    }
  }

  if (num_atoms < atom_capacity) {
    Fp.resize(static_cast<std::size_t>(atom_capacity) * annmb.dim);
    sum_fxyz.resize(
      static_cast<std::size_t>(atom_capacity) * (paramb.n_max_angular + 1) * NUM_OF_ABC);
    num_atoms = atom_capacity;
  }

  lammps_spin_types.resize(static_cast<std::size_t>(atom_capacity));
  lammps_spin_spins_aos.resize(static_cast<std::size_t>(atom_capacity) * 3);
  lammps_spin_descriptor.resize(static_cast<std::size_t>(inum) * annmb.dim);

#if defined(_OPENMP)
  const int num_threads = omp_get_max_threads();
  const bool use_parallel_atoms = num_threads > 1 && atom_capacity > 256;
#else
  const int num_threads = 1;
  const bool use_parallel_atoms = false;
#endif

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_atoms)
#endif
  for (int atom = 0; atom < atom_capacity; ++atom) {
    lammps_spin_types[atom] = type_map[type[atom]];
    const double mu = spins[atom][3];
    for (int d = 0; d < 3; ++d) {
      lammps_spin_spins_aos[static_cast<std::size_t>(atom) * 3 + d] =
        mu * spins[atom][d];
    }
  }

  std::fill(Fp.begin(), Fp.end(), 0.0);
  std::fill(sum_fxyz.begin(), sum_fxyz.end(), 0.0);
  total_potential = 0.0;
  std::fill(total_virial, total_virial + 6, 0.0);

  LammpsRadialEdgeCacheView lammps_radial_cache;
  LammpsAngularEdgeCacheView lammps_angular_cache;
  const int n_max_radial_plus_1 = paramb.n_max_radial + 1;
  const int n_max_angular_plus_1 = paramb.n_max_angular + 1;
#if defined(_OPENMP)
  const bool use_lammps_radial_cache = true;
  const bool use_lammps_angular_cache = omp_get_max_threads() > 1;
#else
  const bool use_lammps_radial_cache = false;
  const bool use_lammps_angular_cache = false;
#endif
  if (use_lammps_radial_cache && inum > 0 && n_max_radial_plus_1 > 0) {
    lammps_radial_edge_offsets.resize(inum + 1);
    int edge_count = 0;
    for (int ii = 0; ii < inum; ++ii) {
      const int i = ilist[ii];
      lammps_radial_edge_offsets[ii] = edge_count;
      edge_count += NN[i];
    }
    lammps_radial_edge_offsets[inum] = edge_count;
    if (edge_count > 0) {
      const std::size_t edge_count_size = static_cast<std::size_t>(edge_count);
      const std::size_t coefficient_count =
        edge_count_size * static_cast<std::size_t>(n_max_radial_plus_1);
      lammps_radial_edge_neighbors.resize(edge_count_size);
      lammps_radial_edge_x12.resize(edge_count_size);
      lammps_radial_edge_y12.resize(edge_count_size);
      lammps_radial_edge_z12.resize(edge_count_size);
      lammps_radial_edge_d12.resize(edge_count_size);
      lammps_radial_edge_gnp.resize(coefficient_count);

      lammps_radial_cache.num_centers = inum;
      lammps_radial_cache.n_max_radial_plus_1 = n_max_radial_plus_1;
      lammps_radial_cache.offsets = lammps_radial_edge_offsets.data();
      lammps_radial_cache.neighbors = lammps_radial_edge_neighbors.data();
      lammps_radial_cache.x12 = lammps_radial_edge_x12.data();
      lammps_radial_cache.y12 = lammps_radial_edge_y12.data();
      lammps_radial_cache.z12 = lammps_radial_edge_z12.data();
      lammps_radial_cache.d12 = lammps_radial_edge_d12.data();
      lammps_radial_cache.gnp = lammps_radial_edge_gnp.data();
    }
  }
  if (use_lammps_angular_cache && inum > 0 && n_max_angular_plus_1 > 0) {
    lammps_angular_edge_offsets.resize(inum + 1);
    int edge_count = 0;
    for (int ii = 0; ii < inum; ++ii) {
      const int i = ilist[ii];
      lammps_angular_edge_offsets[ii] = edge_count;
      edge_count += NN[i];
    }
    lammps_angular_edge_offsets[inum] = edge_count;
    if (edge_count > 0) {
      const std::size_t edge_count_size = static_cast<std::size_t>(edge_count);
      const std::size_t coefficient_count =
        edge_count_size * static_cast<std::size_t>(n_max_angular_plus_1);
      lammps_angular_edge_neighbors.resize(edge_count_size);
      lammps_angular_edge_x12.resize(edge_count_size);
      lammps_angular_edge_y12.resize(edge_count_size);
      lammps_angular_edge_z12.resize(edge_count_size);
      lammps_angular_edge_d12.resize(edge_count_size);
      lammps_angular_edge_gn.resize(coefficient_count);
      lammps_angular_edge_gnp.resize(coefficient_count);

      lammps_angular_cache.num_centers = inum;
      lammps_angular_cache.n_max_angular_plus_1 = n_max_angular_plus_1;
      lammps_angular_cache.offsets = lammps_angular_edge_offsets.data();
      lammps_angular_cache.neighbors = lammps_angular_edge_neighbors.data();
      lammps_angular_cache.x12 = lammps_angular_edge_x12.data();
      lammps_angular_cache.y12 = lammps_angular_edge_y12.data();
      lammps_angular_cache.z12 = lammps_angular_edge_z12.data();
      lammps_angular_cache.d12 = lammps_angular_edge_d12.data();
      lammps_angular_cache.gn = lammps_angular_edge_gn.data();
      lammps_angular_cache.gnp = lammps_angular_edge_gnp.data();
    }
  }

#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  prepare_table_for_lammps(inum, ilist, NN, NL, type, type_map);
#endif

  find_descriptor_for_lammps(
    paramb, annmb, atom_capacity, inum, ilist, NN, NL, type, type_map, pos,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gnp_radial.data(), gn_angular.data(), gnp_angular.data(),
#endif
    Fp.data(), sum_fxyz.data(), total_potential, potential, &lammps_radial_cache,
    &lammps_angular_cache, ann_q_group, ann_hidden, ann_coeff, ann_fp_group,
    nullptr, true, lammps_spin_descriptor.data());

  LammpsThreadLocalScratchView lammps_scratch;
#if defined(_OPENMP)
  if (inum > 0) {
    const int force_rows =
      infer_lammps_touched_rows(
        inum, ilist, NN, NL, lammps_touched_rows, lammps_touched_marks,
        lammps_touched_stamp);
    if (force_rows > 0) {
      const std::size_t force_size = static_cast<std::size_t>(num_threads) * 3 * force_rows;
      const std::size_t total_virial_size =
        static_cast<std::size_t>(num_threads) * kLammpsTotalVirialStride;
      const std::size_t virial_size = static_cast<std::size_t>(num_threads) * 9 * force_rows;
      const std::size_t mforce_size = static_cast<std::size_t>(num_threads) * 3 * force_rows;
      const std::size_t spin_transfer_size =
        static_cast<std::size_t>(num_threads) * 9 * force_rows;
      if (lammps_force_private.size() < force_size) {
        lammps_force_private.resize(force_size);
      }
      if (lammps_total_virial_private.size() < total_virial_size) {
        lammps_total_virial_private.resize(total_virial_size);
      }
      if (virial && lammps_virial_private.size() < virial_size) {
        lammps_virial_private.resize(virial_size);
      }
      if (lammps_mforce_private.size() < mforce_size) {
        lammps_mforce_private.resize(mforce_size);
      }
      if (spin_transfer && lammps_spin_transfer_private.size() < spin_transfer_size) {
        lammps_spin_transfer_private.resize(spin_transfer_size);
      }

      lammps_scratch.force_rows = force_rows;
      lammps_scratch.num_threads = num_threads;
      lammps_scratch.dense_rows =
        static_cast<std::size_t>(force_rows) * 3 <= lammps_touched_rows.size() * 4;
      lammps_scratch.touched_rows = &lammps_touched_rows;
      lammps_scratch.force_private = lammps_force_private.data();
      lammps_scratch.total_virial_private = lammps_total_virial_private.data();
      lammps_scratch.virial_private = virial ? lammps_virial_private.data() : nullptr;
      lammps_scratch.mforce_private = lammps_mforce_private.data();
      lammps_scratch.spin_transfer_private = spin_transfer
        ? lammps_spin_transfer_private.data()
        : nullptr;
      zero_lammps_thread_local_scratch(lammps_scratch, virial != nullptr);
    }
  }
#endif

  const bool fused_spin = compute_spin_lammps_fused_center(
    paramb, annmb, atom_capacity, inum, ilist, NN, lammps_spin_types.data(),
    lammps_spin_spins_aos.data(), spin_baseline, &lammps_radial_cache,
    lammps_spin_descriptor.data(), Fp.data(), total_potential, potential,
    &lammps_scratch, phase_timing ? &spin_phase : nullptr);

  if (!fused_spin) {
    lammps_spin_spins_soa.resize(static_cast<std::size_t>(atom_capacity) * 3);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_atoms)
#endif
    for (int atom = 0; atom < atom_capacity; ++atom) {
      for (int d = 0; d < 3; ++d) {
        lammps_spin_spins_soa[static_cast<std::size_t>(d) * atom_capacity + atom] =
          lammps_spin_spins_aos[static_cast<std::size_t>(atom) * 3 + d];
      }
    }
    find_spin_descriptor_for_lammps(
      paramb, annmb, atom_capacity, inum, ilist, NN, NL, lammps_spin_types.data(), pos,
      lammps_spin_spins_soa.data(), lammps_spin_descriptor.data(), lammps_spin_cache,
      phase_timing ? &spin_phase : nullptr, &lammps_radial_cache);

    apply_spin_ann_for_lammps(
      paramb, annmb, inum, ilist, lammps_spin_types.data(), spin_baseline,
      lammps_spin_descriptor.data(), Fp.data(), total_potential, potential,
      use_parallel_atoms, num_threads);
  }

  find_force_radial_for_lammps(
    paramb, annmb, nlocal, inum, ilist, NN, NL, type, type_map, pos, Fp.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gnp_radial.data(),
#endif
    &lammps_radial_cache, force, total_virial, virial, &lammps_scratch);
  find_force_angular_for_lammps(
    paramb, annmb, nlocal, inum, ilist, NN, NL, type, type_map, pos, Fp.data(),
    sum_fxyz.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_angular.data(), gnp_angular.data(),
#endif
    force, total_virial, virial, &lammps_angular_cache, &lammps_scratch);
  if (zbl.enabled) {
    find_force_ZBL_for_lammps(
      paramb, zbl, inum, ilist, NN, NL, type, type_map, pos, force, total_virial, virial,
      total_potential, potential, &lammps_scratch);
  }

  if (!fused_spin) {
    find_spin_force_for_lammps(
      paramb, annmb, atom_capacity, inum, ilist, lammps_spin_types.data(),
      lammps_spin_spins_soa.data(), lammps_spin_cache, Fp.data(), force, mforce,
      total_virial, virial, spin_transfer, &lammps_scratch, &lammps_spin_gradient_scratch,
      phase_timing ? &spin_phase : nullptr);
  }

#if defined(_OPENMP)
  if (lammps_spin_scratch_active(&lammps_scratch)) {
    reduce_lammps_thread_local_force_virial(
      lammps_scratch, force, total_virial, virial, mforce, spin_transfer);
  }
#endif

  if (phase_timing) {
    NepPhaseTotals& totals = nep_phase_timer_state().lammps;
    ++totals.calls;
    totals.active_atoms += nlocal;
    totals.centers += inum;
    for (int ii = 0; ii < inum; ++ii) {
      totals.neighbors += NN[ilist[ii]];
    }
    totals.spin_setup += spin_phase.setup;
    totals.spin_edges += spin_phase.edges;
    totals.spin_unpack += spin_phase.unpack;
    totals.spin_merge += spin_phase.merge;
    totals.spin_contract += spin_phase.contract;
    totals.spin_chiral += spin_phase.chiral;
    totals.spin_copy += spin_phase.copy;
    totals.spin_ann += spin_phase.ann;
    totals.spin_gradient += spin_phase.gradient_nonchiral + spin_phase.gradient_chiral;
    totals.spin_gradient_nonchiral += spin_phase.gradient_nonchiral;
    totals.spin_gradient_chiral += spin_phase.gradient_chiral;
  }
}

bool NEP::set_dftd3_para_one(
  const std::string& functional_input,
  const std::string& functional_library,
  const double s6,
  const double a1,
  const double s8,
  const double a2)
{
  if (functional_input == functional_library) {
    dftd3.s6 = s6;
    dftd3.a1 = a1;
    dftd3.s8 = s8;
    dftd3.a2 = a2 * dftd3para::Bohr;
    return true;
  }
  return false;
}

void NEP::set_dftd3_para_all(
  const std::string& functional_input,
  const double rc_potential,
  const double rc_coordination_number)
{

  dftd3.rc_radial = rc_potential;
  dftd3.rc_angular = rc_coordination_number;

  std::string functional = functional_input;
  std::transform(functional.begin(), functional.end(), functional.begin(), [](unsigned char c) {
    return std::tolower(c);
  });

  bool valid = false;
  valid = valid || set_dftd3_para_one(functional, "b1b95", 1.000, 0.2092, 1.4507, 5.5545);
  valid = valid || set_dftd3_para_one(functional, "b2gpplyp", 0.560, 0.0000, 0.2597, 6.3332);
  valid = valid || set_dftd3_para_one(functional, "b2plyp", 0.640, 0.3065, 0.9147, 5.0570);
  valid = valid || set_dftd3_para_one(functional, "b3lyp", 1.000, 0.3981, 1.9889, 4.4211);
  valid = valid || set_dftd3_para_one(functional, "b3pw91", 1.000, 0.4312, 2.8524, 4.4693);
  valid = valid || set_dftd3_para_one(functional, "b97d", 1.000, 0.5545, 2.2609, 3.2297);
  valid = valid || set_dftd3_para_one(functional, "bhlyp", 1.000, 0.2793, 1.0354, 4.9615);
  valid = valid || set_dftd3_para_one(functional, "blyp", 1.000, 0.4298, 2.6996, 4.2359);
  valid = valid || set_dftd3_para_one(functional, "bmk", 1.000, 0.1940, 2.0860, 5.9197);
  valid = valid || set_dftd3_para_one(functional, "bop", 1.000, 0.4870, 3.295, 3.5043);
  valid = valid || set_dftd3_para_one(functional, "bp86", 1.000, 0.3946, 3.2822, 4.8516);
  valid = valid || set_dftd3_para_one(functional, "bpbe", 1.000, 0.4567, 4.0728, 4.3908);
  valid = valid || set_dftd3_para_one(functional, "camb3lyp", 1.000, 0.3708, 2.0674, 5.4743);
  valid = valid || set_dftd3_para_one(functional, "dsdblyp", 0.500, 0.0000, 0.2130, 6.0519);
  valid = valid || set_dftd3_para_one(functional, "hcth120", 1.000, 0.3563, 1.0821, 4.3359);
  valid = valid || set_dftd3_para_one(functional, "hf", 1.000, 0.3385, 0.9171, 2.883);
  valid = valid || set_dftd3_para_one(functional, "hse-hjs", 1.000, 0.3830, 2.3100, 5.685);
  valid = valid || set_dftd3_para_one(functional, "lc-wpbe08", 1.000, 0.3919, 1.8541, 5.0897);
  valid = valid || set_dftd3_para_one(functional, "lcwpbe", 1.000, 0.3919, 1.8541, 5.0897);
  valid = valid || set_dftd3_para_one(functional, "m11", 1.000, 0.0000, 2.8112, 10.1389);
  valid = valid || set_dftd3_para_one(functional, "mn12l", 1.000, 0.0000, 2.2674, 9.1494);
  valid = valid || set_dftd3_para_one(functional, "mn12sx", 1.000, 0.0983, 1.1674, 8.0259);
  valid = valid || set_dftd3_para_one(functional, "mpw1b95", 1.000, 0.1955, 1.0508, 6.4177);
  valid = valid || set_dftd3_para_one(functional, "mpwb1k", 1.000, 0.1474, 0.9499, 6.6223);
  valid = valid || set_dftd3_para_one(functional, "mpwlyp", 1.000, 0.4831, 2.0077, 4.5323);
  valid = valid || set_dftd3_para_one(functional, "n12sx", 1.000, 0.3283, 2.4900, 5.7898);
  valid = valid || set_dftd3_para_one(functional, "olyp", 1.000, 0.5299, 2.6205, 2.8065);
  valid = valid || set_dftd3_para_one(functional, "opbe", 1.000, 0.5512, 3.3816, 2.9444);
  valid = valid || set_dftd3_para_one(functional, "otpss", 1.000, 0.4634, 2.7495, 4.3153);
  valid = valid || set_dftd3_para_one(functional, "pbe", 1.000, 0.4289, 0.7875, 4.4407);
  valid = valid || set_dftd3_para_one(functional, "pbe0", 1.000, 0.4145, 1.2177, 4.8593);
  valid = valid || set_dftd3_para_one(functional, "pbe38", 1.000, 0.3995, 1.4623, 5.1405);
  valid = valid || set_dftd3_para_one(functional, "pbesol", 1.000, 0.4466, 2.9491, 6.1742);
  valid = valid || set_dftd3_para_one(functional, "ptpss", 0.750, 0.000, 0.2804, 6.5745);
  valid = valid || set_dftd3_para_one(functional, "pw6b95", 1.000, 0.2076, 0.7257, 6.375);
  valid = valid || set_dftd3_para_one(functional, "pwb6k", 1.000, 0.1805, 0.9383, 7.7627);
  valid = valid || set_dftd3_para_one(functional, "pwpb95", 0.820, 0.0000, 0.2904, 7.3141);
  valid = valid || set_dftd3_para_one(functional, "revpbe", 1.000, 0.5238, 2.3550, 3.5016);
  valid = valid || set_dftd3_para_one(functional, "revpbe0", 1.000, 0.4679, 1.7588, 3.7619);
  valid = valid || set_dftd3_para_one(functional, "revpbe38", 1.000, 0.4309, 1.4760, 3.9446);
  valid = valid || set_dftd3_para_one(functional, "revssb", 1.000, 0.4720, 0.4389, 4.0986);
  valid = valid || set_dftd3_para_one(functional, "rpbe", 1.000, 0.1820, 0.8318, 4.0094);
  valid = valid || set_dftd3_para_one(functional, "rpw86pbe", 1.000, 0.4613, 1.3845, 4.5062);
  valid = valid || set_dftd3_para_one(functional, "scan", 1.000, 0.5380, 0.0000, 5.42);
  valid = valid || set_dftd3_para_one(functional, "sogga11x", 1.000, 0.1330, 1.1426, 5.7381);
  valid = valid || set_dftd3_para_one(functional, "ssb", 1.000, -0.0952, -0.1744, 5.2170);
  valid = valid || set_dftd3_para_one(functional, "tpss", 1.000, 0.4535, 1.9435, 4.4752);
  valid = valid || set_dftd3_para_one(functional, "tpss0", 1.000, 0.3768, 1.2576, 4.5865);
  valid = valid || set_dftd3_para_one(functional, "tpssh", 1.000, 0.4529, 2.2382, 4.6550);
  valid = valid || set_dftd3_para_one(functional, "b2kplyp", 0.64, 0.0000, 0.1521, 7.1916);
  valid = valid || set_dftd3_para_one(functional, "dsd-pbep86", 0.418, 0.0000, 0.0000, 5.6500);
  valid = valid || set_dftd3_para_one(functional, "b97m", 1.0000, -0.0780, 0.1384, 5.5946);
  valid = valid || set_dftd3_para_one(functional, "wb97x", 1.0000, 0.0000, 0.2641, 5.4959);
  valid = valid || set_dftd3_para_one(functional, "wb97m", 1.0000, 0.5660, 0.3908, 3.1280);

  if (!valid) {
    throw std::invalid_argument(
      "The " + functional +
      " functional is not supported for DFT-D3 with BJ damping.");
  }
};
