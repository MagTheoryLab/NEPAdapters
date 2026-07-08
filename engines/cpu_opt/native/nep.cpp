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
#include <stdexcept>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(NEP_ADAPTERS_CPU_OPT_USE_CBLAS)
#if defined(NEP_ADAPTERS_CPU_OPT_USE_ACCELERATE)
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
      totals.spin_contract + totals.spin_chiral + totals.spin_copy + totals.spin_gradient;
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

int spin_openmp_threads(const int N)
{
#if defined(_OPENMP)
  const int cap = N <= 1024 ? 8 : 16;
  return std::max(1, std::min(omp_get_max_threads(), cap));
#else
  (void)N;
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

#if defined(NEP_ADAPTERS_CPU_OPT_USE_CBLAS)
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
  std::vector<double>& ann_fp_group_workspace)
{
#if defined(NEP_ADAPTERS_CPU_OPT_USE_CBLAS)
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
      double fc12;
      double rc = paramb.rc_radial_pair[t12];
      double rcinv = paramb.rcinv_radial_pair[t12];
      find_fc(rc, rcinv, d12, fc12);
      double fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      const double* c_pair =
        annmb.c_radial_pair.data() + static_cast<std::size_t>(t12) *
                                      (paramb.n_max_radial + 1) *
                                      (paramb.basis_size_radial + 1);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gn12 = 0.0;
        const double* c_n = c_pair + n * (paramb.basis_size_radial + 1);
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          gn12 += fn12[k] * c_n[k];
        }
        q[n] += gn12;
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
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      double s[NUM_OF_ABC] = {0.0};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int index = i1 * N + n1;
        int n2 = g_NL_angular[index];
        int t2 = g_type[n2];
        int t12 = t1 * paramb.num_types + t2;
        double r12[3] = {g_x12_angular[index], g_y12_angular[index], g_z12_angular[index]};
        double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
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
#else
        double fc12;
        double rc = paramb.rc_angular_pair[t12];
        double rcinv = paramb.rcinv_angular_pair[t12];
        find_fc(rc, rcinv, d12, fc12);
        double fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        const double* c_pair =
          annmb.c_angular_pair.data() + static_cast<std::size_t>(t12) *
                                         (paramb.n_max_angular + 1) *
                                         (paramb.basis_size_angular + 1);
        double gn12 = 0.0;
        const double* c_n = c_pair + n * (paramb.basis_size_angular + 1);
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          gn12 += fn12[k] * c_n[k];
        }
        accumulate_s(paramb.L_max, d12, r12[0], r12[1], r12[2], gn12, s);
#endif
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
#if defined(NEP_ADAPTERS_CPU_OPT_USE_CBLAS)
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
  double* total_virial_private = nullptr;
  double* virial_private = nullptr;
};

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
  return scratch && scratch->num_threads > 1 && scratch->force_rows > 0 &&
         scratch->touched_rows && scratch->force_private && scratch->total_virial_private;
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
    scratch.total_virial_private, scratch.total_virial_private + static_cast<std::size_t>(num_threads) * 6,
    0.0);

  const bool parallel_rows = use_parallel_lammps_scratch_reduce(scratch);
#pragma omp parallel for schedule(static) if (parallel_rows)
  for (int tid = 0; tid < num_threads; ++tid) {
    double* local_force = scratch.force_private + static_cast<std::size_t>(tid) * 3 * force_rows;
    if (scratch.dense_rows) {
      std::fill(local_force, local_force + static_cast<std::size_t>(3) * force_rows, 0.0);
    } else {
      for (int row : touched_rows) {
        local_force[0 * force_rows + row] = 0.0;
        local_force[1 * force_rows + row] = 0.0;
        local_force[2 * force_rows + row] = 0.0;
      }
    }

    if (use_virial && scratch.virial_private) {
      double* local_virial = scratch.virial_private + static_cast<std::size_t>(tid) * 9 * force_rows;
      if (scratch.dense_rows) {
        std::fill(local_virial, local_virial + static_cast<std::size_t>(9) * force_rows, 0.0);
      } else {
        for (int d = 0; d < 9; ++d) {
          double* local_virial_d = local_virial + static_cast<std::size_t>(d) * force_rows;
          for (int row : touched_rows) {
            local_virial_d[row] = 0.0;
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
  double** g_virial)
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
      const std::size_t base = (static_cast<std::size_t>(tid) * 3) * force_rows + n;
      fx += scratch.force_private[base + static_cast<std::size_t>(0) * force_rows];
      fy += scratch.force_private[base + static_cast<std::size_t>(1) * force_rows];
      fz += scratch.force_private[base + static_cast<std::size_t>(2) * force_rows];
    }
    g_force[n][0] += fx;
    g_force[n][1] += fy;
    g_force[n][2] += fz;
  }

  for (int d = 0; d < 6; ++d) {
    double sum = 0.0;
    for (int tid = 0; tid < num_threads; ++tid) {
      sum += scratch.total_virial_private[static_cast<std::size_t>(tid) * 6 + d];
    }
    g_total_virial[d] += sum;
  }

  if (!g_virial || !scratch.virial_private) {
    return;
  }

#pragma omp parallel for schedule(static) if (parallel_rows)
  for (int i = 0; i < rows_to_reduce; ++i) {
    const int n = scratch.dense_rows ? i : touched_rows[i];
    for (int d = 0; d < 9; ++d) {
      double sum = 0.0;
      for (int tid = 0; tid < num_threads; ++tid) {
        sum += scratch.virial_private[
          (static_cast<std::size_t>(tid) * 9 + d) * force_rows + n];
      }
      g_virial[n][d] += sum;
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
  double* g_virial)
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
      double fc12, fcp12;
      double rc = paramb.rc_radial_pair[t12];
      double rcinv = paramb.rcinv_radial_pair[t12];
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      std::array<double, MAX_NUM_N> fn12;
      std::array<double, MAX_NUM_N> fnp12;
      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12.data(), fnp12.data());
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
        double tmp12 = fp_center[n] * gnp12 * d12inv;
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
  double* g_virial)
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
      double fc12, fcp12;
      double rc = paramb.rc_angular_pair[t12];
      double rcinv = paramb.rcinv_angular_pair[t12];
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

      std::array<double, MAX_NUM_N> fn12;
      std::array<double, MAX_NUM_N> fnp12;
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12.data(), fnp12.data());
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

void zero_total_charge(const int N, double* g_charge)
{
  double mean_charge = 0.0;
  for (int n = 0; n < N; ++n) {
    mean_charge += g_charge[n];
  }
  mean_charge /= N;
  for (int n = 0; n < N; ++n) {
    g_charge[n] -= mean_charge;
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
  std::vector<double>& ann_fp_group_workspace)
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
#if defined(NEP_ADAPTERS_CPU_OPT_USE_CBLAS)
  const bool use_batched_ann = paramb.version != 5;
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
#if defined(NEP_ADAPTERS_CPU_OPT_USE_CBLAS)
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
        scratch->total_virial_private + static_cast<std::size_t>(tid) * 6;
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
            local_force[0 * force_rows + n2] -= f12[0];
            local_force[1 * force_rows + n2] -= f12[1];
            local_force[2 * force_rows + n2] -= f12[2];

            center_virial[0] -= r12[0] * f12[0]; // xx
            center_virial[1] -= r12[1] * f12[1]; // yy
            center_virial[2] -= r12[2] * f12[2]; // zz
            center_virial[3] -= r12[0] * f12[1]; // xy
            center_virial[4] -= r12[0] * f12[2]; // xz
            center_virial[5] -= r12[1] * f12[2]; // yz
            if (local_virial) {
              local_virial[0 * force_rows + n2] -= r12[0] * f12[0]; // xx
              local_virial[1 * force_rows + n2] -= r12[1] * f12[1]; // yy
              local_virial[2 * force_rows + n2] -= r12[2] * f12[2]; // zz
              local_virial[3 * force_rows + n2] -= r12[0] * f12[1]; // xy
              local_virial[4 * force_rows + n2] -= r12[0] * f12[2]; // xz
              local_virial[5 * force_rows + n2] -= r12[1] * f12[2]; // yz
              local_virial[6 * force_rows + n2] -= r12[1] * f12[0]; // yx
              local_virial[7 * force_rows + n2] -= r12[2] * f12[0]; // zx
              local_virial[8 * force_rows + n2] -= r12[2] * f12[1]; // zy
            }
          }
          local_force[0 * force_rows + n1] += center_fx;
          local_force[1 * force_rows + n1] += center_fy;
          local_force[2 * force_rows + n1] += center_fz;
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

            local_force[0 * force_rows + n1] += f12[0];
            local_force[1 * force_rows + n1] += f12[1];
            local_force[2 * force_rows + n1] += f12[2];
            local_force[0 * force_rows + n2] -= f12[0];
            local_force[1 * force_rows + n2] -= f12[1];
            local_force[2 * force_rows + n2] -= f12[2];

            local_total_virial[0] -= r12[0] * f12[0]; // xx
            local_total_virial[1] -= r12[1] * f12[1]; // yy
            local_total_virial[2] -= r12[2] * f12[2]; // zz
            local_total_virial[3] -= r12[0] * f12[1]; // xy
            local_total_virial[4] -= r12[0] * f12[2]; // xz
            local_total_virial[5] -= r12[1] * f12[2]; // yz
            if (local_virial) {
              local_virial[0 * force_rows + n2] -= r12[0] * f12[0]; // xx
              local_virial[1 * force_rows + n2] -= r12[1] * f12[1]; // yy
              local_virial[2 * force_rows + n2] -= r12[2] * f12[2]; // zz
              local_virial[3 * force_rows + n2] -= r12[0] * f12[1]; // xy
              local_virial[4 * force_rows + n2] -= r12[0] * f12[2]; // xz
              local_virial[5 * force_rows + n2] -= r12[1] * f12[2]; // yz
              local_virial[6 * force_rows + n2] -= r12[1] * f12[0]; // yx
              local_virial[7 * force_rows + n2] -= r12[2] * f12[0]; // zx
              local_virial[8 * force_rows + n2] -= r12[2] * f12[1]; // zy
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
        scratch->total_virial_private + static_cast<std::size_t>(tid) * 6;
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
            local_force[0 * force_rows + n2] -= f12[0];
            local_force[1 * force_rows + n2] -= f12[1];
            local_force[2 * force_rows + n2] -= f12[2];

            center_virial[0] -= r12[0] * f12[0]; // xx
            center_virial[1] -= r12[1] * f12[1]; // yy
            center_virial[2] -= r12[2] * f12[2]; // zz
            center_virial[3] -= r12[0] * f12[1]; // xy
            center_virial[4] -= r12[0] * f12[2]; // xz
            center_virial[5] -= r12[1] * f12[2]; // yz
            if (local_virial) {
              local_virial[0 * force_rows + n2] -= r12[0] * f12[0]; // xx
              local_virial[1 * force_rows + n2] -= r12[1] * f12[1]; // yy
              local_virial[2 * force_rows + n2] -= r12[2] * f12[2]; // zz
              local_virial[3 * force_rows + n2] -= r12[0] * f12[1]; // xy
              local_virial[4 * force_rows + n2] -= r12[0] * f12[2]; // xz
              local_virial[5 * force_rows + n2] -= r12[1] * f12[2]; // yz
              local_virial[6 * force_rows + n2] -= r12[1] * f12[0]; // yx
              local_virial[7 * force_rows + n2] -= r12[2] * f12[0]; // zx
              local_virial[8 * force_rows + n2] -= r12[2] * f12[1]; // zy
            }
          }
          local_force[0 * force_rows + n1] += center_fx;
          local_force[1 * force_rows + n1] += center_fy;
          local_force[2 * force_rows + n1] += center_fz;
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

          local_force[0 * force_rows + n1] += f12[0];
          local_force[1 * force_rows + n1] += f12[1];
          local_force[2 * force_rows + n1] += f12[2];
          local_force[0 * force_rows + n2] -= f12[0];
          local_force[1 * force_rows + n2] -= f12[1];
          local_force[2 * force_rows + n2] -= f12[2];

          local_total_virial[0] -= r12[0] * f12[0]; // xx
          local_total_virial[1] -= r12[1] * f12[1]; // yy
          local_total_virial[2] -= r12[2] * f12[2]; // zz
          local_total_virial[3] -= r12[0] * f12[1]; // xy
          local_total_virial[4] -= r12[0] * f12[2]; // xz
          local_total_virial[5] -= r12[1] * f12[2]; // yz
          if (local_virial) {
            local_virial[0 * force_rows + n2] -= r12[0] * f12[0]; // xx
            local_virial[1 * force_rows + n2] -= r12[1] * f12[1]; // yy
            local_virial[2 * force_rows + n2] -= r12[2] * f12[2]; // zz
            local_virial[3 * force_rows + n2] -= r12[0] * f12[1]; // xy
            local_virial[4 * force_rows + n2] -= r12[0] * f12[2]; // xz
            local_virial[5 * force_rows + n2] -= r12[1] * f12[2]; // yz
            local_virial[6 * force_rows + n2] -= r12[1] * f12[0]; // yx
            local_virial[7 * force_rows + n2] -= r12[2] * f12[0]; // zx
            local_virial[8 * force_rows + n2] -= r12[2] * f12[1]; // zy
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
        scratch->total_virial_private + static_cast<std::size_t>(tid) * 6;
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
          local_force[0 * force_rows + n1] += f12[0];
          local_force[1 * force_rows + n1] += f12[1];
          local_force[2 * force_rows + n1] += f12[2];
          local_force[0 * force_rows + n2] -= f12[0];
          local_force[1 * force_rows + n2] -= f12[1];
          local_force[2 * force_rows + n2] -= f12[2];
          local_total_virial[0] -= r12[0] * f12[0]; // xx
          local_total_virial[1] -= r12[1] * f12[1]; // yy
          local_total_virial[2] -= r12[2] * f12[2]; // zz
          local_total_virial[3] -= r12[0] * f12[1]; // xy
          local_total_virial[4] -= r12[0] * f12[2]; // xz
          local_total_virial[5] -= r12[1] * f12[2]; // yz
          if (local_virial) {
            local_virial[0 * force_rows + n2] -= r12[0] * f12[0]; // xx
            local_virial[1 * force_rows + n2] -= r12[1] * f12[1]; // yy
            local_virial[2 * force_rows + n2] -= r12[2] * f12[2]; // zz
            local_virial[3 * force_rows + n2] -= r12[0] * f12[1]; // xy
            local_virial[4 * force_rows + n2] -= r12[0] * f12[2]; // xz
            local_virial[5 * force_rows + n2] -= r12[1] * f12[2]; // yz
            local_virial[6 * force_rows + n2] -= r12[1] * f12[0]; // yx
            local_virial[7 * force_rows + n2] -= r12[2] * f12[0]; // zx
            local_virial[8 * force_rows + n2] -= r12[2] * f12[1]; // zy
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
    std::cout << "Standard exception:\n";
    std::cout << "    File:          " << filename << std::endl;
    std::cout << "    Line:          " << line << std::endl;
    std::cout << "    Error message: " << e.what() << std::endl;
    exit(1);
  }
  return value;
}

double get_double_from_token(const std::string& token, const char* filename, const int line)
{
  double value = 0;
  try {
    value = std::stod(token);
  } catch (const std::exception& e) {
    std::cout << "Standard exception:\n";
    std::cout << "    File:          " << filename << std::endl;
    std::cout << "    Line:          " << line << std::endl;
    std::cout << "    Error message: " << e.what() << std::endl;
    exit(1);
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
  const double powx[5] = {1.0, x, x2, x2 * x, x2 * x2};
  const double powy[5] = {1.0, y, y2, y2 * y, y2 * y2};
  const double powz[5] = {1.0, z, z2, z2 * z, z2 * z2};
  for (int k = 0; k < kSpinDeg2Count; ++k) {
    m2[k] = powx[kSpinDeg2Exp[k][0]] * powy[kSpinDeg2Exp[k][1]] * powz[kSpinDeg2Exp[k][2]];
  }
  for (int k = 0; k < kSpinDeg3Count; ++k) {
    m3[k] = powx[kSpinDeg3Exp[k][0]] * powy[kSpinDeg3Exp[k][1]] * powz[kSpinDeg3Exp[k][2]];
  }
  for (int k = 0; k < kSpinDeg4Count; ++k) {
    m4[k] = powx[kSpinDeg4Exp[k][0]] * powy[kSpinDeg4Exp[k][1]] * powz[kSpinDeg4Exp[k][2]];
  }
}

double dot_spin_terms(const double* a, const double* b, const int count)
{
  double sum = 0.0;
  for (int k = 0; k < count; ++k) {
    sum += a[k] * b[k];
  }
  return sum;
}

double spin_packed_value(const double* packed, const int degree, const int* counts)
{
  return packed[spin_monomial_index(degree, counts[0], counts[1], counts[2])];
}

void unpack_rank3_spin_stf(const double* raw, double* out)
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

void unpack_rank4_spin_stf(const double* raw, double* out)
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

void project_rank3_spin_gradient(const double* grad, double* terms, double* derivatives)
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

void project_rank4_spin_gradient(const double* grad, double* terms, double* derivatives)
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

void rank3_stf_edge(const std::array<double, 3>& u, double* out)
{
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 3; ++c) {
        const double trace =
          ((a == b) ? u[c] : 0.0) +
          ((a == c) ? u[b] : 0.0) +
          ((b == c) ? u[a] : 0.0);
        out[(a * 3 + b) * 3 + c] = u[a] * u[b] * u[c] - trace / 5.0;
      }
    }
  }
}

void rank4_stf_edge(const std::array<double, 3>& u, double* out)
{
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < 3; ++d) {
          const double six =
            ((a == b) ? u[c] * u[d] : 0.0) +
            ((a == c) ? u[b] * u[d] : 0.0) +
            ((a == d) ? u[b] * u[c] : 0.0) +
            ((b == c) ? u[a] * u[d] : 0.0) +
            ((b == d) ? u[a] * u[c] : 0.0) +
            ((c == d) ? u[a] * u[b] : 0.0);
          const double three =
            ((a == b && c == d) ? 1.0 : 0.0) +
            ((a == c && b == d) ? 1.0 : 0.0) +
            ((a == d && b == c) ? 1.0 : 0.0);
          out[((a * 3 + b) * 3 + c) * 3 + d] =
            u[a] * u[b] * u[c] * u[d] - six / 7.0 + three / 35.0;
        }
      }
    }
  }
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

void add_density(
  std::vector<double>& density,
  const int C,
  const int atom,
  const double* values,
  const int width,
  const double* weights,
  const double weight_scale = 1.0)
{
  for (int c = 0; c < C; ++c) {
    double* out = density.data() + (static_cast<std::size_t>(atom) * C + c) * width;
    const double weight = weight_scale * weights[c];
    for (int k = 0; k < width; ++k) {
      out[k] += weight * values[k];
    }
  }
}

struct SpinEdge {
  int i;
  int j;
  int t12;
  double dist;
  std::array<double, 3> rhat;
  std::array<double, 3> si;
  std::array<double, 3> sj;
  std::array<double, MAX_NUM_N> weights;
  std::array<double, MAX_NUM_N> weight_derivatives;
  double dot;
  double sj2;
  double ri_dot_si;
  double ri_dot_sj;
  double bond_axis;
};

struct SpinCache {
  std::vector<SpinEdge> edges;
  std::vector<double> rho0;
  std::vector<double> raw1;
  std::vector<double> l1_rdot;
  std::vector<double> l1_cross;
  std::vector<double> l1_stf;
  std::vector<double> angular2;
  std::vector<double> angular3;
  std::vector<double> angular4;
  std::vector<double> geom;
  std::vector<double> polars;
  std::vector<double> octupoles;
  std::vector<double> hexadecapoles;
  std::vector<double> chirals;
  std::vector<double> pseudodevs;
  std::vector<double> rho0_dot;
  std::vector<double> raw1_dot;
};

struct SpinPhaseBreakdown {
  double setup = 0.0;
  double edges = 0.0;
  double unpack = 0.0;
  double merge = 0.0;
  double contract = 0.0;
  double chiral = 0.0;
  double copy = 0.0;
  double gradient_nonchiral = 0.0;
  double gradient_chiral = 0.0;
};

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
  SpinPhaseBreakdown* phase = nullptr)
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
  std::vector<double> q(static_cast<std::size_t>(N) * paramb.spin_dim, 0.0);
  SpinCache local_cache;
  SpinCache& cache = cache_out ? *cache_out : local_cache;
  cache = SpinCache{};

  auto qref = [&](const int atom, const int dim) -> double& {
    return q[static_cast<std::size_t>(atom) * paramb.spin_dim + dim];
  };
  auto spin = [&](const int atom, const int component) -> double {
    return spins[static_cast<std::size_t>(component) * N + atom];
  };
  auto active = [&](const std::vector<int>& mask, const int t) -> bool {
    return mask.empty() || mask[static_cast<std::size_t>(t)] != 0;
  };

  for (int atom = 0; atom < N; ++atom) {
    const bool dof = active(paramb.spin_dof_type_active, type[atom]);
    const double sx = spin(atom, 0);
    const double sy = spin(atom, 1);
    const double sz = spin(atom, 2);
    const double s2 = sx * sx + sy * sy + sz * sz;
    qref(atom, 0) = dof ? s2 : 0.0;
    qref(atom, 1) = dof ? s2 * s2 : 0.0;
  }

  cache.rho0.assign(static_cast<std::size_t>(N) * C * 3, 0.0);
  cache.raw1.assign(static_cast<std::size_t>(N) * C * 9, 0.0);
  cache.l1_rdot.assign(static_cast<std::size_t>(N) * C, 0.0);
  cache.l1_cross.assign(static_cast<std::size_t>(N) * C * 3, 0.0);
  cache.l1_stf.assign(static_cast<std::size_t>(N) * C * 9, 0.0);
  cache.angular2.assign(static_cast<std::size_t>(N) * C * 15, 0.0);
  cache.angular3.assign(static_cast<std::size_t>(N) * C * 21, 0.0);
  cache.angular4.assign(static_cast<std::size_t>(N) * C * 27, 0.0);
  cache.geom.assign(static_cast<std::size_t>(N) * C * 9, 0.0);
  cache.rho0_dot.assign(static_cast<std::size_t>(N) * C * 3, 0.0);
  cache.raw1_dot.assign(static_cast<std::size_t>(N) * C * 9, 0.0);
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
    polars.assign(static_cast<std::size_t>(N) * C * 3, 0.0);
    octupoles.assign(static_cast<std::size_t>(N) * chiC * 27, 0.0);
    hexadecapoles.assign(static_cast<std::size_t>(N) * chiC * 81, 0.0);
    octupoles_raw.assign(static_cast<std::size_t>(N) * chiC * kSpinDeg3Count, 0.0);
    hexadecapoles_raw.assign(static_cast<std::size_t>(N) * chiC * kSpinDeg4Count, 0.0);
    chirals.assign(static_cast<std::size_t>(N) * chiC, 0.0);
    pseudodevs.assign(static_cast<std::size_t>(N) * C * 9, 0.0);
  }

#if defined(_OPENMP)
  const int num_threads = spin_openmp_threads(N);
#else
  const int num_threads = 1;
#endif
  const bool use_parallel_edges = num_threads > 1 && N > 8;
  const bool keep_edges = cache_out || paramb.spin_chiral;
  std::vector<std::vector<SpinEdge>> private_edges(
    keep_edges && use_parallel_edges ? static_cast<std::size_t>(num_threads) : 0);
  if (keep_edges && !use_parallel_edges) {
    cache.edges.reserve(static_cast<std::size_t>(N) * 8);
  }
  add_phase(&SpinPhaseBreakdown::setup);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_edges)
#endif
  for (int i = 0; i < N; ++i) {
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    for (int n = 0; n < NN[i]; ++n) {
      const int index = n * N + i;
      const int j = NL[index];
      const double dx = x12[index];
      const double dy = y12[index];
      const double dz = z12[index];
      const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (d <= 1.0e-12 || d >= paramb.spin_cutoff_radial) {
        continue;
      }
      if (!active(paramb.spin_dof_type_active, type[i]) ||
          !active(paramb.spin_env_type_active, type[j])) {
        continue;
      }
      double fc = 0.0;
      double fcp = 0.0;
      double fn[MAX_NUM_N];
      double fnp[MAX_NUM_N];
      const double rcinv = 1.0 / paramb.spin_cutoff_radial;
      if (cache_out) {
        find_fc_and_fcp(paramb.spin_cutoff_radial, rcinv, d, fc, fcp);
        find_fn_and_fnp(paramb.spin_basis_size, rcinv, d, fc, fcp, fn, fnp);
      } else {
        find_fc(paramb.spin_cutoff_radial, rcinv, d, fc);
        find_fn(paramb.spin_basis_size, rcinv, d, fc, fn);
      }
      SpinEdge edge;
      edge.i = i;
      edge.j = j;
      edge.t12 = type[i] * paramb.num_types + type[j];
      edge.dist = d;
      edge.rhat = {dx / d, dy / d, dz / d};
      edge.si = {spin(i, 0), spin(i, 1), spin(i, 2)};
      edge.sj = {spin(j, 0), spin(j, 1), spin(j, 2)};
      edge.weights.fill(0.0);
      edge.weight_derivatives.fill(0.0);
      for (int c = 0; c < C; ++c) {
        double w = 0.0;
        double dw = 0.0;
        for (int k = 0; k < B; ++k) {
          const std::size_t idx =
            ((static_cast<std::size_t>(c) * B + k) * paramb.num_types_sq) + edge.t12;
          w += fn[k] * annmb.c_spin[idx];
          if (cache_out) {
            dw += fnp[k] * annmb.c_spin[idx];
          }
        }
        edge.weights[c] = w;
        edge.weight_derivatives[c] = dw;
      }
      edge.dot = edge.si[0] * edge.sj[0] + edge.si[1] * edge.sj[1] + edge.si[2] * edge.sj[2];
      edge.sj2 = edge.sj[0] * edge.sj[0] + edge.sj[1] * edge.sj[1] + edge.sj[2] * edge.sj[2];
      edge.ri_dot_si =
        edge.rhat[0] * edge.si[0] + edge.rhat[1] * edge.si[1] + edge.rhat[2] * edge.si[2];
      edge.ri_dot_sj =
        edge.rhat[0] * edge.sj[0] + edge.rhat[1] * edge.sj[1] + edge.rhat[2] * edge.sj[2];
      edge.bond_axis = edge.ri_dot_si * edge.ri_dot_sj;

      int scalar_offset = 2;
      const double scalars[4] = {
        edge.dot, edge.dot * edge.dot, edge.sj2, edge.bond_axis};
      for (int term = 0; term < 4; ++term) {
        for (int c = 0; c < C; ++c) {
          qref(edge.i, scalar_offset + c) += edge.weights[c] * scalars[term];
        }
        scalar_offset += C;
      }

      const double sj_value[3] = {edge.sj[0], edge.sj[1], edge.sj[2]};
      add_density(rho0, C, edge.i, sj_value, 3, edge.weights.data());
      double raw1_value[9];
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          raw1_value[3 * a + b] = edge.rhat[a] * edge.sj[b];
        }
      }
      add_density(raw1, C, edge.i, raw1_value, 9, edge.weights.data());
      if (l_max >= 1) {
        const double rdot =
          edge.rhat[0] * edge.sj[0] + edge.rhat[1] * edge.sj[1] + edge.rhat[2] * edge.sj[2];
        add_density(l1_rdot, C, edge.i, &rdot, 1, edge.weights.data());
        const double cross_value[3] = {
          edge.rhat[1] * edge.sj[2] - edge.rhat[2] * edge.sj[1],
          edge.rhat[2] * edge.sj[0] - edge.rhat[0] * edge.sj[2],
          edge.rhat[0] * edge.sj[1] - edge.rhat[1] * edge.sj[0]};
        add_density(l1_cross, C, edge.i, cross_value, 3, edge.weights.data());
        const auto stf = stf_outer3(edge.rhat, edge.sj);
        add_density(l1_stf, C, edge.i, stf.data(), 9, edge.weights.data());
      }
      for (int ell = 2; ell <= l_max; ++ell) {
        double ylm[9];
        const int ylm_width = real_spherical_harmonics_spin(edge.rhat, ell, ylm);
        double value[27];
        int width = 0;
        for (int m = 0; m < ylm_width; ++m) {
          const double y = ylm[m];
          value[width++] = y * edge.sj[0];
          value[width++] = y * edge.sj[1];
          value[width++] = y * edge.sj[2];
        }
        add_density(
          ell == 2 ? angular2 : ell == 3 ? angular3 : angular4,
          C, edge.i, value, width, edge.weights.data());
      }
      const auto rr = stf_outer3(edge.rhat, edge.rhat);
      add_density(geom, C, edge.i, rr.data(), 9, edge.weights.data());
      if (paramb.spin_chiral) {
        add_density(polars, C, edge.i, edge.rhat.data(), 3, edge.weights.data());
        double m2[kSpinDeg2Count];
        double m3[kSpinDeg3Count];
        double m4[kSpinDeg4Count];
        fill_spin_monomials(edge.rhat, m2, m3, m4);
        add_density(octupoles_raw, chiC, edge.i, m3, kSpinDeg3Count, edge.weights.data());
        add_density(hexadecapoles_raw, chiC, edge.i, m4, kSpinDeg4Count, edge.weights.data());
      }
      add_density(rho0_dot, C, edge.i, sj_value, 3, edge.weights.data(), edge.dot);
      add_density(raw1_dot, C, edge.i, raw1_value, 9, edge.weights.data(), edge.dot);
      if (keep_edges) {
        if (use_parallel_edges) {
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
    for (int atom = 0; atom < N; ++atom) {
      for (int c = 0; c < chiC; ++c) {
        const std::size_t raw_base = static_cast<std::size_t>(atom) * chiC + c;
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

  if (keep_edges && use_parallel_edges) {
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
    for (int atom = 0; atom < N; ++atom) {
      for (int c = 0; c < C; ++c) {
        const double* av = a.data() + (static_cast<std::size_t>(atom) * C + c) * width;
        const double* bv = b.data() + (static_cast<std::size_t>(atom) * C + c) * width;
        double sum = 0.0;
        for (int k = 0; k < width; ++k) {
          sum += av[k] * bv[k];
        }
        qref(atom, offset + c) = sum;
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
  for (int atom = 0; atom < N; ++atom) {
    const std::array<double, 3> s = {spin(atom, 0), spin(atom, 1), spin(atom, 2)};
    for (int c = 0; c < C; ++c) {
      const double* g = geom.data() + (static_cast<std::size_t>(atom) * C + c) * 9;
      double value = 0.0;
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          value += s[a] * g[3 * a + b] * s[b];
        }
      }
      qref(atom, offset + c) = value;
    }
  }
  offset += C;
  contract(rho0, rho0_dot, 3);
  if (l_max >= 1) {
    contract(raw1, raw1_dot, 9);
  }
  add_phase(&SpinPhaseBreakdown::contract);

  if (paramb.spin_chiral) {
    auto block = [&](const std::vector<double>& v, const int atom, const int c, const int width) {
      return v.data() + (static_cast<std::size_t>(atom) * C + c) * width;
    };
    auto chi_block = [&](const std::vector<double>& v, const int atom, const int c, const int width) {
      return v.data() + (static_cast<std::size_t>(atom) * chiC + c) * width;
    };
    for (int atom = 0; atom < N; ++atom) {
      for (int c = 0; c < chiC; ++c) {
        const double* Q = block(geom, atom, c, 9);
        const double* O = chi_block(octupoles, atom, c, 27);
        const double* H = chi_block(hexadecapoles, atom, c, 81);
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
        chirals[static_cast<std::size_t>(atom) * chiC + c] = value;
      }
    }
    for (const SpinEdge& edge : cache.edges) {
      for (int c = 0; c < C; ++c) {
        const double* Q = block(geom, edge.i, c, 9);
        std::array<double, 3> Qu = {0.0, 0.0, 0.0};
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            Qu[a] += Q[3 * a + b] * edge.rhat[b];
          }
        }
        const std::array<double, 3> axis = cross3(edge.rhat, Qu);
        const auto pseudo = stf_outer3(axis, edge.rhat);
        double* out = pseudodevs.data() + (static_cast<std::size_t>(edge.i) * C + c) * 9;
        for (int k = 0; k < 9; ++k) {
          out[k] += edge.weights[c] * pseudo[k];
        }
      }
    }
    for (const SpinEdge& edge : cache.edges) {
      const std::array<double, 3> spin_cross = cross3(edge.si, edge.sj);
      for (int c = 0; c < chiC; ++c) {
        qref(edge.i, offset + c) += edge.weights[c] * dot3(spin_cross, edge.rhat) *
                                    chirals[static_cast<std::size_t>(edge.i) * chiC + c];
      }
      int chiral_offset = offset + chiC;
      for (int c = 0; c < C; ++c) {
        const double* p = block(polars, edge.i, c, 3);
        const std::array<double, 3> polar = {p[0], p[1], p[2]};
        const std::array<double, 3> axis = cross3(polar, edge.rhat);
        qref(edge.i, chiral_offset + c) += edge.weights[c] * dot3(spin_cross, axis);
      }
      chiral_offset += C;
      for (int c = 0; c < C; ++c) {
        const double* P = block(pseudodevs, edge.i, c, 9);
        std::array<double, 3> axis = {0.0, 0.0, 0.0};
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            axis[a] += P[3 * a + b] * edge.rhat[b];
          }
        }
        qref(edge.i, chiral_offset + c) += edge.weights[c] * dot3(spin_cross, axis);
      }
    }
  }
  add_phase(&SpinPhaseBreakdown::chiral);

  for (int atom = 0; atom < N; ++atom) {
    for (int d = 0; d < paramb.spin_dim; ++d) {
      descriptor_soa[(offset0 + d) * N + atom] =
        qref(atom, d) * paramb.q_scaler[offset0 + d];
    }
  }
  add_phase(&SpinPhaseBreakdown::copy);
}

void add_stf_outer_gradient(
  const double* grad,
  const std::array<double, 3>& a,
  const std::array<double, 3>& b,
  std::array<double, 3>& grad_a,
  std::array<double, 3>& grad_b)
{
  const double trace_grad = (grad[0] + grad[4] + grad[8]) / 3.0;
  for (int p = 0; p < 3; ++p) {
    double ga = -trace_grad * b[p];
    double gb = -trace_grad * a[p];
    for (int q = 0; q < 3; ++q) {
      ga += 0.5 * (grad[3 * p + q] + grad[3 * q + p]) * b[q];
      gb += 0.5 * (grad[3 * p + q] + grad[3 * q + p]) * a[q];
    }
    grad_a[p] += ga;
    grad_b[p] += gb;
  }
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
  const int* type,
  const SpinCache& cache,
  const double* Fp,
  std::vector<double>& grad_spin,
  double* force,
  double* virial)
{
  if (!paramb.spin_chiral || cache.edges.empty()) {
    return;
  }
  const int C = paramb.spin_compress;
  const int chiC = std::min(2, C);
  const int base_offset = spin_descriptor_dim(C, paramb.spin_l_max, false);
  const int offset0 = paramb.struct_dim;
  const std::size_t edge_count = cache.edges.size();
  std::vector<double> grad_weight(edge_count * C, 0.0);
  std::vector<double> grad_rhat(edge_count * 3, 0.0);
  std::vector<double> grad_si(edge_count * 3, 0.0);
  std::vector<double> grad_sj(edge_count * 3, 0.0);
  std::vector<double> grad_Q(static_cast<std::size_t>(N) * C * 9, 0.0);
  std::vector<double> grad_O(static_cast<std::size_t>(N) * chiC * 27, 0.0);
  std::vector<double> grad_H(static_cast<std::size_t>(N) * chiC * 81, 0.0);
  std::vector<double> grad_chi(static_cast<std::size_t>(N) * chiC, 0.0);
  std::vector<double> grad_polar(static_cast<std::size_t>(N) * C * 3, 0.0);
  std::vector<double> grad_pseudodev(static_cast<std::size_t>(N) * C * 9, 0.0);

#if defined(_OPENMP)
  const int num_threads = spin_openmp_threads(N);
#else
  const int num_threads = 1;
#endif
  const bool use_parallel_edges = num_threads > 1 && edge_count > 32;

  auto fp = [&](const int atom, const int dim) {
    return Fp[static_cast<std::size_t>(atom) * annmb.dim + offset0 + dim];
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

  std::vector<double> grad_chi_private(
    use_parallel_edges ? static_cast<std::size_t>(num_threads) * grad_chi.size() : 0, 0.0);
  std::vector<double> grad_polar_private(
    use_parallel_edges ? static_cast<std::size_t>(num_threads) * grad_polar.size() : 0, 0.0);
  std::vector<double> grad_pseudodev_private(
    use_parallel_edges ? static_cast<std::size_t>(num_threads) * grad_pseudodev.size() : 0, 0.0);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_edges)
#endif
  for (std::ptrdiff_t edge_index = 0; edge_index < static_cast<std::ptrdiff_t>(edge_count); ++edge_index) {
    const std::size_t e = static_cast<std::size_t>(edge_index);
    const SpinEdge& edge = cache.edges[e];
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
    const std::array<double, 3> x = cross3(edge.si, edge.sj);
    std::array<double, 3> gx = {0.0, 0.0, 0.0};
    std::array<double, 3> gu = {0.0, 0.0, 0.0};
    for (int c = 0; c < chiC; ++c) {
      const double alpha = fp(edge.i, base_offset + c);
      const double chi = cache.chirals[static_cast<std::size_t>(edge.i) * chiC + c];
      const double xu = dot3(x, edge.rhat);
      egw(e, c) += alpha * xu * chi;
      local_grad_chi[static_cast<std::size_t>(edge.i) * chiC + c] +=
        alpha * edge.weights[c] * xu;
      for (int d = 0; d < 3; ++d) {
        gx[d] += alpha * edge.weights[c] * chi * edge.rhat[d];
        gu[d] += alpha * edge.weights[c] * chi * x[d];
      }
    }
    int chiral_offset = base_offset + chiC;
    for (int c = 0; c < C; ++c) {
      const double alpha = fp(edge.i, chiral_offset + c);
      const double* p = cblockC(cache.polars, edge.i, c, 3);
      const std::array<double, 3> polar = {p[0], p[1], p[2]};
      const std::array<double, 3> axis = cross3(polar, edge.rhat);
      const double xa = dot3(x, axis);
      egw(e, c) += alpha * xa;
      std::array<double, 3> gaxis = {0.0, 0.0, 0.0};
      for (int d = 0; d < 3; ++d) {
        gx[d] += alpha * edge.weights[c] * axis[d];
        gaxis[d] = alpha * edge.weights[c] * x[d];
      }
      const std::array<double, 3> gp = cross3(edge.rhat, gaxis);
      const std::array<double, 3> gu_part = cross3(gaxis, polar);
      double* grad_p = local_grad_polar + (static_cast<std::size_t>(edge.i) * C + c) * 3;
      for (int d = 0; d < 3; ++d) {
        grad_p[d] += gp[d];
        gu[d] += gu_part[d];
      }
    }
    chiral_offset += C;
    for (int c = 0; c < C; ++c) {
      const double alpha = fp(edge.i, chiral_offset + c);
      const double* P = cblockC(cache.pseudodevs, edge.i, c, 9);
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
        gx[d] += alpha * edge.weights[c] * axis[d];
        gaxis[d] = alpha * edge.weights[c] * x[d];
      }
      double* gP = local_grad_pseudodev + (static_cast<std::size_t>(edge.i) * C + c) * 9;
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          gP[3 * a + b] += gaxis[a] * edge.rhat[b];
          gu[b] += P[3 * a + b] * gaxis[a];
        }
      }
    }
    const std::array<double, 3> gsi = cross3(edge.sj, gx);
    const std::array<double, 3> gsj = cross3(gx, edge.si);
    add_edge_vec(grad_si, e, gsi);
    add_edge_vec(grad_sj, e, gsj);
    add_edge_vec(grad_rhat, e, gu);
  }
  reduce_private(grad_chi, grad_chi_private);
  reduce_private(grad_polar, grad_polar_private);
  reduce_private(grad_pseudodev, grad_pseudodev_private);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_edges)
#endif
  for (int atom = 0; atom < N; ++atom) {
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

  std::vector<double> grad_Q_private(
    use_parallel_edges ? static_cast<std::size_t>(num_threads) * grad_Q.size() : 0, 0.0);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_edges)
#endif
  for (std::ptrdiff_t edge_index = 0; edge_index < static_cast<std::ptrdiff_t>(edge_count); ++edge_index) {
    const std::size_t e = static_cast<std::size_t>(edge_index);
    const SpinEdge& edge = cache.edges[e];
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    double* local_grad_Q = use_parallel_edges
      ? grad_Q_private.data() + static_cast<std::size_t>(tid) * grad_Q.size()
      : grad_Q.data();
    for (int c = 0; c < C; ++c) {
      const double* gp = cblockC(grad_polar, edge.i, c, 3);
      egw(e, c) += gp[0] * edge.rhat[0] + gp[1] * edge.rhat[1] + gp[2] * edge.rhat[2];
      for (int d = 0; d < 3; ++d) {
        eg3(grad_rhat, e, d) += edge.weights[c] * gp[d];
      }

      const double* gPd = cblockC(grad_pseudodev, edge.i, c, 9);
      const double* Q = cblockC(cache.geom, edge.i, c, 9);
      std::array<double, 3> Qu = {0.0, 0.0, 0.0};
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          Qu[a] += Q[3 * a + b] * edge.rhat[b];
        }
      }
      const std::array<double, 3> pseudo_axis = cross3(edge.rhat, Qu);
      const auto pseudo = stf_outer3(pseudo_axis, edge.rhat);
      double dot = 0.0;
      double gpseudo[9] = {0.0};
      for (int k = 0; k < 9; ++k) {
        dot += gPd[k] * pseudo[k];
        gpseudo[k] = gPd[k] * edge.weights[c];
      }
      egw(e, c) += dot;
      std::array<double, 3> g_axis = {0.0, 0.0, 0.0};
      std::array<double, 3> gu2 = {0.0, 0.0, 0.0};
      add_stf_outer_gradient(gpseudo, pseudo_axis, edge.rhat, g_axis, gu2);
      const std::array<double, 3> g_u_cross = cross3(Qu, g_axis);
      const std::array<double, 3> g_Qu = cross3(g_axis, edge.rhat);
      double* gQ = local_grad_Q + (static_cast<std::size_t>(edge.i) * C + c) * 9;
      for (int a = 0; a < 3; ++a) {
        eg3(grad_rhat, e, a) += gu2[a] + g_u_cross[a];
        for (int b = 0; b < 3; ++b) {
          gQ[3 * a + b] += g_Qu[a] * edge.rhat[b];
          eg3(grad_rhat, e, b) += Q[3 * a + b] * g_Qu[a];
        }
      }
    }
  }
  reduce_private(grad_Q, grad_Q_private);

  std::vector<double> grad_Q_terms(static_cast<std::size_t>(N) * C * kSpinDeg2Count, 0.0);
  std::vector<double> grad_O_terms(static_cast<std::size_t>(N) * chiC * kSpinDeg3Count, 0.0);
  std::vector<double> grad_O_derivatives(
    static_cast<std::size_t>(N) * chiC * 3 * kSpinDeg2Count, 0.0);
  std::vector<double> grad_H_terms(static_cast<std::size_t>(N) * chiC * kSpinDeg4Count, 0.0);
  std::vector<double> grad_H_derivatives(
    static_cast<std::size_t>(N) * chiC * 3 * kSpinDeg3Count, 0.0);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_edges)
#endif
  for (int atom = 0; atom < N; ++atom) {
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
      const double* q_terms = cblockC(grad_Q_terms, edge.i, c, kSpinDeg2Count);
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
      const double* o_terms = cblockChi(grad_O_terms, edge.i, c, kSpinDeg3Count);
      const double* o_derivatives =
        cblockChi(grad_O_derivatives, edge.i, c, 3 * kSpinDeg2Count);
      const double* h_terms = cblockChi(grad_H_terms, edge.i, c, kSpinDeg4Count);
      const double* h_derivatives =
        cblockChi(grad_H_derivatives, edge.i, c, 3 * kSpinDeg3Count);
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

  std::vector<double> force_private(
    use_parallel_edges ? static_cast<std::size_t>(num_threads) * 3 * N : 0, 0.0);
  std::vector<double> grad_spin_private(
    use_parallel_edges ? static_cast<std::size_t>(num_threads) * 3 * N : 0, 0.0);
  std::vector<double> virial_private(
    use_parallel_edges ? static_cast<std::size_t>(num_threads) * 9 : 0, 0.0);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_edges)
#endif
  for (std::ptrdiff_t edge_index = 0; edge_index < static_cast<std::ptrdiff_t>(edge_count); ++edge_index) {
    const std::size_t e = static_cast<std::size_t>(edge_index);
    const SpinEdge& edge = cache.edges[e];
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    double* local_force = use_parallel_edges
      ? force_private.data() + static_cast<std::size_t>(tid) * 3 * N
      : force;
    double* local_grad_spin = use_parallel_edges
      ? grad_spin_private.data() + static_cast<std::size_t>(tid) * 3 * N
      : grad_spin.data();
    double* local_virial = use_parallel_edges
      ? virial_private.data() + static_cast<std::size_t>(tid) * 9
      : nullptr;
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
      local_force[static_cast<std::size_t>(d) * N + edge.i] += grad_rij[d];
      local_force[static_cast<std::size_t>(d) * N + edge.j] -= grad_rij[d];
      local_grad_spin[static_cast<std::size_t>(d) * N + edge.i] += grad_si[e * 3 + d];
      local_grad_spin[static_cast<std::size_t>(d) * N + edge.j] += grad_sj[e * 3 + d];
    }
    for (int a = 0; a < 3; ++a) {
      const double rij_a = edge.rhat[a] * edge.dist;
      for (int b = 0; b < 3; ++b) {
        if (use_parallel_edges) {
          local_virial[a * 3 + b] -= rij_a * grad_rij[b];
        } else {
          virial[static_cast<std::size_t>(a * 3 + b) * N] -= rij_a * grad_rij[b];
        }
      }
    }
  }

  if (use_parallel_edges) {
    for (int tid = 0; tid < num_threads; ++tid) {
      const double* local_force = force_private.data() + static_cast<std::size_t>(tid) * 3 * N;
      const double* local_grad_spin =
        grad_spin_private.data() + static_cast<std::size_t>(tid) * 3 * N;
      const double* local_virial = virial_private.data() + static_cast<std::size_t>(tid) * 9;
      for (int atom = 0; atom < N; ++atom) {
        for (int d = 0; d < 3; ++d) {
          const std::size_t idx = static_cast<std::size_t>(d) * N + atom;
          force[idx] += local_force[idx];
          grad_spin[idx] += local_grad_spin[idx];
        }
      }
      for (int d = 0; d < 9; ++d) {
        virial[static_cast<std::size_t>(d) * N] += local_virial[d];
      }
    }
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
  SpinPhaseBreakdown* phase = nullptr)
{
  auto phase_mark = NepPhaseClock::now();
  const int C = paramb.spin_compress;
  const int l_max = paramb.spin_l_max;
  const int offset0 = paramb.struct_dim;
  std::vector<double> grad_spin(static_cast<std::size_t>(N) * 3, 0.0);
  auto fp = [&](const int atom, const int dim) {
    return Fp[static_cast<std::size_t>(atom) * annmb.dim + offset0 + dim];
  };
  auto spin = [&](const int atom, const int d) {
    return spins[static_cast<std::size_t>(d) * N + atom];
  };
  auto active = [&](const std::vector<int>& mask, const int t) {
    return mask.empty() || mask[static_cast<std::size_t>(t)] != 0;
  };
  auto block = [&](const std::vector<double>& v, const int atom, const int c, const int width) {
    return v.data() + (static_cast<std::size_t>(atom) * C + c) * width;
  };

  for (int atom = 0; atom < N; ++atom) {
    if (!active(paramb.spin_dof_type_active, type[atom])) {
      continue;
    }
    const double sx = spin(atom, 0);
    const double sy = spin(atom, 1);
    const double sz = spin(atom, 2);
    const double s2 = sx * sx + sy * sy + sz * sz;
    const double scale = 2.0 * fp(atom, 0) + 4.0 * fp(atom, 1) * s2;
    grad_spin[atom] += scale * sx;
    grad_spin[static_cast<std::size_t>(N) + atom] += scale * sy;
    grad_spin[static_cast<std::size_t>(2) * N + atom] += scale * sz;
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
  for (int atom = 0; atom < N; ++atom) {
    if (!active(paramb.spin_dof_type_active, type[atom])) {
      continue;
    }
    const std::array<double, 3> s = {spin(atom, 0), spin(atom, 1), spin(atom, 2)};
    for (int c = 0; c < C; ++c) {
      const double alpha = fp(atom, geom_offset + c);
      const double* g = block(cache.geom, atom, c, 9);
      for (int a = 0; a < 3; ++a) {
        double gs = 0.0;
        for (int b = 0; b < 3; ++b) {
          gs += (g[3 * a + b] + g[3 * b + a]) * s[b];
        }
        grad_spin[static_cast<std::size_t>(a) * N + atom] += alpha * gs;
      }
    }
  }

  std::vector<double> rho0_pull(static_cast<std::size_t>(N) * C * 3, 0.0);
  std::vector<double> rho0_dot_pull(static_cast<std::size_t>(N) * C * 3, 0.0);
  std::vector<double> l1_pull(l_max >= 1 ? static_cast<std::size_t>(N) * C * 9 : 0, 0.0);
  std::vector<double> l1_dot_pull(l_max >= 1 ? static_cast<std::size_t>(N) * C * 9 : 0, 0.0);
  std::vector<double> geom_pull(static_cast<std::size_t>(N) * C * 9, 0.0);

  auto mutable_block = [&](std::vector<double>& v, const int atom, const int c, const int width) {
    return v.data() + (static_cast<std::size_t>(atom) * C + c) * width;
  };

  for (int atom = 0; atom < N; ++atom) {
    for (int c = 0; c < C; ++c) {
      const double alpha0 = fp(atom, rho0_offset + c);
      const double alpha0_dot = fp(atom, rho0_dot_offset + c);
      const double* rho = block(cache.rho0, atom, c, 3);
      const double* rho_dot = block(cache.rho0_dot, atom, c, 3);
      double* rho_out = mutable_block(rho0_pull, atom, c, 3);
      double* rho_dot_out = mutable_block(rho0_dot_pull, atom, c, 3);
      for (int d = 0; d < 3; ++d) {
        rho_out[d] = 2.0 * alpha0 * rho[d] + alpha0_dot * rho_dot[d];
        rho_dot_out[d] = alpha0_dot * rho[d];
      }

      const double alpha_geom = fp(atom, geom_offset + c);
      const std::array<double, 3> s = {spin(atom, 0), spin(atom, 1), spin(atom, 2)};
      const auto ss = stf_outer3(s, s);
      double* geom_out = mutable_block(geom_pull, atom, c, 9);
      for (int k = 0; k < 9; ++k) {
        geom_out[k] = alpha_geom * ss[k];
      }
    }
  }

  if (l_max >= 1) {
    for (int atom = 0; atom < N; ++atom) {
      for (int c = 0; c < C; ++c) {
        double* mat = mutable_block(l1_pull, atom, c, 9);
        double* mat_dot = mutable_block(l1_dot_pull, atom, c, 9);
        const double alpha_rdot = fp(atom, l1_rdot_offset + c);
        const double alpha_cross = fp(atom, l1_cross_offset + c);
        const double alpha_stf = fp(atom, l1_stf_offset + c);
        const double alpha_raw1 = fp(atom, raw1_dot_offset + c);
        const double rdot = *block(cache.l1_rdot, atom, c, 1);
        const double* cross = block(cache.l1_cross, atom, c, 3);
        const double* stf = block(cache.l1_stf, atom, c, 9);
        const double* raw = block(cache.raw1, atom, c, 9);
        const double* raw_dot = block(cache.raw1_dot, atom, c, 9);
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

#if defined(_OPENMP)
  const int num_threads = spin_openmp_threads(N);
#else
  const int num_threads = 1;
#endif
  const bool use_parallel_edges = num_threads > 1 && cache.edges.size() > 32;
  std::vector<double> force_private(
    use_parallel_edges ? static_cast<std::size_t>(num_threads) * 3 * N : 0, 0.0);
  std::vector<double> grad_spin_private(
    use_parallel_edges ? static_cast<std::size_t>(num_threads) * 3 * N : 0, 0.0);
  std::vector<double> virial_private(
    use_parallel_edges ? static_cast<std::size_t>(num_threads) * 9 : 0, 0.0);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(num_threads) if (use_parallel_edges)
#endif
  for (std::ptrdiff_t edge_index = 0; edge_index < static_cast<std::ptrdiff_t>(cache.edges.size()); ++edge_index) {
    const SpinEdge& edge = cache.edges[static_cast<std::size_t>(edge_index)];
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    double* local_force = use_parallel_edges
      ? force_private.data() + static_cast<std::size_t>(tid) * 3 * N
      : force;
    double* local_grad_spin = use_parallel_edges
      ? grad_spin_private.data() + static_cast<std::size_t>(tid) * 3 * N
      : grad_spin.data();
    double* local_virial = use_parallel_edges
      ? virial_private.data() + static_cast<std::size_t>(tid) * 9
      : nullptr;
    auto add_local_grad_spin = [&](const int atom, const std::array<double, 3>& g) {
      for (int d = 0; d < 3; ++d) {
        local_grad_spin[static_cast<std::size_t>(d) * N + atom] += g[d];
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
      grad_sj[d] += 2.0 * g_sj2 * edge.sj[d];
    }
    offset += C;
    const double g_axis = add_scalar(offset, edge.bond_axis);
    for (int d = 0; d < 3; ++d) {
      grad_si[d] += g_axis * edge.ri_dot_sj * edge.rhat[d];
      grad_sj[d] += g_axis * edge.ri_dot_si * edge.rhat[d];
      grad_rhat[d] += g_axis * (edge.ri_dot_sj * edge.si[d] + edge.ri_dot_si * edge.sj[d]);
    }
    offset += C;

    auto add_density_gradient = [&](const int width, const std::vector<double>& density,
                           const double* value,
                           const double mod_value,
                           const bool has_mod,
                           const int q_offset,
                           const std::vector<double>* other_density,
                           double* grad_value,
                           double& grad_mod) {
      for (int c = 0; c < C; ++c) {
        const double* self = block(density, edge.i, c, width);
        const double* other = other_density ? block(*other_density, edge.i, c, width) : nullptr;
        const double alpha = fp(edge.i, q_offset + c);
        const double m = has_mod ? mod_value : 1.0;
        double grad_weight_c = grad_weight[c];
        for (int k = 0; k < width; ++k) {
          const double gd = other ? alpha * other[k] : 2.0 * alpha * self[k];
          grad_weight_c += gd * value[k] * m;
          grad_value[k] += gd * edge.weights[c] * m;
          if (has_mod) {
            grad_mod += gd * edge.weights[c] * value[k];
          }
        }
        grad_weight[c] = grad_weight_c;
      }
    };

    auto apply_angular = [&](const int ell, const double* ylm, const int ylm_width, const double* ge) {
      double grad_ylm[9] = {0.0};
      for (int m = 0; m < ylm_width; ++m) {
        for (int d = 0; d < 3; ++d) {
          const double g = ge[m * 3 + d];
          grad_sj[d] += g * ylm[m];
          grad_ylm[m] += g * edge.sj[d];
        }
      }
      add_real_spherical_harmonics_gradient(edge.rhat, ell, grad_ylm, grad_rhat);
    };

    for (int c = 0; c < C; ++c) {
      const double* b = block(rho0_pull, edge.i, c, 3);
      const double* b_dot = block(rho0_dot_pull, edge.i, c, 3);
      const double u0 = b[0] + edge.dot * b_dot[0];
      const double u1 = b[1] + edge.dot * b_dot[1];
      const double u2 = b[2] + edge.dot * b_dot[2];
      const double w = edge.weights[c];
      grad_weight[c] += edge.sj[0] * u0 + edge.sj[1] * u1 + edge.sj[2] * u2;
      grad_sj[0] += w * u0;
      grad_sj[1] += w * u1;
      grad_sj[2] += w * u2;
      grad_dot += w * (edge.sj[0] * b_dot[0] + edge.sj[1] * b_dot[1] + edge.sj[2] * b_dot[2]);
    }
    offset += C;

    for (int c = 0; c < C; ++c) {
      const double* K = block(geom_pull, edge.i, c, 9);
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
        const double* M = block(l1_pull, edge.i, c, 9);
        const double* Md = block(l1_dot_pull, edge.i, c, 9);
        const double v10 = M[0] * edge.sj[0] + M[1] * edge.sj[1] + M[2] * edge.sj[2];
        const double v11 = M[3] * edge.sj[0] + M[4] * edge.sj[1] + M[5] * edge.sj[2];
        const double v12 = M[6] * edge.sj[0] + M[7] * edge.sj[1] + M[8] * edge.sj[2];
        const double v20 = Md[0] * edge.sj[0] + Md[1] * edge.sj[1] + Md[2] * edge.sj[2];
        const double v21 = Md[3] * edge.sj[0] + Md[4] * edge.sj[1] + Md[5] * edge.sj[2];
        const double v22 = Md[6] * edge.sj[0] + Md[7] * edge.sj[1] + Md[8] * edge.sj[2];
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
        value[width++] = y * edge.sj[0];
        value[width++] = y * edge.sj[1];
        value[width++] = y * edge.sj[2];
      }
      double ge[27] = {0.0};
      add_density_gradient(width, angular, value, 1.0, false, offset, nullptr, ge, grad_dot);
      apply_angular(ell, ylm, ylm_width, ge);
      offset += C;
    }

    offset += C;
    offset += C;
    if (l_max >= 1) {
      offset += C;
    }

    for (int d = 0; d < 3; ++d) {
      grad_si[d] += grad_dot * edge.sj[d];
      grad_sj[d] += grad_dot * edge.si[d];
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
      local_force[static_cast<std::size_t>(d) * N + edge.i] += grad_rij[d];
      local_force[static_cast<std::size_t>(d) * N + edge.j] -= grad_rij[d];
    }
    for (int a = 0; a < 3; ++a) {
      const double rij_a = edge.rhat[a] * edge.dist;
      for (int b = 0; b < 3; ++b) {
        if (use_parallel_edges) {
          local_virial[a * 3 + b] -= rij_a * grad_rij[b];
        } else {
          virial[static_cast<std::size_t>(a * 3 + b) * N] -= rij_a * grad_rij[b];
        }
      }
    }
    add_local_grad_spin(edge.i, grad_si);
    add_local_grad_spin(edge.j, grad_sj);
  }

  if (use_parallel_edges) {
    for (int tid = 0; tid < num_threads; ++tid) {
      const double* local_force = force_private.data() + static_cast<std::size_t>(tid) * 3 * N;
      const double* local_grad_spin =
        grad_spin_private.data() + static_cast<std::size_t>(tid) * 3 * N;
      const double* local_virial = virial_private.data() + static_cast<std::size_t>(tid) * 9;
      for (int atom = 0; atom < N; ++atom) {
        for (int d = 0; d < 3; ++d) {
          const std::size_t idx = static_cast<std::size_t>(d) * N + atom;
          force[idx] += local_force[idx];
          grad_spin[idx] += local_grad_spin[idx];
        }
      }
      for (int d = 0; d < 9; ++d) {
        virial[static_cast<std::size_t>(d) * N] += local_virial[d];
      }
    }
  }

  if (phase) {
    phase->gradient_nonchiral += nep_phase_elapsed(phase_mark);
  }
  add_spin_chiral_gradient(
    paramb, annmb, N, type, cache, Fp, grad_spin, force, virial);
  if (phase) {
    phase->gradient_chiral += nep_phase_elapsed(phase_mark);
  }

  for (int atom = 0; atom < N; ++atom) {
    for (int d = 0; d < 3; ++d) {
      mforce[static_cast<std::size_t>(d) * N + atom] -= grad_spin[static_cast<std::size_t>(d) * N + atom];
    }
  }
}

} // namespace

NEP::NEP() {}

NEP::NEP(const std::string& potential_filename) { init_from_file(potential_filename, true); }

void NEP::init_from_file(const std::string& potential_filename, const bool is_rank_0)
{
  std::ifstream input(potential_filename);
  if (!input.is_open()) {
    std::cout << "Failed to open " << potential_filename << std::endl;
    exit(1);
  }

  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() < 3) {
    print_tokens(tokens);
    std::cout << "The first line of nep.txt should have at least 3 items." << std::endl;
    exit(1);
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
    std::cout << tokens[0] << " is an unsupported NEP model." << std::endl;
    exit(1);
  }

  paramb.num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (tokens.size() != 2 + paramb.num_types) {
    print_tokens(tokens);
    std::cout << "The first line of nep.txt should have " << paramb.num_types << " atom symbols."
              << std::endl;
    exit(1);
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
        throw std::runtime_error("only spin_scaler 1 is supported by cpu_opt");
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
    if (paramb.spin_basis_size + 1 > MAX_NUM_N || paramb.spin_compress > MAX_NUM_N) {
      throw std::runtime_error("spin basis is too large for cpu_opt");
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
      print_tokens(tokens);
      std::cout << "This line should be zbl rc_inner rc_outer [zbl_factor]." << std::endl;
      exit(1);
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
    print_tokens(tokens);
    std::cout << "cutoff should have 4 or num_types * 2 + 2 parameters.\n";
    exit(1);
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
    print_tokens(tokens);
    std::cout << "This line should be n_max n_max_radial n_max_angular." << std::endl;
    exit(1);
  }
  paramb.n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);

  // basis_size 10 8
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    print_tokens(tokens);
    std::cout << "This line should be basis_size basis_size_radial basis_size_angular."
              << std::endl;
    exit(1);
  }
  paramb.basis_size_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.basis_size_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);

  // l_max
  tokens = get_tokens(input);
  if (tokens.size() < 4) {
    print_tokens(tokens);
    std::cout << "l_max line should have 3 to 6 values." << std::endl;
    exit(1);
  }

  paramb.L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.num_L = paramb.L_max;

  int tok2 = get_int_from_token(tokens[2], __FILE__, __LINE__);
  if (tok2 > 1) {
    // old format: has_q_222 encoded as 0 or 2, tokens[3] = has_q_1111 (0 or 1)
    int L_max_5body = get_int_from_token(tokens[3], __FILE__, __LINE__);
    if (tok2 == 2) {
      paramb.has_q_222 = 1;
      paramb.num_L += 1;
    }
    if (L_max_5body == 1) {
      paramb.has_q_1111 = 1;
      paramb.num_L += 1;
    }
  } else {
    // new format: explicit 0/1 boolean flags
    paramb.has_q_222 = tok2;
    paramb.has_q_1111 = get_int_from_token(tokens[3], __FILE__, __LINE__);
    if (tokens.size() >= 5)
      paramb.has_q_112 = get_int_from_token(tokens[4], __FILE__, __LINE__);
    if (tokens.size() >= 6)
      paramb.has_q_123 = get_int_from_token(tokens[5], __FILE__, __LINE__);
    if (tokens.size() >= 7)
      paramb.has_q_233 = get_int_from_token(tokens[6], __FILE__, __LINE__);
    if (tokens.size() >= 8)
      paramb.has_q_134 = get_int_from_token(tokens[7], __FILE__, __LINE__);
    paramb.num_L += paramb.has_q_222 + paramb.has_q_1111 + paramb.has_q_112
                  + paramb.has_q_123 + paramb.has_q_233 + paramb.has_q_134;
  }

  paramb.dim_angular = (paramb.n_max_angular + 1) * paramb.num_L;

  // ANN
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    print_tokens(tokens);
    std::cout << "This line should be ANN num_neurons 0." << std::endl;
    exit(1);
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
  if (num_atoms < N) {
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
  std::vector<double>& virial)
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
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  prepare_table_small_box(
    N, NN_radial.data(), NL_radial.data(), NN_angular.data(), NL_angular.data(), type.data());
  if (phase_timing) {
    phase_table = nep_phase_elapsed(phase_mark);
  }
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
  if (phase_timing) {
    phase_descriptor = nep_phase_elapsed(phase_mark);
  }

  find_force_radial_small_box(
    false, paramb, annmb, N, NN_radial.data(), NL_radial.data(), type.data(), r12.data(),
    r12.data() + size_x12, r12.data() + size_x12 * 2, Fp.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gnp_radial.data(),
#endif
    force.data(), force.data() + N, force.data() + N * 2, virial.data());
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
    force.data(), force.data() + N, force.data() + N * 2, virial.data());
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
  std::vector<double>& mforce)
{
  const std::size_t N = type.size();
  if (N != potential.size() || N * 3 != force.size() || N * 9 != virial.size() ||
      N * annmb.dim != descriptor.size() || N * 3 != mforce.size()) {
    throw std::runtime_error("spin output sizes are inconsistent");
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
    force.data(), virial.data(), mforce.data(), phase_timing ? &spin_phase : nullptr);
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

  zero_total_charge(N, charge.data());

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
      const std::size_t total_virial_size = static_cast<std::size_t>(num_threads) * 6;
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
  int N,
  int*,
  int*,
  int**,
  int* type,
  int* type_map,
  double** pos,
  double** spins,
  double& total_potential,
  double total_virial[6],
  double* potential,
  double** force,
  double** mforce,
  double** virial)
{
  if (!spins) {
    throw std::runtime_error("spin LAMMPS path requires spins");
  }
  std::vector<int> mapped_type(static_cast<std::size_t>(N));
  std::vector<double> box(9, 0.0);
  std::vector<double> position(static_cast<std::size_t>(N) * 3, 0.0);
  std::vector<double> spin_soa(static_cast<std::size_t>(N) * 3, 0.0);
  double min_pos[3] = {0.0, 0.0, 0.0};
  double max_pos[3] = {0.0, 0.0, 0.0};
  if (N > 0) {
    for (int d = 0; d < 3; ++d) {
      min_pos[d] = pos[0][d];
      max_pos[d] = pos[0][d];
    }
    for (int atom = 1; atom < N; ++atom) {
      for (int d = 0; d < 3; ++d) {
        min_pos[d] = std::min(min_pos[d], pos[atom][d]);
        max_pos[d] = std::max(max_pos[d], pos[atom][d]);
      }
    }
  }
  const double padding = paramb.rc_radial_max + 1.0;
  for (int i = 0; i < 3; ++i) {
    const double span = max_pos[i] - min_pos[i];
    box[i * 3 + i] = std::max(span + 2.0 * padding, 2.6 * paramb.rc_radial_max);
  }
  for (int atom = 0; atom < N; ++atom) {
    mapped_type[atom] = type_map[type[atom]];
    for (int d = 0; d < 3; ++d) {
      position[static_cast<std::size_t>(d) * N + atom] = pos[atom][d] - min_pos[d] + padding;
      spin_soa[static_cast<std::size_t>(d) * N + atom] = spins[atom][d];
    }
  }
  std::vector<double> pe(static_cast<std::size_t>(N), 0.0);
  std::vector<double> force_soa(static_cast<std::size_t>(N) * 3, 0.0);
  std::vector<double> virial_soa(static_cast<std::size_t>(N) * 9, 0.0);
  std::vector<double> descriptor(static_cast<std::size_t>(N) * annmb.dim, 0.0);
  std::vector<double> mforce_soa(static_cast<std::size_t>(N) * 3, 0.0);
  compute(mapped_type, box, position, spin_soa, pe, force_soa, virial_soa, descriptor, mforce_soa);

  total_potential = 0.0;
  std::fill(total_virial, total_virial + 6, 0.0);
  for (int atom = 0; atom < nlocal; ++atom) {
    total_potential += pe[atom];
    if (potential) {
      potential[atom] = pe[atom];
    }
    for (int d = 0; d < 3; ++d) {
      force[atom][d] += force_soa[static_cast<std::size_t>(d) * N + atom];
      if (mforce) {
        mforce[atom][d] += mforce_soa[static_cast<std::size_t>(d) * N + atom];
      }
    }
  }
  auto raw = [&](const int comp, const int atom) {
    return virial_soa[static_cast<std::size_t>(comp) * N + atom];
  };
  for (int atom = 0; atom < nlocal; ++atom) {
    total_virial[0] += raw(0, atom);
    total_virial[1] += raw(4, atom);
    total_virial[2] += raw(8, atom);
    total_virial[3] += raw(1, atom);
    total_virial[4] += raw(2, atom);
    total_virial[5] += raw(5, atom);
    if (virial) {
      virial[atom][0] += raw(0, atom);
      virial[atom][1] += raw(4, atom);
      virial[atom][2] += raw(8, atom);
      virial[atom][3] += raw(1, atom);
      virial[atom][4] += raw(2, atom);
      virial[atom][5] += raw(5, atom);
      virial[atom][6] += raw(3, atom);
      virial[atom][7] += raw(6, atom);
      virial[atom][8] += raw(7, atom);
    }
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
    std::cout << "The " << functional
              << " functional is not supported for DFT-D3 with BJ damping.\n"
              << std::endl;
    exit(1);
  }
};
