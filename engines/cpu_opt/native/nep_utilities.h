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

#pragma once

#include <cmath>
#include <cstddef>

#if defined(__clang__) || defined(__GNUC__)
#define NEP_RESTRICT __restrict__
#else
#define NEP_RESTRICT
#endif

namespace
{
const int MAX_NEURON = 120; // maximum number of neurons in the hidden layer
const int MN = 1000;        // maximum number of neighbors for one atom
const int NUM_OF_ABC = 80;  // 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 for L_max = 8
const int MAX_NUM_N = 17;   // basis_size_radial+1 = 16+1
const int MAX_DIM = 256;
const int MAX_DIM_ANGULAR = 90;
const double C3B[NUM_OF_ABC] = {
  0.238732414637843, 0.119366207318922, 0.119366207318922, 0.099471839432435, 0.596831036594608,
  0.596831036594608, 0.149207759148652, 0.149207759148652, 0.139260575205408, 0.104445431404056,
  0.104445431404056, 1.044454314040563, 1.044454314040563, 0.174075719006761, 0.174075719006761,
  0.011190581936149, 0.223811638722978, 0.223811638722978, 0.111905819361489, 0.111905819361489,
  1.566681471060845, 1.566681471060845, 0.195835183882606, 0.195835183882606, 0.013677377921960,
  0.102580334414698, 0.102580334414698, 2.872249363611549, 2.872249363611549, 0.119677056817148,
  0.119677056817148, 2.154187022708661, 2.154187022708661, 0.215418702270866, 0.215418702270866,
  0.004041043476943, 0.169723826031592, 0.169723826031592, 0.106077391269745, 0.106077391269745,
  0.424309565078979, 0.424309565078979, 0.127292869523694, 0.127292869523694, 2.800443129521260,
  2.800443129521260, 0.233370260793438, 0.233370260793438, 0.004662742473395, 0.004079899664221,
  0.004079899664221, 0.024479397985326, 0.024479397985326, 0.012239698992663, 0.012239698992663,
  0.538546755677165, 0.538546755677165, 0.134636688919291, 0.134636688919291, 3.500553911901575,
  3.500553911901575, 0.250039565135827, 0.250039565135827, 0.000082569397966, 0.005944996653579,
  0.005944996653579, 0.104037441437634, 0.104037441437634, 0.762941237209318, 0.762941237209318,
  0.114441185581398, 0.114441185581398, 5.950941650232678, 5.950941650232678, 0.141689086910302,
  0.141689086910302, 4.250672607309055, 4.250672607309055, 0.265667037956816, 0.265667037956816};
const double C4B[5] = {
  -0.007499480826664, -0.134990654879954, 0.067495327439977, 0.404971964639861, -0.809943929279723};
const double C5B[3] = {0.026596810706114, 0.053193621412227, 0.026596810706114};
const double C4B2[5] = {
  0.027493550848847, 0.164961305093080, -0.013746775424423, 0.041240326273270, 0.082480652546540};
const double C4B_123[7] = {
  -0.008418146349617, -0.016836292699234, -0.033672585398469, -0.042090731748086,
  -0.067345170796937, -0.084181463496172, -0.168362926992344};
const double C4B_233[10] = {
  0.008572620635186, 0.009644198214584, 0.019288396429168, 0.025717861905558, 0.026789439484956,
  0.032147327381947, 0.038576792858337, 0.128589309527790, 0.192883964291685, 0.321473273819474};
const double C4B_134[10] = {
  0.003645164295772, 0.004860219061029, 0.006075273826286, 0.018225821478859, 0.024301095305146,
  0.036451642957719, 0.042526916784005, 0.072903285915437, 0.085053833568010, 0.255161500704030};

const double Z_COEFFICIENT_1[2][2] = {{0.0, 1.0}, {1.0, 0.0}};

const double Z_COEFFICIENT_2[3][3] = {{-1.0, 0.0, 3.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 0.0}};

const double Z_COEFFICIENT_3[4][4] = {
  {0.0, -3.0, 0.0, 5.0}, {-1.0, 0.0, 5.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}};

const double Z_COEFFICIENT_4[5][5] = {
  {3.0, 0.0, -30.0, 0.0, 35.0},
  {0.0, -3.0, 0.0, 7.0, 0.0},
  {-1.0, 0.0, 7.0, 0.0, 0.0},
  {0.0, 1.0, 0.0, 0.0, 0.0},
  {1.0, 0.0, 0.0, 0.0, 0.0}};

const double Z_COEFFICIENT_5[6][6] = {
  {0.0, 15.0, 0.0, -70.0, 0.0, 63.0}, {1.0, 0.0, -14.0, 0.0, 21.0, 0.0},
  {0.0, -1.0, 0.0, 3.0, 0.0, 0.0},    {-1.0, 0.0, 9.0, 0.0, 0.0, 0.0},
  {0.0, 1.0, 0.0, 0.0, 0.0, 0.0},     {1.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

const double Z_COEFFICIENT_6[7][7] = {
  {-5.0, 0.0, 105.0, 0.0, -315.0, 0.0, 231.0}, {0.0, 5.0, 0.0, -30.0, 0.0, 33.0, 0.0},
  {1.0, 0.0, -18.0, 0.0, 33.0, 0.0, 0.0},      {0.0, -3.0, 0.0, 11.0, 0.0, 0.0, 0.0},
  {-1.0, 0.0, 11.0, 0.0, 0.0, 0.0, 0.0},       {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

const double Z_COEFFICIENT_7[8][8] = {{0.0, -35.0, 0.0, 315.0, 0.0, -693.0, 0.0, 429.0},
                                      {-5.0, 0.0, 135.0, 0.0, -495.0, 0.0, 429.0, 0.0},
                                      {0.0, 15.0, 0.0, -110.0, 0.0, 143.0, 0.0, 0.0},
                                      {3.0, 0.0, -66.0, 0.0, 143.0, 0.0, 0.0, 0.0},
                                      {0.0, -3.0, 0.0, 13.0, 0.0, 0.0, 0.0, 0.0},
                                      {-1.0, 0.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                      {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                      {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

const double Z_COEFFICIENT_8[9][9] = {
  {35.0, 0.0, -1260.0, 0.0, 6930.0, 0.0, -12012.0, 0.0, 6435.0},
  {0.0, -35.0, 0.0, 385.0, 0.0, -1001.0, 0.0, 715.0, 0.0},
  {-1.0, 0.0, 33.0, 0.0, -143.0, 0.0, 143.0, 0.0, 0.0},
  {0.0, 3.0, 0.0, -26.0, 0.0, 39.0, 0.0, 0.0, 0.0},
  {1.0, 0.0, -26.0, 0.0, 65.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, -1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {-1.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

template <int L>
inline double z_coefficient(const int n1, const int n2)
{
  if constexpr (L == 1) {
    return Z_COEFFICIENT_1[n1][n2];
  } else if constexpr (L == 2) {
    return Z_COEFFICIENT_2[n1][n2];
  } else if constexpr (L == 3) {
    return Z_COEFFICIENT_3[n1][n2];
  } else if constexpr (L == 4) {
    return Z_COEFFICIENT_4[n1][n2];
  } else if constexpr (L == 5) {
    return Z_COEFFICIENT_5[n1][n2];
  } else if constexpr (L == 6) {
    return Z_COEFFICIENT_6[n1][n2];
  } else if constexpr (L == 7) {
    return Z_COEFFICIENT_7[n1][n2];
  } else {
    return Z_COEFFICIENT_8[n1][n2];
  }
}

const double K_C_SP = 14.399645; // 1/(4*PI*epsilon_0)
const double PI = 3.141592653589793;
const double PI_HALF = 1.570796326794897;
const int NUM_ELEMENTS = 94;
const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};
double COVALENT_RADIUS[NUM_ELEMENTS] = {
  0.426667, 0.613333, 1.6,     1.25333, 1.02667, 1.0,     0.946667, 0.84,    0.853333, 0.893333,
  1.86667,  1.66667,  1.50667, 1.38667, 1.46667, 1.36,    1.32,     1.28,    2.34667,  2.05333,
  1.77333,  1.62667,  1.61333, 1.46667, 1.42667, 1.38667, 1.33333,  1.32,    1.34667,  1.45333,
  1.49333,  1.45333,  1.53333, 1.46667, 1.52,    1.56,    2.52,     2.22667, 1.96,     1.85333,
  1.76,     1.65333,  1.53333, 1.50667, 1.50667, 1.44,    1.53333,  1.64,    1.70667,  1.68,
  1.68,     1.64,     1.76,    1.74667, 2.78667, 2.34667, 2.16,     1.96,    2.10667,  2.09333,
  2.08,     2.06667,  2.01333, 2.02667, 2.01333, 2.0,     1.98667,  1.98667, 1.97333,  2.04,
  1.94667,  1.82667,  1.74667, 1.64,    1.57333, 1.54667, 1.48,     1.49333, 1.50667,  1.76,
  1.73333,  1.73333,  1.81333, 1.74667, 1.84,    1.89333, 2.68,     2.41333, 2.22667,  2.10667,
  2.02667,  2.04,     2.05333, 2.06667};

void complex_product(const double a, const double b, double& real_part, double& imag_part)
{
  const double real_temp = real_part;
  real_part = a * real_temp - b * imag_part;
  imag_part = a * imag_part + b * real_temp;
}

void apply_ann_one_layer(
  const int dim,
  const int num_neurons1,
  const double* w0,
  const double* b0,
  const double* w1,
  const double* b1,
  double* q,
  double& energy,
  double* energy_derivative,
  double* latent_space,
  bool need_B_projection,
  double* B_projection)
{
  if (need_B_projection) {
    for (int n = 0; n < num_neurons1; ++n) {
      const double* w0_n = w0 + n * dim;
      double w0_times_q = 0.0;
      for (int d = 0; d < dim; ++d) {
        w0_times_q += w0_n[d] * q[d];
      }
      double x1 = tanh(w0_times_q - b0[n]);
      double tan_der = 1.0 - x1 * x1;
      double w1_tan_der = w1[n] * tan_der;
      double w1_x1 = w1[n] * x1;

      for (int d = 0; d < dim; ++d)
        B_projection[n * (dim + 2) + d] = w1_tan_der * q[d];
      B_projection[n * (dim + 2) + dim] = -w1_tan_der;
      B_projection[n * (dim + 2) + dim + 1] = x1;

      latent_space[n] = w1_x1; // also try x1
      energy += w1_x1;
      for (int d = 0; d < dim; ++d) {
        energy_derivative[d] += w1_tan_der * w0_n[d];
      }
    }
    energy -= b1[0];
    return;
  }

  for (int n = 0; n < num_neurons1; ++n) {
    const double* w0_n = w0 + n * dim;
    double w0_times_q = 0.0;
    for (int d = 0; d < dim; ++d) {
      w0_times_q += w0_n[d] * q[d];
    }
    double x1 = tanh(w0_times_q - b0[n]);
    double tan_der = 1.0 - x1 * x1;
    double w1_tan_der = w1[n] * tan_der;
    double w1_x1 = w1[n] * x1;

    latent_space[n] = w1_x1; // also try x1
    energy += w1_x1;
    for (int d = 0; d < dim; ++d) {
      energy_derivative[d] += w1_tan_der * w0_n[d];
    }
  }
  energy -= b1[0];
}

void apply_ann_one_layer_charge(
  const int N_des,
  const int N_neu,
  const double* w0,
  const double* b0,
  const double* w1,
  const double* b1,
  double* q,
  double& energy,
  double* energy_derivative,
  double& charge,
  double* charge_derivative)
{
  for (int n = 0; n < N_neu; ++n) {
    double w0_times_q = 0.0;
    for (int d = 0; d < N_des; ++d) {
      w0_times_q += w0[n * N_des + d] * q[d];
    }
    double x1 = tanh(w0_times_q - b0[n]);
    double tanh_der = 1.0 - x1 * x1;
    energy += w1[n] * x1;
    charge += w1[n + N_neu] * x1;
    for (int d = 0; d < N_des; ++d) {
      double y1 = tanh_der * w0[n * N_des + d];
      energy_derivative[d] += w1[n] * y1;
      charge_derivative[d] += w1[n + N_neu] * y1;
    }
  }
  energy -= b1[0];
}

void apply_ann_one_layer_nep5(
  const int dim,
  const int num_neurons1,
  const double* w0,
  const double* b0,
  const double* w1,
  const double* b1,
  double* q,
  double& energy,
  double* energy_derivative,
  double* latent_space)
{
  for (int n = 0; n < num_neurons1; ++n) {
    double w0_times_q = 0.0;
    for (int d = 0; d < dim; ++d) {
      w0_times_q += w0[n * dim + d] * q[d];
    }
    double x1 = tanh(w0_times_q - b0[n]);
    latent_space[n] = w1[n] * x1; // also try x1
    energy += w1[n] * x1;
    for (int d = 0; d < dim; ++d) {
      double y1 = (1.0 - x1 * x1) * w0[n * dim + d];
      energy_derivative[d] += w1[n] * y1;
    }
  }
  energy -= w1[num_neurons1] + b1[0]; // typewise bias + common bias
}

void find_fc(double rc, double rcinv, double d12, double& fc)
{
  if (d12 < rc) {
    double x = d12 * rcinv;
    fc = 0.5 * cos(PI * x) + 0.5;
  } else {
    fc = 0.0;
  }
}

void find_fc_and_fcp(double rc, double rcinv, double d12, double& fc, double& fcp)
{
  if (d12 < rc) {
    double x = d12 * rcinv;
    fc = 0.5 * cos(PI * x) + 0.5;
    fcp = -PI_HALF * sin(PI * x);
    fcp *= rcinv;
  } else {
    fc = 0.0;
    fcp = 0.0;
  }
}

void find_fc_and_fcp_zbl(double r1, double r2, double d12, double& fc, double& fcp)
{
  if (d12 < r1) {
    fc = 1.0;
    fcp = 0.0;
  } else if (d12 < r2) {
    double pi_factor = PI / (r2 - r1);
    fc = cos(pi_factor * (d12 - r1)) * 0.5 + 0.5;
    fcp = -sin(pi_factor * (d12 - r1)) * pi_factor * 0.5;
  } else {
    fc = 0.0;
    fcp = 0.0;
  }
}

void find_phi_and_phip_zbl(double a, double b, double x, double& phi, double& phip)
{
  double tmp = a * exp(-b * x);
  phi += tmp;
  phip -= b * tmp;
}

void find_f_and_fp_zbl(
  double zizj,
  double a_inv,
  double rc_inner,
  double rc_outer,
  double d12,
  double d12inv,
  double& f,
  double& fp)
{
  double x = d12 * a_inv;
  f = fp = 0.0;
  double Zbl_para[8] = {0.18175, 3.1998, 0.50986, 0.94229, 0.28022, 0.4029, 0.02817, 0.20162};
  find_phi_and_phip_zbl(Zbl_para[0], Zbl_para[1], x, f, fp);
  find_phi_and_phip_zbl(Zbl_para[2], Zbl_para[3], x, f, fp);
  find_phi_and_phip_zbl(Zbl_para[4], Zbl_para[5], x, f, fp);
  find_phi_and_phip_zbl(Zbl_para[6], Zbl_para[7], x, f, fp);
  f *= zizj;
  fp *= zizj * a_inv;
  fp = fp * d12inv - f * d12inv * d12inv;
  f *= d12inv;
  double fc, fcp;
  find_fc_and_fcp_zbl(rc_inner, rc_outer, d12, fc, fcp);
  fp = fp * fc + f * fcp;
  f *= fc;
}

void find_f_and_fp_zbl(
  double* zbl_para, double zizj, double a_inv, double d12, double d12inv, double& f, double& fp)
{
  double x = d12 * a_inv;
  f = fp = 0.0;
  find_phi_and_phip_zbl(zbl_para[2], zbl_para[3], x, f, fp);
  find_phi_and_phip_zbl(zbl_para[4], zbl_para[5], x, f, fp);
  find_phi_and_phip_zbl(zbl_para[6], zbl_para[7], x, f, fp);
  find_phi_and_phip_zbl(zbl_para[8], zbl_para[9], x, f, fp);
  f *= zizj;
  fp *= zizj * a_inv;
  fp = fp * d12inv - f * d12inv * d12inv;
  f *= d12inv;
  double fc, fcp;
  find_fc_and_fcp_zbl(zbl_para[0], zbl_para[1], d12, fc, fcp);
  fp = fp * fc + f * fcp;
  f *= fc;
}

void find_fn(const int n, const double rcinv, const double d12, const double fc12, double& fn)
{
  if (n == 0) {
    fn = fc12;
  } else if (n == 1) {
    double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
    fn = (x + 1.0) * 0.5 * fc12;
  } else {
    double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
    double t0 = 1.0;
    double t1 = x;
    double t2;
    for (int m = 2; m <= n; ++m) {
      t2 = 2.0 * x * t1 - t0;
      t0 = t1;
      t1 = t2;
    }
    fn = (t2 + 1.0) * 0.5 * fc12;
  }
}

void find_fn_and_fnp(
  const int n,
  const double rcinv,
  const double d12,
  const double fc12,
  const double fcp12,
  double& fn,
  double& fnp)
{
  if (n == 0) {
    fn = fc12;
    fnp = fcp12;
  } else if (n == 1) {
    double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
    fn = (x + 1.0) * 0.5;
    fnp = 2.0 * (d12 * rcinv - 1.0) * rcinv * fc12 + fn * fcp12;
    fn *= fc12;
  } else {
    double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
    double t0 = 1.0;
    double t1 = x;
    double t2;
    double u0 = 1.0;
    double u1 = 2.0 * x;
    double u2;
    for (int m = 2; m <= n; ++m) {
      t2 = 2.0 * x * t1 - t0;
      t0 = t1;
      t1 = t2;
      u2 = 2.0 * x * u1 - u0;
      u0 = u1;
      u1 = u2;
    }
    fn = (t2 + 1.0) * 0.5;
    fnp = n * u0 * 2.0 * (d12 * rcinv - 1.0) * rcinv;
    fnp = fnp * fc12 + fn * fcp12;
    fn *= fc12;
  }
}

void find_fn(const int n_max, const double rcinv, const double d12, const double fc12, double* fn)
{
  double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
  fn[0] = 1.0;
  fn[1] = x;
  for (int m = 2; m <= n_max; ++m) {
    fn[m] = 2.0 * x * fn[m - 1] - fn[m - 2];
  }
  for (int m = 0; m <= n_max; ++m) {
    fn[m] = (fn[m] + 1.0) * 0.5 * fc12;
  }
}

void find_fn_and_fnp(
  const int n_max,
  const double rcinv,
  const double d12,
  const double fc12,
  const double fcp12,
  double* fn,
  double* fnp)
{
  double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
  fn[0] = 1.0;
  fnp[0] = 0.0;
  fn[1] = x;
  fnp[1] = 1.0;
  double u0 = 1.0;
  double u1 = 2.0 * x;
  double u2;
  for (int m = 2; m <= n_max; ++m) {
    fn[m] = 2.0 * x * fn[m - 1] - fn[m - 2];
    fnp[m] = m * u1;
    u2 = 2.0 * x * u1 - u0;
    u0 = u1;
    u1 = u2;
  }
  for (int m = 0; m <= n_max; ++m) {
    fn[m] = (fn[m] + 1.0) * 0.5;
    fnp[m] *= 2.0 * (d12 * rcinv - 1.0) * rcinv;
    fnp[m] = fnp[m] * fc12 + fn[m] * fcp12;
    fn[m] *= fc12;
  }
}

void get_f12_4body(
  const double d12,
  const double d12inv,
  const double fn,
  const double fnp,
  const double Fp,
  const double* s,
  const double* r12,
  double* f12)
{
  double fn_factor = Fp * fn;
  double fnp_factor = Fp * fnp * d12inv;
  double y20 = (3.0 * r12[2] * r12[2] - d12 * d12);

  // derivative wrt s[0]
  double tmp0 = C4B[0] * 3.0 * s[0] * s[0] + C4B[1] * (s[1] * s[1] + s[2] * s[2]) +
                C4B[2] * (s[3] * s[3] + s[4] * s[4]);
  double tmp1 = tmp0 * y20 * fnp_factor;
  double tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] - tmp2 * 2.0 * r12[0];
  f12[1] += tmp1 * r12[1] - tmp2 * 2.0 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2 * 4.0 * r12[2];

  // derivative wrt s[1]
  tmp0 = C4B[1] * s[0] * s[1] * 2.0 - C4B[3] * s[3] * s[1] * 2.0 + C4B[4] * s[2] * s[4];
  tmp1 = tmp0 * r12[0] * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * r12[2];
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2 * r12[0];

  // derivative wrt s[2]
  tmp0 = C4B[1] * s[0] * s[2] * 2.0 + C4B[3] * s[3] * s[2] * 2.0 + C4B[4] * s[1] * s[4];
  tmp1 = tmp0 * r12[1] * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1] + tmp2 * r12[2];
  f12[2] += tmp1 * r12[2] + tmp2 * r12[1];

  // derivative wrt s[3]
  tmp0 = C4B[2] * s[0] * s[3] * 2.0 + C4B[3] * (s[2] * s[2] - s[1] * s[1]);
  tmp1 = tmp0 * (r12[0] * r12[0] - r12[1] * r12[1]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * 2.0 * r12[0];
  f12[1] += tmp1 * r12[1] - tmp2 * 2.0 * r12[1];
  f12[2] += tmp1 * r12[2];

  // derivative wrt s[4]
  tmp0 = C4B[2] * s[0] * s[4] * 2.0 + C4B[4] * s[1] * s[2];
  tmp1 = tmp0 * (2.0 * r12[0] * r12[1]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * 2.0 * r12[1];
  f12[1] += tmp1 * r12[1] + tmp2 * 2.0 * r12[0];
  f12[2] += tmp1 * r12[2];
}

void get_f12_5body(
  const double d12,
  const double d12inv,
  const double fn,
  const double fnp,
  const double Fp,
  const double* s,
  const double* r12,
  double* f12)
{
  double fn_factor = Fp * fn;
  double fnp_factor = Fp * fnp * d12inv;
  double s1_sq_plus_s2_sq = s[1] * s[1] + s[2] * s[2];

  // derivative wrt s[0]
  double tmp0 = C5B[0] * 4.0 * s[0] * s[0] * s[0] + C5B[1] * s1_sq_plus_s2_sq * 2.0 * s[0];
  double tmp1 = tmp0 * r12[2] * fnp_factor;
  double tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2;

  // derivative wrt s[1]
  tmp0 = C5B[1] * s[0] * s[0] * s[1] * 2.0 + C5B[2] * s1_sq_plus_s2_sq * s[1] * 4.0;
  tmp1 = tmp0 * r12[0] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2;
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2];

  // derivative wrt s[2]
  tmp0 = C5B[1] * s[0] * s[0] * s[2] * 2.0 + C5B[2] * s1_sq_plus_s2_sq * s[2] * 4.0;
  tmp1 = tmp0 * r12[1] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1] + tmp2;
  f12[2] += tmp1 * r12[2];
}

void get_f12_4body_2(
  const double d12,
  const double d12inv,
  const double fn1,
  const double fnp1,
  const double fn2,
  const double fnp2,
  const double Fp,
  const double* s1,
  const double* s2,
  const double* r12,
  double* f12)
{
  double fn_factor = Fp * fn2;
  double fnp_factor = Fp * fnp2 * d12inv;

  // derivative wrt s2[0]
  double tmp0 = C4B2[0] * s1[0] * s1[0] + C4B2[2] * (s1[1] * s1[1] + s1[2] * s1[2]);
  double tmp1 = tmp0 * (3.0 * r12[2] * r12[2] - d12 * d12) * fnp_factor;
  double tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] - tmp2 * 2.0 * r12[0];
  f12[1] += tmp1 * r12[1] - tmp2 * 2.0 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2 * 4.0 * r12[2];

  // derivative wrt s2[1]
  tmp0 = C4B2[1] * s1[0] * s1[1];
  tmp1 = tmp0 * r12[0] * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * r12[2];
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2 * r12[0];

  // derivative wrt s2[2]
  tmp0 = C4B2[1] * s1[0] * s1[2];
  tmp1 = tmp0 * r12[1] * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1] + tmp2 * r12[2];
  f12[2] += tmp1 * r12[2] + tmp2 * r12[1];

  // derivative wrt s2[3]
  tmp0 = C4B2[3] * (s1[1] * s1[1] - s1[2] * s1[2]);
  tmp1 = tmp0 * (r12[0] * r12[0] - r12[1] * r12[1]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * 2.0 * r12[0];
  f12[1] += tmp1 * r12[1] - tmp2 * 2.0 * r12[1];
  f12[2] += tmp1 * r12[2];

  // derivative wrt s2[4]
  tmp0 = C4B2[4] * s1[1] * s1[2];
  tmp1 = tmp0 * (2.0 * r12[0] * r12[1]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * 2.0 * r12[1];
  f12[1] += tmp1 * r12[1] + tmp2 * 2.0 * r12[0];
  f12[2] += tmp1 * r12[2];

  fn_factor = Fp * fn1;
  fnp_factor = Fp * fnp1 * d12inv;

  // derivative wrt s1[0]
  tmp0 = C4B2[0] * 2.0 * s1[0] * s2[0] + C4B2[1] * (s1[1] * s2[1] + s1[2] * s2[2]);
  tmp1 = tmp0 * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2;

  // derivative wrt s1[1]
  tmp0 = C4B2[1] * s1[0] * s2[1] + C4B2[2] * s1[1] * s2[0] * 2.0 + C4B2[3] * s1[1] * s2[3] * 2.0 +
         C4B2[4] * s1[2] * s2[4];
  tmp1 = tmp0 * r12[0] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2;
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2];

  // derivative wrt s1[2]
  tmp0 = C4B2[1] * s1[0] * s2[2] + C4B2[2] * s1[2] * s2[0] * 2.0 - C4B2[3] * s1[2] * s2[3] * 2.0 +
         C4B2[4] * s1[1] * s2[4];
  tmp1 = tmp0 * r12[1] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1] + tmp2;
  f12[2] += tmp1 * r12[2];
}

void get_f12_4body_123(
  const double d12,
  const double d12inv,
  const double fn1,
  const double fnp1,
  const double fn2,
  const double fnp2,
  const double fn3,
  const double fnp3,
  const double Fp,
  const double* s1,
  const double* s2,
  const double* s3,
  const double* r12,
  double* f12)
{
  // s1
  double fn_factor = Fp * fn1;
  double fnp_factor = Fp * fnp1 * d12inv;

  // derivative wrt s1[0]
  double tmp0 = C4B_123[5] * s3[3] * s2[3] + C4B_123[5] * s3[4] * s2[4] +
                C4B_123[4] * s3[2] * s2[2] + C4B_123[1] * s2[0] * s3[0] +
                C4B_123[4] * s2[1] * s3[1];
  double tmp1 = tmp0 * r12[2] * fnp_factor;
  double tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2;

  // derivative wrt s1[1]
  tmp0 = -C4B_123[0] * s3[2] * s2[4] + C4B_123[6] * s3[3] * s2[1] + C4B_123[6] * s3[4] * s2[2] +
         C4B_123[3] * s3[5] * s2[3] + C4B_123[3] * s3[6] * s2[4] - C4B_123[2] * s2[1] * s3[0] +
         C4B_123[1] * s2[0] * s3[1] - C4B_123[0] * s2[3] * s3[1];
  tmp1 = tmp0 * r12[0] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2;
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2];

  // derivative wrt s1[2]
  tmp0 = +C4B_123[6] * s3[4] * s2[1] - C4B_123[6] * s3[3] * s2[2] + C4B_123[3] * s3[6] * s2[3] -
         C4B_123[3] * s3[5] * s2[4] + C4B_123[1] * s3[2] * s2[0] + C4B_123[0] * s3[2] * s2[3] -
         C4B_123[2] * s2[2] * s3[0] - C4B_123[0] * s2[4] * s3[1];
  tmp1 = tmp0 * r12[1] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1] + tmp2;
  f12[2] += tmp1 * r12[2];

  // s2
  fn_factor = Fp * fn2;
  fnp_factor = Fp * fnp2 * d12inv;

  // derivative wrt s2[0]
  tmp0 = C4B_123[1] * (s3[2] * s1[2] + s1[0] * s3[0] + s1[1] * s3[1]);
  tmp1 = tmp0 * (3.0 * r12[2] * r12[2] - d12 * d12) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] - tmp2 * 2.0 * r12[0];
  f12[1] += tmp1 * r12[1] - tmp2 * 2.0 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2 * 4.0 * r12[2];

  // derivative wrt s2[1]
  tmp0 = C4B_123[6] * s3[4] * s1[2] + C4B_123[4] * s1[0] * s3[1] + C4B_123[6] * s1[1] * s3[3] -
         C4B_123[2] * s1[1] * s3[0];
  tmp1 = tmp0 * r12[0] * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * r12[2];
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2 * r12[0];

  // derivative wrt s2[2]
  tmp0 = -C4B_123[6] * s3[3] * s1[2] + C4B_123[4] * s3[2] * s1[0] - C4B_123[2] * s1[2] * s3[0] +
         C4B_123[6] * s1[1] * s3[4];
  tmp1 = tmp0 * r12[1] * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1] + tmp2 * r12[2];
  f12[2] += tmp1 * r12[2] + tmp2 * r12[1];

  // derivative wrt s2[3]
  tmp0 = +C4B_123[5] * s1[0] * s3[3] + C4B_123[3] * s3[6] * s1[2] + C4B_123[0] * s3[2] * s1[2] +
         C4B_123[3] * s1[1] * s3[5] - C4B_123[0] * s1[1] * s3[1];
  tmp1 = tmp0 * (r12[0] * r12[0] - r12[1] * r12[1]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * 2.0 * r12[0];
  f12[1] += tmp1 * r12[1] - tmp2 * 2.0 * r12[1];
  f12[2] += tmp1 * r12[2];

  // derivative wrt s2[4]
  tmp0 = +C4B_123[5] * s1[0] * s3[4] - C4B_123[3] * s3[5] * s1[2] - C4B_123[0] * s3[2] * s1[1] -
         C4B_123[0] * s1[2] * s3[1] + C4B_123[3] * s1[1] * s3[6];
  tmp1 = tmp0 * (2.0 * r12[0] * r12[1]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * 2.0 * r12[1];
  f12[1] += tmp1 * r12[1] + tmp2 * 2.0 * r12[0];
  f12[2] += tmp1 * r12[2];

  // s3
  fn_factor = Fp * fn3;
  fnp_factor = Fp * fnp3 * d12inv;

  // derivative wrt s3[0]
  tmp0 = C4B_123[1] * s1[0] * s2[0] - C4B_123[2] * (s1[2] * s2[2] + s1[1] * s2[1]);
  tmp1 = tmp0 * (5.0 * r12[2] * r12[2] - 3.0 * d12 * d12) * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] - tmp2 * 6.0 * r12[2] * r12[0];
  f12[1] += tmp1 * r12[1] - tmp2 * 6.0 * r12[2] * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2 * (9.0 * r12[2] * r12[2] - 3.0 * d12 * d12);

  // derivative wrt s3[1]
  tmp0 = C4B_123[4] * s1[0] * s2[1] + C4B_123[1] * s1[1] * s2[0] -
         C4B_123[0] * (s1[2] * s2[4] + s1[1] * s2[3]);
  tmp1 = tmp0 * (5.0 * r12[2] * r12[2] - d12 * d12) * r12[0] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] +=
    tmp1 * r12[0] + tmp2 * (4.0 * r12[2] * r12[2] - 3.0 * r12[0] * r12[0] - r12[1] * r12[1]);
  f12[1] += tmp1 * r12[1] - tmp2 * (2.0 * r12[0] * r12[1]);
  f12[2] += tmp1 * r12[2] + tmp2 * (8.0 * r12[0] * r12[2]);

  // derivative wrt s3[2]
  tmp0 = C4B_123[4] * s1[0] * s2[2] + C4B_123[1] * s1[2] * s2[0] +
         C4B_123[0] * (s1[2] * s2[3] - s1[1] * s2[4]);
  tmp1 = tmp0 * (5.0 * r12[2] * r12[2] - d12 * d12) * r12[1] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] - tmp2 * (2.0 * r12[0] * r12[1]);
  f12[1] +=
    tmp1 * r12[1] + tmp2 * (4.0 * r12[2] * r12[2] - r12[0] * r12[0] - 3.0 * r12[1] * r12[1]);
  f12[2] += tmp1 * r12[2] + tmp2 * (8.0 * r12[1] * r12[2]);

  // derivative wrt s3[3]
  tmp0 = -C4B_123[6] * s1[2] * s2[2] + C4B_123[5] * s1[0] * s2[3] + C4B_123[6] * s1[1] * s2[1];
  tmp1 = tmp0 * (r12[0] * r12[0] - r12[1] * r12[1]) * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * (2.0 * r12[0] * r12[2]);
  f12[1] += tmp1 * r12[1] - tmp2 * (2.0 * r12[1] * r12[2]);
  f12[2] += tmp1 * r12[2] + tmp2 * (r12[0] * r12[0] - r12[1] * r12[1]);

  // derivative wrt s3[4]
  tmp0 = C4B_123[6] * (s1[2] * s2[1] + s1[1] * s2[2]) + C4B_123[5] * s1[0] * s2[4];
  tmp1 = tmp0 * (2.0 * r12[0] * r12[1] * r12[2]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * (2.0 * r12[1] * r12[2]);
  f12[1] += tmp1 * r12[1] + tmp2 * (2.0 * r12[0] * r12[2]);
  f12[2] += tmp1 * r12[2] + tmp2 * (2.0 * r12[0] * r12[1]);

  // derivative wrt s3[5]
  tmp0 = C4B_123[3] * (-s1[2] * s2[4] + s1[1] * s2[3]);
  tmp1 = tmp0 * (r12[0] * r12[0] - 3.0 * r12[1] * r12[1]) * r12[0] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * (3.0 * (r12[0] * r12[0] - r12[1] * r12[1]));
  f12[1] += tmp1 * r12[1] - tmp2 * (6.0 * r12[0] * r12[1]);
  f12[2] += tmp1 * r12[2];

  // derivative wrt s3[6]
  tmp0 = C4B_123[3] * (s1[2] * s2[3] + s1[1] * s2[4]);
  tmp1 = tmp0 * (3.0 * r12[0] * r12[0] - r12[1] * r12[1]) * r12[1] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * (6.0 * r12[0] * r12[1]);
  f12[1] += tmp1 * r12[1] + tmp2 * (3.0 * (r12[0] * r12[0] - r12[1] * r12[1]));
  f12[2] += tmp1 * r12[2];
}

void get_f12_4body_233(
  const double d12,
  const double d12inv,
  const double fn2,
  const double fnp2,
  const double fn3,
  const double fnp3,
  const double Fp,
  const double* s2,
  const double* s3,
  const double* r12,
  double* f12)
{
  double fn_factor2 = Fp * fn2;
  double fnp_factor2 = Fp * fnp2 * d12inv;
  double fn_factor3 = Fp * fn3;
  double fnp_factor3 = Fp * fnp3 * d12inv;

  // s2[0]
  double tmp0 = C4B_233[0] * (s3[0] * s3[0]) + C4B_233[1] * (s3[2] * s3[2] + s3[1] * s3[1]) +
                C4B_233[4] * (-s3[5] * s3[5] - s3[6] * s3[6]);
  double tmp1 = tmp0 * (3.0 * r12[2] * r12[2] - d12 * d12) * fnp_factor2;
  double tmp2 = tmp0 * fn_factor2;
  f12[0] += tmp1 * r12[0] - tmp2 * 2.0 * r12[0];
  f12[1] += tmp1 * r12[1] - tmp2 * 2.0 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2 * 4.0 * r12[2];

  // s2[1]
  tmp0 = C4B_233[3] * (s3[0] * s3[1]) + C4B_233[8] * (s3[3] * s3[1] + s3[2] * s3[4]) +
         C4B_233[9] * (s3[4] * s3[6] + s3[5] * s3[3]);
  tmp1 = tmp0 * r12[0] * r12[2] * fnp_factor2;
  tmp2 = tmp0 * fn_factor2;
  f12[0] += tmp1 * r12[0] + tmp2 * r12[2];
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2 * r12[0];

  // s2[2]
  tmp0 = C4B_233[3] * (s3[2] * s3[0]) + C4B_233[8] * (s3[4] * s3[1] - s3[2] * s3[3]) +
         C4B_233[9] * (s3[3] * s3[6] - s3[5] * s3[4]);
  tmp1 = tmp0 * r12[1] * r12[2] * fnp_factor2;
  tmp2 = tmp0 * fn_factor2;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1] + tmp2 * r12[2];
  f12[2] += tmp1 * r12[2] + tmp2 * r12[1];

  // s2[3]
  tmp0 = C4B_233[2] * (-s3[2] * s3[2] + s3[1] * s3[1]) +
         C4B_233[5] * (-s3[5] * s3[1] - s3[2] * s3[6]) + C4B_233[7] * (-s3[3] * s3[0]);
  tmp1 = tmp0 * (r12[0] * r12[0] - r12[1] * r12[1]) * fnp_factor2;
  tmp2 = tmp0 * fn_factor2;
  f12[0] += tmp1 * r12[0] + tmp2 * 2.0 * r12[0];
  f12[1] += tmp1 * r12[1] - tmp2 * 2.0 * r12[1];
  f12[2] += tmp1 * r12[2];

  // s2[4]
  tmp0 = C4B_233[5] * (-s3[6] * s3[1] + s3[2] * s3[5]) + C4B_233[6] * (s3[2] * s3[1]) +
         C4B_233[7] * (-s3[4] * s3[0]);
  tmp1 = tmp0 * (2.0 * r12[0] * r12[1]) * fnp_factor2;
  tmp2 = tmp0 * fn_factor2;
  f12[0] += tmp1 * r12[0] + tmp2 * 2.0 * r12[1];
  f12[1] += tmp1 * r12[1] + tmp2 * 2.0 * r12[0];
  f12[2] += tmp1 * r12[2];

  // s3[0]
  tmp0 = 2.0 * C4B_233[0] * s2[0] * s3[0] + C4B_233[3] * (s2[1] * s3[1] + s2[2] * s3[2]) +
         C4B_233[7] * (-s2[3] * s3[3] - s2[4] * s3[4]);
  tmp1 = tmp0 * (5.0 * r12[2] * r12[2] - 3.0 * d12 * d12) * r12[2] * fnp_factor3;
  tmp2 = tmp0 * fn_factor3;
  f12[0] += tmp1 * r12[0] - tmp2 * 6.0 * r12[2] * r12[0];
  f12[1] += tmp1 * r12[1] - tmp2 * 6.0 * r12[2] * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2 * (9.0 * r12[2] * r12[2] - 3.0 * d12 * d12);

  // s3[1]
  tmp0 = 2.0 * C4B_233[1] * s2[0] * s3[1] + 2.0 * C4B_233[2] * s2[3] * s3[1] +
         C4B_233[3] * (s2[1] * s3[0]) + C4B_233[5] * (-s2[4] * s3[6] - s2[3] * s3[5]) +
         C4B_233[6] * (s2[4] * s3[2]) + C4B_233[8] * (s2[1] * s3[3] + s2[2] * s3[4]);
  tmp1 = tmp0 * (5.0 * r12[2] * r12[2] - d12 * d12) * r12[0] * fnp_factor3;
  tmp2 = tmp0 * fn_factor3;
  f12[0] +=
    tmp1 * r12[0] + tmp2 * (4.0 * r12[2] * r12[2] - 3.0 * r12[0] * r12[0] - r12[1] * r12[1]);
  f12[1] += tmp1 * r12[1] - tmp2 * (2.0 * r12[0] * r12[1]);
  f12[2] += tmp1 * r12[2] + tmp2 * (8.0 * r12[0] * r12[2]);

  // s3[2]
  tmp0 = 2.0 * C4B_233[1] * s2[0] * s3[2] - 2.0 * C4B_233[2] * s2[3] * s3[2] +
         C4B_233[3] * (s2[2] * s3[0]) + C4B_233[5] * (-s2[3] * s3[6] + s2[4] * s3[5]) +
         C4B_233[6] * (s2[4] * s3[1]) + C4B_233[8] * (s2[1] * s3[4] - s2[2] * s3[3]);
  tmp1 = tmp0 * (5.0 * r12[2] * r12[2] - d12 * d12) * r12[1] * fnp_factor3;
  tmp2 = tmp0 * fn_factor3;
  f12[0] += tmp1 * r12[0] - tmp2 * (2.0 * r12[0] * r12[1]);
  f12[1] +=
    tmp1 * r12[1] + tmp2 * (4.0 * r12[2] * r12[2] - r12[0] * r12[0] - 3.0 * r12[1] * r12[1]);
  f12[2] += tmp1 * r12[2] + tmp2 * (8.0 * r12[1] * r12[2]);

  // s3[3]
  tmp0 = C4B_233[7] * (-s2[3] * s3[0]) + C4B_233[8] * (s2[1] * s3[1] - s2[2] * s3[2]) +
         C4B_233[9] * (s2[2] * s3[6] + s2[1] * s3[5]);
  tmp1 = tmp0 * (r12[0] * r12[0] - r12[1] * r12[1]) * r12[2] * fnp_factor3;
  tmp2 = tmp0 * fn_factor3;
  f12[0] += tmp1 * r12[0] + tmp2 * (2.0 * r12[0] * r12[2]);
  f12[1] += tmp1 * r12[1] - tmp2 * (2.0 * r12[1] * r12[2]);
  f12[2] += tmp1 * r12[2] + tmp2 * (r12[0] * r12[0] - r12[1] * r12[1]);

  // s3[4]
  tmp0 = C4B_233[7] * (-s2[4] * s3[0]) + C4B_233[8] * (s2[2] * s3[1] + s2[1] * s3[2]) +
         C4B_233[9] * (s2[1] * s3[6] - s2[2] * s3[5]);
  tmp1 = tmp0 * (2.0 * r12[0] * r12[1] * r12[2]) * fnp_factor3;
  tmp2 = tmp0 * fn_factor3;
  f12[0] += tmp1 * r12[0] + tmp2 * (2.0 * r12[1] * r12[2]);
  f12[1] += tmp1 * r12[1] + tmp2 * (2.0 * r12[0] * r12[2]);
  f12[2] += tmp1 * r12[2] + tmp2 * (2.0 * r12[0] * r12[1]);

  // s3[5]
  tmp0 = -2.0 * C4B_233[4] * s2[0] * s3[5] + C4B_233[5] * (s2[4] * s3[2] - s2[3] * s3[1]) +
         C4B_233[9] * (s2[1] * s3[3] - s2[2] * s3[4]);
  tmp1 = tmp0 * (r12[0] * r12[0] - 3.0 * r12[1] * r12[1]) * r12[0] * fnp_factor3;
  tmp2 = tmp0 * fn_factor3;
  f12[0] += tmp1 * r12[0] + tmp2 * (3.0 * (r12[0] * r12[0] - r12[1] * r12[1]));
  f12[1] += tmp1 * r12[1] - tmp2 * (6.0 * r12[0] * r12[1]);
  f12[2] += tmp1 * r12[2];

  // s3[6]
  tmp0 = -2.0 * C4B_233[4] * s2[0] * s3[6] + C4B_233[5] * (-s2[3] * s3[2] - s2[4] * s3[1]) +
         C4B_233[9] * (s2[1] * s3[4] + s2[2] * s3[3]);
  tmp1 = tmp0 * (3.0 * r12[0] * r12[0] - r12[1] * r12[1]) * r12[1] * fnp_factor3;
  tmp2 = tmp0 * fn_factor3;
  f12[0] += tmp1 * r12[0] + tmp2 * (6.0 * r12[0] * r12[1]);
  f12[1] += tmp1 * r12[1] + tmp2 * (3.0 * (r12[0] * r12[0] - r12[1] * r12[1]));
  f12[2] += tmp1 * r12[2];
}

void get_f12_4body_134(
  const double d12,
  const double d12inv,
  const double fn1,
  const double fnp1,
  const double fn3,
  const double fnp3,
  const double fn4,
  const double fnp4,
  const double Fp,
  const double* s1,
  const double* s3,
  const double* s4,
  const double* r12,
  double* f12)
{
  // s1
  double fn_factor = Fp * fn1;
  double fnp_factor = Fp * fnp1 * d12inv;

  double tmp0 = C4B_134[1] * s4[0] * s3[0] +
                C4B_134[5] * (s3[2] * s4[2] + s4[1] * s3[1]) +
                C4B_134[7] * (s3[3] * s4[3] + s3[4] * s4[4]) +
                C4B_134[8] * (s3[5] * s4[5] + s3[6] * s4[6]);
  double tmp1 = tmp0 * r12[2] * fnp_factor;
  double tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2;

  tmp0 = -C4B_134[0] * s4[0] * s3[1] +
         C4B_134[2] * (-s3[5] * s4[3] - s3[6] * s4[4]) +
         C4B_134[3] * (s3[2] * s4[4] + s4[3] * s3[1]) +
         C4B_134[4] * s4[1] * s3[0] +
         C4B_134[5] * (-s3[3] * s4[1] - s3[4] * s4[2]) +
         C4B_134[6] * (s3[5] * s4[7] + s3[6] * s4[8]) +
         C4B_134[9] * (s3[3] * s4[5] + s3[4] * s4[6]);
  tmp1 = tmp0 * r12[0] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2;
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2];

  tmp0 = -C4B_134[0] * s3[2] * s4[0] +
         C4B_134[2] * (-s3[6] * s4[3] + s3[5] * s4[4]) +
         C4B_134[3] * (-s3[2] * s4[3] + s4[4] * s3[1]) +
         C4B_134[4] * s4[2] * s3[0] +
         C4B_134[5] * (-s3[4] * s4[1] + s3[3] * s4[2]) +
         C4B_134[6] * (-s3[6] * s4[7] + s3[5] * s4[8]) +
         C4B_134[9] * (-s3[4] * s4[5] + s3[3] * s4[6]);
  tmp1 = tmp0 * r12[1] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1] + tmp2;
  f12[2] += tmp1 * r12[2];

  // s3
  fn_factor = Fp * fn3;
  fnp_factor = Fp * fnp3 * d12inv;

  tmp0 = C4B_134[1] * s1[0] * s4[0] +
         C4B_134[4] * (s1[1] * s4[1] + s1[2] * s4[2]);
  tmp1 = tmp0 * (5.0 * r12[2] * r12[2] - 3.0 * d12 * d12) * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] - tmp2 * 6.0 * r12[2] * r12[0];
  f12[1] += tmp1 * r12[1] - tmp2 * 6.0 * r12[2] * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2 * (9.0 * r12[2] * r12[2] - 3.0 * d12 * d12);

  tmp0 = -C4B_134[0] * s1[1] * s4[0] +
         C4B_134[3] * (s1[1] * s4[3] + s1[2] * s4[4]) +
         C4B_134[5] * s1[0] * s4[1];
  tmp1 = tmp0 * (5.0 * r12[2] * r12[2] - d12 * d12) * r12[0] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * (4.0 * r12[2] * r12[2] - 3.0 * r12[0] * r12[0] - r12[1] * r12[1]);
  f12[1] += tmp1 * r12[1] - tmp2 * (2.0 * r12[0] * r12[1]);
  f12[2] += tmp1 * r12[2] + tmp2 * (8.0 * r12[0] * r12[2]);

  tmp0 = -C4B_134[0] * s4[0] * s1[2] +
         C4B_134[3] * (-s4[3] * s1[2] + s1[1] * s4[4]) +
         C4B_134[5] * s1[0] * s4[2];
  tmp1 = tmp0 * (5.0 * r12[2] * r12[2] - d12 * d12) * r12[1] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] - tmp2 * (2.0 * r12[0] * r12[1]);
  f12[1] += tmp1 * r12[1] + tmp2 * (4.0 * r12[2] * r12[2] - r12[0] * r12[0] - 3.0 * r12[1] * r12[1]);
  f12[2] += tmp1 * r12[2] + tmp2 * (8.0 * r12[1] * r12[2]);

  tmp0 = C4B_134[5] * (-s1[1] * s4[1] + s1[2] * s4[2]) +
         C4B_134[7] * s1[0] * s4[3] +
         C4B_134[9] * (s1[1] * s4[5] + s1[2] * s4[6]);
  tmp1 = tmp0 * (r12[0] * r12[0] - r12[1] * r12[1]) * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * (2.0 * r12[0] * r12[2]);
  f12[1] += tmp1 * r12[1] - tmp2 * (2.0 * r12[1] * r12[2]);
  f12[2] += tmp1 * r12[2] + tmp2 * (r12[0] * r12[0] - r12[1] * r12[1]);

  tmp0 = C4B_134[5] * (-s1[1] * s4[2] - s1[2] * s4[1]) +
         C4B_134[7] * s1[0] * s4[4] +
         C4B_134[9] * (s1[1] * s4[6] - s1[2] * s4[5]);
  tmp1 = tmp0 * (2.0 * r12[0] * r12[1] * r12[2]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * (2.0 * r12[1] * r12[2]);
  f12[1] += tmp1 * r12[1] + tmp2 * (2.0 * r12[0] * r12[2]);
  f12[2] += tmp1 * r12[2] + tmp2 * (2.0 * r12[0] * r12[1]);

  tmp0 = C4B_134[2] * (-s1[1] * s4[3] + s1[2] * s4[4]) +
         C4B_134[6] * (s1[1] * s4[7] + s1[2] * s4[8]) +
         C4B_134[8] * s1[0] * s4[5];
  tmp1 = tmp0 * (r12[0] * r12[0] - 3.0 * r12[1] * r12[1]) * r12[0] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * (3.0 * (r12[0] * r12[0] - r12[1] * r12[1]));
  f12[1] += tmp1 * r12[1] - tmp2 * (6.0 * r12[0] * r12[1]);
  f12[2] += tmp1 * r12[2];

  tmp0 = C4B_134[2] * (-s1[1] * s4[4] - s1[2] * s4[3]) +
         C4B_134[6] * (s1[1] * s4[8] - s1[2] * s4[7]) +
         C4B_134[8] * s1[0] * s4[6];
  tmp1 = tmp0 * (3.0 * r12[0] * r12[0] - r12[1] * r12[1]) * r12[1] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * (6.0 * r12[0] * r12[1]);
  f12[1] += tmp1 * r12[1] + tmp2 * (3.0 * (r12[0] * r12[0] - r12[1] * r12[1]));
  f12[2] += tmp1 * r12[2];

  // s4
  fn_factor = Fp * fn4;
  fnp_factor = Fp * fnp4 * d12inv;

  tmp0 = C4B_134[0] * (-s3[2] * s1[2] - s1[1] * s3[1]) +
         C4B_134[1] * s1[0] * s3[0];
  tmp1 = tmp0 * (35.0 * r12[2]*r12[2]*r12[2]*r12[2] - 30.0 * d12*d12 * r12[2]*r12[2] +
                 3.0 * d12*d12*d12*d12) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * 12.0 * r12[0] * (r12[0]*r12[0] + r12[1]*r12[1] - 4.0*r12[2]*r12[2]);
  f12[1] += tmp1 * r12[1] + tmp2 * 12.0 * r12[1] * (r12[0]*r12[0] + r12[1]*r12[1] - 4.0*r12[2]*r12[2]);
  f12[2] += tmp1 * r12[2] + tmp2 * 16.0 * r12[2] * (-3.0*r12[0]*r12[0] - 3.0*r12[1]*r12[1] + 2.0*r12[2]*r12[2]);

  tmp0 = C4B_134[4] * s1[1] * s3[0] +
         C4B_134[5] * (s1[0] * s3[1] - s1[1] * s3[3] - s1[2] * s3[4]);
  tmp1 = tmp0 * (7.0 * r12[2]*r12[2] - 3.0 * d12*d12) * r12[0] * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * r12[2] * (-9.0*r12[0]*r12[0] - 3.0*r12[1]*r12[1] + 4.0*r12[2]*r12[2]);
  f12[1] += tmp1 * r12[1] - tmp2 * 6.0 * r12[0] * r12[1] * r12[2];
  f12[2] += tmp1 * r12[2] - tmp2 * 3.0 * r12[0] * (r12[0]*r12[0] + r12[1]*r12[1] - 4.0*r12[2]*r12[2]);

  tmp0 = C4B_134[4] * s1[2] * s3[0] +
         C4B_134[5] * (s1[0] * s3[2] - s1[1] * s3[4] + s1[2] * s3[3]);
  tmp1 = tmp0 * (7.0 * r12[2]*r12[2] - 3.0 * d12*d12) * r12[1] * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] - tmp2 * 6.0 * r12[0] * r12[1] * r12[2];
  f12[1] += tmp1 * r12[1] + tmp2 * r12[2] * (-3.0*r12[0]*r12[0] - 9.0*r12[1]*r12[1] + 4.0*r12[2]*r12[2]);
  f12[2] += tmp1 * r12[2] - tmp2 * 3.0 * r12[1] * (r12[0]*r12[0] + r12[1]*r12[1] - 4.0*r12[2]*r12[2]);

  tmp0 = C4B_134[2] * (-s1[1] * s3[5] - s1[2] * s3[6]) +
         C4B_134[3] * (-s3[2] * s1[2] + s1[1] * s3[1]) +
         C4B_134[7] * s1[0] * s3[3];
  tmp1 = tmp0 * (7.0 * r12[2]*r12[2] - d12*d12) * (r12[0]*r12[0] - r12[1]*r12[1]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] - tmp2 * 4.0 * r12[0] * (r12[0]*r12[0] - 3.0*r12[2]*r12[2]);
  f12[1] += tmp1 * r12[1] + tmp2 * 4.0 * r12[1] * (r12[1]*r12[1] - 3.0*r12[2]*r12[2]);
  f12[2] += tmp1 * r12[2] + tmp2 * 12.0 * r12[2] * (r12[0]*r12[0] - r12[1]*r12[1]);

  tmp0 = C4B_134[2] * (-s1[1] * s3[6] + s1[2] * s3[5]) +
         C4B_134[3] * (s1[1] * s3[2] + s1[2] * s3[1]) +
         C4B_134[7] * s1[0] * s3[4];
  tmp1 = tmp0 * (7.0 * r12[2]*r12[2] - d12*d12) * 2.0 * r12[0] * r12[1] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] - tmp2 * 2.0 * r12[1] * (3.0*r12[0]*r12[0] + r12[1]*r12[1] - 6.0*r12[2]*r12[2]);
  f12[1] += tmp1 * r12[1] - tmp2 * 2.0 * r12[0] * (r12[0]*r12[0] + 3.0*r12[1]*r12[1] - 6.0*r12[2]*r12[2]);
  f12[2] += tmp1 * r12[2] + tmp2 * 24.0 * r12[0] * r12[1] * r12[2];

  tmp0 = C4B_134[8] * s1[0] * s3[5] +
         C4B_134[9] * (s1[1] * s3[3] - s1[2] * s3[4]);
  tmp1 = tmp0 * (r12[0]*r12[0] - 3.0*r12[1]*r12[1]) * r12[0] * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * 3.0 * r12[2] * (r12[0]*r12[0] - r12[1]*r12[1]);
  f12[1] += tmp1 * r12[1] - tmp2 * 6.0 * r12[0] * r12[1] * r12[2];
  f12[2] += tmp1 * r12[2] + tmp2 * r12[0] * (r12[0]*r12[0] - 3.0*r12[1]*r12[1]);

  tmp0 = C4B_134[8] * s1[0] * s3[6] +
         C4B_134[9] * (s1[1] * s3[4] + s1[2] * s3[3]);
  tmp1 = tmp0 * (3.0*r12[0]*r12[0] - r12[1]*r12[1]) * r12[1] * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * 6.0 * r12[0] * r12[1] * r12[2];
  f12[1] += tmp1 * r12[1] + tmp2 * 3.0 * r12[2] * (r12[0]*r12[0] - r12[1]*r12[1]);
  f12[2] += tmp1 * r12[2] + tmp2 * r12[1] * (3.0*r12[0]*r12[0] - r12[1]*r12[1]);

  tmp0 = C4B_134[6] * (s1[1] * s3[5] - s1[2] * s3[6]);
  tmp1 = tmp0 * (r12[0]*r12[0]*r12[0]*r12[0] - 6.0*r12[0]*r12[0]*r12[1]*r12[1] +
                 r12[1]*r12[1]*r12[1]*r12[1]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * 4.0 * r12[0] * (r12[0]*r12[0] - 3.0*r12[1]*r12[1]);
  f12[1] += tmp1 * r12[1] + tmp2 * 4.0 * r12[1] * (-3.0*r12[0]*r12[0] + r12[1]*r12[1]);
  f12[2] += tmp1 * r12[2];

  tmp0 = C4B_134[6] * (s1[1] * s3[6] + s1[2] * s3[5]);
  tmp1 = tmp0 * 4.0 * r12[0] * r12[1] * (r12[0]*r12[0] - r12[1]*r12[1]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * 4.0 * r12[1] * (3.0*r12[0]*r12[0] - r12[1]*r12[1]);
  f12[1] += tmp1 * r12[1] + tmp2 * 4.0 * r12[0] * (r12[0]*r12[0] - 3.0*r12[1]*r12[1]);
  f12[2] += tmp1 * r12[2];
}

template <int L>
void calculate_s_one(
  const int n, const int n_max_angular_plus_1, const double* Fp, const double* sum_fxyz, double* s)
{
  const int L_minus_1 = L - 1;
  const int L_twice_plus_1 = 2 * L + 1;
  const int L_square_minus_1 = L * L - 1;
  double Fp_factor = 2.0 * Fp[L_minus_1 * n_max_angular_plus_1 + n];
  s[0] = sum_fxyz[n * NUM_OF_ABC + L_square_minus_1] * C3B[L_square_minus_1] * Fp_factor;
  Fp_factor *= 2.0;
  for (int k = 1; k < L_twice_plus_1; ++k) {
    s[k] = sum_fxyz[n * NUM_OF_ABC + L_square_minus_1 + k] * C3B[L_square_minus_1 + k] * Fp_factor;
  }
}

template <int L>
void accumulate_f12_one(
  const double d12inv,
  const double fn,
  const double fnp,
  const double* s,
  const double* r12,
  double* f12)
{
  const double dx[3] = {
    (1.0 - r12[0] * r12[0]) * d12inv, -r12[0] * r12[1] * d12inv, -r12[0] * r12[2] * d12inv};
  const double dy[3] = {
    -r12[0] * r12[1] * d12inv, (1.0 - r12[1] * r12[1]) * d12inv, -r12[1] * r12[2] * d12inv};
  const double dz[3] = {
    -r12[0] * r12[2] * d12inv, -r12[1] * r12[2] * d12inv, (1.0 - r12[2] * r12[2]) * d12inv};

  double z_pow[L + 1] = {1.0};
  for (int n = 1; n <= L; ++n) {
    z_pow[n] = r12[2] * z_pow[n - 1];
  }

  double real_part = 1.0;
  double imag_part = 0.0;
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    double z_factor = 0.0;
    double dz_factor = 0.0;
    for (int n2 = n2_start; n2 <= L - n1; n2 += 2) {
      const double coefficient = z_coefficient<L>(n1, n2);
      z_factor += coefficient * z_pow[n2];
      if (n2 > 0) {
        dz_factor += coefficient * n2 * z_pow[n2 - 1];
      }
    }
    if (n1 == 0) {
      for (int d = 0; d < 3; ++d) {
        f12[d] += s[0] * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
      }
    } else {
      double real_part_n1 = n1 * real_part;
      double imag_part_n1 = n1 * imag_part;
      for (int d = 0; d < 3; ++d) {
        double real_part_dx = dx[d];
        double imag_part_dy = dy[d];
        complex_product(real_part_n1, imag_part_n1, real_part_dx, imag_part_dy);
        f12[d] += (s[2 * n1 - 1] * real_part_dx + s[2 * n1 - 0] * imag_part_dy) * z_factor * fn;
      }
      complex_product(r12[0], r12[1], real_part, imag_part);
      const double xy_temp = s[2 * n1 - 1] * real_part + s[2 * n1 - 0] * imag_part;
      for (int d = 0; d < 3; ++d) {
        f12[d] += xy_temp * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
      }
    }
  }
}

void accumulate_f12(
  const int L_max,
  const int num_L,
  const int n,
  const int n_max_angular_plus_1,
  const double d12,
  const double* r12,
  double fn,
  double fnp,
  const double* Fp,
  const double* sum_fxyz,
  double* f12)
{
  const double fn_original = fn;
  const double fnp_original = fnp;
  const double d12inv = 1.0 / d12;
  const double r12unit[3] = {r12[0] * d12inv, r12[1] * d12inv, r12[2] * d12inv};

  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  if (num_L >= L_max + 2) {
    double s1[3] = {
      sum_fxyz[n * NUM_OF_ABC + 0], sum_fxyz[n * NUM_OF_ABC + 1], sum_fxyz[n * NUM_OF_ABC + 2]};
    get_f12_5body(d12, d12inv, fn, fnp, Fp[(L_max + 1) * n_max_angular_plus_1 + n], s1, r12, f12);
  }

  if (L_max >= 1) {
    double s1[3];
    calculate_s_one<1>(n, n_max_angular_plus_1, Fp, sum_fxyz, s1);
    accumulate_f12_one<1>(d12inv, fn_original, fnp_original, s1, r12unit, f12);
  }

  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  if (num_L >= L_max + 1) {
    double s2[5] = {
      sum_fxyz[n * NUM_OF_ABC + 3], sum_fxyz[n * NUM_OF_ABC + 4], sum_fxyz[n * NUM_OF_ABC + 5],
      sum_fxyz[n * NUM_OF_ABC + 6], sum_fxyz[n * NUM_OF_ABC + 7]};
    get_f12_4body(d12, d12inv, fn, fnp, Fp[L_max * n_max_angular_plus_1 + n], s2, r12, f12);
  }

  if (L_max >= 2) {
    double s2[5];
    calculate_s_one<2>(n, n_max_angular_plus_1, Fp, sum_fxyz, s2);
    accumulate_f12_one<2>(d12inv, fn_original, fnp_original, s2, r12unit, f12);
  }

  if (L_max >= 3) {
    double s3[7];
    calculate_s_one<3>(n, n_max_angular_plus_1, Fp, sum_fxyz, s3);
    accumulate_f12_one<3>(d12inv, fn_original, fnp_original, s3, r12unit, f12);
  }

  if (L_max >= 4) {
    double s4[9];
    calculate_s_one<4>(n, n_max_angular_plus_1, Fp, sum_fxyz, s4);
    accumulate_f12_one<4>(d12inv, fn_original, fnp_original, s4, r12unit, f12);
  }

  if (L_max >= 5) {
    double s5[11];
    calculate_s_one<5>(n, n_max_angular_plus_1, Fp, sum_fxyz, s5);
    accumulate_f12_one<5>(d12inv, fn_original, fnp_original, s5, r12unit, f12);
  }

  if (L_max >= 6) {
    double s6[13];
    calculate_s_one<6>(n, n_max_angular_plus_1, Fp, sum_fxyz, s6);
    accumulate_f12_one<6>(d12inv, fn_original, fnp_original, s6, r12unit, f12);
  }

  if (L_max >= 7) {
    double s7[15];
    calculate_s_one<7>(n, n_max_angular_plus_1, Fp, sum_fxyz, s7);
    accumulate_f12_one<7>(d12inv, fn_original, fnp_original, s7, r12unit, f12);
  }

  if (L_max >= 8) {
    double s8[17];
    calculate_s_one<8>(n, n_max_angular_plus_1, Fp, sum_fxyz, s8);
    accumulate_f12_one<8>(d12inv, fn_original, fnp_original, s8, r12unit, f12);
  }
}

void accumulate_f12_extra_terms(
  const int L_max,
  const int has_q_222,
  const int has_q_1111,
  const int has_q_112,
  const int has_q_123,
  const int has_q_233,
  const int has_q_134,
  const int num_L,
  const int n,
  const int n_max_angular_plus_1,
  const double d12,
  const double d12inv,
  const double* r12,
  double fn,
  double fnp,
  const double* Fp,
  const double* sum_fxyz,
  double* f12)
{
  if (num_L <= L_max) {
    return;
  }

  int L_index = L_max;
  double s1[3] = {
    sum_fxyz[n * NUM_OF_ABC + 0], sum_fxyz[n * NUM_OF_ABC + 1], sum_fxyz[n * NUM_OF_ABC + 2]};
  double s2[5] = {
    sum_fxyz[n * NUM_OF_ABC + 3], sum_fxyz[n * NUM_OF_ABC + 4], sum_fxyz[n * NUM_OF_ABC + 5],
    sum_fxyz[n * NUM_OF_ABC + 6], sum_fxyz[n * NUM_OF_ABC + 7]};

  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double fnp2 = fnp * d12inv - fn * d12inv * d12inv;
  double fn2 = fn * d12inv;

  if (has_q_222) {
    get_f12_4body(d12, d12inv, fn2, fnp2, Fp[(L_index++) * n_max_angular_plus_1 + n], s2, r12, f12);
  }
  if (has_q_1111) {
    get_f12_5body(d12, d12inv, fn, fnp, Fp[(L_index++) * n_max_angular_plus_1 + n], s1, r12, f12);
  }
  if (has_q_112) {
    get_f12_4body_2(
      d12, d12inv, fn, fnp, fn2, fnp2, Fp[(L_index++) * n_max_angular_plus_1 + n], s1, s2, r12, f12);
  }

  if (has_q_123 || has_q_233) {
    double fnp3 = fnp2 * d12inv - fn2 * d12inv * d12inv;
    double fn3 = fn2 * d12inv;
    double s3[7] = {
      sum_fxyz[n * NUM_OF_ABC + 8],  sum_fxyz[n * NUM_OF_ABC + 9],
      sum_fxyz[n * NUM_OF_ABC + 10], sum_fxyz[n * NUM_OF_ABC + 11],
      sum_fxyz[n * NUM_OF_ABC + 12], sum_fxyz[n * NUM_OF_ABC + 13],
      sum_fxyz[n * NUM_OF_ABC + 14]};
    if (has_q_123) {
      get_f12_4body_123(
        d12, d12inv, fn, fnp, fn2, fnp2, fn3, fnp3,
        Fp[(L_index++) * n_max_angular_plus_1 + n], s1, s2, s3, r12, f12);
    }
    if (has_q_233) {
      get_f12_4body_233(
        d12, d12inv, fn2, fnp2, fn3, fnp3,
        Fp[(L_index++) * n_max_angular_plus_1 + n], s2, s3, r12, f12);
    }
  }
  if (has_q_134) {
    double fnp3 = fnp2 * d12inv - fn2 * d12inv * d12inv;
    double fn3 = fn2 * d12inv;
    double s3[7] = {
      sum_fxyz[n * NUM_OF_ABC + 8],  sum_fxyz[n * NUM_OF_ABC + 9],
      sum_fxyz[n * NUM_OF_ABC + 10], sum_fxyz[n * NUM_OF_ABC + 11],
      sum_fxyz[n * NUM_OF_ABC + 12], sum_fxyz[n * NUM_OF_ABC + 13],
      sum_fxyz[n * NUM_OF_ABC + 14]};
    double fnp4 = fnp3 * d12inv - fn3 * d12inv * d12inv;
    double fn4 = fn3 * d12inv;
    double s4[9] = {
      sum_fxyz[n * NUM_OF_ABC + 15], sum_fxyz[n * NUM_OF_ABC + 16],
      sum_fxyz[n * NUM_OF_ABC + 17], sum_fxyz[n * NUM_OF_ABC + 18],
      sum_fxyz[n * NUM_OF_ABC + 19], sum_fxyz[n * NUM_OF_ABC + 20],
      sum_fxyz[n * NUM_OF_ABC + 21], sum_fxyz[n * NUM_OF_ABC + 22],
      sum_fxyz[n * NUM_OF_ABC + 23]};
    get_f12_4body_134(
      d12, d12inv, fn, fnp, fn3, fnp3, fn4, fnp4,
      Fp[(L_index++) * n_max_angular_plus_1 + n], s1, s3, s4, r12, f12);
  }
}

void accumulate_f12_q222_q1111_all_n(
  const int has_q_222,
  const int has_q_1111,
  const int n_max_angular_plus_1,
  const double d12,
  const double d12inv,
  const double* NEP_RESTRICT r12,
  const double* NEP_RESTRICT fn,
  const double* NEP_RESTRICT fnp,
  const double* NEP_RESTRICT q222_derivatives,
  const double* NEP_RESTRICT q1111_derivatives,
  double* NEP_RESTRICT f12)
{
  const double d12inv2 = d12inv * d12inv;
  const double x = r12[0];
  const double y = r12[1];
  const double z = r12[2];
  const double y20 = 3.0 * z * z - d12 * d12;
  const double xz = x * z;
  const double yz = y * z;
  const double xx_minus_yy = x * x - y * y;
  const double two_xy = 2.0 * x * y;
  double f12_x = f12[0];
  double f12_y = f12[1];
  double f12_z = f12[2];

  for (int n = 0; n < n_max_angular_plus_1; ++n) {
    const double fn1 = fn[n] * d12inv;
    const double fnp1 = fnp[n] * d12inv - fn[n] * d12inv2;

    if (has_q_222) {
      const double* q222 = q222_derivatives + n * 5;
      const double fn2 = fn1 * d12inv;
      const double fnp2 = fnp1 * d12inv - fn1 * d12inv2;
      const double fnp2_factor = fnp2 * d12inv;

      double tmp1 = q222[0] * y20 * fnp2_factor;
      double tmp2 = q222[0] * fn2;
      f12_x += tmp1 * x - tmp2 * 2.0 * x;
      f12_y += tmp1 * y - tmp2 * 2.0 * y;
      f12_z += tmp1 * z + tmp2 * 4.0 * z;

      tmp1 = q222[1] * xz * fnp2_factor;
      tmp2 = q222[1] * fn2;
      f12_x += tmp1 * x + tmp2 * z;
      f12_y += tmp1 * y;
      f12_z += tmp1 * z + tmp2 * x;

      tmp1 = q222[2] * yz * fnp2_factor;
      tmp2 = q222[2] * fn2;
      f12_x += tmp1 * x;
      f12_y += tmp1 * y + tmp2 * z;
      f12_z += tmp1 * z + tmp2 * y;

      tmp1 = q222[3] * xx_minus_yy * fnp2_factor;
      tmp2 = q222[3] * fn2;
      f12_x += tmp1 * x + tmp2 * 2.0 * x;
      f12_y += tmp1 * y - tmp2 * 2.0 * y;
      f12_z += tmp1 * z;

      tmp1 = q222[4] * two_xy * fnp2_factor;
      tmp2 = q222[4] * fn2;
      f12_x += tmp1 * x + tmp2 * 2.0 * y;
      f12_y += tmp1 * y + tmp2 * 2.0 * x;
      f12_z += tmp1 * z;
    }

    if (has_q_1111) {
      const double* q1111 = q1111_derivatives + n * 3;
      const double fnp1_factor = fnp1 * d12inv;

      double tmp1 = q1111[0] * z * fnp1_factor;
      double tmp2 = q1111[0] * fn1;
      f12_x += tmp1 * x;
      f12_y += tmp1 * y;
      f12_z += tmp1 * z + tmp2;

      tmp1 = q1111[1] * x * fnp1_factor;
      tmp2 = q1111[1] * fn1;
      f12_x += tmp1 * x + tmp2;
      f12_y += tmp1 * y;
      f12_z += tmp1 * z;

      tmp1 = q1111[2] * y * fnp1_factor;
      tmp2 = q1111[2] * fn1;
      f12_x += tmp1 * x;
      f12_y += tmp1 * y + tmp2;
      f12_z += tmp1 * z;
    }
  }
  f12[0] = f12_x;
  f12[1] = f12_y;
  f12[2] = f12_z;
}

void accumulate_f12_q222_q1111_contracted_all_n(
  const int has_q_222,
  const int has_q_1111,
  const int n_max_angular_plus_1,
  const double d12,
  const double d12inv,
  const double* NEP_RESTRICT r12,
  const double* NEP_RESTRICT fn,
  const double* NEP_RESTRICT fnp,
  const double* NEP_RESTRICT q222_derivatives,
  const double* NEP_RESTRICT q1111_derivatives,
  double* NEP_RESTRICT f12)
{
  const double d12inv2 = d12inv * d12inv;
  const double x = r12[0];
  const double y = r12[1];
  const double z = r12[2];
  const double y20 = 3.0 * z * z - d12 * d12;
  const double xz = x * z;
  const double yz = y * z;
  const double xx_minus_yy = x * x - y * y;
  const double two_xy = 2.0 * x * y;
  double f12_x = f12[0];
  double f12_y = f12[1];
  double f12_z = f12[2];

  double q222_fn2[5] = {0.0};
  double q222_fnp2_factor[5] = {0.0};
  double q1111_fn1[3] = {0.0};
  double q1111_fnp1_factor[3] = {0.0};

  for (int n = 0; n < n_max_angular_plus_1; ++n) {
    const double fn1 = fn[n] * d12inv;
    const double fnp1 = fnp[n] * d12inv - fn[n] * d12inv2;

    if (has_q_222) {
      const double* q222 = q222_derivatives + n * 5;
      const double fn2 = fn1 * d12inv;
      const double fnp2 = fnp1 * d12inv - fn1 * d12inv2;
      const double fnp2_factor = fnp2 * d12inv;
      for (int k = 0; k < 5; ++k) {
        q222_fn2[k] += q222[k] * fn2;
        q222_fnp2_factor[k] += q222[k] * fnp2_factor;
      }
    }

    if (has_q_1111) {
      const double* q1111 = q1111_derivatives + n * 3;
      const double fnp1_factor = fnp1 * d12inv;
      for (int k = 0; k < 3; ++k) {
        q1111_fn1[k] += q1111[k] * fn1;
        q1111_fnp1_factor[k] += q1111[k] * fnp1_factor;
      }
    }
  }

  if (has_q_222) {
    double tmp1 = q222_fnp2_factor[0] * y20;
    double tmp2 = q222_fn2[0];
    f12_x += tmp1 * x - tmp2 * 2.0 * x;
    f12_y += tmp1 * y - tmp2 * 2.0 * y;
    f12_z += tmp1 * z + tmp2 * 4.0 * z;

    tmp1 = q222_fnp2_factor[1] * xz;
    tmp2 = q222_fn2[1];
    f12_x += tmp1 * x + tmp2 * z;
    f12_y += tmp1 * y;
    f12_z += tmp1 * z + tmp2 * x;

    tmp1 = q222_fnp2_factor[2] * yz;
    tmp2 = q222_fn2[2];
    f12_x += tmp1 * x;
    f12_y += tmp1 * y + tmp2 * z;
    f12_z += tmp1 * z + tmp2 * y;

    tmp1 = q222_fnp2_factor[3] * xx_minus_yy;
    tmp2 = q222_fn2[3];
    f12_x += tmp1 * x + tmp2 * 2.0 * x;
    f12_y += tmp1 * y - tmp2 * 2.0 * y;
    f12_z += tmp1 * z;

    tmp1 = q222_fnp2_factor[4] * two_xy;
    tmp2 = q222_fn2[4];
    f12_x += tmp1 * x + tmp2 * 2.0 * y;
    f12_y += tmp1 * y + tmp2 * 2.0 * x;
    f12_z += tmp1 * z;
  }

  if (has_q_1111) {
    double tmp1 = q1111_fnp1_factor[0] * z;
    double tmp2 = q1111_fn1[0];
    f12_x += tmp1 * x;
    f12_y += tmp1 * y;
    f12_z += tmp1 * z + tmp2;

    tmp1 = q1111_fnp1_factor[1] * x;
    tmp2 = q1111_fn1[1];
    f12_x += tmp1 * x + tmp2;
    f12_y += tmp1 * y;
    f12_z += tmp1 * z;

    tmp1 = q1111_fnp1_factor[2] * y;
    tmp2 = q1111_fn1[2];
    f12_x += tmp1 * x;
    f12_y += tmp1 * y + tmp2;
    f12_z += tmp1 * z;
  }

  f12[0] = f12_x;
  f12[1] = f12_y;
  f12[2] = f12_z;
}

void accumulate_f12_from_contracted_l1(
  const double* NEP_RESTRICT contracted_fn,
  const double* NEP_RESTRICT contracted_fnp_factor,
  const double* NEP_RESTRICT r12,
  double* NEP_RESTRICT f12)
{
  double f12_x = f12[0];
  double f12_y = f12[1];
  double f12_z = f12[2];

  double tmp1 = contracted_fnp_factor[0] * r12[2];
  double tmp2 = contracted_fn[0];
  f12_x += tmp1 * r12[0];
  f12_y += tmp1 * r12[1];
  f12_z += tmp1 * r12[2] + tmp2;

  tmp1 = contracted_fnp_factor[1] * r12[0];
  tmp2 = contracted_fn[1];
  f12_x += tmp1 * r12[0] + tmp2;
  f12_y += tmp1 * r12[1];
  f12_z += tmp1 * r12[2];

  tmp1 = contracted_fnp_factor[2] * r12[1];
  tmp2 = contracted_fn[2];
  f12_x += tmp1 * r12[0];
  f12_y += tmp1 * r12[1] + tmp2;
  f12_z += tmp1 * r12[2];

  f12[0] = f12_x;
  f12[1] = f12_y;
  f12[2] = f12_z;
}

void accumulate_f12_from_contracted_l2(
  const double d12,
  const double* NEP_RESTRICT contracted_fn,
  const double* NEP_RESTRICT contracted_fnp_factor,
  const double* NEP_RESTRICT r12,
  double* NEP_RESTRICT f12)
{
  const double y20 = 3.0 * r12[2] * r12[2] - d12 * d12;
  const double xz = r12[0] * r12[2];
  const double yz = r12[1] * r12[2];
  const double xx_minus_yy = r12[0] * r12[0] - r12[1] * r12[1];
  const double two_xy = 2.0 * r12[0] * r12[1];
  double f12_x = f12[0];
  double f12_y = f12[1];
  double f12_z = f12[2];

  double tmp1 = contracted_fnp_factor[0] * y20;
  double tmp2 = contracted_fn[0];
  f12_x += tmp1 * r12[0] - tmp2 * 2.0 * r12[0];
  f12_y += tmp1 * r12[1] - tmp2 * 2.0 * r12[1];
  f12_z += tmp1 * r12[2] + tmp2 * 4.0 * r12[2];

  tmp1 = contracted_fnp_factor[1] * xz;
  tmp2 = contracted_fn[1];
  f12_x += tmp1 * r12[0] + tmp2 * r12[2];
  f12_y += tmp1 * r12[1];
  f12_z += tmp1 * r12[2] + tmp2 * r12[0];

  tmp1 = contracted_fnp_factor[2] * yz;
  tmp2 = contracted_fn[2];
  f12_x += tmp1 * r12[0];
  f12_y += tmp1 * r12[1] + tmp2 * r12[2];
  f12_z += tmp1 * r12[2] + tmp2 * r12[1];

  tmp1 = contracted_fnp_factor[3] * xx_minus_yy;
  tmp2 = contracted_fn[3];
  f12_x += tmp1 * r12[0] + tmp2 * 2.0 * r12[0];
  f12_y += tmp1 * r12[1] - tmp2 * 2.0 * r12[1];
  f12_z += tmp1 * r12[2];

  tmp1 = contracted_fnp_factor[4] * two_xy;
  tmp2 = contracted_fn[4];
  f12_x += tmp1 * r12[0] + tmp2 * 2.0 * r12[1];
  f12_y += tmp1 * r12[1] + tmp2 * 2.0 * r12[0];
  f12_z += tmp1 * r12[2];

  f12[0] = f12_x;
  f12[1] = f12_y;
  f12[2] = f12_z;
}

void accumulate_f12_from_contracted_l3(
  const double d12,
  const double* NEP_RESTRICT contracted_fn,
  const double* NEP_RESTRICT contracted_fnp_factor,
  const double* NEP_RESTRICT r12,
  double* NEP_RESTRICT f12)
{
  const double x = r12[0];
  const double y = r12[1];
  const double z = r12[2];
  const double d12sq = d12 * d12;
  double f12_x = f12[0];
  double f12_y = f12[1];
  double f12_z = f12[2];

  double tmp1 = contracted_fnp_factor[0] * (5.0 * z * z - 3.0 * d12sq) * z;
  double tmp2 = contracted_fn[0];
  f12_x += tmp1 * x - tmp2 * 6.0 * z * x;
  f12_y += tmp1 * y - tmp2 * 6.0 * z * y;
  f12_z += tmp1 * z + tmp2 * (9.0 * z * z - 3.0 * d12sq);

  tmp1 = contracted_fnp_factor[1] * (5.0 * z * z - d12sq) * x;
  tmp2 = contracted_fn[1];
  f12_x += tmp1 * x + tmp2 * (4.0 * z * z - 3.0 * x * x - y * y);
  f12_y += tmp1 * y - tmp2 * (2.0 * x * y);
  f12_z += tmp1 * z + tmp2 * (8.0 * x * z);

  tmp1 = contracted_fnp_factor[2] * (5.0 * z * z - d12sq) * y;
  tmp2 = contracted_fn[2];
  f12_x += tmp1 * x - tmp2 * (2.0 * x * y);
  f12_y += tmp1 * y + tmp2 * (4.0 * z * z - x * x - 3.0 * y * y);
  f12_z += tmp1 * z + tmp2 * (8.0 * y * z);

  tmp1 = contracted_fnp_factor[3] * (x * x - y * y) * z;
  tmp2 = contracted_fn[3];
  f12_x += tmp1 * x + tmp2 * (2.0 * x * z);
  f12_y += tmp1 * y - tmp2 * (2.0 * y * z);
  f12_z += tmp1 * z + tmp2 * (x * x - y * y);

  tmp1 = contracted_fnp_factor[4] * (2.0 * x * y * z);
  tmp2 = contracted_fn[4];
  f12_x += tmp1 * x + tmp2 * (2.0 * y * z);
  f12_y += tmp1 * y + tmp2 * (2.0 * x * z);
  f12_z += tmp1 * z + tmp2 * (2.0 * x * y);

  tmp1 = contracted_fnp_factor[5] * (x * x - 3.0 * y * y) * x;
  tmp2 = contracted_fn[5];
  f12_x += tmp1 * x + tmp2 * (3.0 * (x * x - y * y));
  f12_y += tmp1 * y - tmp2 * (6.0 * x * y);
  f12_z += tmp1 * z;

  tmp1 = contracted_fnp_factor[6] * (3.0 * x * x - y * y) * y;
  tmp2 = contracted_fn[6];
  f12_x += tmp1 * x + tmp2 * (6.0 * x * y);
  f12_y += tmp1 * y + tmp2 * (3.0 * (x * x - y * y));
  f12_z += tmp1 * z;

  f12[0] = f12_x;
  f12[1] = f12_y;
  f12[2] = f12_z;
}

void accumulate_f12_from_contracted_l4(
  const double d12,
  const double* NEP_RESTRICT contracted_fn,
  const double* NEP_RESTRICT contracted_fnp_factor,
  const double* NEP_RESTRICT r12,
  double* NEP_RESTRICT f12)
{
  const double x = r12[0];
  const double y = r12[1];
  const double z = r12[2];
  const double x2 = x * x;
  const double y2 = y * y;
  const double z2 = z * z;
  const double d2 = d12 * d12;
  const double d4 = d2 * d2;
  double f12_x = f12[0];
  double f12_y = f12[1];
  double f12_z = f12[2];

  double tmp1 = contracted_fnp_factor[0] * (35.0 * z2 * z2 - 30.0 * d2 * z2 + 3.0 * d4);
  double tmp2 = contracted_fn[0];
  f12_x += tmp1 * x + tmp2 * 12.0 * x * (x2 + y2 - 4.0 * z2);
  f12_y += tmp1 * y + tmp2 * 12.0 * y * (x2 + y2 - 4.0 * z2);
  f12_z += tmp1 * z + tmp2 * 16.0 * z * (-3.0 * x2 - 3.0 * y2 + 2.0 * z2);

  tmp1 = contracted_fnp_factor[1] * (7.0 * z2 - 3.0 * d2) * x * z;
  tmp2 = contracted_fn[1];
  f12_x += tmp1 * x + tmp2 * z * (-9.0 * x2 - 3.0 * y2 + 4.0 * z2);
  f12_y += tmp1 * y - tmp2 * 6.0 * x * y * z;
  f12_z += tmp1 * z - tmp2 * 3.0 * x * (x2 + y2 - 4.0 * z2);

  tmp1 = contracted_fnp_factor[2] * (7.0 * z2 - 3.0 * d2) * y * z;
  tmp2 = contracted_fn[2];
  f12_x += tmp1 * x - tmp2 * 6.0 * x * y * z;
  f12_y += tmp1 * y + tmp2 * z * (-3.0 * x2 - 9.0 * y2 + 4.0 * z2);
  f12_z += tmp1 * z - tmp2 * 3.0 * y * (x2 + y2 - 4.0 * z2);

  tmp1 = contracted_fnp_factor[3] * (7.0 * z2 - d2) * (x2 - y2);
  tmp2 = contracted_fn[3];
  f12_x += tmp1 * x - tmp2 * 4.0 * x * (x2 - 3.0 * z2);
  f12_y += tmp1 * y + tmp2 * 4.0 * y * (y2 - 3.0 * z2);
  f12_z += tmp1 * z + tmp2 * 12.0 * z * (x2 - y2);

  tmp1 = contracted_fnp_factor[4] * (7.0 * z2 - d2) * 2.0 * x * y;
  tmp2 = contracted_fn[4];
  f12_x += tmp1 * x - tmp2 * 2.0 * y * (3.0 * x2 + y2 - 6.0 * z2);
  f12_y += tmp1 * y - tmp2 * 2.0 * x * (x2 + 3.0 * y2 - 6.0 * z2);
  f12_z += tmp1 * z + tmp2 * 24.0 * x * y * z;

  tmp1 = contracted_fnp_factor[5] * (x2 - 3.0 * y2) * x * z;
  tmp2 = contracted_fn[5];
  f12_x += tmp1 * x + tmp2 * 3.0 * z * (x2 - y2);
  f12_y += tmp1 * y - tmp2 * 6.0 * x * y * z;
  f12_z += tmp1 * z + tmp2 * x * (x2 - 3.0 * y2);

  tmp1 = contracted_fnp_factor[6] * (3.0 * x2 - y2) * y * z;
  tmp2 = contracted_fn[6];
  f12_x += tmp1 * x + tmp2 * 6.0 * x * y * z;
  f12_y += tmp1 * y + tmp2 * 3.0 * z * (x2 - y2);
  f12_z += tmp1 * z + tmp2 * y * (3.0 * x2 - y2);

  tmp1 = contracted_fnp_factor[7] * (x2 * x2 - 6.0 * x2 * y2 + y2 * y2);
  tmp2 = contracted_fn[7];
  f12_x += tmp1 * x + tmp2 * 4.0 * x * (x2 - 3.0 * y2);
  f12_y += tmp1 * y + tmp2 * 4.0 * y * (-3.0 * x2 + y2);
  f12_z += tmp1 * z;

  tmp1 = contracted_fnp_factor[8] * 4.0 * x * y * (x2 - y2);
  tmp2 = contracted_fn[8];
  f12_x += tmp1 * x + tmp2 * 4.0 * y * (3.0 * x2 - y2);
  f12_y += tmp1 * y + tmp2 * 4.0 * x * (x2 - 3.0 * y2);
  f12_z += tmp1 * z;

  f12[0] = f12_x;
  f12[1] = f12_y;
  f12[2] = f12_z;
}

void scale_f12_q222_q1111_all_n(
  const int L_max,
  const int has_q_222,
  const int has_q_1111,
  const int n_max_angular_plus_1,
  const double* NEP_RESTRICT Fp,
  const double* NEP_RESTRICT sum_fxyz,
  double* NEP_RESTRICT q222_derivatives,
  double* NEP_RESTRICT q1111_derivatives)
{
  int L_index = L_max;
  const int q222_L_index = has_q_222 ? L_index++ : -1;
  const int q1111_L_index = has_q_1111 ? L_index++ : -1;

  for (int n = 0; n < n_max_angular_plus_1; ++n) {
    const int offset = n * NUM_OF_ABC;
    if (has_q_222) {
      const double fp = Fp[q222_L_index * n_max_angular_plus_1 + n];
      const double s0 = sum_fxyz[offset + 3];
      const double s1 = sum_fxyz[offset + 4];
      const double s2 = sum_fxyz[offset + 5];
      const double s3 = sum_fxyz[offset + 6];
      const double s4 = sum_fxyz[offset + 7];
      double* q222 = q222_derivatives + n * 5;
      q222[0] = (C4B[0] * 3.0 * s0 * s0 + C4B[1] * (s1 * s1 + s2 * s2) +
                 C4B[2] * (s3 * s3 + s4 * s4)) *
                fp;
      q222[1] = (C4B[1] * s0 * s1 * 2.0 - C4B[3] * s3 * s1 * 2.0 + C4B[4] * s2 * s4) *
                fp;
      q222[2] = (C4B[1] * s0 * s2 * 2.0 + C4B[3] * s3 * s2 * 2.0 + C4B[4] * s1 * s4) *
                fp;
      q222[3] = (C4B[2] * s0 * s3 * 2.0 + C4B[3] * (s2 * s2 - s1 * s1)) * fp;
      q222[4] = (C4B[2] * s0 * s4 * 2.0 + C4B[4] * s1 * s2) * fp;
    }

    if (has_q_1111) {
      const double fp = Fp[q1111_L_index * n_max_angular_plus_1 + n];
      const double s0 = sum_fxyz[offset + 0];
      const double s1 = sum_fxyz[offset + 1];
      const double s2 = sum_fxyz[offset + 2];
      const double s1_sq_plus_s2_sq = s1 * s1 + s2 * s2;
      double* q1111 = q1111_derivatives + n * 3;
      q1111[0] =
        (C5B[0] * 4.0 * s0 * s0 * s0 + C5B[1] * s1_sq_plus_s2_sq * 2.0 * s0) * fp;
      q1111[1] =
        (C5B[1] * s0 * s0 * s1 * 2.0 + C5B[2] * s1_sq_plus_s2_sq * s1 * 4.0) * fp;
      q1111[2] =
        (C5B[1] * s0 * s0 * s2 * 2.0 + C5B[2] * s1_sq_plus_s2_sq * s2 * 4.0) * fp;
    }
  }
}

void scale_sum_fxyz_3body_all_n(
  const int L_max,
  const int n_max_angular_plus_1,
  const double* NEP_RESTRICT Fp,
  const double* NEP_RESTRICT sum_fxyz,
  double* NEP_RESTRICT scaled_sum_fxyz)
{
  const int L_stop = L_max < 8 ? L_max : 8;
  for (int L = 1; L <= L_stop; ++L) {
    const int s_index = L * L - 1;
    for (int n = 0; n < n_max_angular_plus_1; ++n) {
      const int offset = n * NUM_OF_ABC;
      const double fp = Fp[(L - 1) * n_max_angular_plus_1 + n];
      scaled_sum_fxyz[offset + s_index] =
        sum_fxyz[offset + s_index] * C3B[s_index] * 2.0 * fp;
      for (int n1 = 1; n1 <= L; ++n1) {
        const int sa_index = s_index + 2 * n1 - 1;
        const int sb_index = s_index + 2 * n1;
        scaled_sum_fxyz[offset + sa_index] =
          sum_fxyz[offset + sa_index] * C3B[sa_index] * 4.0 * fp;
        scaled_sum_fxyz[offset + sb_index] =
          sum_fxyz[offset + sb_index] * C3B[sb_index] * 4.0 * fp;
      }
    }
  }
}

void accumulate_f12(
  const int L_max,
  const int has_q_222,
  const int has_q_1111,
  const int has_q_112,
  const int has_q_123,
  const int has_q_233,
  const int has_q_134,
  const int num_L,
  const int n,
  const int n_max_angular_plus_1,
  const double d12,
  const double* r12,
  double fn,
  double fnp,
  const double* Fp,
  const double* sum_fxyz,
  double* f12)
{
  const double fn_original = fn;
  const double fnp_original = fnp;
  const double d12inv = 1.0 / d12;
  const double r12unit[3] = {r12[0] * d12inv, r12[1] * d12inv, r12[2] * d12inv};

  if (L_max >= 1) {
    double s1[3];
    calculate_s_one<1>(n, n_max_angular_plus_1, Fp, sum_fxyz, s1);
    accumulate_f12_one<1>(d12inv, fn_original, fnp_original, s1, r12unit, f12);
  }
  if (L_max >= 2) {
    double s2[5];
    calculate_s_one<2>(n, n_max_angular_plus_1, Fp, sum_fxyz, s2);
    accumulate_f12_one<2>(d12inv, fn_original, fnp_original, s2, r12unit, f12);
  }
  if (L_max >= 3) {
    double s3[7];
    calculate_s_one<3>(n, n_max_angular_plus_1, Fp, sum_fxyz, s3);
    accumulate_f12_one<3>(d12inv, fn_original, fnp_original, s3, r12unit, f12);
  }
  if (L_max >= 4) {
    double s4[9];
    calculate_s_one<4>(n, n_max_angular_plus_1, Fp, sum_fxyz, s4);
    accumulate_f12_one<4>(d12inv, fn_original, fnp_original, s4, r12unit, f12);
  }
  if (L_max >= 5) {
    double s5[11];
    calculate_s_one<5>(n, n_max_angular_plus_1, Fp, sum_fxyz, s5);
    accumulate_f12_one<5>(d12inv, fn_original, fnp_original, s5, r12unit, f12);
  }
  if (L_max >= 6) {
    double s6[13];
    calculate_s_one<6>(n, n_max_angular_plus_1, Fp, sum_fxyz, s6);
    accumulate_f12_one<6>(d12inv, fn_original, fnp_original, s6, r12unit, f12);
  }
  if (L_max >= 7) {
    double s7[15];
    calculate_s_one<7>(n, n_max_angular_plus_1, Fp, sum_fxyz, s7);
    accumulate_f12_one<7>(d12inv, fn_original, fnp_original, s7, r12unit, f12);
  }
  if (L_max >= 8) {
    double s8[17];
    calculate_s_one<8>(n, n_max_angular_plus_1, Fp, sum_fxyz, s8);
    accumulate_f12_one<8>(d12inv, fn_original, fnp_original, s8, r12unit, f12);
  }

  accumulate_f12_extra_terms(
    L_max, has_q_222, has_q_1111, has_q_112, has_q_123, has_q_233, has_q_134, num_L, n,
    n_max_angular_plus_1, d12, d12inv, r12, fn_original, fnp_original, Fp, sum_fxyz, f12);
}

template <int L>
void accumulate_f12_one_all_n(
  const int n_max_angular_plus_1,
  const double d12inv,
  const double* NEP_RESTRICT fn,
  const double* NEP_RESTRICT fnp,
  const double* NEP_RESTRICT scaled_sum_fxyz,
  const double* NEP_RESTRICT r12,
  double* NEP_RESTRICT f12)
{
  const double dx[3] = {
    (1.0 - r12[0] * r12[0]) * d12inv, -r12[0] * r12[1] * d12inv, -r12[0] * r12[2] * d12inv};
  const double dy[3] = {
    -r12[0] * r12[1] * d12inv, (1.0 - r12[1] * r12[1]) * d12inv, -r12[1] * r12[2] * d12inv};
  const double dz[3] = {
    -r12[0] * r12[2] * d12inv, -r12[1] * r12[2] * d12inv, (1.0 - r12[2] * r12[2]) * d12inv};

  double z_pow[L + 1] = {1.0};
  for (int n = 1; n <= L; ++n) {
    z_pow[n] = r12[2] * z_pow[n - 1];
  }

  double f12_x = f12[0];
  double f12_y = f12[1];
  double f12_z = f12[2];
  const int s_index = L * L - 1;
  double real_part = 1.0;
  double imag_part = 0.0;
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    double z_factor = 0.0;
    double dz_factor = 0.0;
    for (int n2 = n2_start; n2 <= L - n1; n2 += 2) {
      const double coefficient = z_coefficient<L>(n1, n2);
      z_factor += coefficient * z_pow[n2];
      if (n2 > 0) {
        dz_factor += coefficient * n2 * z_pow[n2 - 1];
      }
    }

    if (n1 == 0) {
      const int s0_index = s_index;
      for (int n = 0; n < n_max_angular_plus_1; ++n) {
        const double s0 = scaled_sum_fxyz[n * NUM_OF_ABC + s0_index];
        const double radial_factor = z_factor * fnp[n];
        const double angular_factor = dz_factor * fn[n];
        f12_x += s0 * (radial_factor * r12[0] + angular_factor * dz[0]);
        f12_y += s0 * (radial_factor * r12[1] + angular_factor * dz[1]);
        f12_z += s0 * (radial_factor * r12[2] + angular_factor * dz[2]);
      }
    } else {
      double real_part_n1 = n1 * real_part;
      double imag_part_n1 = n1 * imag_part;
      double real_part_dx0 = dx[0];
      double real_part_dx1 = dx[1];
      double real_part_dx2 = dx[2];
      double imag_part_dy0 = dy[0];
      double imag_part_dy1 = dy[1];
      double imag_part_dy2 = dy[2];
      complex_product(real_part_n1, imag_part_n1, real_part_dx0, imag_part_dy0);
      complex_product(real_part_n1, imag_part_n1, real_part_dx1, imag_part_dy1);
      complex_product(real_part_n1, imag_part_n1, real_part_dx2, imag_part_dy2);

      const int sa_index = s_index + 2 * n1 - 1;
      const int sb_index = s_index + 2 * n1;
      complex_product(r12[0], r12[1], real_part, imag_part);
      for (int n = 0; n < n_max_angular_plus_1; ++n) {
        const double sa = scaled_sum_fxyz[n * NUM_OF_ABC + sa_index];
        const double sb = scaled_sum_fxyz[n * NUM_OF_ABC + sb_index];
        const double xy_temp = sa * real_part + sb * imag_part;
        const double z_fn = z_factor * fn[n];
        const double z_fnp = z_factor * fnp[n];
        const double dz_fn = dz_factor * fn[n];
        f12_x += (sa * real_part_dx0 + sb * imag_part_dy0) * z_fn;
        f12_x += xy_temp * (z_fnp * r12[0] + dz_fn * dz[0]);
        f12_y += (sa * real_part_dx1 + sb * imag_part_dy1) * z_fn;
        f12_y += xy_temp * (z_fnp * r12[1] + dz_fn * dz[1]);
        f12_z += (sa * real_part_dx2 + sb * imag_part_dy2) * z_fn;
        f12_z += xy_temp * (z_fnp * r12[2] + dz_fn * dz[2]);
      }
    }
  }
  f12[0] = f12_x;
  f12[1] = f12_y;
  f12[2] = f12_z;
}

template <int L>
void accumulate_f12_one_contracted(
  const double d12inv,
  const double* NEP_RESTRICT contracted_fn,
  const double* NEP_RESTRICT contracted_fnp,
  const double* NEP_RESTRICT r12,
  double* NEP_RESTRICT f12)
{
  const double dx[3] = {
    (1.0 - r12[0] * r12[0]) * d12inv, -r12[0] * r12[1] * d12inv, -r12[0] * r12[2] * d12inv};
  const double dy[3] = {
    -r12[0] * r12[1] * d12inv, (1.0 - r12[1] * r12[1]) * d12inv, -r12[1] * r12[2] * d12inv};
  const double dz[3] = {
    -r12[0] * r12[2] * d12inv, -r12[1] * r12[2] * d12inv, (1.0 - r12[2] * r12[2]) * d12inv};

  double z_pow[L + 1] = {1.0};
  for (int n = 1; n <= L; ++n) {
    z_pow[n] = r12[2] * z_pow[n - 1];
  }

  double f12_x = f12[0];
  double f12_y = f12[1];
  double f12_z = f12[2];
  const int s_index = L * L - 1;
  double real_part = 1.0;
  double imag_part = 0.0;
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    double z_factor = 0.0;
    double dz_factor = 0.0;
    for (int n2 = n2_start; n2 <= L - n1; n2 += 2) {
      const double coefficient = z_coefficient<L>(n1, n2);
      z_factor += coefficient * z_pow[n2];
      if (n2 > 0) {
        dz_factor += coefficient * n2 * z_pow[n2 - 1];
      }
    }

    if (n1 == 0) {
      const double sum_fn = contracted_fn[s_index];
      const double sum_fnp = contracted_fnp[s_index];
      f12_x += z_factor * sum_fnp * r12[0] + dz_factor * sum_fn * dz[0];
      f12_y += z_factor * sum_fnp * r12[1] + dz_factor * sum_fn * dz[1];
      f12_z += z_factor * sum_fnp * r12[2] + dz_factor * sum_fn * dz[2];
    } else {
      double real_part_n1 = n1 * real_part;
      double imag_part_n1 = n1 * imag_part;
      double real_part_dx0 = dx[0];
      double real_part_dx1 = dx[1];
      double real_part_dx2 = dx[2];
      double imag_part_dy0 = dy[0];
      double imag_part_dy1 = dy[1];
      double imag_part_dy2 = dy[2];
      complex_product(real_part_n1, imag_part_n1, real_part_dx0, imag_part_dy0);
      complex_product(real_part_n1, imag_part_n1, real_part_dx1, imag_part_dy1);
      complex_product(real_part_n1, imag_part_n1, real_part_dx2, imag_part_dy2);

      const int sa_index = s_index + 2 * n1 - 1;
      const int sb_index = s_index + 2 * n1;
      complex_product(r12[0], r12[1], real_part, imag_part);
      const double sa_fn = contracted_fn[sa_index];
      const double sb_fn = contracted_fn[sb_index];
      const double sa_fnp = contracted_fnp[sa_index];
      const double sb_fnp = contracted_fnp[sb_index];
      const double xy_fn = sa_fn * real_part + sb_fn * imag_part;
      const double xy_fnp = sa_fnp * real_part + sb_fnp * imag_part;

      f12_x += (sa_fn * real_part_dx0 + sb_fn * imag_part_dy0) * z_factor;
      f12_x += xy_fnp * z_factor * r12[0] + xy_fn * dz_factor * dz[0];
      f12_y += (sa_fn * real_part_dx1 + sb_fn * imag_part_dy1) * z_factor;
      f12_y += xy_fnp * z_factor * r12[1] + xy_fn * dz_factor * dz[1];
      f12_z += (sa_fn * real_part_dx2 + sb_fn * imag_part_dy2) * z_factor;
      f12_z += xy_fnp * z_factor * r12[2] + xy_fn * dz_factor * dz[2];
    }
  }
  f12[0] = f12_x;
  f12[1] = f12_y;
  f12[2] = f12_z;
}

template <int L_MAX>
void accumulate_f12_3body_contracted_one_lmax(
  const int n_max_angular_plus_1,
  const double d12,
  const double* NEP_RESTRICT r12,
  const double* NEP_RESTRICT fn,
  const double* NEP_RESTRICT fnp,
  const double* NEP_RESTRICT scaled_sum_fxyz,
  double* NEP_RESTRICT f12)
{
  const double d12inv = 1.0 / d12;
  const double r12unit[3] = {r12[0] * d12inv, r12[1] * d12inv, r12[2] * d12inv};
  constexpr int abc_count = L_MAX * (L_MAX + 2);
  double contracted_fn[abc_count] = {0.0};
  double contracted_fnp[abc_count] = {0.0};

  for (int n = 0; n < n_max_angular_plus_1; ++n) {
    const double fn_n = fn[n];
    const double fnp_n = fnp[n];
    const double* row = scaled_sum_fxyz + n * NUM_OF_ABC;
    for (int abc = 0; abc < abc_count; ++abc) {
      contracted_fn[abc] += row[abc] * fn_n;
      contracted_fnp[abc] += row[abc] * fnp_n;
    }
  }

  if constexpr (L_MAX >= 1) {
    accumulate_f12_one_contracted<1>(d12inv, contracted_fn, contracted_fnp, r12unit, f12);
  }
  if constexpr (L_MAX >= 2) {
    accumulate_f12_one_contracted<2>(d12inv, contracted_fn, contracted_fnp, r12unit, f12);
  }
  if constexpr (L_MAX >= 3) {
    accumulate_f12_one_contracted<3>(d12inv, contracted_fn, contracted_fnp, r12unit, f12);
  }
  if constexpr (L_MAX >= 4) {
    accumulate_f12_one_contracted<4>(d12inv, contracted_fn, contracted_fnp, r12unit, f12);
  }
  if constexpr (L_MAX >= 5) {
    accumulate_f12_one_contracted<5>(d12inv, contracted_fn, contracted_fnp, r12unit, f12);
  }
  if constexpr (L_MAX >= 6) {
    accumulate_f12_one_contracted<6>(d12inv, contracted_fn, contracted_fnp, r12unit, f12);
  }
  if constexpr (L_MAX >= 7) {
    accumulate_f12_one_contracted<7>(d12inv, contracted_fn, contracted_fnp, r12unit, f12);
  }
  if constexpr (L_MAX >= 8) {
    accumulate_f12_one_contracted<8>(d12inv, contracted_fn, contracted_fnp, r12unit, f12);
  }
}

void accumulate_f12_3body_contracted_all_n(
  const int L_max,
  const int n_max_angular_plus_1,
  const double d12,
  const double* NEP_RESTRICT r12,
  const double* NEP_RESTRICT fn,
  const double* NEP_RESTRICT fnp,
  const double* NEP_RESTRICT scaled_sum_fxyz,
  double* NEP_RESTRICT f12)
{
  if (L_max == 1) {
    accumulate_f12_3body_contracted_one_lmax<1>(
      n_max_angular_plus_1, d12, r12, fn, fnp, scaled_sum_fxyz, f12);
  } else if (L_max == 2) {
    accumulate_f12_3body_contracted_one_lmax<2>(
      n_max_angular_plus_1, d12, r12, fn, fnp, scaled_sum_fxyz, f12);
  } else if (L_max == 3) {
    accumulate_f12_3body_contracted_one_lmax<3>(
      n_max_angular_plus_1, d12, r12, fn, fnp, scaled_sum_fxyz, f12);
  } else if (L_max == 4) {
    accumulate_f12_3body_contracted_one_lmax<4>(
      n_max_angular_plus_1, d12, r12, fn, fnp, scaled_sum_fxyz, f12);
  } else if (L_max == 5) {
    accumulate_f12_3body_contracted_one_lmax<5>(
      n_max_angular_plus_1, d12, r12, fn, fnp, scaled_sum_fxyz, f12);
  } else if (L_max == 6) {
    accumulate_f12_3body_contracted_one_lmax<6>(
      n_max_angular_plus_1, d12, r12, fn, fnp, scaled_sum_fxyz, f12);
  } else if (L_max == 7) {
    accumulate_f12_3body_contracted_one_lmax<7>(
      n_max_angular_plus_1, d12, r12, fn, fnp, scaled_sum_fxyz, f12);
  } else {
    accumulate_f12_3body_contracted_one_lmax<8>(
      n_max_angular_plus_1, d12, r12, fn, fnp, scaled_sum_fxyz, f12);
  }
}

void accumulate_f12_extra_terms_contracted_all_n(
  const int L_max,
  const int has_q_222,
  const int has_q_1111,
  const int has_q_112,
  const int has_q_123,
  const int has_q_233,
  const int has_q_134,
  const int num_L,
  const int n_max_angular_plus_1,
  const double d12,
  const double d12inv,
  const double* NEP_RESTRICT r12,
  const double* NEP_RESTRICT fn,
  const double* NEP_RESTRICT fnp,
  const double* NEP_RESTRICT Fp,
  const double* NEP_RESTRICT sum_fxyz,
  double* NEP_RESTRICT f12)
{
  if (num_L <= L_max || (!has_q_112 && !has_q_123 && !has_q_233 && !has_q_134)) {
    return;
  }

  const double d12inv2 = d12inv * d12inv;
  double l1_fn[3] = {0.0};
  double l1_fnp[3] = {0.0};
  double l2_fn[5] = {0.0};
  double l2_fnp[5] = {0.0};
  double l3_fn[7] = {0.0};
  double l3_fnp[7] = {0.0};
  double l4_fn[9] = {0.0};
  double l4_fnp[9] = {0.0};

  for (int n = 0; n < n_max_angular_plus_1; ++n) {
    const int offset = n * NUM_OF_ABC;
    const double s1[3] = {sum_fxyz[offset + 0], sum_fxyz[offset + 1], sum_fxyz[offset + 2]};
    const double s2[5] = {
      sum_fxyz[offset + 3], sum_fxyz[offset + 4], sum_fxyz[offset + 5],
      sum_fxyz[offset + 6], sum_fxyz[offset + 7]};
    const double fn1 = fn[n] * d12inv;
    const double fnp1 = fnp[n] * d12inv - fn[n] * d12inv2;
    const double fn2 = fn1 * d12inv;
    const double fnp2 = fnp1 * d12inv - fn1 * d12inv2;

    int L_index = L_max;
    if (has_q_222) {
      ++L_index;
    }
    if (has_q_1111) {
      ++L_index;
    }

    if (has_q_112) {
      const double fp = Fp[(L_index++) * n_max_angular_plus_1 + n];
      const double fn1_factor = fp * fn1;
      const double fnp1_factor = fp * fnp1 * d12inv;
      const double fn2_factor = fp * fn2;
      const double fnp2_factor = fp * fnp2 * d12inv;

      double tmp0 = C4B2[0] * s1[0] * s1[0] + C4B2[2] * (s1[1] * s1[1] + s1[2] * s1[2]);
      l2_fn[0] += tmp0 * fn2_factor;
      l2_fnp[0] += tmp0 * fnp2_factor;
      tmp0 = C4B2[1] * s1[0] * s1[1];
      l2_fn[1] += tmp0 * fn2_factor;
      l2_fnp[1] += tmp0 * fnp2_factor;
      tmp0 = C4B2[1] * s1[0] * s1[2];
      l2_fn[2] += tmp0 * fn2_factor;
      l2_fnp[2] += tmp0 * fnp2_factor;
      tmp0 = C4B2[3] * (s1[1] * s1[1] - s1[2] * s1[2]);
      l2_fn[3] += tmp0 * fn2_factor;
      l2_fnp[3] += tmp0 * fnp2_factor;
      tmp0 = C4B2[4] * s1[1] * s1[2];
      l2_fn[4] += tmp0 * fn2_factor;
      l2_fnp[4] += tmp0 * fnp2_factor;

      tmp0 = C4B2[0] * 2.0 * s1[0] * s2[0] + C4B2[1] * (s1[1] * s2[1] + s1[2] * s2[2]);
      l1_fn[0] += tmp0 * fn1_factor;
      l1_fnp[0] += tmp0 * fnp1_factor;
      tmp0 = C4B2[1] * s1[0] * s2[1] + C4B2[2] * s1[1] * s2[0] * 2.0 +
             C4B2[3] * s1[1] * s2[3] * 2.0 + C4B2[4] * s1[2] * s2[4];
      l1_fn[1] += tmp0 * fn1_factor;
      l1_fnp[1] += tmp0 * fnp1_factor;
      tmp0 = C4B2[1] * s1[0] * s2[2] + C4B2[2] * s1[2] * s2[0] * 2.0 -
             C4B2[3] * s1[2] * s2[3] * 2.0 + C4B2[4] * s1[1] * s2[4];
      l1_fn[2] += tmp0 * fn1_factor;
      l1_fnp[2] += tmp0 * fnp1_factor;
    }

    if (has_q_123 || has_q_233 || has_q_134) {
      const double s3[7] = {
        sum_fxyz[offset + 8],  sum_fxyz[offset + 9],  sum_fxyz[offset + 10],
        sum_fxyz[offset + 11], sum_fxyz[offset + 12], sum_fxyz[offset + 13],
        sum_fxyz[offset + 14]};
      const double fn3 = fn2 * d12inv;
      const double fnp3 = fnp2 * d12inv - fn2 * d12inv2;

      if (has_q_123) {
        const double fp = Fp[(L_index++) * n_max_angular_plus_1 + n];
        const double fn1_factor = fp * fn1;
        const double fnp1_factor = fp * fnp1 * d12inv;
        const double fn2_factor = fp * fn2;
        const double fnp2_factor = fp * fnp2 * d12inv;
        const double fn3_factor = fp * fn3;
        const double fnp3_factor = fp * fnp3 * d12inv;

        double tmp0 = C4B_123[5] * s3[3] * s2[3] + C4B_123[5] * s3[4] * s2[4] +
                      C4B_123[4] * s3[2] * s2[2] + C4B_123[1] * s2[0] * s3[0] +
                      C4B_123[4] * s2[1] * s3[1];
        l1_fn[0] += tmp0 * fn1_factor;
        l1_fnp[0] += tmp0 * fnp1_factor;
        tmp0 = -C4B_123[0] * s3[2] * s2[4] + C4B_123[6] * s3[3] * s2[1] +
               C4B_123[6] * s3[4] * s2[2] + C4B_123[3] * s3[5] * s2[3] +
               C4B_123[3] * s3[6] * s2[4] - C4B_123[2] * s2[1] * s3[0] +
               C4B_123[1] * s2[0] * s3[1] - C4B_123[0] * s2[3] * s3[1];
        l1_fn[1] += tmp0 * fn1_factor;
        l1_fnp[1] += tmp0 * fnp1_factor;
        tmp0 = C4B_123[6] * s3[4] * s2[1] - C4B_123[6] * s3[3] * s2[2] +
               C4B_123[3] * s3[6] * s2[3] - C4B_123[3] * s3[5] * s2[4] +
               C4B_123[1] * s3[2] * s2[0] + C4B_123[0] * s3[2] * s2[3] -
               C4B_123[2] * s2[2] * s3[0] - C4B_123[0] * s2[4] * s3[1];
        l1_fn[2] += tmp0 * fn1_factor;
        l1_fnp[2] += tmp0 * fnp1_factor;

        tmp0 = C4B_123[1] * (s3[2] * s1[2] + s1[0] * s3[0] + s1[1] * s3[1]);
        l2_fn[0] += tmp0 * fn2_factor;
        l2_fnp[0] += tmp0 * fnp2_factor;
        tmp0 = C4B_123[6] * s3[4] * s1[2] + C4B_123[4] * s1[0] * s3[1] +
               C4B_123[6] * s1[1] * s3[3] - C4B_123[2] * s1[1] * s3[0];
        l2_fn[1] += tmp0 * fn2_factor;
        l2_fnp[1] += tmp0 * fnp2_factor;
        tmp0 = -C4B_123[6] * s3[3] * s1[2] + C4B_123[4] * s3[2] * s1[0] -
               C4B_123[2] * s1[2] * s3[0] + C4B_123[6] * s1[1] * s3[4];
        l2_fn[2] += tmp0 * fn2_factor;
        l2_fnp[2] += tmp0 * fnp2_factor;
        tmp0 = C4B_123[5] * s1[0] * s3[3] + C4B_123[3] * s3[6] * s1[2] +
               C4B_123[0] * s3[2] * s1[2] + C4B_123[3] * s1[1] * s3[5] -
               C4B_123[0] * s1[1] * s3[1];
        l2_fn[3] += tmp0 * fn2_factor;
        l2_fnp[3] += tmp0 * fnp2_factor;
        tmp0 = C4B_123[5] * s1[0] * s3[4] - C4B_123[3] * s3[5] * s1[2] -
               C4B_123[0] * s3[2] * s1[1] - C4B_123[0] * s1[2] * s3[1] +
               C4B_123[3] * s1[1] * s3[6];
        l2_fn[4] += tmp0 * fn2_factor;
        l2_fnp[4] += tmp0 * fnp2_factor;

        tmp0 = C4B_123[1] * s1[0] * s2[0] - C4B_123[2] * (s1[2] * s2[2] + s1[1] * s2[1]);
        l3_fn[0] += tmp0 * fn3_factor;
        l3_fnp[0] += tmp0 * fnp3_factor;
        tmp0 = C4B_123[4] * s1[0] * s2[1] + C4B_123[1] * s1[1] * s2[0] -
               C4B_123[0] * (s1[2] * s2[4] + s1[1] * s2[3]);
        l3_fn[1] += tmp0 * fn3_factor;
        l3_fnp[1] += tmp0 * fnp3_factor;
        tmp0 = C4B_123[4] * s1[0] * s2[2] + C4B_123[1] * s1[2] * s2[0] +
               C4B_123[0] * (s1[2] * s2[3] - s1[1] * s2[4]);
        l3_fn[2] += tmp0 * fn3_factor;
        l3_fnp[2] += tmp0 * fnp3_factor;
        tmp0 = -C4B_123[6] * s1[2] * s2[2] + C4B_123[5] * s1[0] * s2[3] +
               C4B_123[6] * s1[1] * s2[1];
        l3_fn[3] += tmp0 * fn3_factor;
        l3_fnp[3] += tmp0 * fnp3_factor;
        tmp0 = C4B_123[6] * (s1[2] * s2[1] + s1[1] * s2[2]) + C4B_123[5] * s1[0] * s2[4];
        l3_fn[4] += tmp0 * fn3_factor;
        l3_fnp[4] += tmp0 * fnp3_factor;
        tmp0 = C4B_123[3] * (-s1[2] * s2[4] + s1[1] * s2[3]);
        l3_fn[5] += tmp0 * fn3_factor;
        l3_fnp[5] += tmp0 * fnp3_factor;
        tmp0 = C4B_123[3] * (s1[2] * s2[3] + s1[1] * s2[4]);
        l3_fn[6] += tmp0 * fn3_factor;
        l3_fnp[6] += tmp0 * fnp3_factor;
      }

      if (has_q_233) {
        const double fp = Fp[(L_index++) * n_max_angular_plus_1 + n];
        const double fn2_factor = fp * fn2;
        const double fnp2_factor = fp * fnp2 * d12inv;
        const double fn3_factor = fp * fn3;
        const double fnp3_factor = fp * fnp3 * d12inv;

        double tmp0 = C4B_233[0] * s3[0] * s3[0] + C4B_233[1] * (s3[2] * s3[2] + s3[1] * s3[1]) -
                      C4B_233[4] * (s3[5] * s3[5] + s3[6] * s3[6]);
        l2_fn[0] += tmp0 * fn2_factor;
        l2_fnp[0] += tmp0 * fnp2_factor;
        tmp0 = C4B_233[3] * s3[0] * s3[1] + C4B_233[8] * (s3[3] * s3[1] + s3[2] * s3[4]) +
               C4B_233[9] * (s3[4] * s3[6] + s3[5] * s3[3]);
        l2_fn[1] += tmp0 * fn2_factor;
        l2_fnp[1] += tmp0 * fnp2_factor;
        tmp0 = C4B_233[3] * s3[2] * s3[0] + C4B_233[8] * (s3[4] * s3[1] - s3[2] * s3[3]) +
               C4B_233[9] * (s3[3] * s3[6] - s3[5] * s3[4]);
        l2_fn[2] += tmp0 * fn2_factor;
        l2_fnp[2] += tmp0 * fnp2_factor;
        tmp0 = C4B_233[2] * (-s3[2] * s3[2] + s3[1] * s3[1]) -
               C4B_233[5] * (s3[5] * s3[1] + s3[2] * s3[6]) - C4B_233[7] * s3[3] * s3[0];
        l2_fn[3] += tmp0 * fn2_factor;
        l2_fnp[3] += tmp0 * fnp2_factor;
        tmp0 = C4B_233[5] * (-s3[6] * s3[1] + s3[2] * s3[5]) + C4B_233[6] * s3[2] * s3[1] -
               C4B_233[7] * s3[4] * s3[0];
        l2_fn[4] += tmp0 * fn2_factor;
        l2_fnp[4] += tmp0 * fnp2_factor;

        tmp0 = 2.0 * C4B_233[0] * s2[0] * s3[0] + C4B_233[3] * (s2[1] * s3[1] + s2[2] * s3[2]) -
               C4B_233[7] * (s2[3] * s3[3] + s2[4] * s3[4]);
        l3_fn[0] += tmp0 * fn3_factor;
        l3_fnp[0] += tmp0 * fnp3_factor;
        tmp0 = 2.0 * C4B_233[1] * s2[0] * s3[1] + 2.0 * C4B_233[2] * s2[3] * s3[1] +
               C4B_233[3] * s2[1] * s3[0] - C4B_233[5] * (s2[4] * s3[6] + s2[3] * s3[5]) +
               C4B_233[6] * s2[4] * s3[2] + C4B_233[8] * (s2[1] * s3[3] + s2[2] * s3[4]);
        l3_fn[1] += tmp0 * fn3_factor;
        l3_fnp[1] += tmp0 * fnp3_factor;
        tmp0 = 2.0 * C4B_233[1] * s2[0] * s3[2] - 2.0 * C4B_233[2] * s2[3] * s3[2] +
               C4B_233[3] * s2[2] * s3[0] + C4B_233[5] * (-s2[3] * s3[6] + s2[4] * s3[5]) +
               C4B_233[6] * s2[4] * s3[1] + C4B_233[8] * (s2[1] * s3[4] - s2[2] * s3[3]);
        l3_fn[2] += tmp0 * fn3_factor;
        l3_fnp[2] += tmp0 * fnp3_factor;
        tmp0 = -C4B_233[7] * s2[3] * s3[0] + C4B_233[8] * (s2[1] * s3[1] - s2[2] * s3[2]) +
               C4B_233[9] * (s2[2] * s3[6] + s2[1] * s3[5]);
        l3_fn[3] += tmp0 * fn3_factor;
        l3_fnp[3] += tmp0 * fnp3_factor;
        tmp0 = -C4B_233[7] * s2[4] * s3[0] + C4B_233[8] * (s2[2] * s3[1] + s2[1] * s3[2]) +
               C4B_233[9] * (s2[1] * s3[6] - s2[2] * s3[5]);
        l3_fn[4] += tmp0 * fn3_factor;
        l3_fnp[4] += tmp0 * fnp3_factor;
        tmp0 = -2.0 * C4B_233[4] * s2[0] * s3[5] + C4B_233[5] * (s2[4] * s3[2] - s2[3] * s3[1]) +
               C4B_233[9] * (s2[1] * s3[3] - s2[2] * s3[4]);
        l3_fn[5] += tmp0 * fn3_factor;
        l3_fnp[5] += tmp0 * fnp3_factor;
        tmp0 = -2.0 * C4B_233[4] * s2[0] * s3[6] - C4B_233[5] * (s2[3] * s3[2] + s2[4] * s3[1]) +
               C4B_233[9] * (s2[1] * s3[4] + s2[2] * s3[3]);
        l3_fn[6] += tmp0 * fn3_factor;
        l3_fnp[6] += tmp0 * fnp3_factor;
      }

      if (has_q_134) {
        const double s4[9] = {
          sum_fxyz[offset + 15], sum_fxyz[offset + 16], sum_fxyz[offset + 17],
          sum_fxyz[offset + 18], sum_fxyz[offset + 19], sum_fxyz[offset + 20],
          sum_fxyz[offset + 21], sum_fxyz[offset + 22], sum_fxyz[offset + 23]};
        const double fn4 = fn3 * d12inv;
        const double fnp4 = fnp3 * d12inv - fn3 * d12inv2;
        const double fp = Fp[(L_index++) * n_max_angular_plus_1 + n];
        const double fn1_factor = fp * fn1;
        const double fnp1_factor = fp * fnp1 * d12inv;
        const double fn3_factor = fp * fn3;
        const double fnp3_factor = fp * fnp3 * d12inv;
        const double fn4_factor = fp * fn4;
        const double fnp4_factor = fp * fnp4 * d12inv;

        double tmp0 = C4B_134[1] * s4[0] * s3[0] + C4B_134[5] * (s3[2] * s4[2] + s4[1] * s3[1]) +
                      C4B_134[7] * (s3[3] * s4[3] + s3[4] * s4[4]) +
                      C4B_134[8] * (s3[5] * s4[5] + s3[6] * s4[6]);
        l1_fn[0] += tmp0 * fn1_factor;
        l1_fnp[0] += tmp0 * fnp1_factor;
        tmp0 = -C4B_134[0] * s4[0] * s3[1] - C4B_134[2] * (s3[5] * s4[3] + s3[6] * s4[4]) +
               C4B_134[3] * (s3[2] * s4[4] + s4[3] * s3[1]) + C4B_134[4] * s4[1] * s3[0] -
               C4B_134[5] * (s3[3] * s4[1] + s3[4] * s4[2]) +
               C4B_134[6] * (s3[5] * s4[7] + s3[6] * s4[8]) +
               C4B_134[9] * (s3[3] * s4[5] + s3[4] * s4[6]);
        l1_fn[1] += tmp0 * fn1_factor;
        l1_fnp[1] += tmp0 * fnp1_factor;
        tmp0 = -C4B_134[0] * s3[2] * s4[0] + C4B_134[2] * (-s3[6] * s4[3] + s3[5] * s4[4]) +
               C4B_134[3] * (-s3[2] * s4[3] + s4[4] * s3[1]) + C4B_134[4] * s4[2] * s3[0] +
               C4B_134[5] * (-s3[4] * s4[1] + s3[3] * s4[2]) +
               C4B_134[6] * (-s3[6] * s4[7] + s3[5] * s4[8]) +
               C4B_134[9] * (-s3[4] * s4[5] + s3[3] * s4[6]);
        l1_fn[2] += tmp0 * fn1_factor;
        l1_fnp[2] += tmp0 * fnp1_factor;

        tmp0 = C4B_134[1] * s1[0] * s4[0] + C4B_134[4] * (s1[1] * s4[1] + s1[2] * s4[2]);
        l3_fn[0] += tmp0 * fn3_factor;
        l3_fnp[0] += tmp0 * fnp3_factor;
        tmp0 = -C4B_134[0] * s1[1] * s4[0] + C4B_134[3] * (s1[1] * s4[3] + s1[2] * s4[4]) +
               C4B_134[5] * s1[0] * s4[1];
        l3_fn[1] += tmp0 * fn3_factor;
        l3_fnp[1] += tmp0 * fnp3_factor;
        tmp0 = -C4B_134[0] * s4[0] * s1[2] + C4B_134[3] * (-s4[3] * s1[2] + s1[1] * s4[4]) +
               C4B_134[5] * s1[0] * s4[2];
        l3_fn[2] += tmp0 * fn3_factor;
        l3_fnp[2] += tmp0 * fnp3_factor;
        tmp0 = C4B_134[5] * (-s1[1] * s4[1] + s1[2] * s4[2]) + C4B_134[7] * s1[0] * s4[3] +
               C4B_134[9] * (s1[1] * s4[5] + s1[2] * s4[6]);
        l3_fn[3] += tmp0 * fn3_factor;
        l3_fnp[3] += tmp0 * fnp3_factor;
        tmp0 = -C4B_134[5] * (s1[1] * s4[2] + s1[2] * s4[1]) + C4B_134[7] * s1[0] * s4[4] +
               C4B_134[9] * (s1[1] * s4[6] - s1[2] * s4[5]);
        l3_fn[4] += tmp0 * fn3_factor;
        l3_fnp[4] += tmp0 * fnp3_factor;
        tmp0 = C4B_134[2] * (-s1[1] * s4[3] + s1[2] * s4[4]) +
               C4B_134[6] * (s1[1] * s4[7] + s1[2] * s4[8]) + C4B_134[8] * s1[0] * s4[5];
        l3_fn[5] += tmp0 * fn3_factor;
        l3_fnp[5] += tmp0 * fnp3_factor;
        tmp0 = C4B_134[2] * (-s1[1] * s4[4] - s1[2] * s4[3]) +
               C4B_134[6] * (s1[1] * s4[8] - s1[2] * s4[7]) + C4B_134[8] * s1[0] * s4[6];
        l3_fn[6] += tmp0 * fn3_factor;
        l3_fnp[6] += tmp0 * fnp3_factor;

        tmp0 = -C4B_134[0] * (s3[2] * s1[2] + s1[1] * s3[1]) + C4B_134[1] * s1[0] * s3[0];
        l4_fn[0] += tmp0 * fn4_factor;
        l4_fnp[0] += tmp0 * fnp4_factor;
        tmp0 = C4B_134[4] * s1[1] * s3[0] + C4B_134[5] * (s1[0] * s3[1] - s1[1] * s3[3] - s1[2] * s3[4]);
        l4_fn[1] += tmp0 * fn4_factor;
        l4_fnp[1] += tmp0 * fnp4_factor;
        tmp0 = C4B_134[4] * s1[2] * s3[0] + C4B_134[5] * (s1[0] * s3[2] - s1[1] * s3[4] + s1[2] * s3[3]);
        l4_fn[2] += tmp0 * fn4_factor;
        l4_fnp[2] += tmp0 * fnp4_factor;
        tmp0 = C4B_134[2] * (-s1[1] * s3[5] - s1[2] * s3[6]) +
               C4B_134[3] * (-s3[2] * s1[2] + s1[1] * s3[1]) + C4B_134[7] * s1[0] * s3[3];
        l4_fn[3] += tmp0 * fn4_factor;
        l4_fnp[3] += tmp0 * fnp4_factor;
        tmp0 = C4B_134[2] * (-s1[1] * s3[6] + s1[2] * s3[5]) +
               C4B_134[3] * (s1[1] * s3[2] + s1[2] * s3[1]) + C4B_134[7] * s1[0] * s3[4];
        l4_fn[4] += tmp0 * fn4_factor;
        l4_fnp[4] += tmp0 * fnp4_factor;
        tmp0 = C4B_134[8] * s1[0] * s3[5] + C4B_134[9] * (s1[1] * s3[3] - s1[2] * s3[4]);
        l4_fn[5] += tmp0 * fn4_factor;
        l4_fnp[5] += tmp0 * fnp4_factor;
        tmp0 = C4B_134[8] * s1[0] * s3[6] + C4B_134[9] * (s1[1] * s3[4] + s1[2] * s3[3]);
        l4_fn[6] += tmp0 * fn4_factor;
        l4_fnp[6] += tmp0 * fnp4_factor;
        tmp0 = C4B_134[6] * (s1[1] * s3[5] - s1[2] * s3[6]);
        l4_fn[7] += tmp0 * fn4_factor;
        l4_fnp[7] += tmp0 * fnp4_factor;
        tmp0 = C4B_134[6] * (s1[1] * s3[6] + s1[2] * s3[5]);
        l4_fn[8] += tmp0 * fn4_factor;
        l4_fnp[8] += tmp0 * fnp4_factor;
      }
    }
  }

  accumulate_f12_from_contracted_l1(l1_fn, l1_fnp, r12, f12);
  accumulate_f12_from_contracted_l2(d12, l2_fn, l2_fnp, r12, f12);
  accumulate_f12_from_contracted_l3(d12, l3_fn, l3_fnp, r12, f12);
  accumulate_f12_from_contracted_l4(d12, l4_fn, l4_fnp, r12, f12);
}

void accumulate_f12_contracted_all_n(
  const int L_max,
  const int has_q_222,
  const int has_q_1111,
  const int has_q_112,
  const int has_q_123,
  const int has_q_233,
  const int has_q_134,
  const int num_L,
  const int n_max_angular_plus_1,
  const double d12,
  const double* NEP_RESTRICT r12,
  const double* NEP_RESTRICT fn,
  const double* NEP_RESTRICT fnp,
  const double* NEP_RESTRICT Fp,
  const double* NEP_RESTRICT sum_fxyz,
  const double* NEP_RESTRICT scaled_sum_fxyz,
  const double* NEP_RESTRICT q222_derivatives,
  const double* NEP_RESTRICT q1111_derivatives,
  double* NEP_RESTRICT f12)
{
  const double d12inv = 1.0 / d12;
  accumulate_f12_3body_contracted_all_n(
    L_max, n_max_angular_plus_1, d12, r12, fn, fnp, scaled_sum_fxyz, f12);

  if (num_L <= L_max) {
    return;
  }

  if (has_q_222 || has_q_1111) {
    accumulate_f12_q222_q1111_contracted_all_n(
      has_q_222, has_q_1111, n_max_angular_plus_1, d12, d12inv, r12, fn, fnp,
      q222_derivatives, q1111_derivatives, f12);
  }
  accumulate_f12_extra_terms_contracted_all_n(
    L_max, has_q_222, has_q_1111, has_q_112, has_q_123, has_q_233, has_q_134, num_L,
    n_max_angular_plus_1, d12, d12inv, r12, fn, fnp, Fp, sum_fxyz, f12);
}

void accumulate_f12_contracted_lmax4_q222_q1111_n5(
  const double d12,
  const double* NEP_RESTRICT r12,
  const double* NEP_RESTRICT fn,
  const double* NEP_RESTRICT fnp,
  const double* NEP_RESTRICT scaled_sum_fxyz,
  const double* NEP_RESTRICT q222_derivatives,
  const double* NEP_RESTRICT q1111_derivatives,
  double* NEP_RESTRICT f12)
{
  const double d12inv = 1.0 / d12;
  accumulate_f12_3body_contracted_one_lmax<4>(
    5, d12, r12, fn, fnp, scaled_sum_fxyz, f12);
  accumulate_f12_q222_q1111_contracted_all_n(
    1, 1, 5, d12, d12inv, r12, fn, fnp, q222_derivatives, q1111_derivatives, f12);
}

void accumulate_f12_all_n(
  const int L_max,
  const int has_q_222,
  const int has_q_1111,
  const int has_q_112,
  const int has_q_123,
  const int has_q_233,
  const int has_q_134,
  const int num_L,
  const int n_max_angular_plus_1,
  const double d12,
  const double* NEP_RESTRICT r12,
  const double* NEP_RESTRICT fn,
  const double* NEP_RESTRICT fnp,
  const double* NEP_RESTRICT Fp,
  const double* NEP_RESTRICT sum_fxyz,
  const double* NEP_RESTRICT scaled_sum_fxyz,
  const double* NEP_RESTRICT q222_derivatives,
  const double* NEP_RESTRICT q1111_derivatives,
  double* NEP_RESTRICT f12)
{
  const double d12inv = 1.0 / d12;
  const double r12unit[3] = {r12[0] * d12inv, r12[1] * d12inv, r12[2] * d12inv};

  if (L_max >= 1) {
    accumulate_f12_one_all_n<1>(n_max_angular_plus_1, d12inv, fn, fnp, scaled_sum_fxyz, r12unit, f12);
  }
  if (L_max >= 2) {
    accumulate_f12_one_all_n<2>(n_max_angular_plus_1, d12inv, fn, fnp, scaled_sum_fxyz, r12unit, f12);
  }
  if (L_max >= 3) {
    accumulate_f12_one_all_n<3>(n_max_angular_plus_1, d12inv, fn, fnp, scaled_sum_fxyz, r12unit, f12);
  }
  if (L_max >= 4) {
    accumulate_f12_one_all_n<4>(n_max_angular_plus_1, d12inv, fn, fnp, scaled_sum_fxyz, r12unit, f12);
  }
  if (L_max >= 5) {
    accumulate_f12_one_all_n<5>(n_max_angular_plus_1, d12inv, fn, fnp, scaled_sum_fxyz, r12unit, f12);
  }
  if (L_max >= 6) {
    accumulate_f12_one_all_n<6>(n_max_angular_plus_1, d12inv, fn, fnp, scaled_sum_fxyz, r12unit, f12);
  }
  if (L_max >= 7) {
    accumulate_f12_one_all_n<7>(n_max_angular_plus_1, d12inv, fn, fnp, scaled_sum_fxyz, r12unit, f12);
  }
  if (L_max >= 8) {
    accumulate_f12_one_all_n<8>(n_max_angular_plus_1, d12inv, fn, fnp, scaled_sum_fxyz, r12unit, f12);
  }

  if (num_L <= L_max) {
    return;
  }

  if (!has_q_112 && !has_q_123 && !has_q_233 && !has_q_134) {
    accumulate_f12_q222_q1111_all_n(
      has_q_222, has_q_1111, n_max_angular_plus_1, d12, d12inv, r12, fn, fnp,
      q222_derivatives, q1111_derivatives, f12);
  } else {
    for (int n = 0; n < n_max_angular_plus_1; ++n) {
      accumulate_f12_extra_terms(
        L_max, has_q_222, has_q_1111, has_q_112, has_q_123, has_q_233, has_q_134, num_L, n,
        n_max_angular_plus_1, d12, d12inv, r12, fn[n], fnp[n], Fp, sum_fxyz, f12);
    }
  }
}

template <int L>
void accumulate_s_one(
  const double x12, const double y12, const double z12, const double fn, double* s)
{
  int s_index = L * L - 1;
  double z_pow[L + 1] = {1.0};
  for (int n = 1; n <= L; ++n) {
    z_pow[n] = z12 * z_pow[n - 1];
  }
  double real_part = x12;
  double imag_part = y12;
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    double z_factor = 0.0;
    for (int n2 = n2_start; n2 <= L - n1; n2 += 2) {
      z_factor += z_coefficient<L>(n1, n2) * z_pow[n2];
    }
    z_factor *= fn;
    if (n1 == 0) {
      s[s_index++] += z_factor;
    } else {
      s[s_index++] += z_factor * real_part;
      s[s_index++] += z_factor * imag_part;
      complex_product(x12, y12, real_part, imag_part);
    }
  }
}

void accumulate_s(
  const int L_max, const double d12, double x12, double y12, double z12, const double fn, double* s)
{
  double d12inv = 1.0 / d12;
  x12 *= d12inv;
  y12 *= d12inv;
  z12 *= d12inv;
  if (L_max >= 1) {
    accumulate_s_one<1>(x12, y12, z12, fn, s);
  }
  if (L_max >= 2) {
    accumulate_s_one<2>(x12, y12, z12, fn, s);
  }
  if (L_max >= 3) {
    accumulate_s_one<3>(x12, y12, z12, fn, s);
  }
  if (L_max >= 4) {
    accumulate_s_one<4>(x12, y12, z12, fn, s);
  }
  if (L_max >= 5) {
    accumulate_s_one<5>(x12, y12, z12, fn, s);
  }
  if (L_max >= 6) {
    accumulate_s_one<6>(x12, y12, z12, fn, s);
  }
  if (L_max >= 7) {
    accumulate_s_one<7>(x12, y12, z12, fn, s);
  }
  if (L_max >= 8) {
    accumulate_s_one<8>(x12, y12, z12, fn, s);
  }
}

template <int L>
double find_q_one(const double* s)
{
  const int start_index = L * L - 1;
  const int num_terms = 2 * L + 1;
  double q = 0.0;
  for (int k = 1; k < num_terms; ++k) {
    q += C3B[start_index + k] * s[start_index + k] * s[start_index + k];
  }
  q *= 2.0;
  q += C3B[start_index] * s[start_index] * s[start_index];
  return q;
}

void find_q(
  const int L_max,
  const int num_L,
  const int n_max_angular_plus_1,
  const int n,
  const double* s,
  double* q)
{
  if (L_max >= 1) {
    q[0 * n_max_angular_plus_1 + n] = find_q_one<1>(s);
  }
  if (L_max >= 2) {
    q[1 * n_max_angular_plus_1 + n] = find_q_one<2>(s);
  }
  if (L_max >= 3) {
    q[2 * n_max_angular_plus_1 + n] = find_q_one<3>(s);
  }
  if (L_max >= 4) {
    q[3 * n_max_angular_plus_1 + n] = find_q_one<4>(s);
  }
  if (L_max >= 5) {
    q[4 * n_max_angular_plus_1 + n] = find_q_one<5>(s);
  }
  if (L_max >= 6) {
    q[5 * n_max_angular_plus_1 + n] = find_q_one<6>(s);
  }
  if (L_max >= 7) {
    q[6 * n_max_angular_plus_1 + n] = find_q_one<7>(s);
  }
  if (L_max >= 8) {
    q[7 * n_max_angular_plus_1 + n] = find_q_one<8>(s);
  }
  if (num_L >= L_max + 1) {
    q[L_max * n_max_angular_plus_1 + n] =
      C4B[0] * s[3] * s[3] * s[3] + C4B[1] * s[3] * (s[4] * s[4] + s[5] * s[5]) +
      C4B[2] * s[3] * (s[6] * s[6] + s[7] * s[7]) + C4B[3] * s[6] * (s[5] * s[5] - s[4] * s[4]) +
      C4B[4] * s[4] * s[5] * s[7];
  }
  if (num_L >= L_max + 2) {
    double s0_sq = s[0] * s[0];
    double s1_sq_plus_s2_sq = s[1] * s[1] + s[2] * s[2];
    q[(L_max + 1) * n_max_angular_plus_1 + n] = C5B[0] * s0_sq * s0_sq +
                                                C5B[1] * s0_sq * s1_sq_plus_s2_sq +
                                                C5B[2] * s1_sq_plus_s2_sq * s1_sq_plus_s2_sq;
  }
}

void find_q(
  const int L_max,
  const int has_q_222,
  const int has_q_1111,
  const int has_q_112,
  const int has_q_123,
  const int has_q_233,
  const int has_q_134,
  const int n_max_angular_plus_1,
  const int n,
  const double* s,
  double* q)
{
  if (L_max >= 1) {
    q[0 * n_max_angular_plus_1 + n] = find_q_one<1>(s);
  }
  if (L_max >= 2) {
    q[1 * n_max_angular_plus_1 + n] = find_q_one<2>(s);
  }
  if (L_max >= 3) {
    q[2 * n_max_angular_plus_1 + n] = find_q_one<3>(s);
  }
  if (L_max >= 4) {
    q[3 * n_max_angular_plus_1 + n] = find_q_one<4>(s);
  }
  if (L_max >= 5) {
    q[4 * n_max_angular_plus_1 + n] = find_q_one<5>(s);
  }
  if (L_max >= 6) {
    q[5 * n_max_angular_plus_1 + n] = find_q_one<6>(s);
  }
  if (L_max >= 7) {
    q[6 * n_max_angular_plus_1 + n] = find_q_one<7>(s);
  }
  if (L_max >= 8) {
    q[7 * n_max_angular_plus_1 + n] = find_q_one<8>(s);
  }

  int L_index = L_max;

  if (has_q_222) {
    q[(L_index++) * n_max_angular_plus_1 + n] =
      C4B[0] * s[3] * s[3] * s[3] + C4B[1] * s[3] * (s[4] * s[4] + s[5] * s[5]) +
      C4B[2] * s[3] * (s[6] * s[6] + s[7] * s[7]) + C4B[3] * s[6] * (s[5] * s[5] - s[4] * s[4]) +
      C4B[4] * s[4] * s[5] * s[7];
  }

  if (has_q_1111) {
    double s0_sq = s[0] * s[0];
    double s1_sq_plus_s2_sq = s[1] * s[1] + s[2] * s[2];
    q[(L_index++) * n_max_angular_plus_1 + n] = C5B[0] * s0_sq * s0_sq +
                                                C5B[1] * s0_sq * s1_sq_plus_s2_sq +
                                                C5B[2] * s1_sq_plus_s2_sq * s1_sq_plus_s2_sq;
  }

  if (has_q_112) {
    q[(L_index++) * n_max_angular_plus_1 + n] =
      C4B2[0] * s[0] * s[0] * s[3] + C4B2[1] * s[0] * (s[1] * s[4] + s[2] * s[5]) +
      C4B2[2] * s[3] * (s[1] * s[1] + s[2] * s[2]) +
      C4B2[3] * s[6] * (s[1] * s[1] - s[2] * s[2]) + C4B2[4] * s[1] * s[2] * s[7];
  }

  if (has_q_123) {
    double val = 0.0;
    val += C4B_123[6] *
           (s[12] * s[2] * s[4] - s[11] * s[2] * s[5] + s[1] * s[11] * s[4] + s[1] * s[12] * s[5]);
    val += C4B_123[5] * (s[0] * s[11] * s[6] + s[0] * s[12] * s[7]);
    val += C4B_123[3] *
           (s[14] * s[2] * s[6] - s[13] * s[2] * s[7] + s[1] * s[13] * s[6] + s[1] * s[14] * s[7]);
    val += C4B_123[4] * (s[10] * s[0] * s[5] + s[0] * s[4] * s[9]);
    val += C4B_123[1] * (s[10] * s[2] * s[3] + s[0] * s[3] * s[8] + s[1] * s[3] * s[9]);
    val +=
      C4B_123[0] * (s[10] * s[2] * s[6] - s[10] * s[1] * s[7] - s[2] * s[7] * s[9] - s[1] * s[6] * s[9]);
    val += C4B_123[2] * (-s[2] * s[5] * s[8] - s[1] * s[4] * s[8]);
    q[(L_index++) * n_max_angular_plus_1 + n] = val;
  }

  if (has_q_233) {
    double val = 0.0;
    val += C4B_233[0] * (s[3] * s[8] * s[8]);
    val += C4B_233[1] * (s[10] * s[10] * s[3] + s[3] * s[9] * s[9]);
    val += C4B_233[2] * (-s[10] * s[10] * s[6] + s[6] * s[9] * s[9]);
    val += C4B_233[3] * (s[4] * s[8] * s[9] + s[10] * s[5] * s[8]);
    val += C4B_233[4] * (-s[13] * s[13] * s[3] - s[14] * s[14] * s[3]);
    val += C4B_233[5] *
           (-s[14] * s[7] * s[9] - s[13] * s[6] * s[9] - s[10] * s[14] * s[6] + s[10] * s[13] * s[7]);
    val += C4B_233[6] * (s[10] * s[7] * s[9]);
    val += C4B_233[7] * (-s[11] * s[6] * s[8] - s[12] * s[7] * s[8]);
    val += C4B_233[8] *
           (s[11] * s[4] * s[9] + s[12] * s[5] * s[9] + s[10] * s[12] * s[4] - s[10] * s[11] * s[5]);
    val += C4B_233[9] *
           (s[12] * s[14] * s[4] + s[11] * s[14] * s[5] + s[13] * s[11] * s[4] - s[13] * s[12] * s[5]);
    q[(L_index++) * n_max_angular_plus_1 + n] = val;
  }

  if (has_q_134) {
    q[(L_index++) * n_max_angular_plus_1 + n] =
      C4B_134[0] * (-s[10] * s[15] * s[2] - s[1] * s[15] * s[9]) +
      C4B_134[1] * (s[0] * s[15] * s[8]) +
      C4B_134[2] * (-s[1] * s[13] * s[18] - s[1] * s[14] * s[19] -
                    s[2] * s[14] * s[18] + s[2] * s[13] * s[19]) +
      C4B_134[3] * (-s[10] * s[18] * s[2] + s[1] * s[10] * s[19] +
                    s[1] * s[18] * s[9] + s[2] * s[19] * s[9]) +
      C4B_134[4] * (s[1] * s[16] * s[8] + s[2] * s[17] * s[8]) +
      C4B_134[5] * (s[0] * s[10] * s[17] + s[0] * s[16] * s[9] -
                    s[1] * s[11] * s[16] - s[1] * s[12] * s[17] -
                    s[2] * s[12] * s[16] + s[2] * s[11] * s[17]) +
      C4B_134[6] * (s[1] * s[13] * s[22] + s[1] * s[14] * s[23] -
                    s[2] * s[14] * s[22] + s[2] * s[13] * s[23]) +
      C4B_134[7] * (s[0] * s[11] * s[18] + s[0] * s[12] * s[19]) +
      C4B_134[8] * (s[0] * s[13] * s[20] + s[0] * s[14] * s[21]) +
      C4B_134[9] * (s[1] * s[11] * s[20] + s[1] * s[12] * s[21] -
                    s[2] * s[12] * s[20] + s[2] * s[11] * s[21]);
  }
}

#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
#ifndef NEP_TABLE_LENGTH
#define NEP_TABLE_LENGTH 32769
#endif
const int table_length = NEP_TABLE_LENGTH;
const int table_segments = table_length - 1;
const double table_resolution = 1.0 / table_segments;

void find_index_and_weight(
  const double d12_reduced,
  int& index_left,
  int& index_right,
  double& weight_left,
  double& weight_right)
{
  double d12_index = d12_reduced * table_segments;
  index_left = int(d12_index);
  if (index_left == table_segments) {
    --index_left;
  }
  index_right = index_left + 1;
  weight_right = d12_index - index_left;
  weight_left = 1.0 - weight_right;
}

double interpolate_table_value(
  const double* value,
  const double* derivative,
  const std::size_t index_left,
  const std::size_t index_right,
  const double weight_right,
  const double step)
{
  const double t = weight_right;
  const double t2 = t * t;
  const double t3 = t2 * t;
  const double y0 = value[index_left];
  const double y1 = value[index_right];
  const double yp0 = derivative[index_left];
  const double yp1 = derivative[index_right];
  return (2.0 * t3 - 3.0 * t2 + 1.0) * y0 + (t3 - 2.0 * t2 + t) * step * yp0 +
    (-2.0 * t3 + 3.0 * t2) * y1 + (t3 - t2) * step * yp1;
}

double interpolate_table_derivative(
  const double* value,
  const double* derivative,
  const std::size_t index_left,
  const std::size_t index_right,
  const double weight_right,
  const double step)
{
  const double t = weight_right;
  const double t2 = t * t;
  const double inv_step = 1.0 / step;
  const double y0 = value[index_left];
  const double y1 = value[index_right];
  const double yp0 = derivative[index_left];
  const double yp1 = derivative[index_right];
  return (6.0 * t2 - 6.0 * t) * inv_step * y0 + (3.0 * t2 - 4.0 * t + 1.0) * yp0 +
    (-6.0 * t2 + 6.0 * t) * inv_step * y1 + (3.0 * t2 - 2.0 * t) * yp1;
}

void construct_table_radial_or_angular(
  const int* slot_to_pair,
  const std::size_t table_pair_count,
  const int n_max,
  const int basis_size,
  const double* rc_pair,
  const double* rcinv_pair,
  const double* c_pair_major,
  double* gn,
  double* gnp)
{
  for (int table_index = 0; table_index < table_length; ++table_index) {
    for (std::size_t slot = 0; slot < table_pair_count; ++slot) {
      const int t12 = slot_to_pair[slot];
      const double rc = rc_pair[t12];
      const double rcinv = rcinv_pair[t12];
      const double d12 = table_index * table_resolution * rc;
      double fc12, fcp12;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      find_fn_and_fnp(basis_size, rcinv, d12, fc12, fcp12, fn12, fnp12);
      const double* c_pair =
        c_pair_major + t12 * (n_max + 1) * (basis_size + 1);
      for (int n = 0; n <= n_max; ++n) {
        double gn12 = 0.0;
        double gnp12 = 0.0;
        const double* c_n = c_pair + n * (basis_size + 1);
        for (int k = 0; k <= basis_size; ++k) {
          gn12 += fn12[k] * c_n[k];
          gnp12 += fnp12[k] * c_n[k];
        }
        const std::size_t index_all =
          (static_cast<std::size_t>(table_index) * table_pair_count + slot) * (n_max + 1) + n;
        gn[index_all] = gn12;
        gnp[index_all] = gnp12;
      }
    }
  }
}
#endif

} 

#undef NEP_RESTRICT
