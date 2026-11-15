#pragma once

#include "../../common/spin_density_layout.hpp"
#include <array>
#include <cmath>
#include <vector>

namespace nep_adapters::cpu_spin {
using spin_density::Layout;
struct Value { double value = 0; std::array<double, 3> derivative{}; };
struct Edge {
  int neighbor = -1;
  double displacement[3]{}, distance = 0;
  double weights[9]{}, derivatives[9]{};
};
struct EdgeGradient {
  int neighbor;
  double displacement[3]{}, position[3]{}, neighbor_spin[3]{};
};
struct CenterScratch {
  std::vector<double> moments, pull;
  // Forward density and backward contraction are never live together.
  std::vector<double> neighbor_tensor;
  std::vector<Value> center_contraction;
  std::vector<Value> harmonics;
  std::vector<double> harmonic_planes;
  std::vector<EdgeGradient> edge_gradients;
};
inline double power(double x, int n) {
  double result = 1;
  for (int i = 0; i < n; ++i) result *= x;
  return result;
}
inline void spherical(const Layout& layout, const double* x, std::vector<Value>& y) {
  y.assign((layout.lmax + 1) * (layout.lmax + 1), Value{});
  // The harmonic table uses degrees 0..4; reuse powers across its monomials.
  double powers[3][5];
  for (int d = 0; d < 3; ++d) {
    powers[d][0] = 1.0;
    for (int n = 1; n <= layout.lmax && n < 5; ++n)
      powers[d][n] = powers[d][n-1] * x[d];
  }
  for (const auto& h : spin_density::harmonics) {
    if (h.component >= static_cast<int>(y.size())) continue;
    const int p[3] = {h.x, h.y, h.z};
    double val = h.value;
    for (int d = 0; d < 3; ++d) val *= powers[d][p[d]];
    y[h.component].value += val;
    for (int d = 0; d < 3; ++d) {
      if (!p[d]) continue;
      double grad = h.value * p[d];
      for (int k = 0; k < 3; ++k) grad *= powers[k][p[k] - (d == k)];
      y[h.component].derivative[d] += grad;
    }
  }
}
inline void primitives_at(const Layout& layout, const double* s, Value* phi) {
  std::vector<Value> y;
  spherical(layout, s, y);
  const double s2 = s[0]*s[0] + s[1]*s[1] + s[2]*s[2];
  for (const auto& p : layout.primitives) {
    const int exponent = (p.degree - p.l) / 2;
    const double factor = power(s2, exponent);
    const double gradient = exponent ? 2 * exponent * power(s2, exponent - 1) : 0;
    for (int m = 0; m < 2 * p.l + 1; ++m) {
      auto& out = phi[p.offset + m];
      const auto& h = y[p.l * p.l + m];
      out.value = factor * h.value;
      for (int d = 0; d < 3; ++d)
        out.derivative[d] = factor * h.derivative[d] + gradient * s[d] * h.value;
    }
  }
}
// Parameters here start at the path weights; radial mixing has already been
// contracted with the structural angular coefficients by the caller.
inline void forward(const Layout& layout, int atom, const Value* primitives,
                    const double* weights, const std::vector<Edge>& edges,
                    CenterScratch& scratch, double* q) {
  const int cmax = layout.channels;
  const int geometry_count = (layout.lmax + 1) * (layout.lmax + 1);
  scratch.moments.assign(cmax * layout.components, 0);
  scratch.neighbor_tensor.assign(cmax * layout.spin_components * geometry_count, 0.0);
  const Value* center = primitives + static_cast<std::size_t>(atom) * layout.spin_components;
  for (const auto& edge : edges) {
    double u[3];
    for (int d = 0; d < 3; ++d) u[d] = edge.displacement[d] / edge.distance;
    spherical(layout, u, scratch.harmonics);
    const Value* neighbor = primitives + static_cast<std::size_t>(edge.neighbor) * layout.spin_components;
    for (int c = 0; c < cmax; ++c)
      for (int a = 0; a < layout.spin_components; ++a) {
        const double weighted_spin = edge.weights[c] * (a ? neighbor[a].value : 1.0);
        double* density = scratch.neighbor_tensor.data() +
          (c * layout.spin_components + a) * geometry_count;
        for (int g = 0; g < geometry_count; ++g)
          density[g] += weighted_spin * scratch.harmonics[g].value;
      }
  }
  // Couple the aggregated neighbor density once per center, not once per edge.
  for (const auto& term : layout.terms)
    for (int c = 0; c < cmax; ++c) {
      const double* density = scratch.neighbor_tensor.data() + c * layout.spin_components * geometry_count;
      const double product = term.neighbor ? density[term.spin * geometry_count + term.geometry] :
        density[term.geometry] * center[term.spin].value;
      scratch.moments[c * layout.components + term.output] +=
        weights[term.path * cmax + c] * term.coefficient * product;
    }
  const double* center_weights = weights + layout.path_count * cmax;
  for (const auto& term : layout.centers)
    for (int c = 0; c < cmax; ++c)
      scratch.moments[c * layout.components + term.output] +=
        center_weights[term.weight * cmax + c] * center[term.primitive].value;
  int column = 0;
  for (std::size_t p = 1; p < layout.primitives.size(); ++p)
    if (layout.primitives[p].l == 0) q[column++] = center[layout.primitives[p].offset].value;
  for (const auto& sector : layout.sectors)
    for (int c = 0; c < cmax; ++c) {
      const double* h = scratch.moments.data() + c * layout.components + sector.offset;
      double norm = 0;
      for (int m = 0; m < sector.size; ++m) norm += h[m] * h[m];
      q[sector.norm_column + c] = norm / sector.size;
      if (sector.scalar_column >= 0) q[sector.scalar_column + c] = h[0];
    }
}
inline void gradients(const Layout& layout, int atom, const Value* primitives,
                      const double* weights, const std::vector<Edge>& edges,
                      const double* fp, CenterScratch& scratch, double* center_gradient) {
  const int cmax = layout.channels;
  const Value* center = primitives + static_cast<std::size_t>(atom) * layout.spin_components;
  scratch.pull.assign(cmax * layout.components, 0);
  for (const auto& sector : layout.sectors)
    for (int c = 0; c < cmax; ++c) {
      const int offset = c * layout.components + sector.offset;
      for (int m = 0; m < sector.size; ++m)
        scratch.pull[offset + m] = 2 * fp[sector.norm_column + c] *
          scratch.moments[offset + m] / sector.size;
      if (sector.scalar_column >= 0) scratch.pull[offset] += fp[sector.scalar_column + c];
    }
  const double* center_weights = weights + layout.path_count * cmax;
  for (const auto& term : layout.centers)
    for (int c = 0; c < cmax; ++c)
      for (int d = 0; d < 3; ++d)
        center_gradient[d] += scratch.pull[c * layout.components + term.output] *
          center_weights[term.weight * cmax + c] * center[term.primitive].derivative[d];
  int column = 0;
  for (std::size_t p = 1; p < layout.primitives.size(); ++p)
    if (layout.primitives[p].l == 0) {
      for (int d = 0; d < 3; ++d)
        center_gradient[d] += fp[column] * center[layout.primitives[p].offset].derivative[d];
      ++column;
    }
  scratch.edge_gradients.clear();
  const int geometry_count = (layout.lmax + 1) * (layout.lmax + 1);
  scratch.center_contraction.assign(cmax * geometry_count, Value{});
  scratch.harmonic_planes.resize(4 * geometry_count);
  scratch.neighbor_tensor.assign(cmax * layout.spin_components * geometry_count, 0.0);
  // Contract the center adjoints once, before traversing the neighbor edges.
  for (const auto& term : layout.terms)
    for (int c = 0; c < cmax; ++c) {
      const double adjoint = scratch.pull[c * layout.components + term.output] *
        weights[term.path * cmax + c] * term.coefficient;
      if (term.neighbor) {
        scratch.neighbor_tensor[
          (c * layout.spin_components + term.spin) * geometry_count + term.geometry] += adjoint;
      } else {
        auto& contracted = scratch.center_contraction[c * geometry_count + term.geometry];
        const auto& spin = center[term.spin];
        contracted.value += adjoint * spin.value;
        for (int d = 0; d < 3; ++d)
          contracted.derivative[d] += adjoint * spin.derivative[d];
      }
    }
  for (const auto& edge : edges) {
    EdgeGradient grad{};
    grad.neighbor = edge.neighbor;
    double u[3], angular[3]{}, radial = 0;
    for (int d = 0; d < 3; ++d) {
      grad.displacement[d] = edge.displacement[d];
      u[d] = edge.displacement[d] / edge.distance;
    }
    spherical(layout, u, scratch.harmonics);
    // Deinterleave once per edge instead of inside every spin contraction.
    for (int g = 0; g < geometry_count; ++g) {
      const auto& y = scratch.harmonics[g];
      scratch.harmonic_planes[g] = y.value;
      for (int d = 0; d < 3; ++d)
        scratch.harmonic_planes[(d+1)*geometry_count+g] = y.derivative[d];
    }
    const Value* neighbor = primitives + static_cast<std::size_t>(edge.neighbor) * layout.spin_components;
    for (int c = 0; c < cmax; ++c) {
      for (int g = 0; g < geometry_count; ++g) {
        const auto& contracted = scratch.center_contraction[c * geometry_count + g];
        const auto& y = scratch.harmonics[g];
        radial += contracted.value * edge.derivatives[c] * y.value;
        for (int d = 0; d < 3; ++d) {
          angular[d] += contracted.value * edge.weights[c] * y.derivative[d];
          center_gradient[d] += contracted.derivative[d] * edge.weights[c] * y.value;
        }
      }
      for (int a = 1; a < layout.spin_components; ++a) {
        const double* coefficients = scratch.neighbor_tensor.data() +
          (c * layout.spin_components + a) * geometry_count;
        const double* y = scratch.harmonic_planes.data();
        const double* yx = y + geometry_count;
        const double* yy = yx + geometry_count;
        const double* yz = yy + geometry_count;
        double value = 0, dx = 0, dy = 0, dz = 0;
        for (int g = 0; g < geometry_count; ++g) {
          value += coefficients[g] * y[g];
          dx += coefficients[g] * yx[g];
          dy += coefficients[g] * yy[g];
          dz += coefficients[g] * yz[g];
        }
        const double direction[3] = {dx, dy, dz};
        const auto& spin = neighbor[a];
        radial += value * edge.derivatives[c] * spin.value;
        for (int d = 0; d < 3; ++d) {
          angular[d] += edge.weights[c] * direction[d] * spin.value;
          grad.neighbor_spin[d] += edge.weights[c] * value * spin.derivative[d];
        }
      }
    }
    double longitudinal = 0;
    for (int d = 0; d < 3; ++d) longitudinal += angular[d] * u[d];
    for (int d = 0; d < 3; ++d)
      grad.position[d] = radial * u[d] + (angular[d] - longitudinal * u[d]) / edge.distance;
    scratch.edge_gradients.push_back(grad);
  }
}
}  // namespace nep_adapters::cpu_spin
