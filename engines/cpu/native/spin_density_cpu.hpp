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
  std::vector<Value> center, neighbor, harmonics;
  std::vector<EdgeGradient> edge_gradients;
};
inline double power(double x, int n) {
  double result = 1;
  for (int i = 0; i < n; ++i) result *= x;
  return result;
}
inline void spherical(const Layout& layout, const double* x, std::vector<Value>& y) {
  y.assign((layout.lmax + 1) * (layout.lmax + 1), Value{});
  for (const auto& h : spin_density::harmonics) {
    if (h.component >= static_cast<int>(y.size())) continue;
    const int p[3] = {h.x, h.y, h.z};
    double val = h.value;
    for (int d = 0; d < 3; ++d) val *= power(x[d], p[d]);
    y[h.component].value += val;
    for (int d = 0; d < 3; ++d) {
      if (!p[d]) continue;
      double grad = h.value * p[d];
      for (int k = 0; k < 3; ++k) grad *= power(x[k], p[k] - (d == k));
      y[h.component].derivative[d] += grad;
    }
  }
}
inline void primitives_at(const Layout& layout, const double* s, std::vector<Value>& phi) {
  std::vector<Value> y;
  spherical(layout, s, y);
  phi.assign(layout.spin_components, Value{});
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
inline void forward(const Layout& layout, const double* si, const double* spins,
                    const double* weights, const std::vector<Edge>& edges,
                    CenterScratch& scratch, double* q) {
  const int cmax = layout.channels;
  scratch.moments.assign(cmax * layout.components, 0);
  primitives_at(layout, si, scratch.center);
  for (const auto& edge : edges) {
    double u[3];
    for (int d = 0; d < 3; ++d) u[d] = edge.displacement[d] / edge.distance;
    spherical(layout, u, scratch.harmonics);
    primitives_at(layout, spins + 3 * edge.neighbor, scratch.neighbor);
    for (const auto& term : layout.terms) {
      const double spin = (term.neighbor ? scratch.neighbor : scratch.center)[term.spin].value;
      const double product = term.coefficient * scratch.harmonics[term.geometry].value * spin;
      for (int c = 0; c < cmax; ++c)
        scratch.moments[c * layout.components + term.output] +=
          edge.weights[c] * weights[term.path * cmax + c] * product;
    }
  }
  const double* center_weights = weights + layout.path_count * cmax;
  for (const auto& term : layout.centers)
    for (int c = 0; c < cmax; ++c)
      scratch.moments[c * layout.components + term.output] +=
        center_weights[term.weight * cmax + c] * scratch.center[term.primitive].value;
  int column = 0;
  for (std::size_t p = 1; p < layout.primitives.size(); ++p)
    if (layout.primitives[p].l == 0) q[column++] = scratch.center[layout.primitives[p].offset].value;
  for (const auto& sector : layout.sectors)
    for (int c = 0; c < cmax; ++c) {
      const double* h = scratch.moments.data() + c * layout.components + sector.offset;
      double norm = 0;
      for (int m = 0; m < sector.size; ++m) norm += h[m] * h[m];
      q[sector.norm_column + c] = norm / sector.size;
      if (sector.scalar_column >= 0) q[sector.scalar_column + c] = h[0];
    }
}
inline void gradients(const Layout& layout, const double* si, const double* spins,
                      const double* weights, const std::vector<Edge>& edges,
                      const double* fp, CenterScratch& scratch, double* center_gradient) {
  const int cmax = layout.channels;
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
          center_weights[term.weight * cmax + c] * scratch.center[term.primitive].derivative[d];
  int column = 0;
  for (std::size_t p = 1; p < layout.primitives.size(); ++p)
    if (layout.primitives[p].l == 0) {
      for (int d = 0; d < 3; ++d)
        center_gradient[d] += fp[column] * scratch.center[layout.primitives[p].offset].derivative[d];
      ++column;
    }
  scratch.edge_gradients.clear();
  for (const auto& edge : edges) {
    EdgeGradient grad{};
    grad.neighbor = edge.neighbor;
    double u[3], angular[3]{}, radial = 0;
    for (int d = 0; d < 3; ++d) {
      grad.displacement[d] = edge.displacement[d];
      u[d] = edge.displacement[d] / edge.distance;
    }
    spherical(layout, u, scratch.harmonics);
    primitives_at(layout, spins + 3 * edge.neighbor, scratch.neighbor);
    for (const auto& term : layout.terms) {
      const auto& spin = (term.neighbor ? scratch.neighbor : scratch.center)[term.spin];
      const auto& y = scratch.harmonics[term.geometry];
      for (int c = 0; c < cmax; ++c) {
        const double adjoint = scratch.pull[c * layout.components + term.output] *
          weights[term.path * cmax + c] * term.coefficient;
        radial += adjoint * edge.derivatives[c] * y.value * spin.value;
        for (int d = 0; d < 3; ++d) {
          angular[d] += adjoint * edge.weights[c] * y.derivative[d] * spin.value;
          (term.neighbor ? grad.neighbor_spin : center_gradient)[d] +=
            adjoint * edge.weights[c] * y.value * spin.derivative[d];
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
