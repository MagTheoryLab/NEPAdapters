/* Adapted from GPUMD 73f6b09. SPDX-License-Identifier: LGPL-3.0-or-later */
#pragma once
#include "model_protocol.hpp"

namespace nep_adapters::spin_density {
struct PrimitiveBasis {
  Primitive entries[9] = {};
  int count = 0, components = 0, scalars = 0;
#if defined(__CUDACC__)
  __host__ __device__
#endif
  constexpr PrimitiveBasis(int lmax, int order)
  {
    for (int d = 0; d <= order; ++d)
      for (int j = d % 2; j <= d && j <= lmax; j += 2) {
        entries[count++] = {d, j, components};
        components += 2 * j + 1;
        if (d && j == 0)
          ++scalars;
      }
  }
};
constexpr int md_force_lanes(int l, int order) {
  const int h = (l+1)*(l+1);
  return l == 0 ? 1 : (order > 2 && h*PrimitiveBasis(l,order).components >= 256 ? 32 : 4);
}
constexpr bool md_warp_density(int harmonics, int spins) { return harmonics*spins >= 24; }
constexpr bool md_wide(int l, int order, bool soc) { return l == 4 && order == 4 && !soc; }
constexpr bool md_quadratic(int l, int order) { return l >= 2 && order == 2; }
constexpr bool md_compact_rows(int lanes, int elements, bool fused) {
  return lanes == 32 && fused && elements <= 10*32;
}
inline bool md_paired(const Layout& layout, bool soc) {
  const int lanes = md_force_lanes(layout.lmax,layout.order);
  const int harmonics = (layout.lmax+1)*(layout.lmax+1);
  const int elements = harmonics*layout.spin_components;
  return lanes > 1 && !md_quadratic(layout.lmax,layout.order) &&
    !md_wide(layout.lmax,layout.order,soc) &&
    !(md_compact_rows(lanes,elements,layout.channels > 1) && layout.terms.size() <= std::size_t(2*elements));
}
inline std::size_t md_radial_slots(int l, int order, int neighbors) {
  const int lanes = md_force_lanes(l,order);
  return (static_cast<std::size_t>(neighbors) + lanes - 1) / lanes * lanes;
}

struct ProjectionRow {
  int output, begin, count;
};
struct alignas(8) ProjectionTerm {
  unsigned short input;
  short center_spin;
  float weight;
};

struct SpinProjection {
  enum { width = 32 };
  std::vector<ProjectionRow> rows;
  std::vector<ProjectionTerm> packed;
  std::vector<float> neighbor_weights;
  SpinProjection(const Layout& layout, const float* weights, bool transpose, bool paired = false)
  {
    const int harmonics = (layout.lmax + 1) * (layout.lmax + 1);
    const int outputs = transpose ? harmonics * layout.spin_components : layout.components;
    auto density_index = [&](const Term& t) {
      return t.geometry * layout.spin_components + (t.neighbor ? t.spin : 0);
    };
    std::vector<std::vector<int>> by_output(outputs);
    for (int t = 0; t < static_cast<int>(layout.terms.size()); ++t) {
      const Term& term = layout.terms[t];
      const int density = paired ? term.geometry * layout.spin_components + term.spin : density_index(term);
      by_output[transpose ? density : term.output].push_back(t);
    }
    // Center and neighbor couplings for a given (geometry, spin, output) use the same moment.
    std::vector<int> partner(paired ? layout.terms.size() : 0, -1);
    if (paired) {
      for (auto& row : by_output) {
        std::vector<int> centers;
        for (int t : row)
          if (!layout.terms[t].neighbor)
            centers.push_back(t);
        for (int t : row) {
          if (!layout.terms[t].neighbor)
            continue;
          auto match = std::find_if(centers.begin(), centers.end(), [&](int local) {
            return partner[local] < 0 && layout.terms[local].output == layout.terms[t].output;
          });
          if (match == centers.end()) {
            centers.push_back(t);
            partner[t] = t;
          } else {
            partner[*match] = t;
          }
        }
        row.swap(centers);
      }
    }
    for (int m = 0; m < outputs; ++m)
      rows.push_back({m, 0, static_cast<int>(by_output[m].size())});
    // Group similar-length rows so lanes read adjacent entries and finish together.
    std::stable_sort(rows.begin(), rows.end(), [](ProjectionRow a, ProjectionRow b) {
      return a.count > b.count;
    });
    for (int base = 0; base < outputs; base += width) {
      const int start = static_cast<int>(packed.size());
      packed.resize(start + width * rows[base].count * layout.channels);
      if (paired)
        neighbor_weights.resize(packed.size());
      for (int lane = 0; lane < width && base + lane < outputs; ++lane) {
        auto& row = rows[base + lane];
        row.begin = start + lane;
        for (int k = 0; k < row.count; ++k) {
          const int t = by_output[row.output][k];
          const Term& term = layout.terms[t];
          const int input = transpose ? term.output : density_index(term);
          for (int c = 0; c < layout.channels; ++c) {
            const int offset = row.begin + (k * layout.channels + c) * width;
            packed[offset] = {
              static_cast<unsigned short>(input),
              static_cast<short>(term.neighbor ? -1 : term.spin),
              (paired && term.neighbor) ? 0.0f : weights[term.path * layout.channels + c] * term.coefficient};
            if (paired && partner[t] >= 0) {
              const auto& neighbor = layout.terms[partner[t]];
              neighbor_weights[offset] = weights[neighbor.path * layout.channels + c] * neighbor.coefficient;
            }
          }
        }
      }
    }
  }
};
struct ProjectionView {
  const ProjectionRow* rows;
  const ProjectionTerm* terms;
  const float* neighbor_weights;
};
} // namespace nep_adapters::spin_density
