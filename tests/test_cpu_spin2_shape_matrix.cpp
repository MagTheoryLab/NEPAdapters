#include "spin2_polynomial_cpu.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

using nep_adapters::common::SpinPolynomialLayout;
using nep_adapters::cpu_spin2::CenterScratch;
using nep_adapters::cpu_spin2::Edge;

constexpr double kStep = 2.0e-6;
constexpr double kTolerance = 3.0e-6;

int torchnep_descriptor_dim(
    int channels, int l_max, int order, int soc) {
  const int pairs = channels * (channels + 1) / 2;
  int dim = 1 + 2 * channels;
  if (soc != 0 && l_max >= 2) dim += 2 * channels;
  if (order >= 2) {
    dim += 2 * channels;
    if (l_max >= 1) dim += (soc != 0 ? 3 : 1) * channels;
    if (l_max >= 2) dim += channels;
    dim += channels + 2 * pairs;
    if (soc != 0 && l_max >= 1) {
      if (channels >= 2) dim += channels;
      dim += channels;
    }
    if (soc != 0 && l_max >= 2) {
      if (channels >= 2) dim += channels;
      dim += channels;
    }
  }
  if (order >= 3) {
    dim += channels;
    if (soc != 0 && l_max >= 1) {
      if (channels >= 2) dim += channels;
      dim += channels;
    }
    if (soc != 0 && l_max >= 2) {
      if (channels >= 2) dim += channels;
      dim += channels;
      if (channels >= 2) dim += channels;
    }
    if (soc != 0 && l_max >= 1 && channels >= 3) dim += channels;
  }
  return dim;
}

struct ShapeCase {
  SpinPolynomialLayout layout;
  std::array<double, 3> center_spin = {0.43, -0.27, 0.61};
  std::vector<double> spins = {
      0.43, -0.27, 0.61,
      -0.31, 0.72, 0.18,
      0.57, 0.11, -0.46};
  std::vector<Edge> edges{Edge{}, Edge{}};
  std::vector<std::array<double, nep_adapters::cpu_spin2::kMaxChannels>>
      weight_offsets{{}, {}};
  std::vector<double> projection;
  std::vector<double> descriptor_gradient;
};

void refresh_edges(ShapeCase& shape) {
  for (std::size_t edge_index = 0; edge_index < shape.edges.size(); ++edge_index) {
    Edge& edge = shape.edges[edge_index];
    edge.distance = std::sqrt(
        edge.displacement[0] * edge.displacement[0] +
        edge.displacement[1] * edge.displacement[1] +
        edge.displacement[2] * edge.displacement[2]);
    for (int channel = 0; channel < shape.layout.channels; ++channel) {
      edge.weights[channel] = shape.weight_offsets[edge_index][channel] +
          edge.derivatives[channel] * edge.distance;
    }
  }
}

ShapeCase make_shape(int channels, int l_max, int order, int soc) {
  ShapeCase shape;
  shape.layout = nep_adapters::common::make_spin_polynomial_layout(
      channels, l_max, order, soc);
  shape.edges[0].neighbor = 1;
  shape.edges[0].displacement[0] = 1.21;
  shape.edges[0].displacement[1] = -0.37;
  shape.edges[0].displacement[2] = 0.42;
  shape.edges[1].neighbor = 2;
  shape.edges[1].displacement[0] = -0.64;
  shape.edges[1].displacement[1] = 1.08;
  shape.edges[1].displacement[2] = 0.29;
  for (std::size_t edge_index = 0; edge_index < shape.edges.size(); ++edge_index) {
    for (int channel = 0; channel < channels; ++channel) {
      shape.weight_offsets[edge_index][channel] =
          0.17 + 0.013 * channel + 0.021 * static_cast<double>(edge_index);
      shape.edges[edge_index].derivatives[channel] =
          -0.031 + 0.004 * channel - 0.003 * static_cast<double>(edge_index);
    }
  }
  refresh_edges(shape);
  shape.projection.resize(static_cast<std::size_t>(4 * channels * channels));
  for (std::size_t index = 0; index < shape.projection.size(); ++index) {
    shape.projection[index] = 0.09 * std::sin(0.37 * static_cast<double>(index + 1));
  }
  shape.descriptor_gradient.resize(
      static_cast<std::size_t>(shape.layout.descriptor_dim));
  for (std::size_t index = 0; index < shape.descriptor_gradient.size(); ++index) {
    shape.descriptor_gradient[index] =
        0.11 * std::cos(0.23 * static_cast<double>(index + 1));
  }
  return shape;
}

double evaluate(ShapeCase& shape) {
  refresh_edges(shape);
  CenterScratch scratch;
  nep_adapters::cpu_spin2::build_state(
      shape.layout, shape.center_spin.data(), shape.spins.data(),
      shape.edges, scratch.state);
  std::vector<double> descriptor(
      static_cast<std::size_t>(shape.layout.descriptor_dim), 0.0);
  nep_adapters::cpu_spin2::descriptors(
      shape.layout, shape.center_spin.data(), shape.projection.data(),
      scratch.state, descriptor.data());
  double value = 0.0;
  for (int index = 0; index < shape.layout.descriptor_dim; ++index) {
    value += descriptor[index] * shape.descriptor_gradient[index];
  }
  return value;
}

bool close_gradient(
    double analytical, double numerical, int channels, int l_max,
    int order, int soc, const char* variable, int index) {
  const double error = std::abs(analytical - numerical);
  const double scale = 1.0 + std::max(std::abs(analytical), std::abs(numerical));
  if (error <= kTolerance * scale) return true;
  std::cerr << "gradient mismatch C=" << channels
            << " L=" << l_max << " O=" << order << " SOC=" << soc
            << " variable=" << variable << '[' << index << ']'
            << " analytical=" << analytical
            << " numerical=" << numerical
            << " error=" << error << '\n';
  return false;
}

template <typename Access>
double finite_difference(ShapeCase& shape, Access access) {
  double& variable = access(shape);
  const double original = variable;
  variable = original + kStep;
  const double plus = evaluate(shape);
  variable = original - kStep;
  const double minus = evaluate(shape);
  variable = original;
  return (plus - minus) / (2.0 * kStep);
}

bool check_shape(int channels, int l_max, int order, int soc) {
  ShapeCase shape = make_shape(channels, l_max, order, soc);
  const int expected_dim =
      torchnep_descriptor_dim(channels, l_max, order, soc);
  if (shape.layout.descriptor_dim != expected_dim) {
    std::cerr << "descriptor dimension mismatch C=" << channels
              << " L=" << l_max << " O=" << order << " SOC=" << soc
              << " actual=" << shape.layout.descriptor_dim
              << " expected=" << expected_dim << '\n';
    return false;
  }
  CenterScratch scratch;
  nep_adapters::cpu_spin2::build_state(
      shape.layout, shape.center_spin.data(), shape.spins.data(),
      shape.edges, scratch.state);
  double center_gradient[3] = {};
  nep_adapters::cpu_spin2::gradients(
      shape.layout, shape.center_spin.data(), shape.spins.data(),
      shape.projection.data(), shape.edges, shape.descriptor_gradient.data(),
      scratch, center_gradient);

  bool ok = true;
  for (int component = 0; component < 3; ++component) {
    const double numerical = finite_difference(
        shape, [component](ShapeCase& value) -> double& {
          value.spins[component] = value.center_spin[component];
          return value.center_spin[component];
        });
    shape.spins[component] = shape.center_spin[component];
    ok = close_gradient(
        center_gradient[component], numerical, channels, l_max, order, soc,
        "center_spin", component) && ok;
  }
  for (std::size_t edge_index = 0; edge_index < shape.edges.size(); ++edge_index) {
    const int neighbor = shape.edges[edge_index].neighbor;
    for (int component = 0; component < 3; ++component) {
      const int flat = 3 * neighbor + component;
      const double numerical = finite_difference(
          shape, [flat](ShapeCase& value) -> double& { return value.spins[flat]; });
      ok = close_gradient(
          scratch.edge_gradients[edge_index].neighbor_spin[component], numerical,
          channels, l_max, order, soc, "neighbor_spin", flat) && ok;
    }
    for (int component = 0; component < 3; ++component) {
      const double numerical = finite_difference(
          shape, [edge_index, component](ShapeCase& value) -> double& {
            return value.edges[edge_index].displacement[component];
          });
      ok = close_gradient(
          scratch.edge_gradients[edge_index].position[component], numerical,
          channels, l_max, order, soc, "displacement",
          static_cast<int>(3 * edge_index + component)) && ok;
    }
  }
  return ok;
}

bool check_rank_one_pruning_and_survivors() {
  ShapeCase shape = make_shape(1, 2, 3, 1);
  const SpinPolynomialLayout& layout = shape.layout;
  if (layout.descriptor_dim != 19 ||
      layout.coupling_l11_axial >= 0 ||
      layout.coupling_l22_axial >= 0 ||
      layout.coupling_l11_dot_response >= 0 ||
      layout.coupling_l22_dot_response >= 0 ||
      layout.coupling_l112_edge_response >= 0 ||
      layout.coupling_l111_bulk >= 0 ||
      layout.edge_l11_axial < 0 ||
      layout.edge_l22_axial < 0 ||
      layout.edge_l0_moment_gate < 0 ||
      layout.coupling_l111_p_m_x < 0 ||
      layout.coupling_l111_p_qs_x < 0) {
    std::cerr << "rank-one radial pruning contract mismatch\n";
    return false;
  }

  CenterScratch scratch;
  nep_adapters::cpu_spin2::build_state(
      layout, shape.center_spin.data(), shape.spins.data(),
      shape.edges, scratch.state);
  std::vector<double> descriptor(
      static_cast<std::size_t>(layout.descriptor_dim), 0.0);
  nep_adapters::cpu_spin2::descriptors(
      layout, shape.center_spin.data(), shape.projection.data(),
      scratch.state, descriptor.data());
  for (int offset : {
           layout.edge_l11_axial,
           layout.edge_l22_axial,
           layout.edge_l0_moment_gate,
           layout.coupling_l111_p_m_x,
           layout.coupling_l111_p_qs_x}) {
    if (std::abs(descriptor[static_cast<std::size_t>(offset)]) <= 1.0e-12) {
      std::cerr << "rank-one surviving channel is inactive at offset "
                << offset << '\n';
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!check_rank_one_pruning_and_survivors()) return EXIT_FAILURE;
  int checked = 0;
  for (int channels = 1; channels <= 9; ++channels) {
    for (int l_max = 0; l_max <= 2; ++l_max) {
      for (int order = 1; order <= 3; ++order) {
        for (int soc = 0; soc <= 1; ++soc) {
          if (!check_shape(channels, l_max, order, soc)) return EXIT_FAILURE;
          ++checked;
        }
      }
    }
  }
  std::cout << "validated " << checked
            << " spin polynomial shape gradients\n";
  return checked == 162 ? EXIT_SUCCESS : EXIT_FAILURE;
}
