#include "device_model.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace nep_adapters::cuda_backend {
namespace {

constexpr std::array<float, 94> kCovalentRadius = {
    0.426667f, 0.613333f, 1.6f, 1.25333f, 1.02667f, 1.0f, 0.946667f,
    0.84f, 0.853333f, 0.893333f, 1.86667f, 1.66667f, 1.50667f, 1.38667f,
    1.46667f, 1.36f, 1.32f, 1.28f, 2.34667f, 2.05333f, 1.77333f, 1.62667f,
    1.61333f, 1.46667f, 1.42667f, 1.38667f, 1.33333f, 1.32f, 1.34667f,
    1.45333f, 1.49333f, 1.45333f, 1.53333f, 1.46667f, 1.52f, 1.56f, 2.52f,
    2.22667f, 1.96f, 1.85333f, 1.76f, 1.65333f, 1.53333f, 1.50667f,
    1.50667f, 1.44f, 1.53333f, 1.64f, 1.70667f, 1.68f, 1.68f, 1.64f,
    1.76f, 1.74667f, 2.78667f, 2.34667f, 2.16f, 1.96f, 2.10667f, 2.09333f,
    2.08f, 2.06667f, 2.01333f, 2.02667f, 2.01333f, 2.0f, 1.98667f,
    1.98667f, 1.97333f, 2.04f, 1.94667f, 1.82667f, 1.74667f, 1.64f,
    1.57333f, 1.54667f, 1.48f, 1.49333f, 1.50667f, 1.76f, 1.73333f,
    1.73333f, 1.81333f, 1.74667f, 1.84f, 1.89333f, 2.68f, 2.41333f,
    2.22667f, 2.10667f, 2.02667f, 2.04f, 2.05333f, 2.06667f};

constexpr std::array<float, 8> kUniversalZbl = {
    0.18175f, 3.1998f, 0.50986f, 0.94229f,
    0.28022f, 0.4029f, 0.02817f, 0.20162f};

std::size_t symmetric_pair_index(
    std::size_t type1,
    std::size_t type2,
    std::size_t num_types) {
  const std::size_t low = std::min(type1, type2);
  const std::size_t high = std::max(type1, type2);
  return low * num_types - low * (low - 1) / 2 + (high - low);
}

std::size_t checked_count(int value, const char* name) {
  if (value < 0) {
    throw std::runtime_error(std::string(name) + " must be non-negative");
  }
  return static_cast<std::size_t>(value);
}

void append_range(
    std::vector<float>& dst,
    const std::vector<float>& src,
    std::size_t offset,
    std::size_t count) {
  if (offset + count > src.size()) {
    throw std::runtime_error("parameter section exceeds raw parameter buffer");
  }
  dst.insert(dst.end(), src.begin() + offset, src.begin() + offset + count);
}

}  // namespace

HostModelParameters load_host_model_parameters(const std::string& model_path) {
  ParsedModelFile parsed = parse_model_file(model_path);
  const ModelProtocol& protocol = parsed.protocol;
  const std::vector<float>& raw = parsed.parameters_and_q_scaler;

  if (raw.size() != protocol.model_parameter_count + protocol.q_scaler_count) {
    throw std::runtime_error("model parameter count mismatch");
  }

  HostModelParameters packed;
  packed.protocol = protocol;
  packed.has_charge = protocol.charge_mode > 0;
  packed.atomic_numbers = protocol.atomic_numbers;
  packed.spin_baseline = protocol.spin_baseline;
  packed.flexible_zbl_parameters = std::move(parsed.flexible_zbl_parameters);

  const std::size_t dim = checked_count(protocol.descriptor_dim, "descriptor_dim");
  const std::size_t hidden = checked_count(protocol.hidden_neurons, "hidden_neurons");
  const std::size_t hidden2 =
      checked_count(protocol.hidden_neurons2, "hidden_neurons2");
  const std::size_t num_types = checked_count(protocol.num_types, "num_types");
  const std::size_t w0_count = hidden * dim;

  packed.cutoff_radial_pair.resize(num_types * num_types);
  packed.cutoff_angular_pair.resize(num_types * num_types);
  if (protocol.spin_mode != 0) {
    packed.spin_cutoff_pair.resize(num_types * num_types);
  }
  if (protocol.has_zbl) {
    packed.zbl_parameters_pair.resize(num_types * num_types * 10);
  }
  for (std::size_t type1 = 0; type1 < num_types; ++type1) {
    for (std::size_t type2 = 0; type2 < num_types; ++type2) {
      const std::size_t pair = type1 * num_types + type2;
      packed.cutoff_radial_pair[pair] = static_cast<float>(
          0.5 * (protocol.cutoff_radial_by_type[type1] +
                 protocol.cutoff_radial_by_type[type2]));
      packed.cutoff_angular_pair[pair] = static_cast<float>(
          0.5 * (protocol.cutoff_angular_by_type[type1] +
                 protocol.cutoff_angular_by_type[type2]));
      if (protocol.spin_mode != 0) {
        packed.spin_cutoff_pair[pair] = static_cast<float>(
            0.5 * (protocol.spin_cutoff_by_type[type1] +
                   protocol.spin_cutoff_by_type[type2]));
      }
      if (!protocol.has_zbl) {
        continue;
      }
      float* zbl_pair = packed.zbl_parameters_pair.data() + pair * 10;
      if (protocol.flexible_zbl) {
        const std::size_t source_pair =
            symmetric_pair_index(type1, type2, num_types);
        std::copy_n(
            packed.flexible_zbl_parameters.data() + source_pair * 10,
            10,
            zbl_pair);
      } else {
        zbl_pair[0] = static_cast<float>(protocol.zbl_inner);
        zbl_pair[1] = static_cast<float>(protocol.zbl_outer);
        if (protocol.use_typewise_cutoff_zbl) {
          const int zi = protocol.atomic_numbers[type1];
          const int zj = protocol.atomic_numbers[type2];
          if (zi <= 0 || zi > static_cast<int>(kCovalentRadius.size()) ||
              zj <= 0 || zj > static_cast<int>(kCovalentRadius.size())) {
            throw std::runtime_error(
                "typewise ZBL requires atomic numbers within 1..94");
          }
          zbl_pair[0] = 0.0f;
          zbl_pair[1] = std::min(
              zbl_pair[1],
              (kCovalentRadius[static_cast<std::size_t>(zi - 1)] +
               kCovalentRadius[static_cast<std::size_t>(zj - 1)]) *
                  static_cast<float>(protocol.typewise_cutoff_zbl_factor));
        }
        std::copy(kUniversalZbl.begin(), kUniversalZbl.end(), zbl_pair + 2);
      }
    }
  }

  packed.ann_type_major.reserve(protocol.ann_parameter_count);
  std::size_t raw_offset = 0;
  for (std::size_t type = 0; type < num_types; ++type) {
    AnnTypeBlock block;
    block.w0_offset = packed.ann_type_major.size();
    append_range(packed.ann_type_major, raw, raw_offset, w0_count);
    raw_offset += w0_count;

    block.b0_offset = packed.ann_type_major.size();
    append_range(packed.ann_type_major, raw, raw_offset, hidden);
    raw_offset += hidden;

    block.w1_offset = packed.ann_type_major.size();
    if (hidden2 > 0) {
      block.has_second_hidden_layer = true;
      append_range(
          packed.ann_type_major, raw, raw_offset, hidden * hidden2);
      raw_offset += hidden * hidden2;
      block.b1_hidden_offset = packed.ann_type_major.size();
      append_range(packed.ann_type_major, raw, raw_offset, hidden2);
      raw_offset += hidden2;
      block.w2_offset = packed.ann_type_major.size();
      append_range(packed.ann_type_major, raw, raw_offset, hidden2);
      raw_offset += hidden2;
    } else {
      append_range(packed.ann_type_major, raw, raw_offset, hidden);
      raw_offset += hidden;
    }

    if (protocol.charge_mode > 0) {
      block.has_charge_w1 = true;
      block.charge_w1_offset = packed.ann_type_major.size();
      append_range(packed.ann_type_major, raw, raw_offset, hidden);
      raw_offset += hidden;
    }

    if (protocol.version == 5) {
      block.has_extra_bias = true;
      block.extra_bias_offset = packed.ann_type_major.size();
      append_range(packed.ann_type_major, raw, raw_offset, 1);
      raw_offset += 1;
    }

    packed.ann_blocks.push_back(block);
  }

  if (protocol.charge_mode > 0) {
    packed.sqrt_epsilon_inf_offset = packed.ann_type_major.size();
    append_range(packed.ann_type_major, raw, raw_offset, 1);
    raw_offset += 1;
  }

  packed.b1_offset = packed.ann_type_major.size();
  append_range(packed.ann_type_major, raw, raw_offset, 1);
  raw_offset += 1;

  const std::size_t descriptor_offset = raw_offset;
  packed.descriptor_layout.radial_count =
      num_types * num_types *
      (checked_count(protocol.n_max_radial, "n_max_radial") + 1) *
      (checked_count(protocol.basis_size_radial, "basis_size_radial") + 1);
  packed.descriptor_layout.angular_count =
      num_types * num_types *
      (checked_count(protocol.n_max_angular, "n_max_angular") + 1) *
      (checked_count(protocol.basis_size_angular, "basis_size_angular") + 1);
  packed.descriptor_layout.radial_offset = 0;
  packed.descriptor_layout.angular_offset = packed.descriptor_layout.radial_count;
  packed.descriptor_layout.spin_offset = protocol.ordinary_descriptor_parameter_count;
  packed.descriptor_layout.spin_count = protocol.spin_descriptor_parameter_count;
  append_range(
      packed.descriptor_coefficients,
      raw,
      descriptor_offset,
      protocol.descriptor_parameter_count);
  packed.descriptor_coefficients_type_pair_major.assign(
      protocol.ordinary_descriptor_parameter_count,
      0.0f);
  packed.angular_coefficients_center_type_major.assign(
      packed.descriptor_layout.angular_count,
      0.0f);
  const std::size_t type_pairs = num_types * num_types;
  const std::size_t radial_basis_count =
      (checked_count(protocol.n_max_radial, "n_max_radial") + 1) *
      (checked_count(protocol.basis_size_radial, "basis_size_radial") + 1);
  const std::size_t angular_basis_count =
      (checked_count(protocol.n_max_angular, "n_max_angular") + 1) *
      (checked_count(protocol.basis_size_angular, "basis_size_angular") + 1);
  for (std::size_t type_pair = 0; type_pair < type_pairs; ++type_pair) {
    for (std::size_t basis = 0; basis < radial_basis_count; ++basis) {
      packed.descriptor_coefficients_type_pair_major[
          type_pair * radial_basis_count + basis] =
          packed.descriptor_coefficients[basis * type_pairs + type_pair];
    }
    for (std::size_t basis = 0; basis < angular_basis_count; ++basis) {
      packed.descriptor_coefficients_type_pair_major[
          packed.descriptor_layout.radial_count +
          type_pair * angular_basis_count + basis] =
          packed.descriptor_coefficients[
              packed.descriptor_layout.angular_offset +
              basis * type_pairs + type_pair];

      const std::size_t center_type = type_pair / num_types;
      const std::size_t neighbor_type = type_pair % num_types;
      packed.angular_coefficients_center_type_major[
          (center_type * angular_basis_count + basis) * num_types +
          neighbor_type] =
          packed.descriptor_coefficients[
              packed.descriptor_layout.angular_offset +
              basis * type_pairs + type_pair];
    }
  }
  raw_offset += protocol.descriptor_parameter_count;
  append_range(
      packed.spin_projection_parameters,
      raw,
      raw_offset,
      protocol.spin_projection_parameter_count);
  raw_offset += protocol.spin_projection_parameter_count;

  if (raw_offset != protocol.model_parameter_count) {
    throw std::runtime_error("model parameter layout did not consume all model parameters");
  }

  append_range(packed.q_scaler, raw, raw_offset, protocol.q_scaler_count);
  packed.ann_type_major_qscaled = packed.ann_type_major;
  for (std::size_t type = 0; type < num_types; ++type) {
    const AnnTypeBlock& block = packed.ann_blocks[type];
    for (std::size_t neuron = 0; neuron < hidden; ++neuron) {
      for (std::size_t d = 0; d < dim; ++d) {
        packed.ann_type_major_qscaled[
            block.w0_offset + neuron * dim + d] *= packed.q_scaler[d];
      }
    }
  }
  return packed;
}

}  // namespace nep_adapters::cuda_backend
