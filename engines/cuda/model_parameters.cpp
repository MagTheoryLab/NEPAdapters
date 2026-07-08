#include "model_parameters.hpp"

#include <stdexcept>

namespace nep_adapters::cuda_backend {
namespace {

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
  packed.flexible_zbl_parameters = std::move(parsed.flexible_zbl_parameters);

  const std::size_t dim = checked_count(protocol.descriptor_dim, "descriptor_dim");
  const std::size_t hidden = checked_count(protocol.hidden_neurons, "hidden_neurons");
  const std::size_t num_types = checked_count(protocol.num_types, "num_types");
  const std::size_t w0_count = hidden * dim;

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
    append_range(packed.ann_type_major, raw, raw_offset, hidden);
    raw_offset += hidden;

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
  append_range(
      packed.descriptor_coefficients,
      raw,
      descriptor_offset,
      protocol.descriptor_parameter_count);
  packed.descriptor_coefficients_type_pair_major.assign(
      protocol.descriptor_parameter_count,
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
    }
  }
  raw_offset += protocol.descriptor_parameter_count;

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
