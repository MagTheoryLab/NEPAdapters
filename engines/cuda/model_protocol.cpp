#include "model_protocol.hpp"
#include "nep_adapters/detail/model_file.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace nep_adapters::cuda_backend {
namespace {

constexpr int kMaxModelTypes = 118;

std::vector<std::string> split_line(const std::string& line) {
  std::istringstream stream(line);
  std::vector<std::string> tokens;
  std::string token;
  while (stream >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

int parse_int(const std::string& token) {
  std::size_t consumed = 0;
  const int value = std::stoi(token, &consumed);
  if (consumed != token.size()) {
    throw std::runtime_error("invalid integer token");
  }
  return value;
}

double parse_double(const std::string& token) {
  std::size_t consumed = 0;
  const double value = std::stod(token, &consumed);
  if (consumed != token.size() || !std::isfinite(value)) {
    throw std::runtime_error("invalid double token");
  }
  return value;
}

float parse_float_token(const std::string& token) {
  const double value = parse_double(token);
  if (value < -std::numeric_limits<float>::max() ||
      value > std::numeric_limits<float>::max()) {
    throw std::runtime_error("float token out of range");
  }
  return static_cast<float>(value);
}

int atomic_number_from_symbol(const std::string& symbol) {
  static constexpr std::array<const char*, 118> kSymbols = {
      "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
      "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
      "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
      "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
      "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
      "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
      "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
      "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
      "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
      "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
      "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
      "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};
  for (std::size_t index = 0; index < kSymbols.size(); ++index) {
    if (symbol == kSymbols[index]) {
      return static_cast<int>(index) + 1;
    }
  }
  throw std::runtime_error("unknown element symbol for ZBL atomic number");
}

std::vector<std::string> next_tokens(std::istream& input) {
  std::string line;
  while (std::getline(input, line)) {
    std::vector<std::string> tokens = split_line(line);
    if (!tokens.empty()) {
      return tokens;
    }
  }
  return {};
}

bool boolean_flag_from_token(const std::string& token, const char* name) {
  const int value = parse_int(token);
  if (value != 0 && value != 1) {
    throw std::runtime_error(std::string(name) + " must be 0 or 1");
  }
  return value != 0;
}

void parse_version_tag(const std::string& tag, ModelProtocol& protocol) {
  if (tag.find("_dipole") != std::string::npos ||
      tag.find("_polarizability") != std::string::npos) {
    throw UnsupportedModelProtocol(
        "dipole and polarizability models are unsupported by the CUDA backend; "
        "CPU fallback is disabled");
  }
  if (tag.find("_temperature") != std::string::npos) {
    throw UnsupportedModelProtocol(
        "CUDA engine only accepts supported NEP potential models");
  }

  if (tag == "nep4" || tag == "nep4_zbl" ||
      tag == "nep4_spin_density" || tag == "nep4_spin_density_zbl") {
    protocol.version = 4;
  } else if (tag == "nep4_charge1" || tag == "nep4_zbl_charge1") {
    protocol.version = 4;
    protocol.charge_mode = 1;
  } else if (tag == "nep4_charge2" || tag == "nep4_zbl_charge2") {
    protocol.version = 4;
    protocol.charge_mode = 2;
  } else if (tag == "nep4_charge3" || tag == "nep4_zbl_charge3") {
    protocol.version = 4;
    protocol.charge_mode = 3;
  } else if (tag == "nep5" || tag == "nep5_zbl") {
    protocol.version = 5;
  } else {
    throw UnsupportedModelProtocol(
        "CUDA engine only accepts supported NEP4/NEP5 models");
  }
  protocol.has_zbl = tag.find("_zbl") != std::string::npos;
  protocol.spin_mode = tag.find("_spin_density") != std::string::npos ? 1 : 0;
}

void parse_zbl(
    const std::vector<std::string>& tokens,
    ModelProtocol& protocol) {
  if ((tokens.size() != 3 && tokens.size() != 4) || tokens[0] != "zbl") {
    throw std::runtime_error("expected zbl line");
  }
  if (tokens.size() == 4) {
    protocol.typewise_cutoff_zbl_factor = parse_double(tokens[3]);
    if (protocol.typewise_cutoff_zbl_factor <= 0.0) {
      throw std::runtime_error("typewise ZBL cutoff factor must be positive");
    }
    protocol.use_typewise_cutoff_zbl = true;
  }

  protocol.zbl_inner = parse_double(tokens[1]);
  protocol.zbl_outer = parse_double(tokens[2]);
  protocol.flexible_zbl = (protocol.zbl_inner == 0.0 && protocol.zbl_outer == 0.0);
  if (protocol.flexible_zbl && protocol.use_typewise_cutoff_zbl) {
    throw std::runtime_error(
        "flexible ZBL does not accept a typewise cutoff factor");
  }
  if (!protocol.flexible_zbl &&
      (protocol.zbl_inner < 0.0 ||
       protocol.zbl_outer <= protocol.zbl_inner)) {
    throw std::runtime_error("invalid ZBL cutoff range");
  }
}

// Mirrors the ordinary NEP4/NEP5 protocol in torchnep/src/force/nep.cu.
// Keep protocol parsing separate from CUDA execution so later kernel work does
// not quietly redefine the model ABI.
void parse_cutoff(
    const std::vector<std::string>& tokens,
    ModelProtocol& protocol) {
  if (tokens.empty() || tokens[0] != "cutoff") {
    throw std::runtime_error("expected cutoff line");
  }

  const int per_type_tokens = 2 * protocol.num_types + 3;
  const bool uniform_cutoff = tokens.size() == 5;
  const bool per_type_cutoff =
      protocol.num_types > 1 &&
      static_cast<int>(tokens.size()) == per_type_tokens;
  if (!uniform_cutoff && !per_type_cutoff) {
    throw std::runtime_error("invalid cutoff line");
  }
  protocol.cutoff_radial_by_type.resize(
      static_cast<std::size_t>(protocol.num_types));
  protocol.cutoff_angular_by_type.resize(
      static_cast<std::size_t>(protocol.num_types));
  if (uniform_cutoff) {
    const double radial = parse_double(tokens[1]);
    const double angular = parse_double(tokens[2]);
    std::fill(
        protocol.cutoff_radial_by_type.begin(),
        protocol.cutoff_radial_by_type.end(),
        radial);
    std::fill(
        protocol.cutoff_angular_by_type.begin(),
        protocol.cutoff_angular_by_type.end(),
        angular);
  } else {
    protocol.use_typewise_cutoff = true;
    for (int type = 0; type < protocol.num_types; ++type) {
      protocol.cutoff_radial_by_type[static_cast<std::size_t>(type)] =
          parse_double(tokens[static_cast<std::size_t>(1 + 2 * type)]);
      protocol.cutoff_angular_by_type[static_cast<std::size_t>(type)] =
          parse_double(tokens[static_cast<std::size_t>(2 + 2 * type)]);
    }
  }
  protocol.cutoff_radial = *std::max_element(
      protocol.cutoff_radial_by_type.begin(),
      protocol.cutoff_radial_by_type.end());
  protocol.cutoff_neighbor =
      std::max(protocol.cutoff_radial, protocol.zbl_outer);
  protocol.cutoff_angular = *std::max_element(
      protocol.cutoff_angular_by_type.begin(),
      protocol.cutoff_angular_by_type.end());
  protocol.max_neighbors_radial =
      parse_int(tokens[tokens.size() - 2]);
  protocol.max_neighbors_angular =
      parse_int(tokens[tokens.size() - 1]);
  for (int type = 0; type < protocol.num_types; ++type) {
    const double radial =
        protocol.cutoff_radial_by_type[static_cast<std::size_t>(type)];
    const double angular =
        protocol.cutoff_angular_by_type[static_cast<std::size_t>(type)];
    if (radial <= 0.0 || radial > 100.0 ||
        angular <= 0.0 || angular > 100.0) {
      throw std::runtime_error("cutoffs must be within (0, 100]");
    }
  }
  if (protocol.max_neighbors_radial <= 0 ||
      protocol.max_neighbors_angular <= 0) {
    throw std::runtime_error("maximum neighbor counts must be positive");
  }

  const double radial_capacity =
      std::ceil(static_cast<double>(protocol.max_neighbors_radial) * 1.25);
  const double angular_capacity =
      std::ceil(static_cast<double>(protocol.max_neighbors_angular) * 1.25);
  if (radial_capacity > std::numeric_limits<int>::max() ||
      angular_capacity > std::numeric_limits<int>::max()) {
    throw std::runtime_error("maximum neighbor count is too large");
  }
  protocol.neighbor_capacity_radial = static_cast<int>(radial_capacity);
  protocol.neighbor_capacity_angular = static_cast<int>(angular_capacity);
  protocol.cutoff_max =
      std::max({protocol.cutoff_radial, protocol.cutoff_angular, protocol.zbl_outer});
}

void parse_n_max(
    const std::vector<std::string>& tokens,
    ModelProtocol& protocol) {
  if (tokens.size() != 3 || tokens[0] != "n_max") {
    throw std::runtime_error("expected n_max line");
  }
  protocol.n_max_radial = parse_int(tokens[1]);
  protocol.n_max_angular = parse_int(tokens[2]);
}

void parse_basis_size(
    const std::vector<std::string>& tokens,
    ModelProtocol& protocol) {
  if (tokens.size() != 3 || tokens[0] != "basis_size") {
    throw std::runtime_error("expected basis_size line");
  }
  protocol.basis_size_radial = parse_int(tokens[1]);
  protocol.basis_size_angular = parse_int(tokens[2]);
}

void parse_l_max(
    const std::vector<std::string>& tokens,
    ModelProtocol& protocol) {
  if (tokens.size() < 2 || tokens.size() > 8 || tokens[0] != "l_max") {
    throw std::runtime_error("expected l_max line");
  }

  BodyChannelConfig body;
  body.l_max_3body = parse_int(tokens[1]);
  if (tokens.size() >= 3) {
    // GPUMD nep.txt historically encoded q222 as 0/2 and still emits 2
    // when newer q112/q123/q233/q134 fields are present. Treat only this
    // position as a legacy 0/1/2 field; all other switches are strict bools.
    const int q222 = parse_int(tokens[2]);
    if (q222 < 0 || q222 > 2) {
      throw std::runtime_error("has_q_222 must be 0, 1, or legacy value 2");
    }
    body.has_q_222 = q222 != 0;
  }
  if (tokens.size() >= 4) {
    body.has_q_1111 =
        boolean_flag_from_token(tokens[3], "has_q_1111");
  }
  if (tokens.size() >= 5) {
    body.has_q_112 = boolean_flag_from_token(tokens[4], "has_q_112");
  }
  if (tokens.size() >= 6) {
    body.has_q_123 = boolean_flag_from_token(tokens[5], "has_q_123");
  }
  if (tokens.size() >= 7) {
    body.has_q_233 = boolean_flag_from_token(tokens[6], "has_q_233");
  }
  if (tokens.size() >= 8) {
    body.has_q_134 = boolean_flag_from_token(tokens[7], "has_q_134");
  }
  protocol.body_channels = body;
}

void parse_ann(
    const std::vector<std::string>& tokens,
    ModelProtocol& protocol) {
  if (tokens.size() != 3 || tokens[0] != "ANN") {
    throw std::runtime_error("expected ANN line");
  }
  protocol.hidden_neurons = parse_int(tokens[1]);
  if (protocol.hidden_neurons <= 0 || protocol.hidden_neurons > 120) {
    throw std::runtime_error("first ANN hidden layer must be within 1..120");
  }
  protocol.hidden_neurons2 = parse_int(tokens[2]);
  if (protocol.hidden_neurons2 < 0 || protocol.hidden_neurons2 > 120) {
    throw std::runtime_error("second ANN hidden layer must be within 0..120");
  }
  if (protocol.hidden_neurons2 > 0 &&
      (protocol.version != 4 || protocol.charge_mode != 0 ||
       protocol.spin_mode != 0)) {
    throw UnsupportedModelProtocol(
        "two-hidden-layer ANN is supported only for ordinary NEP4 models");
  }
}

void validate_model_ranges(const ModelProtocol& protocol) {
  if (protocol.n_max_radial < 0 || protocol.n_max_radial > 12) {
    throw std::runtime_error("n_max_radial must be within 0..12");
  }
  if (protocol.n_max_angular < 0 || protocol.n_max_angular > 8) {
    throw std::runtime_error("n_max_angular must be within 0..8");
  }
  if (protocol.basis_size_radial < 0 || protocol.basis_size_radial > 16) {
    throw std::runtime_error("basis_size_radial must be within 0..16");
  }
  if (protocol.basis_size_angular < 0 || protocol.basis_size_angular > 12) {
    throw std::runtime_error("basis_size_angular must be within 0..12");
  }

  const BodyChannelConfig& body = protocol.body_channels;
  if (body.l_max_3body < 0 || body.l_max_3body > 8) {
    throw std::runtime_error("l_max_3body must be within 0..8");
  }
  if ((body.has_q_222 || body.has_q_112) && body.l_max_3body < 2) {
    throw std::runtime_error("q222/q112 require l_max_3body >= 2");
  }
  if (body.has_q_1111 && body.l_max_3body < 1) {
    throw std::runtime_error("q1111 requires l_max_3body >= 1");
  }
  if ((body.has_q_123 || body.has_q_233) && body.l_max_3body < 3) {
    throw std::runtime_error("q123/q233 require l_max_3body >= 3");
  }
  if (body.has_q_134 && body.l_max_3body < 4) {
    throw std::runtime_error("q134 requires l_max_3body >= 4");
  }
  const int angular_dim =
      (protocol.n_max_angular + 1) * body.channel_count();
  if (angular_dim > 90) {
    throw std::runtime_error(
        "number of structural angular descriptors must not exceed 90");
  }
}

std::vector<std::string> parse_spin_block(
    std::istream& input, ModelProtocol& p, std::vector<std::string> tokens) {
  if (!p.spin_mode) return tokens;
  if (tokens.size() != 3 || tokens[0] != "spin_mode" || tokens[1] != "1")
    throw std::runtime_error("expected density spin metadata");
  const int count = parse_int(tokens[2]);
  p.spin_dof_type_active.assign(p.num_types, 1);
  for (int line = 0; line < count; ++line) {
    const auto v = next_tokens(input);
    if (v.size() < 2) throw std::runtime_error("truncated spin metadata");
    const auto& key = v[0];
    if (key == "spin_baseline") {
      for (int t = 0; t < p.num_types; ++t) p.spin_baseline.push_back(parse_double(v[t+1]));
    } else if (key == "spin_basis_size") p.spin_basis_size = parse_int(v[1]);
    else if (key == "spin_compress") p.spin_compress = parse_int(v[1]);
    else if (key == "spin_l_max") p.spin_l_max = parse_int(v[1]);
    else if (key == "spin_order") p.spin_order = parse_int(v[1]);
    else if (key == "spin_soc") p.spin_soc = parse_int(v[1]);
    else if (key == "spin_scaler") p.spin_scaler = parse_int(v[1]);
    else if (key == "spin_cutoff") {
      for (int t = 0; t < p.num_types; ++t)
        p.spin_cutoff_by_type.push_back(parse_double(v[v.size() == 2 ? 1 : t+1]));
      p.spin_cutoff_radial = *std::max_element(p.spin_cutoff_by_type.begin(), p.spin_cutoff_by_type.end());
    } else if (key == "spin_type") {
      p.spin_dof_type_active.assign(p.num_types, 0);
      for (std::size_t t = 1; t < v.size(); ++t) {
        auto found = std::find(p.elements.begin(), p.elements.end(), v[t]);
        p.spin_dof_type_active[found - p.elements.begin()] = 1;
      }
    } else throw std::runtime_error("unexpected spin metadata: " + key);
  }
  return next_tokens(input);
}

void finalize_counts(ModelProtocol& protocol) {
  protocol.struct_descriptor_dim =
      protocol.n_max_radial + 1 +
      (protocol.n_max_angular + 1) * protocol.body_channels.channel_count();
  if (protocol.spin_mode) {
    if (!supports_cuda_spin_shape(protocol)) throw UnsupportedModelProtocol("unsupported density spin shape");
    const auto layout = make_spin_density_layout(protocol);
    protocol.spin_descriptor_dim = layout.dimension;
    protocol.spin_parameter_count = layout.parameter_count(protocol.n_max_angular + 1);
  }
  protocol.descriptor_dim =
      protocol.struct_descriptor_dim + protocol.spin_descriptor_dim;

  const std::size_t dim = static_cast<std::size_t>(protocol.descriptor_dim);
  const std::size_t hidden = static_cast<std::size_t>(protocol.hidden_neurons);
  const std::size_t hidden2 =
      static_cast<std::size_t>(protocol.hidden_neurons2);
  const std::size_t types = static_cast<std::size_t>(protocol.num_types);
  if (hidden2 > 0) {
    protocol.ann_parameter_count =
        ((dim + 1) * hidden + (hidden + 2) * hidden2) * types + 1;
  } else if (protocol.version == 4) {
    protocol.ann_parameter_count = (dim + 2) * hidden * types + 1;
  } else {
    protocol.ann_parameter_count = ((dim + 2) * hidden + 1) * types + 1;
  }
  if (protocol.charge_mode > 0) {
    protocol.ann_parameter_count += hidden * types + 1;
  }

  const std::size_t type_pairs = types * types;
  protocol.ordinary_descriptor_parameter_count =
      type_pairs *
      ((static_cast<std::size_t>(protocol.n_max_radial) + 1) *
           (static_cast<std::size_t>(protocol.basis_size_radial) + 1) +
       (static_cast<std::size_t>(protocol.n_max_angular) + 1) *
           (static_cast<std::size_t>(protocol.basis_size_angular) + 1));
  protocol.descriptor_parameter_count =
      protocol.ordinary_descriptor_parameter_count;
  protocol.model_parameter_count =
      protocol.ann_parameter_count + protocol.descriptor_parameter_count +
      protocol.spin_parameter_count;
  protocol.q_scaler_count = dim;
}

std::vector<float> read_scalar_lines(
    std::istream& input,
    std::size_t count,
    const char* section_name) {
  std::vector<float> values;
  values.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const std::vector<std::string> tokens = next_tokens(input);
    if (tokens.empty()) {
      throw std::runtime_error(std::string("unexpected end of ") + section_name);
    }
    if (tokens.size() != 1) {
      throw std::runtime_error(
          std::string("expected one scalar per line in ") + section_name);
    }
    values.push_back(parse_float_token(tokens[0]));
  }
  return values;
}

ModelProtocol parse_model_header(std::istream& input) {
  ModelProtocol protocol;
  std::vector<std::string> tokens = next_tokens(input);
  if (tokens.size() < 3) {
    throw std::runtime_error("malformed model header");
  }

  parse_version_tag(tokens[0], protocol);
  protocol.num_types = parse_int(tokens[1]);
  if (protocol.num_types <= 0 || protocol.num_types > kMaxModelTypes ||
      static_cast<int>(tokens.size()) != 2 + protocol.num_types) {
    throw std::runtime_error("invalid type count in model header");
  }
  protocol.elements.assign(tokens.begin() + 2, tokens.end());
  protocol.atomic_numbers.reserve(protocol.elements.size());
  for (const std::string& element : protocol.elements) {
    protocol.atomic_numbers.push_back(atomic_number_from_symbol(element));
  }

  tokens = next_tokens(input);
  tokens = parse_spin_block(input, protocol, tokens);
  if (protocol.has_zbl) {
    parse_zbl(tokens, protocol);
    tokens = next_tokens(input);
  }

  parse_cutoff(tokens, protocol);
  parse_n_max(next_tokens(input), protocol);
  parse_basis_size(next_tokens(input), protocol);
  parse_l_max(next_tokens(input), protocol);
  parse_ann(next_tokens(input), protocol);
  validate_model_ranges(protocol);
  finalize_counts(protocol);
  return protocol;
}

}  // namespace

int BodyChannelConfig::channel_count() const {
  int count = l_max_3body;
  count += has_q_222 ? 1 : 0;
  count += has_q_1111 ? 1 : 0;
  count += has_q_112 ? 1 : 0;
  count += has_q_123 ? 1 : 0;
  count += has_q_233 ? 1 : 0;
  count += has_q_134 ? 1 : 0;
  return count;
}

int BodyChannelConfig::abc_count() const {
  return (l_max_3body + 1) * (l_max_3body + 1) - 1;
}

ModelProtocol parse_model_protocol(const std::string& model_path) {
  nep_adapters::detail::ModelInput model_input(model_path);
  std::istream& input = model_input.stream();
  ModelProtocol protocol = parse_model_header(input);
  protocol.energy_reference = model_input.energy_reference();
  protocol.spin_type_semantics = model_input.spin_type_semantics();
  return protocol;
}

ParsedModelFile parse_model_file(const std::string& model_path) {
  nep_adapters::detail::ModelInput model_input(model_path);
  std::istream& input = model_input.stream();
  ParsedModelFile parsed;
  parsed.protocol = parse_model_header(input);
  parsed.protocol.energy_reference = model_input.energy_reference();
  parsed.protocol.spin_type_semantics = model_input.spin_type_semantics();
  parsed.parameters_and_q_scaler = read_scalar_lines(
      input,
      parsed.protocol.model_parameter_count + parsed.protocol.q_scaler_count,
      "model parameters");
  if (parsed.protocol.flexible_zbl) {
    const std::size_t type_pairs =
        static_cast<std::size_t>(parsed.protocol.num_types) *
        static_cast<std::size_t>(parsed.protocol.num_types + 1) / 2;
      parsed.flexible_zbl_parameters =
        read_scalar_lines(input, 10 * type_pairs, "flexible zbl parameters");
    double flexible_outer_max = 0.0;
    for (std::size_t pair = 0; pair < type_pairs; ++pair) {
      const double inner =
          parsed.flexible_zbl_parameters[10 * pair];
      const double outer =
          parsed.flexible_zbl_parameters[10 * pair + 1];
      if (inner < 0.0 || outer <= inner) {
        throw std::runtime_error("invalid flexible ZBL cutoff range");
      }
      flexible_outer_max = std::max(flexible_outer_max, outer);
    }
    parsed.protocol.zbl_outer = flexible_outer_max;
    parsed.protocol.cutoff_neighbor =
        std::max(parsed.protocol.cutoff_neighbor, flexible_outer_max);
    parsed.protocol.cutoff_max =
        std::max(parsed.protocol.cutoff_max, flexible_outer_max);
  }
  const std::vector<std::string> trailing = next_tokens(input);
  if (!trailing.empty()) {
    throw std::runtime_error("unexpected trailing data after model parameters");
  }

  return parsed;
}

}  // namespace nep_adapters::cuda_backend
