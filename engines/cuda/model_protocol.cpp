#include "model_protocol.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace nep_adapters::cuda_backend {
namespace {

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
  if (consumed != token.size()) {
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

std::vector<std::string> next_tokens(std::ifstream& input) {
  std::string line;
  while (std::getline(input, line)) {
    std::vector<std::string> tokens = split_line(line);
    if (!tokens.empty()) {
      return tokens;
    }
  }
  return {};
}

bool flag_from_token(const std::string& token) {
  return parse_int(token) != 0;
}

void parse_version_tag(const std::string& tag, ModelProtocol& protocol) {
  if (tag.find("_dipole") != std::string::npos ||
      tag.find("_polarizability") != std::string::npos ||
      tag.find("_temperature") != std::string::npos) {
    throw UnsupportedModelProtocol(
        "CUDA engine only accepts supported NEP potential models");
  }

  if (tag == "nep4" || tag == "nep4_zbl" ||
      tag == "nep4_spin" || tag == "nep4_spin1") {
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
  protocol.spin_mode = tag.find("_spin") != std::string::npos ? 1 : 0;
}

void parse_zbl(
    const std::vector<std::string>& tokens,
    ModelProtocol& protocol) {
  if ((tokens.size() != 3 && tokens.size() != 4) || tokens[0] != "zbl") {
    throw std::runtime_error("expected zbl line");
  }

  protocol.zbl_inner = parse_double(tokens[1]);
  protocol.zbl_outer = parse_double(tokens[2]);
  protocol.flexible_zbl = (protocol.zbl_inner == 0.0 && protocol.zbl_outer == 0.0);
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
  const bool per_type_cutoff = static_cast<int>(tokens.size()) == per_type_tokens;
  if (!uniform_cutoff && !per_type_cutoff) {
    throw std::runtime_error("invalid cutoff line");
  }

  protocol.cutoff_radial = 0.0;
  protocol.cutoff_angular = 0.0;
  if (uniform_cutoff) {
    protocol.cutoff_radial = parse_double(tokens[1]);
    protocol.cutoff_angular = parse_double(tokens[2]);
    protocol.max_neighbors_radial = parse_int(tokens[3]);
    protocol.max_neighbors_angular = parse_int(tokens[4]);
  } else {
    for (int type = 0; type < protocol.num_types; ++type) {
      protocol.cutoff_radial =
          std::max(protocol.cutoff_radial, parse_double(tokens[1 + 2 * type]));
      protocol.cutoff_angular =
          std::max(protocol.cutoff_angular, parse_double(tokens[2 + 2 * type]));
    }
    protocol.max_neighbors_radial = parse_int(tokens[1 + 2 * protocol.num_types]);
    protocol.max_neighbors_angular = parse_int(tokens[2 + 2 * protocol.num_types]);
  }

  protocol.neighbor_capacity_radial =
      static_cast<int>(std::ceil(protocol.max_neighbors_radial * 1.25));
  protocol.neighbor_capacity_angular =
      static_cast<int>(std::ceil(protocol.max_neighbors_angular * 1.25));
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
  if (tokens.size() < 4 || tokens[0] != "l_max") {
    throw std::runtime_error("expected l_max line");
  }

  BodyChannelConfig body;
  body.l_max_3body = parse_int(tokens[1]);
  body.has_q_222 = flag_from_token(tokens[2]);
  body.has_q_1111 = flag_from_token(tokens[3]);
  if (tokens.size() >= 5) {
    body.has_q_112 = flag_from_token(tokens[4]);
  }
  if (tokens.size() >= 6) {
    body.has_q_123 = flag_from_token(tokens[5]);
  }
  if (tokens.size() >= 7) {
    body.has_q_233 = flag_from_token(tokens[6]);
  }
  if (tokens.size() >= 8) {
    body.has_q_134 = flag_from_token(tokens[7]);
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
  if (protocol.hidden_neurons <= 0 || parse_int(tokens[2]) != 0) {
    throw std::runtime_error("invalid ANN line");
  }
}

void parse_spin_header_line(
    const std::vector<std::string>& tokens,
    ModelProtocol& protocol) {
  if (tokens.empty()) {
    return;
  }
  if (tokens[0] == "spin_baseline") {
    if (tokens.size() != static_cast<std::size_t>(1 + protocol.num_types)) {
      throw std::runtime_error("spin_baseline must have one value per type");
    }
    protocol.spin_baseline.resize(static_cast<std::size_t>(protocol.num_types));
    for (int type = 0; type < protocol.num_types; ++type) {
      protocol.spin_baseline[static_cast<std::size_t>(type)] =
          parse_double(tokens[static_cast<std::size_t>(1 + type)]);
    }
  } else if (tokens[0] == "spin_chiral") {
    protocol.spin_chiral = parse_int(tokens[1]);
    if (protocol.spin_chiral != 0 && protocol.spin_chiral != 1) {
      throw std::runtime_error("spin_chiral must be 0 or 1");
    }
  } else if (tokens[0] == "spin_compress") {
    protocol.spin_compress = parse_int(tokens[1]);
  } else if (tokens[0] == "spin_basis_size") {
    protocol.spin_basis_size = parse_int(tokens[1]);
  } else if (tokens[0] == "spin_l_max") {
    protocol.spin_l_max = parse_int(tokens[1]);
  } else if (tokens[0] == "spin_cutoff") {
    protocol.spin_cutoff_radial = parse_double(tokens[1]);
  } else if (tokens[0] == "spin_dof_type" || tokens[0] == "spin_type") {
    protocol.spin_dof_type_active.assign(
        static_cast<std::size_t>(protocol.num_types), 0);
    for (std::size_t i = 1; i < tokens.size(); ++i) {
      const auto found =
          std::find(protocol.elements.begin(), protocol.elements.end(), tokens[i]);
      if (found == protocol.elements.end()) {
        throw std::runtime_error("unknown spin_dof_type in nep.txt");
      }
      protocol.spin_dof_type_active[
          static_cast<std::size_t>(found - protocol.elements.begin())] = 1;
    }
  } else if (tokens[0] == "spin_env_type") {
    protocol.spin_env_type_active.assign(
        static_cast<std::size_t>(protocol.num_types), 0);
    for (std::size_t i = 1; i < tokens.size(); ++i) {
      const auto found =
          std::find(protocol.elements.begin(), protocol.elements.end(), tokens[i]);
      if (found == protocol.elements.end()) {
        throw std::runtime_error("unknown spin_env_type in nep.txt");
      }
      protocol.spin_env_type_active[
          static_cast<std::size_t>(found - protocol.elements.begin())] = 1;
    }
  } else if (tokens[0] == "spin_scaler" || tokens[0] == "spin_n_max") {
    return;
  } else {
    throw std::runtime_error("unknown spin header line");
  }
}

std::vector<std::string> parse_spin_block(
    std::ifstream& input,
    ModelProtocol& protocol,
    std::vector<std::string> tokens) {
  if (!protocol.spin_mode) {
    return tokens;
  }
  if (tokens.empty() || tokens[0] != "spin_mode") {
    throw std::runtime_error("spin model must contain spin_mode line");
  }
  protocol.spin_mode = parse_int(tokens[1]);
  if (protocol.spin_mode != 1) {
    throw std::runtime_error("only spin_mode 1 is supported");
  }
  if (tokens.size() >= 3) {
    const int spin_header_lines = parse_int(tokens[2]);
    for (int line = 0; line < spin_header_lines; ++line) {
      parse_spin_header_line(next_tokens(input), protocol);
    }
    return next_tokens(input);
  }
  tokens = next_tokens(input);
  while (!tokens.empty() && tokens[0].rfind("spin_", 0) == 0) {
    parse_spin_header_line(tokens, protocol);
    tokens = next_tokens(input);
  }
  return tokens;
}

void finalize_counts(ModelProtocol& protocol) {
  protocol.struct_descriptor_dim =
      protocol.n_max_radial + 1 +
      (protocol.n_max_angular + 1) * protocol.body_channels.channel_count();
  if (protocol.spin_mode) {
    if (protocol.spin_compress <= 0 || protocol.spin_l_max < 0 ||
        protocol.spin_l_max > 4) {
      throw std::runtime_error("invalid spin settings");
    }
    if (protocol.spin_basis_size + 1 < protocol.spin_compress) {
      throw std::runtime_error("spin_basis_size must cover spin_compress");
    }
    if (protocol.spin_cutoff_radial <= 0.0) {
      protocol.spin_cutoff_radial = protocol.cutoff_radial;
    }
    protocol.cutoff_radial = std::max(protocol.cutoff_radial, protocol.spin_cutoff_radial);
    protocol.cutoff_max = std::max(protocol.cutoff_max, protocol.spin_cutoff_radial);
    if (protocol.spin_dof_type_active.empty()) {
      protocol.spin_dof_type_active.assign(
          static_cast<std::size_t>(protocol.num_types), 1);
    }
    if (protocol.spin_env_type_active.empty()) {
      protocol.spin_env_type_active = protocol.spin_dof_type_active;
    }
    if (protocol.spin_baseline.empty()) {
      protocol.spin_baseline.assign(static_cast<std::size_t>(protocol.num_types), 0.0);
    }
    protocol.spin_descriptor_dim = make_spin_core_layout(protocol).descriptor_dim;
  }
  protocol.descriptor_dim =
      protocol.struct_descriptor_dim + protocol.spin_descriptor_dim;

  const std::size_t dim = static_cast<std::size_t>(protocol.descriptor_dim);
  const std::size_t hidden = static_cast<std::size_t>(protocol.hidden_neurons);
  const std::size_t types = static_cast<std::size_t>(protocol.num_types);
  if (protocol.version == 4) {
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
  if (protocol.spin_mode) {
    protocol.spin_descriptor_parameter_count =
        type_pairs * static_cast<std::size_t>(protocol.spin_compress) *
        (static_cast<std::size_t>(protocol.spin_basis_size) + 1);
  }
  protocol.descriptor_parameter_count =
      protocol.ordinary_descriptor_parameter_count +
      protocol.spin_descriptor_parameter_count;
  protocol.model_parameter_count =
      protocol.ann_parameter_count + protocol.descriptor_parameter_count;
  protocol.q_scaler_count = dim;
}

std::vector<float> read_scalar_lines(
    std::ifstream& input,
    std::size_t count,
    const char* section_name) {
  std::vector<float> values;
  values.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const std::vector<std::string> tokens = next_tokens(input);
    if (tokens.empty()) {
      throw std::runtime_error(std::string("unexpected end of ") + section_name);
    }
    values.push_back(parse_float_token(tokens[0]));
  }
  return values;
}

ModelProtocol parse_model_header(std::ifstream& input) {
  ModelProtocol protocol;
  std::vector<std::string> tokens = next_tokens(input);
  if (tokens.size() < 3) {
    throw std::runtime_error("malformed model header");
  }

  parse_version_tag(tokens[0], protocol);
  protocol.num_types = parse_int(tokens[1]);
  if (protocol.num_types <= 0 ||
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
  std::ifstream input(model_path);
  if (!input.is_open()) {
    throw std::runtime_error("failed to open model file");
  }
  return parse_model_header(input);
}

ParsedModelFile parse_model_file(const std::string& model_path) {
  std::ifstream input(model_path);
  if (!input.is_open()) {
    throw std::runtime_error("failed to open model file");
  }
  ParsedModelFile parsed;
  parsed.protocol = parse_model_header(input);
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
  }

  return parsed;
}

}  // namespace nep_adapters::cuda_backend
