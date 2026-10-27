#include "nep_adapters/detail/model_file.hpp"
#include "../engines/common/spin_polynomial_layout.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <locale>
#include <map>
#include <set>
#include <stdexcept>

namespace nep_adapters::detail {
namespace {
using Tokens = std::vector<std::string>;

std::string lower(std::string s) {
  for (char& c : s) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
  return s;
}

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error("NEP block model: " + message);
}

Tokens tokenize(std::string line) {
  if (!line.empty() && line.back() == '\r') line.pop_back();
  line = line.substr(0, line.find('#'));
  for (unsigned char c : line) {
    if ((c < 32 && c != '\t') || c == 127) fail("invalid control character");
  }
  std::istringstream input(line);
  input.imbue(std::locale::classic());
  Tokens tokens;
  for (std::string token; input >> token;) tokens.push_back(token);
  return tokens;
}

bool identifier(const std::string& s) {
  for (std::size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (!((c >= 'a' && c <= 'z') || c == '_' ||
          (i > 0 && c >= '0' && c <= '9'))) return false;
  }
  return !s.empty();
}

std::size_t natural(const std::string& token, std::size_t maximum,
                    const std::string& name) {
  if (token.empty()) fail("invalid integer in " + name);
  std::size_t value = 0;
  for (char c : token) {
    if (c < '0' || c > '9' || static_cast<unsigned>(c - '0') > maximum ||
        value > (maximum - static_cast<unsigned>(c - '0')) / 10)
      fail("integer out of range in " + name);
    value = value * 10 + (c - '0');
  }
  return value;
}

double number(const std::string& token, const std::string& name, bool as_float = false) {
  std::istringstream input(token);
  input.imbue(std::locale::classic());
  double value;
  if (!(input >> value) || input.peek() != std::char_traits<char>::eof() ||
      !std::isfinite(value)) fail("invalid finite decimal in " + name);
  if (as_float && (std::abs(value) > std::numeric_limits<float>::max() ||
      (value != 0.0 && static_cast<float>(value) == 0.0f)))
    fail("float out of range in " + name);
  return value;
}

struct Document {
  Tokens elements;
  std::map<std::string, Tokens> fields, blocks;
  std::set<std::string> used_fields, used_blocks;

  const Tokens& field(const std::string& name, std::size_t lo, std::size_t hi) {
    auto it = fields.find(name);
    if (it == fields.end()) fail("missing field " + name);
    if (it->second.size() < lo || it->second.size() > hi)
      fail("incorrect value count in " + name);
    used_fields.insert(name);
    return it->second;
  }
  int integer(const std::string& name, std::size_t i, int maximum) {
    return static_cast<int>(natural(fields.at(name).at(i), maximum, name));
  }
  const Tokens& block(const std::string& name, std::size_t count) {
    auto it = blocks.find(name);
    if (it == blocks.end()) fail("missing block " + name);
    if (it->second.size() != count) fail("incorrect value count in block " + name);
    used_blocks.insert(name);
    for (const auto& value : it->second) number(value, name, true);
    return it->second;
  }
  void check_used() const {
    for (const auto& entry : fields)
      if (!used_fields.count(entry.first)) fail("unhandled field " + entry.first);
    for (const auto& entry : blocks)
      if (!used_blocks.count(entry.first)) fail("unhandled block " + entry.first);
  }
};

Document read_document(std::istream& input) {
  Document doc;
  std::string line;
  std::getline(input, line);
  const auto header = tokenize(line);
  if (header.size() < 3 || lower(header[0]) != "nep") fail("invalid first line");
  const auto types = natural(header[1], 118, "type count");
  if (types == 0 || types != header.size() - 2) fail("element count mismatch");
  std::set<std::string> elements;
  for (std::size_t i = 2; i < header.size(); ++i) {
    auto symbol = lower(header[i]);
    if (symbol.empty()) fail("empty element");
    symbol[0] = static_cast<char>(symbol[0] - 'a' + 'A');
    if (!elements.insert(symbol).second) fail("duplicate element " + symbol);
    doc.elements.push_back(symbol); // Engines validate the chemical symbols.
  }
  std::set<std::string> names;
  std::string active;
  std::size_t remaining = 0, line_number = 1;
  while (std::getline(input, line)) {
    ++line_number;
    const auto tokens = tokenize(line);
    if (tokens.empty()) continue;
    if (remaining) {
      if (tokens.size() > remaining) fail("too many values at line " + std::to_string(line_number));
      for (const auto& token : tokens) number(token, active);
      auto& values = doc.blocks.at(active);
      values.insert(values.end(), tokens.begin(), tokens.end());
      remaining -= tokens.size();
      continue;
    }
    const bool block = tokens[0] == "@";
    if ((block && tokens.size() != 3) || (!block && tokens.size() < 2))
      fail("invalid record at line " + std::to_string(line_number));
    const auto name = lower(tokens[block ? 1 : 0]);
    if (!identifier(name) || !names.insert(name).second)
      fail("invalid or duplicate record " + name);
    if (block) {
      remaining = natural(tokens[2], Tokens().max_size(), name);
      active = name;
      doc.blocks[name] = {};
    } else {
      doc.fields[name] = Tokens(tokens.begin() + 1, tokens.end());
    }
  }
  if (input.bad()) fail("input stream failure");
  if (remaining) fail("truncated block " + active);
  return doc;
}

// Syntax/layout contract: GPUMD spin3-refactor b7d58c1, utilities/nep_*_file.
// Only structural coefficients transpose; magnetic and ANN parameters retain
// their original order. Token strings preserve the CPU engine's FP64 inputs.
std::string normalize(Document& doc, std::vector<double>& reference) {
  if (doc.field("format_version", 1, 1)[0] != "1") fail("unsupported format_version (expected 1)");
  const auto model = lower(doc.field("model", 1, 1)[0]);
  const bool spin = model == "nep_spin" || model == "nep_spin3";
  const bool charge = model == "qnep1" || model == "qnep2";
  const bool response = model == "nep_dipole" || model == "nep_polarizability";
  if (!spin && !charge && !response && model != "nep") fail("unsupported model " + model);
  const std::size_t types = doc.elements.size(), pairs = types * types;
  const bool zbl = doc.fields.count("zbl") != 0;
  if (response && zbl) fail("ZBL is not supported for response models");
  std::string tag = spin ? "nep4_spin3" : "nep4";
  if (zbl) tag += "_zbl";
  if (charge) tag += model == "qnep1" ? "_charge1" : "_charge2";
  if (response) tag += model == "nep_dipole" ? "_dipole" : "_polarizability";
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << tag << ' ' << types;
  for (const auto& element : doc.elements) out << ' ' << element;
  out << '\n';
  auto emit = [&](const std::string& name, const Tokens& values) {
    out << name;
    for (const auto& value : values) out << ' ' << value;
    out << '\n';
  };
  if (!response) {
    const auto& values = doc.field("energy_reference", types, types);
    if (!spin) for (const auto& value : values) reference.push_back(number(value, "energy_reference"));
    else for (const auto& value : values) number(value, "energy_reference");
  }
  int spin_dim = 0, magnetic_count = 0, projection = 0;
  if (spin) {
    const auto key = model == "nep_spin" ? "spin_version" : "spin_mode";
    if (doc.field(key, 1, 1)[0] != (model == "nep_spin" ? "1" : "3"))
      fail(std::string("unsupported ") + key);
    const bool selected = doc.fields.count("spin_type");
    const bool dof = doc.fields.count("spin_dof_type"), env = doc.fields.count("spin_env_type");
    if (selected && (dof || env))
      fail("spin_type cannot be combined with legacy spin_dof_type/spin_env_type");
    // New selectors restrict spin centers, never the geometric environment.
    // Historical files retain their explicit env mask and env=dof default.
    out << "spin_mode 3 " << 9 + (selected ? 2 : dof + env) << '\n';
    emit("spin_baseline", doc.fields.at("energy_reference"));
    for (const auto* name : {"spin_basis_size", "spin_l_max", "spin_compress", "spin_order",
                             "spin_soc", "spin_projection_size", "spin_scaler"})
      emit(name, doc.field(name, 1, 1));
    const int c = doc.integer("spin_compress", 0, 9);
    const int sb = doc.integer("spin_basis_size", 0, 8);
    const int l = doc.integer("spin_l_max", 0, 2);
    const int order = doc.integer("spin_order", 0, 3);
    const int soc = doc.integer("spin_soc", 0, 1);
    projection = doc.integer("spin_projection_size", 0, 324);
    if (c == 0 || order == 0 || sb != 8 || projection != 4 * c * c ||
        doc.integer("spin_scaler", 0, 1) != 1) fail("unsupported Spin3 shape");
    spin_dim = common::make_spin_polynomial_layout(c, l, order, soc, true).descriptor_dim;
    magnetic_count = static_cast<int>(pairs) * c * (sb + 1);
    const auto& cutoff = doc.field("spin_cutoff", 1, types);
    if (cutoff.size() != 1 && cutoff.size() != types) fail("invalid spin_cutoff count");
    for (const auto& value : cutoff) if (number(value, "spin_cutoff") <= 0) fail("invalid spin_cutoff");
    emit("spin_cutoff", cutoff);
    for (const auto* name : {"spin_type", "spin_dof_type", "spin_env_type"}) {
      if (!doc.fields.count(name)) continue;
      auto values = doc.field(name, 1, types);
      std::set<std::string> seen;
      for (auto& value : values) {
        auto found = std::find_if(doc.elements.begin(), doc.elements.end(),
          [&](const auto& element) { return lower(element) == lower(value); });
        if (found == doc.elements.end() || !seen.insert(*found).second) fail(std::string("invalid ") + name);
        value = *found;
      }
      emit(std::string(name) == "spin_type" ? "spin_dof_type" : name, values);
    }
    if (selected) emit("spin_env_type", doc.elements);
  }
  bool flexible_zbl = false;
  if (zbl) {
    const auto& values = doc.field("zbl", 2, 3);
    const double inner = number(values[0], "zbl"), outer = number(values[1], "zbl");
    flexible_zbl = inner == 0 && outer == 0;
    if (flexible_zbl ? values.size() != 2 : inner < 0 || outer <= inner) fail("invalid ZBL cutoffs");
    if (values.size() == 3 && number(values[2], "zbl") <= 0) fail("invalid ZBL factor");
    emit("zbl", values);
  }
  const auto& cutoff = doc.field("cutoff", 4, 2 * types + 2);
  if (cutoff.size() != 4 && cutoff.size() != 2 * types + 2) fail("invalid cutoff count");
  if (charge && cutoff.size() != 4) fail("qNEP requires uniform cutoffs");
  for (std::size_t i = 0; i + 2 < cutoff.size(); i += 2) {
    const double r = number(cutoff[i], "cutoff"), a = number(cutoff[i + 1], "cutoff");
    if (a <= 0 || r < a) fail("invalid cutoffs");
  }
  for (std::size_t i = cutoff.size() - 2; i < cutoff.size(); ++i)
    if (!natural(cutoff[i], std::numeric_limits<int>::max() / 2, "cutoff")) fail("invalid neighbor count");
  emit("cutoff", cutoff);
  emit("n_max", doc.field("n_max", 2, 2));
  emit("basis_size", doc.field("basis_size", 2, 2));
  emit("l_max", doc.field("l_max", 3, 7));
  emit("ANN", doc.field("ann", 2, 2));
  const int nr = doc.integer("n_max", 0, 12), na = doc.integer("n_max", 1, 8);
  const int br = doc.integer("basis_size", 0, 16), ba = doc.integer("basis_size", 1, 12);
  const int l = doc.integer("l_max", 0, 8);
  int channels = l;
  const auto& lmax = doc.fields.at("l_max");
  for (std::size_t i = 1; i < lmax.size(); ++i) {
    const int flag = doc.integer("l_max", i, i == 1 ? 2 : 1);
    const int minimum[] = {0, 2, 1, 2, 3, 3, 4};
    if (flag && l < minimum[i]) fail("l_max flag exceeds angular order");
    channels += flag != 0;
  }
  const int dim = nr + 1 + (na + 1) * channels + spin_dim;
  if ((na + 1) * channels > 90 || dim > (spin ? 199 : 103)) fail("descriptor dimension exceeds supported range");
  const int h1 = doc.integer("ann", 0, 120), h2 = doc.integer("ann", 1, 120);
  if (!h1 || (h2 && model != "nep")) fail("unsupported ANN shape");
  std::size_t ann_count = h2 ? (dim + 1) * h1 + (h1 + 2) * h2 : (dim + 2 + charge) * h1;
  ann_count = ann_count * types + (charge ? 2 : 1);
  if (model == "nep_polarizability") ann_count *= 2;
  const std::size_t radial_basis = (nr + 1) * (br + 1), angular_basis = (na + 1) * (ba + 1);
  auto parameters = doc.block("parameters", ann_count + pairs * (radial_basis + angular_basis) + magnetic_count + projection);
  std::size_t offset = ann_count;
  for (const auto count : {radial_basis, angular_basis}) {
    const Tokens original(parameters.begin() + offset, parameters.begin() + offset + pairs * count);
    for (std::size_t pair = 0; pair < pairs; ++pair)
      for (std::size_t basis = 0; basis < count; ++basis)
        parameters[offset + basis * pairs + pair] = original[pair * count + basis];
    offset += pairs * count;
  }
  for (const auto& value : parameters) out << value << '\n';
  for (const auto& value : doc.block("q_scaler", dim)) {
    if (number(value, "q_scaler", true) <= 0) fail("q_scaler must be positive");
    out << value << '\n';
  }
  if (flexible_zbl) {
    const auto& values = doc.block("zbl_parameters", 5 * types * (types + 1));
    for (std::size_t i = 0; i < values.size(); i += 10) {
      const double inner = number(values[i], "zbl_parameters"), outer = number(values[i + 1], "zbl_parameters");
      if (inner < 0 || outer <= inner) fail("invalid flexible ZBL cutoffs");
    }
    for (const auto& value : values) out << value << '\n';
  }
  doc.check_used();
  return out.str();
}
}  // namespace

ModelInput::ModelInput(const std::string& path) : file_(open_model_input(path)) {
  if (!file_.is_open()) throw std::runtime_error("failed to open NEP model: " + path);
  std::string token;
  file_ >> token;
  file_.clear();
  file_.seekg(0);
  block_format_ = lower(token) == "nep";
  if (block_format_) {
    auto doc = read_document(file_);
    converted_.str(normalize(doc, reference_));
    spin_type_semantics_ = doc.fields.count("spin_type") != 0;
  }
}
}  // namespace nep_adapters::detail
