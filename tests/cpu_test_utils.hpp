#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace cpu_test {

struct Frame {
  std::vector<std::int32_t> types;
  std::vector<double> positions_aos3;
  std::vector<double> reference_forces_aos3;
  double box[9] = {};
  double reference_energy = 0.0;
  double reference_virial_row_major9[9] = {};
  bool has_reference_forces = false;
  bool has_reference_virial = false;
};

struct Matrix {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<double> values;
};

inline std::unordered_map<std::string, std::int32_t> read_type_map(
    const std::string& model_path) {
  std::ifstream input(model_path);
  std::string header;
  std::getline(input, header);

  std::istringstream stream(header);
  std::string tag;
  int num_types = 0;
  stream >> tag >> num_types;

  std::unordered_map<std::string, std::int32_t> map;
  for (int index = 0; index < num_types; ++index) {
    std::string symbol;
    stream >> symbol;
    map[symbol] = index;
  }
  return map;
}

inline bool parse_lattice(const std::string& comment, double* box) {
  const std::string key = "Lattice=\"";
  const std::size_t begin = comment.find(key);
  if (begin == std::string::npos) {
    return false;
  }

  const std::size_t values_begin = begin + key.size();
  const std::size_t end = comment.find('"', values_begin);
  if (end == std::string::npos) {
    return false;
  }

  std::istringstream stream(comment.substr(values_begin, end - values_begin));
  for (int index = 0; index < 9; ++index) {
    if (!(stream >> box[index])) {
      return false;
    }
  }
  return true;
}

inline bool parse_energy(const std::string& comment, double& energy) {
  const std::string key = "energy=";
  const std::size_t begin = comment.find(key);
  if (begin == std::string::npos) {
    return false;
  }

  std::istringstream stream(comment.substr(begin + key.size()));
  stream >> energy;
  return static_cast<bool>(stream);
}

inline bool parse_quoted_doubles(
    const std::string& comment,
    const std::string& key,
    double* values,
    int count) {
  const std::string prefix = key + "=\"";
  const std::size_t begin = comment.find(prefix);
  if (begin == std::string::npos) {
    return false;
  }

  const std::size_t values_begin = begin + prefix.size();
  const std::size_t end = comment.find('"', values_begin);
  if (end == std::string::npos) {
    return false;
  }

  std::istringstream stream(comment.substr(values_begin, end - values_begin));
  for (int index = 0; index < count; ++index) {
    if (!(stream >> values[index])) {
      return false;
    }
  }
  return true;
}

inline Frame read_first_frame(
    const std::string& xyz_path,
    const std::unordered_map<std::string, std::int32_t>& type_map) {
  std::ifstream input(xyz_path);
  std::string line;
  std::getline(input, line);

  const int atom_count = std::stoi(line);
  std::getline(input, line);

  Frame frame;
  frame.types.resize(static_cast<std::size_t>(atom_count));
  frame.positions_aos3.resize(static_cast<std::size_t>(atom_count) * 3);
  frame.reference_forces_aos3.resize(static_cast<std::size_t>(atom_count) * 3);
  if (!parse_lattice(line, frame.box) ||
      !parse_energy(line, frame.reference_energy)) {
    std::exit(EXIT_FAILURE);
  }
  frame.has_reference_virial =
      parse_quoted_doubles(line, "virial", frame.reference_virial_row_major9, 9);

  for (int atom = 0; atom < atom_count; ++atom) {
    std::getline(input, line);
    std::istringstream stream(line);
    std::string symbol;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    stream >> symbol >> x >> y >> z >> fx >> fy >> fz;
    if (!stream) {
      std::exit(EXIT_FAILURE);
    }

    const auto found = type_map.find(symbol);
    if (found == type_map.end()) {
      std::exit(EXIT_FAILURE);
    }

    frame.types[atom] = found->second;
    frame.positions_aos3[3 * atom + 0] = x;
    frame.positions_aos3[3 * atom + 1] = y;
    frame.positions_aos3[3 * atom + 2] = z;
    frame.reference_forces_aos3[3 * atom + 0] = fx;
    frame.reference_forces_aos3[3 * atom + 1] = fy;
    frame.reference_forces_aos3[3 * atom + 2] = fz;
  }
  frame.has_reference_forces = true;

  return frame;
}

inline Frame read_xyz_in(
    const std::string& xyz_path,
    const std::unordered_map<std::string, std::int32_t>& type_map) {
  std::ifstream input(xyz_path);
  int atom_count = 0;
  input >> atom_count;
  if (!input || atom_count <= 0) {
    std::exit(EXIT_FAILURE);
  }

  Frame frame;
  frame.types.resize(static_cast<std::size_t>(atom_count));
  frame.positions_aos3.resize(static_cast<std::size_t>(atom_count) * 3);
  input >> frame.box[0] >> frame.box[3] >> frame.box[6] >> frame.box[1] >>
      frame.box[4] >> frame.box[7] >> frame.box[2] >> frame.box[5] >>
      frame.box[8];

  for (int atom = 0; atom < atom_count; ++atom) {
    std::string symbol;
    input >> symbol >> frame.positions_aos3[3 * atom + 0] >>
        frame.positions_aos3[3 * atom + 1] >>
        frame.positions_aos3[3 * atom + 2];
    const auto found = type_map.find(symbol);
    if (!input || found == type_map.end()) {
      std::exit(EXIT_FAILURE);
    }
    frame.types[atom] = found->second;
  }
  return frame;
}

inline Matrix read_matrix(const std::string& path) {
  std::ifstream input(path);
  std::string marker;
  std::string shape;
  Matrix matrix;
  input >> marker >> shape >> matrix.rows >> matrix.cols;
  if (!input || marker != "#" || shape != "shape" ||
      matrix.rows == 0 || matrix.cols == 0) {
    std::exit(EXIT_FAILURE);
  }

  matrix.values.resize(matrix.rows * matrix.cols, 0.0);
  for (double& value : matrix.values) {
    input >> value;
    if (!input) {
      std::exit(EXIT_FAILURE);
    }
  }
  return matrix;
}

inline Matrix read_plain_matrix(const std::string& path, std::size_t cols) {
  std::ifstream input(path);
  Matrix matrix;
  matrix.cols = cols;
  double value = 0.0;
  while (input >> value) {
    matrix.values.push_back(value);
  }
  if (!input.eof() || cols == 0 || matrix.values.empty() ||
      matrix.values.size() % cols != 0) {
    std::exit(EXIT_FAILURE);
  }
  matrix.rows = matrix.values.size() / cols;
  return matrix;
}

inline bool all_finite(const std::vector<double>& values) {
  for (double value : values) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

}  // namespace cpu_test
