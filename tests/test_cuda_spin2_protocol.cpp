#include "device_model.hpp"
#include "model_protocol.hpp"
#include "nep_adapters/api.h"
#include "nep_adapters/engines/cpu.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
using nep_adapters::cuda_backend::HostModelParameters;
using nep_adapters::cuda_backend::ModelProtocol;
using nep_adapters::cuda_backend::load_host_model_parameters;
using nep_adapters::cuda_backend::parse_model_file;
using nep_adapters::cuda_backend::parse_model_protocol;

int expected_dim(int c, int lmax, int order, int soc) {
  const int pairs = c * (c + 1) / 2;
  int dim = 1 + 2 * c;
  if (soc && lmax >= 2) dim += 2 * c;
  if (order >= 2) {
    dim += 2 * c;
    if (lmax >= 1) dim += (soc ? 3 : 1) * c;
    if (lmax >= 2) dim += c;
    dim += c + 2 * pairs;
    if (soc && lmax >= 1) dim += 2 * c;
    if (soc && lmax >= 2) dim += 2 * c;
  }
  if (order >= 3) {
    dim += c;
    if (soc && lmax >= 1) dim += 2 * c;
    if (soc && lmax >= 2) dim += 3 * c;
    if (soc && lmax >= 1 && c >= 3) dim += c;
  }
  return dim;
}

std::vector<std::string> header(
    int c, int lmax, int order, int soc, bool masks = false) {
  std::vector<std::string> lines = {
      "nep4_spin2 3 Fe Ge C", "spin_mode 2 " + std::to_string(masks ? 11 : 9),
      "spin_baseline -1 -2 -3", "spin_basis_size 8",
      "spin_l_max " + std::to_string(lmax),
      "spin_compress " + std::to_string(c), "spin_cutoff 6",
      "spin_order " + std::to_string(order),
      "spin_soc " + std::to_string(soc),
      "spin_projection_size " + std::to_string(4 * c * c), "spin_scaler 1"};
  if (masks) {
    lines.push_back("spin_dof_type Fe C");
    lines.push_back("spin_env_type Fe Ge C");
  }
  lines.insert(lines.end(), {"cutoff 6 5 64 64", "n_max 0 0",
      "basis_size 0 0", "l_max 0 0 0", "ANN 1 0"});
  return lines;
}

std::string write_lines(const std::string& name,
                        const std::vector<std::string>& lines) {
  const std::string path =
      (std::filesystem::temp_directory_path() / name).string();
  std::ofstream out(path);
  for (const auto& line : lines) out << line << '\n';
  return path;
}

std::string complete(const std::string& name,
                     const std::vector<std::string>& lines,
                     int extra = 0, int truncate = 0) {
  const ModelProtocol p = parse_model_protocol(write_lines(name + ".h", lines));
  const std::size_t count = p.model_parameter_count + p.q_scaler_count;
  const std::string path =
      (std::filesystem::temp_directory_path() / name).string();
  std::ofstream out(path);
  for (const auto& line : lines) out << line << '\n';
  for (std::size_t i = 0; i < count - static_cast<std::size_t>(truncate); ++i)
    out << 0.001 * static_cast<double>(i + 1) << '\n';
  for (int i = 0; i < extra; ++i) out << "9.25\n";
  return path;
}

bool header_fails(const std::vector<std::string>& lines,
                  const std::string& needle) {
  try { (void)parse_model_protocol(write_lines("spin2_bad.nep", lines)); }
  catch (const std::exception& e) {
    return std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

bool file_fails(const std::string& path, const std::string& needle) {
  try { (void)parse_model_file(path); }
  catch (const std::exception& e) {
    return std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc == 2) {
    try {
      const HostModelParameters p = load_host_model_parameters(argv[1]);
      std::cout << "spin_mode=" << p.protocol.spin_mode << '\n'
                << "spin_descriptor_dim=" << p.protocol.spin_descriptor_dim << '\n'
                << "descriptor_dim=" << p.protocol.descriptor_dim << '\n'
                << "projection_count=" << p.spin_projection_parameters.size() << '\n';
      return p.protocol.spin_mode == 2 &&
             p.spin_projection_parameters.size() ==
                 static_cast<std::size_t>(p.protocol.spin_projection_size)
          ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& e) {
      std::cerr << e.what() << '\n';
      return EXIT_FAILURE;
    }
  }
  for (int c = 1; c <= 9; ++c) for (int l = 0; l <= 2; ++l)
    for (int o = 1; o <= 3; ++o) for (int s = 0; s <= 1; ++s) {
      const ModelProtocol p = parse_model_protocol(write_lines(
          "spin2_matrix.nep", header(c, l, o, s)));
      if (p.spin_descriptor_dim != expected_dim(c, l, o, s) ||
          p.spin_projection_size != 4 * c * c ||
          p.spin_projection_parameter_count !=
              static_cast<std::size_t>(4 * c * c)) {
        std::cerr << "dimension mismatch C=" << c << " l=" << l
                  << " O=" << o << " SOC=" << s << '\n';
        return EXIT_FAILURE;
      }
    }

  const auto rep = header(3, 2, 3, 1, true);
  const std::string valid = complete("spin2_valid.nep", rep);
  const HostModelParameters loaded = load_host_model_parameters(valid);
  if (loaded.protocol.spin_descriptor_dim != 79 ||
      loaded.spin_projection_parameters.size() != 36 ||
      loaded.protocol.spin_dof_type_active != std::vector<int>({1, 0, 1}) ||
      loaded.protocol.spin_env_type_active != std::vector<int>({1, 1, 1})) {
    std::cerr << "representative O3-C3 semantics mismatch\n";
    return EXIT_FAILURE;
  }

  auto bad = header(2, 2, 3, 1);
  bad[0] = "nep4_spin3 3 Fe Ge C";
  if (!header_fails(bad, "supported NEP4/NEP5")) return EXIT_FAILURE;
  bad = header(2, 2, 3, 1); bad[9] = "spin_projection_size 17";
  if (!header_fails(bad, "4 * spin_compress^2")) return EXIT_FAILURE;
  bad = header(2, 2, 3, 1); bad[1] = "spin_mode 2";
  if (!header_fails(bad, "requires counted")) return EXIT_FAILURE;
  bad = header(2, 2, 3, 1); bad[10] = "spin_type Fe";
  if (!header_fails(bad, "unknown spin header line")) return EXIT_FAILURE;
  bad = header(2, 2, 3, 1); bad.erase(bad.begin() + 8); bad[1] = "spin_mode 2 8";
  if (!header_fails(bad, "missing required metadata: spin_soc")) return EXIT_FAILURE;
  if (!file_fails(complete("spin2_short.nep", rep, 0, 1), "unexpected end") ||
      !file_fails(complete("spin2_long.nep", rep, 1, 0), "unexpected trailing"))
    return EXIT_FAILURE;

  if (!nep_adapters::register_cpu_engine()) return EXIT_FAILURE;
  NepaModel* cpu_model = nullptr;
  NepaModelInfo cpu_info{};
  if (nepa_load_model("cpu", valid.c_str(), &cpu_model) != NEPA_STATUS_OK ||
      nepa_model_info(cpu_model, &cpu_info) != NEPA_STATUS_OK ||
      cpu_info.descriptor_dim != loaded.protocol.descriptor_dim ||
      (cpu_info.capabilities & NEPA_CAPABILITY_SPIN) == 0) {
    nepa_free_model(cpu_model);
    return EXIT_FAILURE;
  }
  nepa_free_model(cpu_model);
  return EXIT_SUCCESS;
}
