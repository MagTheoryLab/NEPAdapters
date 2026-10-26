#ifndef NEP_ADAPTERS_DETAIL_MODEL_FILE_HPP_
#define NEP_ADAPTERS_DETAIL_MODEL_FILE_HPP_

#include <fstream>
#include <ios>
#include <string>
#include <sstream>
#include <vector>
#include "nep_adapters/api.h"

#if defined(_WIN32)
#include <filesystem>
#endif

namespace nep_adapters::detail {

// Public model paths are UTF-8. On Windows, u8path converts that contract to
// the native wide-character filesystem representation used by file streams.
inline std::ifstream open_model_input(
    const std::string& utf8_path,
    std::ios_base::openmode mode = std::ios_base::in) {
#if defined(_WIN32)
  return std::ifstream(std::filesystem::u8path(utf8_path), mode);
#else
  return std::ifstream(utf8_path, mode);
#endif
}

// Normalize the versioned block syntax to the existing engine input grammar.
// Legacy files are streamed unchanged. Non-spin reference energies remain
// double precision and are applied by the engines, outside the ANN parameters.
class NEP_ADAPTERS_API ModelInput {
 public:
  explicit ModelInput(const std::string& path);
  std::istream& stream() {
    return block_format_ ? static_cast<std::istream&>(converted_) : file_;
  }
  const std::vector<double>& energy_reference() const { return reference_; }

 private:
  bool block_format_ = false;
  std::ifstream file_;
  std::istringstream converted_;
  std::vector<double> reference_;
};

}  // namespace nep_adapters::detail

#endif  // NEP_ADAPTERS_DETAIL_MODEL_FILE_HPP_
