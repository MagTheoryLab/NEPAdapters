#ifndef NEP_ADAPTERS_DETAIL_MODEL_FILE_HPP_
#define NEP_ADAPTERS_DETAIL_MODEL_FILE_HPP_

#include <fstream>
#include <ios>
#include <string>

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

}  // namespace nep_adapters::detail

#endif  // NEP_ADAPTERS_DETAIL_MODEL_FILE_HPP_
