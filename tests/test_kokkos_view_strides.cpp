#include "kokkos_view_strides.hpp"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <tuple>

namespace {

struct RankTwoViewProbe {
  static constexpr int rank = 2;

  void stride(std::size_t* values) const {
    values[0] = 1;
    values[1] = 640;
    values[2] = 1920;
  }
};

}  // namespace

int main() {
  const auto strides =
      nep_adapters::lammps_detail::kokkos_view_strides(RankTwoViewProbe{});
  static_assert(std::tuple_size_v<decltype(strides)> == 3);
  return strides == std::array<std::size_t, 3>{1, 640, 1920}
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
