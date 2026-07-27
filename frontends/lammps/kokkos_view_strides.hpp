#pragma once

#include <array>
#include <cstddef>

namespace nep_adapters::lammps_detail {

template <class View>
auto kokkos_view_strides(const View& view)
    -> std::array<std::size_t, static_cast<std::size_t>(View::rank) + 1> {
  static_assert(View::rank == 2, "expected a rank-2 Kokkos view");
  std::array<std::size_t, static_cast<std::size_t>(View::rank) + 1> strides{};
  view.stride(strides.data());
  return strides;
}

}  // namespace nep_adapters::lammps_detail
