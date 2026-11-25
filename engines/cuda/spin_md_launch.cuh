/* Adapted from GPUMD 73f6b09. SPDX-License-Identifier: LGPL-3.0-or-later */
#pragma once
#include "device_operations.hpp"
#include "spin_md_kernels.cuh"

namespace nep_adapters::spin_density::md {
bool prefer_ghost_coordinates(const View& layout);

// Specialize only the two polynomial ranks. Channel count and SOC coupling
// stay in the model tables, so all such models share the same kernels.
template <int L, int Order>
struct MDPolynomialLayout : View {
  enum : int {
    lmax = L,
    harmonic_capacity = (L + 1) * (L + 1),
    spin_lmax = L < Order ? L : Order,
    spin_components = PrimitiveBasis(L, Order).components,
    primitive_count = PrimitiveBasis(L, Order).count,
    spin_capacity = spin_components,
    // SOC has at most four parity sectors per rank; no-SOC has the full product.
    component_capacity = harmonic_capacity *
      ((spin_lmax + 1) * (spin_lmax + 1) > 4 ? (spin_lmax + 1) * (spin_lmax + 1) : 4),
    tile = spin_density::md_force_lanes(L, Order),
    edge_cached = false
  };
  explicit MDPolynomialLayout(View v) : View(v) {}
  const float* edge_radial = nullptr;
  __device__ __forceinline__ Primitive primitive(int i) const
  {
    constexpr PrimitiveBasis basis(L, Order);
    return basis.entries[i];
  }
  __device__ Sector sector(int i) const { return sectors[i]; }
  __device__ Term term(int i) const { return terms[i]; }
  __device__ Term quadratic_term(int i) const { return terms[i]; }
  __device__ Center center(int i) const { return centers[i]; }
  __device__ Harmonic harmonic(int i) const { return harmonics[i]; }
};

// One full warp writes and consumes different moment components of the same center.
template <int L, int Order>
struct CenterMomentRows<MDPolynomialLayout<L, Order>>
  : std::integral_constant<bool, MDPolynomialLayout<L, Order>::tile == 32> {};

// The L2/O2 term and primitive tables are independent of C. Only the
// descriptor-column stride changes when a sector has more or fewer channels.
template <class Base>
struct MDQuadraticChannels : Base {
  int channels;
  explicit MDQuadraticChannels(int count) : channels(count) {}
  __device__ __forceinline__ Sector sector(int i) const
  {
    constexpr int local_columns = PrimitiveBasis(Base::lmax, Base::order).scalars;
    auto entry = Base::sector(i);
    entry.norm_column =
      local_columns + (entry.norm_column - local_columns) / Base::channels * channels;
    if (entry.scalar_column >= 0)
      entry.scalar_column =
        local_columns + (entry.scalar_column - local_columns) / Base::channels * channels;
    return entry;
  }
};


template<bool UseRadialCache, class Layout, bool Quadratic, class Coordinates, class Real>
void launch(Layout layout, const cuda_backend::ModelProtocol& p, int count,
            Coordinates geometry, const cuda_backend::DeviceModelView& m,
            const cuda_backend::DeviceWorkspaceView& w, bool backward, int virial_mode,
            bool transfer, const Real* coefficients) {
  const int n = static_cast<int>(w.atom_capacity);
  const int plane = n*p.neighbor_capacity_radial;
  constexpr bool ghost_coordinates = std::is_same_v<Coordinates,GhostCoordinates>;
  constexpr int density_elements = Layout::harmonic_capacity * Layout::spin_capacity;
  // With ghost coordinates a small density fits in the subwarp's registers.
  constexpr bool warp_density = std::is_base_of_v<View,Layout> &&
    (!ghost_coordinates || density_elements > SpinProjection::width) &&
    md_warp_density(Layout::harmonic_capacity,Layout::spin_capacity);
  auto* cache = reinterpret_cast<RadialEdge<Real>*>(w.spin_radial_cache);
  const auto* params = m.spin_parameters;
  if (!backward) {
    constexpr int lanes = warp_density ? 32 : Layout::tile;
    auto* geometric = m.spin_transpose.neighbor_weights
      ? w.spin_moments + static_cast<std::size_t>(n)*layout.channels*layout.components : nullptr;
    descriptors<warp_density,UseRadialCache><<<(count*p.spin_compress*lanes+63)/64,64>>>(
      layout,count,n,p.struct_descriptor_dim,p.num_types,p.n_max_angular,p.spin_basis_size,
      geometry,w.types,m.spin_dof_type_active,w.positions_soa3,w.spins_soa3,w.nn_radial,
      w.nl_radial_slot_major,nullptr,plane,m.spin_cutoff_pair,coefficients,params,
      w.spin_moments,w.descriptors,m.spin_forward,geometric,cache);
  } else {
    constexpr bool can_fuse = !Quadratic && std::is_same_v<Real,float> &&
      (Layout::tile == int(SpinProjection::width) || Layout::harmonic_capacity <= 4);
    constexpr bool cache_rows = !Quadratic && Layout::tile > 1 && std::is_same_v<Real,float>;
    auto launch_force = [&](auto fuse, auto cached) {
      constexpr bool fused = decltype(fuse)::value, paired = decltype(cached)::value;
      constexpr int batch = fused ? 3 : 1;
      const int batches = (p.spin_compress+batch-1)/batch;
      forces<Layout,true,Quadratic,fused,paired,UseRadialCache,!ghost_coordinates>
        <<<(count*batches*Layout::tile+63)/64,64>>>(
          layout,count,n,p.struct_descriptor_dim,p.num_types,p.n_max_angular,p.spin_basis_size,
          geometry,w.types,m.spin_dof_type_active,w.positions_soa3,w.spins_soa3,w.nn_radial,
          w.nl_radial_slot_major,nullptr,plane,m.spin_cutoff_pair,coefficients,params,
          w.spin_moments,w.fp,w.force_soa3,w.mforce_soa3,w.virial_soa9,virial_mode,
          w.per_atom_virial_float_soa9,transfer ? w.spin_transfer_soa9 : nullptr,
          m.spin_md_adjoint_order,m.spin_transpose,cache);
    };
    if constexpr (can_fuse) {
      if (p.spin_compress > 1) {
        constexpr bool compact = md_compact_rows(Layout::tile,
          Layout::harmonic_capacity*Layout::spin_capacity,true);
        if constexpr (compact)
          if (!m.spin_transpose.neighbor_weights) {
            launch_force(std::true_type{},std::false_type{});
            return;
          }
        launch_force(std::true_type{},std::bool_constant<cache_rows>{});
        return;
      }
    }
    launch_force(std::false_type{},std::bool_constant<cache_rows>{});
  }
}

// NVCC with MSVC cannot instantiate the nested generic lambdas in this dispatch.
template<int L, int O, class Real>
void launch_order_with_coefficients(
    const cuda_backend::ModelProtocol& p, int count, const Geometry& geometry,
    const cuda_backend::DeviceModelView& m, const cuda_backend::DeviceWorkspaceView& w,
    bool backward, int mode, bool transfer, const Real* coefficients) {
  using Layout = MDPolynomialLayout<L,O>;
  const Layout layout(m.spin_layout);
  constexpr bool quadratic = md_quadratic(L,O);
  if (!w.spin_radial_cache)
    launch<false,Layout,quadratic>(layout,p,count,geometry,m,w,backward,mode,transfer,coefficients);
  // MD can use the narrower force kernel when only center virial is requested.
  else if (w.neighbor_source == 1 && (!backward || (mode < 2 && !transfer)) &&
           prefer_ghost_coordinates(m.spin_layout))
    launch<true,Layout,quadratic>(layout,p,count,GhostCoordinates{},m,w,backward,mode,transfer,coefficients);
  else
    launch<true,Layout,quadratic>(layout,p,count,geometry,m,w,backward,mode,transfer,coefficients);
}

template<int L, int O>
void launch_order(const cuda_backend::ModelProtocol& p, int count, const Geometry& geometry,
                  const cuda_backend::DeviceModelView& m, const cuda_backend::DeviceWorkspaceView& w,
                  bool backward, int mode, bool transfer) {
  if constexpr (md_wide(L,O,false)) {
    if (!p.spin_soc) {
      launch_order_with_coefficients<L,O>(
        p,count,geometry,m,w,backward,mode,transfer,m.spin_radial_coefficients_wide);
      return;
    }
  }
  launch_order_with_coefficients<L,O>(
    p,count,geometry,m,w,backward,mode,transfer,m.spin_radial_coefficients);
}

template<int L>
void launch_order(const cuda_backend::ModelProtocol& p, int count,
                  const cuda_backend::SimulationBox& box, const cuda_backend::DeviceModelView& m,
                  const cuda_backend::DeviceWorkspaceView& w, bool backward, int mode, bool transfer) {
  const Geometry geometry{box, w.neighbor_source, w.atom_to_structure,
    w.boxes_row_major9, w.box_inverse_row_major9, w.pbc_flags3};
  switch (p.spin_order) {
    case 1: launch_order<L,1>(p,count,geometry,m,w,backward,mode,transfer); break;
    case 2:
      if constexpr (L != 2) launch_order<L,2>(p,count,geometry,m,w,backward,mode,transfer);
      break;
    case 3: launch_order<L,3>(p,count,geometry,m,w,backward,mode,transfer); break;
    case 4: launch_order<L,4>(p,count,geometry,m,w,backward,mode,transfer); break;
  }
}
} // namespace nep_adapters::spin_density::md
