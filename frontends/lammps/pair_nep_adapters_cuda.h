/* -*- c++ -*- */
#ifdef PAIR_CLASS
// clang-format off
PairStyle(nep/gpu,PairNEPAdaptersCUDA);
#ifdef LMP_KOKKOS
PairStyle(nep/gpu/kk,PairNEPAdaptersCUDA);
PairStyle(nep/gpu/kk/device,PairNEPAdaptersCUDA);
#endif
// clang-format on
#else

#ifndef LMP_PAIR_NEP_ADAPTERS_CUDA_H
#define LMP_PAIR_NEP_ADAPTERS_CUDA_H

#include "pair_nep_adapters_common.h"

#ifdef LMP_KOKKOS
#include "kokkos_base.h"
#include "kokkos_type.h"
#endif

namespace LAMMPS_NS {

class PairNEPAdaptersCUDA : public PairNEPAdaptersCommon
#ifdef LMP_KOKKOS
    ,
                            public KokkosBase
#endif
{
 public:
  PairNEPAdaptersCUDA(class LAMMPS*);
  ~PairNEPAdaptersCUDA() override;

  void coeff(int, char**) override;
  void compute(int, int) override;
  void init_style() override;

#ifdef LMP_KOKKOS
  int pack_reverse_comm_kokkos(int, int, DAT::tdual_xfloat_1d&) override;
  void unpack_reverse_comm_kokkos(
      int,
      DAT::tdual_int_1d,
      DAT::tdual_xfloat_1d&) override;

 private:
  using KokkosCvatom = Kokkos::DualView<F_FLOAT*[9], Kokkos::LayoutRight, LMPDeviceType>;

  void ensure_kokkos_buffers();
  void ensure_kokkos_type_map();
  void ensure_kokkos_tally_buffers();
  void pack_kokkos_per_atom_tallies();

  Kokkos::View<int*, LMPDeviceType> d_type_map_;
  Kokkos::View<double*, LMPDeviceType> d_total_potential_;
  Kokkos::View<double*, LMPDeviceType> d_total_virial6_;
  DAT::tdual_efloat_1d k_eatom_;
  DAT::tdual_virial_array k_vatom_;
  KokkosCvatom k_cvatom_;
  typename DAT::t_efloat_1d d_eatom_;
  typename DAT::t_virial_array d_vatom_;
  typename KokkosCvatom::t_dev d_cvatom_;
  Kokkos::View<double*, LMPDeviceType> d_lammps_raw9_;
  int d_type_map_length_ = 0;
#else
 private:
#endif
  bool charge_model_ = false;
};

}  // namespace LAMMPS_NS

#endif
#endif
