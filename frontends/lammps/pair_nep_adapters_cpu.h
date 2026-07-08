/* -*- c++ -*- */
#ifdef PAIR_CLASS
// clang-format off
PairStyle(nep/cpu,PairNEPAdaptersCPU);
// clang-format on
#else

#ifndef LMP_PAIR_NEP_ADAPTERS_CPU_H
#define LMP_PAIR_NEP_ADAPTERS_CPU_H

#include "pair_nep_adapters_common.h"

namespace LAMMPS_NS {

class PairNEPAdaptersCPU : public PairNEPAdaptersCommon {
 public:
  PairNEPAdaptersCPU(class LAMMPS*);
};

}  // namespace LAMMPS_NS

#endif
#endif
