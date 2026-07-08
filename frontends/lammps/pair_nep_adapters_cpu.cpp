#include "pair_nep_adapters_cpu.h"

#include "nep_adapters/engines/cpu_nep3.hpp"

using namespace LAMMPS_NS;

PairNEPAdaptersCPU::PairNEPAdaptersCPU(LAMMPS* lmp)
    : PairNEPAdaptersCommon(
          lmp,
          "nep/cpu",
          "cpu_nep3",
          "NEPAdapters CPU",
          &nep_adapters::register_cpu_nep3_engine) {}
