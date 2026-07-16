#include "pair_nep_adapters_cpu.h"

#include "nep_adapters/engines/cpu.hpp"

using namespace LAMMPS_NS;

PairNEPAdaptersCPU::PairNEPAdaptersCPU(LAMMPS* lmp)
    : PairNEPAdaptersCommon(
          lmp,
          "nep/cpu",
          "cpu",
          "NEPAdapters CPU",
          &nep_adapters::register_cpu_engine) {}
