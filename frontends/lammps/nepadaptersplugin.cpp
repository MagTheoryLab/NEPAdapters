#include "lammpsplugin.h"
#include "version.h"

#include "pair_nep_adapters_cpu.h"

using namespace LAMMPS_NS;

namespace {

static Pair* pair_nep_adapters_cpu_creator(LAMMPS* lmp) {
  return new PairNEPAdaptersCPU(lmp);
}

void register_pair_style(
    void* lmp,
    void* handle,
    lammpsplugin_regfunc register_plugin,
    const char* name,
    const char* info,
    lammpsplugin_factory1* creator) {
  lammpsplugin_t plugin;
  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = name;
  plugin.info = info;
  plugin.author = "NEPAdapters developers";
  plugin.creator.v1 = creator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}

}  // namespace

extern "C" void lammpsplugin_init(void* lmp, void* handle, void* regfunc) {
  auto register_plugin = reinterpret_cast<lammpsplugin_regfunc>(regfunc);
  register_pair_style(
      lmp,
      handle,
      register_plugin,
      "nep/cpu",
      "NEPAdapters CPU pair style",
      reinterpret_cast<lammpsplugin_factory1*>(&pair_nep_adapters_cpu_creator));
}
