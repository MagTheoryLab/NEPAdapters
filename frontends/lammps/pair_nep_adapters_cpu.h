/* -*- c++ -*- */
#ifdef PAIR_CLASS
// clang-format off
PairStyle(nep/cpu,PairNEPAdaptersCPU);
// clang-format on
#else

#ifndef LMP_PAIR_NEP_ADAPTERS_CPU_H
#define LMP_PAIR_NEP_ADAPTERS_CPU_H

#include "pair.h"

#include <string>
#include <vector>

struct NepaModel;

namespace LAMMPS_NS {

class PairNEPAdaptersCPU : public Pair {
 public:
  PairNEPAdaptersCPU(class LAMMPS*);
  ~PairNEPAdaptersCPU() override;

  void compute(int, int) override;
  void settings(int, char**) override;
  void coeff(int, char**) override;
  void init_style() override;
  double init_one(int, int) override;

 private:
  void allocate();
  void read_type_map(const std::string& model_path, int narg, char** arg);
  void load_model(const std::string& model_path);

  int* type_map_ = nullptr;
  NepaModel* model_ = nullptr;
  std::string model_filename_;
  double cutoff_ = 0.0;

  std::vector<int> sanitized_numneigh_;
  std::vector<int*> sanitized_firstneigh_;
  std::vector<std::vector<int>> sanitized_neighbors_;
  std::vector<double> potential_;
  std::vector<double> virials_per_atom_;
  std::vector<double*> virial_rows_;
};

}  // namespace LAMMPS_NS

#endif
#endif
