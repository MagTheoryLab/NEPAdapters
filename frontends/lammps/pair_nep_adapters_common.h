/* -*- c++ -*- */
#ifndef LMP_PAIR_NEP_ADAPTERS_COMMON_H
#define LMP_PAIR_NEP_ADAPTERS_COMMON_H

#include "pair.h"

#include <string>
#include <vector>

struct NepaModel;

namespace LAMMPS_NS {

using PairNEPAdaptersRegisterEngine = bool (*)();

class PairNEPAdaptersCommon : public Pair {
 public:
  PairNEPAdaptersCommon(
      class LAMMPS*,
      const char* style_name,
      const char* engine_name,
      const char* label,
      PairNEPAdaptersRegisterEngine register_engine);
  ~PairNEPAdaptersCommon() override;

  void compute(int, int) override;
  void settings(int, char**) override;
  void coeff(int, char**) override;
  void init_style() override;
  double init_one(int, int) override;
  int pack_reverse_comm(int, int, double*) override;
  void unpack_reverse_comm(int, int*, double*) override;

 protected:
  void allocate();
  void read_type_map(const std::string& model_path, int narg, char** arg);
  void load_model(const std::string& model_path);
  void log_loaded_model(const std::string& model_path) const;

  const std::string style_name_;
  const std::string engine_name_;
  const std::string label_;
  PairNEPAdaptersRegisterEngine register_engine_ = nullptr;

  int* type_map_ = nullptr;
  NepaModel* model_ = nullptr;
  std::string model_filename_;
  std::vector<std::string> model_elements_;
  double cutoff_ = 0.0;
  bool spin_model_ = false;

  std::vector<int> sanitized_numneigh_;
  std::vector<int*> sanitized_firstneigh_;
  std::vector<std::vector<int>> sanitized_neighbors_;
  std::vector<double> potential_;
  std::vector<double> virials_per_atom_;
  std::vector<double*> virial_rows_;
  bool classic_reverse_per_atom_virial_ = false;
};

}  // namespace LAMMPS_NS

#endif
