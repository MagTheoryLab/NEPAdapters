#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

struct Atom {
  int type = 0;
  std::array<double, 3> position{};
};

struct RankCase {
  int nlocal = 0;
  int max_neighbors = 0;
  std::vector<int> global_index;
  std::vector<Atom> atoms;
  std::vector<int> neighbor_counts;
  std::vector<int> neighbors;
};

struct Result {
  std::vector<double> energy;
  std::vector<double> force_aos3;
  std::array<double, 9> virial_raw9{};
};

std::vector<Atom> make_atoms() {
  return {
      {0, {0.00, 0.00, 0.00}},
      {0, {1.10, 0.25, -0.10}},
      {0, {0.30, 1.20, 0.35}},
      {0, {1.35, 1.05, 0.62}},
  };
}

RankCase make_rank_case(
    const std::vector<Atom>& global_atoms,
    const std::vector<int>& owned) {
  const int atom_count = static_cast<int>(global_atoms.size());
  RankCase rank;
  rank.nlocal = static_cast<int>(owned.size());
  rank.max_neighbors = atom_count - 1;

  std::vector<int> slot_of_global(static_cast<std::size_t>(atom_count), -1);
  for (int global : owned) {
    if (global < 0 || global >= atom_count ||
        slot_of_global[static_cast<std::size_t>(global)] >= 0) {
      throw std::runtime_error("invalid owned atom list");
    }
    slot_of_global[static_cast<std::size_t>(global)] =
        static_cast<int>(rank.global_index.size());
    rank.global_index.push_back(global);
  }
  for (int global = 0; global < atom_count; ++global) {
    if (slot_of_global[static_cast<std::size_t>(global)] >= 0) {
      continue;
    }
    slot_of_global[static_cast<std::size_t>(global)] =
        static_cast<int>(rank.global_index.size());
    rank.global_index.push_back(global);
  }

  rank.atoms.reserve(rank.global_index.size());
  for (int global : rank.global_index) {
    rank.atoms.push_back(global_atoms[static_cast<std::size_t>(global)]);
  }

  rank.neighbor_counts.assign(static_cast<std::size_t>(rank.nlocal), 0);
  rank.neighbors.assign(
      static_cast<std::size_t>(rank.nlocal) * rank.max_neighbors,
      0);
  for (int local = 0; local < rank.nlocal; ++local) {
    const int center_global = rank.global_index[static_cast<std::size_t>(local)];
    int slot = 0;
    for (int neighbor_global = 0; neighbor_global < atom_count; ++neighbor_global) {
      if (neighbor_global == center_global) {
        continue;
      }
      rank.neighbors[static_cast<std::size_t>(local) +
                     static_cast<std::size_t>(rank.nlocal) * slot] =
          slot_of_global[static_cast<std::size_t>(neighbor_global)];
      ++slot;
    }
    rank.neighbor_counts[static_cast<std::size_t>(local)] = slot;
  }
  return rank;
}

void add_outer_product(
    std::array<double, 9>& virial,
    const std::array<double, 3>& dr,
    const std::array<double, 3>& force) {
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      virial[static_cast<std::size_t>(3 * row + col)] +=
          dr[static_cast<std::size_t>(row)] *
          force[static_cast<std::size_t>(col)];
    }
  }
}

Result run_center_model(const RankCase& rank) {
  constexpr double k = 1.7;
  constexpr double r0 = 0.85;

  Result result;
  result.energy.assign(rank.atoms.size(), 0.0);
  result.force_aos3.assign(rank.atoms.size() * 3, 0.0);

  for (int local = 0; local < rank.nlocal; ++local) {
    const Atom& center = rank.atoms[static_cast<std::size_t>(local)];
    const int neighbor_count = rank.neighbor_counts[static_cast<std::size_t>(local)];
    for (int slot = 0; slot < neighbor_count; ++slot) {
      const int neighbor =
          rank.neighbors[static_cast<std::size_t>(local) +
                         static_cast<std::size_t>(rank.nlocal) * slot];
      const Atom& other = rank.atoms[static_cast<std::size_t>(neighbor)];
      const std::array<double, 3> dr{
          other.position[0] - center.position[0],
          other.position[1] - center.position[1],
          other.position[2] - center.position[2],
      };
      const double r =
          std::sqrt(dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]);
      if (!(r > 0.0)) {
        continue;
      }

      const double delta = r - r0;
      const double directed_energy = 0.25 * k * delta * delta;
      const double force_scale = 0.5 * k * delta / r;
      const std::array<double, 3> force{
          force_scale * dr[0],
          force_scale * dr[1],
          force_scale * dr[2],
      };

      result.energy[static_cast<std::size_t>(local)] += directed_energy;
      for (int dim = 0; dim < 3; ++dim) {
        result.force_aos3[static_cast<std::size_t>(3 * local + dim)] +=
            force[static_cast<std::size_t>(dim)];
        result.force_aos3[static_cast<std::size_t>(3 * neighbor + dim)] -=
            force[static_cast<std::size_t>(dim)];
      }
      add_outer_product(result.virial_raw9, dr, force);
    }
  }

  return result;
}

Result fold_rank_results(
    int global_atom_count,
    const std::vector<RankCase>& ranks,
    const std::vector<Result>& rank_results) {
  Result folded;
  folded.energy.assign(static_cast<std::size_t>(global_atom_count), 0.0);
  folded.force_aos3.assign(static_cast<std::size_t>(global_atom_count) * 3, 0.0);

  for (std::size_t rank_index = 0; rank_index < ranks.size(); ++rank_index) {
    const RankCase& rank = ranks[rank_index];
    const Result& result = rank_results[rank_index];
    for (int local = 0; local < rank.nlocal; ++local) {
      const int global = rank.global_index[static_cast<std::size_t>(local)];
      folded.energy[static_cast<std::size_t>(global)] +=
          result.energy[static_cast<std::size_t>(local)];
    }

    for (std::size_t slot = 0; slot < rank.global_index.size(); ++slot) {
      const int global = rank.global_index[slot];
      for (int dim = 0; dim < 3; ++dim) {
        folded.force_aos3[static_cast<std::size_t>(3 * global + dim)] +=
            result.force_aos3[3 * slot + static_cast<std::size_t>(dim)];
      }
    }

    for (int component = 0; component < 9; ++component) {
      folded.virial_raw9[static_cast<std::size_t>(component)] +=
          result.virial_raw9[static_cast<std::size_t>(component)];
    }
  }

  return folded;
}

double max_abs_diff(const std::vector<double>& lhs, const std::vector<double>& rhs) {
  if (lhs.size() != rhs.size()) {
    return INFINITY;
  }
  double diff = 0.0;
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    diff = std::max(diff, std::abs(lhs[index] - rhs[index]));
  }
  return diff;
}

double max_abs_diff(
    const std::array<double, 9>& lhs,
    const std::array<double, 9>& rhs) {
  double diff = 0.0;
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    diff = std::max(diff, std::abs(lhs[index] - rhs[index]));
  }
  return diff;
}

double norm(const std::vector<double>& values) {
  double sum = 0.0;
  for (double value : values) {
    sum += value * value;
  }
  return std::sqrt(sum);
}

}  // namespace

int main() {
  const std::vector<Atom> atoms = make_atoms();
  const RankCase full_rank = make_rank_case(atoms, {0, 1, 2, 3});
  const Result full = run_center_model(full_rank);

  const std::vector<RankCase> ranks = {
      make_rank_case(atoms, {0, 1}),
      make_rank_case(atoms, {2, 3}),
  };
  std::vector<Result> rank_results;
  rank_results.reserve(ranks.size());
  for (const RankCase& rank : ranks) {
    rank_results.push_back(run_center_model(rank));
  }

  const Result folded =
      fold_rank_results(static_cast<int>(atoms.size()), ranks, rank_results);

  constexpr double tolerance = 1.0e-12;
  const double energy_diff = max_abs_diff(full.energy, folded.energy);
  const double force_diff = max_abs_diff(full.force_aos3, folded.force_aos3);
  const double virial_diff = max_abs_diff(full.virial_raw9, folded.virial_raw9);

  if (norm(full.force_aos3) <= tolerance || energy_diff > tolerance ||
      force_diff > tolerance || virial_diff > tolerance) {
    std::cerr << "domain decomposition contract failed: energy_diff="
              << energy_diff << " force_diff=" << force_diff
              << " virial_diff=" << virial_diff
              << " tolerance=" << tolerance << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
