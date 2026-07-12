"""Focused unit tests for the RTX 4090 spin profiling harness."""

import importlib.util
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "run_4090_spin_lmp_profile.py"
SPEC = importlib.util.spec_from_file_location("spin_profile", MODULE_PATH)
profile = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(profile)


class SpinLmpProfileTest(unittest.TestCase):

  BENCHMARK_OPTIONS = {
      "cubic": 100,
      "spacing": 3.0,
      "warmup": 1,
      "iterations": 3,
      "repeats": 3,
      "layout": "nolegacy",
      "mode": "api",
      "strip_spin_model": False,
  }
  LEGACY_NCU_METRICS = (
      "gpu__time_duration.sum",
      "launch__registers_per_thread",
      "sm__warps_active.avg.pct_of_peak_sustained_active",
      "smsp__warps_eligible.avg.per_cycle_active",
      "smsp__issue_active.avg.pct_of_peak_sustained_active",
      "dram__throughput.avg.pct_of_peak_sustained_elapsed",
      "sm__throughput.avg.pct_of_peak_sustained_elapsed",
      "l1tex__t_sector_hit_rate.pct",
      "lts__t_sector_hit_rate.pct",
  )
  NCU_WORKLOAD_OPTIONS = {
      "cubic": 50,
      "spacing": 3.0,
      "warmup": 1,
      "iterations": 1,
      "layout": "nolegacy",
      "mode": "api",
      "strip_spin_model": False,
      "detail": False,
      "selected_metrics": list(LEGACY_NCU_METRICS),
  }

  def benchmark_evidence(self, latency=10.0, throughput=2.0):
    return {"spin": {"summary": {
        "nolegacy_ms": {"median": latency},
        "nolegacy_matom_per_s": {"median": throughput},
    }}}

  def ncu_evidence(self, duration="100", registers="64", labels=None):
    labels = labels or ("primitive", "chiral", "density")
    return {
        label: {"metrics": [
            {"metric": "gpu__time_duration.sum", "value": duration},
            {"metric": "launch__registers_per_thread", "value": registers},
        ]}
        for label in labels
    }

  def test_detailed_csv_metrics_supports_real_long_form_ncu_raw_csv(self):
    csv_text = "\n".join([
        '==PROF== Connected',
        '"ID","Process ID","Process Name","Host Name","Kernel Name","Context",'
        '"Stream","Section Name","Metric Name","Metric Unit","Metric Value"',
        '"1","10","bench","host","void target_kernel()","1","7",'
        '"GPU Speed Of Light Throughput","gpu__time_duration.sum","nsecond","1000"',
        '"1","10","bench","host","void target_kernel()","1","7",'
        '"Launch Statistics","launch__registers_per_thread","register/thread","64"',
        '"1","10","bench","host","void target_kernel()","1","7",'
        '"Scheduler Statistics","sm__warps_active.avg.pct_of_peak_sustained_active",'
        '"%","75.0"',
        '"1","10","bench","host","void target_kernel()","1","7",'
        '"Memory Workload Analysis","dram__throughput.avg.pct_of_peak_sustained_elapsed",'
        '"%","22.5"',
    ])

    metrics = profile.detailed_ncu_metrics(csv_text)

    self.assertEqual(metrics[0], {"metric": "Kernel Name", "value": "void target_kernel()"})
    self.assertEqual(
        {item["metric"] for item in metrics[1:]},
        {
            "gpu__time_duration.sum",
            "launch__registers_per_thread",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        })

  def test_detailed_csv_metrics_keeps_wide_explicit_metrics_compatibility(self):
    csv_text = "\n".join([
        '"ID","Kernel Name","gpu__time_duration.sum",'
        '"launch__registers_per_thread","unused"',
        '"1","void target_kernel()","1000","64","discard"',
    ])

    metrics = profile.detailed_ncu_metrics(csv_text)

    self.assertEqual(metrics, [
        {"metric": "Kernel Name", "value": "void target_kernel()"},
        {"metric": "gpu__time_duration.sum", "value": "1000"},
        {"metric": "launch__registers_per_thread", "value": "64"},
    ])

  def test_profile_release_cuda_flags_preserve_optimized_lineinfo_build(self):
    flags = profile.profile_release_cuda_flags()

    self.assertIn("-O3", flags)
    self.assertIn("-DNDEBUG", flags)
    self.assertIn("-lineinfo", flags)

  def test_source_provenance_records_reproducible_shape(self):
    binary = {
        "path": "/remote/build/benchmarks/nep_adapters_bench_cuda_lammps_device_layout",
        "sha256": "binary-sha",
        "size_bytes": 1234,
        "mtime_ns": 5678,
    }
    provenance = profile.source_provenance(
        "local-sha",
        "remote-sha",
        "local-commit",
        "main",
        True,
        binary,
        True,
        "-O3 -DNDEBUG",
        "present")

    self.assertEqual(
        provenance,
        {
            "local_source_sha256": "local-sha",
            "remote_source_sha256": "remote-sha",
            "git_commit": "local-commit",
            "git_branch": "main",
            "git_dirty": True,
            "desired_cuda_release_flags": "-O3 -DNDEBUG -lineinfo",
            "configured_cuda_release_flags": "-O3 -DNDEBUG",
            "configured_cuda_release_flags_status": "present",
            "lineinfo": False,
            "build_performed": True,
            "remote_benchmark_executable": binary,
        })

  def test_source_provenance_marks_missing_cuda_release_flags_unknown(self):
    provenance = profile.source_provenance(
        "local-sha",
        "remote-sha",
        "local-commit",
        "main",
        False,
        {"path": "/remote/bench", "sha256": "binary", "size_bytes": 1, "mtime_ns": 2},
        False,
        None,
        "missing_cache")

    self.assertEqual(provenance["desired_cuda_release_flags"], "-O3 -DNDEBUG -lineinfo")
    self.assertIsNone(provenance["configured_cuda_release_flags"])
    self.assertEqual(provenance["configured_cuda_release_flags_status"], "missing_cache")
    self.assertIsNone(provenance["lineinfo"])

  def test_source_inputs_cover_all_benchmark_build_roots(self):
    self.assertEqual(set(profile.SOURCE_INPUTS), {
        "CMakeLists.txt",
        "cmake",
        "include",
        "engines/cpu_common",
        "engines/cpu_opt",
        "engines/cuda",
        "frontends/lammps",
        "src",
        "benchmarks",
        "tools/run_4090_spin_lmp_profile.py",
        profile.MODEL_REL,
    })

  def test_comparison_reports_only_comparable_benchmark_and_ncu_deltas(self):
    baseline = {
        "benchmarks": self.benchmark_evidence(),
        "benchmark_options": self.BENCHMARK_OPTIONS,
        "ncu": self.ncu_evidence(),
        "ncu_workload_options": self.NCU_WORKLOAD_OPTIONS,
    }
    current = {
        "benchmarks": self.benchmark_evidence(latency=8.0, throughput=2.5),
        "benchmark_options": self.BENCHMARK_OPTIONS,
        "ncu": self.ncu_evidence(duration="80"),
        "ncu_workload_options": self.NCU_WORKLOAD_OPTIONS,
    }

    comparison = profile.compare_with_baseline(current, baseline)

    self.assertEqual(comparison["benchmark"]["status"], "comparable")
    self.assertEqual(comparison["benchmark"]["deltas"]["spin_median_latency_ms"], {
        "baseline": 10.0, "current": 8.0, "delta": -2.0, "delta_percent": -20.0})
    self.assertEqual(comparison["benchmark"]["deltas"]["spin_median_throughput_matom_per_s"], {
        "baseline": 2.0, "current": 2.5, "delta": 0.5, "delta_percent": 25.0})
    self.assertEqual(comparison["ncu"]["status"], "comparable")
    self.assertEqual(comparison["ncu"]["deltas"]["primitive"]["gpu__time_duration.sum"], {
        "baseline": 100.0, "current": 80.0, "delta": -20.0, "delta_percent": -20.0})

  def test_comparison_marks_benchmark_option_mismatch_non_comparable(self):
    baseline = {
        "benchmarks": self.benchmark_evidence(),
        "benchmark_options": self.BENCHMARK_OPTIONS,
    }
    current = {
        "benchmarks": self.benchmark_evidence(latency=8.0, throughput=2.5),
        "benchmark_options": {**self.BENCHMARK_OPTIONS, "cubic": 20},
    }

    comparison = profile.compare_with_baseline(current, baseline)

    self.assertEqual(comparison["benchmark"], {
        "status": "non_comparable",
        "reason": "benchmark command options differ",
        "current_options": {**self.BENCHMARK_OPTIONS, "cubic": 20},
        "baseline_options": self.BENCHMARK_OPTIONS,
    })

  def test_comparison_marks_missing_benchmark_evidence_non_comparable(self):
    comparison = profile.compare_with_baseline(
        {"benchmark_options": self.BENCHMARK_OPTIONS},
        {
            "benchmarks": self.benchmark_evidence(),
            "benchmark_options": self.BENCHMARK_OPTIONS,
        })

    self.assertEqual(comparison["benchmark"], {
        "status": "non_comparable",
        "reason": "current summary is missing benchmark evidence",
        "current_options": self.BENCHMARK_OPTIONS,
        "baseline_options": self.BENCHMARK_OPTIONS,
    })

  def test_comparison_marks_missing_benchmark_options_non_comparable(self):
    comparison = profile.compare_with_baseline(
        {"benchmarks": self.benchmark_evidence()},
        {
            "benchmarks": self.benchmark_evidence(),
            "benchmark_options": self.BENCHMARK_OPTIONS,
        })

    self.assertEqual(comparison["benchmark"]["status"], "non_comparable")
    self.assertEqual(
        comparison["benchmark"]["reason"],
        "current summary is missing complete benchmark_options")

  def test_comparison_marks_one_sided_ncu_evidence_non_comparable(self):
    comparison = profile.compare_with_baseline(
        {
            "ncu": self.ncu_evidence(duration="80"),
            "ncu_workload_options": self.NCU_WORKLOAD_OPTIONS,
        },
        {"ncu_workload_options": self.NCU_WORKLOAD_OPTIONS})

    self.assertEqual(comparison["ncu"], {
        "status": "non_comparable",
        "reason": "baseline summary is missing NCU evidence",
        "current_options": self.NCU_WORKLOAD_OPTIONS,
        "baseline_options": self.NCU_WORKLOAD_OPTIONS,
    })

  def test_comparison_marks_detail_vs_legacy_ncu_non_comparable(self):
    detailed_options = {
        **self.NCU_WORKLOAD_OPTIONS,
        "detail": True,
        "selected_metrics": list(profile.NCU_SELECTED_METRICS),
    }
    comparison = profile.compare_with_baseline(
        {"ncu": self.ncu_evidence(duration="80"), "ncu_workload_options": detailed_options},
        {"ncu": self.ncu_evidence(), "ncu_workload_options": self.NCU_WORKLOAD_OPTIONS})

    self.assertEqual(comparison["ncu"]["status"], "non_comparable")
    self.assertEqual(
        comparison["ncu"]["reason"],
        "NCU workload or collection options differ")

  def test_comparison_marks_ncu_label_mismatch_non_comparable(self):
    comparison = profile.compare_with_baseline(
        {
            "ncu": self.ncu_evidence(labels=("primitive", "chiral")),
            "ncu_workload_options": self.NCU_WORKLOAD_OPTIONS,
        },
        {"ncu": self.ncu_evidence(), "ncu_workload_options": self.NCU_WORKLOAD_OPTIONS})

    self.assertEqual(comparison["ncu"]["status"], "non_comparable")
    self.assertEqual(comparison["ncu"]["reason"], "NCU labels do not match expected set")

  def test_comparison_marks_ncu_metric_set_mismatch_non_comparable(self):
    current_ncu = self.ncu_evidence(duration="80")
    current_ncu["primitive"]["metrics"] = [
        {"metric": "gpu__time_duration.sum", "value": "80"},
    ]
    comparison = profile.compare_with_baseline(
        {"ncu": current_ncu, "ncu_workload_options": self.NCU_WORKLOAD_OPTIONS},
        {"ncu": self.ncu_evidence(), "ncu_workload_options": self.NCU_WORKLOAD_OPTIONS})

    self.assertEqual(comparison["ncu"]["status"], "non_comparable")
    self.assertEqual(
        comparison["ncu"]["reason"],
        "NCU numeric metric names differ for label: primitive")

  def test_ncu_workload_options_record_collection_shape(self):
    args = mock.MagicMock(ncu_cubic=50, spacing=3.0, ncu_detail=False)

    self.assertEqual(profile.ncu_workload_options(args), self.NCU_WORKLOAD_OPTIONS)

  def test_source_hash_is_stable_and_excludes_generated_cache_and_temp_files(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      (root / "CMakeLists.txt").write_text("project(hash_fixture)\n", encoding="utf-8")
      source_dir = root / "src"
      source_dir.mkdir()
      (source_dir / "kernel.cu").write_text("__global__ void k() {}\n", encoding="utf-8")
      (source_dir / "kernel.pyc").write_bytes(b"generated")
      (source_dir / "__pycache__").mkdir()
      (source_dir / "__pycache__" / "tool.pyc").write_bytes(b"cache")
      (source_dir / "tmp").mkdir()
      (source_dir / "tmp" / "scratch.cpp").write_text("stale\n", encoding="utf-8")

      inputs = ("CMakeLists.txt", "src")
      first = profile.source_tree_sha256(root, inputs)
      (source_dir / "kernel.pyc").write_bytes(b"different generated output")
      (source_dir / "tmp" / "scratch.cpp").write_text("different stale output\n", encoding="utf-8")
      second = profile.source_tree_sha256(root, inputs)

      self.assertEqual(first, second)
      (source_dir / "kernel.cu").write_text("__global__ void k() { return; }\n", encoding="utf-8")
      self.assertNotEqual(second, profile.source_tree_sha256(root, inputs))

  def test_source_hash_rejects_missing_declared_roots(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      (root / "CMakeLists.txt").write_text("project(hash_fixture)\n", encoding="utf-8")

      with self.assertRaisesRegex(FileNotFoundError, "missing declared source root: src"):
        profile.source_tree_sha256(root, ("CMakeLists.txt", "src"))

  def test_source_hash_includes_cmake_configure_templates(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      cmake_dir = root / "cmake"
      cmake_dir.mkdir()
      template = cmake_dir / "NEPAdaptersConfig.cmake.in"
      template.write_text("set(NEP_ADAPTERS_VERSION @VERSION@)\n", encoding="utf-8")

      first = profile.source_tree_sha256(root, ("cmake",))
      template.write_text("set(NEP_ADAPTERS_VERSION @NEXT_VERSION@)\n", encoding="utf-8")

      self.assertNotEqual(first, profile.source_tree_sha256(root, ("cmake",)))

  def test_detailed_ncu_command_cleans_and_requires_current_text_artifacts(self):
    args = mock.MagicMock(cubic=50, spacing=3.0, warmup=1, iterations=1)
    plan = {
        "source_dir": "/remote/source",
        "build_dir": "/remote/build",
    }

    paths = profile.detailed_ncu_artifact_paths("/remote/runs/run", "primitive")
    script = profile.build_detailed_ncu_script(
        plan,
        args,
        "/remote/runs/run",
        "primitive",
        "build_spin_primitive_cache_c4_l4_warp")

    self.assertEqual(paths["report"], "/remote/runs/run/ncu_primitive_detailed.ncu-rep")
    self.assertNotIn("report", profile.detailed_ncu_text_artifact_names())
    self.assertEqual(
        set(profile.detailed_ncu_text_artifact_names()),
        {"raw_csv", "cuda_source", "sass_source", "binary_sass", "resource_usage"})
    self.assertIn("--set detailed", script)
    self.assertIn("--import-source yes", script)
    self.assertIn("--metrics " + profile.q(",".join(profile.NCU_SELECTED_METRICS)), script)
    self.assertIn("--kernel-name regex:build_spin_primitive_cache_c4_l4_warp", script)
    self.assertIn("--launch-count 1", script)
    self.assertIn("find \"$artifact_dir\" -maxdepth 1 -type f", script)
    self.assertIn("-name \"${artifact_stem}.*\" -delete", script)
    self.assertIn("if [ ! -s", script)
    self.assertNotIn("|| true", next(
        line for line in script.splitlines() if "ncu --target-processes all" in line))

  def test_detailed_ncu_export_failures_do_not_become_nonempty_artifacts(self):
    args = mock.MagicMock(cubic=50, spacing=3.0, warmup=1, iterations=1)
    plan = {
        "source_dir": "/remote/source",
        "build_dir": "/remote/build",
    }

    script = profile.build_detailed_ncu_script(
        plan,
        args,
        "/remote/runs/run",
        "primitive",
        "build_spin_primitive_cache_c4_l4_warp")

    self.assertIn("set -euo pipefail", script)
    self.assertNotIn("ncu_primitive_detailed.cuda.txt 2>&1", script)
    self.assertNotIn("ncu_primitive_detailed.sass.txt 2>&1", script)
    self.assertNotIn("ncu_primitive_detailed.resources.txt 2>&1", script)
    self.assertNotIn("cuobjdump --dump-sass /remote/build/" + profile.BENCH_REL +
                     " 2>/dev/null", script)
    self.assertIn("--print-source cuda", script)
    self.assertIn("> /remote/runs/run/ncu_primitive_detailed.cuda.txt", script)
    self.assertIn("--print-source sass", script)
    self.assertIn("> /remote/runs/run/ncu_primitive_detailed.sass.txt", script)
    self.assertIn("--dump-resource-usage /remote/build/" + profile.BENCH_REL, script)
    self.assertIn("> /remote/runs/run/ncu_primitive_detailed.resources.txt", script)

  def test_legacy_ncu_command_cleans_and_requires_a_current_csv(self):
    args = mock.MagicMock(cubic=50, spacing=3.0, warmup=1, iterations=1)
    plan = {
        "source_dir": "/remote/source",
        "build_dir": "/remote/build",
    }

    script = profile.build_legacy_ncu_script(
        plan,
        args,
        "/remote/runs/run",
        "primitive",
        "build_spin_primitive_cache_c4_l4_warp")

    self.assertIn("rm -f -- /remote/runs/run/ncu_primitive.csv", script)
    self.assertIn("if [ ! -s /remote/runs/run/ncu_primitive.csv ]; then", script)
    self.assertNotIn("|| true", next(
        line for line in script.splitlines() if "ncu --target-processes all" in line))

  def test_cleanup_local_run_artifacts_is_narrow(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      run_dir = root / "diagnostics" / "spin_lmp_4090" / "repeat"
      run_dir.mkdir(parents=True)
      generated = [
          run_dir / "summary.json",
          run_dir / "ncu_primitive_detailed.csv",
          run_dir / "ncu_density_detailed.resources.txt",
      ]
      unrelated = [
          run_dir / "notes.txt",
          run_dir / "ncu_notes.txt",
          run_dir / "ncu_primitive_notes.txt",
      ]
      for path in generated + unrelated:
        path.write_text(path.name, encoding="utf-8")

      profile.cleanup_local_run_artifacts("repeat", root=root)

      self.assertTrue(run_dir.is_dir())
      self.assertTrue(all(not path.exists() for path in generated))
      self.assertTrue(all(path.exists() for path in unrelated))

  def test_local_run_artifacts_reject_path_escaping_run_names(self):
    with self.assertRaisesRegex(ValueError, "single path component"):
      profile.cleanup_local_run_artifacts("../escape")
    with self.assertRaisesRegex(ValueError, "single path component"):
      profile.write_summary({}, "../escape")


if __name__ == "__main__":
  unittest.main()
