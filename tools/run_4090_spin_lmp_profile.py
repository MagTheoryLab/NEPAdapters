#!/usr/bin/env python3
"""Run the spin LAMMPS GPU benchmark/profiling loop on the 4090 host."""

import argparse
import csv
import hashlib
import json
import re
import shlex
import statistics
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HPC_SKILL = Path("/Users/superbing/Desktop/Workspace/my-skills/skills/hpc-platform")
DEFAULT_PLAN = Path(
    "/Users/superbing/Desktop/Workspace/.hpc-platform/plans/"
    "4090-nepadapters-spin-lmp.json"
)
MODEL_REL = "tests/fixtures/spin_chiral_protocol/nep.txt"
BENCH_REL = "benchmarks/nep_adapters_bench_cuda_lammps_device_layout"
PROFILE_CUDA_RELEASE_FLAGS = "-O3 -DNDEBUG -lineinfo"
SOURCE_INPUTS = (
    "CMakeLists.txt",
    "cmake",
    "include",
    "engines/cpu_common",
    "engines/cpu_opt",
    "benchmarks",
    "engines/cuda",
    "frontends/lammps",
    "src",
    "tools/run_4090_spin_lmp_profile.py",
    MODEL_REL,
)
NCU_LABELS = ("primitive", "chiral", "density")
NCU_SELECTED_METRICS = (
    "gpu__time_duration.sum",
    "launch__registers_per_thread",
    "launch__block_size",
    "launch__grid_size",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
)
LEGACY_NCU_SELECTED_METRICS = (
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
BENCHMARK_OPTION_KEYS = frozenset({
    "cubic", "spacing", "warmup", "iterations", "repeats", "layout", "mode",
    "strip_spin_model",
})
NCU_OPTION_KEYS = frozenset({
    "cubic", "spacing", "warmup", "iterations", "layout", "mode",
    "strip_spin_model", "detail", "selected_metrics",
})
SOURCE_ALLOWED_SUFFIXES = frozenset({
    ".c", ".cc", ".cpp", ".cuh", ".cu", ".cxx", ".h", ".hh", ".hpp",
    ".cmake", ".in", ".inl", ".py", ".txt",
})
SOURCE_EXCLUDED_PARTS = frozenset({
    "__pycache__", ".mypy_cache", ".pytest_cache", "CMakeFiles", "cache", "temp",
    "tmp",
})
SOURCE_EXCLUDED_FILE_NAMES = frozenset({"CMakeCache.txt"})
SOURCE_EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo", ".swp", ".tmp"})


def run_local(args, *, cwd=ROOT, input_text=None, capture=True):
  print("+", " ".join(str(arg) for arg in args), file=sys.stderr)
  return subprocess.run(
      [str(arg) for arg in args],
      cwd=cwd,
      input=input_text,
      text=True,
      capture_output=capture,
      check=True)


def ssh_bash(profile, script, *, capture=True):
  return run_local(["ssh", profile, "bash", "-s"], input_text=script, capture=capture)


def q(value):
  return shlex.quote(str(value))


def profile_release_cuda_flags():
  return PROFILE_CUDA_RELEASE_FLAGS


def load_plan(plan_path):
  with open(plan_path, "r", encoding="utf-8") as handle:
    plan = json.load(handle)
  required = ["source_dir", "build_dir", "run_dir"]
  missing = [key for key in required if not plan.get(key)]
  if missing:
    raise SystemExit(f"plan is missing required key(s): {', '.join(missing)}")
  return plan


def sync_workspace(plan_path):
  sync_script = HPC_SKILL / "scripts" / "sync_workspace.py"
  run_local([
      sys.executable,
      sync_script,
      "--plan",
      plan_path,
      "--exclude",
      ".build/***",
      "--exclude",
      "tmp/***",
      "--exclude",
      "diagnostics/***",
      "--exclude",
      "outputs/***",
      "--exclude",
      "runs/***",
      "--exclude",
      "tests_tmp*/***",
      "--allow-full-sync",
      "--reason",
      "benchmark script excludes generated artifacts explicitly",
  ], capture=False)


def remote_env(gpu):
  return textwrap.dedent(f"""
      set -euo pipefail
      export PATH=/home/dwhe/opt/cuda-12.8/bin:/home/dwhe/.local/bin:$PATH
      export LD_LIBRARY_PATH=/home/dwhe/opt/cuda-12.8/lib64:${{LD_LIBRARY_PATH:-}}
      export CUDA_VISIBLE_DEVICES={q(gpu)}
      """)


def build_remote(profile, plan, gpu):
  source_dir = plan["source_dir"]
  build_dir = plan["build_dir"]
  script = remote_env(gpu) + textwrap.dedent(f"""
      mkdir -p {q(build_dir)}
      cd {q(source_dir)}
      cmake -S {q(source_dir)} -B {q(build_dir)} \
        -DCMAKE_BUILD_TYPE=Release \
        -DNEP_ADAPTERS_ENABLE_CPU_NEP3=OFF \
        -DNEP_ADAPTERS_ENABLE_CUDA=ON \
        -DNEP_ADAPTERS_CUDA_ENABLE_DEVICE_RUNTIME=ON \
        -DNEP_ADAPTERS_ENABLE_CPU_OPT=ON \
        -DNEP_ADAPTERS_BUILD_TESTS=ON \
        -DNEP_ADAPTERS_BUILD_BENCHMARKS=ON \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DCMAKE_CUDA_FLAGS_RELEASE={q(profile_release_cuda_flags())}
      cmake --build {q(build_dir)} --clean-first --parallel 16 --target \
        nep_adapters_bench_spin_gpu \
        nep_adapters_bench_cuda_lammps_device_layout \
        nep_adapters_cuda_spin_fixture_test \
        nep_adapters_cuda_spin_chiral_polar_test \
        nep_adapters_cuda_lammps_kk_sim_test
      """)
  ssh_bash(profile, script, capture=False)


def run_correctness(profile, plan, gpu):
  script = remote_env(gpu) + textwrap.dedent(f"""
      cd {q(plan["build_dir"])}
      ctest --output-on-failure -R 'cuda_spin_fixture|cuda_spin_chiral_polar|cuda_lammps_kk_sim'
      """)
  return ssh_bash(profile, script).stdout


def bench_command(plan, args, strip_spin):
  command = [
      f"{plan['build_dir']}/{BENCH_REL}",
      "--model",
      f"{plan['source_dir']}/{MODEL_REL}",
      "--cubic",
      str(args.cubic),
      "--spacing",
      str(args.spacing),
      "--layout",
      "nolegacy",
      "--mode",
      "api",
      "--warmup",
      str(args.warmup),
      "--iterations",
      str(args.iterations),
  ]
  if strip_spin:
    command.append("--strip-spin-model")
  return command


def parse_benchmark(stdout):
  metrics = {}
  for key, value in re.findall(r"([A-Za-z0-9_]+)=([-+0-9.eE]+)", stdout):
    try:
      metrics[key] = float(value)
    except ValueError:
      pass
  return metrics


def parse_pair_profile(stderr):
  samples = []
  for line in stderr.splitlines():
    if "NEPA_PAIR_PROFILE" not in line:
      continue
    samples.append(parse_benchmark(line))
  return samples


def run_benchmark_case(profile, plan, gpu, args, strip_spin, repeats, profile_pair=0):
  command = bench_command(plan, args, strip_spin)
  env_line = ""
  if profile_pair > 0:
    env_line = f"export NEP_ADAPTERS_PROFILE_PAIR={profile_pair}\n"
  script = remote_env(gpu) + env_line + textwrap.dedent(f"""
      cd {q(plan["source_dir"])}
      for i in $(seq 1 {int(repeats)}); do
        {' '.join(q(part) for part in command)}
      done
      """)
  completed = ssh_bash(profile, script)
  stdout = completed.stdout
  stderr = completed.stderr
  runs = []
  for block in re.split(r"(?m)(?=^atoms=)", stdout):
    metrics = parse_benchmark(block)
    if "nolegacy_ms" in metrics:
      runs.append(metrics)
  if not runs:
    metrics = parse_benchmark(stdout)
    if metrics:
      runs.append(metrics)
  pair_profile = parse_pair_profile(stderr)
  steady_pair_profile = [
      sample for sample in pair_profile if sample.get("rebuild", 0.0) == 0.0
  ]
  return {
      "command": command,
      "stdout": stdout,
      "stderr": stderr,
      "runs": runs,
      "pair_profile": pair_profile,
      "pair_profile_summary": summarize_runs(pair_profile),
      "pair_profile_steady_summary": summarize_runs(steady_pair_profile),
  }


def last_pair_sample(pair_profile):
  if not pair_profile:
    return {}
  return pair_profile[-1]


def summarize_runs(runs):
  if not runs:
    return {}
  result = {}
  for key in sorted({key for run in runs for key in run}):
    values = [run[key] for run in runs if key in run]
    if values:
      result[key] = {
          "median": statistics.median(values),
          "min": min(values),
          "max": max(values),
      }
  return result


def run_nsys(profile, plan, gpu, args, run_dir):
  prefix = f"{run_dir}/nsys_spin_lmp_1m"
  command = bench_command(plan, args, False)
  script = remote_env(gpu) + textwrap.dedent(f"""
      if ! command -v nsys >/dev/null 2>&1; then
        echo '{{"skipped":"nsys not found"}}'
        exit 0
      fi
      mkdir -p {q(run_dir)}
      cd {q(plan["source_dir"])}
      nsys profile --trace=cuda --sample=none --cpuctxsw=none \
        --force-overwrite true -o {q(prefix)} \
        {' '.join(q(part) for part in command)} \
        > {q(prefix + ".log")} 2>&1
      nsys stats --report cuda_gpu_kern_sum {q(prefix + ".nsys-rep")} \
        > {q(prefix + ".kernels.txt")} 2>&1 || true
      python3 - <<'PY'
import json
from pathlib import Path
kernels = Path({str(prefix + ".kernels.txt")!r})
print(json.dumps({{
  "report": {str(prefix + ".nsys-rep")!r},
  "log": {str(prefix + ".log")!r},
  "kernels": str(kernels),
  "kernels_head": "\\n".join(kernels.read_text().splitlines()[:80]) if kernels.exists() else ""
}}))
PY
      """)
  return json.loads(ssh_bash(profile, script).stdout.strip().splitlines()[-1])


def detailed_ncu_metrics(text):
  lines = text.splitlines()
  header_index = None
  for index, line in enumerate(lines):
    if line.startswith('"ID",') or line.startswith("ID,"):
      header_index = index
      break
  if header_index is None:
    return []
  reader = csv.DictReader(lines[header_index:])
  parsed_rows = list(reader)
  if not parsed_rows:
    return []
  if "Metric Name" in reader.fieldnames and "Metric Value" in reader.fieldnames:
    kernel = next((row.get("Kernel Name", "") for row in parsed_rows
                   if row.get("Kernel Name", "")), "")
    selected_values = {}
    for row in parsed_rows:
      metric = row.get("Metric Name", "")
      if metric in NCU_SELECTED_METRICS and metric not in selected_values:
        selected_values[metric] = row.get("Metric Value", "")
    if not kernel:
      return []
    return ([{"metric": "Kernel Name", "value": kernel}] + [
        {"metric": metric, "value": selected_values[metric]}
        for metric in NCU_SELECTED_METRICS if metric in selected_values
    ])
  for row in parsed_rows:
    kernel = row.get("Kernel Name", "")
    if kernel:
      return ([{"metric": "Kernel Name", "value": kernel}] + [
          {"metric": metric, "value": row[metric]}
          for metric in NCU_SELECTED_METRICS if metric in row
      ])
  return []


def parse_ncu_csv_text(text):
  return detailed_ncu_metrics(text)


def is_allowed_source_file(relative_path):
  if any(part in SOURCE_EXCLUDED_PARTS for part in relative_path.parts[:-1]):
    return False
  if relative_path.name in SOURCE_EXCLUDED_FILE_NAMES:
    return False
  if relative_path.suffix.lower() in SOURCE_EXCLUDED_SUFFIXES:
    return False
  return relative_path.suffix.lower() in SOURCE_ALLOWED_SUFFIXES


def source_tree_sha256(root, source_inputs=SOURCE_INPUTS):
  root = Path(root)
  digest = hashlib.sha256()
  for relative in source_inputs:
    path = root / relative
    if not path.exists():
      raise FileNotFoundError(f"missing declared source root: {relative}")
    paths = [path] if path.is_file() else sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix())
    for item in paths:
      source_relative = item.relative_to(root).as_posix()
      if not is_allowed_source_file(Path(source_relative)):
        continue
      digest.update(source_relative.encode("utf-8"))
      digest.update(b"\0")
      digest.update(item.read_bytes())
  return digest.hexdigest()


def git_value(*args):
  return run_local(["git", *args]).stdout.strip()


def source_provenance(local_source_sha256, remote_source_sha256, git_commit,
                      git_branch, git_dirty, remote_benchmark_executable,
                      build_performed, configured_cuda_release_flags,
                      configured_cuda_release_flags_status):
  return {
      "local_source_sha256": local_source_sha256,
      "remote_source_sha256": remote_source_sha256,
      "git_commit": git_commit,
      "git_branch": git_branch,
      "git_dirty": git_dirty,
      "desired_cuda_release_flags": profile_release_cuda_flags(),
      "configured_cuda_release_flags": configured_cuda_release_flags,
      "configured_cuda_release_flags_status": configured_cuda_release_flags_status,
      "lineinfo": (
          "-lineinfo" in configured_cuda_release_flags
          if configured_cuda_release_flags is not None else None),
      "build_performed": build_performed,
      "remote_benchmark_executable": remote_benchmark_executable,
  }


def collect_source_provenance(profile, plan, gpu, build_performed):
  local_source_sha256 = source_tree_sha256(ROOT)
  git_commit = git_value("rev-parse", "HEAD")
  git_branch = git_value("branch", "--show-current")
  git_dirty = bool(git_value("status", "--porcelain"))
  script = remote_env(gpu) + textwrap.dedent(f"""
      cd {q(plan["source_dir"])}
      python3 - <<'PY'
import hashlib
import importlib.util
import json
from pathlib import Path

root = Path.cwd()
module_path = root / "tools/run_4090_spin_lmp_profile.py"
spec = importlib.util.spec_from_file_location("spin_profile_hash", module_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
benchmark = Path({str(Path(plan["build_dir"]) / BENCH_REL)!r})
if not benchmark.is_file():
  raise FileNotFoundError(f"remote benchmark executable is missing: {{benchmark}}")
digest = hashlib.sha256()
with benchmark.open("rb") as handle:
  for chunk in iter(lambda: handle.read(1024 * 1024), b""):
    digest.update(chunk)
stat = benchmark.stat()
cache = Path({str(Path(plan["build_dir"]) / "CMakeCache.txt")!r})
configured_cuda_release_flags = None
configured_cuda_release_flags_status = "missing_cache"
if cache.is_file():
  configured_cuda_release_flags_status = "missing_key"
  for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
    prefix = "CMAKE_CUDA_FLAGS_RELEASE:STRING="
    if line.startswith(prefix):
      configured_cuda_release_flags = line[len(prefix):]
      configured_cuda_release_flags_status = "present"
      break
print(json.dumps({{
    "source_sha256": module.source_tree_sha256(root),
    "remote_benchmark_executable": {{
        "path": str(benchmark),
        "sha256": digest.hexdigest(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }},
    "configured_cuda_release_flags": configured_cuda_release_flags,
    "configured_cuda_release_flags_status": configured_cuda_release_flags_status,
}}))
PY
      """)
  remote = json.loads(ssh_bash(profile, script).stdout.strip().splitlines()[-1])
  return source_provenance(
      local_source_sha256,
      remote["source_sha256"],
      git_commit,
      git_branch,
      git_dirty,
      remote["remote_benchmark_executable"],
      build_performed,
      remote["configured_cuda_release_flags"],
      remote["configured_cuda_release_flags_status"])


def build_legacy_ncu_script(plan, args, run_dir, label, kernel_regex, gpu="0"):
  csv_path = f"{run_dir}/ncu_{label}.csv"
  command = bench_command(plan, args, False)
  return remote_env(gpu) + textwrap.dedent(f"""
      mkdir -p {q(run_dir)}
      rm -f -- {q(csv_path)}
      if ! command -v ncu >/dev/null 2>&1; then
        echo '{{"skipped":"ncu not found"}}'
        exit 0
      fi
      cd {q(plan["source_dir"])}
      ncu --target-processes all --kernel-name {q("regex:" + kernel_regex)} \
        --launch-count 1 \
        --metrics {q(",".join(LEGACY_NCU_SELECTED_METRICS))} \
        --csv --page raw --log-file {q(csv_path)} \
        {' '.join(q(part) for part in command)} >/dev/null 2>&1
      if [ ! -s {q(csv_path)} ]; then
        echo "legacy NCU did not produce a current raw CSV" >&2
        exit 1
      fi
      python3 - <<PY
import json
from pathlib import Path
path = Path({str(csv_path)!r})
print(json.dumps({{"csv": str(path), "text": path.read_text() if path.exists() else ""}}))
PY
      """)


def run_ncu_one(profile, plan, gpu, args, run_dir, label, kernel_regex,
                detailed=False):
  if detailed:
    return run_ncu_detail_one(
        profile, plan, gpu, args, run_dir, label, kernel_regex)
  script = build_legacy_ncu_script(
      plan, args, run_dir, label, kernel_regex, gpu)
  payload = json.loads(ssh_bash(profile, script).stdout.strip().splitlines()[-1])
  text = payload.pop("text", "")
  payload["metrics"] = parse_ncu_csv_text(text)
  return payload


def detailed_ncu_artifact_paths(run_dir, label):
  prefix = f"{run_dir}/ncu_{label}_detailed"
  return {
      "report": prefix + ".ncu-rep",
      "raw_csv": prefix + ".csv",
      "cuda_source": prefix + ".cuda.txt",
      "sass_source": prefix + ".sass.txt",
      "binary_sass": prefix + ".binary_sass.txt",
      "resource_usage": prefix + ".resources.txt",
  }


def detailed_ncu_text_artifact_names():
  return (
      "raw_csv",
      "cuda_source",
      "sass_source",
      "binary_sass",
      "resource_usage",
  )


def benchmark_workload_options(args):
  return {
      "cubic": args.cubic,
      "spacing": args.spacing,
      "warmup": args.warmup,
      "iterations": args.iterations,
      "repeats": args.repeats,
      "layout": "nolegacy",
      "mode": "api",
      "strip_spin_model": False,
  }


def ncu_workload_options(args):
  detailed = bool(args.ncu_detail)
  return {
      "cubic": args.ncu_cubic,
      "spacing": args.spacing,
      "warmup": 1,
      "iterations": 1,
      "layout": "nolegacy",
      "mode": "api",
      "strip_spin_model": False,
      "detail": detailed,
      "selected_metrics": list(
          NCU_SELECTED_METRICS if detailed else LEGACY_NCU_SELECTED_METRICS),
  }


def build_detailed_ncu_script(plan, args, run_dir, label, kernel_regex, gpu="0"):
  paths = detailed_ncu_artifact_paths(run_dir, label)
  binary_path = f"{plan['build_dir']}/{BENCH_REL}"
  command = bench_command(plan, args, False)
  required_paths = [paths["report"], *[
      paths[name] for name in detailed_ncu_text_artifact_names()
  ]]
  return remote_env(gpu) + textwrap.dedent(f"""
      artifact_dir=$(dirname {q(paths["report"])})
      artifact_stem=$(basename {q(paths["report"][:-len(".ncu-rep")])})
      mkdir -p "$artifact_dir"
      find "$artifact_dir" -maxdepth 1 -type f -name "${{artifact_stem}}.*" -delete
      if ! command -v ncu >/dev/null 2>&1; then
        echo '{{"skipped":"ncu not found"}}'
        exit 0
      fi
      cd {q(plan["source_dir"])}
      ncu --target-processes all --kernel-name {q("regex:" + kernel_regex)} --launch-count 1 --set detailed --metrics {q(",".join(NCU_SELECTED_METRICS))} --import-source yes --export {q(paths["report"][:-len(".ncu-rep")])} --csv --page raw --log-file {q(paths["raw_csv"])} \
        {' '.join(q(part) for part in command)}
      if [ ! -s {q(paths["report"])} ] || [ ! -s {q(paths["raw_csv"])} ]; then
        echo "detailed NCU did not produce a current report and raw CSV" >&2
        exit 1
      fi
      ncu --import {q(paths["report"])} --page source --print-source cuda \
        > {q(paths["cuda_source"])}
      ncu --import {q(paths["report"])} --page source --print-source sass \
        > {q(paths["sass_source"])}
      if ! command -v cuobjdump >/dev/null 2>&1; then
        echo "cuobjdump is required for detailed NCU artifacts" >&2
        exit 1
      fi
      cuobjdump --dump-sass {q(binary_path)} | \
        awk -v pattern={q(kernel_regex)} \
        '/Function : / {{ keep = ($0 ~ pattern) }} keep {{ print }}' \
        > {q(paths["binary_sass"])}
      cuobjdump --dump-resource-usage {q(binary_path)} \
        > {q(paths["resource_usage"])}
      for artifact in {' '.join(q(path) for path in required_paths)}; do
        if [ ! -s "$artifact" ]; then
          echo "required detailed artifact is missing or empty: $artifact" >&2
          exit 1
        fi
      done
      python3 - <<'PY'
import json

paths = {{
    "report": {paths["report"]!r},
    "raw_csv": {paths["raw_csv"]!r},
    "cuda_source": {paths["cuda_source"]!r},
    "sass_source": {paths["sass_source"]!r},
    "binary_sass": {paths["binary_sass"]!r},
    "resource_usage": {paths["resource_usage"]!r},
}}
text_names = {detailed_ncu_text_artifact_names()!r}
with open(paths["raw_csv"], "r", encoding="utf-8") as handle:
  csv_text = handle.read()
print(json.dumps({{
    "report": paths["report"],
    "text_artifacts": {{name: paths[name] for name in text_names}},
    "csv_text": csv_text,
}}))
PY
      """)


def run_ncu_detail_one(profile, plan, gpu, args, run_dir, label, kernel_regex):
  script = build_detailed_ncu_script(
      plan, args, run_dir, label, kernel_regex, gpu)
  payload = json.loads(ssh_bash(profile, script).stdout.strip().splitlines()[-1])
  csv_text = payload.pop("csv_text", "")
  payload["metrics"] = detailed_ncu_metrics(csv_text)
  return payload


def run_ncu(profile, plan, gpu, args, run_dir):
  small = argparse.Namespace(**vars(args))
  small.cubic = args.ncu_cubic
  small.warmup = 1
  small.iterations = 1
  return {
      "primitive": run_ncu_one(
          profile,
          plan,
          gpu,
          small,
          run_dir,
          "primitive",
          "build_spin_descriptor_core_streaming",
          args.ncu_detail),
      "chiral": run_ncu_one(
          profile,
          plan,
          gpu,
          small,
          run_dir,
          "chiral",
          "accumulate_spin_chiral_forces",
          args.ncu_detail),
      "density": run_ncu_one(
          profile,
          plan,
          gpu,
          small,
          run_dir,
          "density",
          "accumulate_spin_density_forces_tile_f32",
          args.ncu_detail),
  }


def metric_values(metrics):
  values = {}
  for item in metrics:
    try:
      values[item["metric"]] = float(item["value"])
    except (KeyError, TypeError, ValueError):
      continue
  return values


def delta_record(current, baseline):
  delta = current - baseline
  return {
      "baseline": baseline,
      "current": current,
      "delta": delta,
      "delta_percent": 100.0 * delta / baseline if baseline else None,
  }


def spin_median(summary, metric):
  return summary.get("benchmarks", {}).get("spin", {}).get("summary", {}).get(
      metric, {}).get("median")


def options_are_complete(options, expected_keys):
  return isinstance(options, dict) and set(options) == expected_keys


def benchmark_evidence(summary):
  latency = spin_median(summary, "nolegacy_ms")
  throughput = spin_median(summary, "nolegacy_Matom_per_s")
  if latency is None or throughput is None:
    return None
  return {
      "spin_median_latency_ms": latency,
      "spin_median_throughput_matom_per_s": throughput,
  }


def comparison_record(current_options, baseline_options):
  return {
      "current_options": current_options,
      "baseline_options": baseline_options,
  }


def compare_with_baseline(current, baseline):
  comparison = {}
  current_benchmark_options = current.get("benchmark_options")
  baseline_benchmark_options = baseline.get("benchmark_options")
  benchmark = comparison_record(
      current_benchmark_options, baseline_benchmark_options)
  current_benchmark = benchmark_evidence(current)
  baseline_benchmark = benchmark_evidence(baseline)
  if current_benchmark is None:
    benchmark.update({
        "status": "non_comparable",
        "reason": "current summary is missing benchmark evidence",
    })
  elif baseline_benchmark is None:
    benchmark.update({
        "status": "non_comparable",
        "reason": "baseline summary is missing benchmark evidence",
    })
  elif not options_are_complete(current_benchmark_options, BENCHMARK_OPTION_KEYS):
    benchmark.update({
        "status": "non_comparable",
        "reason": "current summary is missing complete benchmark_options",
    })
  elif not options_are_complete(baseline_benchmark_options, BENCHMARK_OPTION_KEYS):
    benchmark.update({
        "status": "non_comparable",
        "reason": "baseline summary is missing complete benchmark_options",
    })
  elif current_benchmark_options != baseline_benchmark_options:
    benchmark.update({
        "status": "non_comparable",
        "reason": "benchmark command options differ",
    })
  else:
    benchmark.update({
        "status": "comparable",
        "deltas": {
            name: delta_record(current_benchmark[name], baseline_benchmark[name])
            for name in sorted(current_benchmark)
        },
    })
  comparison["benchmark"] = benchmark

  current_ncu = current.get("ncu")
  baseline_ncu = baseline.get("ncu")
  if current_ncu or baseline_ncu:
    current_options = current.get("ncu_workload_options")
    baseline_options = baseline.get("ncu_workload_options")
    ncu_comparison = comparison_record(current_options, baseline_options)
    if not isinstance(current_ncu, dict) or not current_ncu:
      ncu_comparison.update({
          "status": "non_comparable",
          "reason": "current summary is missing NCU evidence",
      })
    elif not isinstance(baseline_ncu, dict) or not baseline_ncu:
      ncu_comparison.update({
          "status": "non_comparable",
          "reason": "baseline summary is missing NCU evidence",
      })
    elif not options_are_complete(current_options, NCU_OPTION_KEYS):
      ncu_comparison.update({
          "status": "non_comparable",
          "reason": "current summary is missing complete ncu_workload_options",
      })
    elif not options_are_complete(baseline_options, NCU_OPTION_KEYS):
      ncu_comparison.update({
          "status": "non_comparable",
          "reason": "baseline summary is missing complete ncu_workload_options",
      })
    elif current_options != baseline_options:
      ncu_comparison.update({
          "status": "non_comparable",
          "reason": "NCU workload or collection options differ",
      })
    elif set(current_ncu) != set(NCU_LABELS) or set(baseline_ncu) != set(NCU_LABELS):
      ncu_comparison.update({
          "status": "non_comparable",
          "reason": "NCU labels do not match expected set",
      })
    else:
      deltas = {}
      metric_mismatch = None
      for label in NCU_LABELS:
        current_profile = current_ncu[label]
        baseline_profile = baseline_ncu[label]
        current_metrics = metric_values(current_profile.get("metrics", []))
        baseline_metrics = metric_values(baseline_profile.get("metrics", []))
        if not current_metrics or not baseline_metrics or (
            set(current_metrics) != set(baseline_metrics)):
          metric_mismatch = label
          break
        deltas[label] = {
            metric: delta_record(current_metrics[metric], baseline_metrics[metric])
            for metric in sorted(current_metrics)
        }
      if metric_mismatch:
        ncu_comparison.update({
            "status": "non_comparable",
            "reason": f"NCU numeric metric names differ for label: {metric_mismatch}",
        })
      else:
        ncu_comparison.update({
            "status": "comparable",
            "deltas": deltas,
        })
    comparison["ncu"] = ncu_comparison
  return comparison


def copy_text_artifacts(profile, ncu, run_name):
  local_dir = local_run_directory(run_name)
  local_dir.mkdir(parents=True, exist_ok=True)
  for label, payload in ncu.items():
    remote_artifacts = payload.get("text_artifacts", {})
    local_artifacts = {}
    for name, remote_path in remote_artifacts.items():
      local_path = local_dir / Path(remote_path).name
      run_local(["scp", f"{profile}:{remote_path}", str(local_path)])
      local_artifacts[name] = str(local_path)
    if remote_artifacts:
      payload["artifacts"] = {
          "remote_report": payload.get("report"),
          "remote_text": remote_artifacts,
          "local_text": local_artifacts,
      }


def validate_run_name(run_name):
  path = Path(run_name)
  if (not run_name or run_name in (".", "..") or path.is_absolute() or
      "/" in run_name):
    raise ValueError("run_name must be a single path component")
  return run_name


def local_run_directory(run_name, root=ROOT):
  return Path(root) / "diagnostics" / "spin_lmp_4090" / validate_run_name(run_name)


def local_ncu_artifact_names():
  suffixes = (".csv", ".cuda.txt", ".sass.txt", ".binary_sass.txt", ".resources.txt")
  return tuple(
      f"ncu_{label}_detailed{suffix}"
      for label in NCU_LABELS for suffix in suffixes)


def cleanup_local_run_artifacts(run_name, root=ROOT):
  local_dir = local_run_directory(run_name, root)
  if not local_dir.is_dir():
    return
  targets = [local_dir / "summary.json"] + [
      local_dir / name for name in local_ncu_artifact_names()
  ]
  for path in targets:
    if path.is_file() or path.is_symlink():
      path.unlink()


def write_summary(summary, run_name):
  local_dir = local_run_directory(run_name)
  local_dir.mkdir(parents=True, exist_ok=True)
  path = local_dir / "summary.json"
  path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
  return path


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--profile", default="4090")
  parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
  parser.add_argument("--gpu", default="0")
  parser.add_argument("--run-name", default="")
  parser.add_argument("--cubic", type=int, default=100)
  parser.add_argument("--spacing", type=float, default=3.0)
  parser.add_argument("--warmup", type=int, default=1)
  parser.add_argument("--iterations", type=int, default=3)
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--ncu-cubic", type=int, default=50)
  parser.add_argument(
      "--ncu-detail",
      action="store_true",
      help="collect source-correlated detailed NCU text artifacts")
  parser.add_argument(
      "--baseline-summary",
      type=Path,
      help="compare this run with a prior summary.json")
  parser.add_argument("--profile-pair", type=int, default=3)
  parser.add_argument(
      "--profile-mode",
      choices=["benchmark", "nsys", "ncu", "all"],
      default="all")
  parser.add_argument("--quick", action="store_true")
  parser.add_argument("--skip-sync", action="store_true")
  parser.add_argument("--skip-build", action="store_true")
  parser.add_argument("--skip-correctness", action="store_true")
  args = parser.parse_args()

  if args.quick:
    args.cubic = 50
    args.iterations = 2
    args.repeats = 1
    args.profile_mode = "benchmark"

  plan = load_plan(args.plan)
  run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
  remote_run_dir = f"{plan['run_dir'].rstrip('/')}/{run_name}"
  cleanup_local_run_artifacts(run_name)

  if not args.skip_sync:
    sync_workspace(args.plan)
  if not args.skip_build:
    build_remote(args.profile, plan, args.gpu)

  provenance = collect_source_provenance(
      args.profile, plan, args.gpu, build_performed=not args.skip_build)

  summary = {
      "run_name": run_name,
      "profile": args.profile,
      "gpu": args.gpu,
      "plan": str(args.plan),
      "remote_run_dir": remote_run_dir,
      "source_provenance": provenance,
      "benchmark_options": benchmark_workload_options(args),
      "ncu_workload_options": ncu_workload_options(args),
  }

  if not args.skip_correctness:
    summary["correctness_stdout"] = run_correctness(args.profile, plan, args.gpu)

  if args.profile_mode in ("benchmark", "all"):
    spin = run_benchmark_case(
        args.profile, plan, args.gpu, args, False, args.repeats, args.profile_pair)
    structural = run_benchmark_case(
        args.profile, plan, args.gpu, args, True, args.repeats, 0)
    summary["benchmarks"] = {
        "spin": {**spin, "summary": summarize_runs(spin["runs"])},
        "stripped_structural": {
            **structural,
            "summary": summarize_runs(structural["runs"]),
        },
    }
    spin_ms = summary["benchmarks"]["spin"]["summary"].get("nolegacy_ms", {}).get("median")
    struct_ms = summary["benchmarks"]["stripped_structural"]["summary"].get(
        "nolegacy_ms", {}).get("median")
    if spin_ms and struct_ms:
      summary["spin_to_structural_ms_ratio"] = spin_ms / struct_ms

  if args.profile_mode in ("nsys", "all"):
    summary["nsys"] = run_nsys(args.profile, plan, args.gpu, args, remote_run_dir)

  if args.profile_mode in ("ncu", "all"):
    summary["ncu"] = run_ncu(args.profile, plan, args.gpu, args, remote_run_dir)
    if args.ncu_detail:
      copy_text_artifacts(args.profile, summary["ncu"], run_name)

  if args.baseline_summary:
    with open(args.baseline_summary, "r", encoding="utf-8") as handle:
      summary["comparison"] = compare_with_baseline(summary, json.load(handle))
    summary["baseline_summary"] = str(args.baseline_summary)

  summary_path = write_summary(summary, run_name)
  print(json.dumps({
      "summary": str(summary_path),
      "remote_run_dir": remote_run_dir,
      "spin_to_structural_ms_ratio": summary.get("spin_to_structural_ms_ratio"),
  }, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
