#!/usr/bin/env python3
"""Run the spin LAMMPS GPU benchmark/profiling loop on the 4090 host."""

import argparse
import csv
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
        -DCMAKE_CUDA_ARCHITECTURES=89
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


def parse_ncu_csv_text(text):
  lines = text.splitlines()
  header_index = None
  for index, line in enumerate(lines):
    if line.startswith('"ID",') or line.startswith("ID,"):
      header_index = index
      break
  if header_index is None:
    return []
  selected = [
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
  ]
  reader = csv.DictReader(lines[header_index:])
  rows = []
  for row in reader:
    kernel = row.get("Kernel Name", "")
    if kernel:
      rows.append({"metric": "Kernel Name", "value": kernel})
      for metric in selected:
        if metric in row:
          rows.append({"metric": metric, "value": row[metric]})
      break
  return rows


def run_ncu_one(profile, plan, gpu, args, run_dir, label, kernel_regex):
  csv_path = f"{run_dir}/ncu_{label}.csv"
  command = bench_command(plan, args, False)
  script = remote_env(gpu) + textwrap.dedent(f"""
      if ! command -v ncu >/dev/null 2>&1; then
        echo '{{"skipped":"ncu not found"}}'
        exit 0
      fi
      mkdir -p {q(run_dir)}
      cd {q(plan["source_dir"])}
      ncu --target-processes all --kernel-name {q("regex:" + kernel_regex)} \
        --launch-count 1 \
        --metrics gpu__time_duration.sum,launch__registers_per_thread,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__warps_eligible.avg.per_cycle_active,\
smsp__issue_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct \
        --csv --page raw --log-file {q(csv_path)} \
        {' '.join(q(part) for part in command)} >/dev/null 2>&1 || true
      python3 - <<PY
import json
from pathlib import Path
path = Path({str(csv_path)!r})
print(json.dumps({{"csv": str(path), "text": path.read_text() if path.exists() else ""}}))
PY
      """)
  payload = json.loads(ssh_bash(profile, script).stdout.strip().splitlines()[-1])
  text = payload.pop("text", "")
  payload["metrics"] = parse_ncu_csv_text(text)
  return payload


def run_ncu(profile, plan, gpu, args, run_dir):
  small = argparse.Namespace(**vars(args))
  small.cubic = args.ncu_cubic
  small.warmup = 1
  small.iterations = 1
  return {
      "chiral": run_ncu_one(
          profile,
          plan,
          gpu,
          small,
          run_dir,
          "chiral",
          "accumulate_spin_chiral_forces"),
      "density": run_ncu_one(
          profile,
          plan,
          gpu,
          small,
          run_dir,
          "density",
          "accumulate_spin_density_forces_c4_l4_(pull|block_f32)"),
  }


def write_summary(summary, run_name):
  local_dir = ROOT / "diagnostics" / "spin_lmp_4090" / run_name
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

  if not args.skip_sync:
    sync_workspace(args.plan)
  if not args.skip_build:
    build_remote(args.profile, plan, args.gpu)

  summary = {
      "run_name": run_name,
      "profile": args.profile,
      "gpu": args.gpu,
      "plan": str(args.plan),
      "remote_run_dir": remote_run_dir,
      "benchmark_options": {
          "cubic": args.cubic,
          "spacing": args.spacing,
          "warmup": args.warmup,
          "iterations": args.iterations,
          "repeats": args.repeats,
      },
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

  summary_path = write_summary(summary, run_name)
  print(json.dumps({
      "summary": str(summary_path),
      "remote_run_dir": remote_run_dir,
      "spin_to_structural_ms_ratio": summary.get("spin_to_structural_ms_ratio"),
  }, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
