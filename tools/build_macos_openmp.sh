#!/bin/bash
set -euo pipefail

# Homebrew bottles can require a newer macOS than the wheel deployment target.
project_root=$(cd "$(dirname "$0")/.." && pwd)
work="$project_root/.build/wheel-openmp"
mkdir -p "$work/source"
curl --fail --location --silent --show-error --retry 3 \
  https://github.com/llvm/llvm-project/releases/download/llvmorg-23.1.5/llvm-project-23.1.5.src.tar.xz \
  --output "$work/llvm.tar.xz"
printf '%s  %s\n' \
  fb50b1eac93ddeb3aa8dab4cfe324dff74f4a9c2bdee6160c9e36188f8d06a6e \
  "$work/llvm.tar.xz" | shasum -a 256 --check
tar -xf "$work/llvm.tar.xz" -C "$work/source" --strip-components=1
cmake -S "$work/source/runtimes" -B "$work/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$work/install" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
  -DLLVM_ENABLE_RUNTIMES=openmp \
  -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
  -DOPENMP_ENABLE_OMPT_TOOLS=OFF \
  -DLIBOMP_INSTALL_ALIASES=OFF
cmake --build "$work/build" --parallel 3
cmake --install "$work/build"
# Delocate must find the runtime before wheel-local library IDs are normalized.
install_name_tool -id "$work/install/lib/libomp.dylib" "$work/install/lib/libomp.dylib"
