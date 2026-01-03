#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

mkdir -p "${BUILD_DIR}"

CMAKE_GENERATOR="Ninja"
if ! command -v ninja >/dev/null 2>&1; then
  CMAKE_GENERATOR="Unix Makefiles"
fi

cmake -S "${ROOT_DIR}/backend" -B "${BUILD_DIR}" -G "${CMAKE_GENERATOR}" \
  -DGGML_CUDA=ON \
  -DWHISPER_BUILD_EXAMPLES=OFF \
  -DWHISPER_BUILD_TESTS=OFF

cmake --build "${BUILD_DIR}" -j
