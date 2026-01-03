#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${1:-${ROOT_DIR}/models/ggml-small.en.bin}"
PUBLIC_DIR="${ROOT_DIR}/frontend"

shift || true

"${ROOT_DIR}/build/realtime_server" \
  --model "${MODEL_PATH}" \
  --public "${PUBLIC_DIR}" \
  "$@"
