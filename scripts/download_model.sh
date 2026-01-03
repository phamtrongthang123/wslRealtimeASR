#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="${1:-small.en}"

mkdir -p "${ROOT_DIR}/models"
"${ROOT_DIR}/third_party/whisper.cpp/models/download-ggml-model.sh" "${MODEL_NAME}" "${ROOT_DIR}/models"
