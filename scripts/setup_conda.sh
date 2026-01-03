#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="wslrealtimeasr"

if conda env list | awk '{print $1}' | grep -q "^${ENV_NAME}$"; then
  conda env update -f "${ROOT_DIR}/environment.yml" --prune
else
  conda env create -f "${ROOT_DIR}/environment.yml"
fi

echo "Conda env ready. Activate with: conda activate ${ENV_NAME}"
