#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${SCRIPT_DIR}/../MoE" && pwd)}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-DynamicPTQ}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] conda is not available. Set up conda first." >&2
  exit 1
fi
conda activate "${CONDA_ENV_NAME}"

MODEL_PATH="${MODEL_PATH:-mistralai/Mixtral-8x7B-v0.1}"
ZETA="${ZETA:-3.0}"
ORDERS_OUT="${ORDERS_OUT:-${WORKSPACE_ROOT}/outputs/orders.json}"

mkdir -p "$(dirname "${ORDERS_OUT}")"

echo "[INFO] stage=orders model=${MODEL_PATH}"

PYTHONPATH="${WORKSPACE_ROOT}:${PYTHONPATH:-}" python "${WORKSPACE_ROOT}/src/compute_orders.py" \
  --model_path "${MODEL_PATH}" \
  --out "${ORDERS_OUT}" \
  --zeta "${ZETA}"
