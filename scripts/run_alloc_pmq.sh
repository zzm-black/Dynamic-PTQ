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
AVG_BITS_LIST="${AVG_BITS_LIST:-2.75 2.5 2.25 2.0 1.75}"
ALLOC_OUT_DIR="${ALLOC_OUT_DIR:-${WORKSPACE_ROOT}/outputs/alloc_pmq}"

mkdir -p "${ALLOC_OUT_DIR}"

read -r -a AVG_BITS_ARRAY <<< "${AVG_BITS_LIST}"

echo "[INFO] stage=alloc_pmq model=${MODEL_PATH}"

PYTHONPATH="${WORKSPACE_ROOT}:${PYTHONPATH:-}" python "${WORKSPACE_ROOT}/tools/allocators.py" \
  --model_path "${MODEL_PATH}" \
  --out_dir "${ALLOC_OUT_DIR}" \
  --avg_bits "${AVG_BITS_ARRAY[@]}"
