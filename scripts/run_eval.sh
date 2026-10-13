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

EXP_NAME="${EXP_NAME:-}"
CKPT="${CKPT:-}"
TP="${TP:-2}"
BACKEND="${BACKEND:-vllm}"
TASKS="${TASKS:-piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande,mathqa,mmlu}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.78}"

if [[ -z "${CKPT}" ]]; then
  if [[ -z "${EXP_NAME}" ]]; then
    echo "[ERROR] Set EXP_NAME or CKPT." >&2
    exit 1
  fi
  CKPT="${WORKSPACE_ROOT}/ckpts/${EXP_NAME}"
fi
if [[ ! -d "${CKPT}" ]]; then
  echo "[ERROR] checkpoint dir not found: ${CKPT}" >&2
  exit 1
fi

OUT="${OUT:-${WORKSPACE_ROOT}/results/$(basename "${CKPT}").json}"
mkdir -p "$(dirname "${OUT}")"

echo "[INFO] stage=eval ckpt=${CKPT} backend=${BACKEND} tp=${TP}"

PYTHONPATH="${WORKSPACE_ROOT}:${PYTHONPATH:-}" python "${WORKSPACE_ROOT}/src/eval_benchmarks.py" \
  --ckpt "${CKPT}" \
  --out "${OUT}" \
  --tp "${TP}" \
  --backend "${BACKEND}" \
  --tasks "${TASKS}" \
  --max_model_len "${MAX_MODEL_LEN}" \
  --gpu_util "${GPU_UTIL}"
