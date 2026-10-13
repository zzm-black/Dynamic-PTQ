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
ORDER="${ORDER:-ours_jr}"
AVG_BITS="${AVG_BITS:-2.5}"
ORDER_FILE="${ORDER_FILE:-${WORKSPACE_ROOT}/outputs/orders.json}"
BIT_CONFIG_FILE="${BIT_CONFIG_FILE:-}"
MIXED_TYPE="${MIXED_TYPE:-mixed}"
UNIFORM_BITS="${UNIFORM_BITS:-}"
HIGH_BIT_LEVEL="${HIGH_BIT_LEVEL:-3}"
MID_BIT_LEVEL="${MID_BIT_LEVEL:-2}"
LOW_BIT_LEVEL="${LOW_BIT_LEVEL:-1}"
ATTN_BIT_LEVEL="${ATTN_BIT_LEVEL:-4}"
GROUPSIZE="${GROUPSIZE:-128}"
PERCDAMP="${PERCDAMP:-0.01}"
NSAMPLES="${NSAMPLES:-128}"
SEED="${SEED:-0}"
DATASET="${DATASET:-wikitext2}"
DEVICE="${DEVICE:-cuda:0}"
GPU="${GPU:-}"

if [[ -n "${GPU}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU}"
fi
export TOKENIZERS_PARALLELISM=false

if [[ "${MIXED_TYPE}" == "uniform" ]]; then
  if [[ -z "${UNIFORM_BITS}" ]]; then
    echo "[ERROR] UNIFORM_BITS must be set when MIXED_TYPE=uniform." >&2
    exit 1
  fi
  EXP_NAME="${EXP_NAME:-uniform${UNIFORM_BITS}}"
else
  if [[ -n "${BIT_CONFIG_FILE}" ]]; then
    EXP_NAME="${EXP_NAME:-$(basename "${BIT_CONFIG_FILE}" .json)}"
  else
    EXP_NAME="${EXP_NAME:-${ORDER}_${AVG_BITS}}"
  fi
fi
OUT_DIR="${OUT_DIR:-${WORKSPACE_ROOT}/ckpts/${EXP_NAME}}"

QUANT_FLAGS=(
  --model_path "${MODEL_PATH}"
  --out "${OUT_DIR}"
  --mixed_type "${MIXED_TYPE}"
  --avg_bits "${AVG_BITS}"
  --high_bit_level "${HIGH_BIT_LEVEL}"
  --mid_bit_level "${MID_BIT_LEVEL}"
  --low_bit_level "${LOW_BIT_LEVEL}"
  --attn_bit_level "${ATTN_BIT_LEVEL}"
  --groupsize "${GROUPSIZE}"
  --percdamp "${PERCDAMP}"
  --nsamples "${NSAMPLES}"
  --seed "${SEED}"
  --dataset "${DATASET}"
  --device "${DEVICE}"
)

if [[ "${MIXED_TYPE}" != "uniform" && -n "${BIT_CONFIG_FILE}" ]]; then
  QUANT_FLAGS+=(--bit_config_file "${BIT_CONFIG_FILE}")
elif [[ "${MIXED_TYPE}" != "uniform" ]]; then
  if [[ ! -f "${ORDER_FILE}" ]]; then
    echo "[ERROR] ORDER_FILE=${ORDER_FILE} not found. Run the orders stage first." >&2
    exit 1
  fi
  QUANT_FLAGS+=(--order_file "${ORDER_FILE}" --order_name "${ORDER}")
fi
if [[ "${MIXED_TYPE}" == "uniform" ]]; then
  QUANT_FLAGS+=(--uniform_bits "${UNIFORM_BITS}")
fi
if [[ "${GPTQ_PROPAGATE:-false}" == "true" ]]; then
  QUANT_FLAGS+=(--gptq_propagate)
fi
if [[ "${RESIDUAL2:-false}" == "true" ]]; then
  QUANT_FLAGS+=(--residual2)
fi

echo "[INFO] stage=quant exp=${EXP_NAME} model=${MODEL_PATH}"

cd "${WORKSPACE_ROOT}"
python src/quantize_mixtral.py "${QUANT_FLAGS[@]}"
