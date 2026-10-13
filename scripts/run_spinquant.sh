#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-DynamicPTQ}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash.hook)"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] conda is not available. Set up conda first." >&2
  exit 1
fi
conda activate "${CONDA_ENV_NAME}"

METHOD="${METHOD:-rtn}"
MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3-8B}"
OPTIMIZED_ROTATION_PATH="${OPTIMIZED_ROTATION_PATH:-}"
HF_TOKEN="${HF_TOKEN:-}"
CALI_DATASET="${CALI_DATASET:-wikitext2}"
EVAL_DATASETS="${EVAL_DATASETS:-wikitext2 c4}"
RUN_LM_EVAL="${RUN_LM_EVAL:-false}"
LM_EVAL_TASKS="${LM_EVAL_TASKS:-piqa hellaswag arc_easy arc_challenge winogrande lambada_openai}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-16}"
W_BITS="${W_BITS:-4}"
A_BITS="${A_BITS:-4}"
K_BITS="${K_BITS:-4}"
V_BITS="${V_BITS:-4}"
K_GROUPSIZE="${K_GROUPSIZE:-128}"
V_GROUPSIZE="${V_GROUPSIZE:-128}"
NSAMPLES="${NSAMPLES:-128}"
SEED="${SEED:-0}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
ENABLE_DEEP="${ENABLE_DEEP:-true}"
DEEP_ACT_BITS="${DEEP_ACT_BITS:-8}"
DEEP_SELECT="${DEEP_SELECT:-ratio}"
DEEP_RATIO="${DEEP_RATIO:-0.1}"
DEEP_LAYERS="${DEEP_LAYERS:-}"
DISTRIBUTE_MODEL="${DISTRIBUTE_MODEL:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${OPTIMIZED_ROTATION_PATH}" || ! -f "${OPTIMIZED_ROTATION_PATH}" ]]; then
  echo "[ERROR] OPTIMIZED_ROTATION_PATH is required and must exist." >&2
  exit 1
fi
if [[ "${ENABLE_DEEP}" == "true" && "${DEEP_SELECT}" == "layers" && -z "${DEEP_LAYERS}" ]]; then
  echo "[ERROR] DEEP_LAYERS must be set when ENABLE_DEEP=true and DEEP_SELECT=layers. Example: 0,1,2,31" >&2
  exit 1
fi

read -r -a EVAL_DATASET_ARRAY <<< "${EVAL_DATASETS}"
read -r -a LM_EVAL_TASK_ARRAY <<< "${LM_EVAL_TASKS}"

HF_TOKEN_FLAGS=()
[[ -n "${HF_TOKEN}" ]] && HF_TOKEN_FLAGS=(--access_token "${HF_TOKEN}")

DEEP_FLAGS=(--no-enable_deep)
if [[ "${ENABLE_DEEP}" == "true" ]]; then
  DEEP_FLAGS=(--enable_deep --deep_act_bits "${DEEP_ACT_BITS}" --deep_select "${DEEP_SELECT}")
  if [[ "${DEEP_SELECT}" == "layers" ]]; then
    DEEP_FLAGS+=(--deep_layers "${DEEP_LAYERS}")
  else
    DEEP_FLAGS+=(--deep_ratio "${DEEP_RATIO}")
  fi
fi

LM_EVAL_FLAGS=()
if [[ "${RUN_LM_EVAL}" == "true" ]]; then
  LM_EVAL_FLAGS=(--lm_eval --lm_eval_tasks "${LM_EVAL_TASK_ARRAY[@]}" --lm_eval_batch_size "${LM_EVAL_BATCH_SIZE}")
fi

METHOD_FLAGS=(--w_rtn)
if [[ "${METHOD}" == "gptq" ]]; then
  METHOD_FLAGS=()
fi

DISTRIBUTE_FLAGS=()
if [[ "${DISTRIBUTE_MODEL}" == "true" ]]; then
  DISTRIBUTE_FLAGS=(--distribute)
fi

MASTER_PORT="${MASTER_PORT:-$((10000 + (${SLURM_JOB_ID:-0} % 50000)))}"
echo "[INFO] backend=spinquant method=${METHOD}"
echo "[INFO] model=${MODEL_PATH}"
echo "[INFO] master_port=${MASTER_PORT}"

cd "${WORKSPACE_ROOT}/SpinQuant"
PYTHONPATH="${WORKSPACE_ROOT}:${PYTHONPATH:-}" torchrun --nnodes=1 --nproc_per_node=1 --master_port "${MASTER_PORT}" ptq.py \
  --input_model "${MODEL_PATH}" \
  --do_train False \
  --do_eval True \
  --per_device_eval_batch_size "${LM_EVAL_BATCH_SIZE}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --fp16 False \
  --bf16 True \
  --save_safetensors False \
  --cali_dataset "${CALI_DATASET}" \
  --eval_datasets "${EVAL_DATASET_ARRAY[@]}" \
  --w_bits "${W_BITS}" --a_bits "${A_BITS}" \
  --k_bits "${K_BITS}" --k_asym --k_groupsize "${K_GROUPSIZE}" \
  --v_bits "${V_BITS}" --v_asym --v_groupsize "${V_GROUPSIZE}" \
  --nsamples "${NSAMPLES}" \
  --seed "${SEED}" \
  --rotate --optimized_rotation_path "${OPTIMIZED_ROTATION_PATH}" \
  "${HF_TOKEN_FLAGS[@]}" \
  "${METHOD_FLAGS[@]}" \
  "${DEEP_FLAGS[@]}" \
  "${LM_EVAL_FLAGS[@]}" \
  "${DISTRIBUTE_FLAGS[@]}"
