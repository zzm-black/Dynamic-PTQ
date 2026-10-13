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
HF_TOKEN="${HF_TOKEN:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_ROOT}/outputs/flatquant}"
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
CALI_BSZ="${CALI_BSZ:-4}"
EPOCHS="${EPOCHS:-15}"
FLAT_LR="${FLAT_LR:-5e-3}"
SEED="${SEED:-0}"
ENABLE_DEEP="${ENABLE_DEEP:-true}"
DEEP_ACT_BITS="${DEEP_ACT_BITS:-8}"
DEEP_SELECT="${DEEP_SELECT:-ratio}"
DEEP_RATIO="${DEEP_RATIO:-0.1}"
DEEP_LAYERS="${DEEP_LAYERS:-}"
DISTRIBUTE_MODEL="${DISTRIBUTE_MODEL:-false}"

if [[ "${ENABLE_DEEP}" == "true" && "${DEEP_SELECT}" == "layers" && -z "${DEEP_LAYERS}" ]]; then
  echo "[ERROR] DEEP_LAYERS must be set when ENABLE_DEEP=true and DEEP_SELECT=layers. Example: 0,1,2,31" >&2
  exit 1
fi

read -r -a EVAL_DATASET_ARRAY <<< "${EVAL_DATASETS}"
read -r -a LM_EVAL_TASK_ARRAY <<< "${LM_EVAL_TASKS}"

HF_TOKEN_FLAGS=()
[[ -n "${HF_TOKEN}" ]] && HF_TOKEN_FLAGS=(--hf_token "${HF_TOKEN}")

MODEL_TAG="${MODEL_PATH//\//_}"
EXP_NAME="${EXP_NAME:-flatquant_${METHOD}_w${W_BITS}a${A_BITS}k${K_BITS}v${V_BITS}_${MODEL_TAG}}"

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
  LM_EVAL_FLAGS=(--lm_eval --tasks "${LM_EVAL_TASK_ARRAY[@]}" --lm_eval_batch_size "${LM_EVAL_BATCH_SIZE}")
fi

METHOD_FLAGS=()
if [[ "${METHOD}" == "gptq" ]]; then
  METHOD_FLAGS=(--gptq)
fi

DISTRIBUTE_FLAGS=()
if [[ "${DISTRIBUTE_MODEL}" == "true" ]]; then
  DISTRIBUTE_FLAGS=(--distribute_model)
fi

echo "[INFO] backend=flatquant method=${METHOD}"
echo "[INFO] model=${MODEL_PATH}"
echo "[INFO] output_dir=${OUTPUT_DIR}/${EXP_NAME}"

cd "${WORKSPACE_ROOT}/FlatQuant"
PYTHONPATH="${WORKSPACE_ROOT}:${PYTHONPATH:-}" python main.py \
  --model "${MODEL_PATH}" \
  --cali_dataset "${CALI_DATASET}" \
  --eval_datasets "${EVAL_DATASET_ARRAY[@]}" \
  --w_bits "${W_BITS}" --a_bits "${A_BITS}" \
  --k_bits "${K_BITS}" --k_asym --k_groupsize "${K_GROUPSIZE}" \
  --v_bits "${V_BITS}" --v_asym --v_groupsize "${V_GROUPSIZE}" \
  --nsamples "${NSAMPLES}" --cali_bsz "${CALI_BSZ}" \
  --epochs "${EPOCHS}" --flat_lr "${FLAT_LR}" \
  --lwc --lac --cali_trans --add_diag \
  --seed "${SEED}" \
  --output_dir "${OUTPUT_DIR}" \
  --exp_name "${EXP_NAME}" \
  "${HF_TOKEN_FLAGS[@]}" \
  "${METHOD_FLAGS[@]}" \
  "${DEEP_FLAGS[@]}" \
  "${LM_EVAL_FLAGS[@]}" \
  "${DISTRIBUTE_FLAGS[@]}"
