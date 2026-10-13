# DynamicPTQ

DynamicPTQ is a unified post-training quantization project covering two
settings with aligned runtime options and a single launcher per part:

- Dense LLMs: DEEP mixed-precision activation quantization across three
  backends (`FlatQuant`, `SpinQuant`, `QuaRot`)
- MoE LLMs: expert-ordering mixed-precision GPTQ quantization for
    Mixture-of-Experts models (`Mixtral-8x7B-v0.1`)

Supported dense model families include Llama-2, Llama-3, Llama-3.1 and
Qwen2.5.

## Repository Layout

- `scripts/DynamicPTQ.sh`: dense backend dispatcher
- `scripts/run_flatquant.sh`: FlatQuant runner
- `scripts/run_spinquant.sh`: SpinQuant runner
- `scripts/run_quarot.sh`: QuaRot runner
- `scripts/MoEPTQ.sh`: MoE stage dispatcher
- `scripts/run_orders.sh`: compute MoE expert orderings
- `scripts/run_quant.sh`: quantize one MoE configuration
- `scripts/run_eval.sh`: evaluate a quantized MoE checkpoint
- `scripts/run_collect.sh`: aggregate MoE results into one table
- `scripts/run_alloc_pmq.sh`: PMQ / Hessian bit allocators
- `scripts/run_alloc_pmq_official.sh`: PMQ allocator with official MC-MoE semantics
- `FlatQuant/`: FlatQuant backend code
- `SpinQuant/`: SpinQuant backend code
- `QuaRot/`: QuaRot backend code (fake-quant path)
- `MoE/`: MoE ordering, quantization, evaluation and allocator code

## Project Goal

- Run dense PTQ with `METHOD=rtn|gptq` on all supported backends
- Configure DEEP layers explicitly with 0-based layer indices, or select them
  adaptively by per-layer jump ratio
- Quantize MoE expert weights under a per-layer average-bit budget with
  different expert orderings deciding which experts receive higher bit levels
- Keep calibration and evaluation setup consistent across experiments

## Quick Start

```bash
cd DynamicPTQ
```

### Create and activate the environment

```bash
conda create -n DynamicPTQ python=3.10 -y
conda activate DynamicPTQ
pip install -r requirements.txt
pip install -e ./FlatQuant/third-party/fast-hadamard-transform
```

### Optional for gated models

```bash
export HF_TOKEN="your_hf_token"
```

# Part I: Dense Models

### Run with defaults

```bash
bash scripts/DynamicPTQ.sh
```

### Direct backend scripts (recommended for clarity)

```bash
bash scripts/run_flatquant.sh
bash scripts/run_spinquant.sh
bash scripts/run_quarot.sh
```

## Dense Repro Commands

### Backend-only runs (DEEP disabled)

FlatQuant + RTN:

```bash
BACKEND=flatquant METHOD=rtn ENABLE_DEEP=false \
MODEL_PATH=meta-llama/Meta-Llama-3-8B \
bash scripts/run_flatquant.sh
```

FlatQuant + GPTQ:

```bash
BACKEND=flatquant METHOD=gptq ENABLE_DEEP=false \
MODEL_PATH=meta-llama/Meta-Llama-3-8B \
bash scripts/run_flatquant.sh
```

SpinQuant + RTN:

```bash
BACKEND=spinquant METHOD=rtn ENABLE_DEEP=false \
MODEL_PATH=meta-llama/Meta-Llama-3-8B \
OPTIMIZED_ROTATION_PATH=/path/to/R.bin \
bash scripts/run_spinquant.sh
```

SpinQuant + GPTQ:

```bash
BACKEND=spinquant METHOD=gptq ENABLE_DEEP=false \
MODEL_PATH=meta-llama/Meta-Llama-3-8B \
OPTIMIZED_ROTATION_PATH=/path/to/R.bin \
bash scripts/run_spinquant.sh
```

QuaRot + RTN:

```bash
BACKEND=quarot METHOD=rtn ENABLE_DEEP=false \
MODEL_PATH=meta-llama/Meta-Llama-3-8B \
bash scripts/run_quarot.sh
```

QuaRot + GPTQ:

```bash
BACKEND=quarot METHOD=gptq ENABLE_DEEP=false \
MODEL_PATH=meta-llama/Meta-Llama-3-8B \
bash scripts/run_quarot.sh
```

### DEEP runs with explicit layers (`DEEP_SELECT=layers`)

```bash
BACKEND=flatquant METHOD=rtn \
ENABLE_DEEP=true DEEP_SELECT=layers DEEP_LAYERS=0,1,2,31 DEEP_ACT_BITS=8 \
MODEL_PATH=meta-llama/Meta-Llama-3-8B \
bash scripts/DynamicPTQ.sh
```

### DEEP runs with adaptive selection (`DEEP_SELECT=ratio`)

The default mode. Per-layer jump ratios are computed on the FP16 model with the
calibration data, and the top `round(DEEP_RATIO * L)` layers are promoted to
`DEEP_ACT_BITS` activation bits; all other layers keep `A_BITS`.

```bash
BACKEND=flatquant METHOD=rtn \
ENABLE_DEEP=true DEEP_SELECT=ratio DEEP_RATIO=0.1 DEEP_ACT_BITS=8 \
MODEL_PATH=meta-llama/Meta-Llama-3-8B \
bash scripts/DynamicPTQ.sh
```

```bash
BACKEND=spinquant METHOD=gptq \
ENABLE_DEEP=true DEEP_SELECT=ratio DEEP_RATIO=0.1 DEEP_ACT_BITS=8 \
MODEL_PATH=meta-llama/Meta-Llama-3-8B \
OPTIMIZED_ROTATION_PATH=/path/to/R.bin \
bash scripts/DynamicPTQ.sh
```

```bash
BACKEND=quarot METHOD=rtn \
ENABLE_DEEP=true DEEP_SELECT=ratio DEEP_RATIO=0.1 DEEP_ACT_BITS=8 \
MODEL_PATH=meta-llama/Meta-Llama-3-8B \
bash scripts/DynamicPTQ.sh
```

### Alternative selection modes

- `DEEP_SELECT=random`: promote a uniformly random set of `round(DEEP_RATIO * L)` layers
- `DEEP_SELECT=bottom`: promote the lowest-ranked layers by jump ratio

```bash
BACKEND=quarot METHOD=rtn \
ENABLE_DEEP=true DEEP_SELECT=random DEEP_RATIO=0.1 \
MODEL_PATH=meta-llama/Llama-2-7b-hf \
bash scripts/run_quarot.sh
```

### Recommended dense full setup

```bash
CALI_DATASET=wikitext2 \
NSAMPLES=128 \
EVAL_DATASETS="wikitext2 c4" \
RUN_LM_EVAL=true \
LM_EVAL_TASKS="piqa hellaswag arc_easy arc_challenge winogrande lambada_openai" \
bash scripts/DynamicPTQ.sh
```

# Part II: MoE Models

DynamicPTQ-MoE quantizes Mixtral-8x7B-v0.1 expert weights with GPTQ under a
per-layer average-bit budget and compares different expert orderings that
decide which experts receive the higher bit levels.

- residual-dynamics orderings: per-expert Jump Ratio (JR) and JR+SNR
- router-norm + MaxVar ordering
- frequency / gate-weight heuristic orderings
- random ordering and uniform bit-width baselines
- calibration-based PMQ and Hessian bit allocators

Evaluation covers WikiText2 perplexity and eight zero-shot tasks (PIQA,
ARC-e, ARC-c, BoolQ, HellaSwag, WinoGrande, MathQA, MMLU).

## MoE Repro Pipeline

The pipeline has four stages: compute orderings, quantize, evaluate, collect.
All MoE artifacts (`outputs/`, `ckpts/`, `results/`) are placed under `MoE/`.

### 1. Compute expert orderings

```bash
MODEL_PATH=mistralai/Mixtral-8x7B-v0.1 \
STAGE=orders \
bash scripts/MoEPTQ.sh
```

Writes `MoE/outputs/orders.json` with all orderings and per-layer statistics.

### 2. Quantize one configuration

```bash
ORDER=ours_jr AVG_BITS=2.5 \
STAGE=quant \
bash scripts/MoEPTQ.sh
```

Available `ORDER` values: `ours_jr`, `ours_jr_snr`, `ours_gated`,
`paper_full`, `paper_lam`, `maxvar`, `freq`, `gatew`, `random`.

Uniform bit-width baselines:

```bash
MIXED_TYPE=uniform UNIFORM_BITS=2 \
STAGE=quant \
bash scripts/MoEPTQ.sh
```

PMQ / Hessian allocator baselines (run an allocator first, then quantize
from its bit-config file):

```bash
STAGE=alloc_pmq bash scripts/MoEPTQ.sh
BIT_CONFIG_FILE=MoE/outputs/alloc_pmq/pmq_2.5.json \
STAGE=quant \
bash scripts/MoEPTQ.sh
```

### 3. Evaluate a quantized checkpoint

```bash
EXP_NAME=ours_jr_2.5 TP=2 \
STAGE=eval \
bash scripts/MoEPTQ.sh
```

Writes `MoE/results/ours_jr_2.5.json`.

### 4. Collect results

```bash
STAGE=collect bash scripts/MoEPTQ.sh
```

Prints one table over all checkpoints and writes `MoE/results/summary.json`.

### Direct stage scripts (recommended for clarity)

```bash
bash scripts/run_orders.sh
bash scripts/run_quant.sh
bash scripts/run_eval.sh
bash scripts/run_collect.sh
bash scripts/run_alloc_pmq.sh
bash scripts/run_alloc_pmq_official.sh
```

## MoE Method Summary

Each layer orders its experts; the first experts in the order receive the
high bit level, the last receive the low bit level, under a three-level
(3, 2, 1) bit scheme that matches the average-bit budget. Non-expert weights
are quantized to `ATTN_BIT_LEVEL` (default 4) bits.

- `ours_jr`: residual-dynamics Jump Ratio, `||d_s|| / ||h||` per expert,
  descending (`h` is the residual stream entering the decoder layer, `d_s`
  is the gate-weighted output of expert `s`)
- `ours_jr_snr`: `0.5 * robust_z(JR) + 0.5 * robust_z(-SNR)`, descending
- `ours_gated`: `0.4 * robust_z(JR) + 0.4 * robust_z(-SNR) + 0.2 * uncertainty`
- `paper_full`: router L2-norm ascending + MaxVar reordering
- `freq` / `gatew`: activation frequency / mean routing weight, descending
- `random`: seeded shuffle

Calibration uses WikiText2 with 128 samples of 2048 tokens, seed 0; GPTQ
uses group size 128 and 1% damping.

## Calibration And Evaluation Data Guide

### Dense backends

Prepare datasets under `FlatQuant/datasets` (or your own mirrored directory):

- Calibration / PPL evaluation:
  - WikiText2: download from [wikitext](https://huggingface.co/datasets/wikitext), place under `FlatQuant/datasets/wikitext`.
  - C4: download from [allenai/c4](https://huggingface.co/datasets/allenai/c4), place under `FlatQuant/datasets/allenai/c4`.

- Commonsense QA evaluation (`lm-eval` tasks):
  - ARC-E / ARC-C: [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) -> `FlatQuant/datasets/ai2_arc`
  - HellaSwag: [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) -> `FlatQuant/datasets/hellaswag`
  - LAMBADA: [EleutherAI/lambada_openai](https://huggingface.co/datasets/EleutherAI/lambada_openai) -> `FlatQuant/datasets/lambada_openai`
  - PIQA: [ybisk/piqa](https://huggingface.co/datasets/ybisk/piqa) -> `FlatQuant/datasets/piqa`
  - WinoGrande: [winogrande](https://huggingface.co/datasets/winogrande) -> `FlatQuant/datasets/winogrande`

### MoE pipeline

- Calibration and perplexity: WikiText2 (`wikitext`, `wikitext-2-raw-v1`),
  downloaded automatically from the Hugging Face Hub on first use.
- Zero-shot tasks: downloaded automatically by `lm-eval` on first use.
- Datasets are cached under the default Hugging Face cache directory.

## Backbone-Specific Notes

### FlatQuant

- Python environment should use `python=3.10`.
- Recommended calibration defaults: `NSAMPLES=128`, `CALI_BSZ=4`, `EPOCHS=15`, `FLAT_LR=5e-3`.
- Typical low-bit setup is `W_BITS=4 A_BITS=4 K_BITS=4 V_BITS=4`.
- For gated models (for example Llama family), export `HF_TOKEN` before running.
- Optional attention speedup: `pip install flash-attn --no-build-isolation`.

### SpinQuant

- `OPTIMIZED_ROTATION_PATH` is mandatory and must point to an existing rotation file
  (can be produced with `SpinQuant/optimize_rotation.py`).
- For large models (for example 70B), set multiple visible GPUs and keep
  `DISTRIBUTE_MODEL=true`.

### QuaRot

- The fake-quant path assumes rotation is enabled (`--rotate` is always passed by the runner).
- LLaMA-family models are the most verified path; test on LLaMA before expanding
  to other families.

### MoE

- The quantize stage runs on a single GPU (a single decoder layer is
  resident on device at a time); use `GPU` to select it.
- The orders and allocators stages load the model with `device_map=auto`
  over all visible GPUs.
- The eval stage loads the full fp16 checkpoint; for Mixtral-8x7B use
  `TP=2` (or more) GPUs with the vLLM backend, or `BACKEND=hf` with
  `device_map=auto`.
- If quantization and evaluation live in different conda environments, set
  `CONDA_ENV_NAME` per stage.
- Each quantized checkpoint is about 90 GB (fp16 fake-quant weights).

### Model-family hints (dense)

- Llama-2 / Llama-3 / Llama-3.1 / Qwen2.5 are the most verified dense paths.
- For 70B-class checkpoints, prefer multi-GPU execution and keep
  `DISTRIBUTE_MODEL=true`.
- With `DEEP_SELECT=layers`, start from `DEEP_LAYERS=0,1,2,<last_layer_id>` and
  tune only after a stable baseline run.

## Output Layout

### Dense backends

- `${OUTPUT_DIR}/${EXP_NAME}/` (FlatQuant, override with `OUTPUT_DIR` and `EXP_NAME`)
- `QuaRot` and `SpinQuant` print metrics to stdout; redirect or tee as needed

### MoE pipeline

- `MoE/outputs/orders.json`: expert orderings and per-layer statistics
- `MoE/outputs/alloc_pmq/`, `MoE/outputs/alloc_pmq_official/`: allocator bit configs
- `MoE/ckpts/${EXP_NAME}/`: quantized checkpoint plus `meta.json`
- `MoE/results/${EXP_NAME}.json`: per-task evaluation results
- `MoE/results/summary.json`: aggregated table

## Main Configuration

### Dense backends

- `BACKEND`: `flatquant|spinquant|quarot`
- `METHOD`: `rtn|gptq`
- `MODEL_PATH`: model id or local model directory
- `ENABLE_DEEP`: `true|false`
- `DEEP_SELECT`: `layers|ratio|random|bottom`
- `DEEP_LAYERS`: comma-separated 0-based layer ids, used when `DEEP_SELECT=layers` (for example: `0,1,2,31`)
- `DEEP_RATIO`: fraction of layers promoted when `DEEP_SELECT` is `ratio|random|bottom`
- `DEEP_ACT_BITS`: activation bits for DEEP-selected layers
- `W_BITS`, `A_BITS`, `K_BITS`, `V_BITS`: PTQ precision setup
- `K_GROUPSIZE`, `V_GROUPSIZE`: group sizes for KV quantization, default `128`
- `CALI_DATASET`, `NSAMPLES`, `CALI_BSZ`: calibration setup
- `EVAL_DATASETS`: PPL evaluation datasets
- `RUN_LM_EVAL`: `true|false`
- `LM_EVAL_TASKS`: space-separated lm-eval tasks

### MoE pipeline

- `STAGE`: `orders|quant|eval|collect|alloc_pmq|alloc_pmq_official`
- `MODEL_PATH`: model id or local model directory, default `mistralai/Mixtral-8x7B-v0.1`
- `ORDER`: ordering name for the quantize stage, default `ours_jr`
- `AVG_BITS`: average expert bits, default `2.5`
- `ORDER_FILE`: orderings json, default `MoE/outputs/orders.json`
- `BIT_CONFIG_FILE`: explicit `{layer: {expert: bits}}` json for allocator baselines
- `MIXED_TYPE`: `mixed|uniform`, default `mixed`
- `UNIFORM_BITS`: expert bits when `MIXED_TYPE=uniform`
- `HIGH_BIT_LEVEL`, `MID_BIT_LEVEL`, `LOW_BIT_LEVEL`: expert bit levels, default `3`, `2`, `1`
- `ATTN_BIT_LEVEL`: non-expert weight bits, default `4`
- `GROUPSIZE`, `PERCDAMP`, `NSAMPLES`, `SEED`, `DATASET`: GPTQ setup, default `128`, `0.01`, `128`, `0`, `wikitext2`
- `DEVICE`, `GPU`: quantization device and GPU selection
- `GPTQ_PROPAGATE`, `RESIDUAL2`: optional quantizer variants, default `false`
- `EXP_NAME`, `OUT_DIR`: checkpoint naming, default `${ORDER}_${AVG_BITS}`
- `TP`, `BACKEND`, `TASKS`, `MAX_MODEL_LEN`, `GPU_UTIL`: evaluation setup, default `2`, `vllm`, eight tasks, `4096`, `0.78`
- `ZETA`, `ORDERS_OUT`: ordering stage setup, default `3.0`, `MoE/outputs/orders.json`
- `AVG_BITS_LIST`, `ALLOC_OUT_DIR`, `SEQS`: allocator stage setup

### Common

- `CONDA_ENV_NAME`: conda environment activated by the runners, default `DynamicPTQ`
- `WORKSPACE_ROOT`: dense runners use the repository root; MoE stage runners use `MoE/`

## Notes

- The launchers auto-detect their workspace root from script location.
- For the SpinQuant backend, `OPTIMIZED_ROTATION_PATH` is required.
- No hardcoded personal paths or tokens are required.

## Acknowledgements

This project builds on and integrates ideas and implementations from:

- `FlatQuant`
- `SpinQuant`
- `QuaRot`
- GPTQ
- the official `moe_quantization` repository (ICLR 2026)
- Mixture-Compressor-MoE (MC-MoE)
- `lm-eval` and `vLLM`

We thank the original authors and maintainers of these projects for their
valuable open-source contributions.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. See LICENSE for details.
