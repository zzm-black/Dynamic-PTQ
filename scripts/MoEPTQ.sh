#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
DynamicPTQ MoE stage dispatcher

Usage:
  bash scripts/MoEPTQ.sh [--stage orders|quant|eval|collect|alloc_pmq|alloc_pmq_official] [extra args]

This script only dispatches to stage-specific scripts:
  - scripts/run_orders.sh
  - scripts/run_quant.sh
  - scripts/run_eval.sh
  - scripts/run_collect.sh
  - scripts/run_alloc_pmq.sh
  - scripts/run_alloc_pmq_official.sh

Examples:
  STAGE=orders MODEL_PATH=mistralai/Mixtral-8x7B-v0.1 bash scripts/MoEPTQ.sh

  STAGE=quant ORDER=ours_jr AVG_BITS=2.5 bash scripts/MoEPTQ.sh
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STAGE="${STAGE:-quant}"
if [[ "${1:-}" == "--stage" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "[ERROR] --stage requires a value." >&2
    exit 1
  fi
  STAGE="${2}"
  shift 2
fi
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

case "${STAGE}" in
  orders|quant|eval|collect|alloc_pmq|alloc_pmq_official) ;;
  *)
    echo "[ERROR] Unsupported STAGE=${STAGE}. Use orders|quant|eval|collect|alloc_pmq|alloc_pmq_official." >&2
    exit 1
    ;;
esac

exec "${SCRIPT_DIR}/run_${STAGE}.sh" "$@"
