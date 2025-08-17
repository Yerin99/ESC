#!/usr/bin/env bash
set -euo pipefail

# Grid evaluation for BART checkpoints with top-k and top-p sampling.
# - Distributes 6 jobs (k: 10/30/50, p: 0.85/0.9/0.95) across GPUs 0,2,3
# - Each GPU runs its 2 jobs sequentially; GPUs run in parallel
#
# Usage (relative paths OK; run from repo root):
#   # 1) Evaluate a fine-tuned checkpoint
#   bash scripts/eval_bart_grid.sh \
#     outputs/bart_base_e10_greedy/checkpoint-7980 \
#     facebook/bart-base \
#     outputs/eval_bart_ckpt7980_grid
#
#   # 2) Evaluate a pretrained HF model (no checkpoint)
#   bash scripts/eval_bart_grid.sh \
#     pretrained \
#     facebook/bart-base \
#     outputs/eval_bart_pretrained_grid
#
# Args:
#   1) CKPT_DIR        : path to checkpoint directory (contains model.safetensors)
#   2) TOKENIZER_DIR   : tokenizer source (default: facebook/bart-base)
#   3) OUT_ROOT        : output root directory (default: outputs/eval_bart_grid_<ckpt_basename>)
#   4) BATCH_SIZE      : per-device eval batch size (default: 16)
#   5) SEED            : random seed (default: 42)

ARG1=${1:-""}
TOKENIZER_DIR=${2:-facebook/bart-base}
# Determine mode
PRETRAINED_MODE=0
if [[ -z "${ARG1}" || "${ARG1}" == "pretrained" || "${ARG1}" == "-" ]]; then
  PRETRAINED_MODE=1
  CKPT_DIR=""
  CKPT_BASE="pretrained"
else
  PRETRAINED_MODE=0
  CKPT_DIR="${ARG1}"
  CKPT_BASE=$(basename "${CKPT_DIR}")
fi
# Determine repo root relative to this script (.. from scripts/)
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
OUT_ROOT=${3:-"outputs/eval_bart_grid_${CKPT_BASE}"}
BATCH_SIZE=${4:-16}
SEED=${5:-42}

mkdir -p "${OUT_ROOT}/logs"

run_eval() {
  local gpu="$1"; shift
  local name="$1"; shift
  local extra_args=("$@")

  echo "[$(date "+%F %T")] GPU=${gpu} START ${name}" | tee -a "${OUT_ROOT}/logs/${name}.log"
  (
    cd "${REPO_ROOT}"
    if [[ ${PRETRAINED_MODE} -eq 1 ]]; then
      CUDA_VISIBLE_DEVICES="${gpu}" ${PY_BIN:-python3} eval_bart.py \
        --pretrained "${TOKENIZER_DIR}" \
        --tokenizer_dir "${TOKENIZER_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        --seed "${SEED}" \
        "${extra_args[@]}" \
        --output_dir "${OUT_ROOT}/${name}"
    else
      CUDA_VISIBLE_DEVICES="${gpu}" ${PY_BIN:-python3} eval_bart.py \
        --model_dir "${CKPT_DIR}" \
        --tokenizer_dir "${TOKENIZER_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        --seed "${SEED}" \
        "${extra_args[@]}" \
        --output_dir "${OUT_ROOT}/${name}"
    fi
  ) >> "${OUT_ROOT}/logs/${name}.log" 2>&1
  echo "[$(date "+%F %T")] GPU=${gpu} DONE  ${name}" | tee -a "${OUT_ROOT}/logs/${name}.log"
}

# GPU 0: k=10, k=30 (sequential)
(
  run_eval 0 k10 --top_k 10
  run_eval 0 k30 --top_k 30
) &

# GPU 2: k=50, p=0.85 (sequential)
(
  run_eval 2 k50 --top_k 50
  run_eval 2 p085 --top_p 0.85
) &

# GPU 3: p=0.9, p=0.95 (sequential)
(
  run_eval 3 p090 --top_p 0.9
  run_eval 3 p095 --top_p 0.95
) &

wait
echo "[$(date "+%F %T")] All evaluations finished. Outputs -> ${OUT_ROOT}"


