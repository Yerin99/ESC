#!/usr/bin/env bash
# run_bart.sh
# Run bart.py with various generation settings.
# GPU allocation
#   GPU 1 : Beam Search experiments (sequentially)
#   GPU 2 : Top-k Sampling experiments (sequentially)
#   GPU 3 : Top-p Sampling experiments (sequentially)

set -u  # undefined vars are errors

LOG_DIR="logs/bart"
mkdir -p "$LOG_DIR"
echo "Logging experiments to $LOG_DIR"

EPOCHS=1
BATCH=16
EARLY_STOPPING_PATIENCE=None
BASE_ARGS="--epochs $EPOCHS --batch_size $BATCH --early_stopping_patience $EARLY_STOPPING_PATIENCE"

run_single_experiment() {
  local cuda_device=$1 run_name=$2 specific_args=$3
  local log_file="$LOG_DIR/${run_name}.log"
  local out_dir="outputs/${run_name}"

  echo "========== Starting ${run_name} on GPU ${cuda_device} =========="
  CUDA_VISIBLE_DEVICES=$cuda_device python bart.py \
    $BASE_ARGS \
    --output_dir "$out_dir" \
    $specific_args \
    >"$log_file" 2>&1

  local exit_code=$?
  if [ $exit_code -ne 0 ]; then
    echo "Run $run_name on GPU $cuda_device FAILED with code $exit_code. See $log_file" | tee -a "$LOG_DIR/failed_runs.log"
  else
    echo "Run $run_name on GPU $cuda_device COMPLETED." | tee -a "$LOG_DIR/success_runs.log"
  fi
}

beam_experiments() {
  echo -e "\n\n======== BEAM SEARCH EXPERIMENTS ON GPU 1 ========"
  local beams=(1 2 3 4 5 6 7 8)
  for beam in "${beams[@]}"; do
    run_single_experiment 1 "beam${beam}" "--num_beams ${beam}"
  done
}

topk_experiments() {
  echo -e "\n\n======== TOP-K SAMPLING EXPERIMENTS ON GPU 2 ========"
  local k_vals=(10 30 50)
  for k in "${k_vals[@]}"; do
    run_single_experiment 2 "sample_k${k}" "--top_k ${k}"
  done
}

topp_experiments() {
  echo -e "\n\n======== TOP-P SAMPLING EXPERIMENTS ON GPU 3 ========"
  local p_vals=(0.85 0.9 0.95)
  for p in "${p_vals[@]}"; do
    run_single_experiment 3 "sample_p${p}" "--top_p ${p}"
  done
}

# Launch three groups in parallel; each group runs sequentially on its GPU
beam_experiments &
topk_experiments &
topp_experiments &

wait
echo "All experiments finished."
