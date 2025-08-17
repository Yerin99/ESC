#!/usr/bin/env bash
# run_all.sh
# Sequentially executes all major experiment scripts.
# 1. Runs all BART experiments and waits for completion.
# 2. Runs all BlenderBot experiments and waits for completion.

set -e # Exit immediately if a command exits with a non-zero status.

# Make sure we are in the project root directory
cd "$(dirname "$0")/.."

echo "======== STARTING BART EXPERIMENTS (run_bart.sh) ========"
bash scripts/run_bart.sh
echo "======== BART EXPERIMENTS FINISHED ========"

echo -e "\n\n======== STARTING BLENDERBOT EXPERIMENTS (run_blenderbot.sh) ========"
bash scripts/run_blenderbot.sh
echo "======== BLENDERBOT EXPERIMENTS FINISHED ========"

echo -e "\n\nAll experiments completed successfully."
