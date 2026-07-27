#!/bin/bash
# Non-player eval stages for one (config, trial): dataset generation + dataset
# evaluations + model diagnostics + plotting + per-trial summary. Player eval
# (a0.eval.player) and the plateau sweep (a0.eval.player_sweep) are run
# separately/manually and are NOT included here.
#
# Assumes a Python venv with the deps is ALREADY active (the caller sets it up).
# Used by:
#   - slurm/eval_a0_gpu.sh              (manual / standalone non-player eval)
#   - slurm/train_a0_gpu.sh terminal    (auto: runs once training completes)
#
# Usage: bash slurm/run_nonplayer_eval.sh <config.json> <trial_no>
set -u
CONFIG_FILE="$1"
TRIAL_NO="$2"

time python -m a0.eval3.training_data      "$CONFIG_FILE" "$TRIAL_NO"
time python -m a0.eval3.generate_datasets  "$CONFIG_FILE" "$TRIAL_NO"
time python -m a0.eval3.dataset_evaluation "$CONFIG_FILE" "$TRIAL_NO"
time python -m a0.eval.model_diagnostics   "$CONFIG_FILE" "$TRIAL_NO"
time python -m a0.eval3.plotting           "$CONFIG_FILE" "$TRIAL_NO"
time python -m a0.eval3.extract_summary    "$CONFIG_FILE" "$TRIAL_NO"
