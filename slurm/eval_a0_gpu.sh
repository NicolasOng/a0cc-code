#!/bin/bash
#SBATCH --account=aip-nathanst
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-node=1

module load cuda/12.6 cudnn/9.10 python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements_drac_cuda12.txt

# Read the (config_path, trial_no) pair for this array task from the tasks file.
# Usage:
#   sbatch --array=1-N eval_a0_gpu.sh path/to/tasks.txt
#   sbatch eval_a0_gpu.sh config/config.json 1   # legacy single-job mode
TASKS_FILE_OR_CONFIG="${1:-config/config.json}"

if [ -n "$SLURM_ARRAY_TASK_ID" ] && [ -f "$TASKS_FILE_OR_CONFIG" ] && [[ "$TASKS_FILE_OR_CONFIG" == *.txt ]]; then
    LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS_FILE_OR_CONFIG")
    if [ -z "$LINE" ]; then
        echo "ERROR: no line ${SLURM_ARRAY_TASK_ID} in $TASKS_FILE_OR_CONFIG"
        exit 1
    fi
    CONFIG_FILE=$(echo "$LINE" | awk '{print $1}')
    TRIAL_NO=$(echo "$LINE" | awk '{print $2}')
else
    CONFIG_FILE="$TASKS_FILE_OR_CONFIG"
    TRIAL_NO="${2}"
fi

echo "Using configuration file: $CONFIG_FILE"
echo "Using trial number: $TRIAL_NO"

time python -m a0.eval3.training_data "$CONFIG_FILE" "$TRIAL_NO"
time python -m a0.eval3.generate_datasets "$CONFIG_FILE" "$TRIAL_NO"
time python -m a0.eval3.dataset_evaluation "$CONFIG_FILE" "$TRIAL_NO"
time python -m a0.eval.model_diagnostics "$CONFIG_FILE" "$TRIAL_NO"
time python -m a0.eval.player "$CONFIG_FILE" "$TRIAL_NO"
time python -m a0.eval3.plotting "$CONFIG_FILE" "$TRIAL_NO"
time python -m a0.eval3.extract_summary "$CONFIG_FILE" "$TRIAL_NO"

# ---------------------------------------------------------------------------
# Roll-up (opt-in via RUN_ROLLUP=1): after this trial's eval, refresh the
# per-setting combined plots and — if SWEEP_DIR is set — the whole-sweep
# aggregate, so they stay current as training/eval progress. A *blocking*
# flock serializes concurrent array tasks that would otherwise write the same
# combine/aggregate outputs at once. combine writes into the per-setting dir
# (dirname of the config); aggregate writes into SWEEP_DIR.
# ---------------------------------------------------------------------------
if [ "${RUN_ROLLUP:-0}" = "1" ]; then
    HP_DIR=$(dirname "$CONFIG_FILE")

    combine_rollup() {
        time python -m a0.eval.combine_merge "$CONFIG_FILE"
        time python -m a0.eval.combine_plot "$CONFIG_FILE"
        time python -m a0.eval3.combine "$CONFIG_FILE"
        time python -m a0.eval.combine_summary "$CONFIG_FILE"
    }
    echo "Roll-up: combine for $CONFIG_FILE (blocking lock $HP_DIR/.combine.lock)"
    ( flock 9; combine_rollup ) 9>"$HP_DIR/.combine.lock"

    if [ -n "$SWEEP_DIR" ]; then
        aggregate_rollup() {
            time python aggregate_sweep.py "$SWEEP_DIR"
            time python plot_sweep.py "$SWEEP_DIR"
        }
        echo "Roll-up: aggregate for $SWEEP_DIR (blocking lock $SWEEP_DIR/.aggregate.lock)"
        ( flock 9; aggregate_rollup ) 9>"$SWEEP_DIR/.aggregate.lock"
    fi
fi
