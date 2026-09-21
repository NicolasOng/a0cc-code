#!/bin/bash
# NOTE: this is the Alliance/Compute Canada allocation this work ran under.
# Change it to your own, or override per-stage without editing this file via
# the "slurm" block in a sweep spec (sbatch CLI flags beat #SBATCH lines).
#SBATCH --account=aip-nathanst
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G

module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements_drac.txt

# Read the (config_path, trial_no) pair for this array task from the tasks file.
# Usage:
#   sbatch --array=1-N eval_a0_cpu.sh path/to/tasks.txt
#   sbatch eval_a0_cpu.sh config/config.json 1   # legacy single-job mode
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

# Roll-up (opt-in via RUN_ROLLUP=1): refresh per-setting combined plots and, if
# SWEEP_DIR is set, the whole-sweep aggregate. Blocking flock serializes
# concurrent array tasks. See slurm/eval_a0_gpu.sh for the full rationale.
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
