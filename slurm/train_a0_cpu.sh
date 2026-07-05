#!/bin/bash
#SBATCH --account=aip-nathanst
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G

module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements_drac.txt

# Read the (config_path, trial_no) pair for this array task from the tasks file.
# Usage:
#   sbatch --array=1-N train_a0_cpu.sh path/to/tasks.txt
#   sbatch train_a0_cpu.sh config/config.json 1   # legacy single-job mode
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

# ---------------------------------------------------------------------------
# Train-only stage; eval runs separately (slurm/eval_a0_cpu.sh). See the header
# comment in slurm/train_a0_gpu.sh for the self-resubmit (AUTO_RESUBMIT) design.
# ---------------------------------------------------------------------------
AUTO_RESUBMIT="${AUTO_RESUBMIT:-0}"
CHAIN_IDX="${CHAIN_IDX:-0}"
MAX_CHAIN="${MAX_CHAIN:-0}"
SELF_SCRIPT="${SELF_SCRIPT:-slurm/train_a0_cpu.sh}"
EVAL_SCRIPT="${EVAL_SCRIPT:-slurm/eval_a0_cpu.sh}"

REMAINING=$(python -m a0.train.check_progress "$CONFIG_FILE" "$TRIAL_NO")
echo "Iterations remaining: $REMAINING  (auto_resubmit=$AUTO_RESUBMIT, chain $CHAIN_IDX/$MAX_CHAIN)"

if [ "$AUTO_RESUBMIT" = "1" ] && [ -n "$SLURM_JOB_ID" ] && [ "${REMAINING:-0}" -gt 0 ]; then
    if [ "${AUTO_EVAL:-1}" = "1" ]; then
        echo "Queuing paired eval job (afterany:$SLURM_JOB_ID)."
        sbatch --dependency="afterany:$SLURM_JOB_ID" $EVAL_SBATCH_FLAGS \
            --export=ALL "$EVAL_SCRIPT" "$CONFIG_FILE" "$TRIAL_NO"
    fi
    if [ "$CHAIN_IDX" -lt "$MAX_CHAIN" ]; then
        NEXT_IDX=$((CHAIN_IDX + 1))
        echo "Queuing continuation train job (chain $NEXT_IDX/$MAX_CHAIN, afterany:$SLURM_JOB_ID)."
        sbatch --dependency="afterany:$SLURM_JOB_ID" $TRAIN_SBATCH_FLAGS \
            --export=ALL,CHAIN_IDX=$NEXT_IDX "$SELF_SCRIPT" "$CONFIG_FILE" "$TRIAL_NO"
    else
        echo "Max chain ($MAX_CHAIN) reached; not resubmitting a continuation."
    fi
fi

if [ "${REMAINING:-0}" -le 0 ]; then
    echo "Training already complete for this (config, trial); nothing to do."
    exit 0
fi

time python -m a0.train.alphazero "$CONFIG_FILE" "$TRIAL_NO"
