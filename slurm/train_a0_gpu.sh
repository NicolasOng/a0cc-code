#!/bin/bash
#SBATCH --account=aip-nathanst
#SBATCH --time=48:00:00
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
#   sbatch --array=1-N train_a0_gpu.sh path/to/tasks.txt
#   sbatch train_a0_gpu.sh config/config.json 1   # legacy single-job mode
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
# Train-only stage. Evaluation runs in a separate job (slurm/eval_a0_gpu.sh),
# wired up by submit_sweep.py.
#
# Self-resubmitting training (opt-in via AUTO_RESUBMIT=1):
#   a0.train.alphazero already resumes from the most recent model_<i>.pkl and
#   trains up to config.training_iterations. To train past this job's wall clock
#   (e.g. a 1-day job that should "keep going"), we queue a CONTINUATION job now
#   (dependency=afterany on THIS job) so a timeout-kill is picked up
#   automatically, plus a paired EVAL job so each training chunk gets evaluated.
#   The chain self-terminates when training is complete (0 iterations remaining)
#   or CHAIN_IDX reaches MAX_CHAIN (safety cap). To just extend an existing run,
#   bump config.training_iterations and resubmit.
#
# NOTE: resuming re-initializes the optimizer — Adam moments are not
# checkpointed — so persist_optimizer_state continuity is lost across restarts.
# ---------------------------------------------------------------------------
AUTO_RESUBMIT="${AUTO_RESUBMIT:-0}"
CHAIN_IDX="${CHAIN_IDX:-0}"
MAX_CHAIN="${MAX_CHAIN:-0}"
SELF_SCRIPT="${SELF_SCRIPT:-slurm/train_a0_gpu.sh}"
EVAL_SCRIPT="${EVAL_SCRIPT:-slurm/eval_a0_gpu.sh}"

REMAINING=$(python -m a0.train.check_progress "$CONFIG_FILE" "$TRIAL_NO")
echo "Iterations remaining: $REMAINING  (auto_resubmit=$AUTO_RESUBMIT, chain $CHAIN_IDX/$MAX_CHAIN)"

if [ "$AUTO_RESUBMIT" = "1" ] && [ -n "$SLURM_JOB_ID" ] && [ "${REMAINING:-0}" -gt 0 ]; then
    # Paired eval of this chunk's checkpoints (runs once this job ends, finished
    # or timed out). $EVAL_SBATCH_FLAGS is optional extra sbatch flags.
    if [ "${AUTO_EVAL:-1}" = "1" ]; then
        echo "Queuing paired eval job (afterany:$SLURM_JOB_ID)."
        sbatch --dependency="afterany:$SLURM_JOB_ID" $EVAL_SBATCH_FLAGS \
            --export=ALL "$EVAL_SCRIPT" "$CONFIG_FILE" "$TRIAL_NO"
    fi
    # Continuation to keep training past this job's wall clock.
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
