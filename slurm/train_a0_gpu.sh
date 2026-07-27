#!/bin/bash
#SBATCH --account=aip-nathanst
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-node=1

# CC module system isn't inherited by non-login `ssh host 'sbatch ...'`; source it.
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
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
    # Chained jobs are not array tasks, so without --output they'd land as
    # ./slurm-%j.out; route them next to the sweep's other logs instead.
    CHAIN_LOG_DIR="$(dirname "$(dirname "$CONFIG_FILE")")/slurm_logs"
    mkdir -p "$CHAIN_LOG_DIR"
    HP_ID="$(basename "$(dirname "$CONFIG_FILE")")"
    # Continuation to keep training past this job's wall clock. (No per-chunk
    # eval anymore — the non-player eval runs once at the terminal slot below,
    # and player eval / plateau sweep / combine are submitted manually.)
    if [ "$CHAIN_IDX" -lt "$MAX_CHAIN" ]; then
        NEXT_IDX=$((CHAIN_IDX + 1))
        echo "Queuing continuation train job (chain $NEXT_IDX/$MAX_CHAIN, afterany:$SLURM_JOB_ID)."
        sbatch --dependency="afterany:$SLURM_JOB_ID" $TRAIN_SBATCH_FLAGS \
            --output="$CHAIN_LOG_DIR/train_chain_${HP_ID}_t${TRIAL_NO}_%j.out" \
            --export=ALL,CHAIN_IDX=$NEXT_IDX "$SELF_SCRIPT" "$CONFIG_FILE" "$TRIAL_NO"
    else
        echo "Max chain ($MAX_CHAIN) reached; not resubmitting a continuation."
    fi
fi

if [ "${REMAINING:-0}" -le 0 ]; then
    # Terminal slot: training is complete. In the auto-resubmit chain this is the
    # pre-queued continuation job that finds nothing left to train — so it runs
    # the non-player eval once, on the GPU it was already allocated, then exits.
    # (Player eval, plateau sweep, and combine/aggregate are submitted manually.)
    echo "Training complete for ($CONFIG_FILE, trial $TRIAL_NO); running non-player eval."
    bash slurm/run_nonplayer_eval.sh "$CONFIG_FILE" "$TRIAL_NO"
    exit 0
fi

time python -m a0.train.alphazero "$CONFIG_FILE" "$TRIAL_NO"
