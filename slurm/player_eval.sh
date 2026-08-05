#!/bin/bash
#SBATCH --account=aip-nathanst
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1

# Manual player-eval WIN-RATE CURVE (a0.eval.player): win rate vs the baseline
# over training iterations at the single player_eval_mcts_samples budget, strided
# by player_eval_stride. Resumable (skips already-scored checkpoints), so it can
# be chained with --dependency=afterany to run past a 12h wall clock.
# The plateau MCTS sweep is a separate script (slurm/player_sweep.sh).
#
# CC module system isn't inherited by non-login `ssh host 'sbatch ...'`; source it.
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load cuda/12.6 cudnn/9.10 python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements_drac_cuda12.txt

# Reads a (config_path, trial_no) pair per array task from a tasks file. Usage:
#   sbatch --array=1-N slurm/player_eval.sh path/to/tasks.txt
#   sbatch slurm/player_eval.sh config/config.json 1   # single-trial mode
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

# NOTE: no solve-data staging here — the player curve plays vs the DIST/Manhattan
# baseline (make_baseline) and never reads the solve file, so staging would just
# waste time/space (esp. the 15G 49-4 file, once per chain link).

# Self-resubmitting chain (opt-in via AUTO_RESUBMIT=1): the eval is resumable
# (skips already-scored checkpoints), so if the full curve doesn't fit in this
# job's wall clock we queue a continuation now (afterany on THIS job) that picks
# up where we left off. The chain self-terminates when no checkpoints remain
# (or CHAIN_IDX hits MAX_CHAIN). Mirrors slurm/train_a0_gpu.sh.
AUTO_RESUBMIT="${AUTO_RESUBMIT:-0}"
CHAIN_IDX="${CHAIN_IDX:-0}"
MAX_CHAIN="${MAX_CHAIN:-0}"
SELF_SCRIPT="${SELF_SCRIPT:-slurm/player_eval.sh}"

# tail -1: player_progress prints the count last; when the results pkl is absent
# (fresh/deleted) load_series emits a "series not found" WARNING to stdout first,
# which would otherwise poison the integer test below and break the chain.
REMAINING=$(python -m a0.eval.player_progress "$CONFIG_FILE" "$TRIAL_NO" | tail -1)
echo "Player-eval checkpoints remaining: $REMAINING  (auto_resubmit=$AUTO_RESUBMIT, chain $CHAIN_IDX/$MAX_CHAIN)"

if [ "$AUTO_RESUBMIT" = "1" ] && [ -n "$SLURM_JOB_ID" ] && [ "${REMAINING:-0}" -gt 0 ] && [ "$CHAIN_IDX" -lt "$MAX_CHAIN" ]; then
    # Chained jobs aren't array tasks, so route their logs next to the sweep's.
    CHAIN_LOG_DIR="$(dirname "$(dirname "$CONFIG_FILE")")/slurm_logs"
    mkdir -p "$CHAIN_LOG_DIR"
    HP_ID="$(basename "$(dirname "$CONFIG_FILE")")"
    NEXT_IDX=$((CHAIN_IDX + 1))
    echo "Queuing player-eval continuation (chain $NEXT_IDX/$MAX_CHAIN, afterany:$SLURM_JOB_ID)."
    sbatch --dependency="afterany:$SLURM_JOB_ID" \
        --output="$CHAIN_LOG_DIR/player_eval_chain_${HP_ID}_t${TRIAL_NO}_%j.out" \
        --export=ALL,CHAIN_IDX=$NEXT_IDX "$SELF_SCRIPT" "$CONFIG_FILE" "$TRIAL_NO"
fi

if [ "${REMAINING:-0}" -le 0 ]; then
    echo "Player eval complete for ($CONFIG_FILE, trial $TRIAL_NO); nothing to do."
    exit 0
fi

echo "Player-curve eval: config=$CONFIG_FILE trial=$TRIAL_NO"
time python -m a0.eval.player "$CONFIG_FILE" "$TRIAL_NO"
