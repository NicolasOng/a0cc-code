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

# Stage the solve file to node-local NVMe so GT lookups mmap a fast local copy.
bash slurm/stage_solve_data.sh "$CONFIG_FILE"

echo "Player-curve eval: config=$CONFIG_FILE trial=$TRIAL_NO"
time python -m a0.eval.player "$CONFIG_FILE" "$TRIAL_NO"
