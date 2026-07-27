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

# Non-player eval only. Player eval (a0.eval.player) and the plateau sweep
# (a0.eval.player_sweep) are submitted manually; combine/aggregate roll-up is
# also manual (slurm/combine_a0.sh). This same stage set runs automatically from
# the terminal slot of slurm/train_a0_gpu.sh once training completes.
bash slurm/run_nonplayer_eval.sh "$CONFIG_FILE" "$TRIAL_NO"
