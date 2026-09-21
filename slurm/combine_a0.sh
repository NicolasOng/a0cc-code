#!/bin/bash
# NOTE: this is the Alliance/Compute Canada allocation this work ran under.
# Change it to your own, or override per-stage without editing this file via
# the "slurm" block in a sweep spec (sbatch CLI flags beat #SBATCH lines).
#SBATCH --account=aip-nathanst
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G

# CC module system is not inherited by non-login `ssh host 'sbatch ...'`; source
# it explicitly so this script works however it's submitted.
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements_drac.txt

# Read the per-HP config path for this array task from an hp_list file.
# Usage:
#   sbatch --array=1-N combine_a0.sh path/to/hp_list.txt
#   sbatch combine_a0.sh path/to/config.json   # legacy single-HP mode
ARG="${1:-config/config.json}"

if [ -n "$SLURM_ARRAY_TASK_ID" ] && [ -f "$ARG" ] && [[ "$ARG" == *.txt ]]; then
    CONFIG_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ARG")
    if [ -z "$CONFIG_FILE" ]; then
        echo "ERROR: no line ${SLURM_ARRAY_TASK_ID} in $ARG"
        exit 1
    fi
else
    CONFIG_FILE="$ARG"
fi

echo "Using configuration file: $CONFIG_FILE"

time python -m a0.eval.combine_merge "$CONFIG_FILE"
time python -m a0.eval.combine_plot "$CONFIG_FILE"

time python -m a0.eval3.combine "$CONFIG_FILE"

time python -m a0.eval.combine_summary "$CONFIG_FILE"
