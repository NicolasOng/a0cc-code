#!/bin/bash
#SBATCH --account=aip-nathanst
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements_drac.txt

# Set the configuration file path, defaulting to config/config.json if not provided
CONFIG_FILE="${1:-config/config.json}"
TRIAL_NO="${2}"

echo "Using configuration file: $CONFIG_FILE"
echo "Using trial number: $TRIAL_NO"

time python -m scripts.checkpointing_test "$CONFIG_FILE" "$TRIAL_NO"

sbatch drac_job5.sh "$CONFIG_FILE" "$TRIAL_NO"

echo "Checkpointing test job submitted."
