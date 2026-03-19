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

# Set the configuration file path, defaulting to config/config.json if not provided
CONFIG_FILE="${1:-config/config.json}"
TRIAL_NO="${2}"

echo "Using configuration file: $CONFIG_FILE"
echo "Using trial number: $TRIAL_NO"

time python -m a0.train.alphazero "$CONFIG_FILE" "$TRIAL_NO"
time python -m a0.eval.generate_datasets "$CONFIG_FILE" "$TRIAL_NO"
time python -m a0.eval.dataset_evaluation "$CONFIG_FILE" "$TRIAL_NO"
time python -m a0.eval.training_data "$CONFIG_FILE" "$TRIAL_NO"
time python -m a0.eval.plotting "$CONFIG_FILE" "$TRIAL_NO"
