#!/bin/bash
#SBATCH --account=def-nathanst-ab
#SBATCH --time=70:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements_drac.txt

# Set the configuration file path, defaulting to config/config.json if not provided
CONFIG_FILE="${1:-config/config.json}" 

time python -m scripts.evaluate_mcts "$CONFIG_FILE"
