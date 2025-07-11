#!/bin/bash
#SBATCH --account=def-nathanst-ab
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements_drac.txt

# Set the configuration file path, defaulting to config/config.json if not provided
CONFIG_FILE="${1:-config/config.json}" 

time python -m a0.train.dataset "$CONFIG_FILE"
