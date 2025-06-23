#!/bin/bash
#SBATCH --account=def-nathanst-ab
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements_drac.txt

time python -m a0.train.alpha_zero
time python -m a0.eval.generate_datasets
time python -m a0.eval.dataset_evaluation
