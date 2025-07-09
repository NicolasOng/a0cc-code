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

time python -m python2.parallel.jax_async_play
time python3 -m python2.evaluations.state_evaluation.data.make_dataset
time python3 -m python2.evaluations.state_evaluation.static_states.evaluate_outcome
