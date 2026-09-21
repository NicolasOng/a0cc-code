#!/bin/bash
# NOTE: this is the Alliance/Compute Canada allocation this work ran under.
# Change it to your own, or override per-stage without editing this file via
# the "slurm" block in a sweep spec (sbatch CLI flags beat #SBATCH lines).
#SBATCH --account=aip-nathanst
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements_drac.txt

# Usage:
#   sbatch aggregate_sweep.sh output/<sweep_name>/
SWEEP_DIR="${1}"
if [ -z "$SWEEP_DIR" ]; then
    echo "ERROR: usage: sbatch aggregate_sweep.sh <sweep_dir>"
    exit 1
fi

echo "Aggregating sweep: $SWEEP_DIR"

time python aggregate_sweep.py "$SWEEP_DIR"
time python plot_sweep.py "$SWEEP_DIR"
