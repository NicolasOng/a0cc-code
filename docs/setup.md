# Setup

## The requirements files do not install from PyPI as written

Both requirements files pin versions with a `+computecanada` local suffix:

```
jax==0.4.35+computecanada
flax==0.10.4+computecanada
numpy==2.2.2+computecanada
```

Those builds exist only in the Digital Research Alliance of Canada wheelhouse,
which is what `pip install --no-index` reads on the cluster. **Off-cluster, pip
will not find them.** Install the same version numbers with the suffix stripped:

```sh
python3.11 -m venv .venv && source .venv/bin/activate
pip install jax==0.4.35 jaxlib==0.4.34 flax==0.10.4 optax==0.2.5 \
            numpy==2.2.2 scipy==1.15.1 matplotlib==3.10.0 dill==0.4.0 pytest
```

Python 3.11 is what the cluster jobs load and what the code was developed
against. The type annotations use `X | None` syntax throughout, so 3.10 is the
practical floor.

## Which requirements file is which

|  | File | Difference |
|---|---|---|
| CPU | `requirements_drac.txt` | Base set |
| GPU | `requirements_drac_cuda12.txt` | Adds `jax_cuda12_pjrt` and `jax_cuda12_plugin` |

That is the *only* difference between them — everything else is pinned
identically. Stages that touch a GPU (training, dataset evaluation, player eval,
the plateau sweep) use the CUDA file; the combine stage, which only merges
pickles, uses the CPU one.

## On the cluster

Every job script builds its own throwaway virtualenv in `$SLURM_TMPDIR`:

```sh
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load cuda/12.6 cudnn/9.10 python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r requirements_drac_cuda12.txt
```

Two details that are easy to lose:

- **The `source /cvmfs/...` line is load-bearing.** The module system is not
  inherited when jobs are submitted non-interactively (`ssh host 'sbatch ...'`),
  and without it `module load` is not a command. Every script sources it
  explicitly so it works however it is submitted.
- **The venv is rebuilt per job**, on node-local storage. This costs a minute or
  two per job but means no shared environment to drift.

For an interactive session on a login node, `slurm/create_python_env.sh` builds
the same environment and prints the `source` line to activate it.

### Allocation

The job scripts carry `#SBATCH --account=aip-nathanst`, the allocation this work
ran under. Change it to your own. You can override it without editing the files
via the `slurm` block in a sweep spec — sbatch CLI flags beat in-script `#SBATCH`
lines:

```json
{ "slurm": { "train": { "account": "def-yourpi" } } }
```

### Resources

The final runs used **16 cores, `--mem=64G`, 1 GPU, `num_workers=24`**, baked
into the slurm scripts and config. Self-play is latency-bound rather than
CPU- or GPU-saturated, so throughput tracks worker count more than core count;
16 cores with 24 workers measured no worse than 32 cores and bills at half the
rate.

## Running scripts

Modules are invoked with `-m` from the repo root:

```sh
python -m a0.train.alphazero config/config.json 1
```

`config.py` reads its config path from `sys.argv[1]` **at import time**, so the
config path is a positional argument to every stage, not a flag. The trial
number is `sys.argv[2]`.

Files under `scripts/` are run as paths rather than modules (their dated names
are not valid Python identifiers), and most need the repo root on the path:

```sh
PYTHONPATH=. python scripts/2026-05-29_single_agent_bfs.py --board-size 4 --num-pieces 3
```

## Tests

```sh
pytest
```

26 tests, a few seconds, no GPU and no solve file needed. `conftest.py`
normalises `sys.argv` before collection so the import-time config read above
does not break the run.
