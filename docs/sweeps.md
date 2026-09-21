# Sweeps

A sweep is a grid of hyperparameter points × trials. `submit_sweep.py` expands
the grid, materialises one self-contained config per point, writes the task
files, and submits the training chain.

```sh
python3 submit_sweep.py sweeps/td_lambda36-6.json
python3 submit_sweep.py sweeps/td_lambda36-6.json --dry-run
```

It is **stdlib-only on purpose** — no jax, no numpy — so it runs on a login node
without activating a venv.

`--dry-run` still writes the sweep directory and its per-arm configs; what it
skips is the `sbatch` call, printing the commands instead. So it is safe to
inspect a new sweep with, but it is not a no-op on an existing one.

## Spec format

```json
{
    "name": "td_lambda36-6",
    "base_config": "config/config-36-6.json",
    "num_trials": 4,
    "auto_resubmit": true,
    "grid":   { "td_lambda": [0.0, 0.5, 0.75, 0.9, 1.0] },
    "points": [ { "alternative_target": "interpolated_td_lambda" } ]
}
```

Required: `name`, `base_config`, `num_trials`, and at least one of `grid` /
`points`.

- **`grid`** — cartesian product of the axes.
- **`points`** — explicit configurations, appended to whatever the grid
  produced and deduplicated by hash. Useful for adding one baseline or odd-arm
  without multiplying it through every other axis.
- **Coupled axes** — a grid axis whose values are *dicts* varies its named
  parameters together, and the key is just a label. This is how a block of
  settings gets held constant across a swept axis without exploding the product:

  ```json
  "grid": {
      "config":    [ { "alternative_target": "td_lambda", "training_iterations": 200 } ],
      "td_lambda": [0.0, 0.5, 1.0]
  }
  ```

  Three points, not six.

- **`slurm`** — per-stage `#SBATCH` overrides, passed as sbatch CLI flags, which
  beat in-script directives:

  ```json
  "slurm": { "train": { "time": "24:00:00", "cpus-per-task": 16 } }
  ```

  The thesis sweeps deliberately carry **no** `slurm` block, so the resource
  settings baked into the job scripts win.

## What it writes

```
sweep-output/<name>/
    <hp_hash>/config.json      one self-contained config per HP point
    <hp_hash>/trial_<n>/...    results land here
    tasks.txt                  "<config_path> <trial>" per line  (arms × trials)
    hp_list.txt                one config path per line          (arms)
    hp_index.txt               hash → the settings it stands for
    sweep.json                 a copy of the spec
    slurm_logs/
```

Each per-arm `config.json` is **fully materialised** — defaults, base config and
overrides merged at submission time. It does not reference the spec, so a
finished arm stays interpretable after the spec changes.

The hash is derived from the HP overrides, so the same point keeps the same
directory across resubmissions.

## Chaining and what is manual

`submit_sweep.py` submits **only the training chain**. With
`auto_resubmit: true`, each training job queues its own continuation until
training completes or `max_resubmissions` is hit, and the terminal job runs the
non-player eval.

When it finishes it prints the follow-up commands. They are manual because they
are expensive and you rarely want all of them:

```sh
# player-curve eval
sbatch --array=1-<tasks> --export=ALL,AUTO_RESUBMIT=1,CHAIN_IDX=0,MAX_CHAIN=20 \
  --output=sweep-output/<sw>/slurm_logs/player_eval_%A_%a.out \
  slurm/player_eval.sh sweep-output/<sw>/tasks.txt

# plateau MCTS sweep
sbatch --array=1-<tasks> \
  --output=sweep-output/<sw>/slurm_logs/player_sweep_%A_%a.out \
  slurm/player_sweep.sh sweep-output/<sw>/tasks.txt

# combine, one task per arm
sbatch --array=1-<arms> \
  --output=sweep-output/<sw>/slurm_logs/combine_%A_%a.out \
  slurm/combine_a0.sh sweep-output/<sw>/hp_list.txt
```

`<tasks>` = arms × trials (the line count of `tasks.txt`); `<arms>` = the line
count of `hp_list.txt`.

Note `submit_sweep.py` reports job ids as `-> job N`, not sbatch's usual
"Submitted batch job N".

## Local mode

Runs the same stages sequentially on the current machine, no slurm:

```sh
python3 submit_sweep.py sweeps/foo.json --local
python3 submit_sweep.py sweeps/foo.json --local --stages a0.eval3.plotting \
        --skip-combine --skip-aggregate
```

`--stages` is the practical use: re-running one stage over an existing sweep —
regenerating plots after a plotting change, say — without retraining.

## Extending a finished sweep

**To train further**, raise `training_iterations` and resubmit with
`--reuse-configs`. Training resumes from the latest checkpoint.

**To add seeds, do not re-run `submit_sweep.py`.** It rewrites `tasks.txt` for
trials `1..N` and would re-touch finished ones. Instead:

1. Write a supplemental tasks file listing `<hp_config> <trial>` for the new
   trial numbers only, across all arms.
2. Submit training against that file, with the same `AUTO_RESUBMIT` exports.
3. Player-eval the new trials the same way.
4. Re-combine with a wider range (`A0_COMBINE_TRIALS=1-8`). Combine globs all
   `trial_*`, so the confidence intervals absorb the new seeds.

## Things that will bite you

- **Clear per-trial player pkls before an eval-only re-run if the methodology
  changed.** `a0.eval.player` never re-scores a checkpoint it has already done,
  so old and new protocols merge into one curve with a step in it.
- **Combine warnings are benign while a run is still training** — GT, accuracy
  and summary series do not exist until the terminal non-player eval runs.
- **Player-eval chains self-terminate when they catch up** to the checkpoints
  that existed when they started, so a run that is still training needs a final
  catch-up pass after training ends.
- **Exclude `trial_*` when downloading.** See
  [pipeline.md](pipeline.md#downloading-results).
