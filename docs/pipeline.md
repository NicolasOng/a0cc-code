# The pipeline

One unit of work is a **`(config, trial)` pair**. The trial number is both the
random seed and the output subdirectory, so trials are independent and
parallelisable, and the spread across them is what the error bars measure.

Every stage takes the same two positional arguments:

```sh
python -m <stage.module> <config.json> <trial_no>
```

## Stages in order

```
  1. training          a0.train.alphazero
       |
  2. non-player eval   slurm/run_nonplayer_eval.sh   (6 stages, GPU)
       |
  3. player eval       a0.eval.player                (manual)
     plateau sweep     a0.eval.player_sweep          (manual)
       |
  4. combine           slurm/combine_a0.sh           (per HP, across trials)
```

Stages 1 and 2 are wired together and run hands-off. Stage 3 is submitted by
hand because it is the most expensive part of the whole pipeline. Stage 4 is
submitted by hand once the trials you want to merge have finished.

---

## 1. Training

```sh
python -m a0.train.alphazero config/config.json 1
```

The standard AlphaZero loop — self-play with MCTS, append to a replay buffer,
train the network, repeat — with the value target chosen by
`alternative_target` (see [configuration.md](configuration.md)). Each iteration
writes `training/model_<i>.pkl`.

**It resumes by itself.** On start it finds the most recent `model_<i>.pkl` and
continues to `training_iterations`. To extend a finished run, raise
`training_iterations` and resubmit; to restart, remove the checkpoints.

One caveat: **the optimizer is not checkpointed.** Adam moments are rebuilt on
resume, so `persist_optimizer_state` continuity does not survive a restart. A
run split across five 12-hour jobs is not bit-identical to the same run in one.

### Chaining past the wall clock

`slurm/train_a0_gpu.sh` with `AUTO_RESUBMIT=1` queues a continuation job
(`--dependency=afterany` on itself) before doing any work, so a timeout is
picked up automatically. The chain stops when
`python -m a0.train.check_progress` reports zero iterations remaining, or when
`CHAIN_IDX` hits `MAX_CHAIN`.

The job that finds nothing left to train is the **terminal slot**: it runs the
non-player eval on the GPU it already holds, then exits. That is how stage 2
gets triggered without a separate submission.

---

## 2. Non-player evaluation

`slurm/run_nonplayer_eval.sh <config> <trial>` runs six stages in order. It
assumes a venv is already active — the caller sets it up.

| Stage | What it does |
|---|---|
| `a0.eval3.training_data` | Walks the self-play data and scores it against ground truth: target accuracy, value bias, per-iteration game stats, the accuracy heatmaps |
| `a0.eval3.generate_datasets` | Builds the held-out evaluation sets (states seen in training, neighbours, random states) |
| `a0.eval3.dataset_evaluation` | Runs every checkpoint over those datasets → the accuracy curves |
| `a0.eval.model_diagnostics` | Per-iteration network probes: pre-activation stats, dead-neuron fraction, pre-tanh value spread. This is the collapse detector |
| `a0.eval3.plotting` | Renders the per-trial plots |
| `a0.eval3.extract_summary` | Scalar metrics → `summary.json` |

**This needs the solve file.** With `do_gt_evals: false` (i.e. 36-6) the
ground-truth analyses are skipped and only playing strength is measured.

---

## 3. Player evaluation

Both are submitted manually. Both play against `make_baseline` — a DIST /
Manhattan-heuristic MCTS player that reads **no** solve file, so these run on
boards with no ground truth.

### Win-rate curve — `a0.eval.player`

Win rate against the baseline over training iterations, at a single search
budget, taking every `player_eval_stride`-th checkpoint. The methodology used
for the thesis:

> 512 MCTS sims · `rollout_type=none` · `rollout_depth=-1` · `c_puct=0.25` ·
> 128 games per side (256 total) · stride 4

Roughly 20 min/checkpoint on 25-6, ~34 on 36-6, ~30 on 49-4.

**It is incremental**: checkpoints already scored in the results pkl are
skipped, never re-scored. That makes it resumable and chainable — and it is
also a trap. If the methodology changes, old entries are *kept*, and the curve
silently mixes two protocols. Delete the per-trial pkls before re-running under
changed settings.

### Plateau MCTS sweep — `a0.eval.player_sweep`

The last `plateau_sweep_num_checkpoints` converged checkpoints, played at
several search budgets (`plateau_sweep_mcts_samples`), averaged.

The point is that **heavy search masks value quality**: give MCTS 512
simulations and a mediocre value head still plays well. Low budgets strip that
away, so this is the measurement where value-target differences show up.

---

## 4. Combine

`slurm/combine_a0.sh <hp_list.txt>` merges across trials for one HP point:

```
a0.eval.combine_merge   →  merge per-trial series
a0.eval.combine_plot    →  plots with cross-trial confidence bands
a0.eval3.combine        →  the eval3 series and merged heatmaps
a0.eval.combine_summary →  merged summary
```

Trial directories are found by globbing `trial_*`. To merge a subset — for
example when some seeds are fully evaluated and others are not — set
`A0_COMBINE_TRIALS`:

```sh
A0_COMBINE_TRIALS=1-5 sbatch ... slurm/combine_a0.sh .../hp_list.txt
```

That filter applies to **all** series, player and non-player alike.

Merged output lands in `<arm>/eval/merged_*.pkl`. Warnings about missing series
are expected while a run is still training — only the player curves exist until
the terminal non-player eval fires. Re-combine after training completes.

---

## Where things land

```
<output_dir>/trial_<N>/
    training/     model_<i>.pkl, replay buffer checkpoints
    eval/         per-trial result series (.pkl) and summary.json
    plots/        per-trial figures
    logs/         run logs
    stats/  validations/  datasets/
```

`output_dir` comes from the config and already includes `trial_<N>` — every
directory above is created lazily, on first write, so a stage that does not
produce plots does not leave an empty `plots/`.

### The result files that matter

| Metric | Per-trial file | Needs GT |
|---|---|:--:|
| Win rate vs baseline | `player_evaluation_results.pkl` | no |
| Value-head accuracy | `seen_nd_eval.pkl`, `random_nd_eval.pkl` | yes |
| Policy-head accuracy | `seen_nt_eval.pkl`, `random_nt_eval.pkl` | yes |
| **Training-target accuracy** | `gamedata_alt_acc.pkl` | yes |
| Game outcome stats | `gamedata_stats.pkl` | no |
| Training losses/accuracies | `training_metrics.pkl` | no |

`nd` = value head, `nt` = policy head. `seen` = states that appeared in
training; `random` = uniformly sampled states.

`gamedata_alt_acc.pkl` is the central one: the accuracy of the **targets
AlphaZero trained on**, measured against ground truth. Do not confuse it with
`alt_targets_dataset_eval`, which compares the model to its own targets rather
than to the truth.

### Downloading results

Exclude `trial_*` when pulling a sweep off the cluster. Each trial keeps
per-checkpoint plots; recursing into them pulls tens of thousands of PNGs and
many gigabytes. Merged results are a few hundred megabytes.
