# Experiments

The final thesis campaign was **ten settings**, run between 2026-08-02 and
2026-08-28. This page maps each to the sweep spec that produced it, so a result
can be traced back to the command that generated it.

## The ten settings

| Setting | Spec | Board | Arms | Iters | GT | What it asks |
|---|---|:--:|--:|--:|:--:|---|
| `td_lambda36-6` | [sweeps/td_lambda36-6.json](../sweeps/td_lambda36-6.json) | 36-6 | 6 | 200 | ✗ | The main value-target sweep: λ ∈ {0, .5, .75, .9, 1} plus interpolated |
| `td_lambda49-4` | [sweeps/td_lambda49-4.json](../sweeps/td_lambda49-4.json) | 49-4 | 6 | 200 | ✓ | Same sweep on the largest solved board |
| `td_lambda25-6` *(eval-only)* | — | 25-6 | 5 | 200 | ✓ | The λ sweep on 25-6. Trained earlier; this campaign only re-evaluated it |
| `td_lambda16-3` *(eval-only)* | [sweeps/td_lambda16-3.json](../sweeps/td_lambda16-3.json) | 16-3 | 5 | 200 | ✓ | The λ sweep on the smallest board |
| `td_interp25-6` | [sweeps/td_interp25-6.json](../sweeps/td_interp25-6.json) | 25-6 | 1 | 200 | ✓ | The interpolated-λ arm for 25-6, whose five λ arms were already trained |
| `refresh36-6` | [sweeps/refresh36-6.json](../sweeps/refresh36-6.json) | 36-6 | 3 | 100 | ✗ | Target refresh on TD(0) at K ∈ {2, 4, 8} |
| `td0refresh8ep36-6` | [sweeps/td0refresh8ep36-6.json](../sweeps/td0refresh8ep36-6.json) | 36-6 | 1 | 50 | ✗ | **The control for the above.** 8 passes with targets *not* recomputed |
| `td0anneal25-6` | [sweeps/td0anneal25-6.json](../sweeps/td0anneal25-6.json) | 25-6 | 1 | 200 | ✓ | TD(0) refresh with K annealed 8→1 over the first 25 iterations |
| `td0anneal36-6` | [sweeps/td0anneal36-6.json](../sweeps/td0anneal36-6.json) | 36-6 | 1 | 200 | ✗ | Same, 36-6 |
| `td0anneal49-4` | [sweeps/td0anneal49-4.json](../sweeps/td0anneal49-4.json) | 49-4 | 1 | 200 | ✓ | Same, 49-4 |

GT ✓ means a solve file exists, so accuracy curves and heatmaps are available.
**36-6 has none** — those three settings are measured by playing strength only.
See [solve-data.md](solve-data.md).

## Reading the three groups

**The λ sweep** (`td_lambda*`, `td_interp*`) is the core comparison: Monte Carlo
targets (λ=1, stock AlphaZero) against TD(λ) at decreasing λ, on four boards.
Interpolated λ anneals from Monte Carlo toward TD over the first 25 iterations.

**The refresh group** (`refresh36-6`, `td0anneal*`) asks whether repeatedly
recomputing targets within an iteration — fitted value iteration — helps, and
whether a high refresh count is only useful early.

**The control** (`td0refresh8ep36-6`) is the one that makes the refresh group
interpretable. It runs the same 8 passes with `refresh_recompute_targets:
false`, so the targets are computed once and reused. Any gain the refresh arms
show over *this* is refreshing; a gain over the plain baseline could just be
extra epochs.

## Seeds

The specs say `"num_trials": 4`. That is the **initial batch**, not the final
count. All ten settings were finalised at **12 seeds**, added in batches using
the procedure in [sweeps.md](sweeps.md#extending-a-finished-sweep).

The two eval-only settings are the exception: 25 seeds exist for them, combined
as `A0_COMBINE_TRIALS=1-25` for ground-truth accuracy and `1-15` for the player
curve.

To reproduce at full seed count, either set `num_trials` to 12 up front or use
the add-seeds procedure. Do not re-run `submit_sweep.py` against a sweep that
already has results.

## Methodology held constant

Everything below was uniform across the campaign. Deviating from it makes
results non-comparable — in particular the player-eval settings, which were
changed partway through an earlier campaign and produced a curve with a visible
step in it.

- **Resources**: 16 cores, `--mem=64G`, 1 GPU, `num_workers=24`. The specs carry
  no `slurm` block so the job scripts' own values win.
- **Player eval**: 512 MCTS sims, `rollout_type=none`, `rollout_depth=-1`,
  `c_puct=0.25`, 128 games/side (256 total), stride 4.
- **Baseline opponent**: DIST/Manhattan (`make_baseline`), which reads no solve
  file — which is what lets 36-6 be evaluated at all.
- **Boards**: 16-3 = 4×4/3pc · 25-6 = 5×5/6pc · 36-6 = 6×6/6pc · 49-4 = 7×7/4pc.

Approximate cost per checkpoint for the player eval: ~20 min on 25-6, ~34 on
36-6, ~30 on 49-4. Training: ~21 min/iteration on 25-6, ~27 on 36-6, ~44 on
49-4. A 200-iteration 49-4 arm is therefore several days of GPU time before any
evaluation.

## Running one

```sh
python3 submit_sweep.py sweeps/td_lambda36-6.json --dry-run   # inspect
python3 submit_sweep.py sweeps/td_lambda36-6.json             # submit
```

It prints the follow-up player-eval and combine commands when it finishes. See
[sweeps.md](sweeps.md).

To try the machinery cheaply first, point a spec at `config/config-16-3.json`
and drop `training_iterations` — 16-3 has a 79 KB solve file and runs quickly.

## Other specs in `sweeps/`

| Spec | Status |
|---|---|
| `example.json` | Template showing `grid`, `points` and the `slurm` block. Not an experiment |
| `thesis_gt.json` | Ground-truth-target runs (`alternative_target: "gt"`), balanced vs unbalanced. Earlier work — GT targets produced weak players, traced to flat GT policy targets and depthless values rather than a model defect |
| `td_lambda.json` | The original 25-6 λ sweep at 25 seeds, superseded by the per-board `td_lambda*-*.json` specs. It named a `base_config` that no longer exists (`config/config-200.json`) and now points at `config/config.json`. **It therefore inherits `training_iterations: 50`, where the campaign ran this setting at 200** — add `"training_iterations": 200` to the spec if you re-run it |

## Where the analysis lives

The dated investigation notes in the investigations repo are the lab notebook
for these runs — what was tried, what failed, and why methodology settled where
it did. `2026-08-02_thesis-final-runs` holds the campaign dashboard and the full
operational log, including the per-setting job ledger.
