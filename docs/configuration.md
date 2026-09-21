# Configuration

[`config.py`](../config.py) carries a documented attribute for every setting.
This page covers the parts that are not obvious from reading it: how values
resolve, the import-time behaviour that surprises people, and which knobs
actually matter.

## How a value resolves

Three layers, each overriding the last:

```
config/config.json          the full default set — every key lives here
        ↓
<your config>.json          only the keys it changes
        ↓
derived / trial adjustments num_spots, output_dir += trial_<N>/
```

This is why `config/config-16-3.json` is four lines:

```json
{
    "board_size": 4,
    "num_pieces": 3,
    "solve_data": "input/solvedata/CC-SOLVE-BASELINE-16-3.dat"
}
```

Everything else comes from the default. `config/config.json` is therefore **not
an example file** — it is the schema, and deleting a key from it breaks every
config that relies on the default.

Per-arm sweep configs are the exception: `submit_sweep.py` materialises them
fully merged, so they are self-contained.

## Two behaviours to know

**The config path is read from `sys.argv[1]` at import time.**

```python
config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.json"
config = Config(config_path, "config/config.json", trial_num)
```

Importing anything that pulls in `config` therefore parses `argv` immediately.
That is why every stage takes the config as a positional argument rather than a
flag, and why `conftest.py` normalises `argv` before pytest collects — otherwise
collection dies trying to parse a test path as JSON.

**Output directories are created lazily.** `training_dir`, `eval_dir`,
`plot_dir` and friends are properties that `mkdir` on first access. A stage that
writes no plots leaves no empty `plots/`. Reading one of these attributes is a
side effect, not a lookup.

## The settings that matter

### Board

| Key | Notes |
|---|---|
| `board_size`, `num_pieces` | `num_spots = board_size²`, so "25-6" is `board_size: 5, num_pieces: 6` |
| `solve_data` | Path to the solve file — see [solve-data.md](solve-data.md) |
| `do_gt_evals` | `false` disables every ground-truth analysis. Required on 36-6, which has no solve file |
| `backwards_moves`, `sideways_moves`, `repeats_for_draw` | Rules. The thesis used no backwards moves, sideways allowed, 6 repeats = draw |

### Value targets — the subject of the thesis

| Key | Notes |
|---|---|
| `alternative_target` | `gt` \| `gt_value` \| `gt_next_value` \| `td_0` \| `td_lambda` \| `interpolated_td_lambda` \| `normal`. `normal` is stock AlphaZero (Monte Carlo) |
| `td_lambda` | λ for `td_lambda`. λ=1 is the Monte Carlo return, λ=0 is one-step TD |
| `interpolated_td_lambda_iterations` | Warm-up length for the interpolated variant; `null` → 25% of `training_iterations` |

### Target refresh (fitted value iteration)

| Key | Notes |
|---|---|
| `use_target_refresh` | Enables the refresh path instead of plain self-play targets |
| `num_target_refreshes` | K, refresh passes per iteration |
| `refresh_k_decay_iterations` | Decay K→1 linearly over this many iterations; `null` = constant K |
| `refresh_recompute_targets` | `true` = a real refresh each pass. **`false` is the control**: K training passes on frozen targets, i.e. extra epochs without refreshing |

That last flag is what separates "refreshing helped" from "you just trained
longer".

### Search

Training search and evaluation search are configured **separately and
deliberately**, so the evals can measure the value head rather than the search
that hides it:

| Training | Player eval |
|---|---|
| `mcts_samples` | `player_eval_mcts_samples` |
| `rollout_type`, `rollout_depth` | `player_eval_rollout_type`, `player_eval_rollout_depth` |
| `c_puct` | `player_eval_c_puct` |

The thesis trained with random rollouts and `c_puct 1.0`, and evaluated with no
rollouts (pure value head) and `c_puct 0.25`.

### Player evaluation

| Key | Notes |
|---|---|
| `do_player_eval` | Gates `a0.eval.player` |
| `player_eval_stride` | Evaluate every Nth checkpoint. Iteration 0 and the latest are always included |
| `player_eval_num_games` | Games per side, per checkpoint |
| `plateau_sweep_mcts_samples` | Search budgets for the plateau sweep |
| `plateau_sweep_num_checkpoints` | How many converged checkpoints to average |

### Model and optimisation

`num_filters`, `num_resblocks`, `learning_rate`, `weight_decay`,
`value_loss_weight`, `grad_clip_norm`, `value_head_zero_init`,
`persist_optimizer_state`.

`value_head_zero_init` zero-initialises the final value layer so pre-tanh output
starts at 0 — relevant to the value-collapse work.

### Collapse detection

`detect_collapse`, `collapse_detection_iteration`,
`collapse_threshold_pre_tanh`, `max_collapse_retries`. Checks whether the value
head's mean absolute pre-tanh output has run away in the first N iterations, and
optionally restarts the run. Investigation found the collapse to be seed-driven
rather than weighting-driven, so this is diagnostic rather than a fix.

### Dataset balancing

`dataset_balance_method` (`none` | `subsample_buckets` | `weighted_buckets`),
`num_buckets_for_balance`, `max_weight_ratio`.

### Heatmaps

`heatmap_max_distance` bins at full resolution; `heatmap_min_cell_count` is
applied at **plot** time. Re-cutting a heatmap therefore never requires
re-running the ground-truth-bound analysis pass.

### Runtime

| Key | Notes |
|---|---|
| `num_workers` | Self-play worker processes. 24 on a 16-core node — self-play is latency-bound, so oversubscribing helps |
| `persist_replay_buffer` | Checkpoint the replay buffer so a resumed run does not restart it empty |
| `distribution_max_samples` | Cap on raw samples kept per (trial, iteration) in distribution pkls. The plots histogram them anyway, so a subsample is as good at a fraction of the size |

## Shipped configs

| File | Board |
|---|---|
| `config/config.json` | 25-6 — the default and the full schema |
| `config/config-16-3.json` | 16-3 — smallest, fastest, good for a first run |
| `config/config-36-6.json` | 36-6 — sets `do_gt_evals: false` |
| `config/config-49-4.json` | 49-4 — largest solved board |
| `config/config-test.json` | Short 25-6 run for smoke-testing |
| `config/config-slbl.json` | Supervised-learning baseline |
