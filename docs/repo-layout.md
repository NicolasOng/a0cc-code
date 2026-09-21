# Repo layout

## Top level

```
cc/            the game
a0/            AlphaZero
config/        configs (config.json is the schema, not an example)
sweeps/        sweep specs — one per experiment
slurm/         cluster job scripts
scripts/       dated one-off analyses; not maintained
tests/         the test suite
docs/          this documentation
utils/         logging and system-metrics helpers

config.py      config loading; imported by nearly everything
conftest.py    pytest bootstrap (normalises argv before collection)
submit_sweep.py       sweep expansion and submission
aggregate_sweep.py    whole-sweep aggregation → sweep_results.json
plot_sweep.py         plots from that aggregation
plot_player_sweep.py  plateau-sweep plots; self-contained, runs on bare pkls
```

Gitignored, present locally: `input/` (solve files and datasets), `output/` and
`sweep-output/` (results), `logs/`.

## `cc/` — the game

Pure Python, no C++ dependency at runtime.

| Module | |
|---|---|
| `core.py` | Board, Move, Game: rules, move generation, terminal detection |
| `ranking.py` | State ↔ integer rank. Must agree with the C++ solver, or solve files cannot be indexed |
| `solvedata.py` | Reads the packed 2-bit solve databases; mmaps by default |
| `ground_truth.py` | Ground-truth lookups on top of the above |
| `lookups.py`, `stats.py`, `validations.py` | Tables, aggregation, consistency checks |

## `a0/` — AlphaZero

```
a0/
  model.py            network definition, save/load
  model_utils.py      board encoding, policy handling, rotation
  game.py             play a game between two players
  dataset.py          training dataset assembly
  experience_buffer.py

  mcts/               nn.py (model-guided), gt.py (ground-truth),
                      rollout.py, rollout_strategies.py
  players/            a0.py, gt.py, human.py, mcts_rollout.py,
                      model.py, random.py
  graph_search/       bfs.py, mcts.py — non-AlphaZero search

  train/              alphazero.py    the training loop
                      targets.py      value-target construction (TD(λ), refresh)
                      trajectory_buffer.py, buffer_checkpoint.py
                      dataset.py, generate_datasets.py, check_progress.py

  eval/  eval3/       evaluation — see below
  utils/              plotting, state helpers, safe loading
```

The board is always presented to the network from the current player's
perspective, which means rotating it 180° for player O and rotating the policy
back afterwards. Getting that rotation wrong in one direction is a bug that
looks like "player O is weak" rather than like a crash — `model_utils.py` is
where it lives.

## Why there are two eval packages

`a0/eval` and `a0/eval3` are **both live**. Neither can be deleted.

There was never an `eval2`.

`eval3` was written as a newer analysis layer and covers most per-trial stages,
but it was never a full replacement: it *imports from* `a0/eval`, and the
pipeline interleaves the two. From `slurm/run_nonplayer_eval.sh`:

```
a0.eval3.training_data
a0.eval3.generate_datasets
a0.eval3.dataset_evaluation
a0.eval.model_diagnostics      <-- eval
a0.eval3.plotting
a0.eval3.extract_summary
```

and from `slurm/combine_a0.sh`:

```
a0.eval.combine_merge
a0.eval.combine_plot
a0.eval3.combine               <-- eval3
a0.eval.combine_summary
```

What lives where:

| Package | Holds |
|---|---|
| `a0/eval` | Player evaluation (`player.py`, `player_sweep.py`, `player_progress.py`), the plotting and `Series` primitives everything else builds on, combine/merge/summary, model diagnostics, trial selection |
| `a0/eval3` | The per-trial analysis stages, the accuracy-heatmap collectors, refresh-target analysis, and its own combine |

`a0/eval/player.py` is the single most referenced module in the package. If you
are tempted to consolidate these, note that `eval3/extract_summary.py`
deliberately keeps metric keys identical to `eval/extract_summary.py` so
downstream consumers cannot tell which produced a summary — that compatibility
is load-bearing, not incidental.

## `scripts/`

Roughly 70 one-off analyses, most named `YYYY-MM-DD_topic.py`. They are a
record of what was tried, not maintained code: many target code that has since
moved, and they are excluded from the test run.

Run them from the repo root with the root on the path:

```sh
PYTHONPATH=. python scripts/2026-05-29_single_agent_bfs.py --board-size 4 --num-pieces 3
```

A few are genuinely useful:

| Script | |
|---|---|
| `2026-05-29_single_agent_bfs.py` | Generates the BFS distance arrays for the optional DB baseline |
| `play_cc_human.py` | Play the game on the terminal — quickest way to see the rules work |
| `play_vs_trained_model.py` | Play against a checkpoint |
| `2026-07-03_check_losing_policies.py` | Consistency check between solver labels and move-by-move outcomes |

## `slurm/`

| Script | |
|---|---|
| `train_a0_gpu.sh` | Training, with the self-resubmitting chain and the terminal eval slot |
| `run_nonplayer_eval.sh` | The six non-player eval stages; assumes an active venv |
| `eval_a0_gpu.sh` | Standalone non-player eval |
| `player_eval.sh` | Win-rate curve, chainable |
| `player_sweep.sh` | Plateau MCTS sweep |
| `combine_a0.sh` | Cross-trial merge, one array task per HP point |
| `stage_solve_data.sh` | Copies the solve file to node-local NVMe |
| `aggregate_sweep.sh` | Whole-sweep aggregation |
| `create_python_env.sh` | Builds the venv interactively |
| `train_a0_cpu.sh`, `eval_a0_cpu.sh`, `script_gpu.sh` | Older/CPU variants |

Each GPU script sources the CVMFS profile before `module load`, because the
module system is not inherited by non-interactive submission.

## History

The repo carries ~525 commits going back to 2025.

`a0_new/` — a from-scratch rewrite started Jan 2026 and abandoned in March —
was removed during the 2026-09 cleanup and remains in history if it is ever
wanted. The thesis was produced entirely with `a0/`.
