# a0cc — AlphaZero on strongly solved Chinese Checkers

Research code for the MSc thesis *Investigating AlphaZero's Value Targets Using
Strongly Solved Chinese Checkers* (Nicolas Ong, supervised by Nathan Sturtevant,
University of Alberta).

Chinese Checkers is strongly solved on small boards, so the game-theoretic value
of **every** state is known. That turns questions normally answered indirectly —
through win rates and Elo — into direct measurements: you can ask what fraction
of the value head's predictions are actually correct, and what fraction of the
targets AlphaZero trains on are actually correct. This repo is the machinery for
asking those questions, and for comparing Monte Carlo value targets against
TD(λ) and fitted-value-iteration alternatives.

The thesis itself lives in a separate repo, and the investigation notes behind
each result in a third. See [Related repos](#related-repos).

## Quick start

```sh
# 1. environment (see docs/setup.md — off-cluster you must drop the
#    +computecanada version suffixes, they are not on PyPI)
python3.11 -m venv .venv && source .venv/bin/activate
pip install jax==0.4.35 jaxlib==0.4.34 flax==0.10.4 optax==0.2.5 \
            numpy==2.2.2 scipy==1.15.1 matplotlib==3.10.0 dill==0.4.0

# 2. ground truth: put a solve file where the config expects it
#    (docs/solve-data.md — this is the one thing you cannot get from this repo)
ls input/solvedata/CC-SOLVE-BASELINE-16-3.dat

# 3. train one run, trial 1, on the smallest board
python -m a0.train.alphazero config/config-16-3.json 1

# 4. evaluate it
bash slurm/run_nonplayer_eval.sh config/config-16-3.json 1
python -m a0.eval.player config/config-16-3.json 1

# 5. tests
pytest
```

Start on **16-3** (4×4 board, 3 pieces each). Its solve file is 79 KB and a run
finishes quickly. The boards used in the thesis — 25-6, 36-6, 49-4 — need a
cluster and, for two of them, a multi-gigabyte solve file.

## Documentation

| Document | What's in it |
|---|---|
| [docs/setup.md](docs/setup.md) | Python environment, the two requirements files, cluster vs. local |
| [docs/solve-data.md](docs/solve-data.md) | Getting the ground-truth solve files — the hard prerequisite |
| [docs/pipeline.md](docs/pipeline.md) | Training → eval → combine, stage by stage, and what each writes |
| [docs/sweeps.md](docs/sweeps.md) | `submit_sweep.py`, sweep specs, job chaining, resuming, adding seeds |
| [docs/configuration.md](docs/configuration.md) | How config resolution works and the knobs that matter |
| [docs/experiments.md](docs/experiments.md) | **Thesis result → sweep spec → config.** Start here to reproduce a figure |
| [docs/repo-layout.md](docs/repo-layout.md) | Directory map, including why `a0/eval` and `a0/eval3` both exist |

## How it fits together

```
cc/          the game: rules, state ranking, and the solve-file reader.
             Pure Python, no C++ dependency at runtime.
a0/          AlphaZero: self-play, MCTS, model, training, evaluation.
config/      board and experiment configs, layered over config/config.json.
sweeps/      sweep specs — the experiments in the thesis, one JSON each.
slurm/       cluster job scripts for each pipeline stage.
scripts/     dated one-off analyses. Not maintained; kept as a record.
```

A single run is `(config, trial)`, where the trial number is the random seed and
also the output subdirectory. A *sweep* is a grid of configs × trials, expanded
and submitted by [submit_sweep.py](submit_sweep.py). Results are merged across
trials by the combine stage, which is where the confidence intervals come from.

## Three things worth knowing before you start

**The solve file is the bottleneck, not the training.** Ground-truth evaluation
needs a strongly solved database for the exact board being trained on. Those are
produced by a separate C++ project and range from 79 KB (16-3) to ~15 GB (49-4).
The 36-6 board has no solve file at all — it would be roughly 300 GB — so runs
there set `do_gt_evals: false` and are measured by playing strength only. See
[docs/solve-data.md](docs/solve-data.md).

**The requirements files are not pip-installable as written.** Both pin
`+computecanada` local versions, which exist only in the Alliance wheelhouse.
Off-cluster, install the same version numbers without the suffix. See
[docs/setup.md](docs/setup.md).

**Evaluation is split across two packages for historical reasons.** `a0/eval3`
is the newer analysis layer, but it imports from `a0/eval`, and the pipeline
interleaves the two. Neither is dead and neither can be deleted. There was never
an `eval2`. See [docs/repo-layout.md](docs/repo-layout.md).

## Related repos

- **Thesis** — [NicolasOng/a0cc-thesis](https://github.com/NicolasOng/a0cc-thesis).
  The document, its figures, and the data behind them.
- **Chinese Checkers (C++)** — the solver that produces the ground-truth
  databases this repo reads, and the reference implementation of the rules.
- **Investigations** — a lab notebook of dated investigations (value-head
  collapse, evaluation stochasticity, the final thesis runs). Most claims in the
  thesis trace back to a numbered note there.

## License

MIT — see [LICENSE](LICENSE).
