# Ground-truth solve data

This is the prerequisite that cannot be satisfied from inside this repo, and the
one most likely to stop a reproduction attempt. Read this before planning a run.

## What it is

A *strongly solved* database: for every legal state of a given board, the
game-theoretic result for the side to move under perfect play — win, loss, or
draw. Two bits per state, packed 32 states to a 64-bit word.

Everything that makes this project different from an ordinary AlphaZero study
depends on it:

- **Value-head accuracy** — is the model's predicted sign correct?
- **Policy-head accuracy** — does the model's preferred move preserve the
  game-theoretic result?
- **Training-target accuracy** — how often is the value AlphaZero is *training
  on* actually correct? This is the central measurement of the thesis.
- **The ground-truth player** and the GT-target training variants.

Win rate against a fixed baseline is the only major metric that does **not**
need it.

## Where it comes from

The solve files are produced by the separate **Chinese Checkers C++ project**,
not by this repo. This repo only reads them, via
[`cc/solvedata.py`](../cc/solvedata.py).

The relevant solver is `BaselineSolver`, which names its output

```
CC-SOLVE-BASELINE-<NUM_SPOTS>-<NUM_PIECES>.dat
```

Board dimensions are **compile-time constants** (`#define NUM_SPOTS` /
`#define NUM_PIECES` in `cc/CCheckers.h`), so you select a board by choosing the
right definition block and rebuilding:

```sh
cd build/gmake && make
```

Then drop the result in `input/solvedata/` in this repo.

## File format

Read by `SolveData.read_solve_data_file`. Little-endian throughout:

```
 16 bytes  header
             8 bytes  number of entries
             8 bytes  memory size (in 64-bit words)
  N bytes  payload: memory_size * 8 bytes, 2 bits per state
```

A state's value is looked up by its rank: word `index >> 5`, bit offset
`(index & 0x1F) << 1`. The layout mirrors `NBitArray<2>` on the C++ side, so the
two implementations must stay in step — if ranking changes there, these files
become unreadable here.

By default the file is **memory-mapped** rather than read into RAM, which is what
makes a 15 GB database usable inside a 64 GB job.

## The boards, and what each costs

| Board | `board_size` | `num_pieces` | Solve file | Config |
|---|:--:|:--:|---|---|
| 16-3 | 4 | 3 | 79 KB | `config/config-16-3.json` |
| 25-6 | 5 | 6 | 2.3 GB | `config/config.json` (the default) |
| 36-6 | 6 | 6 | **none — infeasible** | `config/config-36-6.json` |
| 49-4 | 7 | 4 | ~15 GB | `config/config-49-4.json` |

The `N-M` names are *spots*-*pieces*: `num_spots = board_size²`.

### 36-6 has no solve file

A 36-6 database would be roughly 300 GB, which was not practical. So
`config/config-36-6.json` sets `do_gt_evals: false`, and every 36-6 run is
measured by **playing strength only** — win rate against the baseline, no
accuracy curves and no heatmaps.

This is not a bug and it is not a missing download. If you are looking at a 36-6
sweep and cannot find `seen_nd_eval.pkl` or `gamedata_alt_acc.pkl`, that is why.
Missing GT pkls on **49-4** mean something different — that board is solved, so
they should exist.

## Staging on the cluster

`slurm/stage_solve_data.sh` copies the solve file to node-local NVMe
(`$SLURM_TMPDIR`) before a job starts, and `SolveData._resolve_staged_path`
prefers that copy automatically. Without it, every mmap page fault crosses the
network filesystem; with a 15 GB file and millions of lookups, that dominates
runtime.

Off-cluster, `$SLURM_TMPDIR` is unset and the original path is mapped in place —
the mechanism is a no-op rather than something you need to disable.

Two stages deliberately **skip** staging because they never read the file:

- `slurm/player_eval.sh` — the win-rate curve plays against the DIST/Manhattan
  baseline, which needs no ground truth.
- `slurm/combine_a0.sh` — merges pickles only.

## Other inputs under `input/`

None of these are prerequisites for the main pipeline; they support specific
analyses.

| Path | What | Produced by |
|---|---|---|
| `input/solvedata/*.dat` | Solve databases (above) | C++ solver |
| `input/bfs_db/bfs_<size>_<pieces>.npy` | Single-agent BFS distance-to-goal, for the optional DB baseline (`make_bfs_baseline`) | `scripts/2026-05-29_single_agent_bfs.py` |
| `input/datasets/gtd.pkl` | Cached ground-truth evaluation dataset | `a0/train/generate_datasets.py` |
| `input/nn_slbl/*.pkl` | Supervised-learning baseline checkpoints | `config/config-slbl.json` runs |

The baseline actually used by the thesis player evals is `make_baseline`
(DIST/Manhattan), which reads **no** input file at all. `make_bfs_baseline` is
the exact-distance variant and is the only reason `bfs_db/` exists.

`input/` is gitignored — it is data, and at full size several gigabytes.
