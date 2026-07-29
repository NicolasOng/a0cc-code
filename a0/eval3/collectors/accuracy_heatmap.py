'''
Training-data accuracy heatmaps: iteration (x) x game progress or
distance-from-terminal (y), cell = accuracy of the value targets.

COLUMN SEMANTICS — training-indexed. Column i holds every target whose
*iteration* is i, which on a target-refresh run is not the same as "targets of
games generated at i":

  classic run   turn.alternative_value_target, in its own file's iteration.
  refresh run   the turn.refresh_value_targets tuples, each landing in the
                column named by its own iteration field, all refreshes pooled,
                PLUS the refresh_targets_{i}.pkl sidecar rows — the targets of
                older buffered trajectories retargeted at iteration i, which the
                gamedata traversal cannot see. With both folded in, column i is
                every target created at iteration i regardless of how old the
                trajectory is.

One rule covers both and avoids double-counting: on a refresh run
alternative_value_target IS the refresh-0 target (trajectory_buffer.py sets it
that way), so reading both fields would count refresh 0 twice. Prefer the
tuples, fall back to the scalar.

BINNING is full-resolution here — one row per ply of distance, plus the standard
10 progress buckets — and is CUT at plot time (cap, bin width, min-count mask).
That way re-cutting a heatmap never re-runs the GT-bound traversal.

Rows D0..D_MAX are always emitted, NaN where empty, so runs with different
maximum game lengths still share a key set and merge_series can combine seeds.
Anything beyond D_MAX lands in the DOVER overflow row rather than being dropped.

Unfinished games (turn-limit timeouts) have no terminal state, so they are
excluded from the distance axis. They ARE included on the progress axis, which
keeps that axis's marginal identical to the existing 1D progress accuracy graph.

Outputs (eval_dir):
  {name}_iter_distance_acc.pkl   x = iteration, keys "<metric> D{d}" / "<metric> DOVER"
  {name}_iter_progress_acc.pkl   x = iteration, keys "<metric> P{bucket}"
'''
from typing import Callable

import numpy as np

from a0.utils.plotting import Series, save_series
from a0.utils.load_training_data import refresh_targets_generator
from a0.eval3.collectors.base import (
    Collector, GameInfo, TurnInfo, progress_bucket_for, progress_bucket_upper_bounds,
)

from config import config
from utils.log import get_logger
logger = get_logger(__name__)

# label for the "further from the terminal than D_MAX" row
OVERFLOW = "OVER"

# per-cell accumulator layout
_TOTAL, _CORRECT, _TOTAL_ND, _CORRECT_ND, _ABS_ERROR = range(5)

# (metric key prefix, whether it is normalised by the ND total rather than the
# plain total, index into the accumulator) — drives both emission and naming
_METRICS: list[tuple[str, bool, int]] = [
    ("Value Accuracy", False, _CORRECT),
    ("Value Accuracy ND", True, _CORRECT_ND),
    ("Target MAE", False, _ABS_ERROR),
]

def experienced_targets(ti: TurnInfo) -> list[tuple[int, float]]:
    '''The MC outcome actually experienced in self-play — the control series.'''
    return [(ti.iteration, float(ti.experienced_outcome))]

def alt_targets(ti: TurnInfo) -> list[tuple[int, float]]:
    '''
    The targets the value head trains on, as (1-based iteration, target) pairs.
    Refresh tuples store 0-based training iterations; every other eval3 series
    reports 1-based, so they are converted here.
    '''
    if ti.refresh_value_targets:
        return [(iteration + 1, float(target)) for iteration, _refresh, target in ti.refresh_value_targets]
    if ti.alternative_value_target is None:
        return []
    return [(ti.iteration, float(ti.alternative_value_target))]

class AccuracyHeatmapCollector(Collector):
    '''
    Accumulates value-target accuracy into (iteration, bin) cells on two axes.
    `get_targets` returns the (iteration, target) pairs a turn contributes, so
    the classic/refresh difference lives in one function instead of the loop.
    '''

    def __init__(
        self,
        name: str,
        get_targets: Callable[[TurnInfo], list[tuple[int, float]]] = alt_targets,
        n_progress_buckets: int = 10,
        max_distance: int | None = None,
        fold_refresh_sidecars: bool = False,
    ):
        self._name = name
        self._get_targets = get_targets
        self._n_progress_buckets = n_progress_buckets
        self._max_distance = max_distance if max_distance is not None else config.heatmap_max_distance
        self._fold_refresh_sidecars = fold_refresh_sidecars
        # (generation iteration, game index) -> trajectory facts the sidecar rows
        # need but don't carry: game length, whether it finished, and the GT
        # outcome per turn. Only built when folding, since it is dead weight
        # otherwise. ~4MB of float32 for a 200-iteration run at 5k states each.
        self._games: dict[tuple[int, int], tuple[int, bool, np.ndarray]] = {}
        # iteration -> bin -> accumulator
        self._distance: dict[int, dict[int | str, list[float]]] = {}
        self._progress: dict[int, dict[int, list[float]]] = {}
        # iteration -> [target entries, turns] → mean targets per turn, which is
        # K on a refresh run. Surfaces a scheduled change in K as a visible line
        # rather than letting it silently change what a column averages over.
        self._entries_per_turn: dict[int, list[int]] = {}

    @staticmethod
    def _cell() -> list[float]:
        return [0.0] * 5

    def _accumulate(
        self,
        store: dict,
        iteration: int,
        bin_key: int | str,
        target: float,
        gt_outcome: float,
    ) -> None:
        cell = store.setdefault(iteration, {}).setdefault(bin_key, self._cell())
        sign = float(np.sign(target))
        correct = (sign == gt_outcome)
        cell[_TOTAL] += 1
        if correct:
            cell[_CORRECT] += 1
        # ND: drop states drawn under GT or whose target carries no sign — the
        # same convention as AccuracyCollector and RefreshTargetAccuracyCollector
        if sign != 0.0 and gt_outcome != 0.0:
            cell[_TOTAL_ND] += 1
            if correct:
                cell[_CORRECT_ND] += 1
        cell[_ABS_ERROR] += abs(target - gt_outcome)

    def _distance_bin(self, distance: int) -> int | str:
        return distance if distance <= self._max_distance else OVERFLOW

    def on_game(self, gi: GameInfo) -> None:
        pass

    def on_turn(self, ti: TurnInfo) -> None:
        if self._fold_refresh_sidecars:
            self._index_turn(ti)

        targets = self._get_targets(ti)
        if not targets:
            return

        progress_bin = progress_bucket_for(ti.progress, self._n_progress_buckets)
        # an unfinished game has no terminal state to measure distance from
        distance = (ti.game_length - 1 - ti.turn_index) if ti.ended else None

        for iteration, target in targets:
            self._accumulate(self._progress, iteration, progress_bin, target, ti.gt_outcome)
            if distance is not None:
                self._accumulate(self._distance, iteration, self._distance_bin(distance), target, ti.gt_outcome)

        counts = self._entries_per_turn.setdefault(targets[0][0], [0, 0])
        counts[0] += len(targets)
        counts[1] += 1

    def on_iteration_end(self, iteration: int) -> None:
        pass

    def _index_turn(self, ti: TurnInfo) -> None:
        '''
        Records the trajectory facts a sidecar row can't supply. Keyed by
        0-based generation iteration to match the sidecar rows (ti.iteration is
        1-based, like every other eval3 series).
        '''
        key = (ti.iteration - 1, ti.game_index)
        entry = self._games.get(key)
        if entry is None:
            entry = (ti.game_length, ti.ended, np.full(ti.game_length, np.nan, dtype=np.float32))
            self._games[key] = entry
        entry[2][ti.turn_index] = ti.gt_outcome

    def fold_refresh_sidecars(self) -> None:
        '''
        Adds the sidecar rows — targets of OLDER buffered trajectories retargeted
        during each iteration — into the same cells, so a column ends up holding
        every target created at that iteration rather than only those of the
        trajectories generated then.

        Streams one sidecar file at a time and bins immediately. Deliberately
        does NOT build a per-game record store the way _collect_target_records
        does: at ~50k buffer positions x K refreshes x hundreds of iterations
        that store is multi-GB, while the bins are a fixed ~20k cells.

        Rows are (generation_iteration, game_index, turn_idx, rank, iteration,
        refresh, target), with both iteration fields 0-based.
        '''
        rows_seen = 0
        rows_folded = 0
        orphan_rows = 0
        orphan_games: set[tuple[int, int]] = set()
        unfinished_rows = 0

        for _file_iteration, payload in refresh_targets_generator(config.training_dir, config.training_iterations):
            for generation_iteration, game_index, turn_index, _rank, iteration, refresh, target in payload["rows"]:
                rows_seen += 1
                entry = self._games.get((generation_iteration, game_index))
                if entry is None:
                    # generation gamedata pkl missing — chained job with its own
                    # training_dir, or rolled back. The row carries `rank`, so GT
                    # is recoverable, but game length isn't, so there is no
                    # distance or progress to bin it into.
                    orphan_rows += 1
                    orphan_games.add((generation_iteration, game_index))
                    continue

                game_length, ended, gt_outcomes = entry
                if not 0 <= turn_index < game_length:
                    orphan_rows += 1
                    continue
                gt_outcome = float(gt_outcomes[turn_index])
                if np.isnan(gt_outcome):
                    orphan_rows += 1
                    continue

                column = iteration + 1
                progress = (turn_index * 100) // game_length
                self._accumulate(
                    self._progress, column,
                    progress_bucket_for(progress, self._n_progress_buckets),
                    float(target), gt_outcome,
                )
                if ended:
                    distance = game_length - 1 - turn_index
                    self._accumulate(self._distance, column, self._distance_bin(distance),
                                     float(target), gt_outcome)
                else:
                    unfinished_rows += 1

                counts = self._entries_per_turn.setdefault(column, [0, 0])
                counts[0] += 1
                # every turn contributes exactly one row per refresh, so the
                # refresh-0 rows count the distinct turns behind this column
                if refresh == 0:
                    counts[1] += 1
                rows_folded += 1

        if rows_seen == 0:
            logger.info(f"AccuracyHeatmapCollector '{self._name}': no refresh sidecars (classic run); nothing to fold.")
            return
        logger.info(
            f"AccuracyHeatmapCollector '{self._name}': folded {rows_folded}/{rows_seen} sidecar rows "
            f"({len(self._games)} indexed trajectories, {unfinished_rows} rows from unfinished games "
            f"kept on the progress axis only)."
        )
        if orphan_rows:
            logger.warning(
                f"AccuracyHeatmapCollector '{self._name}': dropped {orphan_rows} sidecar rows with no "
                f"trajectory context ({len(orphan_games)} games missing from the gamedata files). "
                f"Expected if this run was chained from a different training_dir or rolled back."
            )

    def _build_series(
        self,
        store: dict[int, dict],
        bin_keys: list[int | str],
        label: str,
    ) -> Series:
        '''
        Pivots {iteration: {bin: accumulator}} into a Series with x = iteration
        and one key per (metric, bin). Cells with no samples are NaN so they read
        as missing rather than as 0% accuracy.
        '''
        keys: list[str] = []
        for metric, _is_nd, _idx in _METRICS:
            keys += [f"{metric} {label}{b}" for b in bin_keys]
        keys += [f"Count {label}{b}" for b in bin_keys]
        keys += [f"Count ND {label}{b}" for b in bin_keys]
        keys.append("Mean Refresh Count")

        series = Series(keys)
        for iteration in sorted(store.keys()):
            series.x.append(iteration)
            bins = store[iteration]
            for b in bin_keys:
                cell = bins.get(b)
                total = cell[_TOTAL] if cell else 0
                total_nd = cell[_TOTAL_ND] if cell else 0
                for metric, is_nd, idx in _METRICS:
                    denominator = total_nd if is_nd else total
                    value = (cell[idx] / denominator) if (cell and denominator) else float("nan")
                    series.ys[f"{metric} {label}{b}"].append(value)
                series.ys[f"Count {label}{b}"].append(total)
                series.ys[f"Count ND {label}{b}"].append(total_nd)
            entries, turns = self._entries_per_turn.get(iteration, [0, 0])
            series.ys["Mean Refresh Count"].append(entries / turns if turns else float("nan"))
        return series

    def finalize(self) -> None:
        # must run before the pivot: the sidecar rows belong in the same cells,
        # and the traversal index they join against is only complete now
        if self._fold_refresh_sidecars:
            self.fold_refresh_sidecars()

        if not self._progress:
            logger.info(f"AccuracyHeatmapCollector '{self._name}': no targets found; skipping save.")
            return

        # fixed key sets so runs with different game lengths still merge
        distance_bins: list[int | str] = list(range(self._max_distance + 1)) + [OVERFLOW]
        progress_bins: list[int | str] = list(progress_bucket_upper_bounds(self._n_progress_buckets))

        distance_series = self._build_series(self._distance, distance_bins, "D")
        progress_series = self._build_series(self._progress, progress_bins, "P")

        save_series(distance_series, f"{config.eval_dir}/{self._name}_iter_distance_acc.pkl")
        save_series(progress_series, f"{config.eval_dir}/{self._name}_iter_progress_acc.pkl")

        overflow = sum(
            bins[OVERFLOW][_TOTAL] for bins in self._distance.values() if OVERFLOW in bins
        )
        total = sum(cell[_TOTAL] for bins in self._progress.values() for cell in bins.values())
        logger.info(
            f"AccuracyHeatmapCollector '{self._name}': saved "
            f"{self._name}_iter_distance_acc.pkl + {self._name}_iter_progress_acc.pkl "
            f"({len(distance_series.x)} iterations, D0..D{self._max_distance}+overflow, "
            f"n={int(total)} target entries, {int(overflow)} beyond D{self._max_distance})"
        )
