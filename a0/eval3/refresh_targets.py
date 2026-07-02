'''
Analyses for the target-refresh training path (config.use_target_refresh).

Data sources produced during training:
  - turn.refresh_value_targets: (iteration, refresh, target) tuples on each
    turn in gamedata_{i}.pkl — the game's GENERATION-iteration refreshes.
  - refresh_targets_{i}.pkl sidecars: rows (generation_iteration, game_index,
    turn_idx, rank, iteration, refresh, target) — retargets of OLDER buffered
    games during iteration i. Duplicate ranks are expected (targets are
    trajectory-dependent) and counted per occurrence.
Iterations inside tuples/rows are 0-based training iterations; all Series here
report them 1-based to match the other eval files.

Outputs (eval_dir):
  - refresh_targets_acc.pkl          value accuracy vs GT per refresh,
                                     generation-iteration games (via the
                                     RefreshTargetAccuracyCollector, which
                                     training_data.run_collectors registers)
  - refresh_targets_sidecar_acc.pkl  same for buffered retargets (rank-joined)
  - refresh_target_delta.pkl         mean/max |target change| per refresh
                                     transition per iteration
  - refresh_delta_by_distance.pkl    mean |target change| binned by
                                     distance-from-terminal per transition
                                     (the propagation-front data)
NaN is the missing-value convention (skipped by matplotlib, treated as
missing by merge_series).
'''
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from cc.ground_truth import GroundTruth
from a0.utils.plotting import Series, save_series
from a0.utils.load_training_data import game_data_generator, refresh_targets_generator
from a0.eval3.collectors.base import Collector, GameInfo, TurnInfo

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

# transition label for the delta between the last refresh of one iteration and
# refresh 0 of the next (computed for games that stayed in the buffer)
CROSS_ITERATION = "Cross-Iteration"

def _transition_sort_key(name: str) -> tuple[int, int]:
    # "R0->R1" ... sorted numerically, Cross-Iteration last
    if name == CROSS_ITERATION:
        return (1, 0)
    return (0, int(name.split("->")[0][1:]))

def _accuracy_series(per_iteration: dict[int, dict[int, list[int]]]) -> Series | None:
    '''
    Builds a Series from {0-based iteration: {refresh: [correct, total,
    correct_nd, total_nd]}} with keys "Value Accuracy R{k}" / ND variant.
    Returns None when there is no data.
    '''
    if not per_iteration:
        return None
    iterations = sorted(per_iteration.keys())
    refreshes = sorted({r for counts in per_iteration.values() for r in counts})
    keys = [f"Value Accuracy R{r}" for r in refreshes] + [f"Value Accuracy ND R{r}" for r in refreshes]
    series = Series(keys)
    for it in iterations:
        series.x.append(it + 1)  # 1-based, matching the other eval series
        counts = per_iteration[it]
        for r in refreshes:
            c = counts.get(r)
            if c is None:
                series.ys[f"Value Accuracy R{r}"].append(float("nan"))
                series.ys[f"Value Accuracy ND R{r}"].append(float("nan"))
            else:
                correct, total, correct_nd, total_nd = c
                series.ys[f"Value Accuracy R{r}"].append(correct / total if total else float("nan"))
                series.ys[f"Value Accuracy ND R{r}"].append(correct_nd / total_nd if total_nd else float("nan"))
    return series

class RefreshTargetAccuracyCollector(Collector):
    '''
    Value accuracy vs GT of the per-refresh targets on generation-iteration
    games, from turn.refresh_value_targets during the shared gamedata
    traversal. Sign-match convention (matches AccuracyCollector/alt_outcome);
    ND excludes turns where either the target sign or GT outcome is 0.
    → refresh_targets_acc.pkl (skipped entirely on non-refresh runs)
    '''

    def __init__(self):
        # 0-based iteration -> refresh -> [correct, total, correct_nd, total_nd]
        self._per_iteration: dict[int, dict[int, list[int]]] = {}

    def on_game(self, gi: GameInfo) -> None:
        pass

    def on_turn(self, ti: TurnInfo) -> None:
        tuples = ti.refresh_value_targets
        if not tuples:
            return
        for iteration, refresh, target in tuples:
            counts = self._per_iteration.setdefault(iteration, {}).setdefault(refresh, [0, 0, 0, 0])
            sign = float(np.sign(target))
            correct = (sign == ti.gt_outcome)
            counts[1] += 1
            if correct:
                counts[0] += 1
            if sign != 0.0 and ti.gt_outcome != 0.0:
                counts[3] += 1
                if correct:
                    counts[2] += 1

    def on_iteration_end(self, iteration: int) -> None:
        pass

    def finalize(self) -> None:
        series = _accuracy_series(self._per_iteration)
        if series is None:
            logger.info("RefreshTargetAccuracyCollector: no refresh_value_targets found; skipping save.")
            return
        save_series(series, f"{config.eval_dir}/refresh_targets_acc.pkl")
        logger.info(f"RefreshTargetAccuracyCollector: saved refresh_targets_acc.pkl ({len(series.x)} iterations).")

def compute_and_save_sidecar_accuracy(gt: GroundTruth) -> None:
    '''
    Value accuracy vs GT of the refreshed targets on buffered (older) games,
    from the sidecar rows, joined to GT by rank (rank -> unrank -> outcome,
    cached per rank). Duplicate ranks count per occurrence.
    → refresh_targets_sidecar_acc.pkl
    '''
    per_iteration: dict[int, dict[int, list[int]]] = {}
    outcome_cache: dict[int, float] = {}

    for _, payload in refresh_targets_generator(config.training_dir, config.training_iterations):
        for _, _, _, rank, iteration, refresh, target in payload["rows"]:
            outcome = outcome_cache.get(rank)
            if outcome is None:
                outcome = gt.get_outcome(gt.unrank(rank))
                outcome_cache[rank] = outcome
            counts = per_iteration.setdefault(iteration, {}).setdefault(refresh, [0, 0, 0, 0])
            sign = float(np.sign(target))
            correct = (sign == outcome)
            counts[1] += 1
            if correct:
                counts[0] += 1
            if sign != 0.0 and outcome != 0.0:
                counts[3] += 1
                if correct:
                    counts[2] += 1

    series = _accuracy_series(per_iteration)
    if series is None:
        logger.info("compute_and_save_sidecar_accuracy: no sidecar rows found; skipping save.")
        return
    save_series(series, f"{config.eval_dir}/refresh_targets_sidecar_acc.pkl")
    logger.info(f"compute_and_save_sidecar_accuracy: saved refresh_targets_sidecar_acc.pkl "
                f"({len(series.x)} iterations, {len(outcome_cache)} unique ranks).")

def _collect_target_records() -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], dict[tuple[int, int], NDArray[np.float32]]]]:
    '''
    Merges gamedata tuples and sidecar rows into per-game target histories:
      game_lengths[(gen_iter, game_index)] = T
      records[(gen_iter, game_index)][(iteration, refresh)] = targets (T,), NaN where unrecorded
    '''
    game_lengths: dict[tuple[int, int], int] = {}
    records: dict[tuple[int, int], dict[tuple[int, int], NDArray[np.float32]]] = defaultdict(dict)

    for file_i, game_data_list in game_data_generator(config.training_dir, config.training_iterations):
        gen_iter = file_i - 1
        for game_index, gd in enumerate(game_data_list):
            num_turns = len(gd.turn_data)
            if num_turns == 0:
                continue
            key = (gen_iter, game_index)
            game_lengths[key] = num_turns
            for turn_idx, turn in enumerate(gd.turn_data):
                tuples = getattr(turn, "refresh_value_targets", None)
                if not tuples:
                    continue
                for iteration, refresh, target in tuples:
                    arr = records[key].get((iteration, refresh))
                    if arr is None:
                        arr = np.full(num_turns, np.nan, dtype=np.float32)
                        records[key][(iteration, refresh)] = arr
                    arr[turn_idx] = target

    skipped_games: set[tuple[int, int]] = set()
    for _, payload in refresh_targets_generator(config.training_dir, config.training_iterations):
        for gen_iter, game_index, turn_idx, _, iteration, refresh, target in payload["rows"]:
            key = (gen_iter, game_index)
            num_turns = game_lengths.get(key)
            if num_turns is None:
                # generation gamedata pkl missing — no trajectory context
                skipped_games.add(key)
                continue
            arr = records[key].get((iteration, refresh))
            if arr is None:
                arr = np.full(num_turns, np.nan, dtype=np.float32)
                records[key][(iteration, refresh)] = arr
            arr[turn_idx] = target
    if skipped_games:
        logger.warning(f"_collect_target_records: {len(skipped_games)} sidecar games missing from gamedata files; skipped.")

    return game_lengths, records

def compute_and_save_delta_analyses() -> None:
    '''
    Walks per-game target histories chronologically and accumulates
    |target change| per transition (R{k-1}->R{k} within an iteration, plus
    Cross-Iteration: last refresh of iter i-1 -> refresh 0 of iter i).
    → refresh_target_delta.pkl      x = iteration (attributed to the later
                                    iteration), keys "Delta Mean/Max {trans}"
    → refresh_delta_by_distance.pkl x = distance-from-terminal (plies),
                                    keys per transition (propagation front)
    '''
    game_lengths, records = _collect_target_records()
    if not any(records.values()):
        logger.info("compute_and_save_delta_analyses: no refresh target records found; skipping.")
        return

    # iteration -> transition -> [sum, count, max]
    per_iter: dict[int, dict[str, list[float]]] = defaultdict(dict)
    # transition -> distance -> [sum, count]
    front: dict[str, dict[int, list[float]]] = defaultdict(dict)

    for key, recs in records.items():
        num_turns = game_lengths[key]
        ordered = sorted(recs.keys())  # (iteration, refresh): lexicographic == chronological
        for (it_prev, r_prev), (it_cur, r_cur) in zip(ordered, ordered[1:]):
            if it_cur == it_prev and r_cur == r_prev + 1:
                transition = f"R{r_prev}->R{r_cur}"
            elif it_cur == it_prev + 1 and r_cur == 0:
                transition = CROSS_ITERATION
            else:
                continue  # gap (missing file / partial data): not a real transition
            delta = np.abs(recs[(it_cur, r_cur)] - recs[(it_prev, r_prev)])
            valid = np.where(~np.isnan(delta))[0]
            if valid.size == 0:
                continue
            acc = per_iter[it_cur].setdefault(transition, [0.0, 0, 0.0])
            acc[0] += float(np.nansum(delta))
            acc[1] += int(valid.size)
            acc[2] = max(acc[2], float(np.nanmax(delta)))
            for turn_idx in valid:
                distance = num_turns - 1 - int(turn_idx)
                f = front[transition].setdefault(distance, [0.0, 0])
                f[0] += float(delta[turn_idx])
                f[1] += 1

    transitions = sorted({t for accs in per_iter.values() for t in accs}, key=_transition_sort_key)

    # per-iteration delta series
    iterations = sorted(per_iter.keys())
    delta_series = Series([f"Delta Mean {t}" for t in transitions] + [f"Delta Max {t}" for t in transitions])
    for it in iterations:
        delta_series.x.append(it + 1)
        for t in transitions:
            acc = per_iter[it].get(t)
            delta_series.ys[f"Delta Mean {t}"].append(acc[0] / acc[1] if acc else float("nan"))
            delta_series.ys[f"Delta Max {t}"].append(acc[2] if acc else float("nan"))
    save_series(delta_series, f"{config.eval_dir}/refresh_target_delta.pkl")
    logger.info(f"compute_and_save_delta_analyses: saved refresh_target_delta.pkl ({len(iterations)} iterations, {len(transitions)} transitions).")

    # propagation-front series (aggregated over all iterations)
    distances = sorted({d for bins in front.values() for d in bins})
    front_series = Series(transitions)
    for d in distances:
        front_series.x.append(d)
        for t in transitions:
            f = front[t].get(d)
            front_series.ys[t].append(f[0] / f[1] if f else float("nan"))
    save_series(front_series, f"{config.eval_dir}/refresh_delta_by_distance.pkl")
    logger.info(f"compute_and_save_delta_analyses: saved refresh_delta_by_distance.pkl ({len(distances)} distances).")

def run_refresh_target_analyses(gt: GroundTruth | None) -> None:
    '''
    All refresh-target analyses that don't ride the collector traversal
    (that one is RefreshTargetAccuracyCollector, registered in run_collectors).
    Cheap no-ops on non-refresh runs.
    '''
    logger.info("run_refresh_target_analyses: delta analyses (tuples + sidecar)...")
    compute_and_save_delta_analyses()
    if gt is not None:
        logger.info("run_refresh_target_analyses: sidecar accuracy vs GT...")
        compute_and_save_sidecar_accuracy(gt)
    else:
        logger.info("run_refresh_target_analyses: gt off; skipping sidecar accuracy.")

def main():
    setup_logging(level=20, log_dir=config.log_dir, process_name="refresh_targets")
    gt = GroundTruth() if config.do_gt_evals else None
    run_refresh_target_analyses(gt)

if __name__ == "__main__":
    main()
