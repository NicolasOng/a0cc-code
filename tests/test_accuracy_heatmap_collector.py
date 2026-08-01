'''
Unit tests for AccuracyHeatmapCollector's per-cell metrics.

Focused on "GT Win Rate ND", the chance baseline: it has to share the ND
denominator of "Value Accuracy ND" exactly, or it is a baseline for a different
number than the one it is plotted against. The parity structure it is meant to
expose (winner on move at even distances, loser at odd) is asserted end-to-end
on a synthetic game rather than trusted from the plots.
'''
import numpy as np
import pytest

from a0.eval3.collectors.accuracy_heatmap import (
    AccuracyHeatmapCollector, experienced_targets,
)
from a0.eval3.collectors.base import (
    TurnInfo, progress_bucket_for, progress_bucket_upper_bounds,
)


def _turn(iteration, turn_index, game_length, gt_outcome, target,
          experienced=None, ended=True):
    '''A TurnInfo carrying only the fields this collector reads.'''
    n = 4
    return TurnInfo(
        iteration=iteration,
        board=None,  # type: ignore[arg-type]
        winner=None,
        turn_index=turn_index,
        game_length=game_length,
        gt_outcome=float(gt_outcome),
        experienced_outcome=float(experienced if experienced is not None else target),
        gt_policy=np.zeros(n, dtype=np.float32),
        experienced_policy=np.zeros(n, dtype=np.float32),
        is_trivial=False,
        progress=(turn_index * 100) // game_length,
        alternative_value_target=float(target),
        alternative_policy_target=None,
        ended=ended,
        game_index=0,
    )


def _run(turns, **kwargs):
    c = AccuracyHeatmapCollector("t", **kwargs)
    for t in turns:
        c.on_turn(t)
    bins = list(range(c._max_distance + 1)) + ["OVER"]
    return c._build_series(c._distance, bins, "D")


def test_win_rate_is_the_fraction_of_gt_wins_in_the_nd_population():
    # one distance bin (D0), four turns: 3 GT wins, 1 GT loss -> 0.75
    turns = [_turn(1, 1, 2, gt, 1.0) for gt in (1, 1, 1, -1)]
    s = _run(turns)
    assert s.ys["GT Win Rate ND D0"][0] == pytest.approx(0.75)
    assert s.ys["Count ND D0"][0] == 4


def test_gt_draws_are_excluded_from_the_win_rate_denominator():
    # 2 wins, 1 loss, 2 GT draws. Draws leave the ND population entirely, so the
    # rate is 2/3, NOT 2/5 — the same exclusion Value Accuracy ND applies.
    turns = [_turn(1, 1, 2, gt, 1.0) for gt in (1, 1, -1, 0, 0)]
    s = _run(turns)
    assert s.ys["GT Win Rate ND D0"][0] == pytest.approx(2.0 / 3.0)
    assert s.ys["Count ND D0"][0] == 3
    assert s.ys["Count D0"][0] == 5


def test_unsigned_targets_leave_the_nd_population_too():
    # a target of exactly 0 has no sign to score, so ND drops it even though GT
    # is decisive. The win rate must drop it as well to stay the baseline for
    # Value Accuracy ND rather than for some larger sample.
    turns = [_turn(1, 1, 2, 1, 1.0), _turn(1, 1, 2, -1, 0.0)]
    s = _run(turns)
    assert s.ys["Count ND D0"][0] == 1
    assert s.ys["GT Win Rate ND D0"][0] == pytest.approx(1.0)


def test_win_rate_alternates_with_distance_parity():
    '''
    The structural claim the plots rest on: value is from the player-to-move's
    perspective and that perspective flips every ply, so on a decisively won
    game the GT outcome alternates +1 / -1 with distance from terminal.
    '''
    length = 8
    turns = [
        _turn(1, i, length, 1.0 if (length - 1 - i) % 2 == 0 else -1.0, 1.0)
        for i in range(length)
    ]
    s = _run(turns)
    for d in range(length):
        assert s.ys[f"GT Win Rate ND D{d}"][0] == pytest.approx(
            1.0 if d % 2 == 0 else 0.0), f"parity broken at D{d}"


def test_win_rate_is_nan_where_no_states_landed():
    s = _run([_turn(1, 1, 2, 1, 1.0)])
    assert np.isnan(s.ys["GT Win Rate ND D5"][0])


def test_unfinished_games_are_absent_from_the_distance_axis():
    # a turn-limit timeout has no terminal state, so it contributes nothing to
    # any distance bin. Not "a row of NaN" — the iteration never enters the
    # distance store at all, so the series has no x entry for it.
    s = _run([_turn(1, 1, 2, 1, 1.0, ended=False)])
    assert s.x == []
    assert s.ys["GT Win Rate ND D0"] == []


def test_unfinished_games_still_reach_the_progress_axis():
    # the counterpart of the above, and the reason the two axes disagree: the
    # progress marginal has to stay identical to the existing 1D progress graph,
    # which includes timeouts.
    c = AccuracyHeatmapCollector("t")
    t = _turn(1, 1, 2, 1, 1.0, ended=False)
    c.on_turn(t)
    bins = progress_bucket_upper_bounds(c._n_progress_buckets)
    bucket = progress_bucket_for(t.progress, c._n_progress_buckets)
    s = c._build_series(c._progress, list(bins), "P")
    assert s.x == [1]
    assert s.ys[f"GT Win Rate ND P{bucket}"][0] == pytest.approx(1.0)


def test_experienced_and_alt_targets_can_see_different_nd_populations():
    '''
    The two collectors filter ND by their OWN target's sign, so their win rates
    are baselines for their own accuracy and need not match. Here the alt target
    is unsigned while the experienced outcome is not.
    '''
    turns = [_turn(1, 1, 2, -1, 0.0, experienced=1.0)]
    alt = _run(turns)
    exp = _run(turns, get_targets=experienced_targets)
    assert alt.ys["Count ND D0"][0] == 0
    assert exp.ys["Count ND D0"][0] == 1
    assert exp.ys["GT Win Rate ND D0"][0] == pytest.approx(0.0)
