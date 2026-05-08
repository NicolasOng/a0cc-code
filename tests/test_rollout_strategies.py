import random

from cc.core import Board, Game, Move, Player, Point, Tile

from a0.mcts.rollout_strategies import (
    EvaluatorType,
    PolicyType,
    ZeroEval,
    DistEval,
    LBDistEval,
    RandomPolicy,
    RandomForwardPolicy,
    BestPlayoutPolicy,
    BackPlayoutPolicy,
    goal_corner_for,
    goal_distance,
    filter_forward,
    forward_progress,
    make_evaluator,
    make_policy,
    _pick_best_random_tie,
)


def _empty_board(board_size: int = 7, home_size: int = 3) -> Board:
    return Board(board_size=board_size, home_size=home_size)


def test_goal_corner_for():
    print("Testing goal_corner_for...")
    assert goal_corner_for(Player.PLAYER_X, 7) == (6, 6)
    assert goal_corner_for(Player.PLAYER_O, 7) == (0, 0)
    assert goal_corner_for(Player.PLAYER_X, 4) == (3, 3)
    assert goal_corner_for(Player.PLAYER_O, 4) == (0, 0)


def test_goal_distance():
    print("Testing goal_distance (raw + exclude_within)...")
    # piece at goal corner -> 0
    assert goal_distance([Point(6, 6)], (6, 6)) == 0
    # piece at opposite corner -> Manhattan = 12
    assert goal_distance([Point(0, 0)], (6, 6)) == 12
    # multiple pieces -> sum
    assert goal_distance([Point(0, 0), Point(6, 6), Point(3, 3)], (6, 6)) == 12 + 0 + 6

    # exclude_within: pieces with d < threshold are skipped
    pts = [Point(6, 5), Point(5, 5), Point(2, 6)]  # dist 1, 2, 4
    assert goal_distance(pts, (6, 6)) == 7
    assert goal_distance(pts, (6, 6), exclude_within=3) == 4  # drops d=1, d=2
    assert goal_distance(pts, (6, 6), exclude_within=5) == 0  # drops everything


def test_filter_forward_player_x():
    print("Testing filter_forward for PLAYER_X (forward = is_down)...")
    state = _empty_board()
    state.current_player = Player.PLAYER_X
    forward_moves = [Move(3, 3, 4, 4), Move(3, 3, 3, 4), Move(3, 3, 4, 3)]   # is_down
    backward_moves = [Move(3, 3, 2, 2), Move(3, 3, 2, 3), Move(3, 3, 3, 2)]  # is_up
    sideways_moves = [Move(3, 3, 4, 2), Move(3, 3, 2, 4)]                    # is_sideways
    moves = forward_moves + backward_moves + sideways_moves
    filtered = filter_forward(state, moves)
    assert set(map(str, filtered)) == set(map(str, forward_moves))


def test_filter_forward_player_o():
    print("Testing filter_forward for PLAYER_O (forward = is_up)...")
    state = _empty_board()
    state.current_player = Player.PLAYER_O
    forward_moves = [Move(3, 3, 2, 2), Move(3, 3, 2, 3), Move(3, 3, 3, 2)]   # is_up
    backward_moves = [Move(3, 3, 4, 4), Move(3, 3, 3, 4), Move(3, 3, 4, 3)]  # is_down
    moves = forward_moves + backward_moves
    filtered = filter_forward(state, moves)
    assert set(map(str, filtered)) == set(map(str, forward_moves))


def test_forward_progress_sign():
    print("Testing forward_progress sign (forward is positive for both players)...")
    down_move = Move(0, 0, 1, 1)  # anti-diag delta = +2
    up_move = Move(2, 2, 1, 1)    # anti-diag delta = -2
    assert forward_progress(down_move, Player.PLAYER_X) == 2
    assert forward_progress(down_move, Player.PLAYER_O) == -2
    assert forward_progress(up_move, Player.PLAYER_X) == -2
    assert forward_progress(up_move, Player.PLAYER_O) == 2


def test_zero_eval():
    print("Testing ZeroEval...")
    state = Game(7, 6).board
    assert ZeroEval().evaluate(state) == 0.0


def test_dist_eval_near_win_better_than_start():
    print("Testing DistEval prefers near-win to start...")
    e = DistEval(Player.PLAYER_X, 7)

    start = Game(7, 6).board
    start_value = e.evaluate(start)

    # Near-win for X: X pieces fully in BR goal triangle, O scattered mid-board
    # (far from O's goal corner (0,0)). The start state is symmetric and would
    # otherwise score the same — we need O *not* close to (0,0).
    near_win = _empty_board()
    near_win.current_player = Player.PLAYER_X
    near_win.apply_points(
        [Point(6, 6), Point(5, 6), Point(6, 5), Point(4, 6), Point(5, 5), Point(6, 4)],
        Tile.PLAYER_X,
    )
    near_win.apply_points(
        [Point(3, 3), Point(3, 4), Point(4, 3), Point(2, 5), Point(5, 2), Point(3, 5)],
        Tile.PLAYER_O,
    )
    near_win_value = e.evaluate(near_win)

    assert near_win_value > start_value, f"near-win {near_win_value} should beat start {start_value}"


def test_lbdist_invariance_within_goal():
    print("Testing LBDistEval is invariant to in-goal piece arrangement...")
    # Two boards differing only in how 5 X pieces are arranged inside the goal triangle.
    # All 5 are within Manhattan distance < home_size (3) of (6, 6), so LBDist excludes them.
    # The single out-of-goal X piece at (0,0) is identical, as are all O pieces.
    common_o = [Point(0, 1), Point(1, 0), Point(0, 2), Point(1, 1), Point(2, 0), Point(0, 3)]

    base = _empty_board()
    base.current_player = Player.PLAYER_X
    base.apply_points([Point(0, 0)], Tile.PLAYER_X)
    base.apply_points([Point(6, 6), Point(5, 6), Point(6, 5), Point(4, 6), Point(5, 5)], Tile.PLAYER_X)
    base.apply_points(common_o, Tile.PLAYER_O)

    # `other` differs from `base` by replacing a d=1 in-goal piece with a d=2 in-goal piece;
    # this ensures DistEval (which counts them) sees a different value, while LBDist (which
    # excludes everything with d < 3) sees both as equivalent.
    other = _empty_board()
    other.current_player = Player.PLAYER_X
    other.apply_points([Point(0, 0)], Tile.PLAYER_X)
    other.apply_points([Point(6, 6), Point(6, 5), Point(4, 6), Point(5, 5), Point(6, 4)], Tile.PLAYER_X)
    other.apply_points(common_o, Tile.PLAYER_O)

    e = LBDistEval(Player.PLAYER_X, 7, 3)
    assert e.evaluate(base) == e.evaluate(other)

    # Sanity: regular DistEval should NOT be invariant (it does count those pieces)
    e_dist = DistEval(Player.PLAYER_X, 7)
    assert e_dist.evaluate(base) != e_dist.evaluate(other)


def test_zero_eval_terminal_value():
    print("Testing ZeroEval.terminal_value preserves ground-truth ±1/0...")
    state = Game(7, 6).board
    e = ZeroEval()
    assert e.terminal_value(state, Player.PLAYER_X, Player.PLAYER_X) == 1.0
    assert e.terminal_value(state, Player.PLAYER_O, Player.PLAYER_X) == -1.0
    assert e.terminal_value(state, Player.PLAYER_X, Player.PLAYER_O) == -1.0
    assert e.terminal_value(state, Player.PLAYER_O, Player.PLAYER_O) == 1.0
    assert e.terminal_value(state, None, Player.PLAYER_X) == 0.0


def test_dist_eval_terminal_value_matches_evaluate():
    print("Testing DistEval.terminal_value reuses evaluate (single value scale)...")
    # Build an X-winning terminal state
    win_state = _empty_board()
    win_state.current_player = Player.PLAYER_O  # O to move when X just won
    win_state.apply_points(
        [Point(6, 6), Point(5, 6), Point(6, 5), Point(4, 6), Point(5, 5), Point(6, 4)],
        Tile.PLAYER_X,
    )
    win_state.apply_points(
        [Point(3, 3), Point(3, 4), Point(4, 3), Point(2, 5), Point(5, 2), Point(3, 5)],
        Tile.PLAYER_O,
    )
    e = DistEval(Player.PLAYER_X, 7)
    # terminal_value should equal evaluate at the same state — same value scale.
    # winner argument is ignored by DistEval (the eval already encodes who's winning).
    assert e.terminal_value(win_state, Player.PLAYER_X, Player.PLAYER_X) == e.evaluate(win_state)
    assert e.terminal_value(win_state, None, Player.PLAYER_X) == e.evaluate(win_state)
    # And the value should be strongly positive (X is winning by a lot)
    assert e.evaluate(win_state) > 0


def test_best_playout_picks_longest_forward():
    print("Testing BestPlayoutPolicy picks the longest forward move...")
    state = _empty_board()
    state.current_player = Player.PLAYER_X
    state.apply_points([Point(0, 0)], Tile.PLAYER_X)
    moves = [
        Move(0, 0, 1, 1),  # forward progress = 2
        Move(0, 0, 0, 1),  # forward progress = 1
        Move(0, 0, 2, 2),  # forward progress = 4 (best, unique)
        Move(0, 0, 1, 0),  # forward progress = 1
    ]
    chosen = BestPlayoutPolicy(epsilon=0.0).choose(state, moves, Game(7, 6))
    assert (chosen.end.x, chosen.end.y) == (2, 2)


def test_back_playout_picks_rear_piece():
    print("Testing BackPlayoutPolicy picks moves from the rear-most piece...")
    state = _empty_board()
    state.current_player = Player.PLAYER_X
    state.apply_points([Point(0, 0), Point(4, 4)], Tile.PLAYER_X)
    moves = [
        Move(4, 4, 5, 5),  # forward piece (dist 4 from goal)
        Move(4, 4, 5, 4),  # forward piece
        Move(0, 0, 1, 1),  # rear piece (dist 12 from goal) — should be picked
        Move(0, 0, 0, 1),  # rear piece
    ]
    chosen = BackPlayoutPolicy(epsilon=0.0).choose(state, moves, Game(7, 6))
    assert (chosen.start.x, chosen.start.y) == (0, 0)


def test_best_playout_picks_least_bad_when_no_forward():
    print("Testing Best/Back pick least-bad move when no forward moves available...")
    state = _empty_board()
    state.current_player = Player.PLAYER_X
    state.apply_points([Point(3, 3)], Tile.PLAYER_X)
    # All moves are backward for X (is_up). Forward progress = -2, -1, -1.
    # Least-bad (largest forward_progress) is the -1 moves; tie-broken at random.
    only_backward = [
        Move(3, 3, 2, 2),  # anti-diag delta -2 -> forward_progress -2 (worst)
        Move(3, 3, 2, 3),  # anti-diag delta -1 -> forward_progress -1
        Move(3, 3, 3, 2),  # anti-diag delta -1 -> forward_progress -1
    ]
    least_bad = {(2, 3), (3, 2)}
    for _ in range(10):
        chosen = BestPlayoutPolicy(epsilon=0.0).choose(state, only_backward, Game(7, 6))
        assert (chosen.end.x, chosen.end.y) in least_bad, \
            f"BestPlayout fallback should pick a -1 move, got {chosen}"
        chosen = BackPlayoutPolicy(epsilon=0.0).choose(state, only_backward, Game(7, 6))
        assert (chosen.end.x, chosen.end.y) in least_bad, \
            f"BackPlayout fallback should pick a -1 move, got {chosen}"


def test_factories_return_expected_types():
    print("Testing factories return the expected concrete types...")
    assert isinstance(make_evaluator(EvaluatorType.NONE,   Player.PLAYER_X, 7, 3), ZeroEval)
    assert isinstance(make_evaluator(EvaluatorType.DIST,   Player.PLAYER_X, 7, 3), DistEval)
    assert isinstance(make_evaluator(EvaluatorType.LBDIST, Player.PLAYER_X, 7, 3), LBDistEval)
    assert isinstance(make_policy(PolicyType.RANDOM),               RandomPolicy)
    assert isinstance(make_policy(PolicyType.FORWARD),              RandomForwardPolicy)
    assert isinstance(make_policy(PolicyType.BEST,  epsilon=0.1),   BestPlayoutPolicy)
    assert isinstance(make_policy(PolicyType.BACK,  epsilon=0.1),   BackPlayoutPolicy)


def test_pick_best_random_tie():
    print("Testing _pick_best_random_tie returns a max-scoring move, breaks ties uniformly-ish...")
    moves = [
        Move(0, 0, 1, 1),  # score 2 (anti-diag delta)
        Move(0, 0, 2, 2),  # score 4 (max, tied below)
        Move(0, 0, 3, 1),  # score 4 (max, tied above)
        Move(0, 0, 0, 1),  # score 1
    ]
    winners = {(2, 2), (3, 1)}
    rng_state = random.getstate()
    try:
        random.seed(0)
        seen = set()
        for _ in range(50):
            choice = _pick_best_random_tie(moves, lambda m: m.diagonal_distances()[1])
            assert (choice.end.x, choice.end.y) in winners, f"unexpected choice {choice}"
            seen.add((choice.end.x, choice.end.y))
        assert seen == winners, f"only saw {seen}"
    finally:
        random.setstate(rng_state)


if __name__ == "__main__":
    test_goal_corner_for()
    test_goal_distance()
    test_filter_forward_player_x()
    test_filter_forward_player_o()
    test_forward_progress_sign()
    test_zero_eval()
    test_zero_eval_terminal_value()
    test_dist_eval_near_win_better_than_start()
    test_dist_eval_terminal_value_matches_evaluate()
    test_lbdist_invariance_within_goal()
    test_best_playout_picks_longest_forward()
    test_back_playout_picks_rear_piece()
    test_best_playout_picks_least_bad_when_no_forward()
    test_factories_return_expected_types()
    test_pick_best_random_tie()
    print("All rollout strategy tests passed.")
