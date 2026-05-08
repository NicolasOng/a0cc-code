"""
Microbenchmark for the rollout strategies.

Each config calls MCTSRolloutPlayer.select_move on a fixed seed-set of
boards (start position + several random walks at varying depth). Single
process, no opponent, fixed RNG — so timing differences between configs
reflect the strategies' actual per-iteration cost rather than parallelism
jitter or game-length effects.

Reports mean / std time per select_move, time per MCTS iteration, and the
multiplier vs the (NONE, RANDOM) baseline.
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import random
import statistics
import time

from cc.core import Board, Game

from a0.mcts.rollout_strategies import EvaluatorType, PolicyType
from a0.players.mcts_rollout import MCTSRolloutPlayer

from config import config


MCTS_ITERATIONS = 256
SEED = 0
RANDOM_WALK_DEPTHS = (0, 8, 20, 40, 60)   # 0 = starting position
TRIALS_PER_BOARD = 2                       # repeats for each (config, board)


def _make_eval_game() -> Game:
    return Game(
        config.board_size,
        config.num_pieces,
        no_reverse_moves=not config.backwards_moves,
        no_illegal_moves=not config.illegal_moves,
        no_side_moves=not config.sideways_moves,
    )


def make_test_boards() -> list[tuple[int, Board]]:
    """A list of (depth, board) pairs from a fixed-seed random walk off the start."""
    rng = random.Random(SEED)
    g = _make_eval_game()
    out: list[tuple[int, Board]] = []

    state = Board()
    state.copy_board(g.board)
    last_emitted = -1
    for depth in range(max(RANDOM_WALK_DEPTHS) + 1):
        if depth in RANDOM_WALK_DEPTHS and depth != last_emitted:
            snap = Board()
            snap.copy_board(state)
            out.append((depth, snap))
            last_emitted = depth
        if g.get_done(state):
            break
        moves = g.generate_moves_for_given_board(state)
        if not moves:
            break
        m = rng.choice(moves)
        new = Board()
        new.copy_board(state)
        new.apply_move(m)
        state = new
    return out


def make_player(evaluator: EvaluatorType, policy: PolicyType, epsilon: float = 0.0) -> MCTSRolloutPlayer:
    return MCTSRolloutPlayer(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        no_reverse_moves=not config.backwards_moves,
        no_illegal_moves=not config.illegal_moves,
        no_side_moves=not config.sideways_moves,
        mcts_iterations=MCTS_ITERATIONS,
        evaluator=evaluator,
        policy=policy,
        policy_epsilon=epsilon,
    )


def benchmark(player: MCTSRolloutPlayer, boards: list[tuple[int, Board]]) -> list[float]:
    g = player.game
    times: list[float] = []
    for _depth, board in boards:
        moves = g.generate_moves_for_given_board(board)
        if not moves:
            continue
        for _ in range(TRIALS_PER_BOARD):
            t0 = time.perf_counter()
            player.select_move(board, moves)
            t1 = time.perf_counter()
            times.append(t1 - t0)
    return times


def main() -> None:
    boards = make_test_boards()
    print(f"Benchmark: {len(boards)} boards (depths={[d for d, _ in boards]}), "
          f"{TRIALS_PER_BOARD} trials each, mcts_iterations={MCTS_ITERATIONS}")

    configs: list[tuple[str, EvaluatorType, PolicyType, float]] = [
        ('baseline',      EvaluatorType.NONE,   PolicyType.RANDOM,  0.0),
        ('none_forward',  EvaluatorType.NONE,   PolicyType.FORWARD, 0.0),
        ('dist_random',   EvaluatorType.DIST,   PolicyType.RANDOM,  0.0),
        ('dist_forward',  EvaluatorType.DIST,   PolicyType.FORWARD, 0.0),
        ('dist_best',     EvaluatorType.DIST,   PolicyType.BEST,    0.1),
        ('dist_back',     EvaluatorType.DIST,   PolicyType.BACK,    0.1),
        ('lbdist_best',   EvaluatorType.LBDIST, PolicyType.BEST,    0.1),
        ('lbdist_back',   EvaluatorType.LBDIST, PolicyType.BACK,    0.1),
    ]

    # warm up the baseline once so first-call overhead doesn't skew it
    warmup = make_player(EvaluatorType.NONE, PolicyType.RANDOM)
    benchmark(warmup, boards[:1])

    results: list[tuple[str, float, float, float]] = []
    for name, ev, po, eps in configs:
        player = make_player(ev, po, eps)
        times = benchmark(player, boards)
        mean = statistics.mean(times)
        stdev = statistics.stdev(times) if len(times) > 1 else 0.0
        per_iter_us = (mean / MCTS_ITERATIONS) * 1e6
        results.append((name, mean, stdev, per_iter_us))
        print(f"  {name:<14} mean={mean:.3f}s std={stdev:.3f}s per_iter={per_iter_us:.1f}us")

    baseline_mean = next(m for n, m, _, _ in results if n == 'baseline')

    print(f"\n=== Per-config timing ({len(boards) * TRIALS_PER_BOARD} samples each) ===")
    print(f"{'config':<14} {'mean_s':>8} {'std_s':>8} {'us/iter':>9} {'vs_baseline':>13}")
    for name, mean, stdev, per_iter_us in results:
        ratio = mean / baseline_mean
        marker = '' if name == 'baseline' else f"{ratio:.2f}x"
        print(f"{name:<14} {mean:>8.3f} {stdev:>8.3f} {per_iter_us:>9.1f} {marker:>13}")


if __name__ == "__main__":
    main()
