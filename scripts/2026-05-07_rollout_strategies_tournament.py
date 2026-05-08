"""
Round-robin tournament between MCTSRolloutPlayer configs.

Each pair plays NUM_GAMES total — half with player A as P1, half as P2 — so
there's no first-mover bias. Output:
  - EV matrix (row vs col, from row's perspective)
  - Standings sorted by total EV across all opponents
  - Mean game duration per config (avg over all its matchups)

Why round-robin: at any reasonable iteration budget, every contender beats
the (NONE, RANDOM) baseline by a wide margin, so a 1-vs-baseline experiment
saturates near 100% and the ranking among contenders is unreadable. Pitting
contenders against each other puts them on more even footing.
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import math
import multiprocessing
import pickle
from datetime import datetime

import numpy as np

from a0.eval.player import MatchupStats, run_matchup
from a0.game import PlayerClass
from a0.mcts.rollout_strategies import EvaluatorType, PolicyType
from a0.players.mcts_rollout import MCTSRolloutPlayer

from config import config

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)


NUM_GAMES = 64              # games per pair (half as P1, half as P2)
MCTS_ITERATIONS = 512        # tunable: higher = stronger play, slower
EPSILON = 0.1               # for Best/Back rollout policies

# Per-evaluator UCT exploration constant. Defaults are theoretical anchors
# scaled to the current config: √2 (Hoeffding) for NONE, and half-range/√2 for
# DIST/LBDIST where half-range = num_pieces * 2 * (board_size - 1).
def _anchor_c(evaluator: EvaluatorType) -> float:
    if evaluator == EvaluatorType.NONE:
        return math.sqrt(2)
    half_range = config.num_pieces * 2 * (config.board_size - 1)
    return half_range / math.sqrt(2)

TUNED_C = {
    EvaluatorType.NONE: _anchor_c(EvaluatorType.NONE),
    EvaluatorType.DIST: _anchor_c(EvaluatorType.DIST) * 0.25,  # empircally found from c calibration
    EvaluatorType.LBDIST: _anchor_c(EvaluatorType.LBDIST),
}

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
        c=TUNED_C[evaluator],
    )


def _merge(a: MatchupStats, b: MatchupStats) -> MatchupStats:
    return MatchupStats(
        wins=a.wins + b.wins,
        losses=a.losses + b.losses,
        draws_repeat=a.draws_repeat + b.draws_repeat,
        draws_timeout=a.draws_timeout + b.draws_timeout,
        num_turns=a.num_turns + b.num_turns,
        durations=a.durations + b.durations,
    )


def _mirror(s: MatchupStats) -> MatchupStats:
    """Flip the result to the opponent's perspective; turn/duration data is shared."""
    return MatchupStats(
        wins=s.losses,
        losses=s.wins,
        draws_repeat=s.draws_repeat,
        draws_timeout=s.draws_timeout,
        num_turns=list(s.num_turns),
        durations=list(s.durations),
    )


def round_robin(
    players: dict[str, PlayerClass],
    num_games: int,
    game_log_path: str | None = None,
) -> dict[str, dict[str, MatchupStats]]:
    names = list(players.keys())
    results: dict[str, dict[str, MatchupStats]] = {a: {} for a in names}

    half = num_games // 2
    other_half = num_games - half

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            logger.info(f"=== {a} vs {b} ===")
            s_p1 = run_matchup(players[a], players[b], half,
                               focal_first=True,
                               game_log_path=game_log_path, log_games=1,
                               label=f'{a}_vs_{b}_p1')
            s_p2 = run_matchup(players[a], players[b], other_half,
                               focal_first=False,
                               game_log_path=game_log_path, log_games=1,
                               label=f'{a}_vs_{b}_p2')
            combined = _merge(s_p1, s_p2)
            results[a][b] = combined
            results[b][a] = _mirror(combined)
            logger.info(f"  {a} vs {b}: EV={combined.expected_value:+.3f} "
                        f"(W={combined.wins} L={combined.losses} "
                        f"Dr={combined.draws_repeat} Dt={combined.draws_timeout})")
    return results


def _print_ev_matrix(names: list[str], results: dict[str, dict[str, MatchupStats]]) -> None:
    print("\n=== EV matrix (row vs column, from row's perspective) ===")
    col_w = max(12, max(len(n) for n in names) + 1)
    print(f"{'':<{col_w}}" + "".join(f"{n:>10}" for n in names) + f"{'total':>10}")
    for a in names:
        row = f"{a:<{col_w}}"
        total = 0.0
        for b in names:
            if a == b:
                row += f"{'-':>10}"
            else:
                ev = results[a][b].expected_value
                row += f"{ev:>+10.3f}"
                total += ev
        row += f"{total:>+10.3f}"
        print(row)


def _print_standings(names: list[str], results: dict[str, dict[str, MatchupStats]]) -> None:
    print("\n=== Standings (sorted by total EV across all opponents) ===")
    rows = []
    for a in names:
        wins = sum(results[a][b].wins for b in names if b != a)
        losses = sum(results[a][b].losses for b in names if b != a)
        draws = sum(results[a][b].draws_repeat + results[a][b].draws_timeout for b in names if b != a)
        n = wins + losses + draws
        ev = (wins - losses) / n if n > 0 else 0.0
        durs = [d for b in names if b != a for d in results[a][b].durations]
        mean_dur = float(np.mean(durs)) if durs else 0.0
        rows.append((a, ev, wins, losses, draws, n, mean_dur))
    rows.sort(key=lambda r: -r[1])
    print(f"{'config':<16} {'EV':>8} {'W':>5} {'L':>5} {'D':>5} {'N':>5} {'mean_dur_s':>11}")
    for name, ev, w, l, d, n, dur in rows:
        print(f"{name:<16} {ev:>+8.3f} {w:>5} {l:>5} {d:>5} {n:>5} {dur:>11.2f}")


def main() -> None:
    setup_logging(level=20, log_dir=config.log_dir, process_name='rollout_strategies_tournament')
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    players: dict[str, PlayerClass] = {
        'none_random':       make_player(EvaluatorType.NONE,   PolicyType.RANDOM),
        'none_forward':   make_player(EvaluatorType.NONE,   PolicyType.FORWARD),
        'dist_random':    make_player(EvaluatorType.DIST,   PolicyType.RANDOM),
        'dist_forward':   make_player(EvaluatorType.DIST,   PolicyType.FORWARD),
        'dist_best':      make_player(EvaluatorType.DIST,   PolicyType.BEST,    epsilon=EPSILON),
        'dist_back':      make_player(EvaluatorType.DIST,   PolicyType.BACK,    epsilon=EPSILON),
        'lbdist_best':    make_player(EvaluatorType.LBDIST, PolicyType.BEST,    epsilon=EPSILON),
        'lbdist_back':    make_player(EvaluatorType.LBDIST, PolicyType.BACK,    epsilon=EPSILON),
    }
    names = list(players.keys())

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"{config.eval_dir}/rollout_strategies_round_robin_{timestamp}.pkl"
    game_log_path = f"{config.eval_dir}/rollout_strategies_round_robin_games_{timestamp}.json"

    n_pairs = len(names) * (len(names) - 1) // 2
    logger.info(f"Round-robin: {len(names)} configs ({n_pairs} pairs), "
                f"{NUM_GAMES} games per pair, mcts_iterations={MCTS_ITERATIONS}")
    results = round_robin(players, NUM_GAMES, game_log_path=game_log_path)

    # save raw results so they can be re-analyzed without re-running
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump({'names': names, 'results': results,
                     'num_games': NUM_GAMES, 'mcts_iterations': MCTS_ITERATIONS,
                     'epsilon': EPSILON}, f)
    logger.info(f"Saved results to {output_path}")

    _print_ev_matrix(names, results)
    _print_standings(names, results)


if __name__ == "__main__":
    main()
