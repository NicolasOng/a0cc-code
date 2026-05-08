"""
Pre-tournament c calibration for MCTSRolloutPlayer rollout strategies.

For each evaluator family (NONE, DIST, LBDIST), sweep c around a theoretical
anchor and pick the value that wins a self-play round-robin within the family.
Self-play isolates c as the only varying parameter — no cross-eval contamination.

Output: best c per evaluator, ready to paste into the main tournament script.

Cost: 3 evaluators * 3 c-values * 3 pairs * NUM_GAMES games. With NUM_GAMES=16
that's 144 games; at MCTS_ITERATIONS=512 with 32 workers, runs in ~minutes.
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import math
import multiprocessing
import pickle
from datetime import datetime

from a0.eval.player import MatchupStats, run_matchup
from a0.mcts.rollout_strategies import EvaluatorType, PolicyType
from a0.players.mcts_rollout import MCTSRolloutPlayer

from config import config

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)


NUM_GAMES = 8                                # per pair (half as P1, half as P2)
MCTS_ITERATIONS = 256                         # match the main tournament
EPSILON = 0.1
CALIBRATION_POLICY = PolicyType.FORWARD       # neutral non-trivial policy

def anchor_c_for(evaluator: EvaluatorType) -> float:
    """Theoretical c starting point per evaluator, scaled to the current config.

    NONE: rewards in [-1, 1] (terminal ±1, depth-cap 0) → Hoeffding's √2.
    DIST/LBDIST: raw integer rewards. Half-range ≈ num_pieces * 2 * (board_size - 1)
        since each piece can be at most 2*(board_size-1) Manhattan from its goal
        corner. Anchor at half-range / √2 (a "half-Hoeffding" — Hoeffding is loose
        in practice, so we start a factor of 2 below it).
    """
    if evaluator == EvaluatorType.NONE:
        return math.sqrt(2)
    half_range = config.num_pieces * 2 * (config.board_size - 1)
    return half_range / math.sqrt(2)


ANCHOR_C: dict[EvaluatorType, float] = {ev: anchor_c_for(ev) for ev in EvaluatorType}

# Multiplicative sweep around the anchor — coarse on purpose; we just want the
# right basin, then the main tournament tells us if we need a finer sweep.
SWEEP_FACTORS = (0.25, 1.0, 4.0)


def make_player(evaluator: EvaluatorType, policy: PolicyType, epsilon: float, c: float) -> MCTSRolloutPlayer:
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
        c=c,
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
    return MatchupStats(
        wins=s.losses,
        losses=s.wins,
        draws_repeat=s.draws_repeat,
        draws_timeout=s.draws_timeout,
        num_turns=list(s.num_turns),
        durations=list(s.durations),
    )


def calibrate_evaluator(
    evaluator: EvaluatorType, anchor: float,
) -> tuple[float, list[float], dict[float, dict[float, MatchupStats]]]:
    """Round-robin c-sweep within one evaluator. Returns (best_c, sorted_cs, results)."""
    cs = sorted(anchor * f for f in SWEEP_FACTORS)
    logger.info(f"  Sweeping c ∈ {[f'{c:g}' for c in cs]}")

    players = {c: make_player(evaluator, CALIBRATION_POLICY, EPSILON, c) for c in cs}
    results: dict[float, dict[float, MatchupStats]] = {c: {} for c in cs}

    half = NUM_GAMES // 2
    other_half = NUM_GAMES - half
    for i, c_a in enumerate(cs):
        for c_b in cs[i + 1:]:
            logger.info(f"    c={c_a:g} vs c={c_b:g}")
            s1 = run_matchup(players[c_a], players[c_b], half, focal_first=True,
                             label=f'{evaluator.name}_c{c_a:g}_vs_c{c_b:g}_p1')
            s2 = run_matchup(players[c_a], players[c_b], other_half, focal_first=False,
                             label=f'{evaluator.name}_c{c_a:g}_vs_c{c_b:g}_p2')
            combined = _merge(s1, s2)
            results[c_a][c_b] = combined
            results[c_b][c_a] = _mirror(combined)

    totals = {c: sum(results[c][c2].expected_value for c2 in cs if c2 != c) for c in cs}
    best_c = max(totals, key=lambda c: totals[c])
    return best_c, cs, results


def main() -> None:
    setup_logging(level=20, log_dir=config.log_dir, process_name='c_calibration')
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"{config.eval_dir}/c_calibration_{timestamp}.pkl"

    best_cs: dict[EvaluatorType, float] = {}
    cs_per_eval: dict[EvaluatorType, list[float]] = {}
    all_results: dict[EvaluatorType, dict[float, dict[float, MatchupStats]]] = {}

    for evaluator, anchor in ANCHOR_C.items():
        logger.info(f"=== Calibrating {evaluator.name} (anchor c={anchor:g}) ===")
        best_c, cs, results = calibrate_evaluator(evaluator, anchor)
        best_cs[evaluator] = best_c
        cs_per_eval[evaluator] = cs
        all_results[evaluator] = results
        logger.info(f"  -> best c for {evaluator.name}: {best_c:g}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump({
            'best_cs': {k.name: v for k, v in best_cs.items()},
            'cs_per_eval': {k.name: v for k, v in cs_per_eval.items()},
            'all_results': {k.name: v for k, v in all_results.items()},
            'num_games': NUM_GAMES,
            'mcts_iterations': MCTS_ITERATIONS,
            'epsilon': EPSILON,
            'calibration_policy': CALIBRATION_POLICY.name,
            'sweep_factors': SWEEP_FACTORS,
            'anchor_c': {k.name: v for k, v in ANCHOR_C.items()},
        }, f)
    logger.info(f"Saved results to {output_path}")

    # --- Summary ---
    print(f"\n=== Calibration results "
          f"(policy={CALIBRATION_POLICY.name}, {NUM_GAMES} games/pair, {MCTS_ITERATIONS} iters) ===")
    for evaluator in ANCHOR_C:
        anchor = ANCHOR_C[evaluator]
        cs = cs_per_eval[evaluator]
        results = all_results[evaluator]
        print(f"\n{evaluator.name} (anchor={anchor:g})")
        print(f"  {'c':>10} {'total EV':>10} {'W-L by opponent':>30}")
        for c in cs:
            total_ev = sum(results[c][c2].expected_value for c2 in cs if c2 != c)
            opps = ", ".join(f"vs {c2:g}: {results[c][c2].wins}-{results[c][c2].losses}"
                              for c2 in cs if c2 != c)
            mark = " <-- best" if c == best_cs[evaluator] else ""
            print(f"  {c:>10g} {total_ev:>+10.3f}   {opps}{mark}")

    print("\n=== Paste into the tournament's make_player call ===")
    print("TUNED_C = {")
    for evaluator, c in best_cs.items():
        print(f"    EvaluatorType.{evaluator.name}: {c:.4g},")
    print("}")


if __name__ == "__main__":
    main()
