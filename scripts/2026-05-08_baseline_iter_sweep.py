"""
Sweep the baseline player's mcts_iterations to find the operating point where
a trained A0Player wins enough to be visible-as-improving but not so much that
the gap is saturated. Also reports per-iter game timing for SLURM budgeting.

Output: per-iter table with trained EV, W-L-D, mean/max per-game CPU duration,
matchup wall time, and per-game wall time. The per-game wall time is the
useful number for budgeting larger eval runs.
"""
import sys
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import math
import multiprocessing
import pickle
import time
from datetime import datetime

import numpy as np

from a0.eval.player import (
    BASELINE_EPSILON,
    MatchupStats,
    _baseline_c,
    run_matchup,
)
from a0.mcts.rollout_strategies import EvaluatorType, PolicyType
from a0.model import load_model
from a0.players.a0 import A0Player
from a0.players.mcts_rollout import MCTSRolloutPlayer

from config import config

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)


# The checkpoint to sweep. Pass a path as argv[2] (argv[1] is the config, read
# by config.py) or set A0_MODEL_PATH.
MODEL_PATH = (sys.argv[2] if len(sys.argv) > 2
              else os.environ.get("A0_MODEL_PATH", "output/training/model_49.pkl"))

ITER_SWEEP = (64, 128, 256, 512, 1024, 2048)
NUM_GAMES = 64                       # half as P1 (trained), half as P2
TARGET_EV_BAND = (0.2, 0.7)          # what counts as a useful operating point
EVAL_NUM_GAMES_FOR_BUDGET = 64       # we estimate runtime for an eval at this N


# Mirror a0.eval.player.make_baseline but parametric on iters.
# (Keep this in sync with player.py's make_baseline.)
def make_baseline(iters: int) -> MCTSRolloutPlayer:
    return MCTSRolloutPlayer(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        no_reverse_moves=not config.backwards_moves,
        no_illegal_moves=True,
        no_side_moves=not config.sideways_moves,
        mcts_iterations=iters,
        rollout_depth=8,
        evaluator=EvaluatorType.DIST,
        policy=PolicyType.BEST,
        policy_epsilon=BASELINE_EPSILON,
        c=_baseline_c(),
    )


def make_trained() -> A0Player:
    model = load_model(MODEL_PATH, training=False)
    return A0Player(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        model=model,
        exploit=True,
        mcts_samples=config.mcts_samples,
        no_reverse_moves=not config.backwards_moves,
        no_illegal_moves=not config.illegal_moves,
        no_side_moves=not config.sideways_moves,
        rollout_type=config.rollout_type,
        rollout_depth=config.rollout_depth,
        policy_type=config.policy_type,
        epsilon=config.epsilon,
        dirichlet_epsilon=config.dirichlet_epsilon,
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


def main() -> None:
    setup_logging(level=20, log_dir=config.log_dir, process_name='baseline_iter_sweep')
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    trained = make_trained()
    half = NUM_GAMES // 2
    other_half = NUM_GAMES - half

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"{config.eval_dir}/baseline_iter_sweep_{timestamp}.pkl"

    rows: list[dict] = []
    overall_start = time.perf_counter()
    for iters in ITER_SWEEP:
        logger.info(f"=== mcts_iterations={iters} ===")
        baseline = make_baseline(iters)

        wall0 = time.perf_counter()
        s_p1 = run_matchup(trained, baseline, half, focal_first=True,
                           label=f'trained_p1_baseline_iter{iters}')
        s_p2 = run_matchup(trained, baseline, other_half, focal_first=False,
                           label=f'trained_p2_baseline_iter{iters}')
        wall = time.perf_counter() - wall0

        combined = _merge(s_p1, s_p2)  # from trained's perspective
        durs = np.asarray(combined.durations, dtype=float) if combined.durations else np.array([0.0])
        rows.append({
            'iters': iters,
            'wins': combined.wins,
            'losses': combined.losses,
            'draws': combined.draws_repeat + combined.draws_timeout,
            'ev': combined.expected_value,
            'mean_dur_cpu_s': float(durs.mean()),
            'max_dur_cpu_s': float(durs.max()),
            'matchup_wall_s': wall,
            'per_game_wall_s': wall / NUM_GAMES,
        })
        logger.info(f"  iters={iters}: trained EV={combined.expected_value:+.3f} "
                    f"({combined.wins}-{combined.losses}-{combined.draws_repeat + combined.draws_timeout}) "
                    f"matchup_wall={wall:.1f}s, per-game wall={wall / NUM_GAMES:.2f}s")

    total_wall = time.perf_counter() - overall_start

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump({
            'rows': rows,
            'num_games': NUM_GAMES,
            'iter_sweep': ITER_SWEEP,
            'model_path': MODEL_PATH,
            'num_workers': config.num_workers,
            'board_size': config.board_size,
            'num_pieces': config.num_pieces,
            'total_wall_s': total_wall,
        }, f)
    logger.info(f"Saved results to {output_path}")

    # --- Summary ---
    print(f"\n=== Baseline mcts_iterations sweep "
          f"({NUM_GAMES} games/level, {config.num_workers} workers, "
          f"model={os.path.basename(MODEL_PATH)}) ===")
    print(f"{'iters':>6} {'EV':>7} {'W':>4} {'L':>4} {'D':>4} "
          f"{'mean_cpu_s':>11} {'max_cpu_s':>11} {'wall_s':>9} {'s/game':>9}")
    for r in rows:
        in_band = TARGET_EV_BAND[0] <= r['ev'] <= TARGET_EV_BAND[1]
        mark = " <-- in band" if in_band else ""
        print(f"{r['iters']:>6} {r['ev']:>+7.3f} {r['wins']:>4} {r['losses']:>4} {r['draws']:>4} "
              f"{r['mean_dur_cpu_s']:>11.2f} {r['max_dur_cpu_s']:>11.2f} "
              f"{r['matchup_wall_s']:>9.1f} {r['per_game_wall_s']:>9.2f}{mark}")
    print(f"\nTotal sweep wall time: {total_wall:.1f}s ({total_wall / 60:.1f} min)")

    # --- Recommendation ---
    in_band = [r for r in rows if TARGET_EV_BAND[0] <= r['ev'] <= TARGET_EV_BAND[1]]
    if in_band:
        target = sum(TARGET_EV_BAND) / 2
        best = min(in_band, key=lambda r: abs(r['ev'] - target))
        print(f"\nRecommended baseline iter count: {best['iters']} "
              f"(trained EV={best['ev']:+.3f}, ~{best['per_game_wall_s']:.2f}s/game wall)")
    else:
        print(f"\nNo iter level produced trained EV in target band {TARGET_EV_BAND}.")
        print("  - If EV is consistently near +1: trained is too strong. Use a weaker checkpoint.")
        print("  - If EV is consistently near -1: trained is too weak. Use a stronger checkpoint.")
        print("  - Or extend ITER_SWEEP at the appropriate end.")

    # --- SLURM budget hint ---
    print(f"\n=== SLURM budget for an eval of {EVAL_NUM_GAMES_FOR_BUDGET} games "
          f"({config.num_workers} workers) ===")
    scale = EVAL_NUM_GAMES_FOR_BUDGET / NUM_GAMES
    print(f"{'iters':>6} {'projected_wall':>16}")
    for r in rows:
        proj = r['matchup_wall_s'] * scale
        print(f"{r['iters']:>6} {proj:>10.1f}s ({proj / 60:>5.1f} min)")
    print("(Linear extrapolation from this sweep's per-game wall time. "
          "Add ~10-30% headroom for variance and process startup.)")


if __name__ == "__main__":
    main()
