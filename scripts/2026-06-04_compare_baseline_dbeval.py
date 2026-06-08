'''Head-to-head comparison: the DistEval baseline (make_baseline) vs the DB-eval
player (make_bfs_baseline, which uses the exact single-agent BFS distance plus the
disentangled-rollout early-exit).

Each player is run on both sides (DB-eval as P1 and as P2) to cancel any first-move
advantage, and results are reported from the DB-eval player's perspective. Per-side
numbers are also printed so first-move bias is visible.

Number of games per side comes from the GAMES env var (default 50); sys.argv is left
to config.py (argv[1] = config path, argv[2] = trial num), matching the other scripts.

Run:
    python scripts/2026-06-04_compare_baseline_dbeval.py            # 50 games per side
    GAMES=200 python scripts/2026-06-04_compare_baseline_dbeval.py  # 200 games per side
'''
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from a0.eval.player import make_baseline, make_bfs_baseline, run_matchup, MatchupStats
from config import config
from utils.log import get_logger, setup_logging

logger = get_logger(__name__)

#GAMES_PER_SIDE = int(os.environ.get("GAMES", "50"))
GAMES_PER_SIDE = 32


def _combine(a: MatchupStats, b: MatchupStats) -> MatchupStats:
    '''Merge two focal-perspective MatchupStats (DB-eval as P1 + DB-eval as P2).'''
    return MatchupStats(
        wins=a.wins + b.wins,
        losses=a.losses + b.losses,
        draws_repeat=a.draws_repeat + b.draws_repeat,
        draws_timeout=a.draws_timeout + b.draws_timeout,
        num_turns=a.num_turns + b.num_turns,
        durations=a.durations + b.durations,
    )


def _fmt(stats: MatchupStats) -> str:
    return (f"EV={stats.expected_value:+.3f} (W={stats.wins} L={stats.losses} "
            f"Dr={stats.draws_repeat} Dt={stats.draws_timeout})")


def _print_summary(stats: MatchupStats) -> None:
    n = max(stats.num_games, 1)
    print()
    print("=" * 60)
    print("  DB-eval player  vs  DistEval baseline")
    print("  (results from the DB-eval player's perspective)")
    print("=" * 60)
    print(f"  games:           {stats.num_games}")
    print(f"  wins:            {stats.wins} ({stats.wins / n:.1%})")
    print(f"  losses:          {stats.losses} ({stats.losses / n:.1%})")
    print(f"  draws (repeat):  {stats.draws_repeat} ({stats.draws_repeat / n:.1%})")
    print(f"  draws (timeout): {stats.draws_timeout} ({stats.draws_timeout / n:.1%})")
    print(f"  expected value:  {stats.expected_value:+.3f}   [(W - L) / N, in [-1, 1]]")
    print("=" * 60)


def main() -> None:
    setup_logging(level=20, log_dir=config.log_dir, process_name='compare_baseline_dbeval')
    logger.info(f"Comparing on board {config.board_size}, {config.num_pieces} pieces; "
                f"{GAMES_PER_SIDE} games per side.")

    db_player = make_bfs_baseline()   # focal: the new DB-eval player
    baseline = make_baseline()        # opponent: the DistEval baseline

    logger.info("DB-eval as P1 vs baseline...")
    s1 = run_matchup(db_player, baseline, GAMES_PER_SIDE, focal_first=True, label='db_p1')

    logger.info("DB-eval as P2 vs baseline...")
    s2 = run_matchup(db_player, baseline, GAMES_PER_SIDE, focal_first=False, label='db_p2')

    print(f"\nDB-eval as P1: {_fmt(s1)}")
    print(f"DB-eval as P2: {_fmt(s2)}")
    _print_summary(_combine(s1, s2))


if __name__ == "__main__":
    main()
