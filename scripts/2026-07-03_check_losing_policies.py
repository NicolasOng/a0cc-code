"""
TEMPORARY diagnostic script.

Sweeps board states (all of them for 16-3, a random sample for 25-6) and, for
every state that the solver marks as a LOSS for the side to move, checks whether
its 1-ply policy is "trivial".

Definition
----------
A losing state means: under perfect play, the side to move loses no matter what.
Therefore *every* legal move must lead to a position the opponent wins, i.e. all
1-ply outcomes should be -1 (trivial). If we find a losing state whose 1-ply
outcomes are NOT all -1, that is an inconsistency between the loss label and the
move-by-move outcomes (a bug in rank/unrank, move-gen, or the solve data).

Usage
-----
    # 16-3: enumerate every state
    python check_losing_policies.py config/config-16-3.json all

    # 25-6: sample N random states (default 200000)
    python check_losing_policies.py config/config.json 200000

argv[1] -> config path (picked up by config.py)
argv[2] -> "all"/"0" to enumerate everything, or an integer sample count.
"""
import sys
import random

from config import config
from cc.ground_truth import GroundTruth
from cc.solvedata import SolveData


def main() -> None:
    # argv[2]: "all"/"0" => full enumeration, otherwise a sample count.
    n_samples: int | None = 200_000
    if len(sys.argv) > 2:
        arg = sys.argv[2]
        n_samples = None if arg in ("all", "0") else int(arg)

    gt = GroundTruth()
    # mmap the solve file so we don't pull the 2.4 GB 25-6 table into RAM.
    gt.l.solve_data = SolveData(config.solve_data, mmap=True)

    max_rank = gt.get_max_rank()
    print(f"config      = {config.path}")
    print(f"board/pieces= {config.board_size}x{config.board_size} / {config.num_pieces}")
    print(f"solve_data  = {config.solve_data}")
    print(f"max_rank    = {max_rank:,}")

    if n_samples is None or n_samples >= max_rank:
        ranks = range(max_rank)
        total = max_rank
        print(f"mode        = enumerating all {max_rank:,} states\n")
    else:
        rng = random.Random(0)
        ranks = (rng.randrange(max_rank) for _ in range(n_samples))
        total = n_samples
        print(f"mode        = sampling {n_samples:,} random states (seed=0)\n")

    checked = illegal = terminal = losing = nontrivial = 0
    anomalies = []

    for rank in ranks:
        board = gt.unrank(rank)
        checked += 1

        # skip illegal states (solver code 3)
        if gt.is_illegal(board):
            illegal += 1
            continue

        # only interested in states that are a loss for the side to move
        if gt.get_outcome(board) != -1.0:
            continue

        # a terminal loss has no moves to analyse
        if gt.is_terminal(board):
            terminal += 1
            continue

        moves, outcomes = gt.get_1ply_policy_moves(board)
        if not moves:
            terminal += 1
            continue

        losing += 1
        # trivial == every move leads to a loss (all outcomes -1)
        if not all(o == -1.0 for o in outcomes):
            nontrivial += 1
            anomalies.append((rank, board, moves, outcomes))
            if len(anomalies) <= 25:
                print(f"ANOMALY  rank={rank}  to_move={board.current_player}")
                print(board)
                for m, o in zip(moves, outcomes):
                    print(f"    {m} -> {o:+.0f}")
                print()

        if checked % 200_000 == 0:
            print(f"...checked={checked:,}/{total:,}  losing={losing:,}  nontrivial={nontrivial}")

    print("\n==== SUMMARY ====")
    print(f"checked                = {checked:,}")
    print(f"illegal (skipped)      = {illegal:,}")
    print(f"terminal (skipped)     = {terminal:,}")
    print(f"losing states examined = {losing:,}")
    print(f"NON-TRIVIAL losing     = {nontrivial:,}")
    if nontrivial == 0:
        print("\nAll losing states are trivial (every move loses). Consistent. ✅")
    else:
        print(f"\n⚠️  Found {nontrivial:,} losing states whose policy is NOT all-losing.")


if __name__ == "__main__":
    main()
