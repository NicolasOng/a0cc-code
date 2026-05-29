"""
Single-agent BFS over P1 piece configurations.

Port of DoBFS from InitialParallelSolver.cpp (the chinese-checkers C++ solver).
Seeds at the P1 goal configuration (rank = maxP1Rank - 1) and explores outward
via legal P1 moves with the opponent absent from the board. The resulting
depth array gives, for each P1 piece arrangement, the minimum number of
single-agent P1 moves needed to reach the goal.

Output: a numpy int16 array of length maxP1Rank = C(board_size**2, num_pieces).
bfs[x] is the BFS depth of P1 rank x. -1 means unreachable (which would later
hang GetSearchOrder if used as an ordering input).

Usage:
    python scripts/2026-05-29_single_agent_bfs.py --board-size 4 --num-pieces 2
    python scripts/2026-05-29_single_agent_bfs.py --board-size 7 --num-pieces 4 --output bfs_7_4.npy
"""
import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from cc.core import Game
from cc.ranking import CCState, CCPSRank12


def run_bfs(board_size: int, num_pieces: int, verbose: bool = True) -> np.ndarray:
    num_spots = board_size ** 2

    ranker = CCPSRank12(num_spots, 2, num_pieces)
    # no_illegal_moves=False: we want raw piece movements, not game-rule-filtered moves.
    game = Game(board_size=board_size, num_pieces=num_pieces, no_illegal_moves=False)

    max_p1 = ranker.get_max_p1_rank()
    bfs = np.full(max_p1, -1, dtype=np.int16)
    bfs[max_p1 - 1] = 0

    parent = CCState(num_spots, num_pieces, 2)
    child = CCState(num_spots, num_pieces, 2)

    depth = 0
    total = 1
    while True:
        written = 0
        frontier = np.where(bfs == depth)[0]
        for x in frontier:
            x = int(x)
            ranker.unrank_p1(x, parent)
            board = parent.get_board()
            moves = game.generate_moves_for_given_board(board)
            for move in moves:
                board.apply_move(move)
                child.initialize_from_board(board)
                # Mirrors the C++ trick at InitialParallelSolver.cpp:129 — force
                # to_move = 0 so rank_p1 takes the unflipped branch.
                child.to_move = 0
                new_rank = ranker.rank_p1(child)
                if bfs[new_rank] == -1:
                    bfs[new_rank] = depth + 1
                    written += 1
                board.undo_move(move)

        total += written
        if verbose:
            print(f"Depth {depth} complete. {written} new. {total} of {max_p1} complete")

        if written == 0:
            break
        depth += 1

    return bfs


def print_stats(bfs: np.ndarray, elapsed: float) -> None:
    max_p1 = len(bfs)
    unreachable = int(np.sum(bfs == -1))
    reachable = max_p1 - unreachable
    max_depth = int(bfs.max())
    ranks_at_max = int(np.sum(bfs == max_depth))
    start_depth = int(bfs[0])

    print()
    print("=== BFS complete ===")
    print(f"Wallclock:           {elapsed:.2f} s")
    print(f"Reachable:           {reachable:,} / {max_p1:,} ({100.0 * reachable / max_p1:.2f}%)")
    print(f"Unreachable (-1):    {unreachable:,}")
    print(f"Max depth:           {max_depth}")
    print(f"Ranks at max depth:  {ranks_at_max:,}")
    print(f"Start state depth:   {start_depth}    (bfs[0] — min single-agent moves to win)")
    if unreachable > 0:
        print()
        print("WARNING: some ranks are unreachable. This array would hang GetSearchOrder.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-size", type=int, required=True)
    parser.add_argument("--num-pieces", type=int, required=True)
    parser.add_argument("--output", type=str, default=None,
                        help="Output .npy path (default: bfs_<size>_<pieces>.npy)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-depth progress prints")
    args = parser.parse_args()

    output_path = args.output or f"bfs_{args.board_size}_{args.num_pieces}.npy"

    start = time.perf_counter()
    bfs = run_bfs(args.board_size, args.num_pieces, verbose=not args.quiet)
    elapsed = time.perf_counter() - start

    np.save(output_path, bfs)
    print_stats(bfs, elapsed)
    print()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
