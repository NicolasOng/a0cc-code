from cc.core import Board
from cc.ground_truth import GroundTruth

import random

gt = GroundTruth()
for i in range(1):
    r = random.randint(0, gt.get_max_rank() - 1)
    board = gt.unrank(r)
    print(f"rank: {r}")
    print(board.board_view())
    print(board.current_player)
    print(f"trivial: {gt.is_trivial(board)}")
    print(f"outcome: {gt.get_outcome(board)}")

    board.flip_vertical()
    print(board.board_view())
    print(f"trivial: {gt.is_trivial(board)}")
    print(f"outcome: {gt.get_outcome(board)}")
