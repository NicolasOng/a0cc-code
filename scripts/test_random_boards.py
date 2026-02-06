from cc.core import Board, Game
from cc.ground_truth import GroundTruth
from a0.model_utils import Policy

import random

gt = GroundTruth()
cc = Game(5, 6, False, False, False)
for i in range(1):
    r = random.randint(0, gt.get_max_rank() - 1)
    board = gt.unrank(r)
    moves = cc.generate_moves_for_given_board(board)

    print(f"rank: {r}")
    print(board.visualize_move_ends(moves))
    print(board.current_player)
    print(f"trivial: {gt.is_trivial(board)}")
    print(f"outcome: {gt.get_outcome(board)}")

    probs: list[float] = list(range(len(moves)))

    p = Policy(len(board.board))
    p.set_logits_from_moves(moves, probs, rotate_180=False)
    print(p.get_move_probabilities(moves))
    p.move_to_policy_index(moves[0])
    
    p.flip_policy(horizontal=False)
    board.flip_vertical()
    moves = cc.generate_moves_for_given_board(board)
    print(board.visualize_move_ends(moves))
    print(p.get_move_probabilities(moves))
    p.move_to_policy_index(moves[0])

    board.flip_vertical()
    print(board.board_view())
    print(f"trivial: {gt.is_trivial(board)}")
    print(f"outcome: {gt.get_outcome(board)}")


