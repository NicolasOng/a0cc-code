from a0.eval.training_data import get_baseline_accuracy
from cc.ground_truth import GroundTruth
from cc.core import Board, Game
from scripts.inspect_player_policy import get_random_board

import numpy as np

from config import config

def baseline_acc():
    cc = Game(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        no_reverse_moves=not config.backwards_moves
    )
    gt = GroundTruth()
    max_rank = gt.get_max_rank()
    board_set: set[Board] = set()
    for i in range(100):
        # get a random rank:
        rank = np.random.randint(0, max_rank - 1)
        b = gt.unrank(rank)
        # if not gt.is_trivial(b):# and not cc.get_done(b):
        #     board_set.add(b)
        # b = get_random_board(gt, gt.cc)
        board_set.add(b)
        
    value, policy, value_nd, policy_nt = get_baseline_accuracy(list(board_set), gt)
    print(f"Baseline Accuracies: {value:.2%}, {value_nd:.2%}, {policy:.2%}, {policy_nt:.2%}")

baseline_acc()