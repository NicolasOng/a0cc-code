from __future__ import annotations
from typing import Any

from cc.core import Board, Move
from cc.ground_truth import GroundTruth
from a0.model_utils import Policy

class GroundTruthPlayer:
    def __init__(self, print_info: bool = False):
        # data from solve data file and game details in config
        self.gt = GroundTruth()
        self.print_info = print_info

    def select_move(self, state: Board, moves: list[Move]) -> tuple[Move, Any]:
        '''
        Selects a move based on the ground truth solve data.
        '''
        # get the ground truth moves and the outcome of each move.
        gt_moves, gt_outcomes = self.gt.get_1ply_policy_moves(state)

        # put this info in a policy object
        p = Policy(len(state.board))
        p.set_logits_from_moves(gt_moves, gt_outcomes, rotate_180=False)
        p.set_legal_moves(moves)
        p.apply_softmax(temperature=1.0, mask=True)

        # select a move based on the resultant policy distribution
        selected_move = p.get_best_move()
        #selected_move = p.sample_move(42)

        # print some data
        if self.print_info:
            total_moves = len(gt_moves)
            count_wins = gt_outcomes.count(1)
            count_losses = gt_outcomes.count(-1)
            count_draws = gt_outcomes.count(0)
            print("GT Player Info:"
                f"\nTotal Moves: {total_moves},"
                f"\nWins: {count_wins} ({count_wins / total_moves if total_moves > 0 else 0:.2%}),"
                f"\nLosses: {count_losses} ({count_losses / total_moves if total_moves > 0 else 0:.2%}),"
                f"\nDraws: {count_draws} ({count_draws / total_moves if total_moves > 0 else 0:.2%})")


        # return the selected move and the mcts policy distribution
        return selected_move, None
