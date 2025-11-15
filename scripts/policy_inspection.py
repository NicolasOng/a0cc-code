from cc.core import Game, Move, Player, Board
from a0.mcts.nn import MCTS_NN
from a0.graph_search.mcts import MCTS
from a0.model import load_model
from cc.ground_truth import GroundTruth
from a0.model_utils import board_to_input, Policy
from a0.players.a0 import A0Player
from scripts.sl_on_policy_head import generate_mcts_policy
from a0.eval.dataset_evaluation import policy_accuracy_function

import random

import jax.numpy as jnp
import numpy as np

from config import config

def get_random_board(gt: GroundTruth, cc: Game) -> Board:
    while True:
        max_rank = gt.get_max_rank()
        rank = random.randint(0, max_rank - 1)
        random_board = gt.unrank(rank)
        if not gt.is_trivial(random_board) and not cc.get_done(random_board):
            break
    return random_board

def print_board(board: Board) -> None:
    print(board.board_view())
    print(board.current_player)

def get_player(model_no: int) -> A0Player:
    trained_model = load_model(config.training_dir + f"model_{model_no}.pkl")
    player = A0Player(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        model=trained_model,
        exploit=True,
        mcts_samples=64,
        no_reverse_moves=True,
        no_side_moves=False
    )
    return player

def print_policy(policy, board: Board, moves: list[Move]) -> None:
    '''
    Prints the policy distribution
    '''
    p = Policy(len(board.board))
    p.set_logits(np.array(policy), rotate_180=board.current_player == Player.PLAYER_O)
    move_probs = p.get_move_probabilities(moves)
    for prob in move_probs:
        print(f"{prob:.2%}", end=" ")
    #print(move_probs)
    print(f"\n{sum(move_probs):.2f}")  # should be 1.0

def main():
    gt = GroundTruth()
    g = Game(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        no_reverse_moves=True,
        no_illegal_moves=False,
        no_side_moves=False
    )

    random_board = get_random_board(gt, g)
    moves = g.generate_moves_for_given_board(random_board)
    print_board(random_board)

    gt_policy = gt.get_1ply_policy_prob_dist_list(random_board, for_model=True)
    print_policy(gt_policy, random_board, moves)

    player = get_player(450)
    _, policy = player.get_value_and_policy(random_board, moves)
    print_policy(policy[0], random_board, moves)
    print(policy_accuracy_function(policy[0], np.array(gt_policy)))

    mcts_gt_policy, _ = generate_mcts_policy(random_board, g, 0.2, 64)
    print_policy(mcts_gt_policy, random_board, moves)
    print(policy_accuracy_function(mcts_gt_policy, np.array(gt_policy)))

main()
