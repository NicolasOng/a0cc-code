import time

from a0.model import AlphaZeroModel, load_model
from cc.core import Board, Move, Player, Game
from a0.model_utils import board_to_input, Policy
from a0.eval.dataset_evaluation import policy_accuracy_function
from cc.ground_truth import GroundTruth
from a0.model_utils import get_value_head_policy, get_policy_head_policy, get_value_head_policy2

from scripts.policy_inspection import print_policy
from scripts.sl_on_policy_head import get_n_random_states

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def combined_head_policy(model: AlphaZeroModel, state: Board, moves: list[Move], blend: float = 0.5, for_model: bool = False) -> NDArray[np.float32]:
    php, _ = get_policy_head_policy(model, state, moves, for_model=for_model)
    vhp, _ = get_value_head_policy(model, state, moves, for_model=for_model)
    combined_policy = blend * php + (1 - blend) * vhp
    return np.array(combined_policy / np.sum(combined_policy), dtype=np.float32)

def test_accuracies(n: int):
    gt = GroundTruth()
    model = load_model(config.training_dir + "model_450.pkl")
    game = Game(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        no_reverse_moves=False,
        no_illegal_moves=False,
        no_side_moves=False
    )
    print("Generating random states...")
    states = get_n_random_states(n, remove_trivial=True, remove_terminal=True)

    total_boards = 0
    ph_acc_count = 0
    vh_acc_count = 0
    ch_acc_count = 0
    for state in states:
        print(f"Testing board {total_boards + 1}/{len(states)}")
        legal_moves = game.generate_moves_for_given_board(state)

        gt_policy = gt.get_1ply_policy_prob_dist_list(state, for_model=False)

        ph_policy, _ = get_policy_head_policy(model, state, legal_moves, for_model=False)
        vh_policy, _ = get_value_head_policy(model, state, legal_moves, for_model=False)
        #ch_policy = combined_head_policy(model, state, legal_moves, blend=0.5, for_model=False)

        # print(state.board_view())
        # print(state.current_player)
        # # this function assumes the policy is for model. need to fix.
        # print_policy(list(ph_policy), state, legal_moves)
        # print_policy(list(vh_policy), state, legal_moves)

        ph_acc = policy_accuracy_function(ph_policy, np.array(gt_policy))
        vh_acc = policy_accuracy_function(vh_policy, np.array(gt_policy))
        #ch_acc = policy_accuracy_function(ch_policy, np.array(gt_policy))

        total_boards += 1
        if ph_acc:
            ph_acc_count += 1
        if vh_acc:
            vh_acc_count += 1
        #if ch_acc:
        #    ch_acc_count += 1
    
    logger.log(25, f"Policy Head Accuracy: {ph_acc_count}/{total_boards} = {ph_acc_count / total_boards:.2%}")
    logger.log(25, f"Value Head Accuracy: {vh_acc_count}/{total_boards} = {vh_acc_count / total_boards:.2%}")
    #logger.info(f"Combined Head Accuracy: {ch_acc_count}/{total_boards} = {ch_acc_count / total_boards:.2%}")

def inspect_new_policies():
    gt = GroundTruth()
    model = load_model(config.training_dir + "model_450.pkl")
    game = Game(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        no_reverse_moves=False,
        no_illegal_moves=False,
        no_side_moves=False
    )
    states = get_n_random_states(10, remove_trivial=True, remove_terminal=True)

    for state in states:
        legal_moves = game.generate_moves_for_given_board(state)

        gt_policy = gt.get_1ply_policy_prob_dist_list(state, for_model=False)

        ph_policy, _ = get_policy_head_policy(model, state, legal_moves, for_model=False)
        vh_policy, _ = get_value_head_policy(model, state, legal_moves, for_model=False)

        ph_acc = policy_accuracy_function(ph_policy, np.array(gt_policy))
        vh_acc = policy_accuracy_function(vh_policy, np.array(gt_policy))

        print("State:")
        print(state.board_view())
        print(state.current_player)
        print("Ground Truth Policy:")
        print_policy(list(gt_policy), state, legal_moves, policy_is_for_model=False)
        print("Policy Head Policy:")
        print_policy(list(ph_policy), state, legal_moves, policy_is_for_model=False)
        print(f"Policy Head Accuracy: {ph_acc:.2%}")
        print("Value Head Policy:")
        print_policy(list(vh_policy), state, legal_moves, policy_is_for_model=False)
        print(f"Value Head Accuracy: {vh_acc:.2%}")

def time_new_policy_functions():
    model = load_model(config.training_dir + "model_450.pkl")
    game = Game(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        no_reverse_moves=False,
        no_illegal_moves=False,
        no_side_moves=False
    )
    states = get_n_random_states(100, remove_trivial=True, remove_terminal=True)

    logger.info("Timing new policy functions on random states...")
    logger.info("Starting value head 2 timing...")
    start_time = time.time()
    for state in states:
        legal_moves = game.generate_moves_for_given_board(state)
        vh_policy, _ = get_value_head_policy2(model, state, legal_moves, for_model=True)
    vh2_time = time.time() - start_time

    logger.info("Starting policy head timing...")
    start_time = time.time()
    for state in states:
        legal_moves = game.generate_moves_for_given_board(state)
        ph_policy, _ = get_policy_head_policy(model, state, legal_moves, for_model=True)
    ph_time = time.time() - start_time

    logger.info("Starting value head timing...")
    start_time = time.time()
    for state in states:
        legal_moves = game.generate_moves_for_given_board(state)
        vh_policy, _ = get_value_head_policy(model, state, legal_moves, for_model=True)
    vh_time = time.time() - start_time

    logger.info(f"Policy Head Time for 1000 states: {ph_time:.4f} seconds")
    logger.info(f"Value Head Time for 1000 states: {vh_time:.4f} seconds")
    logger.info(f"Value Head 2 Time for 1000 states: {vh2_time:.4f} seconds")

def verify_policy_equivalence():
    model = load_model(config.training_dir + "model_450.pkl")
    game = Game(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        no_reverse_moves=False,
        no_illegal_moves=False,
        no_side_moves=False
    )
    states = get_n_random_states(100, remove_trivial=True, remove_terminal=True)

    for state in states:
        legal_moves = game.generate_moves_for_given_board(state)
        vh_policy1, _ = get_value_head_policy(model, state, legal_moves, for_model=True)
        vh_policy2, _ = get_value_head_policy2(model, state, legal_moves, for_model=True)

        if not np.allclose(vh_policy1, vh_policy2):
            logger.error("Value head policies do not match!")
            print("State:")
            print(state.board_view())
            print("Legal Moves:")
            for move in legal_moves:
                print(move)
            print("Value Head Policy 1:")
            #print(vh_policy1)
            print_policy(list(vh_policy1), state, legal_moves, policy_is_for_model=True)
            print("Value Head Policy 2:")
            #print(vh_policy2)
            print_policy(list(vh_policy2), state, legal_moves, policy_is_for_model=True)
            return
    logger.info("All value head policies match between the two implementations.")

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="new_policy_functions"
    )

    #inspect_new_policies()
    #test_accuracies(100000)
    #time_new_policy_functions()
    verify_policy_equivalence()
