from a0.model import AlphaZeroModel, load_model
from cc.core import Board, Move, Player, Game
from a0.model_utils import board_to_input, Policy
from a0.eval.dataset_evaluation import policy_accuracy_function
from cc.ground_truth import GroundTruth
from scripts.policy_inspection import print_policy

from scripts.sl_on_policy_head import get_n_random_states

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def get_value_head_policy(model: AlphaZeroModel, state: Board, moves: list[Move], for_model: bool = False) -> tuple[NDArray[np.float32], Policy]:
    rotate = state.current_player == Player.PLAYER_O if for_model else False

    values: list[float] = []
    for move in moves:
        state.apply_move(move)
        value, _ = model(jnp.array(board_to_input(state)))
        values.append(float(value[0][0]))
        state.undo_move(move)

    p = Policy(len(state.board))
    p.set_logits_from_moves(moves, values, rotate_180=False)
    p.set_legal_moves(moves)
    p.apply_softmax(temperature=1.0, mask=True)
    if rotate:
        p.rotate_policy()
    
    return p.policy, p

def get_policy_head_policy(model: AlphaZeroModel, state: Board, moves: list[Move], for_model: bool = False) -> tuple[NDArray[np.float32], Policy]:
    '''
    Returns the policy distribution over the given moves for the given state using the model.
    If for_model is True, the policy is rotated according to the model's perspective.
    '''
    rotate = state.current_player == Player.PLAYER_O if for_model else False

    _, policy = model(jnp.array(board_to_input(state)))

    p = Policy(len(state.board))
    p.set_logits(np.array(policy[0]), rotate_180=False)
    p.set_legal_moves(moves)
    p.apply_softmax(temperature=1.0, mask=True)
    if rotate:
        p.rotate_policy()
    
    return p.policy, p

def combined_head_policy(model: AlphaZeroModel, state: Board, moves: list[Move], blend: float = 0.5, for_model: bool = False) -> NDArray[np.float32]:
    php, _ = get_policy_head_policy(model, state, moves, for_model=for_model)
    vhp, _ = get_value_head_policy(model, state, moves, for_model=for_model)
    combined_policy = blend * php + (1 - blend) * vhp
    return np.array(combined_policy / np.sum(combined_policy), dtype=np.float32)

def test_accuracies():
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
    states = get_n_random_states(100, remove_trivial=True, remove_terminal=True)

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
    
    logger.info(f"Policy Head Accuracy: {ph_acc_count}/{total_boards} = {ph_acc_count / total_boards:.2%}")
    logger.info(f"Value Head Accuracy: {vh_acc_count}/{total_boards} = {vh_acc_count / total_boards:.2%}")
    #logger.info(f"Combined Head Accuracy: {ch_acc_count}/{total_boards} = {ch_acc_count / total_boards:.2%}")

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="new_policy_functions"
    )

    test_accuracies()
