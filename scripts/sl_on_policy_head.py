import random

from cc.core import Board
from cc.ground_truth import GroundTruth
from a0.dataset import Dataset
from a0.model_utils import board_to_input
from a0.eval.training_data import game_data_generator
from a0.train.dataset import train_model_epochs, plot_model_performance
from a0.model import AlphaZeroModel
from a0.eval.dataset_evaluation import evaluate_model, policy_accuracy_function

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from flax import nnx
import jax

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def get_n_random_states(n: int, remove_trivial: bool = True) -> list[Board]:
    '''
    Returns a list of n random unique board states.
    '''
    gt = GroundTruth()
    max_rank = gt.get_max_rank()
    state_set: set[Board] = set()
    while len(state_set) < n:
        rank = random.randint(0, max_rank - 1)
        board = gt.unrank(rank)
        if remove_trivial and gt.is_trivial(board):
            continue
        state_set.add(board)
    return list(state_set)

def create_gtd_from_states(boards: list[Board]) -> Dataset:
    '''
    Creates a ground truth dataset from a list of board states.
    This uses the 1ply prob dist over winning moves as the policy target.
    '''
    gt = GroundTruth()
    states: list[jnp.ndarray] = []
    values: list[float] = []
    policies: list[jnp.ndarray] = []
    gt = GroundTruth()
    for board in tqdm(boards):
        states.append(jnp.array(board_to_input(board))) # (1, board_size, board_size, 2)
        values.append(gt.get_outcome(board)) # float
        policies.append(jnp.array(gt.get_1ply_policy_prob_dist_list(board, for_model=True))) # (board_size ** 4,)
    # load all this into a Dataset object
    logger.info("Creating Dataset object with ground truth values...")
    jnp_states = jnp.concatenate(states, axis=0) # (N, board_size, board_size, 2)
    jnp_values = jnp.array(values).reshape(-1, 1)  # (N, 1)
    jnp_policies = jnp.stack(policies) # (N, board_size ** 4)
    logger.info(f"states.shape: {jnp_states.shape}, values.shape: {jnp_values.shape}, policies.shape: {jnp_policies.shape}")
    gtv_dataset = Dataset(batch_size=256)
    gtv_dataset.set(jnp_states, jnp_values, jnp_policies)
    return gtv_dataset

def create_random_gtd_from_states(boards: list[Board]) -> Dataset:
    '''
    Creates a ground truth dataset from n random unique board states.
    This chooses a random winning move as the policy target.
    '''
    gt = GroundTruth()
    states: list[jnp.ndarray] = []
    values: list[float] = []
    policies: list[jnp.ndarray] = []
    gt = GroundTruth()
    for board in tqdm(boards):
        states.append(jnp.array(board_to_input(board))) # (1, board_size, board_size, 2)
        values.append(gt.get_outcome(board)) # float
        policies.append(jnp.array(gt.get_random_best_move_prob_dist_list(board, for_model=True))) # (board_size ** 4,)
    # load all this into a Dataset object
    logger.info("Creating Dataset object with ground truth values...")
    jnp_states = jnp.concatenate(states, axis=0) # (N, board_size, board_size, 2)
    jnp_values = jnp.array(values).reshape(-1, 1)  # (N, 1)
    jnp_policies = jnp.stack(policies) # (N, board_size ** 4)
    logger.info(f"states.shape: {jnp_states.shape}, values.shape: {jnp_values.shape}, policies.shape: {jnp_policies.shape}")
    gtv_dataset = Dataset(batch_size=256)
    gtv_dataset.set(jnp_states, jnp_values, jnp_policies)
    return gtv_dataset

def create_random_dataset_from_states(boards: list[Board]) -> Dataset:
    '''
    Creates a dataset from n random unique board states.
    This chooses a random valid move as the policy target.
    '''
    gt = GroundTruth()
    states: list[jnp.ndarray] = []
    values: list[float] = []
    policies: list[jnp.ndarray] = []
    gt = GroundTruth()
    for board in tqdm(boards):
        states.append(jnp.array(board_to_input(board))) # (1, board_size, board_size, 2)
        values.append(random.choice([0, 1])) # float
        policies.append(jnp.array(gt.get_random_valid_move_prob_dist_list(board, for_model=True))) # (board_size ** 4,)
    # load all this into a Dataset object
    logger.info("Creating Dataset object with ground truth values...")
    jnp_states = jnp.concatenate(states, axis=0) # (N, board_size, board_size, 2)
    jnp_values = jnp.array(values).reshape(-1, 1)  # (N, 1)
    jnp_policies = jnp.stack(policies) # (N, board_size ** 4)
    logger.info(f"states.shape: {jnp_states.shape}, values.shape: {jnp_values.shape}, policies.shape: {jnp_policies.shape}")
    gtv_dataset = Dataset(batch_size=256)
    gtv_dataset.set(jnp_states, jnp_values, jnp_policies)
    return gtv_dataset

def create_dataset_from_selfplay(remove_trivial: bool = True) -> tuple[Dataset, list[Board]]:
    # get the unique boards (with their outcome/policy) from the training data
    gt = GroundTruth()
    total_boards = 0
    acc_count = 0
    states: list[jnp.ndarray] = []
    values: list[float] = []
    policies: list[jnp.ndarray] = []
    boards: set[Board] = set()
    logger.info("Getting unique boards from training data, with their experienced outcome...")
    game_data_lists = game_data_generator(config.training_dir, config.training_iterations)
    for _, game_data_list in tqdm(game_data_lists):
        for game_data in game_data_list:
            game_winner = game_data.winner
            turn_data = game_data.turn_data
            for turn in turn_data:
                # get the board and its experienced outcome (current player perspective)
                board = turn.board
                experienced_board = jnp.array(board_to_input(board)) # (1, board_size, board_size, 2)
                experienced_outcome = 0.0 if game_winner is None else 1.0 if game_winner == board.current_player else -1.0
                experienced_policy = turn.player_data

                if remove_trivial and gt.is_trivial(board):
                    continue

                gt_policy = gt.get_1ply_policy_prob_dist_list(board, for_model=True)
                acc = policy_accuracy_function(np.array(experienced_policy), np.array(gt_policy))
                total_boards += 1
                if acc:
                    acc_count += 1

                states.append(experienced_board)
                values.append(experienced_outcome)
                policies.append(experienced_policy)
                boards.add(board)
            
    # load all this into a Dataset object
    jnp_states = jnp.concatenate(states, axis=0) # (N, board_size, board_size, 2)
    jnp_values = jnp.array(values).reshape(-1, 1)  # (N, 1)
    jnp_policies = jnp.stack(policies) # (N, board_size ** 4)
    logger.info(f"states.shape: {jnp_states.shape}, values.shape: {jnp_values.shape}, policies.shape: {jnp_policies.shape}")
    sp_dataset = Dataset(batch_size=256)
    sp_dataset.set(jnp_states, jnp_values, jnp_policies)

    logger.info(f"Policy accuracy against ground truth on self-play data (NT Boards): {acc_count / total_boards if total_boards > 0 else 0.0:.2%} ({acc_count} / {total_boards})")

    return sp_dataset, list(boards)

def check_random_policy_acc(boards: list[Board]) -> float:
    '''
    Checks the policy accuracy of a random policy on the given boards.
    '''
    gt = GroundTruth()
    correct = 0
    total = 0
    for board in tqdm(boards):
        true_policy = gt.get_1ply_policy_prob_dist_list(board, for_model=False)
        random_policy = gt.get_random_valid_move_prob_dist_list(board, for_model=False)
        acc: bool = policy_accuracy_function(np.array(random_policy), np.array(true_policy))
        if acc:
            correct += 1
        total += 1
    logger.info(f"Random policy accuracy: {correct / total if total > 0 else 0.0:.2%} ({correct} / {total})")
    return correct / total if total > 0 else 0.0

def train_and_plot(fn: str, dataset: Dataset, eval_dataset: Dataset, num_epochs: int = 10) -> AlphaZeroModel:
    logger.info(f"Training model '{fn}' for {num_epochs} epochs...")

    model = AlphaZeroModel(
        board_size=config.board_size,
        num_filters=256,
        training=True,
        rngs=nnx.Rngs({'params': jax.random.PRNGKey(1)})
    )

    test_dataset = dataset.split_off_test(len(dataset) // 10, shuffle=True)

    trained_model, dsd = train_model_epochs(model, dataset, num_epochs, save="None", plot=False, test_dataset=test_dataset)
    plot_model_performance(fn, [dsd])

    loss, value_loss, policy_loss, value_accuracy, policy_accuracy = evaluate_model(trained_model, eval_dataset)
    logger.info(f"For model '{fn}':")
    logger.info(f"Evaluation on eval dataset - Loss: {loss}, Value Loss: {value_loss}, Policy Loss: {policy_loss}, Value Accuracy: {value_accuracy}, Policy Accuracy: {policy_accuracy}")

    return trained_model

def main():
    # TODO: do with larger n, also can delete the value-less ones if no effect.
    # create a dataset from self-play data
    sp_dataset, sp_boards = create_dataset_from_selfplay(remove_trivial=True)
    logger.info("Self-play dataset created.")

    # create a ground truth dataset from the self-play boards
    spgtv_dataset = create_gtd_from_states(sp_boards)
    logger.info("selfplay Ground truth dataset created.")

    sp_n = len(sp_dataset)
    logger.info(f"Self-play dataset size: {sp_n}")
    # get n random unique board states
    n = sp_n
    logger.info(f"Generating {n} random unique board states...")
    boards = get_n_random_states(n, remove_trivial=True)

    # create a ground truth dataset from these states
    gtv_dataset = create_gtd_from_states(boards)
    logger.info("Ground truth dataset created.")

    # create a random ground truth dataset from these states
    random_gtv_dataset = create_random_gtd_from_states(boards)
    logger.info("Random ground truth dataset created.")

    # create a random dataset from these states
    random_dataset = create_random_dataset_from_states(boards)
    logger.info("Random dataset created.")

    # check the random policy accuracy on the gtv boards
    check_random_policy_acc(boards)

    # train a model on the gtv dataset + evaluate
    train_and_plot("sl_on_policy_head_gtv", gtv_dataset, gtv_dataset, num_epochs=10)

    # train a model on the random gtv dataset + evaluate
    train_and_plot("sl_on_policy_head_random_gtv", random_gtv_dataset, gtv_dataset, num_epochs=10)

    # train a model on the random dataset + evaluate
    train_and_plot("sl_on_policy_head_random", random_dataset, gtv_dataset, num_epochs=10)

    # train a model on the self-play dataset + evaluate
    m = train_and_plot("sl_on_policy_head_selfplay", sp_dataset, gtv_dataset, num_epochs=10)
    # also eval on itself
    loss, value_loss, policy_loss, value_accuracy, policy_accuracy = evaluate_model(m, sp_dataset)
    logger.info(f"Evaluation on sp dataset - Loss: {loss}, Value Loss: {value_loss}, Policy Loss: {policy_loss}, Value Accuracy: {value_accuracy}, Policy Accuracy: {policy_accuracy}")
    loss, value_loss, policy_loss, value_accuracy, policy_accuracy = evaluate_model(m, spgtv_dataset)
    logger.info(f"Evaluation on spgtv dataset - Loss: {loss}, Value Loss: {value_loss}, Policy Loss: {policy_loss}, Value Accuracy: {value_accuracy}, Policy Accuracy: {policy_accuracy}")

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="sl_on_policy_head"
    )

    main()
