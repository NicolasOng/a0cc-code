import random

from tqdm import tqdm

from config import config
from cc.core import Player
from a0.dataset import Dataset
from cc.ground_truth import GroundTruth

from a0.players.a0 import board_to_input

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

import pickle
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def generate_ground_truth_dataset(num_states: int | None = None, prob_dist: bool = True):
    '''
    Generates a dataset of ground truth data for the game.
    It either does the number of states specified, or all states if None is specified.
    '''
    # create useful objects
    gt = GroundTruth()

    # decide how many states to generate (if None specified, generate all states)
    max_rank = gt.get_max_rank()
    n = max_rank if num_states is None else num_states

    # create lists to hold the data
    states: list[NDArray[np.float32]] = [] # (board_size, board_size, 2)
    values: list[NDArray[np.float32]] = [] # (1,)
    policies: list[NDArray[np.float32]] = [] # (board_size ** 4)

    # for the amount of ranks specified,
    for i in tqdm(range(n)):
        # get the rank to generate (if generating all states, use the index as the rank)
        cur_rank = i if num_states is None else random.randint(0, max_rank - 1)
        
        # unrank the current rank to get the state
        board = gt.unrank(cur_rank)

        # skip if the current player is O
        # this is because the outcomes/policies are symmetric
        # we skip if num_states is specified, as we want to generate exactly num_states
        if num_states is None and board.current_player == Player.PLAYER_O:
            continue

        # convert the board to a model input (board_size, board_size, 2)
        board_input = board_to_input(board)[0]
        # get the outcome of the state from the solve data
        # this is from the perspective of the current player (1=win, -1=loss, 0=draw/illegal)
        # (1,)
        outcome = np.array([gt.get_outcome(board)])
        # get the ideal policy for the state, based on the solve data
        # (board_size ** 4,)
        if prob_dist:
            policy = np.array(gt.get_1ply_policy_prob_dist_list(board, for_model=True))
        else:
            policy = np.array(gt.get_1ply_policy_outcomes_list(board, for_model=True))

        # put these into the lists
        states.append(board_input)
        values.append(outcome)
        policies.append(policy)

    jnp_states = jnp.stack(states, 0) # (board_size, board_size, 2) -> (N, board_size, board_size, 2)
    jnp_values = jnp.stack(values, 0)  # (1,) -> (N, 1)
    jnp_policies = jnp.stack(policies, 0) # (board_size ** 4) -> (N, board_size ** 4)
    
    # add these to a dataset
    dataset = Dataset(256)
    dataset.set(jnp_states, jnp_values, jnp_policies)

    # save the dataset
    logger.info("Saving ground truth dataset to file...")
    output_path = f"{config.dataset_dir}/gtd.pkl"
    with open(output_path, 'wb') as file:
        pickle.dump(dataset, file)
    logger.info(f"Ground truth dataset saved to {output_path}.")

def load_ground_truth_dataset() -> Dataset:
    dataset_path = config.dataset_dir + '/gtd.pkl'
    with open(dataset_path, 'rb') as file:
        gt_dataset: Dataset = pickle.load(file)
    logger.info(f"Loaded dataset from {dataset_path}.")
    return gt_dataset

def generate_random_dataset(num_states: int | None = None):
    # create useful objects
    gt = GroundTruth()

    # decide how many states to generate (if None specified, generate all states)
    max_rank = gt.get_max_rank()
    n = max_rank if num_states is None else num_states

    # create lists to hold the data
    states = [] # (board_size, board_size, 2)
    values = [] # (1,)
    policies = [] # (board_size ** 4)

    # for the amount of ranks specified,
    for i in tqdm(range(n)):
        # get the rank to generate (if generating all states, use the index as the rank)
        cur_rank = i if num_states is None else random.randint(0, max_rank - 1)
        
        # unrank the current rank to get the state
        board = gt.unrank(cur_rank)

        # convert the board to a model input (board_size, board_size, 2)
        board_input = board_to_input(board)[0]
        # choose a random outcome for the state (1=win, -1=loss, 0=draw/illegal)
        outcome = np.random.choice([-1, 1], p=[0.5, 0.5])
        outcome = np.array([outcome])  # Convert to shape (1,)
        # get a random policy for the state, based on the solve data
        # TODO: implement this - for now, blank.
        
        # put these into the lists
        states.append(board_input)
        values.append(outcome)
        policies.append(jnp.zeros((config.board_size ** 4,)))
    
    jnp_states = jnp.stack(states, 0) # (board_size, board_size) -> (N, board_size, board_size)
    jnp_values = jnp.stack(values, 0)  # (1,) -> (N, 1)
    jnp_policies = jnp.stack(policies, 0) # (board_size ** 4) -> (N, board_size ** 4)

    # add these to a dataset
    dataset = Dataset(256)
    dataset.set(jnp_states, jnp_values, jnp_policies)

    # save the dataset
    logger.info("Saving random dataset to file...")
    output_path = f"{config.dataset_dir}/rd.pkl"
    with open(output_path, 'wb') as file:
        pickle.dump(dataset, file)
    logger.info(f"Random dataset saved to {output_path}.")

def main():
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="gen_train_datasets"
    )
    logger.info("Generating datasets...")

    generate_ground_truth_dataset(num_states=865000, prob_dist=True)

    #generate_random_dataset(865000)

    logger.info("Finished generating datasets.")

if __name__ == "__main__":
    main()
