import random

from tqdm import tqdm

from config import config
from cc.ranking import CCDefaultRank, CCState
from cc.core import Game, Player
from cc.lookups import CCBaselineSolver
from a0.dataset import Dataset

from a0.players.a0 import board_to_input

import jax.numpy as jnp
import numpy as np

import pickle
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def generate_ground_truth_dataset(num_states: int | None = None):
    '''
    Generates a dataset of ground truth data for the game.
    It either does the number of states specified, or all states if None is specified.
    '''
    # create useful objects
    r = CCDefaultRank(config.num_spots, config.num_players, config.num_pieces)
    s = CCState(config.num_spots, config.num_pieces, config.num_players)
    l = CCBaselineSolver(config.solve_data, config.num_spots, config.num_players, config.num_pieces)

    # decide how many states to generate (if None specified, generate all states)
    max_rank = r.get_max_rank()
    n = max_rank if num_states is None else num_states

    # create the jnp arrays to hold the data
    jnp_states = jnp.zeros((n, config.board_size, config.board_size, 2)) # (N, board_size, board_size, 2)
    jnp_values = jnp.zeros((n, 1)) # (N, 1)
    jnp_policies = jnp.zeros((n, config.board_size ** 4)) # (N, board_size ** 4)

    # for the amount of ranks specified,
    for i in tqdm(range(n)):
        # get the rank to generate (if generating all states, use the index as the rank)
        cur_rank = i if num_states is None else random.randint(0, max_rank)
        
        # unrank the current rank to get the state
        r.unrank(cur_rank, s)
        # get the board from the state
        board = s.get_board()

        # convert the board to a model input (board_size, board_size, 2)
        board_input = board_to_input(board)[0]
        # get the outcome of the state from the solve data
        # this is from the perspective of the current player (1=win, -1=loss, 0=draw/illegal)
        # (1,)
        outcome = np.array([l.get_outcome(board)])
        # get the ideal policy for the state, based on the solve data
        # TODO: implement this - for now, blank.
        
        # put these into the jnp arrays
        jnp_states.at[i].set(board_input)
        jnp_values.at[i].set(outcome)
        #jnp_policies[i] = jnp.zeros((config.board_size ** 4,))
    
    # add these to a dataset
    dataset = Dataset(n, 256, True)
    dataset.set(jnp_states, jnp_values, jnp_policies)

    # save the dataset
    logger.info("Saving ground truth dataset to file...")
    output_path = f"{config.eval_dir}/gtd.pkl"
    with open(output_path, 'wb') as file:
        pickle.dump(dataset, file)
    logger.info(f"Ground truth dataset saved to {output_path}.")

def main():
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="gen_train_datasets"
    )
    logger.info("Generating datasets...")

    generate_ground_truth_dataset()

    logger.info("Finished generating datasets.")

if __name__ == "__main__":
    main()
