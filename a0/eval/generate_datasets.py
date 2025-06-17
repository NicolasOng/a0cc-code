import pickle
import random

import jax.numpy as jnp
from tqdm import tqdm

from cc.core import Board
from cc.lookups import CCBaselineSolver
from a0.dataset import Dataset, TrainingData
from a0.game import GameData
from a0.train.alphazero import board_to_input

from config import config

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def training_ground_truth_values(n: int | None = None) -> None:
    # 1. read all of config.training_dir + f"gamedata_{i + 1}.pkl", where i is from 0 to config.training_iterations
    # these are all list[GameData] objects
    game_data_list: list[GameData] = []
    for i in tqdm(range(config.training_iterations)):
        file_path = f"{config.training_dir}/gamedata_{i + 1}.pkl"
        with open(file_path, 'rb') as file:
            data: list[GameData] = pickle.load(file)
            game_data_list.extend(data)
    logger.info(f"Loaded {len(game_data_list)} game data objects from {config.training_dir}.")

    # 2. put all of the boards into a single set (to avoid duplicates)
    total_boards = 0
    boards_set: set[Board] = set()
    for game_data in tqdm(game_data_list):
        turn_data = game_data.turn_data
        for turn in turn_data:
            boards_set.add(turn.board)
            total_boards += 1
    boards: list[Board] = list(boards_set)
    
    logger.info(f"Found {len(boards)} unique boards in the game data, out of {total_boards} total boards.")
    if n is not None: boards = random.sample(boards, n)

    # 3. for each board, get the value from the solve data file, and convert it to a model input
    # then add these to a jnp array
    # values = jnp.zeros((len(boards), 1))
    # policies = jnp.zeros((len(boards), config.board_size ** 4))
    states: list[jnp.ndarray] = []
    values: list[float] = []
    solver = CCBaselineSolver(config.solve_data, config.num_spots, config.num_players, config.num_pieces)
    for board in tqdm(boards):
        values.append(solver.get_outcome(board))
        states.append(board_to_input(board))
    
    # 4. load all this into a static Dataset object
    jnp_states = jnp.stack(states) # (N, board_size, board_size)
    jnp_values = jnp.array(values) [:, None]  # Add [:, None] to make its shape (N, 1)
    jnp_policies = jnp.zeros((len(boards), config.board_size ** 4)) # (N, board_size ** 4)
    gtv_dataset = Dataset(size=len(boards), batch_size=256, static=True)
    gtv_dataset.set(jnp_states, jnp_values, jnp_policies)

    # 5. save the Dataset object to config.data_folder + "training_gtv.pkl"
    output_path = f"{config.data_folder}/training_gtv.pkl"
    with open(output_path, 'wb') as file:
        pickle.dump(gtv_dataset, file)
    logger.info(f"Ground truth values saved to {output_path}.")

def main():
    setup_logging(
        level=20,
        log_dir="logs/",
        process_name="generate_datasets"
    )
    logger.info("Starting ground truth values generation...")

    # Generate ground truth values for all boards
    training_ground_truth_values()

    logger.info("Ground truth values generation completed.")

if __name__ == "__main__":
    main()
