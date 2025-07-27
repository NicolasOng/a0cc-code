import pickle
import random
import copy

import jax.numpy as jnp
from tqdm import tqdm

from cc.core import Board
from cc.lookups import CCBaselineSolver
from cc.ranking import CCDefaultRank, CCState
from a0.dataset import Dataset, TrainingData
from a0.game import GameData
from a0.train.alphazero import board_to_input
from a0.eval.training_data import game_data_generator, training_data_generator

from config import config

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def training_ground_truth_values(n: int | None = None) -> None:
    '''
    Generates and saves a dataset of ground truth values for all unique boards
    from the training data generated during training/self-play.
    The dataset contains the board and the target value, not the policy. (TODO)
    '''
    logger.info("Generating ground truth values from training data...")
    # 1. read all of config.training_dir + f"gamedata_{i + 1}.pkl", where i is from 0 to config.training_iterations
    # these are all list[GameData] objects
    logger.info("Loading game data from training directory...")
    game_data_list: list[GameData] = []
    for i in tqdm(range(config.training_iterations)):
        file_path = f"{config.training_dir}/gamedata_{i + 1}.pkl"
        with open(file_path, 'rb') as file:
            data: list[GameData] = pickle.load(file)
            game_data_list.extend(data)
    logger.info(f"Loaded {len(game_data_list)} game data objects from {config.training_dir}.")

    # 2. put all of the boards into a single set (to avoid duplicates)
    logger.info("Extracting unique boards from game data...")
    total_boards = 0
    boards_set: set[Board] = set()
    for game_data in tqdm(game_data_list):
        turn_data = game_data.turn_data
        for turn in turn_data:
            boards_set.add(turn.board)
            total_boards += 1
    boards: list[Board] = list(boards_set)
    
    num_unique = len(boards_set)
    logger.info(f"Found {num_unique} unique boards in the game data, out of {total_boards} total boards ({(num_unique/total_boards)*100:.2f}%).")
    if n is not None: boards = random.sample(boards, n)

    # 3. for each board, get the value from the solve data file, and convert it to a model input
    # then add these to a jnp array
    # values = jnp.zeros((len(boards), 1))
    # policies = jnp.zeros((len(boards), config.board_size ** 4))
    logger.info("Get ground truth values for each board (current player perspective)...")
    states: list[jnp.ndarray] = []
    values: list[float] = []
    solver = CCBaselineSolver(config.solve_data, config.num_spots, config.num_players, config.num_pieces)
    for board in tqdm(boards):
        values.append(solver.get_outcome(board))
        states.append(jnp.array(board_to_input(board)))
    
    # 4. load all this into a static Dataset object
    logger.info("Creating Dataset object with ground truth values...")
    jnp_states = jnp.stack(states) # (N, board_size, board_size)
    jnp_values = jnp.array(values) [:, None]  # Add [:, None] to make its shape (N, 1)
    jnp_policies = jnp.zeros((len(boards), config.board_size ** 4)) # (N, board_size ** 4)
    gtv_dataset = Dataset(max_size=len(boards), batch_size=256, static=True)
    gtv_dataset.set(jnp_states, jnp_values, jnp_policies)

    # 5. save the Dataset object to config.data_folder + "training_gtv.pkl"
    output_path = f"{config.eval_dir}/training_gtv.pkl"
    with open(output_path, 'wb') as file:
        pickle.dump(gtv_dataset, file)
    logger.info(f"Ground truth values saved to {output_path}.")

def random_ground_truth_values(n: int = 1000) -> None:
    '''
    Generates and saves a dataset of ground truth values for all unique boards
    from a list of randomly generated boards.
    The dataset contains the board and the target value, not the policy. (TODO)
    '''
    logger.info(f"Generating ground truth values from {n} random boards...")

    # 1. generate a list of random boards
    logger.info("Generating random boards...")
    r = CCDefaultRank(config.num_spots, config.num_players, config.num_pieces)
    s = CCState(config.num_spots, config.num_pieces, config.num_players)
    max_rank = r.get_max_rank()
    board_set: set[Board] = set()
    while len(board_set) < n:
        r.unrank(random.randint(0, max_rank), s)
        b = s.get_board()
        board_set.add(b)
    boards: list[Board] = list(board_set)
        
    logger.info(f"Generated {len(boards)} boards.")

    # 3. for each board, get the value from the solve data file, and convert it to a model input
    # then add these to a jnp array
    # values = jnp.zeros((len(boards), 1))
    # policies = jnp.zeros((len(boards), config.board_size ** 4))
    logger.info("Get ground truth values for each board (current player perspective)...")
    states: list[jnp.ndarray] = []
    values: list[float] = []
    solver = CCBaselineSolver(config.solve_data, config.num_spots, config.num_players, config.num_pieces)
    for board in tqdm(boards):
        values.append(solver.get_outcome(board))
        states.append(jnp.array(board_to_input(board)))
    
    # 4. load all this into a static Dataset object
    logger.info("Creating Dataset object with ground truth values...")
    jnp_states = jnp.stack(states) # (N, board_size, board_size, 2)
    jnp_values = jnp.array(values) [:, None]  # Add [:, None] to make its shape (N, 1)
    jnp_policies = jnp.zeros((len(boards), config.board_size ** 4)) # (N, board_size ** 4)
    gtv_dataset = Dataset(max_size=len(boards), batch_size=256, static=True)
    gtv_dataset.set(jnp_states, jnp_values, jnp_policies)

    # 5. save the Dataset object to config.data_folder + "random_gtv.pkl"
    logger.info("Saving ground truth values to file...")
    output_path = f"{config.eval_dir}/random_gtv.pkl"
    with open(output_path, 'wb') as file:
        pickle.dump(gtv_dataset, file)
    logger.info(f"Ground truth values saved to {output_path}.")

def training_datasets() -> None:
    '''
    Generates a list of datasets based on the
    training data generated during training/self-play.
    Each dataset contains the training data generated in a single training iteration.
    '''
    logger.info("Generating training datasets from training data...")
    # load all the training data objects from config.training_dir
    training_data_lists: list[list[TrainingData]] = []
    for i in tqdm(range(config.training_iterations)):
        file_path = f"{config.training_dir}/training_set_{i + 1}.pkl"
        with open(file_path, 'rb') as file:
            data: list[TrainingData] = pickle.load(file)
            training_data_lists.append(data)
    logger.info(f"Loaded {len(training_data_lists)} training data lists from {config.training_dir}.")

    datasets: list[Dataset] = []
    # create a Dataset object for each training data list
    # and add the training data to it
    for i, training_data_list in tqdm(enumerate(training_data_lists)):
        replay_buffer = Dataset(
            max_size=config.replay_buffer_size,
            batch_size=config.training_batch_size,
            static=False
        )
        for training_data in training_data_list:
            replay_buffer.add(training_data)
        datasets.append(replay_buffer)
    
    # save the datasets
    output_path = f"{config.eval_dir}/training_datasets.pkl"
    with open(output_path, 'wb') as file:
        pickle.dump(datasets, file)
    logger.info(f"List of training datasets saved to {output_path}.")

def get_unique_boards_from_training_data() -> set[Board]:
    '''
    Extracts all unique boards from the training data generated during training/self-play.
    Returns a set of unique Board objects.
    '''
    logger.info("Getting unique boards from training data...")
    # 1. get the game data lists from the training data generator
    # these are all list[GameData] objects, where each GameData list
    # contains games generated in a single training iteration
    logger.info("Generating game data lists from training data...")
    game_data_lists = game_data_generator()
    
    # 2. put all of the boards into a single set (to avoid duplicates)
    logger.info("Extracting unique boards from game data...")
    total_boards = 0
    boards_set: set[Board] = set()
    for game_data_list in tqdm(game_data_lists):
        for game_data in tqdm(game_data_list):
            turn_data = game_data.turn_data
            for turn in turn_data:
                boards_set.add(turn.board)
                total_boards += 1
    
    num_unique = len(boards_set)
    logger.info(f"Found {num_unique} unique boards in the game data, out of {total_boards} total boards ({(num_unique/total_boards)*100:.2f}%).")
    return boards_set

def training_experienced_values(n: int | None = None) -> None:
    '''
    Generates and saves a dataset of experienced values for all unique boards
    from the training data generated during training/self-play.
    The dataset contains the board, the experienced value, and the experienced policy.
    The experienced value is the outcome of the game from the perspective of the current player.
    '''
    logger.info("Generating experienced values from training data...")

    # 1. get the unique boards (with their outcome/policy) from the training data
    logger.info("Getting unique boards from training data, with their experienced outcome...")
    game_data_lists = game_data_generator()
    total_boards = 0
    boards_set: set[TrainingData] = set()
    for game_data_list in tqdm(game_data_lists):
        for game_data in tqdm(game_data_list):
            game_winner = game_data.winner
            turn_data = game_data.turn_data
            for turn in turn_data:
                # get the board and its experienced outcome (current player perspective)
                board = turn.board
                experienced_outcome = 0.0 if game_winner is None else 1.0 if game_winner == board.current_player else -1.0
                experienced_policy = turn.player_data
                experienced_target = TrainingData(jnp.array(board_to_input(board)), experienced_outcome, experienced_policy)

                # add this to a set
                boards_set.add(experienced_target)
                total_boards += 1
    
    num_unique = len(boards_set)
    logger.info(f"Found {num_unique} unique boards+value+policy triplets in the game data, out of {total_boards} total boards ({(num_unique/total_boards)*100:.2f}%).")
    
    boards = list(boards_set)
    if n is not None: boards = random.sample(boards, n)
    
    # 4. load all this into a Dataset object
    ev_dataset = Dataset(max_size=len(boards), batch_size=256, static=False)
    for data in tqdm(boards):
        ev_dataset.add(data)
    ev_dataset.convert_to_static()

    # 5. save the Dataset object to config.data_folder + "training_ev.pkl"
    output_path = f"{config.eval_dir}/training_ev.pkl"
    with open(output_path, 'wb') as file:
        pickle.dump(ev_dataset, file)
    logger.info(f"Experienced values saved to {output_path}.")

def main():
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="generate_datasets"
    )
    logger.info("Generating datasets...")

    training_ground_truth_values()

    random_ground_truth_values()

    training_datasets()

    training_experienced_values()

    logger.info("Finished generating datasets.")

if __name__ == "__main__":
    main()
