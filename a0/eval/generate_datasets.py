import pickle
import random
import copy

import jax.numpy as jnp
from tqdm import tqdm

from cc.core import Board, Game
from cc.ground_truth import GroundTruth
from a0.dataset import Dataset
from a0.experience_buffer import ExperienceData
from a0.train.alphazero import board_to_input
from a0.eval.training_data import game_data_generator

from config import config

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def random_ground_truth_values(n: int = 1000) -> None:
    '''
    Generates and saves a dataset of ground truth values for all unique boards
    from a list of randomly generated boards.
    The dataset contains the board and the target value, not the policy. (TODO)
    '''
    logger.info(f"Generating ground truth values from {n} random boards...")

    # 1. generate a list of random boards
    logger.info("Generating random boards...")
    gt = GroundTruth()
    max_rank = gt.get_max_rank()
    board_set: set[Board] = set()
    total = 0
    total_non_trivial = 0
    while len(board_set) < n:
        b = gt.unrank(random.randint(0, max_rank - 1))
        if not gt.is_trivial(b):
            board_set.add(b)
            total_non_trivial += 1
        total += 1
    boards: list[Board] = list(board_set)
        
    logger.info(f"Generated {len(boards)} boards.")
    logger.info(f"Total boards generated: {total}, Non-trivial boards: {total_non_trivial} ({(total_non_trivial/total)*100:.2f}%).")

    # create and save a gtv dataset based on the generated boards
    rgtv_dataset = create_gtv_dataset_from_board_list(boards)
    save_dataset("random_gtv", rgtv_dataset)

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
    game_data_lists = game_data_generator(config.training_dir, config.training_iterations)

    # 2. put all of the boards into a single set (to avoid duplicates)
    logger.info("Extracting unique boards from game data...")
    total_boards = 0
    boards_set: set[Board] = set()
    for _, game_data_list in tqdm(game_data_lists):
        for game_data in game_data_list:
            turn_data = game_data.turn_data
            for turn in turn_data:
                boards_set.add(turn.board)
                total_boards += 1
    
    num_unique = len(boards_set)
    logger.log(25, f"Found {num_unique} unique boards in the game data, out of {total_boards} total boards ({num_unique/total_boards:.2%}).")

    gt = GroundTruth()
    boards_list = [board for board in boards_set if not gt.is_trivial(board)]
    boards_set = set(boards_list)
    logger.log(25, f"Filtered out trivial boards, {len(boards_set)} non-trivial unique boards remain ({len(boards_set)/num_unique:.2%}).")

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
    gt = GroundTruth()
    logger.info("Getting unique boards from training data, with their experienced outcome...")
    game_data_lists = game_data_generator(config.training_dir, config.training_iterations)
    total_boards = 0
    total_non_trivial = 0
    boards_set: set[ExperienceData] = set()
    for _, game_data_list in tqdm(game_data_lists):
        for game_data in game_data_list:
            game_winner = game_data.winner
            turn_data = game_data.turn_data
            for turn in turn_data:
                # get the board and its experienced outcome (current player perspective)
                board = turn.board
                experienced_outcome = 0.0 if game_winner is None else 1.0 if game_winner == board.current_player else -1.0
                experienced_policy = turn.player_data
                experience = ExperienceData(jnp.array(board_to_input(board)), experienced_outcome, experienced_policy)

                # add this to a set
                if not gt.is_trivial(board):
                    boards_set.add(experience)
                    total_non_trivial += 1
                total_boards += 1
    
    num_unique = len(boards_set)
    logger.info(f"Found {total_boards} total boards in the game data, with {total_non_trivial} non-trivial boards ({(total_non_trivial/total_boards)*100:.2f}%).")
    logger.info(f"Found {num_unique} unique and non-trivial boards+value+policy triplets in the game data, out of {total_non_trivial} total boards ({(num_unique/total_non_trivial)*100:.2f}%).")

    boards = list(boards_set)
    if n is not None: boards = random.sample(boards, n)
    
    # 4. load all this into a Dataset object
    ev_dataset = Dataset(batch_size=256)
    e_board = jnp.stack([d.board for d in boards]) # (board_size, board_size) -> (N, board_size, board_size)
    e_value = jnp.array([d.value for d in boards])[:, None] # Add [:, None] to make its shape (N, 1)
    e_policy = jnp.stack([d.policy for d in boards]) # (board_size ** 4) -> (N, board_size ** 4)
    logger.info(f"states.shape: {e_board.shape}, values.shape: {e_value.shape}, policies.shape: {e_policy.shape}")
    ev_dataset.set(e_board, e_value, e_policy)

    # 5. save the Dataset object to config.data_folder + "training_ev.pkl"
    output_path = f"{config.dataset_out_dir}/training_ev.pkl"
    with open(output_path, 'wb') as file:
        pickle.dump(ev_dataset, file)
    logger.info(f"Experienced values saved to {output_path}.")

def create_gtv_dataset_from_board_list(boards: list[Board]):
    '''
    Creates a ground truth value dataset from a list of boards.
    '''
    # for each board, get the value and policy from the solve data and convert it to a model input,
    # then add these to a jnp array
    # values = jnp.zeros((len(boards), 1))
    # policies = jnp.zeros((len(boards), config.board_size ** 4))
    logger.info("Get ground truth values for each board (current player perspective)...")
    states: list[jnp.ndarray] = []
    values: list[float] = []
    policies: list[jnp.ndarray] = []
    gt = GroundTruth()
    for board in tqdm(boards):
        states.append(jnp.array(board_to_input(board)))
        values.append(gt.get_outcome(board))
        policies.append(jnp.array(gt.get_1ply_policy_prob_dist_list(board, for_model=True)))
    
    # load all this into a Dataset object
    logger.info("Creating Dataset object with ground truth values...")
    jnp_states = jnp.stack(states) # (N, board_size, board_size)
    jnp_values = jnp.array(values) [:, None]  # Add [:, None] to make its shape (N, 1)
    jnp_policies = jnp.stack(policies) # (N, board_size ** 4)
    logger.info(f"states.shape: {jnp_states.shape}, values.shape: {jnp_values.shape}, policies.shape: {jnp_policies.shape}")
    gtv_dataset = Dataset(batch_size=256)
    gtv_dataset.set(jnp_states, jnp_values, jnp_policies)
    return gtv_dataset

def save_dataset(fn: str, dataset: Dataset) -> None:
    output_path = f"{config.dataset_out_dir}/{fn}.pkl"
    with open(output_path, 'wb') as file:
        pickle.dump(dataset, file)
    logger.info(f"Dataset saved to {output_path}.")

def get_neighbor_boards(boards: list[Board], boards_set: set[Board], dataset_size: int | None = None) -> list[Board]:
    '''
    Gets all neighboring (child) boards of the given boards.
    Returns a list of unique neighboring boards.
    If dataset_size is specified, returns a random sample of the neighbors.
    Modifies boards_set in-place.
    '''
    cc = Game(config.board_size, config.num_pieces, False, False, False)
    
    neighbor_boards: list[Board] = []
    for board in tqdm(boards):
        # get all possible moves for the current board
        moves = cc.generate_moves_for_given_board(board)
        for move in moves:
            # create a copy of the board and apply the move
            new_board = copy.deepcopy(board)
            new_board.apply_move(move)
            if new_board not in boards_set:
                boards_set.add(new_board)
                neighbor_boards.append(new_board)
    
    num_input = len(boards)
    num_neighbors = len(neighbor_boards)
    logger.info(f"Found {num_neighbors} unique neighbor boards from {num_input} input boards.")

    gt = GroundTruth()
    neighbor_boards = [board for board in neighbor_boards if not gt.is_trivial(board)]
    logger.info(f"Filtered out trivial boards, {len(neighbor_boards)} non-trivial neighbor boards remain ({len(neighbor_boards)/num_neighbors:.2%}).")

    if dataset_size is not None:
        neighbor_boards = random.sample(neighbor_boards, min(dataset_size, len(neighbor_boards)))
        logger.info(f"Trimmed neighbor boards to {len(neighbor_boards)} boards.")
    
    return neighbor_boards

def training_neighbors_gtv(temporary_size: int | None, final_size: int | None = None, num_neighbors: int=2):
    '''
    Generates a set of datasets.
    1. boards seen during training and their ground-truth value values and policies
    2. boards that are children of the seen boards and their GTVs
    3. repeat step 2 for 2-neighbors, 3-neighbors, and so on.
    To prevent running out of memory, the number of states to generate neighbors from is capped.
    Then the final dataset's size is further reduced.
    '''
    # get unique boards seen during training
    boards_set = get_unique_boards_from_training_data()

    # create the list of training boards. trim if necessary.
    training_boards = list(boards_set)
    if temporary_size is not None: training_boards = random.sample(training_boards, min(temporary_size, len(training_boards)))

    # create, trim, and save a Dataset with the GTV for the training boards
    training_gtv_dataset = create_gtv_dataset_from_board_list(training_boards)
    if final_size is not None: training_gtv_dataset.trim(new_size=final_size, shuffle=True)
    save_dataset("training_gtv", training_gtv_dataset)

    # for each neighbor level,
    neighbor_boards = training_boards
    for i in range(num_neighbors):
        logger.info(f"Generating {i+1}-neighbor dataset...")
        # get all the neighboring (children) boards of the previous neighbors,
        # starting with training_boards
        neighbor_boards = get_neighbor_boards(neighbor_boards, boards_set, temporary_size)
        # create, trim, and save a gtv dataset based on the generated boards
        neighbor_gtv_dataset = create_gtv_dataset_from_board_list(neighbor_boards)
        if final_size is not None: neighbor_gtv_dataset.trim(new_size=final_size, shuffle=True)
        save_dataset(f"neighbor_{i+1}_gtv", neighbor_gtv_dataset)


def main():
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="generate_datasets"
    )
    logger.info("Generating datasets...")

    random_ground_truth_values(n=1000)

    training_experienced_values(n=1000)

    training_neighbors_gtv(10000, 1000, 2)

    logger.info("Finished generating datasets.")

if __name__ == "__main__":
    main()
