import pickle
import random
import copy

from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray

from cc.core import Board, Game
from cc.ground_truth import GroundTruth
from a0.dataset import Dataset
from a0.experience_buffer import ExperienceData
from a0.model_utils import board_to_input, input_to_board, Policy, get_legal_move_mask_from_state
from a0.eval.training_data import game_data_generator
from a0.eval.dataset_evaluation import Series, save_series

from config import config

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def random_ground_truth_values(n: int = 1000, gen_non_trivial: bool = True, remove_bias: bool = True) -> None:
    '''
    Generates and saves a dataset of ground truth values for all unique boards
    from a list of randomly generated boards.
    The dataset contains the board, the ground truth value, and the ground truth policy.
    If gen_non_trivial is True, also generates a dataset of non-trivial boards only
    If remove_bias is True, balances the dataset to have an equal number of each wins and losses.
    '''
    logger.info(f"Generating ground truth values from {n} random boards...")

    # 1. generate a list of random boards
    logger.info("Generating random boards...")
    gt = GroundTruth()
    max_rank = gt.get_max_rank()
    board_set: set[Board] = set()
    to_generate = n * 10 if gen_non_trivial else n * 2
    while len(board_set) < to_generate:
        b = gt.unrank(random.randint(0, max_rank - 1))
        board_set.add(b)
    
    non_trivial_boards: list[Board] = []
    if gen_non_trivial:
        non_trivial_boards = remove_trivial_boards(list(board_set))
        logger.info(f"Generated {len(board_set)} unique boards, of which {len(non_trivial_boards)} are non-trivial ({len(non_trivial_boards)/len(board_set):.2%}).")
    else:
        logger.info(f"Generated {len(board_set)} unique boards.")

    # create and save a gtv dataset based on the generated boards
    boards: list[Board] = list(board_set)
    rgtv_dataset = create_gtv_dataset_from_board_list(boards)
    if remove_bias:
        #rgtv_dataset = balance_dataset(rgtv_dataset)
        rgtv_dataset.balance_values()
    rgtv_dataset.trim(new_size=n, shuffle=True)
    save_dataset("random_gtv", rgtv_dataset)

    if gen_non_trivial:
        rgtv_dataset_non_trivial = create_gtv_dataset_from_board_list(non_trivial_boards)
        rgtv_dataset_non_trivial.trim(new_size=n, shuffle=True)
        save_dataset("random_nt_gtv", rgtv_dataset_non_trivial)

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

    return boards_set

def training_experienced_values(n: int | None = None, gen_non_trivial: bool = True, remove_bias: bool = True) -> None:
    '''
    Generates and saves a dataset of experienced values for all unique boards
    from the training data generated during training/self-play.
    The dataset contains the board, the experienced value, and the experienced policy.
    The experienced value is the outcome of the game from the perspective of the current player.
    '''
    logger.info("Generating experienced values from training data...")

    # get the unique boards (with their outcome/policy) from the training data
    gt = GroundTruth()
    logger.info("Getting unique boards from training data, with their experienced outcome...")
    game_data_lists = game_data_generator(config.training_dir, config.training_iterations)
    boards_set: set[ExperienceData] = set()
    boards_set_non_trivial: set[ExperienceData] = set()
    total_boards = 0
    for _, game_data_list in tqdm(game_data_lists):
        for game_data in game_data_list:
            game_winner = game_data.winner
            turn_data = game_data.turn_data
            for turn in turn_data:
                # get the board and its experienced outcome (current player perspective)
                board = turn.board
                experienced_outcome = 0.0 if game_winner is None else 1.0 if game_winner == board.current_player else -1.0
                experienced_policy = turn.player_data
                mask = get_legal_move_mask_from_state(board, for_model=True) # (board_size ** 4,)
                experience = ExperienceData(board_to_input(board), experienced_outcome, experienced_policy, mask)

                # add this to a set
                boards_set.add(experience)
                total_boards += 1

                # also to the non-trivial set if needed
                if gen_non_trivial and not gt.is_trivial(board):
                    boards_set_non_trivial.add(experience)
    
    num_unique = len(boards_set)
    num_unique_non_trivial = len(boards_set_non_trivial)
    logger.info(f"Found {num_unique} unique and boards+value+policy triplets in the game data, out of {total_boards} total boards ({num_unique/total_boards:.2%}).")
    logger.info(f"Found {num_unique_non_trivial} unique and non-trivial boards+value+policy triplets in the game data, out of {total_boards} total boards ({num_unique_non_trivial/total_boards:.2%}).")
    
    # load all this into a Dataset object
    boards = list(boards_set)
    ev_dataset = Dataset(batch_size=256)
    e_board = np.concatenate([d.board for d in boards], axis=0) # (1, board_size, board_size, 2) -> (N, board_size, board_size, 2)
    e_value = np.array([d.value for d in boards]).reshape(-1, 1) # list[float] -> (N, 1)
    e_policy = np.stack([d.policy for d in boards]) # (board_size ** 4) -> (N, board_size ** 4)
    e_mask = np.stack([d.mask for d in boards]) # (board_size ** 4) -> (N, board_size ** 4)
    logger.info(f"states.shape: {e_board.shape}, values.shape: {e_value.shape}, policies.shape: {e_policy.shape}")
    ev_dataset.set(e_board, e_value, e_policy, e_mask)
    if remove_bias:
        #ev_dataset = balance_dataset(ev_dataset)
        ev_dataset.balance_values()
    if n is not None:
        ev_dataset.trim(new_size=n, shuffle=True)

    # save the Dataset object to config.data_folder + "training_ev.pkl"
    output_path = f"{config.dataset_out_dir}/training_ev.pkl"
    with open(output_path, 'wb') as file:
        pickle.dump(ev_dataset, file)
    logger.info(f"Experienced values saved to {output_path}.")

    if gen_non_trivial:
        boards = list(boards_set_non_trivial)
        ev_dataset_non_trivial = Dataset(batch_size=256)
        e_board = np.concatenate([d.board for d in boards], axis=0) # (1, board_size, board_size, 2) -> (N, board_size, board_size, 2)
        e_value = np.array([d.value for d in boards]).reshape(-1, 1) # list[float] -> (N, 1)
        e_policy = np.stack([d.policy for d in boards]) # (board_size ** 4) -> (N, board_size ** 4)
        e_mask = np.stack([d.mask for d in boards]) # (board_size ** 4) -> (N, board_size ** 4)
        logger.info(f"states.shape: {e_board.shape}, values.shape: {e_value.shape}, policies.shape: {e_policy.shape}")
        ev_dataset_non_trivial.set(e_board, e_value, e_policy, e_mask)
        if n is not None:
            ev_dataset_non_trivial.trim(new_size=n, shuffle=True)
        output_path = f"{config.dataset_out_dir}/training_nt_ev.pkl"
        with open(output_path, 'wb') as file:
            pickle.dump(ev_dataset_non_trivial, file)
        logger.info(f"Experienced values for non-trivial boards saved to {output_path}.")

def create_gtv_dataset_from_board_list(boards: list[Board]):
    '''
    Creates a ground truth value dataset from a list of boards.
    '''
    # for each board, get the value and policy from the solve data and convert it to a model input,
    # then add these to a np array
    # values = jnp.zeros((len(boards), 1))
    # policies = jnp.zeros((len(boards), config.board_size ** 4))
    logger.info("Get ground truth values for each board (current player perspective)...")
    states: list[NDArray[np.float32]] = []
    values: list[float] = []
    policies: list[NDArray[np.float32]] = []
    masks: list[NDArray[np.float32]] = []
    gt = GroundTruth()
    for board in tqdm(boards):
        states.append(board_to_input(board)) # (1, board_size, board_size, 2)
        values.append(gt.get_outcome(board)) # float
        policies.append(np.array(gt.get_1ply_policy_prob_dist_list(board, for_model=True))) # (board_size ** 4,)
        masks.append(get_legal_move_mask_from_state(board, for_model=True)) # (board_size ** 4,)
    
    # load all this into a Dataset object
    logger.info("Creating Dataset object with ground truth values...")
    np_states = np.concatenate(states, axis=0) # (N, board_size, board_size, 2)
    np_values = np.array(values).reshape(-1, 1)  # (N, 1)
    np_policies = np.stack(policies) # (N, board_size ** 4)
    np_masks = np.stack(masks) # (N, board_size ** 4)
    logger.info(f"states.shape: {np_states.shape}, values.shape: {np_values.shape}, policies.shape: {np_policies.shape}")
    gtv_dataset = Dataset(batch_size=256)
    gtv_dataset.set(np_states, np_values, np_policies, np_masks)
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
    cc = Game(board_size=config.board_size,
            num_pieces=config.num_pieces,
            repeats_for_draw=-1,
            no_reverse_moves=False,
            no_illegal_moves=False)
    
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

    if dataset_size is not None:
        neighbor_boards = random.sample(neighbor_boards, min(dataset_size, len(neighbor_boards)))
        logger.info(f"Trimmed neighbor boards to {len(neighbor_boards)} boards.")
    
    return neighbor_boards

def training_neighbors_gtv(temporary_size: int | None, final_size: int | None = None, num_neighbors: int=2, gen_non_trivial: bool = True, remove_bias: bool = True):
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
    if remove_bias:
        #training_gtv_dataset = balance_dataset(training_gtv_dataset)
        training_gtv_dataset.balance_values()
    if final_size is not None: training_gtv_dataset.trim(new_size=final_size, shuffle=True)
    save_dataset("training_gtv", training_gtv_dataset)

    # if needed, create, trim, and save a Dataset with the GTV for the non-trivial training boards
    if gen_non_trivial:
        training_boards_nt = remove_trivial_boards(training_boards)
        training_gtv_dataset_non_trivial = create_gtv_dataset_from_board_list(training_boards_nt)
        if final_size is not None: training_gtv_dataset_non_trivial.trim(new_size=final_size, shuffle=True)
        save_dataset("training_nt_gtv", training_gtv_dataset_non_trivial)

    # for each neighbor level,
    neighbor_boards = training_boards
    for i in range(num_neighbors):
        logger.info(f"Generating {i+1}-neighbor dataset...")
        # get all the neighboring (children) boards of the previous neighbors,
        # starting with training_boards
        neighbor_boards = get_neighbor_boards(neighbor_boards, boards_set, temporary_size)
        # create, trim, and save a gtv dataset based on the generated boards
        neighbor_gtv_dataset = create_gtv_dataset_from_board_list(neighbor_boards)
        if remove_bias:
            #neighbor_gtv_dataset = balance_dataset(neighbor_gtv_dataset)
            neighbor_gtv_dataset.balance_values()
        if final_size is not None: neighbor_gtv_dataset.trim(new_size=final_size, shuffle=True)
        save_dataset(f"neighbor_{i+1}_gtv", neighbor_gtv_dataset)

        # if needed, create, trim, and save a Dataset with the GTV for the non-trivial neighbor boards
        if gen_non_trivial:
            neighbor_boards_nt = remove_trivial_boards(neighbor_boards)
            neighbor_gtv_dataset_non_trivial = create_gtv_dataset_from_board_list(neighbor_boards_nt)
            if final_size is not None: neighbor_gtv_dataset_non_trivial.trim(new_size=final_size, shuffle=True)
            save_dataset(f"neighbor_{i+1}_nt_gtv", neighbor_gtv_dataset_non_trivial)

def state_progress_gtv_datasets(num_bins: int = 100, size: int | None = 500, gen_non_trivial: bool = True, remove_bias: bool = True):
    '''
    Generates and saves datasets of ground truth values for boards at different stages of the game.
    The stages are determined by the number of pieces on the board, divided into num_bins bins.
    Each dataset contains boards with a number of pieces within the bin range.
    '''
    logger.info("Generating state progress GTV datasets...")

    # the bins dict holds the boards by their game progress,
    # or how far into the game they appear.
    # for example: 0% to 10%, 10% to 20%, ..., 90% to 100%
    # bins are accessed by their lower bound, e.g. 0, 10, 20, ..., 90
    assert num_bins > 0 and num_bins <= 100, "num_bins must be between 1 and 100 (inclusive)"
    bins: dict[int, set[Board]] = {100 * i // num_bins: set() for i in range(num_bins)}
    bins_nt: dict[int, list[Board]] = {100 * i // num_bins: list() for i in range(num_bins)}
    state_progress_state_nums = Series(["Num States", "Num States NT"])

    # get the game data lists from the training data generator
    game_data_lists = game_data_generator(config.training_dir, config.training_iterations)

    # put all the boards from the game data into the appropriate bins
    logger.info("Extracting boards from game data into bins...")
    for _, game_data_list in tqdm(game_data_lists):
        for game_data in game_data_list:
            game_length = len(game_data.turn_data)
            turn_data = game_data.turn_data
            for turn_no, turn in enumerate(turn_data):
                progress = (turn_no * 100) // game_length
                # find the appropriate bin for this progress
                bin_key = max([k for k in bins.keys() if k <= progress])
                #logger.info(f"Turn {turn_no}/{game_length}, progress: {progress}%, bin: {bin_key}%")
                # add the board to the appropriate bin
                bins[bin_key].add(turn.board)

    # log the number of boards in each bin
    for bin_key, bin_boards in bins.items():
        logger.info(f"Bin {bin_key}: {len(bin_boards)} boards")
        state_progress_state_nums.x.append(bin_key)
        state_progress_state_nums.ys["Num States"].append(len(bin_boards))

    # if needed, remove trivial boards from each bin
    if gen_non_trivial:
        logger.info("Removing trivial boards from each bin...")
        for bin_key, bin_boards in bins.items():
            non_trivial_boards = remove_trivial_boards(list(bin_boards))
            bins_nt[bin_key] = non_trivial_boards
            logger.info(f"Bin {bin_key}: {len(non_trivial_boards)}/{len(bin_boards)} non-trivial boards ({len(non_trivial_boards)/len(bin_boards) if len(bin_boards) > 0 else 0:.2%})")
    
        # log the number of boards in each bin
        for bin_key, bin_boards in bins_nt.items():
            logger.info(f"Bin {bin_key}: {len(bin_boards)} boards")
            state_progress_state_nums.ys["Num States NT"].append(len(bin_boards))
    
    save_series(state_progress_state_nums, f"{config.eval_dir}/state_progress_state_nums.pkl")
    
    # create a dataset for each bin
    logger.info("Creating dataset for each bin...")
    datasets = {bin_key: create_gtv_dataset_from_board_list(list(bin_boards)) for bin_key, bin_boards in bins.items() if bin_boards}
    datasets_nt = {bin_key: create_gtv_dataset_from_board_list(bin_boards) for bin_key, bin_boards in bins_nt.items() if bin_boards} if gen_non_trivial else {}

    # balance each dataset if needed
    logger.info("Balancing datasets to remove bias...")
    if remove_bias:
        for bin_key, dataset in datasets.items():
            #dataset = balance_dataset(dataset)
            dataset.balance_values()
            datasets[bin_key] = dataset
    
    # trim each dataset to the specified size
    logger.info("Trimming datasets to specified size...")
    if size is not None:
        for bin_key, dataset in datasets.items():
            dataset.trim(new_size=size, shuffle=True)
        if gen_non_trivial:
            for bin_key, dataset in datasets_nt.items():
                dataset.trim(new_size=size, shuffle=True)
    
    # save the dataset dict into a single file
    logger.info("Saving state progress GTV datasets...")
    output_path = f"{config.dataset_out_dir}/state_progress_gtv_datasets.pkl"
    with open(output_path, 'wb') as file:
        pickle.dump(datasets, file)
    logger.info(f"State progress GTV datasets saved to {output_path}.")

    if gen_non_trivial:
        output_path = f"{config.dataset_out_dir}/state_progress_nt_gtv_datasets.pkl"
        with open(output_path, 'wb') as file:
            pickle.dump(datasets_nt, file)
        logger.info(f"State progress non-trivial GTV datasets saved to {output_path}.")

def balance_dataset(dataset: Dataset) -> Dataset:
    '''
    Balances the dataset to have an equal number of each value (-1, 0, 1).
    Modifies the dataset in-place and returns it.
    Instead of trimming, it oversamples the minority classes.
    When sampling from the minority class, it mirrors the board to create a new sample.
    '''
    logger.info("Balancing dataset...")
    # get the dataset statistics
    dataset.shuffle()
    num_wins, num_draws, num_losses = dataset.get_distribution()
    logger.info(f"Current distribution: Wins: {num_wins}, Draws: {num_draws}, Losses: {num_losses}")
    if num_wins == num_losses:
        logger.info("Dataset is already balanced.")
        return dataset
    
    # determine which is the minority class and how many samples to add
    min_class = min(num_wins, num_losses)
    max_class = max(num_wins, num_losses)
    to_add = max_class - min_class
    add_wins = num_wins < num_losses
    logger.info(f"Minority class: {'Wins' if add_wins else 'Losses'}, need to add {to_add} samples.")

    # go through the dataset.
    # for each sample in the minority class, create a mirrored board (with b.flip_horizontal()) and add it to the dataset
    # until the dataset is balanced
    # create lists to hold the new samples
    flipped_states: list[NDArray[np.float32]] = []
    flipped_values: list[float] = []
    flipped_policies: list[NDArray[np.float32]] = []
    # for each sample in the dataset
    for i in range(len(dataset.states)):
        # get the board, value, and policy
        cur_value = dataset.values[i, 0]
        cur_board = dataset.states[i]
        cur_policy = dataset.policies[i]
        value = float(cur_value)
        # add the original sample to the new lists
        flipped_states.append(cur_board)
        flipped_values.append(value)
        flipped_policies.append(cur_policy)
        # if we need more flipped samples, and this sample is in the minority class
        if (to_add > 0) and ((add_wins and value == 1.0) or (not add_wins and value == -1.0)):
            # create a mirrored board
            flipped_board = input_to_board(np.array(cur_board))
            flipped_board.flip_horizontal()
            flipped_board_np = np.array(board_to_input(flipped_board))
            # create a mirrored policy
            p = Policy(len(flipped_board.board))
            p.set_logits(np.array(cur_policy), rotate_180=False)
            p.flip_policy(horizontal=True)
            flipped_policy_np = np.array(p.policy)
            # add the new sample to the dataset
            flipped_states.append(flipped_board_np)
            flipped_values.append(value)
            flipped_policies.append(flipped_policy_np)
            # decrement the number of samples to add
            to_add -= 1
    
    # add the new samples to the dataset
    balanced_dataset = Dataset(batch_size=256)
    e_board = np.stack([board for board in flipped_states]) # (board_size, board_size) -> (N, board_size, board_size)
    e_value = np.array([value for value in flipped_values])[:, None] # Add [:, None] to make its shape (N, 1)
    e_policy = np.stack([policy for policy in flipped_policies]) # (board_size ** 4) -> (N, board_size ** 4)
    logger.info(f"states.shape: {e_board.shape}, values.shape: {e_value.shape}, policies.shape: {e_policy.shape}")
    balanced_dataset.set(e_board, e_value, e_policy)
    # shuffle the dataset
    balanced_dataset.shuffle()
    # balance the dataset if there wasn't enough samples in the minority class
    # to duplicate to balance the dataset fully
    balanced_dataset.balance_values()
    # print the new distribution
    num_wins, num_draws, num_losses = balanced_dataset.get_distribution()
    logger.info(f"Current distribution: Wins: {num_wins}, Draws: {num_draws}, Losses: {num_losses}")
    return balanced_dataset

def remove_trivial_boards(boards: list[Board]) -> list[Board]:
    '''
    Removes trivial boards from the given list of boards.
    Returns a new list of non-trivial boards.
    '''
    gt = GroundTruth()
    non_trivial_boards = [board for board in boards if not gt.is_trivial(board)]
    percentage = len(non_trivial_boards) / len(boards) if boards else 0
    logger.info(f"Removed trivial boards. {percentage:.2%} non-trivial boards remain ({len(non_trivial_boards)}/{len(boards)}).")
    return non_trivial_boards

def main():
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="generate_datasets"
    )
    logger.info("Generating datasets...")

    gen_non_trivial = True
    remove_bias = True

    random_ground_truth_values(n=1000, gen_non_trivial=gen_non_trivial, remove_bias=remove_bias)

    training_experienced_values(n=1000, gen_non_trivial=gen_non_trivial, remove_bias=remove_bias)

    training_neighbors_gtv(10000, 1000, 2, gen_non_trivial=gen_non_trivial, remove_bias=remove_bias)

    state_progress_gtv_datasets(100, 500, gen_non_trivial=gen_non_trivial, remove_bias=remove_bias)

    logger.info("Finished generating datasets.")

if __name__ == "__main__":
    main()
