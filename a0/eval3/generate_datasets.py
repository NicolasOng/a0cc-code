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
from a0.eval.generate_datasets import get_unique_boards_from_training_data, save_dataset, get_neighbor_boards

from a0.utils.states import (
    get_gtd_from_states, get_random_states, StateInfo, StatesInfo,
    get_state_info_for_states, get_states_info, log_states_info,
    filter_state_list, balance_gt_values, remove_duplicates, sample_states
)

from config import config

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def get_nd_and_nt_datasets_from_state_list(states: list[Board], gt: GroundTruth, n: int, batch_size: int) -> tuple[Dataset, Dataset]:
    '''
    Gets two Datasets from the given state list - one with non-draw states and one with non-trivial states.
    '''
    # get the state info for the states to use in filtering and logging,
    states_si = get_state_info_for_states(states, gt)

    # log the info for the states before filtering
    logger.info(f"Logging info on states before filtering:")
    log_states_info(get_states_info(states, states_si))

    # filter out unwanted states to create 2 lists,
    # one with draws removed and gt values balanced,
    states_nd, states_nd_si = filter_state_list(
        states,
        states_si,
        remove_draws=True
    )
    states_nd, states_nd_si = balance_gt_values(
        states_nd,
        states_nd_si
    )
    states_nd, states_nd_si = sample_states(
        states_nd,
        n,
        states_nd_si
    )
    logger.info(f"Logging info for states nd states:")
    log_states_info(get_states_info(states_nd, states_nd_si))

    # and one with illegal, trivial, and terminal states removed.
    states_nt, states_nt_si = filter_state_list(
        states,
        states_si,
        remove_illegal=True,
        remove_trivial=True,
        remove_terminal=True
    )
    states_nt, states_nt_si = sample_states(
        states_nt,
        n,
        states_nt_si
    )
    logger.info(f"Logging info for states nt states:")
    log_states_info(get_states_info(states_nt, states_nt_si))

    states_nd_gtd = get_gtd_from_states(states_nd, gt, batch_size, True)
    states_nt_gtd = get_gtd_from_states(states_nt, gt, batch_size, True)

    return states_nd_gtd, states_nt_gtd

def get_and_save_nd_and_nt_datasets_from_state_list(states: list[Board], gt: GroundTruth, n: int, batch_size: int, fn_nd: str, fn_nt: str) -> None:
    '''
    Gets and saves two Datasets from the given state list - one with non-draw states and one with non-trivial states.
    '''
    states_nd_gtd, states_nt_gtd = get_nd_and_nt_datasets_from_state_list(states, gt, n, batch_size)
    save_dataset(fn_nd, states_nd_gtd)
    save_dataset(fn_nt, states_nt_gtd)

def get_and_save_random_states_for_evaluation(n: int, gt: GroundTruth) -> dict[str, list[Board]]:
    '''
    n: the number of states to return for evaluation.
    gt: the ground truth to use for getting the state values and policies.
    Returns the source state list keyed by name, for downstream analysis.
    '''
    # get random states,
    # more than n to account for filtering out unwanted states
    # maybe adjust the mulitple for different game sizes? 25-6: 3x
    random_states = get_random_states(n * 3, gt)
    random_states, _ = remove_duplicates(random_states)

    # get the GTDs for the random states,
    get_and_save_nd_and_nt_datasets_from_state_list(
        random_states,
        gt,
        n,
        256,
        "random_nd",
        "random_nt"
    )

    return {"random": random_states}

def get_and_save_training_neighbors_gtd(gt: GroundTruth, temporary_size: int | None, final_size: int | None = None, num_neighbors: int=2) -> dict[str, list[Board]]:
    '''
    Generates a set of datasets.
    1. boards seen during training and their ground-truth value values and policies
    2. boards that are children of the seen boards and their GTVs
    3. repeat step 2 for 2-neighbors, 3-neighbors, and so on.
    To prevent running out of memory, the number of states to generate neighbors from is capped.
    Then the final dataset's size is further reduced.
    Returns the source state lists keyed by name, for downstream analysis.
    '''
    state_lists: dict[str, list[Board]] = {}

    # get unique boards seen during training
    boards_set = get_unique_boards_from_training_data()

    # create the list of training boards. trim if necessary.
    seen_states = list(boards_set)
    if temporary_size is not None:
        seen_states = random.sample(seen_states, min(temporary_size, len(seen_states)))

    # get and save the nd and nt gtd datasets for the training boards.
    get_and_save_nd_and_nt_datasets_from_state_list(
        seen_states,
        gt,
        n=final_size if final_size is not None else len(seen_states),
        batch_size=256,
        fn_nd="seen_nd",
        fn_nt="seen_nt"
    )
    state_lists["seen"] = seen_states

    # for each neighbor level,
    neighbor_states = seen_states
    for i in range(num_neighbors):
        logger.info(f"Generating {i+1}-neighbor dataset...")
        # get all the neighboring (children) states of the previous neighbors,
        # starting with seen states, then 1-neighbors, then 2-neighbors, and so on.
        neighbor_states = get_neighbor_boards(neighbor_states, boards_set, temporary_size)
        # create, trim, and save a gtv dataset based on the generated boards
        get_and_save_nd_and_nt_datasets_from_state_list(
            neighbor_states,
            gt,
            n=final_size if final_size is not None else len(neighbor_states),
            batch_size=256,
            fn_nd=f"neighbor_{i+1}_nd",
            fn_nt=f"neighbor_{i+1}_nt"
        )
        state_lists[f"neighbor_{i+1}"] = neighbor_states

    return state_lists

def main():
    '''
    Generates nd & nt variants of the the datasets:
    seen, random, and neighbor_[1 to k]
    Also saves the source state lists (pre-nd/nt-filter) to a single
    pickle for downstream analysis (e.g. baseline accuracy).
    '''
    logger.info("Starting dataset generation...")
    gt = GroundTruth()

    state_lists: dict[str, list[Board]] = {}
    state_lists.update(get_and_save_random_states_for_evaluation(1000, gt))
    state_lists.update(get_and_save_training_neighbors_gtd(
        gt,
        temporary_size=10000,
        final_size=1000,
        num_neighbors=2
    ))

    state_lists_path = f"{config.dataset_out_dir}/state_lists.pkl"
    with open(state_lists_path, 'wb') as f:
        pickle.dump(state_lists, f)
    logger.info(f"Saved state lists ({list(state_lists.keys())}) to {state_lists_path}.")

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name='gen_datasets'
    )
    main()
