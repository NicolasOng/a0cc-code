import pickle
import random
import copy

from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray

from cc.core import Board, Game
from cc.ground_truth import GroundTruth, RankUnrank
from a0.dataset import Dataset
from a0.experience_buffer import ExperienceData
from a0.model_utils import board_to_input, input_to_board, Policy, get_legal_move_mask_from_state
from a0.eval.training_data import game_data_generator
from a0.eval.dataset_evaluation import Series, save_series
from a0.eval.generate_datasets import get_unique_boards_from_training_data, save_dataset, get_neighbor_boards

from a0.utils.states import (
    get_gtd_from_states, get_rd_from_states, get_random_states, StateInfo, StatesInfo,
    get_state_info_for_states, get_states_info, log_states_info,
    filter_state_list, balance_gt_values, remove_duplicates, sample_states
)

from config import config

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def get_nd_and_nt_datasets_from_state_list(states: list[Board], gt: GroundTruth, n: int, batch_size: int,
                                           balance: bool = True) -> tuple[Dataset, Dataset]:
    '''
    Gets two Datasets from the given state list - one with non-draw states and one with non-trivial states.

    balance=False skips the win/loss balancing of the nd set. Needed for
    distance-from-terminal buckets, which are inherently one-sided near the end
    of the game (every distance-0 state is a win for the player to move), so
    balancing empties or decimates exactly the rows of most interest. Callers
    that skip it should report the class balance alongside the accuracy, since
    "chance" is then the majority-class rate rather than 0.5.
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
    if balance:
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

def get_random_state_list(n: int, r: RankUnrank) -> list[Board]:
    '''
    Returns a deduped list of ~n*3 random states (the 3x is to absorb
    de-duplication; n itself is the target downstream-dataset size).
    Solve-data-free.
    '''
    random_states = get_random_states(n * 3, r)
    random_states, _ = remove_duplicates(random_states)
    return random_states

def get_training_dataset(n: int, gt: GroundTruth) -> tuple[Dataset, list[Board]]:
    '''
    Gets a dataset that can be used to train a model.
    '''
    # get random states
    states = get_random_states(n * 3, gt)
    states, _ = remove_duplicates(states)

    # get the state info for the states to use in filtering and logging,
    states_si = get_state_info_for_states(states, gt)

    # log the info for the states before filtering
    logger.info(f"Logging info on states before filtering:")
    log_states_info(get_states_info(states, states_si))

    # balance the wins/draws
    states, states_si = balance_gt_values(
        states,
        states_si
    )

    # logging the final info for the states after filtering and balancing
    logger.info(f"Logging info for states after filtering and balancing:")
    log_states_info(get_states_info(states, states_si))

    return get_gtd_from_states(states, gt, batch_size=256, shuffle=True), states

def get_seen_and_neighbor_state_lists(temporary_size: int | None, num_neighbors: int) -> dict[str, list[Board]]:
    '''
    Builds state lists for boards seen during training plus k-neighbor
    expansions. Solve-data-free (uses Game logic only).
    '''
    state_lists: dict[str, list[Board]] = {}

    # get unique boards seen during training
    boards_set = get_unique_boards_from_training_data()

    # create the list of training boards. trim if necessary.
    seen_states = list(boards_set)
    if temporary_size is not None:
        seen_states = random.sample(seen_states, min(temporary_size, len(seen_states)))
    state_lists["seen"] = seen_states

    neighbor_states = seen_states
    for i in range(num_neighbors):
        logger.info(f"Generating {i+1}-neighbor state list...")
        # get all the neighboring (children) states of the previous neighbors,
        # starting with seen states, then 1-neighbors, then 2-neighbors, and so on.
        neighbor_states = get_neighbor_boards(neighbor_states, boards_set, temporary_size)
        state_lists[f"neighbor_{i+1}"] = neighbor_states

    return state_lists


def save_nd_nt_datasets_for_state_lists(state_lists: dict[str, list[Board]], gt: GroundTruth, n: int, batch_size: int) -> None:
    '''
    For each (name, states) entry in state_lists, build and save the nd
    and nt GT-labeled datasets to {dataset_out_dir}/{name}_nd.pkl and
    {name}_nt.pkl. Requires solve data via GroundTruth.
    '''
    for name, states in state_lists.items():
        logger.info(f"Saving nd/nt datasets for source '{name}' ({len(states)} states)...")
        get_and_save_nd_and_nt_datasets_from_state_list(
            states, gt,
            n=n,
            batch_size=batch_size,
            fn_nd=f"{name}_nd",
            fn_nt=f"{name}_nt",
        )


def main():
    '''
    Generates source state lists (random / seen / neighbor_[1..k]) and
    GT-labeled nd & nt dataset variants for each.
      - State lists are produced solve-data-free and saved to
        {dataset_out_dir}/state_lists.pkl.
      - The nd/nt datasets require ground truth and are saved
        per-source to {dataset_out_dir}/{name}_{nd,nt}.pkl.
    '''
    logger.info("Starting dataset generation...")

    # state lists: solve-data-free
    ranker = RankUnrank()
    state_lists: dict[str, list[Board]] = {}
    state_lists["random"] = get_random_state_list(1000, ranker)
    state_lists.update(get_seen_and_neighbor_state_lists(
        temporary_size=10000,
        num_neighbors=2,
    ))

    state_lists_path = f"{config.dataset_out_dir}/state_lists.pkl"
    with open(state_lists_path, 'wb') as f:
        pickle.dump(state_lists, f)
    logger.info(f"Saved state lists ({list(state_lists.keys())}) to {state_lists_path}.")

    # truly-random dataset: n random samples from state space, no GT-based filtering
    save_dataset("random", get_rd_from_states(get_random_states(1000, ranker), batch_size=256, shuffle=False))

    # nd/nt labeled datasets: requires solve data
    if config.do_gt_evals:
        gt = GroundTruth()
        save_nd_nt_datasets_for_state_lists(state_lists, gt, n=1000, batch_size=256)
    else:
        logger.info("config.do_gt_evals=False; skipping nd/nt labeled dataset generation.")

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name='gen_datasets'
    )
    main()
