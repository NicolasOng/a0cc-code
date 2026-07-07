from typing import Any, Optional, Generic

import random

import numpy as np

from cc.core import Board, Game, Player
from cc.ground_truth import GroundTruth, RankUnrank

from a0.experience_buffer import ExperienceData
from a0.dataset import Dataset
from a0.model_utils import board_to_input, get_legal_move_mask_from_state

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

class StateInfo:
    '''
    A class to hold information about a single state.
    '''
    def __init__(
            self,
            outcome: float,
            winner: Optional[Player],
            is_illegal: bool,
            is_draw: bool,
            is_trivial: bool,
            is_terminal: bool
        ) -> None:
        self.outcome = outcome
        self.winner = winner
        self.is_illegal = is_illegal
        self.is_draw = is_draw
        self.is_trivial = is_trivial
        self.is_terminal = is_terminal
        # TODO: num actions, num winning moves, etc

class StatesInfo:
    '''
    A class to hold information about a list of states.
    '''
    def __init__(self, num_states: int,
                 num_unique_states: int,
                 num_wins: int,
                 num_losses: int,
                 num_draw: int,
                 num_illegal: int,
                 num_trivial: int,
                 num_terminal: int
                 ) -> None:
        self.num_states = num_states
        self.num_unique_states = num_unique_states
        self.num_wins = num_wins
        self.num_losses = num_losses
        self.num_draw = num_draw
        self.num_illegal = num_illegal
        self.num_trivial = num_trivial
        self.num_terminal = num_terminal

def get_random_state(max_rank: int, r: RankUnrank) -> Board:
    '''
    Returns a random state. Accepts any RankUnrank (including GroundTruth);
    callers without solve data can pass a plain RankUnrank().
    '''
    return r.unrank(random.randint(0, max_rank - 1))

def get_random_states(n: int, r: RankUnrank) -> list[Board]:
    '''
    Returns a list of random states.
    Uniqueness is not guaranteed.
    '''
    max_rank = r.get_max_rank()
    return [get_random_state(max_rank, r) for _ in range(n)]

def remove_duplicates(states: list[Board], state_info_list: Optional[list[StateInfo]] = None) -> tuple[list[Board], Optional[list[StateInfo]]]:
    '''
    Removes duplicate states from the given list.
    '''
    before_n = len(states)
    seen: set[Board] = set()
    keep_indices: list[int] = []
    for i, s in enumerate(states):
        if s not in seen:
            seen.add(s)
            keep_indices.append(i)
    states = [states[i] for i in keep_indices]
    after_n = len(states)
    logger.info(f"Removed {before_n - after_n} duplicate states, {after_n} unique states remain. That's {100 * (before_n - after_n) / before_n:.2f}% duplicates.")
    if state_info_list is not None:
        state_info_list = [state_info_list[i] for i in keep_indices]
    return states, state_info_list

def sample_states(states: list[Board], n: int, state_info_list: Optional[list[StateInfo]] = None) -> tuple[list[Board], Optional[list[StateInfo]]]:
    '''
    Returns n randomly selected states from the given list.
    '''
    before_n = len(states)
    indices = random.sample(range(len(states)), min(n, len(states)))
    sampled_states = [states[i] for i in indices]
    logger.info(f"sample_states: {before_n} -> {n} states ({100 * n / before_n:.2f}% kept)")
    if state_info_list is not None:
        return sampled_states, [state_info_list[i] for i in indices]
    return sampled_states, None
    
def get_state_info(state: Board, gt: GroundTruth) -> StateInfo:
    '''
    Returns statistics about the given states.
    '''
    si = StateInfo(
        outcome=gt.get_outcome(state),
        winner=gt.get_winner(state),
        is_illegal=gt.is_illegal(state),
        is_draw=gt.is_draw(state),
        is_trivial=gt.is_trivial(state),
        is_terminal=gt.is_terminal(state)
    )
    return si

def get_state_info_for_states(states: list[Board], gt: GroundTruth) -> list[StateInfo]:
    '''
    Returns statistics about the given states.
    '''
    return [get_state_info(state, gt) for state in states]

def get_states_info(states: list[Board], state_info_list: list[StateInfo]) -> StatesInfo:
    '''
    Returns statistics about the given states.
    '''
    state_set = set(states)
    num_states = len(states)
    num_unique_states = len(state_set)
    num_wins = sum(1 for si in state_info_list if si.outcome > 0)
    num_losses = sum(1 for si in state_info_list if si.outcome < 0)
    num_draw = sum(1 for si in state_info_list if si.outcome == 0)
    num_illegal = sum(1 for si in state_info_list if si.is_illegal)
    num_trivial = sum(1 for si in state_info_list if si.is_trivial)
    num_terminal = sum(1 for si in state_info_list if si.is_terminal)
    si = StatesInfo(
        num_states=num_states,
        num_unique_states=num_unique_states,
        num_wins=num_wins,
        num_losses=num_losses,
        num_draw=num_draw,
        num_illegal=num_illegal,
        num_trivial=num_trivial,
        num_terminal=num_terminal
    )
    return si

def log_states_info(si: StatesInfo) -> None:
    '''
    Logs statistics about the given state list.
    '''
    n = si.num_states
    logger.info(f"Total states: {n}, Unique states: {si.num_unique_states} ({si.num_unique_states / n:.2%})")
    logger.info(f"Wins: {si.num_wins} ({si.num_wins / n:.2%}), Losses: {si.num_losses} ({si.num_losses / n:.2%}), Draws: {si.num_draw} ({si.num_draw / n:.2%}), Illegal: {si.num_illegal} ({si.num_illegal / n:.2%}), Trivial: {si.num_trivial} ({si.num_trivial / n:.2%}), Terminal: {si.num_terminal} ({si.num_terminal / n:.2%})")

def filter_state_list(
        states: list[Board],
        state_info_list: list[StateInfo],
        remove_draws: bool = False,
        remove_illegal: bool = False,
        remove_trivial: bool = False,
        remove_terminal: bool = False,
        remove_wins: bool = False,
        remove_losses: bool = False
        ) -> tuple[list[Board], list[StateInfo]]:
    '''
    Filters the given state list based on the provided criteria.
    Returns the filtered states and their corresponding StateInfo.
    '''
    def keep(si: StateInfo) -> bool:
        if remove_draws and si.is_draw:
            return False
        if remove_illegal and si.is_illegal:
            return False
        if remove_trivial and si.is_trivial:
            return False
        if remove_terminal and si.is_terminal:
            return False
        if remove_wins and si.outcome > 0:
            return False
        if remove_losses and si.outcome < 0:
            return False
        return True

    before_n = len(states)
    pairs = [(s, si) for s, si in zip(states, state_info_list) if keep(si)]
    if not pairs:
        logger.info(f"filter_state_list: {before_n} -> 0 states (100.00% removed)")
        return [], []
    filtered_states, filtered_info = zip(*pairs)
    after_n = len(filtered_states)
    logger.info(f"filter_state_list: {before_n} -> {after_n} states ({100 * (before_n - after_n) / before_n:.2f}% removed)")
    return list(filtered_states), list(filtered_info)

def balance_gt_values(states: list[Board], state_info_list: list[StateInfo]) -> tuple[list[Board], list[StateInfo]]:
    '''
    Removes bias from the given state list by removing winning or losing states - whichever is more common.
    Randomly selects which majority-class states to drop.
    '''
    num_wins = sum(1 for si in state_info_list if si.outcome > 0)
    num_losses = sum(1 for si in state_info_list if si.outcome < 0)
    num_to_remove = abs(num_wins - num_losses)
    if num_to_remove == 0:
        pairs = list(zip(states, state_info_list))
        random.shuffle(pairs)
        return [p[0] for p in pairs], [p[1] for p in pairs]

    remove_positive = num_wins > num_losses
    majority_indices = {
        i for i, si in enumerate(state_info_list)
        if (si.outcome > 0) == remove_positive
    }
    drop = set(random.sample(sorted(majority_indices), num_to_remove))
    pairs = [(s, si) for i, (s, si) in enumerate(zip(states, state_info_list)) if i not in drop]
    random.shuffle(pairs)
    return [p[0] for p in pairs], [p[1] for p in pairs]

def convert_state_to_gt_experience(state: Board, gt: GroundTruth) -> ExperienceData:
    '''
    Converts a state to an ExperienceData object using the GTProtocol.
    '''
    board_input = board_to_input(state) # (1, board_size, board_size, 2)
    value = gt.get_outcome(state) # float
    policy = np.array(gt.get_1ply_policy_prob_dist_list(state, for_model=True)) # (board_size ** 4,)
    mask = get_legal_move_mask_from_state(state, for_model=True) # (board_size ** 4,)
    return ExperienceData(board=board_input, value=value, policy=policy, mask=mask)

def convert_state_list_to_gt_experience_list(states: list[Board], gt: GroundTruth) -> list[ExperienceData]:
    '''
    Converts a list of states to a list of ExperienceData objects using the GTProtocol.
    '''
    return [convert_state_to_gt_experience(state, gt) for state in states]

def convert_experience_list_to_dataset(experience_list: list[ExperienceData], batch_size: int) -> Dataset:
    '''
    Converts a list of ExperienceData objects to a Dataset object.
    '''
    new_dataset = Dataset(batch_size)
    new_dataset.set(
        np.stack([d.board[0] for d in experience_list]), # (1, board_size, board_size, 2) -> (N, board_size, board_size, 2)
        np.array([d.value for d in experience_list]) [:, None], # Add [:, None] to make its shape (N, 1)
        np.stack([d.policy for d in experience_list]), # (board_size ** 4) -> (N, board_size ** 4)
        np.stack([d.mask for d in experience_list]) # (board_size ** 4) -> (N, board_size ** 4)
    )
    new_dataset.print_shapes()
    return new_dataset

def get_gtd_from_states(states: list[Board], gt: GroundTruth, batch_size: int, shuffle: bool) -> Dataset:
    '''
    Gets a ground truth dataset from a list of states.
    '''
    a = convert_state_list_to_gt_experience_list(states, gt)
    b = convert_experience_list_to_dataset(a, batch_size)
    if shuffle:
        b.shuffle()
    return b

def get_rd_from_states(states: list[Board], batch_size: int, shuffle: bool) -> Dataset:
    '''
    Gets a random dataset from a list of states.
    The values and policies are random, but the masks are correct.
    '''
    boards = np.stack([board_to_input(state)[0] for state in states]) # (N, board_size, board_size, 2)
    policies = np.random.rand(len(states), config.board_size ** 4).astype(np.float32) # (N, board_size ** 4)
    masks = np.stack([get_legal_move_mask_from_state(state, for_model=True) for state in states]) # (N, board_size ** 4)
    values = np.random.uniform(-1, 1, size=(len(states), 1)).astype(np.float32) # (N, 1)

    dataset = Dataset(batch_size)
    dataset.set(boards, values, policies, masks)
    if shuffle:
        dataset.shuffle()
    return dataset
