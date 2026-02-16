from typing import Any, Optional, Generic

from a0_new.protocols.ground_truth import GTProtocol, PolicyType
from a0_new.protocols.game import T_state, T_action, Player
from a0_new.protocols.model import FullModelOnRaw

import random

import numpy as np

from a0_new.utils.parallel_decorator import parallel

from a0_new.policy import Policy
from a0_new.experience_buffer import ExperienceData
from a0_new.dataset import Dataset
from a0_new.utils.full_experience_data import FullExperienceData, convert_fed_list_to_experience_list

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def get_random_state(max_rank: int, gt: GTProtocol[T_state, T_action]) -> T_state:
    '''
    Returns a random state rank from the GTProtocol.
    '''
    return gt.unrank(random.randint(0, max_rank - 1))

def get_random_states(num_states: int, gt: GTProtocol[T_state, T_action]) -> list[T_state]:
    '''
    Returns a list of random states from the GTProtocol.
    '''
    max_rank = gt.get_max_rank()
    return [get_random_state(max_rank, gt) for _ in range(num_states)]

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
    
def get_state_info(state: T_state, gt: GTProtocol[T_state, T_action]) -> StateInfo:
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

def get_state_info_for_states(states: list[T_state], gt: GTProtocol[T_state, T_action]) -> list[StateInfo]:
    '''
    Returns statistics about the given states.
    '''
    return [get_state_info(state, gt) for state in states]

def get_states_info(states: list[T_state], state_info_list: list[StateInfo]) -> StatesInfo:
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
    logger.info(f"Total states: {si.num_states}, Unique states: {si.num_unique_states}")
    logger.info(f"Wins: {si.num_wins}, Losses: {si.num_losses}, Draws: {si.num_draw}, Illegal: {si.num_illegal}, Trivial: {si.num_trivial}, Terminal: {si.num_terminal}")

def filter_state_list(
        states: list[T_state],
        state_info_list: list[StateInfo],
        remove_draws: bool = False,
        remove_illegal: bool = False,
        remove_trivial: bool = False,
        remove_terminal: bool = False
        ) -> list[T_state]:
    '''
    Filters the given state list based on the provided criteria.
    '''
    filtered_states: list[T_state] = []
    for state, si in zip(states, state_info_list):
        if remove_draws and si.is_draw:
            continue
        if remove_illegal and si.is_illegal:
            continue
        if remove_trivial and si.is_trivial:
            continue
        if remove_terminal and si.is_terminal:
            continue
        filtered_states.append(state)
    return filtered_states

def remove_bias(states: list[T_state], state_info_list: list[StateInfo]) -> list[T_state]:
    '''
    Removes bias from the given state list by removing winning or losing states - whichever is more common.
    Shuffles the states before filtering to avoid any ordering bias.
    '''
    # shuffle states and state_info_list together to avoid ordering bias
    combined = list(zip(states, state_info_list))
    random.shuffle(combined)
    states[:], state_info_list[:] = zip(*combined)
    # remove the more common outcome (win or loss) to reduce bias
    num_wins = sum(1 for si in state_info_list if si.outcome > 0)
    num_losses = sum(1 for si in state_info_list if si.outcome < 0)
    to_remove = 1 if num_wins > num_losses else -1
    num_to_remove = num_wins - num_losses if to_remove == 1 else num_losses - num_wins
    filtered_states: list[T_state] = []
    removed = 0
    for state, si in zip(states, state_info_list):
        if removed >= num_to_remove:
            filtered_states.append(state)
        else:
            if (to_remove == 1 and si.outcome > 0) or (to_remove == -1 and si.outcome < 0):
                removed += 1
            else:
                filtered_states.append(state)
    random.shuffle(filtered_states)
    return filtered_states

def convert_state_to_full_gt_experience(state: T_state, gt: GTProtocol[T_state, T_action]) -> FullExperienceData[T_state, T_action]:
    '''
    Converts a state to a FullExperienceData object using the GTProtocol.
    '''
    value = gt.get_outcome(state)
    policy = gt.get_policy(state, p_type=PolicyType.PROB_DIST)
    return FullExperienceData(state=state, value=value, policy=policy)  

def convert_state_list_to_full_gt_experience_list(states: list[T_state], gt: GTProtocol[T_state, T_action]) -> list[FullExperienceData[T_state, T_action]]:
    '''
    Converts a list of states to a list of FullExperienceData objects using the GTProtocol.
    '''
    return [convert_state_to_full_gt_experience(state, gt) for state in states]

def convert_experience_list_to_dataset(experience_list: list[ExperienceData]) -> Dataset:
    '''
    Converts a list of ExperienceData objects to a Dataset object.
    '''
    dataset = Dataset(config.training_batch_size)
    dataset.set(
        np.stack([exp.state for exp in experience_list]),
        np.array([exp.value for exp in experience_list])[:, None], # Add [:, None] to make it (N, 1)
        np.stack([exp.policy for exp in experience_list]),
        np.stack([exp.mask for exp in experience_list])
    )
    return dataset

def get_gtd_from_states(states: list[T_state], gt: GTProtocol[T_state, T_action], adapter: FullModelOnRaw[Any, T_state, T_action]) -> Dataset:
    '''
    Gets a ground truth dataset from a list of states using the GTProtocol.
    '''
    a = convert_state_list_to_full_gt_experience_list(states, gt)
    b = convert_fed_list_to_experience_list(a, adapter)
    return convert_experience_list_to_dataset(b)
