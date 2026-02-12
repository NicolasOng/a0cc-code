from typing import Any, Optional, Generic

from a0_new.protocols.ground_truth import GTProtocol
from a0_new.protocols.game import T_state, T_action, Player
from a0_new.protocols.model import FullModelOnRaw

import random

import numpy as np

from a0_new.utils.parallel_decorator import parallel

from a0_new.policy import Policy
from a0_new.experience_buffer import ExperienceData
from a0_new.dataset import Dataset

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

class FullExperienceData(Generic[T_state, T_action]):
    def __init__(
            self,
            state: T_state,
            value: float,
            policy: Policy[T_action]
        ) -> None:
        self.state = state
        self.value = value
        self.policy = policy

def get_states_from_fed_list(fed_list: list[FullExperienceData[T_state, T_action]]) -> list[T_state]:
    return [fed.state for fed in fed_list]

def convert_fed_to_experience(full_exp: FullExperienceData[T_state, T_action], adapter: FullModelOnRaw[Any, T_state, T_action]) -> ExperienceData:
    '''
    Converts a FullExperienceData object to an ExperienceData object,
    Using the provided adapter to convert the state and policy to raw formats.
    '''
    return ExperienceData(
        state=adapter.get_raw_state(full_exp.state),
        value=full_exp.value,
        policy=adapter.get_raw_policy(full_exp.state, full_exp.policy),
        mask=adapter.get_legal_actions_mask(full_exp.state, list(full_exp.policy.get_action_list()))
    )

def convert_fed_list_to_experience_list(fed_list: list[FullExperienceData[T_state, T_action]], adapter: FullModelOnRaw[Any, T_state, T_action]) -> list[ExperienceData]:
    '''
    Converts a list of FullExperienceData objects to a list of ExperienceData objects,
    Using the provided adapter to convert the state and policy to raw formats.
    '''
    return [convert_fed_to_experience(fed, adapter) for fed in fed_list]
