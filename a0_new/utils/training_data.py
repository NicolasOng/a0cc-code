from typing import Any, Iterator, Optional, Generic, Generator

from a0_new.protocols.ground_truth import GTProtocol
from a0_new.protocols.game import T_state, T_action, Player
from a0_new.protocols.model import FullModelOnRaw, TrainableModel, T_nnx

import random
import numpy as np
import pickle

from a0_new.utils.parallel_decorator import parallel

from a0_new.policy import Policy
from a0_new.experience_buffer import ExperienceData
from a0_new.dataset import Dataset
from a0_new.play import GameData
from a0_new.train.dataset import DatasetData
from a0_new.utils.full_experience_data import FullExperienceData

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def gamedata_generator(dir: str, n: int) -> Generator[tuple[int, list[GameData[Any, Any]]], None, None]:
    '''
    Loads gamedata from the given path, from 1 to n inclusive
    Returns a generator of tuples (iteration, gamedata),
    where gamedata is a list of GameData objects generated during that iteration's self play.
    If a file doesn't exist, it skips it.
    '''
    for i in range(n):
        file_path = f"{dir}/gamedata_{i + 1}.pkl"
        try:
            with open(file_path, 'rb') as file:
                data: list[GameData[Any, Any]] = pickle.load(file)
                yield i + 1, data
        except Exception as e:
            logger.error(f"Failed to load gamedata {i + 1} at {file_path}: {e}")

def datasetdata_generator(dir: str, n: int) -> Generator[tuple[int, DatasetData], None, None]:
    '''
    Load dataset data from the given path.
    Each dataset data object contains the training data for each iteration.
    Attempts to load each iteration's dataset data (iteration_stats_{i + 1}.pkl) from 1 to n inclusive
    Returns a generator of tuples (iteration, dataset_data),
    where dataset_data is a DatasetData object containing the training data for that iteration.
    If a file doesn't exist, it skips it.
    '''
    for i in range(n):
        file_path = f"{dir}/iteration_stats_{i + 1}.pkl"
        try:
            with open(file_path, 'rb') as file:
                data: DatasetData = pickle.load(file)
                yield i + 1, data
        except Exception as e:
            logger.error(f"Failed to load dataset data {i + 1} at {file_path}: {e}")

def model_generator(model_cls: TrainableModel, dir: str, n: int) -> Generator[tuple[int, TrainableModel], None, None]:
    '''
    Load model data from the given path.
    Each model data object contains the model for each iteration.
    Attempts to load each iteration's model (model_{i + 1}.pkl) from 1 to n inclusive
    Returns a generator of tuples (iteration, model),
    where model is a TrainableModel object for that iteration.
    If a file doesn't exist, it skips it.
    '''
    for i in range(n):
        file_path = f"{dir}/model_{i + 1}.pkl"
        try:
            # Use the class-specific loader
            model = model_cls.load_from_file(file_path)
            yield i + 1, model
        except Exception as e:
            logger.error(f"Failed to load model {i + 1} at {file_path}: {e}")

def get_fed_list_from_gamedata(gamedata: GameData[T_state, T_action]) -> list[FullExperienceData[T_state, T_action]]:
    '''
    Converts a list of GameData objects into a list of FullExperienceData objects.
    '''
    fed_list: list[FullExperienceData[T_state, T_action]] = []
    game_winner = gamedata.winner
    turn_data = gamedata.turn_data
    for turn in turn_data:
        state = turn.state
        # experienced outcome = current player perspective
        outcome = 0.0 if game_winner is None else 1.0 if game_winner == state.get_current_player() else -1.0
        policy = turn.policy
        full_experience = FullExperienceData(
            state=state,
            value=outcome,
            policy=policy
        )
        fed_list.append(full_experience)
    return fed_list

def get_fed_list_from_gamedata_list(gamedata_list: list[GameData[T_state, T_action]]) -> list[FullExperienceData[T_state, T_action]]:
    '''
    Converts a list of GameData objects into a list of FullExperienceData objects.
    '''
    fed_list: list[FullExperienceData[T_state, T_action]] = []
    sub_lists = [get_fed_list_from_gamedata(gamedata) for gamedata in gamedata_list]
    for sublist in sub_lists:
        fed_list.extend(sublist)
    return fed_list

def get_full_experience_data_list_from_training_data(gamedata_list_iterator: Iterator[tuple[int, list[GameData[T_state, T_action]]]]) -> list[FullExperienceData[T_state, T_action]]:
    '''
    Converts a list of GameData objects into a single ExperienceData object.
    '''
    full_experience_data_list: list[FullExperienceData[T_state, T_action]] = []
    sub_lists = [get_fed_list_from_gamedata_list(gamedata_list) for _, gamedata_list in gamedata_list_iterator]
    for sublist in sub_lists:
        full_experience_data_list.extend(sublist)
    return full_experience_data_list
