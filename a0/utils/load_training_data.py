from __future__ import annotations

import pickle
from tqdm import tqdm
from typing import Generator, Any, TYPE_CHECKING
from collections import defaultdict
import random

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from cc.core import Player, Board
from cc.ground_truth import GroundTruth
from a0.game import GameData

if TYPE_CHECKING:
    from a0.train.dataset import DatasetData

import sys
import os

from jax import numpy as jnp
import jax
import optax
from numpy.typing import NDArray

from a0.model import AlphaZeroModel, load_model
from a0.dataset import Dataset

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def game_data_generator(dir: str, n: int) -> Generator[tuple[int, list[GameData]], None, None]:
    '''
    Load game data from the given path, from 1 to n inclusive
    Returns a list of lists of game data, assuming each is named gamedata_<iteration>.pkl
    Each list corresponds to a single training iteration.
    If a file doesn't exist, it skips it.
    '''
    for i in range(n):
        file_path = f"{dir}/gamedata_{i + 1}.pkl"
        try:
            with open(file_path, 'rb') as file:
                data: list[GameData] = pickle.load(file)
                yield i + 1, data
        except Exception as e:
            logger.error(f"Failed to load game data {i} at {file_path}: {e}")

def dataset_data_generator(dir: str, n: int) -> Generator[tuple[int, DatasetData], None, None]:
    '''
    Load dataset data from the given path.
    Each dataset data object contains the training data for each iteration.
    Attempts to load each iteration's dataset data (iteration_stats_{i + 1}.pkl) from 1 to n inclusive
    Returns a list of dataset data.
    '''
    for i in range(n):
        file_path = f"{dir}/iteration_stats_{i + 1}.pkl"
        try:
            with open(file_path, 'rb') as file:
                data: DatasetData = pickle.load(file)
                yield i + 1, data
        except Exception as e:
            logger.error(f"Failed to load dataset data {i} at {file_path}: {e}")

def models_generator_function(dir: str, n: int) -> Generator[tuple[int, AlphaZeroModel], None, None]:
    '''
    Attempts to load [0 to n] AlphaZeroModels from the given directory.
    If one doesn't exist, it just skips it.
    File names are expected to be in the format "model_{i}.pkl" where i is the model number.
    '''
    for i in range(n + 1):
        model_path = f"{dir}/model_{i}.pkl"
        try:
            model = load_model(model_path)
            yield i, model
        except Exception as e:
            logger.error(f"Failed to load model {i} at {model_path}: {e}")

def load_models(dir: str, n: int) -> list[tuple[int, AlphaZeroModel]]:
    '''
    Returns a list of tuples (model_id, AlphaZeroModel).
    Attempts to load models from 0 to n (inclusive) - skips missing models.
    '''
    logger.info(f"Loading {n} models from {dir}...")
    return list(models_generator_function(dir, n))

def dataset_diagnostics_generator(training_dir: str, n: int):
    '''Load dataset_diagnostics_{i}.pkl for iterations 1..n.'''
    for i in range(n):
        file_path = f"{training_dir}/dataset_diagnostics_{i + 1}.pkl"
        try:
            with open(file_path, 'rb') as f:
                data: dict[str, list[float]] = pickle.load(f)
                yield i + 1, data
        except Exception as e:
            logger.error(f"Failed to load dataset diagnostics {i + 1} at {file_path}: {e}")
