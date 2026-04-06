from __future__ import annotations

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
from a0.utils.safe_load import safe_load_pickle

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def game_data_generator(dir: str, n: int) -> Generator[tuple[int, list[GameData]], None, None]:
    '''
    Yields (iteration, game data list) for each gamedata_<iteration>.pkl in 1..n
    that exists. Missing files are skipped (with a warning).
    '''
    for i in range(n):
        file_path = f"{dir}/gamedata_{i + 1}.pkl"
        data: list[GameData] | None = safe_load_pickle(file_path, f"game data {i + 1}")  # type: ignore[assignment]
        if data is not None:
            yield i + 1, data

def dataset_data_generator(dir: str, n: int) -> Generator[tuple[int, DatasetData], None, None]:
    '''
    Yields (iteration, DatasetData) for each iteration_stats_<iteration>.pkl in 1..n
    that exists. Missing files are skipped (with a warning).
    '''
    for i in range(n):
        file_path = f"{dir}/iteration_stats_{i + 1}.pkl"
        data: DatasetData | None = safe_load_pickle(file_path, f"iteration stats {i + 1}")  # type: ignore[assignment]
        if data is not None:
            yield i + 1, data

def models_generator_function(dir: str, n: int) -> Generator[tuple[int, AlphaZeroModel], None, None]:
    '''
    Yields (i, AlphaZeroModel) for model_<i>.pkl in 0..n that exists.
    Models go through a custom load path (load_model), so safe_load_pickle isn't
    used directly; we just check the file exists and let load_model handle the
    flax restore + log any deserialization errors.
    '''
    for i in range(n + 1):
        model_path = f"{dir}/model_{i}.pkl"
        if not os.path.exists(model_path):
            logger.warning(f"model {i} not found at {model_path}; skipping.")
            continue
        try:
            model = load_model(model_path)
            yield i, model
        except Exception as e:
            logger.error(f"Failed to load model {i} at {model_path}: {e}; skipping.")

def load_models(dir: str, n: int) -> list[tuple[int, AlphaZeroModel]]:
    '''
    Returns a list of tuples (model_id, AlphaZeroModel).
    Attempts to load models from 0 to n (inclusive) - skips missing models.
    '''
    logger.info(f"Loading {n} models from {dir}...")
    return list(models_generator_function(dir, n))

def dataset_diagnostics_generator(training_dir: str, n: int) -> Generator[tuple[int, dict[str, list[float]]], None, None]:
    '''
    Yields (iteration, diagnostics dict) for each dataset_diagnostics_<iteration>.pkl
    in 1..n that exists. Missing files are skipped.
    '''
    for i in range(n):
        file_path = f"{training_dir}/dataset_diagnostics_{i + 1}.pkl"
        data: dict[str, list[float]] | None = safe_load_pickle(file_path, f"dataset diagnostics {i + 1}")  # type: ignore[assignment]
        if data is not None:
            yield i + 1, data
