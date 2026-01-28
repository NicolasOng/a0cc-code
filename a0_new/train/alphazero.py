from __future__ import annotations

from typing import Optional, Any, Protocol, TypeVar
import multiprocessing
import concurrent.futures
from concurrent.futures import Future, wait, FIRST_COMPLETED
import os
import pickle
import dill

from a0_new.protocols import A0Game, A0Player, A0Model, A0State, A0Action

import numpy as np
from numpy.typing import NDArray

from config import config

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def alphazero(
        game: A0Game[A0State[A0Action], A0Action],
        player: A0Player[A0Model, A0State[A0Action], A0Action],
        starting_iteration: int
    ) -> None:
    # loading/saving the initial model should be done outside this function

    pass

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="training_alphazero_new"
    )

    logger.info(f"config: {config.path}")
    logger.info(f"output_dir: {config.output_dir}")

    # Set the multiprocessing start method to 'spawn' for compatibility with JAX
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
