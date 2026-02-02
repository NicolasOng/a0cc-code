from __future__ import annotations

from typing import Any

import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from a0_new.protocols.model import A0Model
from a0_new.protocols.game import A0Game
from a0_new.protocols.player import A0Player

import multiprocessing

from a0_new.train.self_play.dynamic_batching import self_play

from config import config

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def train(player: A0Player[A0Model[Any, Any, Any], Any, Any], iteration: int) -> None:
    pass

def save_iteration_data(player: A0Player[A0Model[Any, Any, Any], Any, Any], iteration: int) -> None:
    pass

def alphazero(
        game: A0Game[Any, Any],
        player: A0Player[Any, Any, Any],
        starting_iteration: int
    ) -> None:
    # loading/saving the initial model should be done outside this function

    for iteration in range(starting_iteration, config.training_iterations):
        logger.info(f"Starting training iteration {iteration}")

        # self-play to generate training data
        self_play(game, player, iteration)

        # train the model on the generated data
        train(player, iteration)

        # save the model + other data
        save_iteration_data(player, iteration)


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
