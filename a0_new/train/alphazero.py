from __future__ import annotations

from typing import Any, Union, TypeVar, TypeAlias

import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from a0_new.protocols.model import RecursiveFullOnRawModel, T_nn_model
from a0_new.protocols.game import A0Game, T_state, T_action
from a0_new.protocols.player import FullModelPlayer

import multiprocessing

from a0_new.train.self_play.dynamic_batching import self_play

from a0_new.experience_buffer import ExperienceBuffer, ExperienceData

from config import config

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def train(player: FullModelPlayer[RecursiveFullOnRawModel[T_nn_model], Any, Any], experience_buffer: ExperienceBuffer, iteration: int) -> None:
    pass

def alphazero(
        game: A0Game[Any, Any],
        player: FullModelPlayer[RecursiveFullOnRawModel[T_nn_model], Any, Any],
        starting_iteration: int
    ) -> None:
    # loading/saving the initial model should be done outside this function
    experience_buffer = ExperienceBuffer(config.replay_buffer_size)

    for iteration in range(starting_iteration, config.training_iterations):
        logger.info(f"Starting training iteration {iteration}")

        # self-play to generate + save data
        self_play(game, player, experience_buffer, iteration)

        # train the model on the generated data + save model/data
        train(player, experience_buffer, iteration)
    
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
