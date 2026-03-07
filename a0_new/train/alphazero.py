from __future__ import annotations

from typing import Any, Union, TypeVar, TypeAlias

import os
import time

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from a0_new.protocols.model import RecursiveFullOnRawModel, T_nn_model
from a0_new.protocols.game import A0Game, T_state, T_action
from a0_new.protocols.player import FullModelPlayer

import multiprocessing

#from a0_new.train.self_play.dynamic_batching import self_play
from a0_new.train.self_play.parallel_models import self_play
from a0_new.experience_buffer import ExperienceBuffer, ExperienceData
from a0_new.utils.model import get_full_on_raw_from_player, get_nn_model_from_player, set_raw_model_to_player
from a0_new.train.dataset import train_model_epochs, save_dataset_data, plot_model_performance, stats_from_dataset_data

from config import config

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def train(
        player: FullModelPlayer[RecursiveFullOnRawModel[T_nn_model], Any, Any],
        experience_buffer: ExperienceBuffer,
        iteration: int
    ) -> None:
    train_start = time.time()

    # get the trainable nn model from the player
    model = get_nn_model_from_player(player).get_nn_model()

    # get a dataset from the experience buffer
    dataset = experience_buffer.get_dataset(config.training_batch_size)

    # train the model on the dataset from the experience buffer
    model, train_data = train_model_epochs(
        model=model,
        dataset=dataset,
        num_epochs=1,
        save_type="none",
        plot=False
    )

    # plot, log, and save the model performance metrics in this iteration's training
    i = iteration
    os.makedirs(f"{config.plot_dir}/training_plots", exist_ok=True)
    plot_model_performance(f"training_plots/iteration_{i + 1}", [train_data])
    save_dataset_data(
        f"{config.training_dir}/iteration_stats_{i + 1}.pkl",
        train_data
    )
    total, value_loss, policy_loss, value_accuracy, policy_accuracy = stats_from_dataset_data(train_data)
    logger.log(25, f"Iteration {i + 1} stats: Total Loss: {total:.4f}, Value Loss: {value_loss:.4f}, Policy Loss: {policy_loss:.4f}, Value Accuracy: {value_accuracy:.2%}, Policy Accuracy: {policy_accuracy:.2%}")

    # save the model after each iteration
    if config.training_dir:
        model.save_to_file(config.training_dir + f'model_{i + 1}.pkl')

    train_elapsed = time.time() - train_start
    logger.info(f"Training for iteration {i + 1} completed in {train_elapsed:.1f}s")

def alphazero(
        game: A0Game[Any, Any],
        player: FullModelPlayer[RecursiveFullOnRawModel[T_nn_model], Any, Any],
        starting_iteration: int
    ) -> None:
    logger.info("Starting AlphaZero training")
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
