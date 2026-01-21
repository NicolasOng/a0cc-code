import os

import jax
from flax import nnx

from config import config

from a0.game import GameData
from a0.model import AlphaZeroModel, load_model, save_model
from a0.train.dataset import train_model_epochs, plot_model_performance, DatasetData, save_dataset_data, stats_from_dataset_data
from a0.eval.training_data import GameDataStats, game_data_list_stats
from a0.experience_buffer import ExperienceBuffer, ExperienceData
from a0.train.alphazero import game_data_to_training_set, game_data_to_gt_training_set
from a0.eval.training_data import game_data_generator

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def psuedo_self_play(game_data_list: list[GameData], use_gt: bool = False) -> tuple[list[ExperienceData], list[GameData]]:
    '''
    Args:
        game_data_list: A list of GameData objects, each representing the data from a single iteration.
    Returns:
        The generated training set, containing board states, values, and policies.
        The game data for each game played, which can be used for analysis or debugging.
    '''
    training_set: list[ExperienceData] = []
    for game_data in game_data_list:
        if use_gt:
            experience_data = game_data_to_gt_training_set(game_data)
        else:
            experience_data = game_data_to_training_set(game_data)
        training_set.extend(experience_data)
    
    logger.info(f"Generated training set of size: {len(training_set)}/{config.training_samples}")
    return training_set, game_data_list

def psuedo_train_alphazero(use_gt: bool = False) -> None:
    os.makedirs(config.plot_dir + "psuedo_training_plots", exist_ok=True)

    game_data_lists = game_data_generator(config.training_dir, config.training_iterations)
    
    logger.info("Starting new training from scratch.")

    model = AlphaZeroModel(
        config.board_size,
        training=config.model_training,
        rngs=nnx.Rngs({'params': jax.random.PRNGKey(0)})
    )
    save_model(config.training_dir + f'/psuedo_model_{0}.pkl', model)
    
    experience_buffer = ExperienceBuffer(
        config.replay_buffer_size
    )

    iterations = config.training_iterations
    train_datas: list[DatasetData] = []
    logger.log(25, GameDataStats.get_header())
    for i, game_data_list in game_data_lists:
        print(f"Iteration {i + 1}/{iterations}")
        logger.info(f"Iteration {i + 1}/{iterations}")

        # generate training data with self-play
        training_set, game_data = psuedo_self_play(game_data_list, use_gt=use_gt)

        # generate and print stats about the game data
        game_data_stats = game_data_list_stats(i, game_data)
        logger.log(25, f"{game_data_stats.get_line()}")
        
        # add the training data to the replay buffer
        for example in training_set:
            experience_buffer.add(example)
        
        # train the model on the experiences in the replay buffer
        model, train_data = train_model_epochs(
            model=model,
            dataset=experience_buffer.get_dataset(config.training_batch_size),
            num_epochs=1,
            save="none",
            plot=False,
            test_datasets={}
        )
        train_datas.append(train_data)

        # plot, log, and save the model performance metrics in this iteration's training
        plot_model_performance(f"psuedo_training_plots/iteration_{i + 1}", [train_data])
        save_dataset_data(
            f"{config.training_dir}/psuedo_iteration_stats_{i + 1}.pkl",
            train_data
        )
        total, value_loss, policy_loss, value_accuracy, policy_accuracy = stats_from_dataset_data(train_data)
        logger.log(25, f"Iteration {i + 1} stats: Total Loss: {total:.4f}, Value Loss: {value_loss:.4f}, Policy Loss: {policy_loss:.4f}, Value Accuracy: {value_accuracy:.2%}, Policy Accuracy: {policy_accuracy:.2%}")

        # save the model after each iteration
        if config.training_dir:
            save_model(config.training_dir + f'psuedo_model_{i + 1}.pkl', model)
    
    # after all iterations, plot all the training data
    plot_model_performance("psuedo_training_plots/full_a0", train_datas)

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="psuedo_training_alphazero"
    )

    logger.info(f"config: {config.path}")

    # Example usage
    psuedo_train_alphazero(use_gt=True)
