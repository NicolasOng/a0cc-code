import random

from cc.core import Board, Game, Player
from cc.ground_truth import GroundTruth
from a0.dataset import Dataset
from a0.model_utils import board_to_input, Policy
from a0.eval.training_data import game_data_generator
from a0.train.dataset import train_model_epochs, plot_model_performance
from a0.model import AlphaZeroModel, load_model, save_model
from a0.eval.dataset_evaluation import evaluate_model, policy_accuracy_function
from a0.graph_search.mcts import MCTS
from a0.mcts.gt import MCTS_GT
from a0.players.a0 import A0Player
from scripts.sl_on_policy_head import get_n_random_states, create_gtd_from_states

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from flax import nnx
import jax

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def train_model(fn: str, dataset: Dataset, eval_datasets: dict[str, Dataset], final_dataset: Dataset, num_epochs: int = 1) -> tuple[AlphaZeroModel, float, float]:
    '''
    Trains a model on the given dataset for num_epochs epochs,
    then evaluates it on the eval_dataset.
    Plots the training performance.
    Returns both the trained model and its value and policy accuracy on the final dataset.
    '''
    logger.info(f"Training model '{fn}' for {num_epochs} epochs...")

    model = AlphaZeroModel(
        board_size=config.board_size,
        num_filters=256,
        training=True,
        rngs=nnx.Rngs({'params': jax.random.PRNGKey(1)})
    )

    trained_model, dsd = train_model_epochs(model, dataset, num_epochs, save="None", plot=False, test_datasets=eval_datasets)
    plot_model_performance(fn, [dsd])

    loss, value_loss, policy_loss, value_accuracy, policy_accuracy = evaluate_model(trained_model, final_dataset)
    logger.info(f"For model '{fn}':")
    logger.info(f"Evaluation on final dataset - Loss: {loss:.2%}, Value Loss: {value_loss:.2%}, Policy Loss: {policy_loss:.2%}, Value Accuracy: {value_accuracy:.2%}, Policy Accuracy: {policy_accuracy:.2%}")


    return trained_model, value_accuracy, policy_accuracy

def train_model_to_value_acc(fn: str, target_acc: float, et: float, max_attempts: int) -> AlphaZeroModel:
    '''
    Trains a model on the given dataset until it reaches the target value accuracy,
    evaluating it on the eval_dataset after each training session.
    Plots the training performance.
    '''
    train_n_max = 1048576  # 2^20
    train_n_min = 256  # 2^8
    train_n = 524416
    test_n = 10000
    validation_n = 100000
    logger.info(f"Training model '{fn}' to reach value accuracy of {target_acc:.2%} +/- {et:.2%}...")
    training_dataset = create_gtd_from_states(get_n_random_states(train_n, remove_trivial=False, remove_terminal=True))
    test_dataset = create_gtd_from_states(get_n_random_states(test_n, remove_trivial=False, remove_terminal=True))
    validation_dataset = create_gtd_from_states(get_n_random_states(validation_n, remove_trivial=False, remove_terminal=True))
    achieved_value_acc = 0.0
    achieved_policy_acc = 0.0
    trained_model = None
    n_attempts = 0

    while train_n_min <= train_n_max and n_attempts < max_attempts:
        trained_model, achieved_value_acc, achieved_policy_acc = train_model(
            fn,
            training_dataset,
            {
                "test": test_dataset,
            },
            validation_dataset,
        )

        if achieved_value_acc > target_acc + et:
            logger.info(f"Value accuracy overshot the target: {achieved_value_acc:.2%} > {target_acc + et:.2%}. Reducing amount of states to train on.")
            train_n_max = train_n - 1
        elif achieved_value_acc < target_acc - et:
            logger.info(f"Value accuracy below target: {achieved_value_acc:.2%} < {target_acc - et:.2%}. Increasing amount of states to train on.")
            train_n_min = train_n + 1
        else:
            logger.info(f"Value accuracy within target range: {achieved_value_acc:.2%} ~= {target_acc:.2%}.")
            break
        
        train_n = (train_n_min + train_n_max) // 2
        logger.info(f"Next training will use {train_n} states.")
        training_dataset = create_gtd_from_states(get_n_random_states(train_n, remove_trivial=False, remove_terminal=True))
        n_attempts += 1

    logger.info(f"Final model '{fn}' achieved value accuracy: {achieved_value_acc:.2%}, policy accuracy: {achieved_policy_acc:.2%}.")

    return trained_model

def main():
    acc = 0.80
    trained_model = train_model_to_value_acc(
        fn="a0_model_to_value_acc",
        target_acc=acc,
        et=0.02,
        max_attempts=100,
    )
    save_model(config.training_dir + f"/model_value_acc_{acc:.2f}.pkl", trained_model)

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="train_model_to_acc",
    )

    main()
