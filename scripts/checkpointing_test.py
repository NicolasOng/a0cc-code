import random
import multiprocessing
import concurrent.futures
from concurrent.futures import Future, wait, FIRST_COMPLETED
import os

from cc.core import Board, Game, Player
from cc.ground_truth import GroundTruth
from a0.dataset import Dataset
from a0.train.dataset import DatasetData
from a0.model_utils import board_to_input, Policy, get_legal_move_mask_from_state
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

import pickle

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def get_new_model() -> AlphaZeroModel:
    model = AlphaZeroModel(
        board_size=config.board_size,
        num_filters=256,
        training=True,
        rngs=nnx.Rngs({'params': jax.random.PRNGKey(1)}),
        num_resblocks=3
    )
    return model

def train_on_datasets(model: AlphaZeroModel, dataset: Dataset, eval_datasets: dict[str, Dataset], num_epochs: int = 1) -> tuple[AlphaZeroModel, DatasetData]:
    '''
    Trains the given model on the given dataset for num_epochs epochs,
    then evaluates it on the eval_dataset.
    Plots the training performance.
    '''
    logger.info(f"Training given model for {num_epochs} epochs...")
    trained_model, dsd = train_model_epochs(model, dataset, num_epochs, save="None", plot=False, test_datasets=eval_datasets)
    return trained_model, dsd

def train_with_checkpointing(epochs_per_checkpoint: int = 1, states_per_epoch: int = 1000) -> None:
    # load the model and training data from disk (if they exist)
    if os.path.exists(config.training_dir + "model_checkpoint.pkl"):
        logger.info("Loading model from checkpoint...")
        trained_model = load_model(config.training_dir + "model_checkpoint.pkl", training=True)
    else:
        logger.info("Creating new model...")
        trained_model = get_new_model()
    if os.path.exists(config.training_dir + "datasetdata.pkl"):
        logger.info("Loading dataset data from checkpoint...")
        with open(config.training_dir + "datasetdata.pkl", "rb") as f:
            datasetdata: DatasetData = pickle.load(f)
    else:
        logger.info("Creating new dataset data...")
        datasetdata = DatasetData()

    for e in range(epochs_per_checkpoint):
        logger.info(f"Starting checkpoint epoch {e + 1} of {epochs_per_checkpoint}...")
        # generate a new dataset to train on
        # temp:
        logger.info("Generating new training dataset...")
        dataset = create_gtd_from_states(get_n_random_states(states_per_epoch))

        # train on that dataset
        logger.info("Training on dataset...")
        trained_model, dsd = train_on_datasets(trained_model, dataset, {"eval": dataset}, num_epochs=1)

        # add the training data to the checkpointed
        datasetdata.epoch_data.extend(dsd.epoch_data)

    # plot the model performance so far
    logger.info("Plotting model performance...")
    plot_model_performance("checkpointed_model", [datasetdata])
    # save the checkpointed data and model to disk
    logger.info("Saving checkpointed model and dataset data...")
    with open(config.training_dir + "datasetdata.pkl", "wb") as f:
        pickle.dump(datasetdata, f)
    
    save_model(config.training_dir + "model_checkpoint.pkl", trained_model)

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="checkpointing_test"
    )

    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    train_with_checkpointing(epochs_per_checkpoint=1, states_per_epoch=10000)
