import os

import jax
import jax.numpy as jnp
from flax import nnx

import pickle

from a0.train.generate_datasets import generate_random_gtd
from a0.eval3.generate_datasets import get_nd_and_nt_datasets_from_state_list
from a0.train.dataset import train_model_epochs, plot_model_performance
from a0.model import create_model, save_model
from cc.ground_truth import GroundTruth

from a0.utils.states import (
    get_gtd_from_states, get_rd_from_states, get_random_states, StateInfo, StatesInfo,
    get_state_info_for_states, get_states_info, log_states_info,
    filter_state_list, balance_gt_values, remove_duplicates, sample_states
)

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

from config import config

def train_sl_baseline_player():
    # decide on parameters based on config
    num_epochs = config.training_iterations
    n_random_states = config.training_iterations * config.training_samples

    logger.info(f"Training SL baseline player with {num_epochs} epochs and {n_random_states} random states.")

    # create the model
    seed = int.from_bytes(os.urandom(4))
    model = create_model(
        config.board_size,
        rngs=nnx.Rngs({'params': jax.random.PRNGKey(seed)})
    )

    # generate the dataset
    gt = GroundTruth()
    dataset = generate_random_gtd(gt, n_random_states)

    # generate the eval datasets
    n = 1000
    rs = get_random_states(n * 3, gt)
    rs, _ = remove_duplicates(rs)
    nd, nt = get_nd_and_nt_datasets_from_state_list(rs, gt, n, batch_size=256)

    # train the model for the specified number of epochs
    model, dd = train_model_epochs(model, dataset, num_epochs=num_epochs, plot=False, test_datasets={"nd": nd, "nt": nt})
    plot_model_performance("training", [dd])

    # save the model
    save_model(config.output_dir + "/baseline_sl_player.pkl", model)

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="train_baseline_sl_player"
    )
    logger.info("Generating datasets...")

    train_sl_baseline_player()
