import sys
import os

from typing import Generator

from jax import numpy as jnp
import jax
import optax
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from a0.model import AlphaZeroModel, load_model
from a0.dataset import Dataset
from a0.eval.plotting import Series, save_series

from a0.eval.dataset_evaluation import evaluate_all_models, load_models, load_dataset

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)


def main():
    setup_logging(level=20, log_dir=config.log_dir, process_name='dataset_evaluation')

    logger.info("Starting dataset evaluation...")

    # load the models
    models = load_models(config.training_dir, config.training_iterations)

    # Load the datasets
    seen_nd = load_dataset(f"{config.dataset_out_dir}/seen_nd.pkl")
    random_nd = load_dataset(f"{config.dataset_out_dir}/random_nd.pkl")
    neighbor_nd: list[Dataset] = []
    for i in range(2):
        neighbor_dataset = load_dataset(f"{config.dataset_out_dir}/neighbor_{i+1}_nd.pkl")
        neighbor_nd.append(neighbor_dataset)
    
    # load the datasets (non-trivial)
    seen_nt = load_dataset(f"{config.dataset_out_dir}/seen_nt.pkl")
    random_nt = load_dataset(f"{config.dataset_out_dir}/random_nt.pkl")
    neighbor_nt: list[Dataset] = []
    for i in range(2):
        neighbor_dataset = load_dataset(f"{config.dataset_out_dir}/neighbor_{i+1}_nt.pkl")
        neighbor_nt.append(neighbor_dataset)

    # trim down the datasets to a smaller size for faster evaluation
    n = 1000
    seen_nd.trim(n, shuffle=False)
    random_nd.trim(n, shuffle=False)
    for i in range(2):
        neighbor_nd[i].trim(n, shuffle=False)
    
    seen_nt.trim(n, shuffle=False)
    random_nt.trim(n, shuffle=False)
    for i in range(2):
        neighbor_nt[i].trim(n, shuffle=False)
    
    # Evaluate all models
    evaluate_all_models(models, seen_nd, "seen_nd_eval")
    evaluate_all_models(models, random_nd, "random_nd_eval")
    for i in range(2):
        evaluate_all_models(models, neighbor_nd[i], f"neighbor_{i+1}_nd_eval")

    # Evaluate all models
    evaluate_all_models(models, seen_nt, "seen_nt_eval")
    evaluate_all_models(models, random_nt, "random_nt_eval")
    for i in range(2):
        evaluate_all_models(models, neighbor_nt[i], f"neighbor_{i+1}_nt_eval")

    logger.info("Dataset evaluation completed.")


if __name__ == "__main__":
    main()
