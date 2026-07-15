import csv
import os
import pickle
import random
import time

import numpy as np

from a0.dataset import Dataset

from scripts.sl_on_policy_head import check_random_policy_acc

from a0.eval3.generate_datasets import get_nd_and_nt_datasets_from_state_list, get_random_state_list, get_training_dataset
from cc.ground_truth import GroundTruth
from a0.utils.states import get_gtd_from_states

from a0.train.dataset import DatasetData, make_optimizer, train_model_epochs, plot_model_performance
from a0.model import AlphaZeroModel
from flax import nnx
import jax

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def make_biased_dataset(ds: Dataset, bias: float, n: int) -> Dataset:
    '''
    Creates a biased dataset by removing a fraction of the losing (-1) states from the dataset.
    bias represents how the fraction of winning states in the final dataset.
    1.0 = all winning states, 0.0 = all losing states, 0.5 = balanced dataset.
    After doing this, it samples n states from the remaining states to create the final dataset.
    Then shuffles the dataset and returns it.
    Draw states (value == 0) are excluded entirely — bias is defined over wins/losses only.
    '''
    values = np.asarray(ds.values).flatten()
    win_indices = np.where(values > 0)[0]
    loss_indices = np.where(values < 0)[0]

    num_wins = round(n * bias)
    num_losses = n - num_wins
    assert num_wins <= len(win_indices), f"Need {num_wins} winning states but only {len(win_indices)} available."
    assert num_losses <= len(loss_indices), f"Need {num_losses} losing states but only {len(loss_indices)} available."

    chosen = np.concatenate([
        np.random.choice(win_indices, size=num_wins, replace=False),
        np.random.choice(loss_indices, size=num_losses, replace=False),
    ])

    biased = Dataset(ds.batch_size)
    weights = np.asarray(ds.weights)[chosen] if ds.weights is not None else None
    biased.set(
        np.asarray(ds.states)[chosen],
        np.asarray(ds.values)[chosen],
        np.asarray(ds.policies)[chosen],
        np.asarray(ds.masks)[chosen],
        weights
    )
    biased.shuffle()
    return biased

def train_model(seed: int, dataset: Dataset, num_epochs: int = 10) -> tuple[AlphaZeroModel, DatasetData]:
    model = AlphaZeroModel(
        board_size=config.board_size,
        num_filters=256,
        rngs=nnx.Rngs({'params': jax.random.PRNGKey(seed)}),
        num_resblocks=3
    )
    trained_model, dataset_data = train_model_epochs(
        model=model,
        dataset=dataset,
        num_epochs=num_epochs,
        save="None",
        plot=False,
        test_datasets={},
        optimizer=make_optimizer(model)
    )
    return trained_model, dataset_data

def get_model_value_head_accuracy_and_distribution(model: AlphaZeroModel, dataset: Dataset) -> tuple[float, float, float]:
    '''
    Evaluates the model's value head accuracy on the dataset and returns the accuracy of the value head predictions
    and the mean/std distribution of predicted values (fraction of positive predictions).
    '''
    pred_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    for board_input, value_label, _, _, _ in dataset.jnp_batches():
        value, _ = model.inference(board_input)
        pred_batches.append(np.array(value, dtype=np.float32).flatten())
        label_batches.append(np.array(value_label, dtype=np.float32).flatten())
    predictions = np.concatenate(pred_batches)
    labels = np.concatenate(label_batches)

    # same sign convention as a0.eval.dataset_evaluation.value_accuracy_function
    pred_classification = np.where(predictions <= 0, -1, 1)
    label_classification = np.where(labels <= 0, -1, 1)
    accuracy = float(np.mean(pred_classification == label_classification))

    return accuracy, float(np.mean(predictions)), float(np.std(predictions))

def main():
    num_seeds = 30 # 30
    train_size = 50000 # 50000
    full_train_size = int(train_size * 2.3)

    gt = GroundTruth()
    full_train, _ = get_training_dataset(full_train_size, gt)
    eval_nd, _ = get_nd_and_nt_datasets_from_state_list(get_random_state_list(5000, gt), gt, 5000, 256)

    csv_path = f"{config.stats_dir}/2026-07-15_model_bias_test.csv"

    seeds = [random.randint(0, 1000000) for _ in range(num_seeds)]
    for seed in seeds:
        for b in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            biased_dataset = make_biased_dataset(full_train, b, train_size)
            trained_model, _ = train_model(seed, biased_dataset)
            acc, mean, std = get_model_value_head_accuracy_and_distribution(trained_model, eval_nd)
            logger.info(f"Seed: {seed}, Bias: {b}, Value Head Accuracy: {acc:.4f}, Mean: {mean:.4f}, Std: {std:.4f}")
            write_header = not os.path.exists(csv_path)
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["seed", "bias", "value_accuracy", "pred_mean", "pred_std"])
                writer.writerow([seed, b, f"{acc:.6f}", f"{mean:.6f}", f"{std:.6f}"])


if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="model_bias_test"
    )

    main()
