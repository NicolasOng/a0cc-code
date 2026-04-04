import pickle
import random
from dataclasses import dataclass
from typing import Callable, Generator, Any, Protocol

from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray

from cc.core import Player, Board
from cc.ground_truth import GroundTruth
from a0.game import GameData
from a0.train.dataset import DatasetData, stats_from_dataset_data, plot_model_performance
from a0.eval.plotting import Series, save_series, plot_ridgeline
from a0.eval.dataset_evaluation import policy_accuracy_function, policy_probability_mass_function
from a0.eval.training_data import dataset_data_generator, game_data_generator
from a0.utils.load_training_data import value_diagnostics_generator

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def _log_bucket_distribution(log_fn: Callable[[str], None], label: str, values: list[float], n_buckets: int = 10) -> None:
    '''Log a text histogram of values across n equal-width buckets over [-1, 1].'''
    arr = np.array(values)
    total = len(arr)
    edges = np.linspace(-1, 1, n_buckets + 1)
    counts = np.histogram(arr, bins=edges)[0]
    log_fn(f"{label} ({n_buckets} buckets, {total} samples):")
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        count = int(counts[i])
        bar = "#" * int(40 * count / max(total, 1))
        bracket = "]" if i == n_buckets - 1 else ")"
        log_fn(f"  [{lo:+.2f}, {hi:+.2f}{bracket} {count:6d} ({count/total:.2%}) {bar}")

def get_and_save_value_diagnostics() -> None:
    '''
    For each of pre-balance, post-balance, and model predictions:
      - Write text histograms per iteration to a dedicated text file
      - Save a ridgeline plot summarizing all iterations
    '''
    keys_and_names = [
        ('dataset_values_pre', 'dataset_pre_balance_distributions'),
        ('dataset_values_post', 'dataset_post_balance_distributions'),
        ('model_predictions', 'model_predictions_distributions'),
    ]

    # load all diagnostics first
    all_diags: list[tuple[int, dict[str, list[float]]]] = list(
        value_diagnostics_generator(config.training_dir, config.training_iterations)
    )

    if not all_diags:
        logger.error("No value diagnostics found.")
        return

    for key, name in keys_and_names:
        txt_path = f"{config.eval_dir}{name}.txt"
        with open(txt_path, 'w') as f:
            distributions: list[list[float]] = []
            labels: list[str] = []

            for iteration, diag in all_diags:
                f.write(f"--- Iteration {iteration} ---\n")
                _log_bucket_distribution(lambda line: f.write(line + "\n"), name, diag[key])
                f.write("\n")
                distributions.append(diag[key])
                labels.append(str(iteration))

        logger.info(f"Wrote {name} distributions to {txt_path}")

        # ridgeline plot
        title = name.replace('_', ' ').title()
        plot_ridgeline(
            distributions=distributions,
            labels=labels,
            title=title,
            x_label="Value",
            y_label="Iteration",
            fn=name,
        )
        logger.info(f"Saved ridgeline plot for {name}")

def main():
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="plotting"
    )
    logger.info("plotting...")

    get_and_save_value_diagnostics()

if __name__ == "__main__":
    main()
