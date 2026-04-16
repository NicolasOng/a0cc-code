from typing import Optional

import numpy as np
from tqdm import tqdm

from a0.model import AlphaZeroModel
from a0.dataset import Dataset
from a0.eval.dataset_evaluation import (
    evaluate_all_models,
    evaluate_on_all_datasets,
    load_dataset,
    load_dataset_dict,
    load_models,
)
from a0.utils.plotting import DistributionSeries, save_distribution_series

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)


def collect_value_predictions(model: AlphaZeroModel, evaluation_dataset: Dataset) -> list[float]:
    '''
    Runs the model on the dataset and returns one value prediction per sample
    as a flat list of floats. Does not shuffle the dataset.
    '''
    predictions: list[float] = []
    for batch in evaluation_dataset.jnp_batches():
        board_input, _, _, _, _ = batch
        value, _ = model.inference(board_input)
        predictions.extend(np.array(value, dtype=np.float32).flatten().tolist())
    return predictions


def collect_value_distributions_all_models(
    models: list[tuple[int, AlphaZeroModel]],
    evaluation_dataset: Dataset,
    fn: str,
) -> None:
    '''
    For each model iteration, runs the model over the entire dataset and stores
    every value prediction as one distribution in a DistributionSeries. The
    series has a single trial; merging across runs (via merge_distribution_series)
    is the way to add more trials.
    → {eval_dir}/{fn}.pkl
    '''
    logger.info(f"Collecting value-prediction distributions across models ({fn})...")
    series = DistributionSeries(name=fn)
    for i, model in tqdm(models):
        logger.info(f"Collecting value predictions from model {i}")
        predictions = collect_value_predictions(model, evaluation_dataset)
        series.x.append(i)
        series.trials[0].append(predictions)
    save_distribution_series(series, f"{config.eval_dir}/{fn}.pkl")


def evaluate_if_present(
    models: list[tuple[int, AlphaZeroModel]],
    dataset: Optional[Dataset],
    fn: str,
    n: int,
) -> None:
    '''Trim and evaluate, but skip cleanly if the dataset is missing.'''
    if dataset is None:
        logger.warning(f"Skipping {fn}: dataset not loaded.")
        return
    if not models:
        logger.warning(f"Skipping {fn}: no models loaded.")
        return
    dataset.trim(n, shuffle=False)
    evaluate_all_models(models, dataset, fn)


def collect_distributions_if_present(
    models: list[tuple[int, AlphaZeroModel]],
    dataset: Optional[Dataset],
    fn: str,
    n: int,
) -> None:
    '''Trim and collect value distributions, but skip cleanly if the dataset is missing.'''
    if dataset is None:
        logger.warning(f"Skipping {fn}: dataset not loaded.")
        return
    if not models:
        logger.warning(f"Skipping {fn}: no models loaded.")
        return
    dataset.trim(n, shuffle=False)
    collect_value_distributions_all_models(models, dataset, fn)


def evaluate_dataset_dict_if_present(
    models: list[tuple[int, AlphaZeroModel]],
    dataset_dict: Optional[dict[int, Dataset]],
    fn: str,
) -> None:
    '''
    Evaluate the latest model on a bucket → Dataset dict, skipping cleanly
    if the dict file is missing or no models are loaded.
    '''
    if dataset_dict is None:
        logger.warning(f"Skipping {fn}: dataset dict not loaded.")
        return
    if not models:
        logger.warning(f"Skipping {fn}: no models loaded.")
        return
    evaluate_on_all_datasets(models[-1][1], dataset_dict, fn)


def main():
    setup_logging(level=20, log_dir=config.log_dir, process_name='dataset_evaluation')

    logger.info("Starting dataset evaluation...")

    # load the models (already tolerant — skips missing iterations)
    models = load_models(config.training_dir, config.training_iterations)
    if not models:
        logger.warning("No models loaded; dataset evaluation will save empty series only.")

    n = 1000
    num_neighbors = 2

    # Non-draw datasets (seen / random / neighbor_*)
    evaluate_if_present(models, load_dataset(f"{config.dataset_out_dir}/seen_nd.pkl",   optional=True), "seen_nd_eval",   n)
    evaluate_if_present(models, load_dataset(f"{config.dataset_out_dir}/random_nd.pkl", optional=True), "random_nd_eval", n)
    for i in range(num_neighbors):
        evaluate_if_present(
            models,
            load_dataset(f"{config.dataset_out_dir}/neighbor_{i+1}_nd.pkl", optional=True),
            f"neighbor_{i+1}_nd_eval",
            n,
        )

    # Non-trivial datasets (seen / random / neighbor_*)
    evaluate_if_present(models, load_dataset(f"{config.dataset_out_dir}/seen_nt.pkl",   optional=True), "seen_nt_eval",   n)
    evaluate_if_present(models, load_dataset(f"{config.dataset_out_dir}/random_nt.pkl", optional=True), "random_nt_eval", n)
    for i in range(num_neighbors):
        evaluate_if_present(
            models,
            load_dataset(f"{config.dataset_out_dir}/neighbor_{i+1}_nt.pkl", optional=True),
            f"neighbor_{i+1}_nt_eval",
            n,
        )

    # Experienced + alt-target datasets (saved by training_data.py collectors)
    evaluate_if_present(
        models,
        load_dataset(f"{config.dataset_out_dir}/experienced_dataset.pkl", optional=True),
        "experienced_dataset_eval",
        n,
    )
    evaluate_if_present(
        models,
        load_dataset(f"{config.dataset_out_dir}/alt_targets_dataset.pkl", optional=True),
        "alt_targets_dataset_eval",
        n,
    )

    # Game-progress dataset dicts: evaluate the final model on each bucket
    evaluate_dataset_dict_if_present(
        models,
        load_dataset_dict(f"{config.dataset_out_dir}/game_progress_10_nd.pkl", optional=True),
        "game_progress_10_nd_eval",
    )
    evaluate_dataset_dict_if_present(
        models,
        load_dataset_dict(f"{config.dataset_out_dir}/game_progress_10_nt.pkl", optional=True),
        "game_progress_10_nt_eval",
    )

    # Value-prediction distributions on random_nd
    collect_distributions_if_present(
        models,
        load_dataset(f"{config.dataset_out_dir}/random_nd.pkl", optional=True),
        "random_nd_value_distributions",
        n,
    )

    logger.info("Dataset evaluation completed.")


if __name__ == "__main__":
    main()
