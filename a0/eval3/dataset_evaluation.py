from typing import Optional

from a0.model import AlphaZeroModel
from a0.dataset import Dataset
from a0.eval.dataset_evaluation import evaluate_all_models, load_dataset, load_models

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)


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


def main():
    setup_logging(level=20, log_dir=config.log_dir, process_name='dataset_evaluation')

    logger.info("Starting dataset evaluation...")

    # load the models (already tolerant — skips missing iterations)
    models = load_models(config.training_dir, config.training_iterations)
    if not models:
        logger.warning("No models loaded; dataset evaluation will save empty series only.")

    n = 1000
    num_neighbors = 2

    # Non-draw datasets
    evaluate_if_present(models, load_dataset(f"{config.dataset_out_dir}/seen_nd.pkl",   optional=True), "seen_nd_eval",   n)
    evaluate_if_present(models, load_dataset(f"{config.dataset_out_dir}/random_nd.pkl", optional=True), "random_nd_eval", n)
    for i in range(num_neighbors):
        evaluate_if_present(
            models,
            load_dataset(f"{config.dataset_out_dir}/neighbor_{i+1}_nd.pkl", optional=True),
            f"neighbor_{i+1}_nd_eval",
            n,
        )

    # Non-trivial datasets
    evaluate_if_present(models, load_dataset(f"{config.dataset_out_dir}/seen_nt.pkl",   optional=True), "seen_nt_eval",   n)
    evaluate_if_present(models, load_dataset(f"{config.dataset_out_dir}/random_nt.pkl", optional=True), "random_nt_eval", n)
    for i in range(num_neighbors):
        evaluate_if_present(
            models,
            load_dataset(f"{config.dataset_out_dir}/neighbor_{i+1}_nt.pkl", optional=True),
            f"neighbor_{i+1}_nt_eval",
            n,
        )

    logger.info("Dataset evaluation completed.")


if __name__ == "__main__":
    main()
