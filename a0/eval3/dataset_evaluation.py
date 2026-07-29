import json
import os
import pickle
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from cc.core import Board
from cc.ground_truth import GroundTruth, RankUnrank
from a0.model import AlphaZeroModel
from a0.dataset import Dataset
from a0.eval.dataset_evaluation import (
    evaluate_all_models,
    evaluate_on_all_datasets,
    load_dataset,
    load_dataset_dict,
    load_models,
)
from a0.utils.misc import get_baseline_accuracy
from a0.utils.plotting import DistributionSeries, save_distribution_series, Series, save_series
from a0.utils.safe_load import safe_load_pickle
from a0.utils.value_policy_entropy import build_child_inputs, child_value_entropies, mean_and_ci
from a0.eval3.model_heatmap import run_model_heatmaps

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


def collect_value_policy_entropy(
    models: list[tuple[int, AlphaZeroModel]],
    state_lists_path: str,
    fn: str,
    temperature: float = 1.0,
    batch_size: int = 256,
) -> None:
    '''
    Value-derived policy entropy across models, on the "random" and "seen" board
    lists from state_lists.pkl (actual Boards, needed to enumerate child moves —
    the encoded random.pkl dataset can't provide them). See
    a0.utils.value_policy_entropy for the per-board computation.

    Per model iteration, stores the mean normalized entropy and a 95% CI (across
    boards) for each list as a single Series with "Random"/"Seen" mean lines and
    matching "Random CI"/"Seen CI" half-width lines.
    → {eval_dir}/{fn}.pkl
    '''
    state_lists: Optional[dict[str, list[Board]]] = safe_load_pickle(state_lists_path, "state lists")  # type: ignore[assignment]
    if state_lists is None:
        logger.warning(f"Skipping {fn}: {state_lists_path} not found.")
        return
    if not models:
        logger.warning(f"Skipping {fn}: no models loaded.")
        return

    ranker = RankUnrank()

    # Precompute child inputs once per list (model-independent), so filtering and
    # move enumeration aren't repeated for every model iteration.
    list_specs = [("random", "Random"), ("seen", "Seen")]
    precomputed: dict[str, tuple[NDArray[np.float32], list[int]]] = {}
    for name, _label in list_specs:
        boards = state_lists.get(name)
        if not boards:
            logger.warning(f"{fn}: state list '{name}' missing or empty; its line will be NaN.")
            continue
        child_inputs, counts = build_child_inputs(boards, ranker)
        logger.info(f"{fn}: '{name}' kept {len(counts)}/{len(boards)} boards (non-terminal, >1 move).")
        if counts:
            precomputed[name] = (child_inputs, counts)

    labels = [label for _name, label in list_specs]
    series = Series(labels + [f"{label} CI" for label in labels])

    logger.info(f"Collecting value-derived policy entropy across models ({fn}, T={temperature})...")
    for i, model in tqdm(models):
        series.x.append(i)
        for name, label in list_specs:
            if name not in precomputed:
                series.ys[label].append(float('nan'))
                series.ys[f"{label} CI"].append(float('nan'))
                continue
            child_inputs, counts = precomputed[name]
            entropies = child_value_entropies(model, child_inputs, counts, temperature, batch_size)
            mean, ci = mean_and_ci(entropies)
            series.ys[label].append(mean)
            series.ys[f"{label} CI"].append(ci)

    save_series(series, f"{config.eval_dir}/{fn}.pkl")
    logger.info(f"Saved value-derived policy entropy series to {config.eval_dir}/{fn}.pkl.")


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


def get_and_save_baseline_accuracies() -> None:
    '''
    Loads the source state lists saved by generate_datasets.py and computes
    random-baseline value/policy accuracy for each.
    → {eval_dir}/baseline_accuracies.json
    '''
    state_lists_path = f"{config.dataset_out_dir}/state_lists.pkl"
    if not os.path.exists(state_lists_path):
        logger.warning(f"Skipping baseline accuracies: {state_lists_path} not found.")
        return

    with open(state_lists_path, 'rb') as f:
        state_lists: dict[str, list[Board]] = pickle.load(f)

    gt = GroundTruth()
    results: dict[str, dict[str, float | int]] = {}
    for name, states in state_lists.items():
        if not states:
            logger.warning(f"baseline accuracies: state list '{name}' is empty, skipping.")
            continue
        v_acc, p_acc, v_acc_nd, p_acc_nt = get_baseline_accuracy(states, gt)
        results[name] = {
            "value_accuracy": v_acc,
            "policy_accuracy": p_acc,
            "value_accuracy_nd": v_acc_nd,
            "policy_accuracy_nt": p_acc_nt,
            "n_boards": len(states),
        }
        logger.info(
            f"baseline '{name}' (n={len(states)}): "
            f"v={v_acc:.2%} p={p_acc:.2%} v_nd={v_acc_nd:.2%} p_nt={p_acc_nt:.2%}"
        )

    output_path = f"{config.eval_dir}/baseline_accuracies.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved baseline accuracies for {list(results.keys())} to {output_path}.")


def main():
    setup_logging(level=20, log_dir=config.log_dir, process_name='dataset_evaluation')

    logger.info("Starting dataset evaluation...")

    if config.do_gt_evals:
        get_and_save_baseline_accuracies()
    else:
        logger.info("config.do_gt_evals=False; skipping baseline accuracies.")

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

    # Model-accuracy heatmaps: every checkpoint x every bucket, on both the
    # progress and distance-from-terminal axes. Loads checkpoints lazily, so it
    # doesn't hold the `models` list above on the GPU alongside its own.
    run_model_heatmaps()

    # Value-prediction distributions on random_nd
    collect_distributions_if_present(
        models,
        load_dataset(f"{config.dataset_out_dir}/random_nd.pkl", optional=True),
        "random_nd_value_distributions",
        n,
    )

    # Value-prediction distributions on truly-random sample (no GT required)
    collect_distributions_if_present(
        models,
        load_dataset(f"{config.dataset_out_dir}/random.pkl", optional=True),
        "random_value_distributions",
        n,
    )

    # Value-derived policy entropy on random + seen boards (reads state_lists.pkl
    # for actual Boards so child moves can be enumerated; no GT required).
    collect_value_policy_entropy(
        models,
        f"{config.dataset_out_dir}/state_lists.pkl",
        "value_policy_entropy",
        temperature=1.0,
    )

    logger.info("Dataset evaluation completed.")


if __name__ == "__main__":
    main()
