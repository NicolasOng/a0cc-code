'''
Model-accuracy heatmaps: checkpoint (x) x game progress or distance-from-terminal
(y), cell = how good the model's value/policy head is on that bucket's boards.

The counterpart to the training-data heatmaps in
a0.eval3.collectors.accuracy_heatmap, and deliberately emitted in the SAME
flattened Series layout ("<metric> P<bucket>" / "<metric> D<distance>", x =
iteration), so plot_heatmap, merge_series and the sweep-combine path are shared.

INTERPRETATION — the two heatmaps are NOT cell-to-cell comparable. The buckets
here are a FIXED board set, pooled across all iterations when the datasets were
built, so variation along x is purely model-driven. The training-data heatmap's
board distribution shifts every iteration, so it mixes "targets got better" with
"the states being visited changed".

Batching note: Dataset.jnp_batches drops the final partial batch, so evaluating
a 1000-board bucket at batch 256 would silently score 768 boards and a bucket
smaller than the batch would score none. That is why the existing
evaluate_on_all_datasets sets batch_size = 1 — correct, but far too slow across
hundreds of checkpoints. evaluate_model_padded instead pads the last chunk to a
fixed width and slices the results back (same trick as
TrajectoryReplayBuffer.refresh_and_build_dataset, which also keeps XLA to a
single JIT trace), then weights each chunk by its REAL sample count so the
result equals the batch-size-1 sample mean exactly.
'''
import gc
import os

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from a0.dataset import Dataset
from a0.model import AlphaZeroModel
from a0.utils.load_training_data import models_generator_function
from a0.utils.plotting import Series, save_series
from a0.eval.dataset_evaluation import (
    load_dataset_dict,
    policy_accuracy_batch,
    policy_entropy_batch,
    policy_loss_function,
    value_accuracy_function,
    value_loss_function,
)

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

# masked-out logits for the policy metrics, matching evaluate_model
MASK_VALUE = -1e9

def evaluate_model_padded(
    model: AlphaZeroModel,
    dataset: Dataset,
    batch_size: int = 256,
) -> dict[str, float]:
    '''
    Sample-weighted metrics over the WHOLE dataset, with the final chunk padded
    to `batch_size` and sliced back. Returns the same quantities as
    a0.eval.dataset_evaluation.evaluate_model, but as a dict and without
    shuffling (the caller reuses one dataset across many checkpoints, so
    mutating its order would be a surprise).
    '''
    n = len(dataset)
    if n == 0:
        return {}

    states = np.asarray(dataset.states, dtype=np.float32)
    values = np.asarray(dataset.values, dtype=np.float32)
    policies = np.asarray(dataset.policies, dtype=np.float32)
    masks = np.asarray(dataset.masks, dtype=np.float32)

    totals = {
        "loss": 0.0, "value_loss": 0.0, "policy_loss": 0.0,
        "value_accuracy": 0.0, "policy_accuracy": 0.0, "policy_entropy": 0.0,
    }

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        real = stop - start
        chunk = states[start:stop]
        if real < batch_size:
            # pad to a fixed shape so XLA keeps one trace, then drop the padding
            padding = np.zeros((batch_size - real, *chunk.shape[1:]), dtype=chunk.dtype)
            chunk = np.concatenate([chunk, padding], axis=0)

        pred_value, pred_policy = model.inference(chunk)
        pred_value = np.asarray(pred_value, dtype=np.float32)[:real]
        pred_policy = np.asarray(pred_policy, dtype=np.float32)[:real]

        value_label = values[start:stop]
        policy_label = policies[start:stop]
        policy_mask = masks[start:stop].astype(bool)

        masked_pred = np.where(policy_mask, pred_policy, MASK_VALUE)
        masked_label = np.where(policy_mask, policy_label, 0.0)

        value_loss = value_loss_function(pred_value, value_label)
        policy_loss = policy_loss_function(masked_pred, masked_label)
        batch_metrics = {
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "loss": value_loss + policy_loss,
            "value_accuracy": value_accuracy_function(pred_value, value_label),
            "policy_accuracy": policy_accuracy_batch(masked_pred, masked_label),
            "policy_entropy": policy_entropy_batch(pred_policy, policy_mask),
        }
        # weight by REAL samples: every per-batch metric above is already a mean
        # over the chunk, so this recovers the exact sample mean
        for key, value in batch_metrics.items():
            totals[key] += value * real

    return {key: total / n for key, total in totals.items()}

def evaluate_checkpoints_over_buckets(
    dataset_dict: dict[int, Dataset],
    label: str,
    metrics: list[str],
    fn: str,
    batch_size: int = 256,
    stride: int | None = None,
) -> None:
    '''
    Evaluates every checkpoint on every bucket and saves one flattened heatmap
    Series (x = iteration, keys "<metric> <label><bucket>").

    Checkpoints are loaded ONE AT A TIME and released before the next: the whole
    point of models_generator_function here is that materialising 200+ models
    exhausted GPU memory in the player evals (fixed in 82f8ea7), and this loop
    would hit exactly the same wall.

    `stride` reads config.heatmap_eval_stride if present, defaulting to 1 —
    the eval is cheap enough not to need it, but adding the key later enables it
    without a code change.
    '''
    if not dataset_dict:
        logger.warning(f"{fn}: empty dataset dict; skipping.")
        return

    if stride is None:
        stride = max(1, int(getattr(config, "heatmap_eval_stride", 1)))

    buckets = sorted(dataset_dict.keys())
    keys = [f"{metric} {label}{b}" for metric in metrics for b in buckets]
    keys += [f"Count {label}{b}" for b in buckets]
    series = Series(keys)

    sizes = {b: len(dataset_dict[b]) for b in buckets}
    logger.info(
        f"{fn}: {len(buckets)} buckets (sizes {[sizes[b] for b in buckets]}), "
        f"batch_size={batch_size}, stride={stride}"
    )

    evaluated = 0
    for iteration, model in tqdm(models_generator_function(config.training_dir, config.training_iterations),
                                 desc=f"{fn} checkpoints"):
        if iteration % stride != 0 and iteration != config.training_iterations:
            continue
        series.x.append(iteration)
        for b in buckets:
            results = evaluate_model_padded(model, dataset_dict[b], batch_size)
            for metric in metrics:
                key = _metric_key(metric)
                series.ys[f"{metric} {label}{b}"].append(results.get(key, float("nan")))
            series.ys[f"Count {label}{b}"].append(sizes[b])
        evaluated += 1
        # release this checkpoint's GPU buffers before loading the next
        del model
        gc.collect()

    save_series(series, f"{config.eval_dir}/{fn}.pkl")
    logger.info(f"{fn}: saved {fn}.pkl ({evaluated} checkpoints x {len(buckets)} buckets).")

def _metric_key(metric: str) -> str:
    '''Series label -> evaluate_model_padded dict key.'''
    return {
        "Value Accuracy": "value_accuracy",
        "Policy Accuracy": "policy_accuracy",
        "Value Loss": "value_loss",
        "Policy Loss": "policy_loss",
        "Policy Entropy": "policy_entropy",
        "Loss": "loss",
    }[metric]
