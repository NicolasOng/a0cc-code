'''
Evaluates a trained model on random balanced states to detect bias
in the value head predictions.
Usage: python -m scripts.2026-03-19_model_bias config_file.json trial_num
'''
import numpy as np
import jax.numpy as jnp

from config import config
from cc.core import Game, Player
from cc.ground_truth import GroundTruth
from a0.model import AlphaZeroModel, load_model
from a0.model_utils import board_to_input
from a0.train.alphazero import get_most_recent_model_path

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def get_balanced_random_states(n_per_class: int):
    '''
    Returns n_per_class boards for each GT class (win, loss). Draws are excluded.
    '''
    gt = GroundTruth()
    game = Game(board_size=config.board_size, num_pieces=config.num_pieces)
    max_rank = gt.get_max_rank()

    buckets = {1.0: [], -1.0: []}
    target = n_per_class
    seen = set()

    while any(len(v) < target for v in buckets.values()):
        rank = np.random.randint(0, max_rank)
        if rank in seen:
            continue
        seen.add(rank)
        # if len(seen) % 100 == 0:
        #     logger.info(f"Checked {len(seen)} random states, collected {len(buckets[1.0])} wins and {len(buckets[-1.0])} losses")
        board = gt.unrank(rank)
        outcome = gt.get_outcome(board)
        if outcome == 0.0:
            continue
        if len(buckets[outcome]) < target:
            buckets[outcome].append(board)

    return buckets


def prepare_eval_data(buckets: dict):
    '''Pre-computes the numpy arrays from the buckets so we don't redo it per model.'''
    all_boards = []
    gt_values = []
    for cls in [1.0, -1.0]:
        for board in buckets[cls]:
            all_boards.append(board_to_input(board))
            gt_values.append(cls)
    states = np.concatenate(all_boards, axis=0)  # (N, BS, BS, 2)
    gt_values = np.array(gt_values)  # (N,)
    return states, gt_values


def evaluate_model(model: AlphaZeroModel, states: np.ndarray, gt_values: np.ndarray) -> dict:
    '''Evaluates a model and returns a dict of metrics.'''
    n = len(gt_values)

    # run inference in batches
    pred_values = []
    batch_size = 256
    for i in range(0, n, batch_size):
        batch = jnp.array(states[i:i+batch_size])
        values, _ = model.inference(batch)
        pred_values.append(np.array(values).flatten())
    pred_values = np.concatenate(pred_values)  # (N,)

    # classify predictions by sign
    pred_classes = np.where(pred_values > 0, 1.0, -1.0)
    gt_classes = gt_values

    accuracy = float(np.mean(pred_classes == gt_classes))
    pred_win_pct = float(np.mean(pred_classes == 1.0))
    mean_pred = float(np.mean(pred_values))
    mean_pred_win = float(np.mean(pred_values[gt_classes == 1.0]))
    mean_pred_loss = float(np.mean(pred_values[gt_classes == -1.0]))
    win_recall = float(np.mean(pred_classes[gt_classes == 1.0] == 1.0))
    loss_recall = float(np.mean(pred_classes[gt_classes == -1.0] == -1.0))

    return {
        'accuracy': accuracy,
        'pred_win_pct': pred_win_pct,
        'mean_pred': mean_pred,
        'mean_pred_win': mean_pred_win,
        'mean_pred_loss': mean_pred_loss,
        'win_recall': win_recall,
        'loss_recall': loss_recall,
        'pred_values': pred_values,
    }


def print_detailed_report(iteration: int, metrics: dict, gt_values: np.ndarray):
    '''Prints the full report for a single model.'''
    n = len(gt_values)
    pred_values = metrics['pred_values']
    pred_classes = np.where(pred_values > 0, 1.0, -1.0)
    gt_classes = gt_values
    class_labels = [1.0, -1.0]
    class_names = {1.0: "Win ", -1.0: "Loss"}

    print(f"\n{'='*60}")
    print(f"Model {iteration} — Bias Evaluation (n={n})")
    print(f"{'='*60}")

    print(f"\nGT distribution:")
    for cls in class_labels:
        count = np.sum(gt_classes == cls)
        print(f"  {class_names[cls]}: {count:6d} ({count/n:.1%})")

    print(f"\nPrediction distribution:")
    for cls in class_labels:
        count = np.sum(pred_classes == cls)
        print(f"  {class_names[cls]}: {count:6d} ({count/n:.1%})")

    print(f"\nConfusion matrix (rows=GT, cols=Predicted):")
    print(f"{'':>8} {'Win':>8} {'Loss':>8} {'Recall':>8}")
    for gt_cls in class_labels:
        row = []
        for pred_cls in class_labels:
            count = np.sum((gt_classes == gt_cls) & (pred_classes == pred_cls))
            row.append(count)
        total = sum(row)
        recall = row[class_labels.index(gt_cls)] / total if total > 0 else 0
        print(f"  {class_names[gt_cls]:>5} {row[0]:>8d} {row[1]:>8d} {recall:>8.1%}")

    print(f"{'Prec':>8}", end="")
    for pred_cls in class_labels:
        pred_mask = pred_classes == pred_cls
        tp = np.sum((gt_classes == pred_cls) & pred_mask)
        total = np.sum(pred_mask)
        prec = tp / total if total > 0 else 0
        print(f" {prec:>8.1%}", end="")
    print()

    print(f"\nOverall accuracy: {metrics['accuracy']:.1%}")

    print(f"\nMean predicted value per GT class:")
    print(f"  Win : {metrics['mean_pred_win']:+.4f}")
    print(f"  Loss: {metrics['mean_pred_loss']:+.4f}")
    print(f"  Overall mean: {metrics['mean_pred']:+.4f}")

    print(f"\nPredicted value histogram:")
    bins = np.linspace(-1, 1, 11)
    counts, _ = np.histogram(pred_values, bins=bins)
    max_count = max(counts)
    bar_width = 40
    for i in range(len(counts)):
        lo, hi = bins[i], bins[i+1]
        bar_len = int(counts[i] / max_count * bar_width) if max_count > 0 else 0
        print(f"  [{lo:+.1f},{hi:+.1f}] {'#' * bar_len:<{bar_width}} {counts[i]:>5d}")


def evaluate_all_models(states: np.ndarray, gt_values: np.ndarray):
    '''Loops through all models in training_dir and prints a summary table.'''
    import os
    # find all model files
    model_files = [
        f for f in os.listdir(config.training_dir)
        if f.startswith("model_") and f.endswith(".pkl")
    ]
    iterations = sorted([
        int(f[len("model_"):-len(".pkl")]) for f in model_files
    ])

    if not iterations:
        logger.error("No models found in training directory")
        return

    logger.info(f"Found {len(iterations)} models: {iterations[0]} to {iterations[-1]}")

    all_metrics = []
    for iteration in iterations:
        model_path = config.training_dir + f"model_{iteration}.pkl"
        model = load_model(model_path)
        metrics = evaluate_model(model, states, gt_values)
        all_metrics.append((iteration, metrics))
        # print detailed report for each model
        print_detailed_report(iteration, metrics, gt_values)

    # summary table
    print(f"\n{'='*100}")
    print(f"Summary across all models")
    print(f"{'='*100}")
    print(f"{'Model':>7} {'Acc':>8} {'WinPred%':>9} {'MeanPred':>9} {'MeanWin':>9} {'MeanLoss':>9} {'WinRec':>8} {'LossRec':>8}")
    print(f"{'-'*7} {'-'*8} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*8} {'-'*8}")
    for iteration, m in all_metrics:
        print(f"{iteration:>7d} {m['accuracy']:>8.1%} {m['pred_win_pct']:>9.1%} {m['mean_pred']:>+9.4f} {m['mean_pred_win']:>+9.4f} {m['mean_pred_loss']:>+9.4f} {m['win_recall']:>8.1%} {m['loss_recall']:>8.1%}")


def main():
    n_per_class = 1000
    model_iteration = None  # None = all models, or set to an int (e.g. 5) for a specific model

    logger.info(f"Generating {n_per_class} balanced random states per class...")
    buckets = get_balanced_random_states(n_per_class)
    for cls, name in [(1.0, "Win"), (-1.0, "Loss")]:
        logger.info(f"  {name}: {len(buckets[cls])}")

    states, gt_values = prepare_eval_data(buckets)

    if model_iteration is None:
        evaluate_all_models(states, gt_values)
    else:
        model_path = config.training_dir + f"model_{model_iteration}.pkl"
        model = load_model(model_path)
        logger.info(f"Loaded model from {model_path}")
        metrics = evaluate_model(model, states, gt_values)
        print_detailed_report(model_iteration, metrics, gt_values)

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="model_bias"
    )
    main()
