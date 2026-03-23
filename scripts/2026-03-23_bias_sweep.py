'''
Bias sweep experiment: train models on ground truth data with varying win/loss ratios.
Measures how dataset bias affects model predictions and accuracy.

Usage:
    Timing test:   python -m scripts.2026-03-23_bias_sweep config_file.json
    Full sweep:    python -m scripts.2026-03-23_bias_sweep config_file.json sweep
'''
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import time
import pickle
import multiprocessing
import matplotlib.pyplot as plt

from config import config
from cc.ground_truth import GroundTruth
from a0.model import AlphaZeroModel, create_model
from a0.model_utils import board_to_input, get_legal_move_mask_from_state
from a0.dataset import Dataset
from a0.train.dataset import (
    train_model_epoch, DatasetData, EpochData,
    plot_model_performance, save_dataset_data,
)

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)


def generate_biased_gt_dataset(win_pct: float, total_size: int, seed: int) -> Dataset:
    """
    Generate a GT dataset with a specific win/loss ratio. Draws are excluded.
    win_pct: fraction of wins (e.g., 0.6 means 60% wins, 40% losses)
    total_size: total number of samples
    seed: random seed for reproducibility
    """
    gt = GroundTruth()
    max_rank = gt.get_max_rank()

    target_wins = int(total_size * win_pct)
    target_losses = total_size - target_wins

    rng = np.random.RandomState(seed)

    # collect wins and losses separately
    wins = {'states': [], 'values': [], 'policies': [], 'masks': []}
    losses = {'states': [], 'values': [], 'policies': [], 'masks': []}

    seen = set()
    while len(wins['states']) < target_wins or len(losses['states']) < target_losses:
        rank = rng.randint(0, max_rank)
        if rank in seen:
            continue
        seen.add(rank)

        board = gt.unrank(rank)
        outcome = gt.get_outcome(board)

        if outcome == 0.0:
            continue
        if outcome > 0 and len(wins['states']) >= target_wins:
            continue
        if outcome < 0 and len(losses['states']) >= target_losses:
            continue

        state = board_to_input(board)[0]  # (BS, BS, 2)
        value = np.array([outcome])
        policy = np.array(gt.get_1ply_policy_prob_dist_list(board, for_model=True))
        mask = get_legal_move_mask_from_state(board, for_model=True)

        bucket = wins if outcome > 0 else losses
        bucket['states'].append(state)
        bucket['values'].append(value)
        bucket['policies'].append(policy)
        bucket['masks'].append(mask)

    # combine and build dataset
    all_states = np.stack(wins['states'] + losses['states'])
    all_values = np.stack(wins['values'] + losses['values'])
    all_policies = np.stack(wins['policies'] + losses['policies'])
    all_masks = np.stack(wins['masks'] + losses['masks'])

    dataset = Dataset(config.training_batch_size)
    dataset.set(all_states, all_values, all_policies, all_masks)
    dataset.shuffle()

    return dataset


def generate_balanced_eval_set(n_per_class: int, seed: int) -> Dataset:
    """
    Generate a balanced evaluation Dataset with equal wins and losses.
    Includes GT policies and legal move masks.
    """
    gt = GroundTruth()
    max_rank = gt.get_max_rank()
    rng = np.random.RandomState(seed)

    buckets: dict[float, list] = {1.0: [], -1.0: []}
    seen: set[int] = set()

    while any(len(v) < n_per_class for v in buckets.values()):
        rank = rng.randint(0, max_rank)
        if rank in seen:
            continue
        seen.add(rank)
        board = gt.unrank(rank)
        outcome = gt.get_outcome(board)
        if outcome == 0.0:
            continue
        if len(buckets[outcome]) < n_per_class:
            buckets[outcome].append(board)

    states, values, policies, masks = [], [], [], []
    for cls in [1.0, -1.0]:
        for board in buckets[cls]:
            states.append(board_to_input(board)[0])
            values.append(np.array([cls]))
            policies.append(np.array(gt.get_1ply_policy_prob_dist_list(board, for_model=True)))
            masks.append(get_legal_move_mask_from_state(board, for_model=True))

    dataset = Dataset(config.training_batch_size)
    dataset.set(np.stack(states), np.stack(values), np.stack(policies), np.stack(masks))
    return dataset


def evaluate_model_bias(model: AlphaZeroModel, states: np.ndarray, gt_values: np.ndarray) -> dict:
    """Evaluate model on balanced eval set. Returns dict of metrics."""
    n = len(gt_values)

    # run inference in batches
    pred_values = []
    batch_size = 256
    for i in range(0, n, batch_size):
        batch = jnp.array(states[i:i + batch_size])
        values, _ = model.inference(batch)
        pred_values.append(np.array(values).flatten())
    pred_values = np.concatenate(pred_values)

    pred_classes = np.where(pred_values > 0, 1.0, -1.0)
    gt_classes = gt_values

    return {
        'accuracy': float(np.mean(pred_classes == gt_classes)),
        'pred_win_pct': float(np.mean(pred_classes == 1.0)),
        'mean_pred': float(np.mean(pred_values)),
        'mean_pred_win': float(np.mean(pred_values[gt_classes == 1.0])),
        'mean_pred_loss': float(np.mean(pred_values[gt_classes == -1.0])),
        'win_recall': float(np.mean(pred_classes[gt_classes == 1.0] == 1.0)),
        'loss_recall': float(np.mean(pred_classes[gt_classes == -1.0] == -1.0)),
        'mean_abs_pred': float(np.mean(np.abs(pred_values))),
        'std_pred': float(np.std(pred_values)),
    }


def plot_eval_over_epochs(epoch_evals: list[dict], win_pct: float, trial: int, prefix: str):
    """Plot bias evaluation metrics over training epochs."""
    epochs = list(range(len(epoch_evals)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Bias Evaluation — win_pct={win_pct:.0%}, trial={trial}", fontsize=14)

    # accuracy + recall
    ax = axes[0, 0]
    ax.plot(epochs, [e['accuracy'] for e in epoch_evals], 'k-o', label='Accuracy')
    ax.plot(epochs, [e['win_recall'] for e in epoch_evals], 'b--o', label='Win Recall', markersize=4)
    ax.plot(epochs, [e['loss_recall'] for e in epoch_evals], 'r--o', label='Loss Recall', markersize=4)
    ax.set_ylabel('Rate')
    ax.set_xlabel('Epoch')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title('Accuracy & Recall')
    ax.grid(True, alpha=0.3)

    # mean predictions per class
    ax = axes[0, 1]
    ax.plot(epochs, [e['mean_pred'] for e in epoch_evals], 'k-o', label='Overall')
    ax.plot(epochs, [e['mean_pred_win'] for e in epoch_evals], 'b--o', label='GT=Win', markersize=4)
    ax.plot(epochs, [e['mean_pred_loss'] for e in epoch_evals], 'r--o', label='GT=Loss', markersize=4)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_ylabel('Mean Predicted Value')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.set_title('Mean Predictions')
    ax.grid(True, alpha=0.3)

    # prediction distribution bias
    ax = axes[1, 0]
    ax.plot(epochs, [e['pred_win_pct'] for e in epoch_evals], 'g-o', label='Pred Win %')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Balanced')
    ax.set_ylabel('Fraction Predicted Win')
    ax.set_xlabel('Epoch')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title('Prediction Bias')
    ax.grid(True, alpha=0.3)

    # confidence
    ax = axes[1, 1]
    ax.plot(epochs, [e['mean_abs_pred'] for e in epoch_evals], 'm-o', label='Mean |pred|')
    ax.plot(epochs, [e['std_pred'] for e in epoch_evals], 'c-o', label='Std pred')
    ax.set_ylabel('Value')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.set_title('Confidence & Spread')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/{prefix}_eval.png")
    plt.close()


def run_trial(
    win_pct: float,
    trial: int,
    eval_dataset: Dataset,
    train_dataset: Dataset | None = None,
    dataset_size: int = 100_000,
    num_epochs: int = 10,
) -> dict:
    """
    Run a single bias trial: train epoch-by-epoch with evaluation after each epoch.
    If train_dataset is None, generates one (for standalone/timing use).
    """
    logger.info(f"=== Trial {trial}, win_pct={win_pct:.0%} ===")
    plot_prefix = f"bias_{int(win_pct * 100)}_trial_{trial}"
    eval_states = eval_dataset.states
    eval_gt_values = eval_dataset.values.flatten()

    # generate dataset if not pre-generated
    t_dataset = 0.0
    if train_dataset is None:
        dataset_seed = trial * 1000 + int(win_pct * 100)
        t0 = time.perf_counter()
        train_dataset = generate_biased_gt_dataset(win_pct, dataset_size, seed=dataset_seed)
        t_dataset = time.perf_counter() - t0
        logger.info(f"Generated dataset in {t_dataset:.1f}s")
    train_dataset.print_distribution()

    # create model
    model = create_model(
        board_size=config.board_size,
        rngs=nnx.Rngs({'params': jax.random.PRNGKey(0)})
    )

    # evaluate before training (epoch 0)
    epoch_evals = [evaluate_model_bias(model, eval_states, eval_gt_values)]

    # train epoch-by-epoch, evaluating after each
    test_datasets = {'eval': eval_dataset}
    dataset_data = DatasetData()
    cur_model_no = 0
    t0 = time.perf_counter()
    for epoch in range(num_epochs):
        logger.info(f"Training epoch {epoch + 1}/{num_epochs}...")
        model, epoch_data, cur_model_no = train_model_epoch(
            model, train_dataset, save="none", cur_model_no=cur_model_no,
            test_datasets=test_datasets,
        )
        dataset_data.epoch_data.append(epoch_data)

        # evaluate after this epoch
        metrics = evaluate_model_bias(model, eval_states, eval_gt_values)
        epoch_evals.append(metrics)
        logger.info(
            f"Epoch {epoch + 1} eval: acc={metrics['accuracy']:.1%}, "
            f"pred_win%={metrics['pred_win_pct']:.1%}, "
            f"mean_pred={metrics['mean_pred']:+.4f}, "
            f"win_recall={metrics['win_recall']:.1%}, "
            f"loss_recall={metrics['loss_recall']:.1%}"
        )
    t_train = time.perf_counter() - t0
    logger.info(f"Training completed in {t_train:.1f}s")

    # save data for later reloading/combining
    save_dataset_data(f"{config.stats_dir}/{plot_prefix}_training.pkl", dataset_data)
    with open(f"{config.stats_dir}/{plot_prefix}_eval.pkl", 'wb') as f:
        pickle.dump(epoch_evals, f)
    logger.info(f"Saved training/eval data to {config.stats_dir}/{plot_prefix}_*.pkl")

    # plot training curves
    plot_model_performance(plot_prefix, [dataset_data])
    # plot eval metrics over epochs
    plot_eval_over_epochs(epoch_evals, win_pct, trial, plot_prefix)

    final_eval = epoch_evals[-1]
    logger.info(
        f"Final eval: acc={final_eval['accuracy']:.1%}, "
        f"pred_win%={final_eval['pred_win_pct']:.1%}, "
        f"mean_pred={final_eval['mean_pred']:+.4f}, "
        f"win_recall={final_eval['win_recall']:.1%}, "
        f"loss_recall={final_eval['loss_recall']:.1%}, "
        f"mean_abs={final_eval['mean_abs_pred']:.4f}, "
        f"std={final_eval['std_pred']:.4f}"
    )

    return {
        'win_pct': win_pct,
        'trial': trial,
        'dataset_size': dataset_size,
        'epoch_evals': epoch_evals,
        'dataset_time': t_dataset,
        'train_time': t_train,
        'dataset_data': dataset_data,
    }


def _generate_dataset_worker(args: tuple) -> tuple[int, Dataset]:
    """Worker for parallel dataset generation. Returns (trial, dataset)."""
    win_pct, trial, dataset_size = args
    seed = trial * 1000 + int(win_pct * 100)
    dataset = generate_biased_gt_dataset(win_pct, dataset_size, seed=seed)
    return trial, dataset


def run_sweep(
    win_pcts: list[float],
    num_trials: int = 8,
    dataset_size: int = 100_000,
    num_epochs: int = 10,
    num_workers: int = 8,
):
    """
    Run the full bias sweep. For each bias level, generates all trial datasets
    in parallel (CPU-bound), then trains each trial sequentially (GPU-bound).
    """
    # generate balanced eval set once
    logger.info("Generating balanced eval set...")
    eval_dataset = generate_balanced_eval_set(n_per_class=1000, seed=99999)
    wins = int(np.sum(eval_dataset.values > 0))
    losses = int(np.sum(eval_dataset.values < 0))
    logger.info(f"Eval set: {len(eval_dataset)} states ({wins} wins, {losses} losses)")

    all_results = []
    t_sweep = time.perf_counter()

    for win_pct in win_pcts:
        logger.info(f"\n{'='*60}")
        logger.info(f"Bias level: {win_pct:.0%} wins — generating {num_trials} datasets in parallel...")
        logger.info(f"{'='*60}")

        # generate all datasets for this bias level in parallel
        worker_args = [(win_pct, trial, dataset_size) for trial in range(1, num_trials + 1)]
        t0 = time.perf_counter()
        with multiprocessing.Pool(num_workers) as pool:
            datasets = dict(pool.map(_generate_dataset_worker, worker_args))
        t_gen = time.perf_counter() - t0
        logger.info(f"Generated {num_trials} datasets in {t_gen:.1f}s (parallel)")

        # train each trial sequentially on GPU
        for trial in range(1, num_trials + 1):
            result = run_trial(
                win_pct=win_pct,
                trial=trial,
                eval_dataset=eval_dataset,
                train_dataset=datasets[trial],
                num_epochs=num_epochs,
            )
            all_results.append(result)

    t_sweep = time.perf_counter() - t_sweep
    logger.info(f"\nSweep complete: {len(all_results)} trials in {t_sweep:.1f}s ({t_sweep/60:.1f}m)")
    return all_results


if __name__ == "__main__":
    setup_logging(level=20, log_dir=config.log_dir, process_name="bias_sweep")

    import sys
    mode = sys.argv[3] if len(sys.argv) > 3 else "timing"

    if mode == "sweep" or True:
        multiprocessing.set_start_method('spawn', force=True)
        run_sweep(
            win_pcts=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
            num_trials=8,
            dataset_size=100_000,
            num_epochs=10,
        )
    else:
        # single trial for timing
        logger.info("Generating balanced eval set...")
        eval_dataset = generate_balanced_eval_set(n_per_class=1000, seed=99999)
        wins = int(np.sum(eval_dataset.values > 0))
        losses = int(np.sum(eval_dataset.values < 0))
        logger.info(f"Eval set: {len(eval_dataset)} states ({wins} wins, {losses} losses)")

        logger.info("Running single trial for timing...")
        t_total = time.perf_counter()
        result = run_trial(
            win_pct=0.50,
            trial=1,
            eval_dataset=eval_dataset,
            dataset_size=100_000,
            num_epochs=10,
        )
        t_total = time.perf_counter() - t_total
        logger.info(f"Total: {t_total:.1f}s (dataset={result['dataset_time']:.1f}s, train={result['train_time']:.1f}s)")
