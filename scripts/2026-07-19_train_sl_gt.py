'''
Trains a single SL-on-GT model on an UNBIASED (50/50 win/loss) dataset of
uniformly sampled states, then evaluates it: value-head accuracy/distribution
on a held-out ND dataset, and a player eval against the standard baseline.

Copied and modified from scripts/2026-07-15_model_bias_test.py — same dataset
pipeline with the bias fixed at 0.5, one model instead of the seed x bias grid,
and the model is saved so it can be compared against an alternative_target="gt"
model from the thesis_gt sweep (e.g. with the same probe script / player evals).

Env overrides: SEED, EPOCHS (default 10), TRAIN_SIZE (default 50000).
'''
import os
import random

import numpy as np

from a0.dataset import Dataset

from a0.eval3.generate_datasets import get_nd_and_nt_datasets_from_state_list, get_random_state_list, get_training_dataset
from cc.ground_truth import GroundTruth

from a0.train.dataset import DatasetData, make_optimizer, train_model_epochs
from a0.model import AlphaZeroModel, create_model, save_model
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

def train_model(seed: int, dataset: Dataset, test_datasets: dict[str, Dataset], num_epochs: int = 10) -> tuple[AlphaZeroModel, DatasetData]:
    # create_model (instead of a hardcoded AlphaZeroModel) so the architecture
    # settings (num_filters, num_resblocks, value_head_zero_init, ...) come from
    # the config, matching what the a0 training loop builds
    model = create_model(
        config.board_size,
        rngs=nnx.Rngs({'params': jax.random.PRNGKey(seed)})
    )
    trained_model, dataset_data = train_model_epochs(
        model=model,
        dataset=dataset,
        num_epochs=num_epochs,
        save="None",
        plot=False,
        test_datasets=test_datasets,
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

def evaluate_against_baseline(model: AlphaZeroModel) -> None:
    '''
    Evaluates the trained model (wrapped in an A0Player) against the standard
    MCTS-rollout baseline, saving a Series and the first game's log.
    Copied from scripts/2026-05-27_train_baseline_sl_player.py.
    '''
    import multiprocessing
    from datetime import datetime

    from a0.eval.player import NUM_GAMES, evaluate_references, make_baseline
    from a0.players.a0 import A0Player

    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    player = A0Player(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        model=model,
        exploit=True,
        mcts_samples=config.mcts_samples,
        no_reverse_moves=not config.backwards_moves,
        no_illegal_moves=not config.illegal_moves,
        no_side_moves=not config.sideways_moves,
        rollout_type=config.rollout_type,
        rollout_depth=config.rollout_depth,
        policy_type=config.policy_type,
        epsilon=config.epsilon,
        dirichlet_epsilon=config.dirichlet_epsilon,
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info("Evaluating unbiased SL GT player against the standard baseline...")
    series = evaluate_references(
        references={"sl_gt_unbiased_player": player},
        opponent=make_baseline(),
        num_games=NUM_GAMES,
        output_path=f"{config.eval_dir}/sl_gt_unbiased_evaluation_{timestamp}.pkl",
        log_games=1,
        game_log_path=f"{config.eval_dir}/sl_gt_unbiased_game_logs_{timestamp}.json",
    )
    logger.info(
        f"Evaluation done: ev_p1={series.ys['sl_gt_unbiased_player_ev_p1'][0]:.3f}, "
        f"ev_p2={series.ys['sl_gt_unbiased_player_ev_p2'][0]:.3f}"
    )

def main():
    seed = int(os.environ.get("SEED", str(random.randint(0, 1000000))))
    num_epochs = int(os.environ.get("EPOCHS", "10"))
    train_size = int(os.environ.get("TRAIN_SIZE", "100000"))

    logger.info(f"seed={seed}, epochs={num_epochs}, train_size={train_size}")

    gt = GroundTruth()
    full_train, _ = get_training_dataset(train_size, gt)
    eval_nd, eval_nt = get_nd_and_nt_datasets_from_state_list(get_random_state_list(5000, gt), gt, 5000, 256)

    trained_model, _ = train_model(seed, full_train, {"nd": eval_nd, "nt": eval_nt}, num_epochs)

    acc, mean, std = get_model_value_head_accuracy_and_distribution(trained_model, eval_nd)
    logger.info(f"Seed: {seed}, Value Head Accuracy (nd): {acc:.4f}, Mean: {mean:.4f}, Std: {std:.4f}")

    model_path = f"{config.output_dir}/sl_gt_unbiased_model_seed{seed}.pkl"
    save_model(model_path, trained_model)
    logger.info(f"Saved model to {model_path}")

    evaluate_against_baseline(trained_model)

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="train_sl_gt_unbiased"
    )

    main()
