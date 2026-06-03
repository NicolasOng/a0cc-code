import os

import jax
import jax.numpy as jnp
from flax import nnx

import pickle

from a0.train.generate_datasets import generate_random_gtd
from a0.eval3.generate_datasets import get_nd_and_nt_datasets_from_state_list
from a0.train.dataset import train_model_epochs, plot_model_performance
from a0.model import create_model, save_model
from cc.ground_truth import GroundTruth

from a0.utils.states import (
    get_gtd_from_states, get_rd_from_states, get_random_states, StateInfo, StatesInfo,
    get_state_info_for_states, get_states_info, log_states_info,
    filter_state_list, balance_gt_values, remove_duplicates, sample_states
)

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

from config import config

def evaluate_against_baseline(model):
    """Evaluate the trained model (wrapped in an A0Player) against the standard
    MCTS-rollout baseline, saving a Series and the first game's log."""
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

    num_games = NUM_GAMES

    # --- LOCAL TEST: comment out this line for a full run ---
    # num_games = 4
    # --- end LOCAL TEST ---

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info("Evaluating trained baseline SL player against the standard baseline...")
    series = evaluate_references(
        references={"baseline_sl_player": player},
        opponent=make_baseline(),
        num_games=num_games,
        output_path=f"{config.eval_dir}/baseline_sl_player_evaluation_{timestamp}.pkl",
        log_games=1,
        game_log_path=f"{config.eval_dir}/baseline_sl_player_game_logs_{timestamp}.json",
    )
    logger.info(
        f"Evaluation done: ev_p1={series.ys['baseline_sl_player_ev_p1'][0]:.3f}, "
        f"ev_p2={series.ys['baseline_sl_player_ev_p2'][0]:.3f}"
    )


def train_sl_baseline_player():
    # decide on parameters based on config
    num_epochs = config.training_iterations
    n_random_states = config.training_iterations * config.training_samples
    n = 1000  # number of eval states

    # --- LOCAL TEST: comment out this block for a full run ---
    # num_epochs = 2
    # n_random_states = 512
    # n = 256
    # --- end LOCAL TEST ---

    logger.info(f"Training SL baseline player with {num_epochs} epochs and {n_random_states} random states.")

    # create the model
    seed = int.from_bytes(os.urandom(4))
    model = create_model(
        config.board_size,
        rngs=nnx.Rngs({'params': jax.random.PRNGKey(seed)})
    )

    # generate the dataset
    gt = GroundTruth()
    dataset = generate_random_gtd(gt, n_random_states)

    # generate the eval datasets
    rs = get_random_states(n * 3, gt)
    rs, _ = remove_duplicates(rs)
    nd, nt = get_nd_and_nt_datasets_from_state_list(rs, gt, n, batch_size=256)

    # train the model for the specified number of epochs
    model, dd = train_model_epochs(model, dataset, num_epochs=num_epochs, plot=False, test_datasets={"nd": nd, "nt": nt})
    plot_model_performance("training", [dd])

    # save the model
    save_model(config.output_dir + "/baseline_sl_player.pkl", model)

    return model

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="train_baseline_sl_player"
    )
    logger.info("Generating datasets...")

    model = train_sl_baseline_player()
    evaluate_against_baseline(model)
