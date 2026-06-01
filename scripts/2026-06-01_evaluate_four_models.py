"""
Load four trained A0 model checkpoints, evaluate them against the standard
baseline via a0.eval.player.evaluate_references, and plot the resulting
expected values (P1 / P2) as a grouped bar chart.

Set MODEL_PATHS to the four checkpoint files you want to compare.
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import multiprocessing
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from a0.eval.player import NUM_GAMES, evaluate_references, make_baseline
from a0.model import load_model
from a0.players.a0 import A0Player

from config import config

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)


# Set these to the four checkpoints you want to compare (label -> path).
MODEL_PATHS: dict[str, str] = {
    '163sl': 'output/trial_163/baseline_sl_player.pkl',
    '256sl': 'output/trial_256/baseline_sl_player.pkl',
    '366sl': 'output/trial_366/baseline_sl_player.pkl',
    '494sl': 'output/trial_494/baseline_sl_player.pkl',
}


def make_player(model_path: str) -> A0Player:
    '''Load a checkpoint and wrap it in an A0Player (mirrors get_trained_players).'''
    model = load_model(model_path, training=False)
    return A0Player(
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


def plot_results(names: list[str], ev_p1: list[float], ev_p2: list[float], fn: str) -> None:
    '''Grouped bar chart of P1/P2 expected value per model.'''
    x = np.arange(len(names))
    width = 0.38

    plt.figure(figsize=(16, 9))
    plt.bar(x - width / 2, ev_p1, width, label='as P1')
    plt.bar(x + width / 2, ev_p2, width, label='as P2')
    plt.axhline(0.0, color='black', linewidth=0.8)
    plt.xticks(x, names)
    plt.ylim(-1, 1)
    plt.xlabel('model')
    plt.ylabel('expected value (vs baseline)')
    plt.title(f'Model evaluation vs baseline ({NUM_GAMES} games/side)')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out = f"{config.plot_dir}/{fn}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out)
    plt.close()
    logger.info(f"Saved plot to {out}")


def main() -> None:
    setup_logging(level=20, log_dir=config.log_dir, process_name='evaluate_four_models')
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    references = {name: make_player(path) for name, path in MODEL_PATHS.items()}
    baseline = make_baseline()

    series = evaluate_references(
        references=references,
        opponent=baseline,
        num_games=NUM_GAMES,
        output_path=f"{config.eval_dir}/four_models_evaluation_{timestamp}.pkl",
        log_games=1,
        game_log_path=f"{config.eval_dir}/four_models_game_logs_{timestamp}.json",
    )

    names = list(MODEL_PATHS.keys())
    ev_p1 = [series.ys[f'{name}_ev_p1'][0] for name in names]
    ev_p2 = [series.ys[f'{name}_ev_p2'][0] for name in names]

    plot_results(names, ev_p1, ev_p2, fn=f"four_models_evaluation_{timestamp}")


if __name__ == "__main__":
    main()
