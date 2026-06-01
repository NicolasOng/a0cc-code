"""
Load four trained A0 model checkpoints — each for a DIFFERENT board/player size
— evaluate each against its own standard baseline via
a0.eval.player.evaluate_references, and plot the resulting expected values
(P1 / P2) as a grouped bar chart.

Why subprocesses?
  a0.eval.player reads the global `config` singleton, which is built from
  sys.argv[1] both in this process AND in every spawned game-worker (under the
  'spawn' start method, workers inherit argv and rebuild config from the file).
  So board/player size is fixed per-process and a single process can only
  evaluate models of one board size. To mix sizes we evaluate ONE model per
  subprocess, launching each with its own config file, then aggregate + plot.

Usage:
    python scripts/2026-06-01_evaluate_four_models.py
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
# Put the repo root on sys.path so `a0`/`config` import when run as a script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# (label, config_path, model_path) -- one entry per model.
# IMPORTANT: set the config that matches each model's board/player size.
# The config<NN>-6.json files are board sizes 5/6/7/9 (num_pieces=6).
MODEL_SPECS: list[tuple[str, str, str]] = [
    ('163sl', 'config/163.json', 'output/trial_163/baseline_sl_player.pkl'),
    ('256sl', 'config/256long.json', 'output/trial_256/baseline_sl_player.pkl'),
    ('366sl', 'config/config-366.json', 'output/trial_366/baseline_sl_player.pkl'),
    ('494sl', 'config/494.json', 'output/trial_494/baseline_sl_player.pkl'),
]

# Env-var protocol used to drive a worker invocation of this same script.
_ENV_OUT = 'A0_EVAL_ONE_OUT'
_ENV_LABEL = 'A0_EVAL_ONE_LABEL'
_ENV_MODEL = 'A0_EVAL_ONE_MODEL'


def _run_worker() -> None:
    '''Evaluate a single model (given via env vars) against its baseline.

    argv[1] is the model's config path (consumed by config.py at import), so the
    global `config` here — and in the game-workers spawned by run_matchup — is
    correct for this model's board size.
    '''
    import multiprocessing
    from datetime import datetime

    from a0.eval.player import NUM_GAMES, evaluate_references, make_baseline
    from a0.model import load_model
    from a0.players.a0 import A0Player
    from config import config
    from utils.log import get_logger, setup_logging

    logger = get_logger(__name__)
    setup_logging(level=20, log_dir=config.log_dir, process_name='evaluate_four_models')
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    label = os.environ[_ENV_LABEL]
    model_path = os.environ[_ENV_MODEL]
    out_path = os.environ[_ENV_OUT]

    model = load_model(model_path, training=False)
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
    evaluate_references(
        references={label: player},
        opponent=make_baseline(),
        num_games=NUM_GAMES,
        output_path=out_path,
        log_games=1,
        game_log_path=f"{os.path.dirname(out_path)}/{label}_game_logs_{timestamp}.json",
    )
    logger.info(f"Wrote evaluation series for '{label}' to {out_path}")


def _plot_results(names: list[str], ev_p1: list[float], ev_p2: list[float], out_png: str) -> None:
    '''Grouped bar chart of P1/P2 expected value per model.'''
    import numpy as np
    import matplotlib.pyplot as plt

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
    plt.title('Model evaluation vs baseline (per-model board size)')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png)
    plt.close()
    print(f"Saved plot to {out_png}")


def main() -> None:
    '''Orchestrate: run one eval subprocess per model, then aggregate + plot.'''
    import subprocess
    from datetime import datetime

    from a0.utils.plotting import load_series
    from config import config

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    work_dir = f"{config.eval_dir}/four_models_{timestamp}"
    os.makedirs(work_dir, exist_ok=True)

    names: list[str] = []
    ev_p1: list[float] = []
    ev_p2: list[float] = []

    for label, config_path, model_path in MODEL_SPECS:
        out_path = f"{work_dir}/{label}.pkl"
        print(f"=== Evaluating '{label}' (config={config_path}, model={model_path}) ===")
        env = {
            **os.environ,
            _ENV_OUT: out_path,
            _ENV_LABEL: label,
            _ENV_MODEL: model_path,
        }
        # argv[1] = this model's config; config.py picks it up in the subprocess
        # and in every game-worker it spawns.
        subprocess.run([sys.executable, __file__, config_path], env=env, check=True)

        series = load_series(out_path)
        names.append(label)
        ev_p1.append(series.ys[f'{label}_ev_p1'][0])
        ev_p2.append(series.ys[f'{label}_ev_p2'][0])

    _plot_results(names, ev_p1, ev_p2, out_png=f"{config.plot_dir}/four_models_evaluation_{timestamp}.png")


if __name__ == "__main__":
    if os.environ.get(_ENV_OUT):
        _run_worker()
    else:
        main()
