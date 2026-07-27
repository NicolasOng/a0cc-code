'''Plateau MCTS-sweep player eval.

For the last `plateau_sweep_num_checkpoints` converged checkpoints of a trial,
play `plateau_sweep_num_games` games/side vs the DIST baseline at each search
budget in `plateau_sweep_mcts_samples`, and record the win rate averaged over
those checkpoints. Low search budgets expose value-head quality (which heavy
MCTS otherwise masks), so this is the measurement that should reflect the
value-accuracy differences between td_lambda settings.

Sibling to a0.eval.player (the win-rate *curve*); shares the matchup engine
(build_a0_player / run_matchup / make_baseline). Manually submitted.

Output (per trial): {eval_dir}/player_sweep_results.pkl, a Series with
    x  = mcts_samples budgets
    ys = winrate / winrate_p1 / winrate_p2 (mean over the last-N checkpoints),
         plus *_std / *_ci across those checkpoints.
Aggregate across seeds and plot separately.

Usage: python -m a0.eval.player_sweep <config.json> <trial_no>
'''
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# Disable CUDA-graph command buffers: 32 worker CUDA contexts otherwise exhaust
# the GPU on some drivers (see a0.eval.player memory notes).
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_enable_command_buffer=')

import gc
import multiprocessing

import numpy as np

from a0.model import load_model
from a0.utils.plotting import Series, save_series
from a0.eval.player import (build_a0_player, run_matchup, make_baseline,
                            available_checkpoint_iters)
from config import config
from utils.log import get_logger, setup_logging

logger = get_logger(__name__)


def _winrate(stats) -> float:
    return stats.wins / stats.num_games if stats.num_games else float('nan')


def main() -> None:
    setup_logging(level=20, log_dir=config.log_dir, process_name='player_sweep')
    if not config.do_player_eval:
        logger.info("config.do_player_eval=False; skipping plateau sweep.")
        return
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    budgets = list(config.plateau_sweep_mcts_samples)
    n_ck = config.plateau_sweep_num_checkpoints
    num_games = config.plateau_sweep_num_games

    available = available_checkpoint_iters()
    if not available:
        logger.warning("No checkpoints found; nothing to sweep.")
        return
    checkpoints = available[-n_ck:]
    logger.info(f"Plateau sweep: budgets={budgets}, checkpoints={checkpoints} "
                f"(last {n_ck} of {len(available)}), games/side={num_games}")

    baseline = make_baseline()
    series = Series(ys=['winrate', 'winrate_p1', 'winrate_p2',
                        'winrate_std', 'winrate_p1_std', 'winrate_p2_std',
                        'winrate_ci', 'winrate_p1_ci', 'winrate_p2_ci'])

    for budget in budgets:
        per_ck, per_ck_p1, per_ck_p2 = [], [], []
        for it in checkpoints:
            model = load_model(f"{config.training_dir}/model_{it}.pkl")
            player = build_a0_player(model, budget)
            s1 = run_matchup(player, baseline, num_games, focal_first=True,
                             label=f'sweep_m{budget}_i{it}_p1')
            s2 = run_matchup(player, baseline, num_games, focal_first=False,
                             label=f'sweep_m{budget}_i{it}_p2')
            w1, w2 = _winrate(s1), _winrate(s2)
            wall = (s1.wins + s2.wins) / (s1.num_games + s2.num_games)
            per_ck.append(wall); per_ck_p1.append(w1); per_ck_p2.append(w2)
            logger.info(f"  mcts={budget} iter={it}: winrate={wall:.3f} "
                        f"(p1={w1:.3f} p2={w2:.3f})")
            del model, player
            gc.collect()

        series.x.append(budget)
        for name, vals in [('winrate', per_ck), ('winrate_p1', per_ck_p1),
                           ('winrate_p2', per_ck_p2)]:
            arr = np.asarray(vals, dtype=float)
            std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            series.ys[name].append(float(np.mean(arr)))
            series.ys[f'{name}_std'].append(std)
            series.ys[f'{name}_ci'].append(1.96 * std / max(1, len(arr)) ** 0.5)
        logger.info(f"=> mcts={budget}: mean winrate={series.ys['winrate'][-1]:.3f} "
                    f"+/-{series.ys['winrate_ci'][-1]:.3f} over {len(per_ck)} checkpoints")

    out = f"{config.eval_dir}/player_sweep_results.pkl"
    save_series(series, out)
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
