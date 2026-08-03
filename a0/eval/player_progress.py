"""Print how many checkpoints still need player evaluation for (config, trial).

Mirrors the checkpoint-selection in a0.eval.player.main: the player-curve eval
is incremental (it skips checkpoints already present in the results series), so a
chained job can resume and finish a curve one 12h window couldn't. slurm/player_eval.sh
uses this to decide whether to queue a continuation. Prints a single integer
(0 = nothing left to evaluate for this trial).

Usage: python -m a0.eval.player_progress <config.json> <trial_no>
"""
from a0.utils.plotting import load_series
from a0.eval.player import available_checkpoint_iters, iters_to_evaluate
from config import config


def main() -> None:
    available = available_checkpoint_iters()
    existing = load_series(f"{config.eval_dir}/player_evaluation_results.pkl", optional=True)
    already_done = set(existing.x) if existing is not None else set()
    print(len(iters_to_evaluate(available, already_done)))


if __name__ == "__main__":
    main()
