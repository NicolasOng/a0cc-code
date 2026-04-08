"""
Merge per-trial eval series for a single HP point.

Reads trials from `<config.output_dir>/trial_*/eval/`, merges each pkl across
trials, and writes the merged_*.pkl files to `<config.eval_dir>/`.

Globs for trial directories rather than trusting `config.num_trials`, so partial
sweeps (e.g. when some trials were preempted) still combine cleanly.
"""

import glob

from a0.utils.plotting import load_and_merge_series

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)


def find_trial_eval_dirs() -> list[str]:
    """Glob for per-trial eval directories under the current HP's output dir."""
    pattern = f"{config.output_dir}trial_*/eval/"
    dirs = sorted(glob.glob(pattern))
    return dirs


def main():
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="combining_merge"
    )

    trial_dirs = find_trial_eval_dirs()
    if not trial_dirs:
        logger.error(f"No trial eval directories found matching {config.output_dir}trial_*/eval/")
        return

    logger.info(f"Merging {len(trial_dirs)} trial eval directories from {config.output_dir}")
    for d in trial_dirs:
        logger.info(f"  {d}")

    confidence = 0.95

    load_and_merge_series(trial_dirs, "training_gtv_eval.pkl", confidence)
    load_and_merge_series(trial_dirs, "random_gtv_eval.pkl", confidence)
    load_and_merge_series(trial_dirs, "training_ev_eval.pkl", confidence)
    for i in range(2):
        load_and_merge_series(trial_dirs, f"neighbor_{i+1}_gtv_eval.pkl", confidence)

    load_and_merge_series(trial_dirs, "training_nt_gtv_eval.pkl", confidence)
    load_and_merge_series(trial_dirs, "random_nt_gtv_eval.pkl", confidence)
    load_and_merge_series(trial_dirs, "training_nt_ev_eval.pkl", confidence)
    for i in range(2):
        load_and_merge_series(trial_dirs, f"neighbor_{i+1}_nt_gtv_eval.pkl", confidence)

    load_and_merge_series(trial_dirs, "gamedata_stats.pkl", confidence)
    load_and_merge_series(trial_dirs, "gamedata_acc.pkl", confidence)
    load_and_merge_series(trial_dirs, "gamedata_overall_acc.pkl", confidence)
    load_and_merge_series(trial_dirs, "training_metrics.pkl", confidence)
    load_and_merge_series(trial_dirs, "gamedata_bias.pkl", confidence)
    load_and_merge_series(trial_dirs, "gamedata_overall_bias.pkl", confidence)
    load_and_merge_series(trial_dirs, "gamedata_progress_acc_100.pkl", confidence)
    load_and_merge_series(trial_dirs, "gamedata_progress_acc_10.pkl", confidence)
    load_and_merge_series(trial_dirs, "state_progress_gtv_datasets_eval.pkl", confidence)
    load_and_merge_series(trial_dirs, "state_progress_nt_gtv_datasets_eval.pkl", confidence)
    load_and_merge_series(trial_dirs, "state_progress_state_nums.pkl", confidence)

    num_bins_list = [10]
    for num_bins in num_bins_list:
        load_and_merge_series(trial_dirs, f"num_states_over_game_progress_{num_bins}.pkl", confidence)
        load_and_merge_series(trial_dirs, f"num_unique_states_over_game_progress_{num_bins}.pkl", confidence)
        load_and_merge_series(trial_dirs, f"baseline_accuracy_over_game_progress_{num_bins}.pkl", confidence)
        load_and_merge_series(trial_dirs, f"training_data_accuracy_over_game_progress_{num_bins}.pkl", confidence)
        load_and_merge_series(trial_dirs, f"branching_factor_over_game_progress_{num_bins}.pkl", confidence)


if __name__ == "__main__":
    main()
