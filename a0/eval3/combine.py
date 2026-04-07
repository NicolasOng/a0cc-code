'''
Merge per-trial distribution series across multiple output_dir<i> trial runs
and plot the combined result. Each of the three distribution series produced
by eval3/training_data.py + eval3/dataset_evaluation.py gets merged and
plotted as a shaded ridgeline with cross-trial CI bands.

Trial dirs are built from config.output_dir + config.num_trials, matching the
convention used by a0/eval/combine.py.
'''
from a0.utils.plotting import (
    load_distribution_series,
    load_and_merge_distribution_series,
    plot_shaded_ridgeline,
)
from a0.eval3.plotting import safeplot

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)


DISTRIBUTION_SERIES_FILES = [
    "dataset_pre_balance_distributions",
    "dataset_post_balance_distributions",
    "random_nd_value_distributions",
]


def _get_trial_dirs() -> list[str]:
    '''Build the per-trial eval directory list from config.output_dir.'''
    eval_sub = "/eval/"
    output_dir = config.output_dir[:-1]  # strip trailing slash

    trial_dirs: list[str] = []
    for i in range(config.num_trials):
        #if i + 1 in []: continue
        trial_dir = f"{output_dir}{i+1}{eval_sub}"
        trial_dirs.append(trial_dir)
    return trial_dirs


def merge_all_distribution_series() -> None:
    '''
    Load each distribution series from every trial dir and save the merged
    version to config.eval_dir/merged_<fn>.pkl. Each merge is wrapped so a
    missing file in one trial doesn't block the others.
    '''
    trial_dirs = _get_trial_dirs()
    logger.info(f"Merging distribution series from {len(trial_dirs)} trial dirs...")
    for fn in DISTRIBUTION_SERIES_FILES:
        try:
            load_and_merge_distribution_series(trial_dirs, f"{fn}.pkl")
        except Exception as e:
            logger.warning(f"merge: skipping {fn}: {type(e).__name__}: {e}")


def plot_merged_dataset_pre_balance_distributions() -> None:
    series = load_distribution_series(
        f"{config.eval_dir}/merged_dataset_pre_balance_distributions.pkl"
    )
    plot_shaded_ridgeline(
        trials=series.trials,
        labels=[str(x) for x in series.x],
        title="Dataset Pre Balance Distributions (merged)",
        x_label="Value", y_label="Iteration",
        fn="merged_dataset_pre_balance_distributions",
    )


def plot_merged_dataset_post_balance_distributions() -> None:
    series = load_distribution_series(
        f"{config.eval_dir}/merged_dataset_post_balance_distributions.pkl"
    )
    plot_shaded_ridgeline(
        trials=series.trials,
        labels=[str(x) for x in series.x],
        title="Dataset Post Balance Distributions (merged)",
        x_label="Value", y_label="Iteration",
        fn="merged_dataset_post_balance_distributions",
    )


def plot_merged_random_nd_value_distributions() -> None:
    series = load_distribution_series(
        f"{config.eval_dir}/merged_random_nd_value_distributions.pkl"
    )
    plot_shaded_ridgeline(
        trials=series.trials,
        labels=[str(x) for x in series.x],
        title="Model Value Predictions on random_nd (merged)",
        x_label="Value", y_label="Iteration",
        fn="merged_random_nd_value_distributions",
    )


def main():
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="combining",
    )
    logger.info("combining...")

    merge_all_distribution_series()

    safeplot(plot_merged_dataset_pre_balance_distributions)
    safeplot(plot_merged_dataset_post_balance_distributions)
    safeplot(plot_merged_random_nd_value_distributions)


if __name__ == "__main__":
    main()
