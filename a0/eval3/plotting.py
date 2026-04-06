from typing import Callable

from a0.utils.plotting import (
    load_distribution_series,
    plot_shaded_ridgeline,
)

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)


def safeplot(plot_callable: Callable[[], None]) -> None:
    '''Run a plot function. If anything raises (missing file, missing key,
    bad shape, etc.), log a warning and continue with the next plot.'''
    try:
        plot_callable()
    except Exception as e:
        logger.warning(f"safeplot: skipping {plot_callable.__name__}: {type(e).__name__}: {e}")


def plot_dataset_pre_balance_distributions() -> None:
    series = load_distribution_series(f"{config.eval_dir}/dataset_pre_balance_distributions.pkl")
    plot_shaded_ridgeline(
        trials=series.trials,
        labels=[str(x) for x in series.x],
        title="Dataset Pre Balance Distributions",
        x_label="Value", y_label="Iteration",
        fn="dataset_pre_balance_distributions",
    )


def plot_dataset_post_balance_distributions() -> None:
    series = load_distribution_series(f"{config.eval_dir}/dataset_post_balance_distributions.pkl")
    plot_shaded_ridgeline(
        trials=series.trials,
        labels=[str(x) for x in series.x],
        title="Dataset Post Balance Distributions",
        x_label="Value", y_label="Iteration",
        fn="dataset_post_balance_distributions",
    )


def plot_random_nd_value_distributions() -> None:
    series = load_distribution_series(f"{config.eval_dir}/random_nd_value_distributions.pkl")
    plot_shaded_ridgeline(
        trials=series.trials,
        labels=[str(x) for x in series.x],
        title="Model Value Predictions on random_nd",
        x_label="Value", y_label="Iteration",
        fn="random_nd_value_distributions"
    )


def main():
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="plotting"
    )
    logger.info("plotting...")

    safeplot(plot_dataset_pre_balance_distributions)
    safeplot(plot_dataset_post_balance_distributions)
    safeplot(plot_random_nd_value_distributions)


if __name__ == "__main__":
    main()
