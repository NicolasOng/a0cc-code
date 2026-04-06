from a0.eval.plotting import load_distribution_series, plot_shaded_ridgeline

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def get_and_save_dataset_diagnostics() -> None:
    '''
    Loads the pre/post-balance dataset distribution series saved by
    `get_and_save_dataset_diagnostics_distributions` and renders each as a
    shaded ridgeline plot. Cross-trial CI bands appear automatically when the
    series has been merged across multiple runs; for a single run only the
    mean line is drawn.
    '''
    series_files = [
        ("dataset_pre_balance_distributions", "Dataset Pre Balance Distributions"),
        ("dataset_post_balance_distributions", "Dataset Post Balance Distributions"),
    ]

    for fn, title in series_files:
        series = load_distribution_series(f"{config.eval_dir}/{fn}.pkl")
        if not series.x:
            logger.error(f"Distribution series {fn} has no iterations; skipping plot.")
            continue
        plot_shaded_ridgeline(
            trials=series.trials,
            labels=[str(x) for x in series.x],
            title=title,
            x_label="Value",
            y_label="Iteration",
            fn=fn,
        )
        logger.info(f"Saved shaded ridgeline plot for {fn}")

def main():
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="plotting"
    )
    logger.info("plotting...")

    get_and_save_dataset_diagnostics()

if __name__ == "__main__":
    main()
