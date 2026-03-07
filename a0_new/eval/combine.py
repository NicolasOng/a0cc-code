from a0.eval.combine import load_and_merge_series
from a0.eval.plotting import load_series, plot_shaded_error

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def combine_and_plot():
    # create a list of the output dirs to pull from
    eval_dir = "/eval/"
    output_dir = config.output_dir[:-1]
    outputs: list[str] = []
    for i in range(config.num_trials):
        outputs.append(f"{output_dir}{i+1}{eval_dir}")

    confidence = 0.95

    # merge all eval series across trials
    series_names = [
        "eval_seen_nd",
        "eval_seen_nt",
        "eval_random_nd",
        "eval_random_nt",
    ]
    for name in series_names:
        load_and_merge_series(outputs, f"{name}.pkl", confidence)

    # load the merged series back
    seen_nd = load_series(f"{config.eval_dir}/merged_eval_seen_nd.pkl")
    seen_nt = load_series(f"{config.eval_dir}/merged_eval_seen_nt.pkl")
    random_nd = load_series(f"{config.eval_dir}/merged_eval_random_nd.pkl")
    random_nt = load_series(f"{config.eval_dir}/merged_eval_random_nt.pkl")

    # value accuracy (no-draw states)
    plot_shaded_error(
        "Value Head Accuracy on Ground Truth",
        [
            ("Seen", "±95% CI", seen_nd.x, seen_nd.ys["value_accuracy"], seen_nd.ys["value_accuracy_ci"]),
            ("Random", "±95% CI", random_nd.x, random_nd.ys["value_accuracy"], random_nd.ys["value_accuracy_ci"]),
        ],
        "Iteration", "Accuracy", "merged_value_accuracy_nd"
    )

    # policy accuracy (non-trivial states)
    plot_shaded_error(
        "Policy Head Accuracy on Ground Truth (Non-Trivial States)",
        [
            ("Seen", "±95% CI", seen_nt.x, seen_nt.ys["policy_accuracy"], seen_nt.ys["policy_accuracy_ci"]),
            ("Random", "±95% CI", random_nt.x, random_nt.ys["policy_accuracy"], random_nt.ys["policy_accuracy_ci"]),
        ],
        "Iteration", "Accuracy", "merged_policy_accuracy_nt"
    )

def main():
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="combining"
    )
    combine_and_plot()

if __name__ == "__main__":
    main()
