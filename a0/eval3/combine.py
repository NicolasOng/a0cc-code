'''
Merge per-trial series across multiple trial dirs and produce per-HP plots
with cross-trial confidence bands. Each merge and each plot is wrapped so
missing files in some trials don't block the rest of the pipeline.

Trial dirs are discovered by globbing config.output_dir/trial_*/eval/, matching
the convention used by a0/eval/combine_merge.py.
'''
import glob

from a0.utils.plotting import (
    Series,
    load_series,
    load_distribution_series,
    load_and_merge_series,
    load_and_merge_distribution_series,
    plot_shaded_error,
    plot_shaded_ridgeline,
)
from a0.eval3.plotting import safeplot

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)


# Regular Series files to merge across trials.
SERIES_FILES = [
    # training metrics + game stats
    "training_metrics",
    "gamedata_stats",
    # accuracy / bias (experienced + alt-target)
    "gamedata_acc",
    "gamedata_overall_acc",
    "gamedata_alt_acc",
    "gamedata_alt_overall_acc",
    "gamedata_bias",
    "gamedata_overall_bias",
    "gamedata_alt_bias",
    "gamedata_alt_overall_bias",
    # game-progress series
    "gamedata_10_progress_count",
    "experienced_10_progress_acc",
    "alt_targets_10_progress_acc",
    "gamedata_progress_Baseline Accuracy",
    "gamedata_progress_Branching Factor",
    # per-dataset model eval
    "seen_nd_eval",
    "random_nd_eval",
    "neighbor_1_nd_eval",
    "neighbor_2_nd_eval",
    "seen_nt_eval",
    "random_nt_eval",
    "neighbor_1_nt_eval",
    "neighbor_2_nt_eval",
    "experienced_dataset_eval",
    "alt_targets_dataset_eval",
    # per-bucket model eval
    "game_progress_10_nd_eval",
    "game_progress_10_nt_eval",
]

DISTRIBUTION_SERIES_FILES = [
    "dataset_pre_balance_distributions",
    "dataset_post_balance_distributions",
    "random_nd_value_distributions",
]


def _get_trial_dirs() -> list[str]:
    '''Glob for per-trial eval directories under config.output_dir.'''
    pattern = f"{config.output_dir}trial_*/eval/"
    return sorted(glob.glob(pattern))


def merge_all_series() -> None:
    '''
    Load each regular Series from every trial dir and save the merged version
    to {eval_dir}/merged_<fn>.pkl. Each merge is wrapped so a missing file in
    one trial doesn't block the others.
    '''
    trial_dirs = _get_trial_dirs()
    logger.info(f"Merging {len(SERIES_FILES)} series files from {len(trial_dirs)} trial dirs...")
    for fn in SERIES_FILES:
        try:
            load_and_merge_series(trial_dirs, f"{fn}.pkl")
        except Exception as e:
            logger.warning(f"merge: skipping series {fn}: {type(e).__name__}: {e}")


def merge_all_distribution_series() -> None:
    '''
    Load each DistributionSeries from every trial dir and save the merged
    version to {eval_dir}/merged_<fn>.pkl. Each merge is wrapped so a missing
    file in one trial doesn't block the others.
    '''
    trial_dirs = _get_trial_dirs()
    logger.info(f"Merging {len(DISTRIBUTION_SERIES_FILES)} distribution series files from {len(trial_dirs)} trial dirs...")
    for fn in DISTRIBUTION_SERIES_FILES:
        try:
            load_and_merge_distribution_series(trial_dirs, f"{fn}.pkl")
        except Exception as e:
            logger.warning(f"merge: skipping distribution {fn}: {type(e).__name__}: {e}")


# === merged distribution series plots ===

def plot_merged_dataset_pre_balance_distributions() -> None:
    series = load_distribution_series(f"{config.eval_dir}/merged_dataset_pre_balance_distributions.pkl")
    plot_shaded_ridgeline(
        trials=series.trials,
        labels=[str(x) for x in series.x],
        title="Dataset Pre Balance Distributions (merged)",
        x_label="Value", y_label="Iteration",
        fn="merged_dataset_pre_balance_distributions",
    )


def plot_merged_dataset_post_balance_distributions() -> None:
    series = load_distribution_series(f"{config.eval_dir}/merged_dataset_post_balance_distributions.pkl")
    plot_shaded_ridgeline(
        trials=series.trials,
        labels=[str(x) for x in series.x],
        title="Dataset Post Balance Distributions (merged)",
        x_label="Value", y_label="Iteration",
        fn="merged_dataset_post_balance_distributions",
    )


def plot_merged_random_nd_value_distributions() -> None:
    series = load_distribution_series(f"{config.eval_dir}/merged_random_nd_value_distributions.pkl")
    plot_shaded_ridgeline(
        trials=series.trials,
        labels=[str(x) for x in series.x],
        title="Model Value Predictions on random_nd (merged)",
        x_label="Value", y_label="Iteration",
        fn="merged_random_nd_value_distributions",
    )


# === merged regular series plots ===

def _ci_keys(series: Series, key: str) -> tuple[list[float], list[float]]:
    '''Return (mean, ci) for a key in a merged series.'''
    return series.ys[key], series.ys[f"{key}_ci"]


def plot_merged_training_metrics() -> None:
    s = load_series(f"{config.eval_dir}/merged_training_metrics.pkl")
    plot_shaded_error(
        "Training Performance Metrics (merged)",
        [
            ("Value Accuracy",  "±95% CI", s.x, *_ci_keys(s, "Value Accuracy")),
            ("Policy Accuracy", "±95% CI", s.x, *_ci_keys(s, "Policy Accuracy")),
        ],
        "Iteration", "Performance", "merged_training_metrics", (0, 1),
    )


def plot_merged_gamedata_accuracy() -> None:
    iteration = load_series(f"{config.eval_dir}/merged_gamedata_acc.pkl")
    overall = load_series(f"{config.eval_dir}/merged_gamedata_overall_acc.pkl")
    plot_shaded_error(
        "Training Data Accuracy by Iteration (merged)",
        [
            ("Iter Value Accuracy",       "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Value Accuracy")),
            ("Iter Value Accuracy ND",    "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Value Accuracy ND")),
            ("Iter Policy Accuracy",      "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Policy Accuracy")),
            ("Iter Policy Accuracy NT",   "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Policy Accuracy NT")),
            ("Overall Value Accuracy",    "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Value Accuracy")),
            ("Overall Value Accuracy ND", "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Value Accuracy ND")),
            ("Overall Policy Accuracy",   "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Policy Accuracy")),
            ("Overall Policy Accuracy NT","±95% CI", overall.x,   *_ci_keys(overall,   "Overall Policy Accuracy NT")),
        ],
        "Iterations", "Accuracy", "merged_training_data_accuracy", (0, 1),
    )


def plot_merged_gamedata_alt_accuracy() -> None:
    iteration = load_series(f"{config.eval_dir}/merged_gamedata_alt_acc.pkl")
    overall = load_series(f"{config.eval_dir}/merged_gamedata_alt_overall_acc.pkl")
    plot_shaded_error(
        "Alt-Target Training Data Accuracy by Iteration (merged)",
        [
            ("Iter Value Accuracy",       "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Value Accuracy")),
            ("Iter Value Accuracy ND",    "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Value Accuracy ND")),
            ("Iter Policy Accuracy",      "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Policy Accuracy")),
            ("Iter Policy Accuracy NT",   "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Policy Accuracy NT")),
            ("Overall Value Accuracy",    "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Value Accuracy")),
            ("Overall Value Accuracy ND", "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Value Accuracy ND")),
            ("Overall Policy Accuracy",   "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Policy Accuracy")),
            ("Overall Policy Accuracy NT","±95% CI", overall.x,   *_ci_keys(overall,   "Overall Policy Accuracy NT")),
        ],
        "Iterations", "Accuracy", "merged_training_data_alt_accuracy", (0, 1),
    )


def plot_merged_gamedata_bias() -> None:
    iteration = load_series(f"{config.eval_dir}/merged_gamedata_bias.pkl")
    overall = load_series(f"{config.eval_dir}/merged_gamedata_overall_bias.pkl")
    plot_shaded_error(
        "Training Data Bias by Iteration (merged)",
        [
            ("Win",     "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Win Percent")),
            ("Loss",    "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Loss Percent")),
            ("Draw",    "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Draw Percent")),
            ("Overall Win",  "±95% CI", overall.x, *_ci_keys(overall, "Overall Win Percent")),
            ("Overall Loss", "±95% CI", overall.x, *_ci_keys(overall, "Overall Loss Percent")),
            ("Overall Draw", "±95% CI", overall.x, *_ci_keys(overall, "Overall Draw Percent")),
        ],
        "Training Iteration", "Percentage", "merged_gamedata_bias", (0, 1),
    )


def plot_merged_gamedata_alt_bias() -> None:
    iteration = load_series(f"{config.eval_dir}/merged_gamedata_alt_bias.pkl")
    overall = load_series(f"{config.eval_dir}/merged_gamedata_alt_overall_bias.pkl")
    plot_shaded_error(
        "Alt-Target Training Data Bias by Iteration (merged)",
        [
            ("Win",     "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Win Percent")),
            ("Loss",    "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Loss Percent")),
            ("Draw",    "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Draw Percent")),
            ("Overall Win",  "±95% CI", overall.x, *_ci_keys(overall, "Overall Win Percent")),
            ("Overall Loss", "±95% CI", overall.x, *_ci_keys(overall, "Overall Loss Percent")),
            ("Overall Draw", "±95% CI", overall.x, *_ci_keys(overall, "Overall Draw Percent")),
        ],
        "Training Iteration", "Percentage", "merged_gamedata_alt_bias", (0, 1),
    )


def _plot_merged_gp_target_accuracy(name: str, label: str, fn: str) -> None:
    s = load_series(f"{config.eval_dir}/merged_{name}_10_progress_acc.pkl")
    plot_shaded_error(
        f"{label} Accuracy by Game Progress (merged)",
        [
            ("Value Accuracy",     "±95% CI", s.x, *_ci_keys(s, "Value Accuracy")),
            ("Value Accuracy ND",  "±95% CI", s.x, *_ci_keys(s, "Value Accuracy ND")),
            ("Policy Accuracy",    "±95% CI", s.x, *_ci_keys(s, "Policy Accuracy")),
            ("Policy Accuracy NT", "±95% CI", s.x, *_ci_keys(s, "Policy Accuracy NT")),
        ],
        "Game Progress (%)", "Accuracy", fn, (0, 1),
    )


def plot_merged_gp_experienced_accuracy() -> None:
    _plot_merged_gp_target_accuracy("experienced", "Experienced Targets", "merged_gp_experienced_accuracy")


def plot_merged_gp_alt_targets_accuracy() -> None:
    _plot_merged_gp_target_accuracy("alt_targets", "Alt Targets", "merged_gp_alt_targets_accuracy")


def plot_merged_gp_baseline_accuracy() -> None:
    s = load_series(f"{config.eval_dir}/merged_gamedata_progress_Baseline Accuracy.pkl")
    plot_shaded_error(
        "Baseline Accuracy by Game Progress (merged)",
        [
            ("Value Accuracy",     "±95% CI", s.x, *_ci_keys(s, "Value Accuracy")),
            ("Value Accuracy ND",  "±95% CI", s.x, *_ci_keys(s, "Value Accuracy ND")),
            ("Policy Accuracy",    "±95% CI", s.x, *_ci_keys(s, "Policy Accuracy")),
            ("Policy Accuracy NT", "±95% CI", s.x, *_ci_keys(s, "Policy Accuracy NT")),
        ],
        "Game Progress (%)", "Accuracy", "merged_gp_baseline_accuracy", (0, 1),
    )


def plot_merged_gp_branching_factor() -> None:
    s = load_series(f"{config.eval_dir}/merged_gamedata_progress_Branching Factor.pkl")
    plot_shaded_error(
        "Average Branching Factor by Game Progress (merged)",
        [("Branching Factor", "±95% CI", s.x, *_ci_keys(s, "Branching Factor"))],
        "Game Progress (%)", "Branching Factor", "merged_gp_branching_factor",
    )


def _plot_merged_dataset_eval(name: str, title: str) -> None:
    s = load_series(f"{config.eval_dir}/merged_{name}_eval.pkl")
    plot_shaded_error(
        title,
        [
            ("Value Accuracy",  "±95% CI", s.x, *_ci_keys(s, "value_accuracy")),
            ("Policy Accuracy", "±95% CI", s.x, *_ci_keys(s, "policy_accuracy")),
        ],
        "Iteration", "Accuracy", f"merged_{name}_eval", (0, 1),
    )


def plot_merged_seen_nd_eval()   -> None: _plot_merged_dataset_eval("seen_nd",   "Model Performance on Ground Truth (seen, no draws, merged)")
def plot_merged_random_nd_eval() -> None: _plot_merged_dataset_eval("random_nd", "Model Performance on Ground Truth (random, no draws, merged)")
def plot_merged_seen_nt_eval()   -> None: _plot_merged_dataset_eval("seen_nt",   "Model Performance on Ground Truth (seen, non-trivial, merged)")
def plot_merged_random_nt_eval() -> None: _plot_merged_dataset_eval("random_nt", "Model Performance on Ground Truth (random, non-trivial, merged)")


def plot_merged_neighbor_nd_evals() -> None:
    for i in range(2):
        _plot_merged_dataset_eval(f"neighbor_{i+1}_nd", f"Model Performance on Ground Truth (neighbor {i+1}, no draws, merged)")


def plot_merged_neighbor_nt_evals() -> None:
    for i in range(2):
        _plot_merged_dataset_eval(f"neighbor_{i+1}_nt", f"Model Performance on Ground Truth (neighbor {i+1}, non-trivial, merged)")


def plot_merged_experienced_dataset_eval() -> None:
    _plot_merged_dataset_eval("experienced_dataset", "Model Performance on Experienced Targets Dataset (merged)")


def plot_merged_alt_targets_dataset_eval() -> None:
    _plot_merged_dataset_eval("alt_targets_dataset", "Model Performance on Alt Targets Dataset (merged)")


def plot_merged_game_progress_10_nd_eval() -> None:
    s = load_series(f"{config.eval_dir}/merged_game_progress_10_nd_eval.pkl")
    plot_shaded_error(
        "Final Model Accuracy by Game Progress (no draws, merged)",
        [
            ("Value Accuracy",  "±95% CI", s.x, *_ci_keys(s, "value_accuracy")),
            ("Policy Accuracy", "±95% CI", s.x, *_ci_keys(s, "policy_accuracy")),
        ],
        "Game Progress (%)", "Accuracy", "merged_game_progress_10_nd_eval", (0, 1),
    )


def plot_merged_game_progress_10_nt_eval() -> None:
    s = load_series(f"{config.eval_dir}/merged_game_progress_10_nt_eval.pkl")
    plot_shaded_error(
        "Final Model Accuracy by Game Progress (non-trivial, merged)",
        [
            ("Value Accuracy",  "±95% CI", s.x, *_ci_keys(s, "value_accuracy")),
            ("Policy Accuracy", "±95% CI", s.x, *_ci_keys(s, "policy_accuracy")),
        ],
        "Game Progress (%)", "Accuracy", "merged_game_progress_10_nt_eval", (0, 1),
    )


def plot_merged_full_value_accuracy() -> None:
    seen      = load_series(f"{config.eval_dir}/merged_seen_nd_eval.pkl")
    random_   = load_series(f"{config.eval_dir}/merged_random_nd_eval.pkl")
    n1        = load_series(f"{config.eval_dir}/merged_neighbor_1_nd_eval.pkl")
    n2        = load_series(f"{config.eval_dir}/merged_neighbor_2_nd_eval.pkl")
    iteration = load_series(f"{config.eval_dir}/merged_gamedata_acc.pkl")
    overall   = load_series(f"{config.eval_dir}/merged_gamedata_overall_acc.pkl")
    plot_shaded_error(
        "Value Head Performance on Ground Truth and Training Data (merged)",
        [
            ("Seen",                  "±95% CI", seen.x,      *_ci_keys(seen,      "value_accuracy")),
            ("Neighbor 1",            "±95% CI", n1.x,        *_ci_keys(n1,        "value_accuracy")),
            ("Neighbor 2",            "±95% CI", n2.x,        *_ci_keys(n2,        "value_accuracy")),
            ("Random",                "±95% CI", random_.x,   *_ci_keys(random_,   "value_accuracy")),
            ("Training Data",         "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Value Accuracy")),
            ("Training Data ND",      "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Value Accuracy ND")),
            ("Overall Training Data", "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Value Accuracy")),
            ("Overall Training Data ND", "±95% CI", overall.x, *_ci_keys(overall,  "Overall Value Accuracy ND")),
        ],
        "Iteration", "Accuracy", "merged_full_value_accuracy", (0, 1),
    )


def plot_merged_full_policy_accuracy() -> None:
    seen_nt   = load_series(f"{config.eval_dir}/merged_seen_nt_eval.pkl")
    random_nt = load_series(f"{config.eval_dir}/merged_random_nt_eval.pkl")
    n1_nt     = load_series(f"{config.eval_dir}/merged_neighbor_1_nt_eval.pkl")
    n2_nt     = load_series(f"{config.eval_dir}/merged_neighbor_2_nt_eval.pkl")
    iteration = load_series(f"{config.eval_dir}/merged_gamedata_acc.pkl")
    overall   = load_series(f"{config.eval_dir}/merged_gamedata_overall_acc.pkl")
    plot_shaded_error(
        "Policy Head Performance on Ground Truth (NT) and Training Data (merged)",
        [
            ("Seen NT",                  "±95% CI", seen_nt.x,   *_ci_keys(seen_nt,   "policy_accuracy")),
            ("Neighbor 1 NT",            "±95% CI", n1_nt.x,     *_ci_keys(n1_nt,     "policy_accuracy")),
            ("Neighbor 2 NT",            "±95% CI", n2_nt.x,     *_ci_keys(n2_nt,     "policy_accuracy")),
            ("Random NT",                "±95% CI", random_nt.x, *_ci_keys(random_nt, "policy_accuracy")),
            ("Training Data NT",         "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Policy Accuracy NT")),
            ("Overall Training Data NT", "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Policy Accuracy NT")),
        ],
        "Iteration", "Accuracy", "merged_full_policy_accuracy", (0, 1),
    )


def main():
    setup_logging(level=20, log_dir=config.log_dir, process_name="combining")
    logger.info("combining...")

    merge_all_series()
    merge_all_distribution_series()

    # merged distribution plots
    safeplot(plot_merged_dataset_pre_balance_distributions)
    safeplot(plot_merged_dataset_post_balance_distributions)
    safeplot(plot_merged_random_nd_value_distributions)

    # merged training metrics + game data accuracy / bias
    safeplot(plot_merged_training_metrics)
    safeplot(plot_merged_gamedata_accuracy)
    safeplot(plot_merged_gamedata_alt_accuracy)
    safeplot(plot_merged_gamedata_bias)
    safeplot(plot_merged_gamedata_alt_bias)

    # merged game progress
    safeplot(plot_merged_gp_experienced_accuracy)
    safeplot(plot_merged_gp_alt_targets_accuracy)
    safeplot(plot_merged_gp_baseline_accuracy)
    safeplot(plot_merged_gp_branching_factor)

    # merged per-dataset model eval
    safeplot(plot_merged_seen_nd_eval)
    safeplot(plot_merged_random_nd_eval)
    safeplot(plot_merged_seen_nt_eval)
    safeplot(plot_merged_random_nt_eval)
    safeplot(plot_merged_neighbor_nd_evals)
    safeplot(plot_merged_neighbor_nt_evals)
    safeplot(plot_merged_experienced_dataset_eval)
    safeplot(plot_merged_alt_targets_dataset_eval)

    # merged per-bucket model eval
    safeplot(plot_merged_game_progress_10_nd_eval)
    safeplot(plot_merged_game_progress_10_nt_eval)

    # merged combined plots
    safeplot(plot_merged_full_value_accuracy)
    safeplot(plot_merged_full_policy_accuracy)


if __name__ == "__main__":
    main()
