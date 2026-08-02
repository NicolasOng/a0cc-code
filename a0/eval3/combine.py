'''
Merge per-trial series across multiple trial dirs and produce per-HP plots
with cross-trial confidence bands. Each merge and each plot is wrapped so
missing files in some trials don't block the rest of the pipeline.

Trial dirs are discovered by globbing config.output_dir/trial_*/eval/, matching
the convention used by a0/eval/combine_merge.py.
'''
import glob
import json
import math

import numpy as np
from scipy import stats

from a0.utils.plotting import (
    Series,
    load_series,
    load_distribution_series,
    load_and_merge_series,
    load_and_merge_distribution_series,
    plot_bar_with_error,
    plot_percentile_bands,
    plot_shaded_error,
    plot_shaded_error_groups,
    plot_shaded_ridgeline,
    plot_stacked,
    plot_stacked_proportional,
    plot_value_proportions,
    plot_heatmap,
)
from a0.utils.plotting import pool_heatmap_over_x
from a0.eval3.plotting import (
    safeplot, _reference_names, _plot_accuracy_heatmaps, _plot_model_heatmaps,
    _plot_target_accuracy_lines, _plot_model_accuracy_lines, AXES, MODEL_SPECS,
)

from config import config
from utils.log import get_logger, setup_logging
from a0.eval.trial_selection import select_trial_paths
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
    # BPP / BPPMA (experienced + alt-target)
    "gamedata_bpp",
    "gamedata_overall_bpp",
    "gamedata_bppma",
    "gamedata_overall_bppma",
    "gamedata_alt_bpp",
    "gamedata_alt_overall_bpp",
    "gamedata_alt_bppma",
    "gamedata_alt_overall_bppma",
    # game-progress series
    "gamedata_10_progress_count",
    "experienced_10_progress_acc",
    "alt_targets_10_progress_acc",
    "gamedata_10_progress_baseline_accuracy",
    "gamedata_10_progress_branching_factor",
    # accuracy heatmaps (iteration x progress / distance-from-terminal).
    # Mergeable because the collector always emits the full D0..D_MAX key set,
    # so runs whose games differ in length still share a key set.
    "experienced_iter_distance_acc",
    "experienced_iter_progress_acc",
    "alt_targets_iter_distance_acc",
    "alt_targets_iter_progress_acc",
    # model-accuracy heatmaps (checkpoint x progress / distance) + the class
    # balance of the unbalanced distance buckets
    "model_heatmap_progress_nd",
    "model_heatmap_progress_nt",
    "model_heatmap_distance_nd",
    "model_heatmap_distance_nt",
    "distance_class_balance",
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
    # value-derived policy entropy
    "value_policy_entropy",
    # target-refresh analyses (target-refresh runs only; merge skips when absent)
    "refresh_targets_acc",
    "refresh_targets_sidecar_acc",
    "refresh_target_delta",
    "refresh_delta_by_distance",
    # player evaluation
    "player_evaluation_results",
    "reference_evaluation_results",
    # plateau MCTS sweep (x = mcts budgets); merged -> cross-seed mean + CI
    "player_sweep_results",
]

DISTRIBUTION_SERIES_FILES = [
    "dataset_pre_balance_distributions",
    "dataset_post_balance_distributions",
    "random_nd_value_distributions",
    "random_value_distributions",
]

def _per_refresh_distribution_files() -> list[str]:
    '''Per-refresh distribution series exist only on target-refresh runs; the
    file count is discovered by globbing the trial dirs (K may not be knowable
    from the combine-time config).'''
    names: set[str] = set()
    for trial_dir in _get_trial_dirs():
        for path in glob.glob(f"{trial_dir}dataset_*_balance_distributions_refresh_*.pkl"):
            names.add(path.rsplit("/", 1)[-1].removesuffix(".pkl"))
    return sorted(names)


def _get_trial_dirs() -> list[str]:
    '''Glob for per-trial eval directories under config.output_dir.'''
    pattern = f"{config.output_dir}trial_*/eval/"
    return select_trial_paths(sorted(glob.glob(pattern)))


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


def merge_baseline_accuracies(confidence: float = 0.95) -> None:
    '''
    Merge per-trial baseline_accuracies.json (written by dataset_evaluation.py)
    into a single JSON with mean / std / CI per metric per source state list
    across trials. n_boards is preserved per-trial as a list.
    → {eval_dir}/merged_baseline_accuracies.json
    '''
    trial_dirs = _get_trial_dirs()
    fn = "baseline_accuracies.json"

    trial_results: list[dict[str, dict]] = []
    for d in trial_dirs:
        path = f"{d}{fn}"
        try:
            with open(path, 'r') as f:
                trial_results.append(json.load(f))
        except FileNotFoundError:
            logger.warning(f"merge: skipping baseline accuracies at {path}: not found")
        except Exception as e:
            logger.warning(f"merge: skipping baseline accuracies at {path}: {type(e).__name__}: {e}")

    if not trial_results:
        logger.warning("merge_baseline_accuracies: no trial JSONs found, skipping.")
        return

    all_names: set[str] = set()
    for r in trial_results:
        all_names |= set(r.keys())

    metric_keys = ["value_accuracy", "policy_accuracy", "value_accuracy_nd", "policy_accuracy_nt"]
    nan_to_none = lambda v: None if isinstance(v, float) and math.isnan(v) else v

    merged: dict[str, dict] = {}
    for name in sorted(all_names):
        per_trial = [r[name] for r in trial_results if name in r]
        entry: dict = {"n_trials": len(per_trial)}
        for k in metric_keys:
            vals = np.array([t[k] for t in per_trial if k in t], dtype=np.float64)
            if vals.size == 0:
                entry[k] = None
                entry[f"{k}_std"] = None
                entry[f"{k}_ci"] = None
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            if vals.size >= 2:
                alpha = 1 - confidence
                t_value = stats.t.ppf(1 - alpha / 2, df=vals.size - 1)
                ci = float(t_value * std / np.sqrt(vals.size))
            else:
                ci = float('nan')
            entry[k] = nan_to_none(mean)
            entry[f"{k}_std"] = nan_to_none(std)
            entry[f"{k}_ci"] = nan_to_none(ci)
        entry["n_boards"] = [int(t["n_boards"]) for t in per_trial if "n_boards" in t]
        merged[name] = entry

    output_path = f"{config.eval_dir}/merged_baseline_accuracies.json"
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    logger.info(
        f"merge: saved merged baseline accuracies for {sorted(all_names)} "
        f"({len(trial_results)} trials) to {output_path}."
    )


def merge_all_distribution_series() -> None:
    '''
    Load each DistributionSeries from every trial dir and save the merged
    version to {eval_dir}/merged_<fn>.pkl. Each merge is wrapped so a missing
    file in one trial doesn't block the others.
    '''
    trial_dirs = _get_trial_dirs()
    files = DISTRIBUTION_SERIES_FILES + _per_refresh_distribution_files()
    logger.info(f"Merging {len(files)} distribution series files from {len(trial_dirs)} trial dirs...")
    for fn in files:
        try:
            load_and_merge_distribution_series(trial_dirs, f"{fn}.pkl")
        except Exception as e:
            logger.warning(f"merge: skipping distribution {fn}: {type(e).__name__}: {e}")


# === merged distribution series plots ===

def _plot_merged_distribution_variants(series_fn: str, title: str, fn_base: str) -> None:
    '''Render ridgeline + percentile bands + value-proportion plots for one merged distribution series.'''
    series = load_distribution_series(f"{config.eval_dir}/{series_fn}")
    labels = [str(x) for x in series.x]
    plot_shaded_ridgeline(
        trials=series.trials, labels=labels,
        title=title,
        x_label="Value", y_label="Iteration",
        fn=fn_base,
    )
    plot_percentile_bands(
        trials=series.trials, labels=labels,
        title=f"{title} (percentile bands)",
        x_label="Iteration", y_label="Value",
        fn=f"{fn_base}_bands",
    )
    plot_value_proportions(
        trials=series.trials, labels=labels,
        title=f"{title} (value-bin proportions)",
        x_label="Iteration", y_label="Proportion of samples",
        fn=f"{fn_base}_proportions",
    )


def plot_merged_dataset_pre_balance_distributions() -> None:
    _plot_merged_distribution_variants(
        "merged_dataset_pre_balance_distributions.pkl",
        "Dataset Pre Balance Distributions (merged)",
        "merged_dataset_pre_balance_distributions",
    )


def plot_merged_dataset_post_balance_distributions() -> None:
    _plot_merged_distribution_variants(
        "merged_dataset_post_balance_distributions.pkl",
        "Dataset Post Balance Distributions (merged)",
        "merged_dataset_post_balance_distributions",
    )


def plot_merged_random_nd_value_distributions() -> None:
    _plot_merged_distribution_variants(
        "merged_random_nd_value_distributions.pkl",
        "Model Value Predictions on random_nd (merged)",
        "merged_random_nd_value_distributions",
    )


def plot_merged_random_value_distributions() -> None:
    _plot_merged_distribution_variants(
        "merged_random_value_distributions.pkl",
        "Model Value Predictions on random (merged)",
        "merged_random_value_distributions",
    )


# === merged regular series plots ===

def _ci_keys(series: Series, key: str) -> tuple[list[float], list[float]]:
    '''Return (mean, ci) for a key in a merged series.'''
    return series.ys[key], series.ys[f"{key}_ci"]


def plot_merged_value_policy_entropy() -> None:
    # The per-iteration "Random CI"/"Seen CI" columns (within-run CI across
    # states) are ignored here; cross-trial bands come from the merge's
    # "_ci" columns via _ci_keys.
    s = load_series(f"{config.eval_dir}/merged_value_policy_entropy.pkl")
    plot_shaded_error(
        "Value-Derived Policy Entropy by Iteration (merged)",
        [
            ("Random", "±95% CI", s.x, *_ci_keys(s, "Random")),
            ("Seen",   "±95% CI", s.x, *_ci_keys(s, "Seen")),
        ],
        "Iteration", "Normalized Entropy", "merged_value_policy_entropy", None,
    )


def _merged_refresh_acc_keys(s: Series, nd: bool) -> list[str]:
    prefix = "Value Accuracy ND R" if nd else "Value Accuracy R"
    keys = [k for k in s.ys if k.startswith(prefix) and k[len(prefix):].isdigit()]
    return sorted(keys, key=lambda k: int(k[len(prefix):]))


def _plot_merged_refresh_targets_acc_variant(nd: bool) -> None:
    s = load_series(f"{config.eval_dir}/merged_refresh_targets_acc.pkl")
    side = load_series(f"{config.eval_dir}/merged_refresh_targets_sidecar_acc.pkl", optional=True)
    series = [(k, "±95% CI", s.x, *_ci_keys(s, k)) for k in _merged_refresh_acc_keys(s, nd)]
    if side is not None:
        series += [(f"Buffered {k}", "±95% CI", side.x, *_ci_keys(side, k)) for k in _merged_refresh_acc_keys(side, nd)]
    suffix = "_nd" if nd else ""
    plot_shaded_error(
        f"Refresh Target Value Accuracy{' (no draws)' if nd else ''} vs GT (merged)",
        series,
        "Iteration", "Accuracy", f"merged_refresh_targets_acc{suffix}", (0, 1),
    )


def plot_merged_refresh_targets_acc() -> None:
    _plot_merged_refresh_targets_acc_variant(nd=False)


def plot_merged_refresh_targets_acc_nd() -> None:
    _plot_merged_refresh_targets_acc_variant(nd=True)


def plot_merged_refresh_target_delta() -> None:
    s = load_series(f"{config.eval_dir}/merged_refresh_target_delta.pkl")
    mean_keys = sorted([k for k in s.ys if k.startswith("Delta Mean ") and not k.endswith(("_std", "_ci"))])
    max_keys = sorted([k for k in s.ys if k.startswith("Delta Max ") and not k.endswith(("_std", "_ci"))])
    plot_shaded_error_groups(
        "Refresh Target Delta by Iteration (merged; mean | max)",
        [
            [(k, "±95% CI", s.x, *_ci_keys(s, k)) for k in mean_keys],
            [(k, "±95% CI", s.x, *_ci_keys(s, k)) for k in max_keys],
        ],
        "Iteration", "|target change|", "merged_refresh_target_delta", use_log_y=True,
    )


def plot_merged_refresh_delta_by_distance() -> None:
    s = load_series(f"{config.eval_dir}/merged_refresh_delta_by_distance.pkl")
    keys = [k for k in s.ys if not k.endswith(("_std", "_ci"))]
    plot_shaded_error(
        "Refresh Target Delta by Distance from Terminal (merged propagation front)",
        [(k, "±95% CI", s.x, *_ci_keys(s, k)) for k in keys],
        "Distance from terminal (plies)", "Mean |target change|", "merged_refresh_delta_by_distance", None,
    )


def plot_merged_per_refresh_dataset_distributions() -> None:
    for kind, title in (("pre", "Pre"), ("post", "Post")):
        refresh = 0
        while True:
            fn = f"dataset_{kind}_balance_distributions_refresh_{refresh}"
            if load_distribution_series(f"{config.eval_dir}/merged_{fn}.pkl", optional=True) is None:
                break
            _plot_merged_distribution_variants(
                f"merged_{fn}.pkl",
                f"Dataset {title} Balance Distributions (refresh {refresh}, merged)",
                f"merged_{fn}",
            )
            refresh += 1


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


def plot_merged_training_metrics_with_losses() -> None:
    s = load_series(f"{config.eval_dir}/merged_training_metrics.pkl")
    plot_shaded_error(
        "Training Performance Metrics (with losses, merged)",
        [
            ("Loss",            "±95% CI", s.x, *_ci_keys(s, "Loss")),
            ("Value Loss",      "±95% CI", s.x, *_ci_keys(s, "Value Loss")),
            ("Policy Loss",     "±95% CI", s.x, *_ci_keys(s, "Policy Loss")),
            ("Value Accuracy",  "±95% CI", s.x, *_ci_keys(s, "Value Accuracy")),
            ("Policy Accuracy", "±95% CI", s.x, *_ci_keys(s, "Policy Accuracy")),
        ],
        "Iteration", "Performance", "merged_training_metrics_w_losses",
    )


# === merged gamedata stats ===

def plot_merged_gamedata_total_games() -> None:
    s = load_series(f"{config.eval_dir}/merged_gamedata_stats.pkl")
    plot_shaded_error(
        "Total Games Played by Training Iteration (merged)",
        [("Total Games", "±95% CI", s.x, *_ci_keys(s, "Total Games"))],
        "Training Iteration", "Total Games", "merged_gamedata_total_games",
    )


def plot_merged_gamedata_outcomes_stacked() -> None:
    s = load_series(f"{config.eval_dir}/merged_gamedata_stats.pkl")
    plot_stacked(
        "Game Outcomes by Training Iteration (merged)", s.x,
        [
            ("Player X Wins", s.ys["Player X Wins"]),
            ("Player O Wins", s.ys["Player O Wins"]),
            ("Draws (Repeat)", s.ys["Draws (Repeat)"]),
            ("Draws (Timeout)", s.ys["Draws (Timeout)"]),
        ],
        "Training Iteration", "Number of Games", "merged_gamedata_outcomes_stacked",
    )


def plot_merged_gamedata_outcomes_stacked_proportional() -> None:
    s = load_series(f"{config.eval_dir}/merged_gamedata_stats.pkl")
    plot_stacked_proportional(
        "Game Outcomes by Training Iteration (Proportional, merged)",
        s.x, s.ys["Total Games"],
        [
            ("Player X Wins", s.ys["Player X Wins"]),
            ("Player O Wins", s.ys["Player O Wins"]),
            ("Draws (Repeat)", s.ys["Draws (Repeat)"]),
            ("Draws (Timeout)", s.ys["Draws (Timeout)"]),
        ],
        "Training Iteration", "Proportion of Games", "merged_gamedata_outcomes_stacked_proportional",
    )


def plot_merged_gamedata_outcomes_lines() -> None:
    s = load_series(f"{config.eval_dir}/merged_gamedata_stats.pkl")
    plot_shaded_error(
        "Game Outcomes by Training Iteration (merged)",
        [
            ("Player X Wins",   "±95% CI", s.x, *_ci_keys(s, "Player X Wins")),
            ("Player O Wins",   "±95% CI", s.x, *_ci_keys(s, "Player O Wins")),
            ("Draws (Repeat)",  "±95% CI", s.x, *_ci_keys(s, "Draws (Repeat)")),
            ("Draws (Timeout)", "±95% CI", s.x, *_ci_keys(s, "Draws (Timeout)")),
        ],
        "Training Iteration", "Number of Games", "merged_gamedata_outcomes_lines",
    )


def plot_merged_gamedata_game_length() -> None:
    s = load_series(f"{config.eval_dir}/merged_gamedata_stats.pkl")
    plot_shaded_error(
        "Average Game Length (Turns) by Training Iteration (merged)",
        [("Avg Game Length", "±95% CI", s.x, *_ci_keys(s, "Avg Game Length"))],
        "Training Iteration", "Length (Turns)", "merged_game_length_turns",
    )


def plot_merged_gamedata_game_time() -> None:
    s = load_series(f"{config.eval_dir}/merged_gamedata_stats.pkl")
    plot_shaded_error(
        "Average Game Time by Training Iteration (merged)",
        [("Avg Game Time", "±95% CI", s.x, *_ci_keys(s, "Avg Game Time"))],
        "Training Iteration", "Time (s)", "merged_game_length_time",
    )


def _plot_merged_iteration_accuracy(iteration: Series, overall: Series, title: str, fn: str) -> None:
    plot_shaded_error(
        title,
        [
            ("Iter Value Accuracy",        "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Value Accuracy")),
            ("Iter Value Accuracy ND",     "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Value Accuracy ND")),
            ("Iter Policy Accuracy",       "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Policy Accuracy")),
            ("Iter Policy PM",             "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Policy PM")),
            ("Iter Policy Accuracy NT",    "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Policy Accuracy NT")),
            ("Iter Policy PM NT",          "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Policy PM NT")),
            ("Overall Value Accuracy",     "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Value Accuracy")),
            ("Overall Value Accuracy ND",  "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Value Accuracy ND")),
            ("Overall Policy Accuracy",    "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Policy Accuracy")),
            ("Overall Policy PM",          "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Policy PM")),
            ("Overall Policy Accuracy NT", "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Policy Accuracy NT")),
            ("Overall Policy PM NT",       "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Policy PM NT")),
        ],
        "Iterations", "Accuracy", fn, (0, 1),
    )


def plot_merged_gamedata_accuracy() -> None:
    iteration = load_series(f"{config.eval_dir}/merged_gamedata_acc.pkl")
    overall = load_series(f"{config.eval_dir}/merged_gamedata_overall_acc.pkl")
    _plot_merged_iteration_accuracy(iteration, overall, "Training Data Accuracy by Iteration (merged)", "merged_training_data_accuracy")


def plot_merged_gamedata_alt_accuracy() -> None:
    iteration = load_series(f"{config.eval_dir}/merged_gamedata_alt_acc.pkl")
    overall = load_series(f"{config.eval_dir}/merged_gamedata_alt_overall_acc.pkl")
    _plot_merged_iteration_accuracy(iteration, overall, "Alt-Target Training Data Accuracy by Iteration (merged)", "merged_training_data_alt_accuracy")


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


def plot_merged_gamedata_bpp() -> None:
    iteration = load_series(f"{config.eval_dir}/merged_gamedata_bpp.pkl")
    overall = load_series(f"{config.eval_dir}/merged_gamedata_overall_bpp.pkl")
    plot_shaded_error(
        "Training Data BPP by Iteration (merged)",
        [
            ("Iter Value BPP",        "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Value BPP")),
            ("Iter Value BPP ND",     "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Value BPP ND")),
            ("Iter Policy BPP",       "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Policy BPP")),
            ("Iter Policy BPP NT",    "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Policy BPP NT")),
            ("Overall Value BPP",     "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Value BPP")),
            ("Overall Value BPP ND",  "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Value BPP ND")),
            ("Overall Policy BPP",    "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Policy BPP")),
            ("Overall Policy BPP NT", "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Policy BPP NT")),
        ],
        "Iterations", "BPP", "merged_gamedata_bpp", (0, 1),
    )


def plot_merged_gamedata_bppma() -> None:
    iteration = load_series(f"{config.eval_dir}/merged_gamedata_bppma.pkl")
    overall = load_series(f"{config.eval_dir}/merged_gamedata_overall_bppma.pkl")
    plot_shaded_error(
        "Training Data BPPMA by Iteration (merged)",
        [
            ("Iter Value BPPMA",        "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Value BPPMA")),
            ("Iter Value BPPMA ND",     "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Value BPPMA ND")),
            ("Iter Policy BPPMA",       "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Policy BPPMA")),
            ("Iter Policy BPPMA NT",    "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Policy BPPMA NT")),
            ("Overall Value BPPMA",     "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Value BPPMA")),
            ("Overall Value BPPMA ND",  "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Value BPPMA ND")),
            ("Overall Policy BPPMA",    "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Policy BPPMA")),
            ("Overall Policy BPPMA NT", "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Policy BPPMA NT")),
        ],
        "Iterations", "BPPMA", "merged_gamedata_bppma", (0, 1),
    )


def plot_merged_gamedata_alt_bpp() -> None:
    iteration = load_series(f"{config.eval_dir}/merged_gamedata_alt_bpp.pkl")
    overall = load_series(f"{config.eval_dir}/merged_gamedata_alt_overall_bpp.pkl")
    plot_shaded_error(
        "Alt-Target Training Data BPP by Iteration (merged)",
        [
            ("Iter Value BPP",        "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Value BPP")),
            ("Iter Value BPP ND",     "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Value BPP ND")),
            ("Iter Policy BPP",       "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Policy BPP")),
            ("Iter Policy BPP NT",    "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Policy BPP NT")),
            ("Overall Value BPP",     "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Value BPP")),
            ("Overall Value BPP ND",  "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Value BPP ND")),
            ("Overall Policy BPP",    "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Policy BPP")),
            ("Overall Policy BPP NT", "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Policy BPP NT")),
        ],
        "Iterations", "BPP", "merged_gamedata_alt_bpp", (0, 1),
    )


def plot_merged_gamedata_alt_bppma() -> None:
    iteration = load_series(f"{config.eval_dir}/merged_gamedata_alt_bppma.pkl")
    overall = load_series(f"{config.eval_dir}/merged_gamedata_alt_overall_bppma.pkl")
    plot_shaded_error(
        "Alt-Target Training Data BPPMA by Iteration (merged)",
        [
            ("Iter Value BPPMA",        "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Value BPPMA")),
            ("Iter Value BPPMA ND",     "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Value BPPMA ND")),
            ("Iter Policy BPPMA",       "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Policy BPPMA")),
            ("Iter Policy BPPMA NT",    "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Policy BPPMA NT")),
            ("Overall Value BPPMA",     "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Value BPPMA")),
            ("Overall Value BPPMA ND",  "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Value BPPMA ND")),
            ("Overall Policy BPPMA",    "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Policy BPPMA")),
            ("Overall Policy BPPMA NT", "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Policy BPPMA NT")),
        ],
        "Iterations", "BPPMA", "merged_gamedata_alt_bppma", (0, 1),
    )


def plot_merged_value_accuracies() -> None:
    acc          = load_series(f"{config.eval_dir}/merged_gamedata_acc.pkl")
    overall_acc  = load_series(f"{config.eval_dir}/merged_gamedata_overall_acc.pkl")
    alt_acc      = load_series(f"{config.eval_dir}/merged_gamedata_alt_acc.pkl")
    alt_o_acc    = load_series(f"{config.eval_dir}/merged_gamedata_alt_overall_acc.pkl")
    bppma        = load_series(f"{config.eval_dir}/merged_gamedata_bppma.pkl")
    overall_bppma = load_series(f"{config.eval_dir}/merged_gamedata_overall_bppma.pkl")
    alt_bppma    = load_series(f"{config.eval_dir}/merged_gamedata_alt_bppma.pkl")
    alt_o_bppma  = load_series(f"{config.eval_dir}/merged_gamedata_alt_overall_bppma.pkl")
    plot_shaded_error_groups(
        "Self-Play Value Accuracies (merged)",
        [
            [("Overall Game Outcome Accuracy ND",        "±95% CI", overall_acc.x,   *_ci_keys(overall_acc,   "Overall Value Accuracy ND")),
             ("Game Outcome Accuracy ND",                "±95% CI", acc.x,           *_ci_keys(acc,           "Iteration Value Accuracy ND"))],
            [("Overall Training Data Accuracy ND",       "±95% CI", alt_o_acc.x,     *_ci_keys(alt_o_acc,     "Overall Value Accuracy ND")),
             ("Training Data Accuracy ND",               "±95% CI", alt_acc.x,       *_ci_keys(alt_acc,       "Iteration Value Accuracy ND"))],
            [("Overall Game Outcome Majority Class ND",  "±95% CI", overall_bppma.x, *_ci_keys(overall_bppma, "Overall Value BPPMA ND")),
             ("Value Game Outcome Majority Class ND",    "±95% CI", bppma.x,         *_ci_keys(bppma,         "Iteration Value BPPMA ND"))],
            [("Overall Training Data Majority Class ND", "±95% CI", alt_o_bppma.x,   *_ci_keys(alt_o_bppma,   "Overall Value BPPMA ND")),
             ("Training Data Majority Class ND",         "±95% CI", alt_bppma.x,     *_ci_keys(alt_bppma,     "Iteration Value BPPMA ND"))],
        ],
        "Iterations", "Value", "merged_gamedata_value_summary",
    )


def plot_merged_policy_accuracies() -> None:
    acc          = load_series(f"{config.eval_dir}/merged_gamedata_acc.pkl")
    overall_acc  = load_series(f"{config.eval_dir}/merged_gamedata_overall_acc.pkl")
    alt_acc      = load_series(f"{config.eval_dir}/merged_gamedata_alt_acc.pkl")
    alt_o_acc    = load_series(f"{config.eval_dir}/merged_gamedata_alt_overall_acc.pkl")
    bppma        = load_series(f"{config.eval_dir}/merged_gamedata_bppma.pkl")
    overall_bppma = load_series(f"{config.eval_dir}/merged_gamedata_overall_bppma.pkl")
    alt_bppma    = load_series(f"{config.eval_dir}/merged_gamedata_alt_bppma.pkl")
    alt_o_bppma  = load_series(f"{config.eval_dir}/merged_gamedata_alt_overall_bppma.pkl")
    plot_shaded_error_groups(
        "Self-Play Policy Accuracies (merged)",
        [
            [("Overall Player MCTS Policy Accuracy NT",       "±95% CI", overall_acc.x,   *_ci_keys(overall_acc,   "Overall Policy Accuracy NT")),
             ("Player MCTS Policy Accuracy NT",               "±95% CI", acc.x,           *_ci_keys(acc,           "Iteration Policy Accuracy NT"))],
            [("Overall Training Data Accuracy NT",            "±95% CI", alt_o_acc.x,     *_ci_keys(alt_o_acc,     "Overall Policy Accuracy NT")),
             ("Training Data Accuracy NT",                    "±95% CI", alt_acc.x,       *_ci_keys(alt_acc,       "Iteration Policy Accuracy NT"))],
            [("Overall Player MCTS Policy Majority Class NT", "±95% CI", overall_bppma.x, *_ci_keys(overall_bppma, "Overall Policy BPPMA NT")),
             ("Player MCTS Policy Majority Class NT",         "±95% CI", bppma.x,         *_ci_keys(bppma,         "Iteration Policy BPPMA NT"))],
            [("Overall Training Data Majority Class NT",      "±95% CI", alt_o_bppma.x,   *_ci_keys(alt_o_bppma,   "Overall Policy BPPMA NT")),
             ("Training Data Majority Class NT",              "±95% CI", alt_bppma.x,     *_ci_keys(alt_bppma,     "Iteration Policy BPPMA NT"))],
        ],
        "Iterations", "Policy", "merged_gamedata_policy_summary",
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
            ("Policy PM",          "±95% CI", s.x, *_ci_keys(s, "Policy PM")),
            ("Policy PM NT",       "±95% CI", s.x, *_ci_keys(s, "Policy PM NT")),
        ],
        "Game Progress (%)", "Accuracy", fn, (0, 1),
    )


def plot_merged_gp_experienced_accuracy() -> None:
    _plot_merged_gp_target_accuracy("experienced", "Experienced Targets", "merged_gp_experienced_accuracy")


def plot_merged_gp_alt_targets_accuracy() -> None:
    _plot_merged_gp_target_accuracy("alt_targets", "Alt Targets", "merged_gp_alt_targets_accuracy")


def plot_merged_gp_baseline_accuracy() -> None:
    s = load_series(f"{config.eval_dir}/merged_gamedata_10_progress_baseline_accuracy.pkl")
    plot_shaded_error_groups(
        "Baseline Accuracy by Game Progress (merged)",
        [
            [
                ("Value Accuracy",     "±95% CI", s.x, *_ci_keys(s, "Value Accuracy")),
                ("Value Accuracy ND",  "±95% CI", s.x, *_ci_keys(s, "Value Accuracy ND")),
            ],
            [
                ("Policy Accuracy",    "±95% CI", s.x, *_ci_keys(s, "Policy Accuracy")),
                ("Policy Accuracy NT", "±95% CI", s.x, *_ci_keys(s, "Policy Accuracy NT")),
            ],
        ],
        "Game Progress (%)", "Accuracy", "merged_gp_baseline_accuracy", y_lim=(0, 1),
    )


def plot_merged_gp_branching_factor() -> None:
    s = load_series(f"{config.eval_dir}/merged_gamedata_10_progress_branching_factor.pkl")
    plot_shaded_error(
        "Average Branching Factor by Game Progress (merged)",
        [("Branching Factor", "±95% CI", s.x, *_ci_keys(s, "Branching Factor"))],
        "Game Progress (%)", "Branching Factor", "merged_gp_branching_factor",
    )


def plot_merged_heatmap_alt_targets() -> None:
    _plot_accuracy_heatmaps("alt_targets", "Training targets", merged=True)


def plot_merged_heatmap_experienced() -> None:
    _plot_accuracy_heatmaps("experienced", "Experienced (MC) outcomes", merged=True)


def plot_merged_model_heatmaps() -> None:
    _plot_model_heatmaps(merged=True)


def plot_merged_distance_class_balance() -> None:
    s = load_series(f"{config.eval_dir}/merged_distance_class_balance.pkl")
    plot_shaded_error(
        "Class balance of the distance-from-terminal eval buckets (merged)",
        [
            ("Majority class rate (= chance)", "±95% CI", s.x, *_ci_keys(s, "Majority Class Rate")),
            ("Win rate", "±95% CI", s.x, *_ci_keys(s, "Win Rate")),
        ],
        "Distance from terminal (plies)", "Fraction of bucket",
        "merged_model_distance_class_balance",
        y_lim=(0.0, 1.0),
    )


def _pool_across_trials(
    series_fn: str, metric: str, label: str,
    count_metric: str | None = None, min_count: int | None = None,
) -> tuple[list[int], list[float], list[float]]:
    '''
    Pool each trial along x, then take mean ± 95% CI ACROSS trials per bin.

    Deliberately not computed from the merged series: pooling that would average
    already-averaged numbers and there would be no spread left to put a band on.
    Pooling per trial first keeps one independent value per seed per bin, which
    is what a cross-seed CI has to be built from. Same t-based formula as
    merge_series, so the bands mean the same thing as everywhere else.

    Bins are matched by VALUE, not position — heatmap_matrix trims trailing empty
    rows, so a short-game seed yields fewer rows than a long-game one.
    '''
    per_bin: dict[int, list[float]] = {}
    for d in _get_trial_dirs():
        s = load_series(f"{d}{series_fn}", optional=True)
        if s is None:
            continue
        try:
            bins, pooled, _ = pool_heatmap_over_x(
                s, metric, label, count_metric=count_metric, min_count=min_count)
        except KeyError:
            continue
        for b, v in zip(bins, pooled):
            if np.isfinite(v):
                per_bin.setdefault(b, []).append(float(v))
    if not per_bin:
        raise FileNotFoundError(f"no trial produced {series_fn} [{metric} {label}]")

    xs = sorted(per_bin)
    means, cis = [], []
    for b in xs:
        vals = np.array(per_bin[b], dtype=np.float64)
        n = vals.size
        means.append(float(vals.mean()))
        if n >= 2:
            t = stats.t.ppf(0.975, df=n - 1)
            cis.append(float(t * vals.std(ddof=0) / math.sqrt(n)))
        else:
            cis.append(float("nan"))
    return xs, means, cis


def plot_merged_target_accuracy_pooled() -> None:
    '''(1, merged) Both target types, pooled over iterations, ±95% CI across seeds.'''
    for axis, label, x_label in AXES:
        lines = []
        for name, disp in (("alt_targets", "Training targets"),
                           ("experienced", "Experienced (MC) outcomes")):
            xs, means, cis = _pool_across_trials(
                f"{name}_iter_{axis}_acc.pkl", "Value Accuracy ND", label,
                min_count=config.heatmap_min_cell_count)
            lines.append((disp, "±95% CI", xs, means, cis))
        plot_shaded_error(
            "Training-target value accuracy vs GT, no draws (merged) "
            "- pooled over all iterations",
            lines, x_label, "Value accuracy (no draws)",
            f"merged_target_accuracy_pooled_{axis}",
            y_lim=(0.0, 1.0),
        )


def plot_merged_model_accuracy_pooled() -> None:
    '''(3, merged) Model accuracy pooled over checkpoints, ±95% CI across seeds.'''
    for name, label, x_label, metric, disp in MODEL_SPECS:
        xs, means, cis = _pool_across_trials(
            f"model_heatmap_{name}.pkl", metric, label, count_metric="Count",
            min_count=config.heatmap_min_cell_count)
        plot_shaded_error(
            f"{disp} (merged) - pooled over all checkpoints",
            [(metric, "±95% CI", xs, means, cis)],
            x_label, metric,
            f"merged_model_accuracy_pooled_{name}",
            y_lim=(0.0, 1.0),
        )


def plot_merged_target_accuracy_lines() -> None:
    '''
    (2, merged) One line per iteration, each line the cross-seed mean.

    No CI band here, unlike the pooled plots: this figure already carries ~200
    lines, and 200 shaded bands would be an unreadable wash. The per-cell
    cross-seed CI is exactly what plot_merged_heatmap_seed_agreement shows, and
    the pooled figures carry bands where they can actually be read.
    '''
    _plot_target_accuracy_lines("alt_targets", "Training targets", merged=True)
    _plot_target_accuracy_lines("experienced", "Experienced (MC) outcomes", merged=True)


def plot_merged_model_accuracy_lines() -> None:
    '''(3, merged) One line per checkpoint, each line the cross-seed mean.'''
    _plot_model_accuracy_lines(merged=True)


def plot_merged_gt_win_rate_parity() -> None:
    '''(4, merged) GT win rate by parity, ±95% CI across seeds.'''
    s = load_series(f"{config.eval_dir}/merged_distance_class_balance.pkl")
    mean, ci = _ci_keys(s, "Win Rate")
    groups = []
    for parity, disp in ((0, "win rate, even D"), (1, "win rate, odd D")):
        sel = [(d, m, c) for d, m, c in zip(s.x, mean, ci) if d % 2 == parity]
        groups.append((disp, "±95% CI", [d for d, _, _ in sel],
                       [m for _, m, _ in sel], [c for _, _, c in sel]))
    plot_shaded_error(
        "GT win rate of the player to move, by distance from terminal (merged)",
        groups,
        "Distance from terminal (plies)", "GT win rate (player to move)",
        "merged_gt_win_rate_by_distance_parity",
        y_lim=(0.0, 1.0),
    )


def plot_merged_target_gt_win_rate() -> None:
    '''(4b, merged) Training-target GT win rate, ±95% CI across seeds.'''
    for axis, label, x_label in AXES:
        # alt_targets only, and parity-split on distance only — see the reasoning
        # on a0.eval3.plotting._plot_target_gt_win_rate
        xs, means, cis = _pool_across_trials(
            f"alt_targets_iter_{axis}_acc.pkl", "GT Win Rate ND", label,
            count_metric="Count ND", min_count=config.heatmap_min_cell_count)
        if axis == "distance":
            groups = []
            for parity, tag in ((0, "even D"), (1, "odd D")):
                sel = [(x, m, c) for x, m, c in zip(xs, means, cis) if x % 2 == parity]
                groups.append((f"win rate, {tag}", "±95% CI",
                               [x for x, _, _ in sel], [m for _, m, _ in sel],
                               [c for _, _, c in sel]))
        else:
            groups = [("win rate", "±95% CI", xs, means, cis)]
        plot_shaded_error(
            "GT win rate of the player to move, training-target states (merged) "
            "- pooled over all iterations",
            groups, x_label, "GT win rate (player to move)",
            f"merged_target_gt_win_rate_{axis}",
            y_lim=(0.0, 1.0),
        )


def plot_merged_heatmap_seed_agreement() -> None:
    '''
    Across-seed 95% CI half-width per cell. merge_series emits a "<key>_ci"
    alongside every merged key, so this is free — and on a heatmap it answers the
    question the mean cannot: which parts of the picture actually reproduce
    across seeds, and which are one seed's noise.
    '''
    for name, label_name in (("alt_targets", "Training targets"),
                             ("experienced", "Experienced (MC) outcomes")):
        for axis, axis_label, y_label in (
            ("distance", "D", "Distance from terminal (plies)"),
            ("progress", "P", "Game progress (%)"),
        ):
            s = load_series(f"{config.eval_dir}/merged_{name}_iter_{axis}_acc.pkl")
            plot_heatmap(
                f"{label_name} value accuracy: across-seed 95% CI half-width",
                s, "Value Accuracy ND", axis_label,
                "Training iteration", y_label,
                f"merged_heatmap_{name}_{axis}_value_accuracy_nd_ci",
                cbar_label="CI half-width (lower = more reproducible)",
                key_suffix="_ci",
            )


def plot_merged_gp_state_count() -> None:
    s = load_series(f"{config.eval_dir}/merged_gamedata_10_progress_count.pkl")
    plot_bar_with_error(
        "Number of States in Each Game Progress Bucket (merged)",
        ("Count", s.x, *_ci_keys(s, "Count")),
        "Game Progress (%)", "Count", "merged_gp_state_count",
    )


def plot_merged_gp_unique_state_count() -> None:
    s = load_series(f"{config.eval_dir}/merged_gamedata_10_progress_count.pkl")
    plot_bar_with_error(
        "Number of Unique States in Each Game Progress Bucket (merged)",
        ("Unique", s.x, *_ci_keys(s, "Unique")),
        "Game Progress (%)", "Unique Count", "merged_gp_unique_state_count",
    )


def _plot_merged_dataset_eval(name: str, title: str) -> None:
    s = load_series(f"{config.eval_dir}/merged_{name}_eval.pkl")
    plot_shaded_error(
        title,
        [
            ("Loss",            "±95% CI", s.x, *_ci_keys(s, "loss")),
            ("Value Loss",      "±95% CI", s.x, *_ci_keys(s, "value_loss")),
            ("Policy Loss",     "±95% CI", s.x, *_ci_keys(s, "policy_loss")),
            ("Value Accuracy",  "±95% CI", s.x, *_ci_keys(s, "value_accuracy")),
            ("Policy Accuracy", "±95% CI", s.x, *_ci_keys(s, "policy_accuracy")),
        ],
        "Iteration", "Performance", f"merged_{name}_eval",
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
    iteration = load_series(f"{config.eval_dir}/merged_gamedata_alt_acc.pkl")
    overall   = load_series(f"{config.eval_dir}/merged_gamedata_alt_overall_acc.pkl")
    plot_shaded_error(
        "Value Head Performance on Ground Truth and Training Data (merged)",
        [
            ("Seen",                         "±95% CI", seen.x,      *_ci_keys(seen,      "value_accuracy")),
            ("Neighbor 1",                   "±95% CI", n1.x,        *_ci_keys(n1,        "value_accuracy")),
            ("Neighbor 2",                   "±95% CI", n2.x,        *_ci_keys(n2,        "value_accuracy")),
            ("Random",                       "±95% CI", random_.x,   *_ci_keys(random_,   "value_accuracy")),
            ("Training Data (Alt)",          "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Value Accuracy")),
            ("Training Data ND (Alt)",       "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Value Accuracy ND")),
            ("Overall Training Data (Alt)",  "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Value Accuracy")),
            ("Overall Training Data ND (Alt)","±95% CI", overall.x,  *_ci_keys(overall,   "Overall Value Accuracy ND")),
        ],
        "Iteration", "Accuracy", "merged_full_value_accuracy", (0, 1),
    )


def plot_merged_full_policy_accuracy() -> None:
    seen_nt   = load_series(f"{config.eval_dir}/merged_seen_nt_eval.pkl")
    random_nt = load_series(f"{config.eval_dir}/merged_random_nt_eval.pkl")
    n1_nt     = load_series(f"{config.eval_dir}/merged_neighbor_1_nt_eval.pkl")
    n2_nt     = load_series(f"{config.eval_dir}/merged_neighbor_2_nt_eval.pkl")
    iteration = load_series(f"{config.eval_dir}/merged_gamedata_alt_acc.pkl")
    overall   = load_series(f"{config.eval_dir}/merged_gamedata_alt_overall_acc.pkl")
    plot_shaded_error(
        "Policy Head Performance on Ground Truth (NT) and Training Data (merged)",
        [
            ("Seen NT",                        "±95% CI", seen_nt.x,   *_ci_keys(seen_nt,   "policy_accuracy")),
            ("Neighbor 1 NT",                  "±95% CI", n1_nt.x,     *_ci_keys(n1_nt,     "policy_accuracy")),
            ("Neighbor 2 NT",                  "±95% CI", n2_nt.x,     *_ci_keys(n2_nt,     "policy_accuracy")),
            ("Random NT",                      "±95% CI", random_nt.x, *_ci_keys(random_nt, "policy_accuracy")),
            ("Training Data NT (Alt)",         "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Policy Accuracy NT")),
            ("Overall Training Data NT (Alt)", "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Policy Accuracy NT")),
        ],
        "Iteration", "Accuracy", "merged_full_policy_accuracy", (0, 1),
    )


def plot_merged_full_policy_entropy() -> None:
    seen_nt   = load_series(f"{config.eval_dir}/merged_seen_nt_eval.pkl")
    random_nt = load_series(f"{config.eval_dir}/merged_random_nt_eval.pkl")
    n1_nt     = load_series(f"{config.eval_dir}/merged_neighbor_1_nt_eval.pkl")
    n2_nt     = load_series(f"{config.eval_dir}/merged_neighbor_2_nt_eval.pkl")
    iteration = load_series(f"{config.eval_dir}/merged_gamedata_alt_acc.pkl")
    overall   = load_series(f"{config.eval_dir}/merged_gamedata_alt_overall_acc.pkl")
    plot_shaded_error(
        "Policy Head Entropy on Ground Truth (NT) and Training Data (normalized, merged)",
        [
            ("Seen NT",                        "±95% CI", seen_nt.x,   *_ci_keys(seen_nt,   "policy_entropy")),
            ("Neighbor 1 NT",                  "±95% CI", n1_nt.x,     *_ci_keys(n1_nt,     "policy_entropy")),
            ("Neighbor 2 NT",                  "±95% CI", n2_nt.x,     *_ci_keys(n2_nt,     "policy_entropy")),
            ("Random NT",                      "±95% CI", random_nt.x, *_ci_keys(random_nt, "policy_entropy")),
            ("Training Data NT (Alt)",         "±95% CI", iteration.x, *_ci_keys(iteration, "Iteration Policy Entropy NT")),
            ("Overall Training Data NT (Alt)", "±95% CI", overall.x,   *_ci_keys(overall,   "Overall Policy Entropy NT")),
        ],
        "Iteration", "Normalized Entropy", "merged_full_policy_entropy", (0, 1),
    )


# === merged player evaluation ===

def _rate_ci_keys(s: Series, key: str, denom_key: str) -> tuple[list[float], list[float]]:
    '''Return (rate, rate_ci) where rate = key / denom_key on a merged series.
    num_games is constant per trial, so dividing the merged mean/CI of `key`
    by the merged mean of `denom_key` gives the cross-trial rate and its CI.'''
    rates = [v / d if d > 0 else 0.0 for v, d in zip(s.ys[key], s.ys[denom_key])]
    cis   = [c / d if d > 0 else 0.0 for c, d in zip(s.ys[f"{key}_ci"], s.ys[denom_key])]
    return rates, cis


def plot_merged_ev() -> None:
    player = load_series(f"{config.eval_dir}/merged_player_evaluation_results.pkl")
    # References are evaluated/merged after the players, so an early-timed-out
    # eval can leave the reference series missing. Still plot the model EV in
    # that case, just without the baseline overlays.
    ref    = load_series(f"{config.eval_dir}/merged_reference_evaluation_results.pkl", optional=True)
    x = player.x
    n = len(x)
    groups = [
        [
            ("Model (P1)", "±95% CI", x, *_ci_keys(player, "ev_p1")),
            ("Model (P2)", "±95% CI", x, *_ci_keys(player, "ev_p2")),
        ],
    ]
    for name in (_reference_names(ref) if ref is not None else []):
        ev_p1, ci_p1 = _ci_keys(ref, f"{name}_ev_p1")
        ev_p2, ci_p2 = _ci_keys(ref, f"{name}_ev_p2")
        groups.append([
            (f"{name} (P1, EV={ev_p1[0]:.3f})", "±95% CI", x, [ev_p1[0]] * n, [ci_p1[0]] * n),
            (f"{name} (P2, EV={ev_p2[0]:.3f})", "±95% CI", x, [ev_p2[0]] * n, [ci_p2[0]] * n),
        ])
    plot_shaded_error_groups(
        "Expected Value vs Baseline (merged)",
        groups,
        "Training Iteration", "Expected Value",
        "merged_player_ev", y_lim=(-1.0, 1.0),
    )


def _plot_merged_wld_stacked(side: str, title: str, fn: str) -> None:
    s = load_series(f"{config.eval_dir}/merged_player_evaluation_results.pkl")
    plot_stacked(
        title, s.x,
        [
            ("Wins",            s.ys[f"wins_{side}"]),
            ("Losses",          s.ys[f"losses_{side}"]),
            ("Draws (repeat)",  s.ys[f"draws_repeat_{side}"]),
            ("Draws (timeout)", s.ys[f"draws_timeout_{side}"]),
        ],
        "Training Iteration", "Games", fn,
    )


def plot_merged_wld_p1() -> None:
    _plot_merged_wld_stacked("p1", "Model as P1: W/L/D vs Baseline (merged)", "merged_player_wld_p1")


def plot_merged_wld_p2() -> None:
    _plot_merged_wld_stacked("p2", "Model as P2: W/L/D vs Baseline (merged)", "merged_player_wld_p2")


def plot_merged_wld() -> None:
    s = load_series(f"{config.eval_dir}/merged_player_evaluation_results.pkl")
    x = s.x
    plot_shaded_error_groups(
        "Model W/L/D vs Baseline (merged)",
        [
            [("Wins (P1)",          "±95% CI", x, *_rate_ci_keys(s, "wins_p1",          "num_games_p1")),
             ("Wins (P2)",          "±95% CI", x, *_rate_ci_keys(s, "wins_p2",          "num_games_p2"))],
            [("Losses (P1)",        "±95% CI", x, *_rate_ci_keys(s, "losses_p1",        "num_games_p1")),
             ("Losses (P2)",        "±95% CI", x, *_rate_ci_keys(s, "losses_p2",        "num_games_p2"))],
            [("Draws repeat (P1)",  "±95% CI", x, *_rate_ci_keys(s, "draws_repeat_p1",  "num_games_p1")),
             ("Draws repeat (P2)",  "±95% CI", x, *_rate_ci_keys(s, "draws_repeat_p2",  "num_games_p2"))],
            [("Draws timeout (P1)", "±95% CI", x, *_rate_ci_keys(s, "draws_timeout_p1", "num_games_p1")),
             ("Draws timeout (P2)", "±95% CI", x, *_rate_ci_keys(s, "draws_timeout_p2", "num_games_p2"))],
        ],
        "Training Iteration", "Rate", "merged_player_wld",
    )


def main():
    setup_logging(level=20, log_dir=config.log_dir, process_name="combining")
    logger.info("combining...")

    merge_all_series()
    merge_all_distribution_series()
    merge_baseline_accuracies()

    # distribution series
    safeplot(plot_merged_dataset_pre_balance_distributions)
    safeplot(plot_merged_dataset_post_balance_distributions)
    safeplot(plot_merged_random_nd_value_distributions)
    safeplot(plot_merged_random_value_distributions)

    # value-derived policy entropy
    safeplot(plot_merged_value_policy_entropy)

    # training metrics
    safeplot(plot_merged_training_metrics)
    safeplot(plot_merged_training_metrics_with_losses)

    # gamedata stats
    safeplot(plot_merged_gamedata_total_games)
    safeplot(plot_merged_gamedata_outcomes_stacked)
    safeplot(plot_merged_gamedata_outcomes_stacked_proportional)
    safeplot(plot_merged_gamedata_outcomes_lines)
    safeplot(plot_merged_gamedata_game_length)
    safeplot(plot_merged_gamedata_game_time)

    # target-refresh analyses
    safeplot(plot_merged_refresh_targets_acc)
    safeplot(plot_merged_refresh_targets_acc_nd)
    safeplot(plot_merged_refresh_target_delta)
    safeplot(plot_merged_refresh_delta_by_distance)
    safeplot(plot_merged_per_refresh_dataset_distributions)

    # gamedata accuracy & bias (experienced + alt targets)
    safeplot(plot_merged_gamedata_accuracy)
    safeplot(plot_merged_gamedata_alt_accuracy)
    safeplot(plot_merged_gamedata_bias)
    safeplot(plot_merged_gamedata_alt_bias)
    safeplot(plot_merged_gamedata_bpp)
    safeplot(plot_merged_gamedata_bppma)
    safeplot(plot_merged_gamedata_alt_bpp)
    safeplot(plot_merged_gamedata_alt_bppma)

    # game progress
    safeplot(plot_merged_gp_state_count)
    safeplot(plot_merged_gp_unique_state_count)
    safeplot(plot_merged_gp_experienced_accuracy)
    safeplot(plot_merged_gp_alt_targets_accuracy)
    safeplot(plot_merged_gp_baseline_accuracy)
    safeplot(plot_merged_gp_branching_factor)

    # accuracy heatmaps (iteration x progress / distance-from-terminal)
    safeplot(plot_merged_heatmap_alt_targets)
    safeplot(plot_merged_heatmap_experienced)
    safeplot(plot_merged_heatmap_seed_agreement)

    # model-accuracy heatmaps
    safeplot(plot_merged_model_heatmaps)
    safeplot(plot_merged_distance_class_balance)

    # line-chart readings of the same series
    safeplot(plot_merged_target_accuracy_pooled)
    safeplot(plot_merged_target_accuracy_lines)
    safeplot(plot_merged_model_accuracy_pooled)
    safeplot(plot_merged_model_accuracy_lines)
    safeplot(plot_merged_gt_win_rate_parity)
    safeplot(plot_merged_target_gt_win_rate)

    # per-dataset model evaluation
    safeplot(plot_merged_seen_nd_eval)
    safeplot(plot_merged_random_nd_eval)
    safeplot(plot_merged_seen_nt_eval)
    safeplot(plot_merged_random_nt_eval)
    safeplot(plot_merged_neighbor_nd_evals)
    safeplot(plot_merged_neighbor_nt_evals)
    safeplot(plot_merged_experienced_dataset_eval)
    safeplot(plot_merged_alt_targets_dataset_eval)

    # per-bucket dataset evaluation
    safeplot(plot_merged_game_progress_10_nd_eval)
    safeplot(plot_merged_game_progress_10_nt_eval)

    # combined plots
    safeplot(plot_merged_full_value_accuracy)
    safeplot(plot_merged_full_policy_accuracy)
    safeplot(plot_merged_full_policy_entropy)
    safeplot(plot_merged_value_accuracies)
    safeplot(plot_merged_policy_accuracies)

    # player evaluation
    safeplot(plot_merged_ev)
    safeplot(plot_merged_wld_p1)
    safeplot(plot_merged_wld_p2)
    safeplot(plot_merged_wld)


if __name__ == "__main__":
    main()
