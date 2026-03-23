'''
Combine bias sweep trial results with confidence intervals.

Usage:
    # combine all 8 trials for all bias levels
    python -m scripts.2026-03-23_bias_sweep_combine config_file.json

    # combine specific trials for specific bias levels
    python -m scripts.2026-03-23_bias_sweep_combine config_file.json --trials 1 2 3 4 5 --win-pcts 50 60 70 80 90
'''
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

METRIC_KEYS = [
    'accuracy', 'pred_win_pct', 'mean_pred', 'mean_pred_win',
    'mean_pred_loss', 'win_recall', 'loss_recall', 'mean_abs_pred', 'std_pred',
]


def load_eval_data(win_pct_int: int, trial: int) -> list[dict] | None:
    """Load a single trial's eval data. Returns None if file not found."""
    path = f"{config.stats_dir}/bias_{win_pct_int}_trial_{trial}_eval.pkl"
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logger.warning(f"Not found: {path}")
        return None


def combine_trials(
    win_pct_int: int,
    trials: list[int],
    confidence: float = 0.95,
) -> dict:
    """
    Load and combine eval data across trials for a given bias level.
    Returns dict with per-epoch mean and CI for each metric.
    """
    # load all trials
    all_evals = []
    for trial in trials:
        data = load_eval_data(win_pct_int, trial)
        if data is not None:
            all_evals.append(data)

    if not all_evals:
        logger.error(f"No data found for bias {win_pct_int}%")
        return {}

    n_trials = len(all_evals)
    n_epochs = len(all_evals[0])  # includes epoch 0 (pre-training)
    logger.info(f"Bias {win_pct_int}%: loaded {n_trials} trials, {n_epochs} epochs each")

    result: dict = {'epochs': list(range(n_epochs)), 'n_trials': n_trials}

    for key in METRIC_KEYS:
        # shape: (n_trials, n_epochs)
        values = np.array([[epoch[key] for epoch in trial_data] for trial_data in all_evals])
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0, ddof=1) if n_trials > 1 else np.zeros(n_epochs)

        if n_trials > 1:
            t_crit = stats.t.ppf((1 + confidence) / 2, df=n_trials - 1)
            ci = t_crit * std / np.sqrt(n_trials)
        else:
            ci = np.zeros(n_epochs)

        result[f'{key}_mean'] = mean
        result[f'{key}_std'] = std
        result[f'{key}_ci'] = ci

    return result


def plot_combined_over_epochs(
    combined: dict[int, dict],
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    ylim: tuple[float, float] | None = None,
):
    """Plot a single metric over epochs for all bias levels, with CI shading."""
    plt.figure(figsize=(14, 8))
    cmap = plt.cm.RdYlGn_r  # red for high bias, green for low
    bias_levels = sorted(combined.keys())
    n = len(bias_levels)

    for i, wp in enumerate(bias_levels):
        data = combined[wp]
        epochs = data['epochs']
        mean = data[f'{metric}_mean']
        ci = data[f'{metric}_ci']
        color = cmap(i / max(n - 1, 1))

        plt.plot(epochs, mean, '-o', color=color, label=f'{wp}%', markersize=3)
        plt.fill_between(epochs, mean - ci, mean + ci, alpha=0.15, color=color)

    if ylim:
        plt.ylim(*ylim)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(f"{title} (±95% CI, n={data['n_trials']})")
    plt.legend(title='Win %', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/{filename}.png")
    plt.close()


def plot_final_vs_bias(
    combined: dict[int, dict],
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    ylim: tuple[float, float] | None = None,
):
    """Plot final-epoch metric vs bias level, with CI error bars."""
    bias_levels = sorted(combined.keys())
    means = [combined[wp][f'{metric}_mean'][-1] for wp in bias_levels]
    cis = [combined[wp][f'{metric}_ci'][-1] for wp in bias_levels]

    plt.figure(figsize=(10, 6))
    plt.errorbar(bias_levels, means, yerr=cis, fmt='-o', capsize=4, capthick=1.5)
    if ylim:
        plt.ylim(*ylim)
    plt.xlabel('Training Win %')
    plt.ylabel(ylabel)
    n = combined[bias_levels[0]]['n_trials']
    plt.title(f"{title} (±95% CI, n={n})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/{filename}.png")
    plt.close()


def plot_all(combined: dict[int, dict]):
    """Generate all combined plots."""
    # --- over epochs (one line per bias level) ---
    plot_combined_over_epochs(
        combined, 'accuracy', 'Accuracy Over Training',
        'Accuracy', 'bias_combined_accuracy', ylim=(0, 1),
    )
    plot_combined_over_epochs(
        combined, 'win_recall', 'Win Recall Over Training',
        'Win Recall', 'bias_combined_win_recall', ylim=(0, 1),
    )
    plot_combined_over_epochs(
        combined, 'loss_recall', 'Loss Recall Over Training',
        'Loss Recall', 'bias_combined_loss_recall', ylim=(0, 1),
    )
    plot_combined_over_epochs(
        combined, 'pred_win_pct', 'Prediction Win % Over Training',
        'Fraction Predicted Win', 'bias_combined_pred_win_pct', ylim=(0, 1),
    )
    plot_combined_over_epochs(
        combined, 'mean_pred', 'Mean Predicted Value Over Training',
        'Mean Predicted Value', 'bias_combined_mean_pred',
    )
    plot_combined_over_epochs(
        combined, 'mean_pred_win', 'Mean Prediction (GT=Win) Over Training',
        'Mean Predicted Value', 'bias_combined_mean_pred_win',
    )
    plot_combined_over_epochs(
        combined, 'mean_pred_loss', 'Mean Prediction (GT=Loss) Over Training',
        'Mean Predicted Value', 'bias_combined_mean_pred_loss',
    )
    plot_combined_over_epochs(
        combined, 'mean_abs_pred', 'Confidence (Mean |pred|) Over Training',
        'Mean |Predicted Value|', 'bias_combined_mean_abs_pred',
    )

    # --- final epoch vs bias level ---
    plot_final_vs_bias(
        combined, 'accuracy', 'Final Accuracy vs Training Bias',
        'Accuracy', 'bias_final_accuracy', ylim=(0, 1),
    )
    plot_final_vs_bias(
        combined, 'win_recall', 'Final Win Recall vs Training Bias',
        'Win Recall', 'bias_final_win_recall', ylim=(0, 1),
    )
    plot_final_vs_bias(
        combined, 'loss_recall', 'Final Loss Recall vs Training Bias',
        'Loss Recall', 'bias_final_loss_recall', ylim=(0, 1),
    )
    plot_final_vs_bias(
        combined, 'pred_win_pct', 'Final Prediction Win % vs Training Bias',
        'Fraction Predicted Win', 'bias_final_pred_win_pct', ylim=(0, 1),
    )
    plot_final_vs_bias(
        combined, 'mean_pred', 'Final Mean Prediction vs Training Bias',
        'Mean Predicted Value', 'bias_final_mean_pred',
    )


if __name__ == "__main__":
    setup_logging(level=20, log_dir=config.log_dir, process_name="bias_combine")

    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='config/config.json')
    parser.add_argument('--trials', type=int, nargs='+', default=list(range(1, 9)))
    parser.add_argument('--win-pcts', type=int, nargs='+',
                        default=[50, 55, 60, 65, 70, 75, 80, 85, 90, 95])
    args = parser.parse_args()

    logger.info(f"Combining trials {args.trials} for bias levels {args.win_pcts}%")

    combined: dict[int, dict] = {}
    for wp in args.win_pcts:
        result = combine_trials(wp, args.trials)
        if result:
            combined[wp] = result

    if not combined:
        logger.error("No data to plot")
    else:
        plot_all(combined)
        logger.info(f"Plots saved to {config.plot_dir}")
