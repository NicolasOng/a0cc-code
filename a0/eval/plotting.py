import sys

import pickle
import matplotlib.pyplot as plt
import numpy as np

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

class Series:
    x: list[int]
    ys: dict[str, list[float]]

    def __init__(self, ys: list[str] | None = None):
        self.x = []
        self.ys = {}
        if ys is not None:
            for y in ys:
                self.ys[y] = []

def save_series(series: Series, series_path: str) -> None:
    '''
    Saves a Series object to the given path.
    '''
    logger.info(f"Saving series to {series_path}...")
    with open(series_path, 'wb') as f:
        pickle.dump(series, f)

def load_series(series_path: str) -> Series:
    '''
    loads a series object from a given path
    '''
    logger.info(f"Loading series from {series_path}...")
    try:
        with open(series_path, 'rb') as f:
            series: Series = pickle.load(f)
        logger.info(f"Loaded series from {series_path}.")
    except FileNotFoundError:
        logger.error(f"Series file not found at {series_path}. Please generate the series first.")
        sys.exit()
    except Exception as e:
        logger.error(f"Error loading series: {e}")
        sys.exit()
    return series

def plot_given(title: str, series: list[tuple[str, list[int], list[float]]], x_label: str, y_label: str, fn: str) -> None:
    plt.figure(figsize=(16, 9))
    for label, x_values, y_values in series:
        plt.plot(x_values, y_values, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}{fn}.png")
    plt.clf()

def plot_stacked(title: str, x: list[int], series: list[tuple[str, list[float]]], x_label: str, y_label: str, fn: str) -> None:
    plt.figure(figsize=(16, 9))
    plt.stackplot(x, *[y for _, y in series],
        labels=[l for l, _ in series],
    )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/{fn}.png")
    plt.clf()

def plot_stacked_proportional(title: str, x: list[int], total: list[float], series: list[tuple[str, list[float]]], x_label: str, y_label: str, fn: str) -> None:
    plt.figure(figsize=(16, 9))
    plt.stackplot(
        x,
        *[np.array(y) / np.array(total) for _, y in series],
        labels=[l for l, _ in series],
    )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/{fn}.png")
    plt.clf()

def plot_std_error(title: str, x: list[int], avg: list[float], std: list[float], x_label: str, y_label: str, fn: str):
    plt.figure(figsize=(16, 9))
    plt.errorbar(x, avg, yerr=std, 
                 marker='o', capsize=5, capthick=1, linewidth=1)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/{fn}.png")
    plt.clf()

def plot_shaded_error(title: str, series: list[tuple[str, str, list[int], list[float], list[float]]], x_label: str, y_label: str, fn: str):
    plt.figure(figsize=(16, 9))
    for line_label, fill_label, x_values, y_values, fill_values in series:
        plt.plot(x_values, y_values, label=line_label)
        plt.fill_between(x_values, np.array(y_values) - np.array(fill_values), np.array(y_values) + np.array(fill_values), 
                        alpha=0.3, label=fill_label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/{fn}.png")
    plt.clf()

def main():
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="plotting"
    )
    logger.info("plotting...")

    # load all the series
    train_gt_series = load_series(f"{config.eval_dir}/training_gtv_eval.pkl")
    random_gt_series = load_series(f"{config.eval_dir}/random_gtv_eval.pkl")
    train_ev_series = load_series(f"{config.eval_dir}/training_ev_eval.pkl")
    n_neighbor_gt_series: list[Series] = []
    for i in range(2):
        neighbor_gt_series = load_series(f"{config.eval_dir}/neighbor_{i+1}_gtv_eval.pkl")
        n_neighbor_gt_series.append(neighbor_gt_series)
    
    gamedata_series = load_series(f"{config.eval_dir}/gamedata_stats.pkl")
    gd_accuracy_series = load_series(f"{config.eval_dir}/gamedata_acc.pkl")
    gd_overall_accuracy_series = load_series(f"{config.eval_dir}/gamedata_overall_acc.pkl")
    training_metrics = load_series(f"{config.eval_dir}/training_metrics.pkl")
    gd_bias_series = load_series(f"{config.eval_dir}/gamedata_bias.pkl")
    gd_overall_bias_series = load_series(f"{config.eval_dir}/gamedata_overall_bias.pkl")

    gd_prog_acc_100 = load_series(f"{config.eval_dir}/gamedata_progress_acc_100.pkl")
    gd_prog_acc_10 = load_series(f"{config.eval_dir}/gamedata_progress_acc_10.pkl")
    state_progress_gtv = load_series(f"{config.eval_dir}/state_progress_gtv_datasets_eval.pkl")

    # plot all the series
    plot_given("Model Performance on Ground Truth of States Seen During Training",
               [
                   ("Value Accuracy", train_gt_series.x, train_gt_series.ys["value_accuracy"]),
                   ("Policy Accuracy", train_gt_series.x, train_gt_series.ys["policy_accuracy"])
                ],
               "Iteration", "Accuracy", "training_gtv_eval")
    
    
    plot_given("Model Performance on Ground Truth of Random States",
               [
                   ("Value Accuracy", random_gt_series.x, random_gt_series.ys["value_accuracy"]),
                   ("Policy Accuracy", random_gt_series.x, random_gt_series.ys["policy_accuracy"])
               ],
               "Iteration", "Accuracy", "random_gtv_eval")

    
    plot_given("Model Performance on Experienced Outcomes of States Seen During Training",
               [
                   ("Value Accuracy", train_ev_series.x, train_ev_series.ys["value_accuracy"]),
                   ("Policy Accuracy", train_ev_series.x, train_ev_series.ys["policy_accuracy"])
               ],
               "Iteration", "Accuracy", "training_ev_eval")

    for i in range(2):
        plot_given(
            f"Model Performance on Ground Truth of Neighboring States {i + 1}",
            [
                ("Value Accuracy", n_neighbor_gt_series[i].x, n_neighbor_gt_series[i].ys["value_accuracy"]),
                ("Policy Accuracy", n_neighbor_gt_series[i].x, n_neighbor_gt_series[i].ys["policy_accuracy"])
            ],
            "Iteration", "Accuracy", f"neighbor_{i+1}_gtv_eval"
        )
    
    plot_given("Value Head Model Performance on Ground Truth of States and Training Data Accuracy",
               [
                    ("Seen Accuracy", train_gt_series.x, train_gt_series.ys["value_accuracy"]),
                    ("Neighbor 1 Accuracy", n_neighbor_gt_series[0].x, n_neighbor_gt_series[0].ys["value_accuracy"]),
                    ("Neighbor 2 Accuracy", n_neighbor_gt_series[1].x, n_neighbor_gt_series[1].ys["value_accuracy"]),
                    ("Random Accuracy", random_gt_series.x, random_gt_series.ys["value_accuracy"]),
                    ("Training Data Accuracy", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Value Accuracy"]),
                    ("Overall Training Data Accuracy", gd_overall_accuracy_series.x, gd_overall_accuracy_series.ys["Overall Value Accuracy"])
               ],
               "Iteration", "Accuracy", "full_accuracy_value")

    plot_given("Policy Head Model Performance on Ground Truth of States and Training Data Accuracy",
               [
                    ("Seen Accuracy", train_gt_series.x, train_gt_series.ys["policy_accuracy"]),
                    ("Neighbor 1 Accuracy", n_neighbor_gt_series[0].x, n_neighbor_gt_series[0].ys["policy_accuracy"]),
                    ("Neighbor 2 Accuracy", n_neighbor_gt_series[1].x, n_neighbor_gt_series[1].ys["policy_accuracy"]),
                    ("Random Accuracy", random_gt_series.x, random_gt_series.ys["policy_accuracy"]),
                    ("Training Data Accuracy", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Policy Accuracy"]),
                    ("Training Data PM", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Policy PM"]),
                    ("Overall Training Data Accuracy", gd_overall_accuracy_series.x, gd_overall_accuracy_series.ys["Overall Policy Accuracy"]),
                    ("Overall Training Data PM", gd_overall_accuracy_series.x, gd_overall_accuracy_series.ys["Overall Policy PM"])
               ],
               "Iteration", "Accuracy", "full_accuracy_policy")
    
    plot_given("Training Data Accuracy by Iteration",
               [
                    ("Iteration Value Accuracy", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Value Accuracy"]),
                    ("Experience Buffer", gd_accuracy_series.x, gd_accuracy_series.ys["EB Value Accuracy"]),
                    ("Iteration Policy Accuracy", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Policy Accuracy"]),
                    ("Experience Buffer Policy Accuracy", gd_accuracy_series.x, gd_accuracy_series.ys["EB Policy Accuracy"]),
                    ("Iteration Policy PM", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Policy PM"]),
                    ("Experience Buffer Policy PM", gd_accuracy_series.x, gd_accuracy_series.ys["EB Policy PM"])
               ], "Iterations", "Accuracy", "training_data_accuracy")

    plot_given("Training Performance Metrics",
               [
                   ("Value Accuracy", training_metrics.x, training_metrics.ys["Value Accuracy"]),
                   ("Policy Accuracy", training_metrics.x, training_metrics.ys["Policy Accuracy"])
               ], "Iteration", "Performance", "training_metrics")
    
    plot_given("Training Performance Metrics",
               [
                   ("Loss", training_metrics.x, training_metrics.ys["Loss"]),
                   ("Value Loss", training_metrics.x, training_metrics.ys["Value Loss"]),
                   ("Policy Loss", training_metrics.x, training_metrics.ys["Policy Loss"]),
                   ("Value Accuracy", training_metrics.x, training_metrics.ys["Value Accuracy"]),
                   ("Policy Accuracy", training_metrics.x, training_metrics.ys["Policy Accuracy"])
               ], "Iteration", "Performance", "training_metrics_w_losses")
    
    plot_given("Total Games Played by Training Iteration",
               [
                   ("Total Games", gamedata_series.x, gamedata_series.ys["Total Games"])
               ], "Training Iteration", "Total Games", "gamedata_total_games")
    
    plot_stacked("Game Outcomes by Training Iteration", gamedata_series.x,
                 [
                     ("Player X Wins", gamedata_series.ys["Player X Wins"]),
                     ("Player O Wins", gamedata_series.ys["Player O Wins"]),
                     ("Draws (Repeat)", gamedata_series.ys["Draws (Repeat)"]),
                     ("Draws (Timeout)", gamedata_series.ys["Draws (Timeout)"])
                 ], "Training Iteration", "Number of Games", "gamedata_outcomes_stacked")
    
    plot_stacked_proportional("Game Outcomes by Training Iteration (Proportional)",
                              gamedata_series.x, gamedata_series.ys["Total Games"],
                               [
                                   ("Player X Wins", gamedata_series.ys["Player X Wins"]),
                                   ("Player O Wins", gamedata_series.ys["Player O Wins"]),
                                   ("Draws (Repeat)", gamedata_series.ys["Draws (Repeat)"]),
                                   ("Draws (Timeout)", gamedata_series.ys["Draws (Timeout)"])
                               ], "Training Iteration", "Proportion of Games", "gamedata_outcomes_stacked_proportional")

    plot_given("Game Outcomes by Training Iteration",
               [
                   ("Player X Wins", gamedata_series.x, gamedata_series.ys["Player X Wins"]),
                   ("Player O Wins", gamedata_series.x, gamedata_series.ys["Player O Wins"]),
                   ("Draws (Repeat)", gamedata_series.x, gamedata_series.ys["Draws (Repeat)"]),
                   ("Draws (Timeout)", gamedata_series.x, gamedata_series.ys["Draws (Timeout)"])
               ], "Training Iteration", "Number of Games", "gamedata_outcomes_lines")
    
    plot_std_error("Average Game Length (Turns) by Training Iteration",
                   gamedata_series.x, gamedata_series.ys["Avg Game Length"], gamedata_series.ys["Std Game Length"],
                   "Training Iteration", "Length (Turns)", "game_length_turns")

    plot_std_error("Average Game Length (Time) by Training Iteration",
                   gamedata_series.x, gamedata_series.ys["Avg Game Time"], gamedata_series.ys["Std Game Time"],
                   "Training Iteration", "Length (Time)", "game_length_time")

    plot_given("Training Data Bias by Iteration",
               [
                   ("Win Percentage", gd_bias_series.x, gd_bias_series.ys["Iteration Win Percent"]),
                   ("Loss Percentage", gd_bias_series.x, gd_bias_series.ys["Iteration Loss Percent"]),
                   ("Draw Percentage", gd_bias_series.x, gd_bias_series.ys["Iteration Draw Percent"]),
                   ("Overall Win Percentage", gd_overall_bias_series.x, gd_overall_bias_series.ys["Overall Win Percent"]),
                   ("Overall Loss Percentage", gd_overall_bias_series.x, gd_overall_bias_series.ys["Overall Loss Percent"]),
                   ("Overall Draw Percentage", gd_overall_bias_series.x, gd_overall_bias_series.ys["Overall Draw Percent"])
               ], "Training Iteration", "Percentage", "gamedata_bias")
    
    plot_given("Training Data Accuracy Over Game Progress (100 bins)",
               [
                   ("Value Accuracy", gd_prog_acc_100.x, gd_prog_acc_100.ys["Value Accuracy"]),
                   #("Policy Accuracy", gd_prog_acc_100.x, gd_prog_acc_100.ys["Policy Accuracy"])
               ], "Game Progress (%)", "Accuracy", "gamedata_progress_acc_100")
    
    plot_given("Training Data Accuracy Over Game Progress (10 bins)",
               [
                   ("Value Accuracy", gd_prog_acc_10.x, gd_prog_acc_10.ys["Value Accuracy"]),
                   #("Policy Accuracy", gd_prog_acc_10.x, gd_prog_acc_10.ys["Policy Accuracy"])
               ], "Game Progress (%)", "Accuracy", "gamedata_progress_acc_10")
    
    plot_given("Model Accuracy on Ground Truth Value by State Progress",
               [
                   ("Value Accuracy", state_progress_gtv.x, state_progress_gtv.ys["value_accuracy"]),
                   #("Policy Accuracy", state_progress_gtv.x, state_progress_gtv.ys["policy_accuracy"])
               ], "State Progress (%)", "Accuracy", "state_progress_gtv")

if __name__ == "__main__":
    main()
