import sys
import os

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
    os.makedirs(os.path.dirname(series_path), exist_ok=True)
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
    plt.close()

def plot_given_groups(title: str, groups: list[list[tuple[str, list[int], list[float]]]], x_label: str, y_label: str, fn: str, use_log_y: bool = False) -> None:
    plt.figure(figsize=(16, 9))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']  # Add more if needed
    line_styles = ['-', '--', '-.', ':']  # Solid, dashed, dash-dot, dotted
    for group_idx, group in enumerate(groups):
        color = colors[group_idx % len(colors)]
        for series_idx, (label, x_values, y_values) in enumerate(group):
            line_style = line_styles[series_idx % len(line_styles)]
            plt.plot(x_values, y_values, label=label, color=color, linestyle=line_style)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both')
    if use_log_y:
        plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}{fn}.png")
    plt.close()

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
    plt.close()

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
    plt.close()

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
    plt.close()

def plot_shaded_error(title: str, series: list[tuple[str, str, list[int], list[float], list[float]]], x_label: str, y_label: str, fn: str, y_lim: tuple[float, float] | None = None):
    plt.figure(figsize=(16, 9))
    for line_label, fill_label, x_values, y_values, fill_values in series:
        plt.plot(x_values, y_values, label=line_label)
        plt.fill_between(x_values, np.array(y_values) - np.array(fill_values), np.array(y_values) + np.array(fill_values), 
                        alpha=0.3, label=fill_label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/{fn}.png")
    plt.close()

def plot_bar(title: str, series: tuple[str, list[int], list[float]], x_label: str, y_label: str, fn: str) -> None:
    plt.figure(figsize=(16, 9))
    
    label, x_values, y_values = series
    
    plt.bar(x_values, y_values, label=label)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/{fn}.png")
    plt.close()

def main():
    # I should split this main function into many smaller ones,
    # each of which can fail independently if a series is missing.
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="plotting"
    )
    logger.info("plotting...")

    # load all the series
    # ["loss", "value_loss", "policy_loss", "value_accuracy", "policy_accuracy"]
    train_gt_series = load_series(f"{config.eval_dir}/training_gtv_eval.pkl")
    random_gt_series = load_series(f"{config.eval_dir}/random_gtv_eval.pkl")
    train_ev_series = load_series(f"{config.eval_dir}/training_ev_eval.pkl")
    n_neighbor_gt_series: list[Series] = []
    for i in range(2):
        neighbor_gt_series = load_series(f"{config.eval_dir}/neighbor_{i+1}_gtv_eval.pkl")
        n_neighbor_gt_series.append(neighbor_gt_series)

    train_nt_gt_series = load_series(f"{config.eval_dir}/training_nt_gtv_eval.pkl")
    random_nt_gt_series = load_series(f"{config.eval_dir}/random_nt_gtv_eval.pkl")
    train_nt_ev_series = load_series(f"{config.eval_dir}/training_nt_ev_eval.pkl")
    n_neighbor_nt_gt_series: list[Series] = []
    for i in range(2):
        neighbor_nt_gt_series = load_series(f"{config.eval_dir}/neighbor_{i+1}_nt_gtv_eval.pkl")
        n_neighbor_nt_gt_series.append(neighbor_nt_gt_series)

    # ["Total Games", "Player X Wins", "Player O Wins", "Draws (Repeat)", "Draws (Timeout)", "Avg Game Length", "Avg Game Time", "Std Game Length", "Std Game Time"]
    gamedata_series = load_series(f"{config.eval_dir}/gamedata_stats.pkl")
    # ["Iteration Value Accuracy", "Iteration Value Accuracy ND", "Iteration Policy Accuracy", "Iteration Policy PM", "Iteration Policy Accuracy NT", "Iteration Policy PM NT"]
    gd_accuracy_series = load_series(f"{config.eval_dir}/gamedata_acc.pkl")
    # ["Overall Value Accuracy", "Overall Value Accuracy ND", "Overall Policy Accuracy", "Overall Policy PM", "Overall Policy Accuracy NT", "Overall Policy PM NT"]
    gd_overall_accuracy_series = load_series(f"{config.eval_dir}/gamedata_overall_acc.pkl")
    # ["Loss", "Value Loss", "Policy Loss", "Value Accuracy", "Policy Accuracy"]
    training_metrics = load_series(f"{config.eval_dir}/training_metrics.pkl")
    # ["Iteration Win Percent", "Iteration Loss Percent", "Iteration Draw Percent"]
    gd_bias_series = load_series(f"{config.eval_dir}/gamedata_bias.pkl")
    # ["Overall Win Percent", "Overall Loss Percent", "Overall Draw Percent"]
    gd_overall_bias_series = load_series(f"{config.eval_dir}/gamedata_overall_bias.pkl")

    # ["Num States", "Value Accuracy"]
    gd_prog_acc_100 = load_series(f"{config.eval_dir}/gamedata_progress_acc_100.pkl")
    gd_prog_acc_10 = load_series(f"{config.eval_dir}/gamedata_progress_acc_10.pkl")

    # ["loss", "value_loss", "policy_loss", "value_accuracy", "policy_accuracy"]
    state_progress_gtv = load_series(f"{config.eval_dir}/state_progress_gtv_datasets_eval.pkl")
    state_progress_nt_gtv = load_series(f"{config.eval_dir}/state_progress_nt_gtv_datasets_eval.pkl")
    # ["Num States", "Num States NT"]
    state_progress_state_nums = load_series(f"{config.eval_dir}/state_progress_state_nums.pkl")

    num_bins_list = [10]
    n_num_states_series: list[Series] = []
    n_num_unique_states_series: list[Series] = []
    n_baseline_acc_over_gp_series: list[Series] = []
    n_acc_over_gp_series: list[Series] = []
    n_branching_factor_series: list[Series] = []
    for num_bins in num_bins_list:
        n_num_states_series.append(load_series(f"{config.eval_dir}/num_states_over_game_progress_{num_bins}.pkl"))
        n_num_unique_states_series.append(load_series(f"{config.eval_dir}/num_unique_states_over_game_progress_{num_bins}.pkl"))
        n_baseline_acc_over_gp_series.append(load_series(f"{config.eval_dir}/baseline_accuracy_over_game_progress_{num_bins}.pkl"))
        n_acc_over_gp_series.append(load_series(f"{config.eval_dir}/training_data_accuracy_over_game_progress_{num_bins}.pkl"))
        n_branching_factor_series.append(load_series(f"{config.eval_dir}/branching_factor_over_game_progress_{num_bins}.pkl"))
    
    # ["loss", "value_loss", "policy_loss", "value_accuracy", "policy_accuracy"]
    # mcts_eval_series: list[tuple[int, Series, Series, Series, list[Series]]] = []
    # for mcts_samples in config.eval_mcts_samples:
    #     training_gtv = load_series(f"{config.eval_dir}/mcts_eval_{mcts_samples}/training_gtv_mcts_eval_{mcts_samples}")
    #     random_gtv = load_series(f"{config.eval_dir}/mcts_eval_{mcts_samples}/random_gtv_mcts_eval_{mcts_samples}")
    #     training_ev = load_series(f"{config.eval_dir}/mcts_eval_{mcts_samples}/training_ev_mcts_eval_{mcts_samples}")
    #     num_neighbors = config.eval_neighbors
    #     neighbor_gtvs: list[Series] = []
    #     for i in range(num_neighbors):
    #         neighbor_gtv = load_series(f"{config.eval_dir}/mcts_eval_{mcts_samples}/neighbor_{i+1}_gtv_mcts_eval_{mcts_samples}")
    #         neighbor_gtvs.append(neighbor_gtv)
    #     mcts_eval_series.append((mcts_samples, training_gtv, random_gtv, training_ev, neighbor_gtvs))


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
                    ("Overall Training Data Accuracy", gd_overall_accuracy_series.x, gd_overall_accuracy_series.ys["Overall Value Accuracy"]),
                    ("Training Data Accuracy ND", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Value Accuracy ND"]),
                    ("Overall Training Data Accuracy ND", gd_overall_accuracy_series.x, gd_overall_accuracy_series.ys["Overall Value Accuracy ND"])

               ],
               "Iteration", "Accuracy", "full_accuracy_value")

    plot_given("Policy Head Model Performance on Ground Truth of States and Training Data Accuracy",
               [
                    ("Seen Accuracy", train_gt_series.x, train_gt_series.ys["policy_accuracy"]),
                    ("Neighbor 1 Accuracy", n_neighbor_gt_series[0].x, n_neighbor_gt_series[0].ys["policy_accuracy"]),
                    ("Neighbor 2 Accuracy", n_neighbor_gt_series[1].x, n_neighbor_gt_series[1].ys["policy_accuracy"]),
                    ("Random Accuracy", random_gt_series.x, random_gt_series.ys["policy_accuracy"]),
                    ("Seen Accuracy NT", train_nt_gt_series.x, train_nt_gt_series.ys["policy_accuracy"]),
                    ("Neighbor 1 Accuracy NT", n_neighbor_nt_gt_series[0].x, n_neighbor_nt_gt_series[0].ys["policy_accuracy"]),
                    ("Neighbor 2 Accuracy NT", n_neighbor_nt_gt_series[1].x, n_neighbor_nt_gt_series[1].ys["policy_accuracy"]),
                    ("Random Accuracy NT", random_nt_gt_series.x, random_nt_gt_series.ys["policy_accuracy"]),
                    ("Training Data Accuracy", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Policy Accuracy"]),
                    ("Overall Training Data Accuracy", gd_overall_accuracy_series.x, gd_overall_accuracy_series.ys["Overall Policy Accuracy"]),
                    ("Training Data Accuracy NT", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Policy Accuracy NT"]),
                    ("Overall Training Data Accuracy NT", gd_overall_accuracy_series.x, gd_overall_accuracy_series.ys["Overall Policy Accuracy NT"])
               ],
               "Iteration", "Accuracy", "full_full_accuracy_policy")
    
    plot_given("Policy Head Model Performance on Ground Truth of States and Training Data Accuracy",
               [
                    ("Seen Accuracy NT", train_nt_gt_series.x, train_nt_gt_series.ys["policy_accuracy"]),
                    ("Neighbor 1 Accuracy NT", n_neighbor_nt_gt_series[0].x, n_neighbor_nt_gt_series[0].ys["policy_accuracy"]),
                    ("Neighbor 2 Accuracy NT", n_neighbor_nt_gt_series[1].x, n_neighbor_nt_gt_series[1].ys["policy_accuracy"]),
                    ("Random Accuracy NT", random_nt_gt_series.x, random_nt_gt_series.ys["policy_accuracy"]),
                    ("Training Data Accuracy NT", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Policy Accuracy NT"]),
                    ("Overall Training Data Accuracy NT", gd_overall_accuracy_series.x, gd_overall_accuracy_series.ys["Overall Policy Accuracy NT"])
               ],
               "Iteration", "Accuracy", "full_accuracy_policy")
    
    plot_given("Training Data Accuracy by Iteration",
               [
                    ("Overall Value Accuracy", gd_overall_accuracy_series.x, gd_overall_accuracy_series.ys["Overall Value Accuracy"]),
                    ("Overall Policy Accuracy", gd_overall_accuracy_series.x, gd_overall_accuracy_series.ys["Overall Policy Accuracy"]),
                    ("Overall Policy PM", gd_overall_accuracy_series.x, gd_overall_accuracy_series.ys["Overall Policy PM"]),
                    ("Overall Policy Accuracy NT", gd_overall_accuracy_series.x, gd_overall_accuracy_series.ys["Overall Policy Accuracy NT"]),
                    ("Overall Policy PM NT", gd_overall_accuracy_series.x, gd_overall_accuracy_series.ys["Overall Policy PM NT"]),
                    ("Value Accuracy", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Value Accuracy"]),
                    ("Policy Accuracy", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Policy Accuracy"]),
                    ("Policy PM", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Policy PM"]),
                    ("Policy Accuracy NT", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Policy Accuracy NT"]),
                    ("Policy PM NT", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Policy PM NT"]),
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
    
    plot_given("Training Data Accuracy Over Game Progress",
               [
                   ("Value Accuracy (100 bins)", gd_prog_acc_100.x, gd_prog_acc_100.ys["Value Accuracy"]),
                   ("Value Accuracy (10 bins)", gd_prog_acc_10.x, gd_prog_acc_10.ys["Value Accuracy"])
                   #("Policy Accuracy", gd_prog_acc_100.x, gd_prog_acc_100.ys["Policy Accuracy"])
               ], "Game Progress (%)", "Accuracy", "gamedata_progress_acc")
    
    plot_given("Model Accuracy on Ground Truth Value by State Progress",
               [
                   ("Value Accuracy", state_progress_gtv.x, state_progress_gtv.ys["value_accuracy"]),
                   #("Policy Accuracy", state_progress_gtv.x, state_progress_gtv.ys["policy_accuracy"])
               ], "State Progress (%)", "Accuracy", "state_progress_gtv")

    plot_bar("Number of States in Each Progress Bin (100 bins)",
            ("Num States", gd_prog_acc_100.x, gd_prog_acc_100.ys["Num States"]),
            "State Progress (%)", "Number of States", "gamedata_progress_num_states_100")

    plot_bar("Number of States in Each Progress Bin (10 bins)",
            ("Num States", gd_prog_acc_10.x, gd_prog_acc_10.ys["Num States"]),
            "State Progress (%)", "Number of States", "gamedata_progress_num_states_10")
    
    plot_bar("Number of States in Each Progress Bin",
            ("Num States", state_progress_state_nums.x, state_progress_state_nums.ys["Num States"]),
            "State Progress (%)", "Number of States", "state_progress_num_states")

    plot_bar("Number of Non-Trivial States in Each Progress Bin",
            ("Num States", state_progress_state_nums.x, state_progress_state_nums.ys["Num States NT"]),
            "State Progress (%)", "Number of States", "state_progress_num_states_nt")
    
    for i, num_bin in enumerate(num_bins_list):
        plot_given(f"Number of States and Unique States Over Game Progress (Num Bins: {num_bin})",
                   [
                       ("Num States", n_num_states_series[i].x, n_num_states_series[i].ys["Num States"]),
                       ("Num Unique States", n_num_unique_states_series[i].x, n_num_unique_states_series[i].ys["Num Unique States"])
                   ], "Game Progress (%)", "Number of States", f"gp_num_states_over_game_progress_{num_bin}")
        
        plot_given_groups(f"Baseline Accuracy Over Game Progress (Num Bins: {num_bin})",
                   [
                       [
                            ("Value Accuracy", n_baseline_acc_over_gp_series[i].x, n_baseline_acc_over_gp_series[i].ys["Baseline Value Accuracy"]),
                            ("Value Accuracy (ND)", n_baseline_acc_over_gp_series[i].x, n_baseline_acc_over_gp_series[i].ys["Baseline Value Accuracy (ND)"]),
                       ],
                       [
                            ("Policy Accuracy", n_baseline_acc_over_gp_series[i].x, n_baseline_acc_over_gp_series[i].ys["Baseline Policy Accuracy"]),
                            ("Policy Accuracy (NT)", n_baseline_acc_over_gp_series[i].x, n_baseline_acc_over_gp_series[i].ys["Baseline Policy Accuracy (NT)"]),
                       ]
                   ], "Game Progress (%)", "Baseline Accuracy", f"gp_baseline_accuracy_over_game_progress_{num_bin}")
        
        plot_given_groups(f"Training Data Accuracy Over Game Progress (Num Bins: {num_bin})",
                   [
                       [
                           ("Value Accuracy", n_acc_over_gp_series[i].x, n_acc_over_gp_series[i].ys["Value Accuracy"]),
                           ("Value Accuracy (ND)", n_acc_over_gp_series[i].x, n_acc_over_gp_series[i].ys["Value Accuracy (ND)"]),
                       ], 
                       [
                            ("Policy Accuracy", n_acc_over_gp_series[i].x, n_acc_over_gp_series[i].ys["Policy Accuracy"]),
                            ("Policy Accuracy (NT)", n_acc_over_gp_series[i].x, n_acc_over_gp_series[i].ys["Policy Accuracy (NT)"]),
                       ]
                   ], "Game Progress (%)", "Training Data Accuracy", f"gp_training_data_accuracy_over_game_progress_{num_bin}")
        
        plot_given(f"Branching Factor Over Game Progress (Num Bins: {num_bin})",
                   [
                       ("Average Branching Factor", n_branching_factor_series[i].x, n_branching_factor_series[i].ys["Average Branching Factor"])
                   ], "Game Progress (%)", "Branching Factor", f"gp_branching_factor_over_game_progress_{num_bin}")

if __name__ == "__main__":
    main()
