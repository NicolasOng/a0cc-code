from typing import Callable

from a0.utils.plotting import (
    Series, save_series, load_series,
    plot_given, plot_given_groups,
    plot_stacked, plot_stacked_proportional,
    plot_std_error, plot_shaded_error, plot_ridgeline, plot_bar
)

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def safeplot(plot_callable: Callable[[], None]) -> None:
    '''Run a plot function; log a warning and continue if anything raises.'''
    try:
        plot_callable()
    except Exception as e:
        logger.warning(f"safeplot: skipping {plot_callable.__name__}: {type(e).__name__}: {e}")


def _reference_names(ref: Series) -> list[str]:
    '''Extract reference player names from a reference series' ys keys.'''
    return [key[:-len('_ev_p1')] for key in ref.ys if key.endswith('_ev_p1')]


_NUM_BINS_LIST = [10]


# === player evaluation ===

def plot_ev() -> None:
    player = load_series(f"{config.eval_dir}/player_evaluation_results.pkl")
    ref = load_series(f"{config.eval_dir}/reference_evaluation_results.pkl")
    x = player.x
    n = len(x)
    series = [
        ("Model (P1)", x, player.ys["ev_p1"]),
        ("Model (P2)", x, player.ys["ev_p2"]),
    ]
    for name in _reference_names(ref):
        series.append((f"{name} (P1)", x, [ref.ys[f"{name}_ev_p1"][0]] * n))
        series.append((f"{name} (P2)", x, [ref.ys[f"{name}_ev_p2"][0]] * n))
    plot_given(
        "Expected Value vs Baseline",
        series,
        "Training Iteration", "Expected Value",
        "player_ev",
        y_lim=(-1.0, 1.0),
    )


def plot_wld_p1() -> None:
    s = load_series(f"{config.eval_dir}/player_evaluation_results.pkl")
    plot_stacked(
        "Model as P1: W/L/D vs Baseline", s.x,
        [
            ("Wins",            s.ys["wins_p1"]),
            ("Losses",          s.ys["losses_p1"]),
            ("Draws (repeat)",  s.ys["draws_repeat_p1"]),
            ("Draws (timeout)", s.ys["draws_timeout_p1"]),
        ],
        "Training Iteration", "Games", "player_wld_p1",
    )


def plot_wld_p2() -> None:
    s = load_series(f"{config.eval_dir}/player_evaluation_results.pkl")
    plot_stacked(
        "Model as P2: W/L/D vs Baseline", s.x,
        [
            ("Wins",            s.ys["wins_p2"]),
            ("Losses",          s.ys["losses_p2"]),
            ("Draws (repeat)",  s.ys["draws_repeat_p2"]),
            ("Draws (timeout)", s.ys["draws_timeout_p2"]),
        ],
        "Training Iteration", "Games", "player_wld_p2",
    )


def plot_wld_proportional_p1() -> None:
    s = load_series(f"{config.eval_dir}/player_evaluation_results.pkl")
    plot_stacked_proportional(
        "Model as P1: W/L/D vs Baseline (Proportional)", s.x,
        s.ys["num_games_p1"],
        [
            ("Wins",            s.ys["wins_p1"]),
            ("Losses",          s.ys["losses_p1"]),
            ("Draws (repeat)",  s.ys["draws_repeat_p1"]),
            ("Draws (timeout)", s.ys["draws_timeout_p1"]),
        ],
        "Training Iteration", "Proportion", "player_wld_proportional_p1",
    )


def plot_wld_proportional_p2() -> None:
    s = load_series(f"{config.eval_dir}/player_evaluation_results.pkl")
    plot_stacked_proportional(
        "Model as P2: W/L/D vs Baseline (Proportional)", s.x,
        s.ys["num_games_p2"],
        [
            ("Wins",            s.ys["wins_p2"]),
            ("Losses",          s.ys["losses_p2"]),
            ("Draws (repeat)",  s.ys["draws_repeat_p2"]),
            ("Draws (timeout)", s.ys["draws_timeout_p2"]),
        ],
        "Training Iteration", "Proportion", "player_wld_proportional_p2",
    )


# === head accuracy ===

def plot_value_head_accuracy() -> None:
    # ["loss", "value_loss", "policy_loss", "value_accuracy", "policy_accuracy"]
    train_gt = load_series(f"{config.eval_dir}/training_gtv_eval.pkl")
    random_gt = load_series(f"{config.eval_dir}/random_gtv_eval.pkl")
    neighbor_gt: list[Series] = [
        load_series(f"{config.eval_dir}/neighbor_{i+1}_gtv_eval.pkl")
        for i in range(2)
    ]
    # ["Iteration Value Accuracy", "Iteration Value Accuracy ND", "Iteration Policy Accuracy", "Iteration Policy PM", "Iteration Policy Accuracy NT", "Iteration Policy PM NT"]
    gd_acc = load_series(f"{config.eval_dir}/gamedata_acc.pkl")
    # ["Overall Value Accuracy", "Overall Value Accuracy ND", "Overall Policy Accuracy", "Overall Policy PM", "Overall Policy Accuracy NT", "Overall Policy PM NT"]
    gd_overall_acc = load_series(f"{config.eval_dir}/gamedata_overall_acc.pkl")

    plot_given("Value Head Model Performance on Ground Truth of States and Training Data Accuracy",
        [
            ("Seen Accuracy", train_gt.x, train_gt.ys["value_accuracy"]),
            ("Neighbor 1 Accuracy", neighbor_gt[0].x, neighbor_gt[0].ys["value_accuracy"]),
            ("Neighbor 2 Accuracy", neighbor_gt[1].x, neighbor_gt[1].ys["value_accuracy"]),
            ("Random Accuracy", random_gt.x, random_gt.ys["value_accuracy"]),
            ("Training Data Accuracy", gd_acc.x, gd_acc.ys["Iteration Value Accuracy"]),
            ("Overall Training Data Accuracy", gd_overall_acc.x, gd_overall_acc.ys["Overall Value Accuracy"]),
            ("Training Data Accuracy ND", gd_acc.x, gd_acc.ys["Iteration Value Accuracy ND"]),
            ("Overall Training Data Accuracy ND", gd_overall_acc.x, gd_overall_acc.ys["Overall Value Accuracy ND"])
        ],
        "Iteration", "Accuracy", "full_accuracy_value")


def plot_policy_head_accuracy() -> None:
    train_nt_gt = load_series(f"{config.eval_dir}/training_nt_gtv_eval.pkl")
    random_nt_gt = load_series(f"{config.eval_dir}/random_nt_gtv_eval.pkl")
    neighbor_nt_gt: list[Series] = [
        load_series(f"{config.eval_dir}/neighbor_{i+1}_nt_gtv_eval.pkl")
        for i in range(2)
    ]
    gd_acc = load_series(f"{config.eval_dir}/gamedata_acc.pkl")
    gd_overall_acc = load_series(f"{config.eval_dir}/gamedata_overall_acc.pkl")

    plot_given("Policy Head Model Performance on Ground Truth of States and Training Data Accuracy",
        [
            ("Seen Accuracy NT", train_nt_gt.x, train_nt_gt.ys["policy_accuracy"]),
            ("Neighbor 1 Accuracy NT", neighbor_nt_gt[0].x, neighbor_nt_gt[0].ys["policy_accuracy"]),
            ("Neighbor 2 Accuracy NT", neighbor_nt_gt[1].x, neighbor_nt_gt[1].ys["policy_accuracy"]),
            ("Random Accuracy NT", random_nt_gt.x, random_nt_gt.ys["policy_accuracy"]),
            ("Training Data Accuracy NT", gd_acc.x, gd_acc.ys["Iteration Policy Accuracy NT"]),
            ("Overall Training Data Accuracy NT", gd_overall_acc.x, gd_overall_acc.ys["Overall Policy Accuracy NT"])
        ],
        "Iteration", "Accuracy", "full_accuracy_policy")


def plot_policy_head_accuracy_full() -> None:
    train_gt = load_series(f"{config.eval_dir}/training_gtv_eval.pkl")
    random_gt = load_series(f"{config.eval_dir}/random_gtv_eval.pkl")
    neighbor_gt: list[Series] = [
        load_series(f"{config.eval_dir}/neighbor_{i+1}_gtv_eval.pkl")
        for i in range(2)
    ]
    train_nt_gt = load_series(f"{config.eval_dir}/training_nt_gtv_eval.pkl")
    random_nt_gt = load_series(f"{config.eval_dir}/random_nt_gtv_eval.pkl")
    neighbor_nt_gt: list[Series] = [
        load_series(f"{config.eval_dir}/neighbor_{i+1}_nt_gtv_eval.pkl")
        for i in range(2)
    ]
    gd_acc = load_series(f"{config.eval_dir}/gamedata_acc.pkl")
    gd_overall_acc = load_series(f"{config.eval_dir}/gamedata_overall_acc.pkl")

    plot_given("Policy Head Model Performance on Ground Truth of States and Training Data Accuracy",
        [
            ("Seen Accuracy", train_gt.x, train_gt.ys["policy_accuracy"]),
            ("Neighbor 1 Accuracy", neighbor_gt[0].x, neighbor_gt[0].ys["policy_accuracy"]),
            ("Neighbor 2 Accuracy", neighbor_gt[1].x, neighbor_gt[1].ys["policy_accuracy"]),
            ("Random Accuracy", random_gt.x, random_gt.ys["policy_accuracy"]),
            ("Seen Accuracy NT", train_nt_gt.x, train_nt_gt.ys["policy_accuracy"]),
            ("Neighbor 1 Accuracy NT", neighbor_nt_gt[0].x, neighbor_nt_gt[0].ys["policy_accuracy"]),
            ("Neighbor 2 Accuracy NT", neighbor_nt_gt[1].x, neighbor_nt_gt[1].ys["policy_accuracy"]),
            ("Random Accuracy NT", random_nt_gt.x, random_nt_gt.ys["policy_accuracy"]),
            ("Training Data Accuracy", gd_acc.x, gd_acc.ys["Iteration Policy Accuracy"]),
            ("Overall Training Data Accuracy", gd_overall_acc.x, gd_overall_acc.ys["Overall Policy Accuracy"]),
            ("Training Data Accuracy NT", gd_acc.x, gd_acc.ys["Iteration Policy Accuracy NT"]),
            ("Overall Training Data Accuracy NT", gd_overall_acc.x, gd_overall_acc.ys["Overall Policy Accuracy NT"])
        ],
        "Iteration", "Accuracy", "full_full_accuracy_policy")


def plot_vh_acc_on_experienced() -> None:
    train_ev = load_series(f"{config.eval_dir}/training_ev_eval.pkl")
    plot_given("Model Performance on Experienced Outcomes of States Seen During Training",
        [
            ("Value Accuracy", train_ev.x, train_ev.ys["value_accuracy"]),
            ("Policy Accuracy", train_ev.x, train_ev.ys["policy_accuracy"])
        ],
        "Iteration", "Accuracy", "training_ev_eval")


# === training data accuracy ===

def plot_training_data_accuracy() -> None:
    gd_acc = load_series(f"{config.eval_dir}/gamedata_acc.pkl")
    gd_overall_acc = load_series(f"{config.eval_dir}/gamedata_overall_acc.pkl")
    plot_given("Training Data Accuracy by Iteration",
        [
            ("Overall Value Accuracy", gd_overall_acc.x, gd_overall_acc.ys["Overall Value Accuracy"]),
            ("Overall Policy Accuracy", gd_overall_acc.x, gd_overall_acc.ys["Overall Policy Accuracy"]),
            ("Overall Policy PM", gd_overall_acc.x, gd_overall_acc.ys["Overall Policy PM"]),
            ("Overall Policy Accuracy NT", gd_overall_acc.x, gd_overall_acc.ys["Overall Policy Accuracy NT"]),
            ("Overall Policy PM NT", gd_overall_acc.x, gd_overall_acc.ys["Overall Policy PM NT"]),
            ("Value Accuracy", gd_acc.x, gd_acc.ys["Iteration Value Accuracy"]),
            ("Policy Accuracy", gd_acc.x, gd_acc.ys["Iteration Policy Accuracy"]),
            ("Policy PM", gd_acc.x, gd_acc.ys["Iteration Policy PM"]),
            ("Policy Accuracy NT", gd_acc.x, gd_acc.ys["Iteration Policy Accuracy NT"]),
            ("Policy PM NT", gd_acc.x, gd_acc.ys["Iteration Policy PM NT"]),
        ], "Iterations", "Accuracy", "training_data_accuracy")


def plot_training_accuracy() -> None:
    # ["Loss", "Value Loss", "Policy Loss", "Value Accuracy", "Policy Accuracy"]
    training_metrics = load_series(f"{config.eval_dir}/training_metrics.pkl")
    plot_given("Training Performance Metrics",
        [
            ("Value Accuracy", training_metrics.x, training_metrics.ys["Value Accuracy"]),
            ("Policy Accuracy", training_metrics.x, training_metrics.ys["Policy Accuracy"])
        ], "Iteration", "Performance", "training_metrics")


def plot_training_metrics() -> None:
    training_metrics = load_series(f"{config.eval_dir}/training_metrics.pkl")
    plot_given("Training Performance Metrics",
        [
            ("Loss", training_metrics.x, training_metrics.ys["Loss"]),
            ("Value Loss", training_metrics.x, training_metrics.ys["Value Loss"]),
            ("Policy Loss", training_metrics.x, training_metrics.ys["Policy Loss"]),
            ("Value Accuracy", training_metrics.x, training_metrics.ys["Value Accuracy"]),
            ("Policy Accuracy", training_metrics.x, training_metrics.ys["Policy Accuracy"])
        ], "Iteration", "Performance", "training_metrics_w_losses")


# === game data ===

def plot_total_games() -> None:
    # ["Total Games", "Player X Wins", "Player O Wins", "Draws (Repeat)", "Draws (Timeout)", "Avg Game Length", "Avg Game Time", "Std Game Length", "Std Game Time"]
    s = load_series(f"{config.eval_dir}/gamedata_stats.pkl")
    plot_given("Total Games Played by Training Iteration",
        [
            ("Total Games", s.x, s.ys["Total Games"])
        ], "Training Iteration", "Total Games", "gamedata_total_games")


def plot_game_outcomes_stacked() -> None:
    s = load_series(f"{config.eval_dir}/gamedata_stats.pkl")
    plot_stacked("Game Outcomes by Training Iteration", s.x,
        [
            ("Player X Wins", s.ys["Player X Wins"]),
            ("Player O Wins", s.ys["Player O Wins"]),
            ("Draws (Repeat)", s.ys["Draws (Repeat)"]),
            ("Draws (Timeout)", s.ys["Draws (Timeout)"])
        ], "Training Iteration", "Number of Games", "gamedata_outcomes_stacked")


def plot_game_outcomes_stacked_proportional() -> None:
    s = load_series(f"{config.eval_dir}/gamedata_stats.pkl")
    plot_stacked_proportional("Game Outcomes by Training Iteration (Proportional)",
        s.x, s.ys["Total Games"],
        [
            ("Player X Wins", s.ys["Player X Wins"]),
            ("Player O Wins", s.ys["Player O Wins"]),
            ("Draws (Repeat)", s.ys["Draws (Repeat)"]),
            ("Draws (Timeout)", s.ys["Draws (Timeout)"])
        ], "Training Iteration", "Proportion of Games", "gamedata_outcomes_stacked_proportional")


def plot_game_outcomes_lines() -> None:
    s = load_series(f"{config.eval_dir}/gamedata_stats.pkl")
    plot_given("Game Outcomes by Training Iteration",
        [
            ("Player X Wins", s.x, s.ys["Player X Wins"]),
            ("Player O Wins", s.x, s.ys["Player O Wins"]),
            ("Draws (Repeat)", s.x, s.ys["Draws (Repeat)"]),
            ("Draws (Timeout)", s.x, s.ys["Draws (Timeout)"])
        ], "Training Iteration", "Number of Games", "gamedata_outcomes_lines")


def plot_game_length_turns() -> None:
    s = load_series(f"{config.eval_dir}/gamedata_stats.pkl")
    plot_std_error("Average Game Length (Turns) by Training Iteration",
        s.x, s.ys["Avg Game Length"], s.ys["Std Game Length"],
        "Training Iteration", "Length (Turns)", "game_length_turns")


def plot_game_length_time() -> None:
    s = load_series(f"{config.eval_dir}/gamedata_stats.pkl")
    plot_std_error("Average Game Length (Time) by Training Iteration",
        s.x, s.ys["Avg Game Time"], s.ys["Std Game Time"],
        "Training Iteration", "Length (Time)", "game_length_time")


def plot_gamedata_bias() -> None:
    # ["Iteration Win Percent", "Iteration Loss Percent", "Iteration Draw Percent"]
    bias = load_series(f"{config.eval_dir}/gamedata_bias.pkl")
    # ["Overall Win Percent", "Overall Loss Percent", "Overall Draw Percent"]
    overall_bias = load_series(f"{config.eval_dir}/gamedata_overall_bias.pkl")
    plot_given("Training Data Bias by Iteration",
        [
            ("Win Percentage", bias.x, bias.ys["Iteration Win Percent"]),
            ("Loss Percentage", bias.x, bias.ys["Iteration Loss Percent"]),
            ("Draw Percentage", bias.x, bias.ys["Iteration Draw Percent"]),
            ("Overall Win Percentage", overall_bias.x, overall_bias.ys["Overall Win Percent"]),
            ("Overall Loss Percentage", overall_bias.x, overall_bias.ys["Overall Loss Percent"]),
            ("Overall Draw Percentage", overall_bias.x, overall_bias.ys["Overall Draw Percent"])
        ], "Training Iteration", "Percentage", "gamedata_bias")


# === progress ===

def plot_gamedata_progress_acc() -> None:
    # ["Num States", "Value Accuracy"]
    p100 = load_series(f"{config.eval_dir}/gamedata_progress_acc_100.pkl")
    p10 = load_series(f"{config.eval_dir}/gamedata_progress_acc_10.pkl")
    plot_given("Training Data Accuracy Over Game Progress",
        [
            ("Value Accuracy (100 bins)", p100.x, p100.ys["Value Accuracy"]),
            ("Value Accuracy (10 bins)", p10.x, p10.ys["Value Accuracy"])
        ], "Game Progress (%)", "Accuracy", "gamedata_progress_acc")


def plot_state_progress_gtv() -> None:
    # ["loss", "value_loss", "policy_loss", "value_accuracy", "policy_accuracy"]
    s = load_series(f"{config.eval_dir}/state_progress_gtv_datasets_eval.pkl")
    plot_given("Model Accuracy on Ground Truth Value by State Progress",
        [
            ("Value Accuracy", s.x, s.ys["value_accuracy"]),
        ], "State Progress (%)", "Accuracy", "state_progress_gtv")


def plot_gamedata_progress_num_states_100() -> None:
    p100 = load_series(f"{config.eval_dir}/gamedata_progress_acc_100.pkl")
    plot_bar("Number of States in Each Progress Bin (100 bins)",
        ("Num States", p100.x, p100.ys["Num States"]),
        "State Progress (%)", "Number of States", "gamedata_progress_num_states_100")


def plot_gamedata_progress_num_states_10() -> None:
    p10 = load_series(f"{config.eval_dir}/gamedata_progress_acc_10.pkl")
    plot_bar("Number of States in Each Progress Bin (10 bins)",
        ("Num States", p10.x, p10.ys["Num States"]),
        "State Progress (%)", "Number of States", "gamedata_progress_num_states_10")


def plot_state_progress_num_states() -> None:
    # ["Num States", "Num States NT"]
    s = load_series(f"{config.eval_dir}/state_progress_state_nums.pkl")
    plot_bar("Number of States in Each Progress Bin",
        ("Num States", s.x, s.ys["Num States"]),
        "State Progress (%)", "Number of States", "state_progress_num_states")


def plot_state_progress_num_states_nt() -> None:
    s = load_series(f"{config.eval_dir}/state_progress_state_nums.pkl")
    plot_bar("Number of Non-Trivial States in Each Progress Bin",
        ("Num States", s.x, s.ys["Num States NT"]),
        "State Progress (%)", "Number of States", "state_progress_num_states_nt")


# === game progress per bin ===

def plot_gp_num_states_over_game_progress() -> None:
    for num_bin in _NUM_BINS_LIST:
        ns = load_series(f"{config.eval_dir}/num_states_over_game_progress_{num_bin}.pkl")
        nus = load_series(f"{config.eval_dir}/num_unique_states_over_game_progress_{num_bin}.pkl")
        plot_given(f"Number of States and Unique States Over Game Progress (Num Bins: {num_bin})",
            [
                ("Num States", ns.x, ns.ys["Num States"]),
                ("Num Unique States", nus.x, nus.ys["Num Unique States"])
            ], "Game Progress (%)", "Number of States", f"gp_num_states_over_game_progress_{num_bin}")


def plot_gp_baseline_accuracy_over_game_progress() -> None:
    for num_bin in _NUM_BINS_LIST:
        s = load_series(f"{config.eval_dir}/baseline_accuracy_over_game_progress_{num_bin}.pkl")
        plot_given_groups(f"Baseline Accuracy Over Game Progress (Num Bins: {num_bin})",
            [
                [
                    ("Value Accuracy", s.x, s.ys["Baseline Value Accuracy"]),
                    ("Value Accuracy (ND)", s.x, s.ys["Baseline Value Accuracy (ND)"]),
                ],
                [
                    ("Policy Accuracy", s.x, s.ys["Baseline Policy Accuracy"]),
                    ("Policy Accuracy (NT)", s.x, s.ys["Baseline Policy Accuracy (NT)"]),
                ]
            ], "Game Progress (%)", "Baseline Accuracy", f"gp_baseline_accuracy_over_game_progress_{num_bin}")


def plot_gp_training_data_accuracy_over_game_progress() -> None:
    for num_bin in _NUM_BINS_LIST:
        s = load_series(f"{config.eval_dir}/training_data_accuracy_over_game_progress_{num_bin}.pkl")
        plot_given_groups(f"Training Data Accuracy Over Game Progress (Num Bins: {num_bin})",
            [
                [
                    ("Value Accuracy", s.x, s.ys["Value Accuracy"]),
                    ("Value Accuracy (ND)", s.x, s.ys["Value Accuracy (ND)"]),
                ],
                [
                    ("Policy Accuracy", s.x, s.ys["Policy Accuracy"]),
                    ("Policy Accuracy (NT)", s.x, s.ys["Policy Accuracy (NT)"]),
                ]
            ], "Game Progress (%)", "Training Data Accuracy", f"gp_training_data_accuracy_over_game_progress_{num_bin}")


def plot_gp_branching_factor_over_game_progress() -> None:
    for num_bin in _NUM_BINS_LIST:
        s = load_series(f"{config.eval_dir}/branching_factor_over_game_progress_{num_bin}.pkl")
        plot_given(f"Branching Factor Over Game Progress (Num Bins: {num_bin})",
            [
                ("Average Branching Factor", s.x, s.ys["Average Branching Factor"])
            ], "Game Progress (%)", "Branching Factor", f"gp_branching_factor_over_game_progress_{num_bin}")


# TODO: plot mcts eval series
# def plot_mcts_eval() -> None:
#     # ["loss", "value_loss", "policy_loss", "value_accuracy", "policy_accuracy"]
#     mcts_eval_series: list[tuple[int, Series, Series, Series, list[Series]]] = []
#     for mcts_samples in config.eval_mcts_samples:
#         training_gtv = load_series(f"{config.eval_dir}/mcts_eval_{mcts_samples}/training_gtv_mcts_eval_{mcts_samples}")
#         random_gtv = load_series(f"{config.eval_dir}/mcts_eval_{mcts_samples}/random_gtv_mcts_eval_{mcts_samples}")
#         training_ev = load_series(f"{config.eval_dir}/mcts_eval_{mcts_samples}/training_ev_mcts_eval_{mcts_samples}")
#         num_neighbors = config.eval_neighbors
#         neighbor_gtvs: list[Series] = []
#         for i in range(num_neighbors):
#             neighbor_gtv = load_series(f"{config.eval_dir}/mcts_eval_{mcts_samples}/neighbor_{i+1}_gtv_mcts_eval_{mcts_samples}")
#             neighbor_gtvs.append(neighbor_gtv)
#         mcts_eval_series.append((mcts_samples, training_gtv, random_gtv, training_ev, neighbor_gtvs))


def main():
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="plotting"
    )
    logger.info("plotting...")

    # head accuracy
    safeplot(plot_vh_acc_on_experienced)
    safeplot(plot_value_head_accuracy)
    safeplot(plot_policy_head_accuracy)
    safeplot(plot_policy_head_accuracy_full)
    safeplot(plot_training_data_accuracy)
    safeplot(plot_training_accuracy)
    safeplot(plot_training_metrics)

    # game data
    safeplot(plot_total_games)
    safeplot(plot_game_outcomes_stacked)
    safeplot(plot_game_outcomes_stacked_proportional)
    safeplot(plot_game_outcomes_lines)
    safeplot(plot_game_length_turns)
    safeplot(plot_game_length_time)
    safeplot(plot_gamedata_bias)

    # progress
    safeplot(plot_gamedata_progress_acc)
    safeplot(plot_state_progress_gtv)
    safeplot(plot_gamedata_progress_num_states_100)
    safeplot(plot_gamedata_progress_num_states_10)
    safeplot(plot_state_progress_num_states)
    safeplot(plot_state_progress_num_states_nt)

    # game progress per bin
    safeplot(plot_gp_num_states_over_game_progress)
    safeplot(plot_gp_baseline_accuracy_over_game_progress)
    safeplot(plot_gp_training_data_accuracy_over_game_progress)
    safeplot(plot_gp_branching_factor_over_game_progress)

    # player evaluation
    safeplot(plot_ev)
    safeplot(plot_wld_p1)
    safeplot(plot_wld_p2)
    safeplot(plot_wld_proportional_p1)
    safeplot(plot_wld_proportional_p2)


if __name__ == "__main__":
    main()
