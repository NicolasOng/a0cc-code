from typing import Callable

from a0.utils.plotting import (
    Series,
    load_series,
    load_distribution_series,
    plot_given,
    plot_given_groups,
    plot_stacked,
    plot_stacked_proportional,
    plot_std_error,
    plot_shaded_ridgeline,
    plot_percentile_bands,
    plot_value_proportions,
    plot_bar,
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


# === distribution series plots ===

def _plot_distribution_variants(series_fn: str, title: str, fn_base: str) -> None:
    '''Render ridgeline + percentile bands + value-proportion plots for one distribution series.'''
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


def plot_dataset_pre_balance_distributions() -> None:
    _plot_distribution_variants(
        "dataset_pre_balance_distributions.pkl",
        "Dataset Pre Balance Distributions",
        "dataset_pre_balance_distributions",
    )


def plot_dataset_post_balance_distributions() -> None:
    _plot_distribution_variants(
        "dataset_post_balance_distributions.pkl",
        "Dataset Post Balance Distributions",
        "dataset_post_balance_distributions",
    )


def plot_random_nd_value_distributions() -> None:
    _plot_distribution_variants(
        "random_nd_value_distributions.pkl",
        "Model Value Predictions on random_nd",
        "random_nd_value_distributions",
    )


# === training metrics ===

def plot_training_metrics_accuracy() -> None:
    s = load_series(f"{config.eval_dir}/training_metrics.pkl")
    plot_given(
        "Training Performance Metrics",
        [
            ("Value Accuracy", s.x, s.ys["Value Accuracy"]),
            ("Policy Accuracy", s.x, s.ys["Policy Accuracy"]),
        ],
        "Iteration", "Performance", "training_metrics",
    )


def plot_training_metrics_with_losses() -> None:
    s = load_series(f"{config.eval_dir}/training_metrics.pkl")
    plot_given(
        "Training Performance Metrics (with losses)",
        [
            ("Loss", s.x, s.ys["Loss"]),
            ("Value Loss", s.x, s.ys["Value Loss"]),
            ("Policy Loss", s.x, s.ys["Policy Loss"]),
            ("Value Accuracy", s.x, s.ys["Value Accuracy"]),
            ("Policy Accuracy", s.x, s.ys["Policy Accuracy"]),
        ],
        "Iteration", "Performance", "training_metrics_w_losses",
    )


# === gamedata stats ===

def plot_gamedata_total_games() -> None:
    s = load_series(f"{config.eval_dir}/gamedata_stats.pkl")
    plot_given(
        "Total Games Played by Training Iteration",
        [("Total Games", s.x, s.ys["Total Games"])],
        "Training Iteration", "Total Games", "gamedata_total_games",
    )


def plot_gamedata_outcomes_stacked() -> None:
    s = load_series(f"{config.eval_dir}/gamedata_stats.pkl")
    plot_stacked(
        "Game Outcomes by Training Iteration", s.x,
        [
            ("Player X Wins", s.ys["Player X Wins"]),
            ("Player O Wins", s.ys["Player O Wins"]),
            ("Draws (Repeat)", s.ys["Draws (Repeat)"]),
            ("Draws (Timeout)", s.ys["Draws (Timeout)"]),
        ],
        "Training Iteration", "Number of Games", "gamedata_outcomes_stacked",
    )


def plot_gamedata_outcomes_stacked_proportional() -> None:
    s = load_series(f"{config.eval_dir}/gamedata_stats.pkl")
    plot_stacked_proportional(
        "Game Outcomes by Training Iteration (Proportional)",
        s.x, s.ys["Total Games"],
        [
            ("Player X Wins", s.ys["Player X Wins"]),
            ("Player O Wins", s.ys["Player O Wins"]),
            ("Draws (Repeat)", s.ys["Draws (Repeat)"]),
            ("Draws (Timeout)", s.ys["Draws (Timeout)"]),
        ],
        "Training Iteration", "Proportion of Games", "gamedata_outcomes_stacked_proportional",
    )


def plot_gamedata_outcomes_lines() -> None:
    s = load_series(f"{config.eval_dir}/gamedata_stats.pkl")
    plot_given(
        "Game Outcomes by Training Iteration",
        [
            ("Player X Wins", s.x, s.ys["Player X Wins"]),
            ("Player O Wins", s.x, s.ys["Player O Wins"]),
            ("Draws (Repeat)", s.x, s.ys["Draws (Repeat)"]),
            ("Draws (Timeout)", s.x, s.ys["Draws (Timeout)"]),
        ],
        "Training Iteration", "Number of Games", "gamedata_outcomes_lines",
    )


def plot_gamedata_game_length() -> None:
    s = load_series(f"{config.eval_dir}/gamedata_stats.pkl")
    plot_std_error(
        "Average Game Length (Turns) by Training Iteration",
        s.x, s.ys["Avg Game Length"], s.ys["Std Game Length"],
        "Training Iteration", "Length (Turns)", "game_length_turns",
    )


def plot_gamedata_game_time() -> None:
    s = load_series(f"{config.eval_dir}/gamedata_stats.pkl")
    plot_std_error(
        "Average Game Time by Training Iteration",
        s.x, s.ys["Avg Game Time"], s.ys["Std Game Time"],
        "Training Iteration", "Time (s)", "game_length_time",
    )


# === gamedata accuracy & bias (experienced + alt-target variants) ===

def _plot_iteration_accuracy(iteration: Series, overall: Series, title: str, fn: str) -> None:
    plot_given(
        title,
        [
            ("Overall Value Accuracy", overall.x, overall.ys["Overall Value Accuracy"]),
            ("Overall Value Accuracy ND", overall.x, overall.ys["Overall Value Accuracy ND"]),
            ("Overall Policy Accuracy", overall.x, overall.ys["Overall Policy Accuracy"]),
            ("Overall Policy PM", overall.x, overall.ys["Overall Policy PM"]),
            ("Overall Policy Accuracy NT", overall.x, overall.ys["Overall Policy Accuracy NT"]),
            ("Overall Policy PM NT", overall.x, overall.ys["Overall Policy PM NT"]),
            ("Value Accuracy", iteration.x, iteration.ys["Iteration Value Accuracy"]),
            ("Value Accuracy ND", iteration.x, iteration.ys["Iteration Value Accuracy ND"]),
            ("Policy Accuracy", iteration.x, iteration.ys["Iteration Policy Accuracy"]),
            ("Policy PM", iteration.x, iteration.ys["Iteration Policy PM"]),
            ("Policy Accuracy NT", iteration.x, iteration.ys["Iteration Policy Accuracy NT"]),
            ("Policy PM NT", iteration.x, iteration.ys["Iteration Policy PM NT"]),
        ],
        "Iterations", "Accuracy", fn,
    )


def _plot_iteration_bias(iteration: Series, overall: Series, title: str, fn: str) -> None:
    plot_given(
        title,
        [
            ("Win Percentage", iteration.x, iteration.ys["Iteration Win Percent"]),
            ("Loss Percentage", iteration.x, iteration.ys["Iteration Loss Percent"]),
            ("Draw Percentage", iteration.x, iteration.ys["Iteration Draw Percent"]),
            ("Overall Win Percentage", overall.x, overall.ys["Overall Win Percent"]),
            ("Overall Loss Percentage", overall.x, overall.ys["Overall Loss Percent"]),
            ("Overall Draw Percentage", overall.x, overall.ys["Overall Draw Percent"]),
        ],
        "Training Iteration", "Percentage", fn,
    )


def plot_gamedata_accuracy() -> None:
    iteration = load_series(f"{config.eval_dir}/gamedata_acc.pkl")
    overall = load_series(f"{config.eval_dir}/gamedata_overall_acc.pkl")
    _plot_iteration_accuracy(iteration, overall, "Training Data Accuracy by Iteration", "training_data_accuracy")


def plot_gamedata_alt_accuracy() -> None:
    iteration = load_series(f"{config.eval_dir}/gamedata_alt_acc.pkl")
    overall = load_series(f"{config.eval_dir}/gamedata_alt_overall_acc.pkl")
    _plot_iteration_accuracy(iteration, overall, "Alt-Target Training Data Accuracy by Iteration", "training_data_alt_accuracy")


def plot_gamedata_bias() -> None:
    iteration = load_series(f"{config.eval_dir}/gamedata_bias.pkl")
    overall = load_series(f"{config.eval_dir}/gamedata_overall_bias.pkl")
    _plot_iteration_bias(iteration, overall, "Training Data Bias by Iteration", "gamedata_bias")


def plot_gamedata_alt_bias() -> None:
    iteration = load_series(f"{config.eval_dir}/gamedata_alt_bias.pkl")
    overall = load_series(f"{config.eval_dir}/gamedata_alt_overall_bias.pkl")
    _plot_iteration_bias(iteration, overall, "Alt-Target Training Data Bias by Iteration", "gamedata_alt_bias")


# === gamedata BPP / BPPMA ===

def _plot_iteration_bpp(iteration: Series, overall: Series, title: str, fn: str) -> None:
    plot_given(
        title,
        [
            ("Overall Value BPP", overall.x, overall.ys["Overall Value BPP"]),
            ("Overall Value BPP ND", overall.x, overall.ys["Overall Value BPP ND"]),
            ("Overall Policy BPP", overall.x, overall.ys["Overall Policy BPP"]),
            ("Overall Policy BPP NT", overall.x, overall.ys["Overall Policy BPP NT"]),
            ("Value BPP", iteration.x, iteration.ys["Iteration Value BPP"]),
            ("Value BPP ND", iteration.x, iteration.ys["Iteration Value BPP ND"]),
            ("Policy BPP", iteration.x, iteration.ys["Iteration Policy BPP"]),
            ("Policy BPP NT", iteration.x, iteration.ys["Iteration Policy BPP NT"]),
        ],
        "Iterations", "BPP", fn,
    )


def _plot_iteration_bppma(iteration: Series, overall: Series, title: str, fn: str) -> None:
    plot_given(
        title,
        [
            ("Overall Value BPPMA", overall.x, overall.ys["Overall Value BPPMA"]),
            ("Overall Value BPPMA ND", overall.x, overall.ys["Overall Value BPPMA ND"]),
            ("Overall Policy BPPMA", overall.x, overall.ys["Overall Policy BPPMA"]),
            ("Overall Policy BPPMA NT", overall.x, overall.ys["Overall Policy BPPMA NT"]),
            ("Value BPPMA", iteration.x, iteration.ys["Iteration Value BPPMA"]),
            ("Value BPPMA ND", iteration.x, iteration.ys["Iteration Value BPPMA ND"]),
            ("Policy BPPMA", iteration.x, iteration.ys["Iteration Policy BPPMA"]),
            ("Policy BPPMA NT", iteration.x, iteration.ys["Iteration Policy BPPMA NT"]),
        ],
        "Iterations", "BPPMA", fn,
    )


def plot_gamedata_bpp() -> None:
    iteration = load_series(f"{config.eval_dir}/gamedata_bpp.pkl")
    overall = load_series(f"{config.eval_dir}/gamedata_overall_bpp.pkl")
    _plot_iteration_bpp(iteration, overall, "Training Data BPP by Iteration", "gamedata_bpp")


def plot_gamedata_bppma() -> None:
    iteration = load_series(f"{config.eval_dir}/gamedata_bppma.pkl")
    overall = load_series(f"{config.eval_dir}/gamedata_overall_bppma.pkl")
    _plot_iteration_bppma(iteration, overall, "Training Data BPPMA by Iteration", "gamedata_bppma")


def plot_gamedata_alt_bpp() -> None:
    iteration = load_series(f"{config.eval_dir}/gamedata_alt_bpp.pkl")
    overall = load_series(f"{config.eval_dir}/gamedata_alt_overall_bpp.pkl")
    _plot_iteration_bpp(iteration, overall, "Alt-Target Training Data BPP by Iteration", "gamedata_alt_bpp")


def plot_gamedata_alt_bppma() -> None:
    iteration = load_series(f"{config.eval_dir}/gamedata_alt_bppma.pkl")
    overall = load_series(f"{config.eval_dir}/gamedata_alt_overall_bppma.pkl")
    _plot_iteration_bppma(iteration, overall, "Alt-Target Training Data BPPMA by Iteration", "gamedata_alt_bppma")


def plot_value_accuracies() -> None:
    acc          = load_series(f"{config.eval_dir}/gamedata_acc.pkl")
    overall_acc  = load_series(f"{config.eval_dir}/gamedata_overall_acc.pkl")
    alt_acc      = load_series(f"{config.eval_dir}/gamedata_alt_acc.pkl")
    alt_o_acc    = load_series(f"{config.eval_dir}/gamedata_alt_overall_acc.pkl")
    bppma        = load_series(f"{config.eval_dir}/gamedata_bppma.pkl")
    overall_bppma = load_series(f"{config.eval_dir}/gamedata_overall_bppma.pkl")
    alt_bppma    = load_series(f"{config.eval_dir}/gamedata_alt_bppma.pkl")
    alt_o_bppma  = load_series(f"{config.eval_dir}/gamedata_alt_overall_bppma.pkl")
    plot_given_groups(
        "Self-Play Value Accuracies",
        [
            [("Overall Game Outcome Accuracy ND",     overall_acc.x,   overall_acc.ys["Overall Value Accuracy ND"]),
             ("Game Outcome Accuracy ND",              acc.x,           acc.ys["Iteration Value Accuracy ND"])],
            [("Overall Training Data Accuracy ND", alt_o_acc.x,     alt_o_acc.ys["Overall Value Accuracy ND"]),
             ("Training Data Accuracy ND",         alt_acc.x,       alt_acc.ys["Iteration Value Accuracy ND"])],
            [("Overall Game Outcome Majority Class ND",        overall_bppma.x, overall_bppma.ys["Overall Value BPPMA ND"]),
             ("Value Game Outcome Majority Class ND",                bppma.x,         bppma.ys["Iteration Value BPPMA ND"])],
            [("Overall Training Data Majority Class ND",    alt_o_bppma.x,   alt_o_bppma.ys["Overall Value BPPMA ND"]),
             ("Training Data Majority Class ND",            alt_bppma.x,     alt_bppma.ys["Iteration Value BPPMA ND"])],
        ],
        "Iterations", "Value", "gamedata_value_summary",
    )


def plot_policy_accuracies() -> None:
    acc          = load_series(f"{config.eval_dir}/gamedata_acc.pkl")
    overall_acc  = load_series(f"{config.eval_dir}/gamedata_overall_acc.pkl")
    alt_acc      = load_series(f"{config.eval_dir}/gamedata_alt_acc.pkl")
    alt_o_acc    = load_series(f"{config.eval_dir}/gamedata_alt_overall_acc.pkl")
    bppma        = load_series(f"{config.eval_dir}/gamedata_bppma.pkl")
    overall_bppma = load_series(f"{config.eval_dir}/gamedata_overall_bppma.pkl")
    alt_bppma    = load_series(f"{config.eval_dir}/gamedata_alt_bppma.pkl")
    alt_o_bppma  = load_series(f"{config.eval_dir}/gamedata_alt_overall_bppma.pkl")
    plot_given_groups(
        "Self-Play Policy Accuracies",
        [
            [("Overall Player MCTS Policy Accuracy NT",     overall_acc.x,   overall_acc.ys["Overall Policy Accuracy NT"]),
             ("Player MCTS Policy Accuracy NT",              acc.x,           acc.ys["Iteration Policy Accuracy NT"])],
            [("Overall Training Data Accuracy NT", alt_o_acc.x,     alt_o_acc.ys["Overall Policy Accuracy NT"]),
             ("Training Data Accuracy NT",         alt_acc.x,       alt_acc.ys["Iteration Policy Accuracy NT"])],
            [("Overall Player MCTS Policy Majority Class NT",        overall_bppma.x, overall_bppma.ys["Overall Policy BPPMA NT"]),
             ("Player MCTS Policy Majority Class NT",                bppma.x,         bppma.ys["Iteration Policy BPPMA NT"])],
            [("Overall Training Data Majority Class NT",    alt_o_bppma.x,   alt_o_bppma.ys["Overall Policy BPPMA NT"]),
             ("Training Data Majority Class NT",            alt_bppma.x,     alt_bppma.ys["Iteration Policy BPPMA NT"])],
        ],
        "Iterations", "Policy", "gamedata_policy_summary",
    )


# === game progress (over buckets) ===

def plot_gp_state_count() -> None:
    s = load_series(f"{config.eval_dir}/gamedata_10_progress_count.pkl")
    plot_bar(
        "Number of States in Each Game Progress Bucket",
        ("Count", s.x, s.ys["Count"]),
        "Game Progress (%)", "Count", "gp_state_count",
    )


def plot_gp_unique_state_count() -> None:
    s = load_series(f"{config.eval_dir}/gamedata_10_progress_count.pkl")
    plot_bar(
        "Number of Unique States in Each Game Progress Bucket",
        ("Unique", s.x, s.ys["Unique"]),
        "Game Progress (%)", "Unique Count", "gp_unique_state_count",
    )


def _plot_gp_target_accuracy(name: str, label: str, fn: str) -> None:
    s = load_series(f"{config.eval_dir}/{name}_10_progress_acc.pkl")
    plot_given_groups(
        f"{label} Accuracy by Game Progress",
        [
            [
                ("Value Accuracy", s.x, s.ys["Value Accuracy"]),
                ("Value Accuracy ND", s.x, s.ys["Value Accuracy ND"]),
            ],
            [
                ("Policy Accuracy", s.x, s.ys["Policy Accuracy"]),
                ("Policy Accuracy NT", s.x, s.ys["Policy Accuracy NT"]),
                ("Policy PM", s.x, s.ys["Policy PM"]),
                ("Policy PM NT", s.x, s.ys["Policy PM NT"]),
            ],
        ],
        "Game Progress (%)", "Accuracy", fn,
    )


def plot_gp_experienced_accuracy() -> None:
    _plot_gp_target_accuracy("experienced", "Experienced Targets", "gp_experienced_accuracy")


def plot_gp_alt_targets_accuracy() -> None:
    _plot_gp_target_accuracy("alt_targets", "Alt Targets", "gp_alt_targets_accuracy")


def plot_gp_baseline_accuracy() -> None:
    s = load_series(f"{config.eval_dir}/gamedata_10_progress_baseline_accuracy.pkl")
    plot_given_groups(
        "Baseline Accuracy by Game Progress",
        [
            [
                ("Value Accuracy", s.x, s.ys["Value Accuracy"]),
                ("Value Accuracy ND", s.x, s.ys["Value Accuracy ND"]),
            ],
            [
                ("Policy Accuracy", s.x, s.ys["Policy Accuracy"]),
                ("Policy Accuracy NT", s.x, s.ys["Policy Accuracy NT"]),
            ],
        ],
        "Game Progress (%)", "Accuracy", "gp_baseline_accuracy",
    )


def plot_gp_branching_factor() -> None:
    s = load_series(f"{config.eval_dir}/gamedata_10_progress_branching_factor.pkl")
    plot_given(
        "Average Branching Factor by Game Progress",
        [("Branching Factor", s.x, s.ys["Branching Factor"])],
        "Game Progress (%)", "Branching Factor", "gp_branching_factor",
    )


# === per-dataset model evaluation ===

def _plot_dataset_eval(name: str, title: str) -> None:
    s = load_series(f"{config.eval_dir}/{name}_eval.pkl")
    plot_given(
        title,
        [
            ("Loss",            s.x, s.ys["loss"]),
            ("Value Loss",      s.x, s.ys["value_loss"]),
            ("Policy Loss",     s.x, s.ys["policy_loss"]),
            ("Value Accuracy",  s.x, s.ys["value_accuracy"]),
            ("Policy Accuracy", s.x, s.ys["policy_accuracy"]),
        ],
        "Iteration", "Performance", f"{name}_eval",
    )


def plot_seen_nd_eval()   -> None: _plot_dataset_eval("seen_nd",   "Model Performance on Ground Truth (seen, no draws)")
def plot_random_nd_eval() -> None: _plot_dataset_eval("random_nd", "Model Performance on Ground Truth (random, no draws)")
def plot_seen_nt_eval()   -> None: _plot_dataset_eval("seen_nt",   "Model Performance on Ground Truth (seen, non-trivial)")
def plot_random_nt_eval() -> None: _plot_dataset_eval("random_nt", "Model Performance on Ground Truth (random, non-trivial)")


def plot_neighbor_nd_evals() -> None:
    for i in range(2):
        _plot_dataset_eval(f"neighbor_{i+1}_nd", f"Model Performance on Ground Truth (neighbor {i+1}, no draws)")


def plot_neighbor_nt_evals() -> None:
    for i in range(2):
        _plot_dataset_eval(f"neighbor_{i+1}_nt", f"Model Performance on Ground Truth (neighbor {i+1}, non-trivial)")


def plot_experienced_dataset_eval() -> None:
    _plot_dataset_eval("experienced_dataset", "Model Performance on Experienced Targets Dataset")


def plot_alt_targets_dataset_eval() -> None:
    _plot_dataset_eval("alt_targets_dataset", "Model Performance on Alt Targets Dataset")


# === per-bucket dataset evaluation (last model) ===

def plot_game_progress_10_nd_eval() -> None:
    s = load_series(f"{config.eval_dir}/game_progress_10_nd_eval.pkl")
    plot_given(
        "Final Model Accuracy by Game Progress (no draws)",
        [
            ("Value Accuracy", s.x, s.ys["value_accuracy"]),
            ("Policy Accuracy", s.x, s.ys["policy_accuracy"]),
        ],
        "Game Progress (%)", "Accuracy", "game_progress_10_nd_eval",
    )


def plot_game_progress_10_nt_eval() -> None:
    s = load_series(f"{config.eval_dir}/game_progress_10_nt_eval.pkl")
    plot_given(
        "Final Model Accuracy by Game Progress (non-trivial)",
        [
            ("Value Accuracy", s.x, s.ys["value_accuracy"]),
            ("Policy Accuracy", s.x, s.ys["policy_accuracy"]),
        ],
        "Game Progress (%)", "Accuracy", "game_progress_10_nt_eval",
    )


# === combined model + training-data plots ===

def plot_full_value_accuracy() -> None:
    seen      = load_series(f"{config.eval_dir}/seen_nd_eval.pkl")
    random_   = load_series(f"{config.eval_dir}/random_nd_eval.pkl")
    n1        = load_series(f"{config.eval_dir}/neighbor_1_nd_eval.pkl")
    n2        = load_series(f"{config.eval_dir}/neighbor_2_nd_eval.pkl")
    iteration = load_series(f"{config.eval_dir}/gamedata_alt_acc.pkl")
    overall   = load_series(f"{config.eval_dir}/gamedata_alt_overall_acc.pkl")
    plot_given(
        "Value Head Performance on Ground Truth and Training Data",
        [
            ("Seen", seen.x, seen.ys["value_accuracy"]),
            ("Neighbor 1", n1.x, n1.ys["value_accuracy"]),
            ("Neighbor 2", n2.x, n2.ys["value_accuracy"]),
            ("Random", random_.x, random_.ys["value_accuracy"]),
            ("Training Data (Alt)", iteration.x, iteration.ys["Iteration Value Accuracy"]),
            ("Training Data ND (Alt)", iteration.x, iteration.ys["Iteration Value Accuracy ND"]),
            ("Overall Training Data (Alt)", overall.x, overall.ys["Overall Value Accuracy"]),
            ("Overall Training Data ND (Alt)", overall.x, overall.ys["Overall Value Accuracy ND"]),
        ],
        "Iteration", "Accuracy", "full_value_accuracy",
    )


def plot_full_policy_accuracy() -> None:
    seen_nt   = load_series(f"{config.eval_dir}/seen_nt_eval.pkl")
    random_nt = load_series(f"{config.eval_dir}/random_nt_eval.pkl")
    n1_nt     = load_series(f"{config.eval_dir}/neighbor_1_nt_eval.pkl")
    n2_nt     = load_series(f"{config.eval_dir}/neighbor_2_nt_eval.pkl")
    iteration = load_series(f"{config.eval_dir}/gamedata_alt_acc.pkl")
    overall   = load_series(f"{config.eval_dir}/gamedata_alt_overall_acc.pkl")
    plot_given(
        "Policy Head Performance on Ground Truth (NT) and Training Data",
        [
            ("Seen NT", seen_nt.x, seen_nt.ys["policy_accuracy"]),
            ("Neighbor 1 NT", n1_nt.x, n1_nt.ys["policy_accuracy"]),
            ("Neighbor 2 NT", n2_nt.x, n2_nt.ys["policy_accuracy"]),
            ("Random NT", random_nt.x, random_nt.ys["policy_accuracy"]),
            ("Training Data NT (Alt)", iteration.x, iteration.ys["Iteration Policy Accuracy NT"]),
            ("Overall Training Data NT (Alt)", overall.x, overall.ys["Overall Policy Accuracy NT"]),
        ],
        "Iteration", "Accuracy", "full_policy_accuracy",
    )

# === player evaluation ===

def _reference_names(ref: Series) -> list[str]:
    '''Extract reference player names from a reference series' ys keys.'''
    return [key[:-len('_ev_p1')] for key in ref.ys if key.endswith('_ev_p1')]

def plot_ev() -> None:
    player = load_series(f"{config.eval_dir}/player_evaluation_results.pkl")
    ref = load_series(f"{config.eval_dir}/reference_evaluation_results.pkl")
    x = player.x
    n = len(x)
    groups = [
        [
            ("Model (P1)", x, player.ys["ev_p1"]),
            ("Model (P2)", x, player.ys["ev_p2"]),
        ],
    ]
    for name in _reference_names(ref):
        ev_p1 = ref.ys[f"{name}_ev_p1"][0]
        ev_p2 = ref.ys[f"{name}_ev_p2"][0]
        groups.append([
            (f"{name} (P1, EV={ev_p1:.3f})", x, [ev_p1] * n),
            (f"{name} (P2, EV={ev_p2:.3f})", x, [ev_p2] * n),
        ])
    plot_given_groups(
        "Expected Value vs Baseline",
        groups,
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


def plot_wld() -> None:
    s = load_series(f"{config.eval_dir}/player_evaluation_results.pkl")
    x = s.x
    plot_given_groups(
        "Model W/L/D vs Baseline",
        [
            [("Wins (P1)",          x, s.ys["wins_p1"]),
             ("Wins (P2)",          x, s.ys["wins_p2"])],
            [("Losses (P1)",        x, s.ys["losses_p1"]),
             ("Losses (P2)",        x, s.ys["losses_p2"])],
            [("Draws repeat (P1)",  x, s.ys["draws_repeat_p1"]),
             ("Draws repeat (P2)",  x, s.ys["draws_repeat_p2"])],
            [("Draws timeout (P1)", x, s.ys["draws_timeout_p1"]),
             ("Draws timeout (P2)", x, s.ys["draws_timeout_p2"])],
        ],
        "Training Iteration", "Games", "player_wld",
    )


def main():
    setup_logging(level=20, log_dir=config.log_dir, process_name="plotting")
    logger.info("plotting...")

    # distribution series
    # note: each distribution gets 3 plots
    safeplot(plot_dataset_pre_balance_distributions)
    safeplot(plot_dataset_post_balance_distributions)
    safeplot(plot_random_nd_value_distributions)

    # training metrics
    safeplot(plot_training_metrics_accuracy)
    safeplot(plot_training_metrics_with_losses)

    # gamedata stats
    safeplot(plot_gamedata_total_games)
    safeplot(plot_gamedata_outcomes_stacked)
    safeplot(plot_gamedata_outcomes_stacked_proportional)
    safeplot(plot_gamedata_outcomes_lines)
    safeplot(plot_gamedata_game_length)
    safeplot(plot_gamedata_game_time)

    # gamedata accuracy & bias (experienced + alt targets)
    safeplot(plot_gamedata_accuracy)
    safeplot(plot_gamedata_alt_accuracy)
    safeplot(plot_gamedata_bias)
    safeplot(plot_gamedata_alt_bias)
    safeplot(plot_gamedata_bpp)
    safeplot(plot_gamedata_bppma)
    safeplot(plot_gamedata_alt_bpp)
    safeplot(plot_gamedata_alt_bppma)

    # game progress
    safeplot(plot_gp_state_count)
    safeplot(plot_gp_unique_state_count)
    safeplot(plot_gp_experienced_accuracy)
    safeplot(plot_gp_alt_targets_accuracy)
    safeplot(plot_gp_baseline_accuracy)
    safeplot(plot_gp_branching_factor)

    # per-dataset model evaluation
    safeplot(plot_seen_nd_eval)
    safeplot(plot_random_nd_eval)
    safeplot(plot_seen_nt_eval)
    safeplot(plot_random_nt_eval)
    safeplot(plot_neighbor_nd_evals)
    safeplot(plot_neighbor_nt_evals)
    safeplot(plot_experienced_dataset_eval)
    safeplot(plot_alt_targets_dataset_eval)

    # per-bucket dataset evaluation
    safeplot(plot_game_progress_10_nd_eval)
    safeplot(plot_game_progress_10_nt_eval)

    # combined plots
    safeplot(plot_full_value_accuracy)
    safeplot(plot_full_policy_accuracy)
    safeplot(plot_value_accuracies)
    safeplot(plot_policy_accuracies)

    # player evaluation
    safeplot(plot_ev)
    safeplot(plot_wld_p1)
    safeplot(plot_wld_p2)
    safeplot(plot_wld)


if __name__ == "__main__":
    main()
