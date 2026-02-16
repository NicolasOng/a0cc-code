import sys
import os

import pickle
import matplotlib.pyplot as plt
import numpy as np

from a0.eval.plotting import Series, load_series, plot_given, plot_given_groups, plot_stacked, plot_stacked_proportional, plot_std_error, plot_shaded_error, plot_bar

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def plot_seen_random():
    try:
        seen_nd_series = load_series(f"{config.eval_dir}/eval_seen_nd.pkl")
        seen_nt_series = load_series(f"{config.eval_dir}/eval_seen_nt.pkl")
        random_nd_series = load_series(f"{config.eval_dir}/eval_random_nd.pkl")
        random_nt_series = load_series(f"{config.eval_dir}/eval_random_nt.pkl")
    except Exception as e:
        logger.error(f"Failed to load series for seen vs random plot: {e}")
        return

    plot_given("Value Head Model Performance on Ground Truth of States and Training Data Accuracy",
    [
        ("Seen Accuracy", seen_nd_series.x, seen_nd_series.ys["value_accuracy"]),
        ("Random Accuracy", random_nd_series.x, random_nd_series.ys["value_accuracy"])
    ],
    "Iteration", "Accuracy", "full_accuracy_value")

    plot_given("Policy Head Model Performance on Ground Truth of States and Training Data Accuracy",
    [
        ("Seen Accuracy", seen_nt_series.x, seen_nt_series.ys["policy_accuracy"]),
        ("Random Accuracy", random_nt_series.x, random_nt_series.ys["policy_accuracy"])
    ],
    "Iteration", "Accuracy", "full_accuracy_policy")

def main():
    # I should split this main function into many smaller ones,
    # each of which can fail independently if a series is missing.
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="plotting"
    )
    logger.info("plotting...")

    plot_seen_random()

if __name__ == "__main__":
    main()
