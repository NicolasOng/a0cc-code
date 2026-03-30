import numpy as np
from numpy.typing import NDArray
from scipy import stats

from a0.eval.plotting import Series, load_series, save_series, plot_shaded_error

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def merge_series(series: list[Series], confidence: float = 0.95) -> Series:
    '''
    Combines multiple series into one.
    Assumes that all the series share the same y's.
    Uses the union of all x-values across series. Stats (mean, std, CI) at each x
    are computed from whichever series have a value there.
    The resultant series will contain the mean, standard deviation, and confidence interval for each y in the original series.
    '''
    # find the union of all x-values in all the Series objects
    all_x: set = set()
    for s in series:
        all_x |= set(s.x)

    # Sort to maintain order
    all_x_sorted = sorted(all_x)

    # log how many x-values each series is missing
    for i, s in enumerate(series):
        missing_x = set(all_x_sorted) - set(s.x)
        if missing_x:
            logger.info(f"Series {i} missing {len(missing_x)} x-values ({missing_x}).")

    # build aligned arrays with NaN where a series doesn't have a given x
    y_keys = list(series[0].ys.keys())
    num_x = len(all_x_sorted)
    num_series = len(series)

    merged_ys: dict[str, NDArray[np.float64]] = {}
    for y_key in y_keys:
        y_data = np.full((num_series, num_x), np.nan, dtype=np.float64)
        for s_idx, s in enumerate(series):
            x_to_index = {x_val: i for i, x_val in enumerate(s.x)}
            for x_idx, x_val in enumerate(all_x_sorted):
                if x_val in x_to_index:
                    y_data[s_idx, x_idx] = s.ys[y_key][x_to_index[x_val]]
        merged_ys[y_key] = y_data

    # create a merged series object, where each y in the original Series is replaced with
    # average, std, and CI
    merged_series = Series(y_keys + [y_key + "_std" for y_key in y_keys] + [y_key + "_ci" for y_key in y_keys])

    for y_key in y_keys:
        y_data = merged_ys[y_key]
        # count of non-NaN values at each x position
        n = np.sum(~np.isnan(y_data), axis=0)

        # calculate mean and std ignoring NaNs
        mean = np.nanmean(y_data, axis=0)
        std = np.nanstd(y_data, axis=0)

        # Calculate confidence interval (per-x degrees of freedom)
        alpha = 1 - confidence
        # where n < 2 we can't compute a CI; use NaN
        df = np.maximum(n - 1, 1)  # avoid 0 df for t.ppf
        t_value = stats.t.ppf(1 - alpha/2, df=df)
        confidence_interval = np.where(n >= 2, t_value * std / np.sqrt(n), np.nan)

        # put all these into the merged series object
        merged_series.x = list(all_x_sorted)
        merged_series.ys[y_key] = mean.tolist()
        merged_series.ys[y_key + "_std"] = std.tolist()
        merged_series.ys[y_key + "_ci"] = confidence_interval.tolist()

    return merged_series

def load_and_merge_series(dirs: list[str], series_fn: str, confidence: float = 0.95, save: bool = True) -> Series:
    '''
    Loads and merges series from the specified directories.
    '''
    logger.info(f"Loading and merging series {series_fn} from directories: {dirs} with confidence level {confidence}")
    # load all the series, using the provided directories and series filename.
    series: list[Series] = []
    for dir in dirs:
        series.append(load_series(f"{dir}{series_fn}"))

    merged_series = merge_series(series, confidence)

    if save:
        save_series(merged_series, f"{config.eval_dir}merged_{series_fn}")

    return merged_series

def main():
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="combining"
    )

    # create a list of the output dirs to pull from
    eval_dir = "/eval/"
    output_dir = config.output_dir[:-1]
    outputs: list[str] = []
    for i in range(config.num_trials):
        outputs.append(f"{output_dir}{i+1}{eval_dir}")

    # choose the confidence level for all the plots
    confidence = 0.95

    # load and merge all the series (also save them)
    load_and_merge_series(outputs, "training_gtv_eval.pkl", confidence)
    load_and_merge_series(outputs, "random_gtv_eval.pkl", confidence)
    load_and_merge_series(outputs, "training_ev_eval.pkl", confidence)
    for i in range(2):
        load_and_merge_series(outputs, f"neighbor_{i+1}_gtv_eval.pkl", confidence)
    
    load_and_merge_series(outputs, "training_nt_gtv_eval.pkl", confidence)
    load_and_merge_series(outputs, "random_nt_gtv_eval.pkl", confidence)
    load_and_merge_series(outputs, "training_nt_ev_eval.pkl", confidence)
    for i in range(2):
        load_and_merge_series(outputs, f"neighbor_{i+1}_nt_gtv_eval.pkl", confidence)

    load_and_merge_series(outputs, "gamedata_stats.pkl", confidence)
    load_and_merge_series(outputs, "gamedata_acc.pkl", confidence)
    load_and_merge_series(outputs, "gamedata_overall_acc.pkl", confidence)
    load_and_merge_series(outputs, "training_metrics.pkl", confidence)
    load_and_merge_series(outputs, "gamedata_bias.pkl", confidence)
    load_and_merge_series(outputs, "gamedata_overall_bias.pkl", confidence)
    load_and_merge_series(outputs, "gamedata_progress_acc_100.pkl", confidence)
    load_and_merge_series(outputs, "gamedata_progress_acc_10.pkl", confidence)
    load_and_merge_series(outputs, "state_progress_gtv_datasets_eval.pkl", confidence)
    load_and_merge_series(outputs, "state_progress_nt_gtv_datasets_eval.pkl", confidence)
    load_and_merge_series(outputs, "state_progress_state_nums.pkl", confidence)

    num_bins_list = [10]
    for num_bins in num_bins_list:
        load_and_merge_series(outputs, f"num_states_over_game_progress_{num_bins}.pkl", confidence)
        load_and_merge_series(outputs, f"num_unique_states_over_game_progress_{num_bins}.pkl", confidence)
        load_and_merge_series(outputs, f"baseline_accuracy_over_game_progress_{num_bins}.pkl", confidence)
        load_and_merge_series(outputs, f"training_data_accuracy_over_game_progress_{num_bins}.pkl", confidence)
        load_and_merge_series(outputs, f"branching_factor_over_game_progress_{num_bins}.pkl", confidence)
    
    # load the merged series from disk
    # this step is seperated in case I don't want to recalculate all the series
    train_gt_series = load_series(f"{config.eval_dir}/merged_training_gtv_eval.pkl")
    random_gt_series = load_series(f"{config.eval_dir}/merged_random_gtv_eval.pkl")
    train_ev_series = load_series(f"{config.eval_dir}/merged_training_ev_eval.pkl")
    n_neighbor_gt_series: list[Series] = []
    for i in range(2):
        neighbor_gt_series = load_series(f"{config.eval_dir}/merged_neighbor_{i+1}_gtv_eval.pkl")
        n_neighbor_gt_series.append(neighbor_gt_series)

    train_nt_gt_series = load_series(f"{config.eval_dir}/merged_training_nt_gtv_eval.pkl")
    random_nt_gt_series = load_series(f"{config.eval_dir}/merged_random_nt_gtv_eval.pkl")
    train_nt_ev_series = load_series(f"{config.eval_dir}/merged_training_nt_ev_eval.pkl")
    n_neighbor_nt_gt_series: list[Series] = []
    for i in range(2):
        neighbor_nt_gt_series = load_series(f"{config.eval_dir}/merged_neighbor_{i+1}_nt_gtv_eval.pkl")
        n_neighbor_nt_gt_series.append(neighbor_nt_gt_series)

    gamedata_series = load_series(f"{config.eval_dir}/merged_gamedata_stats.pkl")
    gd_accuracy_series = load_series(f"{config.eval_dir}/merged_gamedata_acc.pkl")
    gd_overall_acc_series = load_series(f"{config.eval_dir}/merged_gamedata_overall_acc.pkl")
    training_metrics = load_series(f"{config.eval_dir}/merged_training_metrics.pkl")
    gd_bias_series = load_series(f"{config.eval_dir}/merged_gamedata_bias.pkl")
    gd_overall_bias_series = load_series(f"{config.eval_dir}/merged_gamedata_overall_bias.pkl")
    gd_prog_acc_100 = load_series(f"{config.eval_dir}/merged_gamedata_progress_acc_100.pkl")
    gd_prog_acc_10 = load_series(f"{config.eval_dir}/merged_gamedata_progress_acc_10.pkl")
    state_progress_gtv = load_series(f"{config.eval_dir}/merged_state_progress_gtv_datasets_eval.pkl")
    state_progress_nt_gtv = load_series(f"{config.eval_dir}/merged_state_progress_nt_gtv_datasets_eval.pkl")
    state_progress_state_nums = load_series(f"{config.eval_dir}/merged_state_progress_state_nums.pkl")

    num_bins_list = [10]
    n_num_states_series: list[Series] = []
    n_num_unique_states_series: list[Series] = []
    n_baseline_acc_over_gp_series: list[Series] = []
    n_acc_over_gp_series: list[Series] = []
    n_branching_factor_series: list[Series] = []
    for num_bins in num_bins_list:
        n_num_states_series.append(load_series(f"{config.eval_dir}/merged_num_states_over_game_progress_{num_bins}.pkl"))
        n_num_unique_states_series.append(load_series(f"{config.eval_dir}/merged_num_unique_states_over_game_progress_{num_bins}.pkl"))
        n_baseline_acc_over_gp_series.append(load_series(f"{config.eval_dir}/merged_baseline_accuracy_over_game_progress_{num_bins}.pkl"))
        n_acc_over_gp_series.append(load_series(f"{config.eval_dir}/merged_training_data_accuracy_over_game_progress_{num_bins}.pkl"))
        n_branching_factor_series.append(load_series(f"{config.eval_dir}/merged_branching_factor_over_game_progress_{num_bins}.pkl"))
    
    # try plotting
    plot_shaded_error("Value Head Model Performance on Ground Truth of States and Training Data Accuracy",
                      [
                          ("Seen Accuracy", "±1σ", train_gt_series.x, train_gt_series.ys["value_accuracy"], train_gt_series.ys["value_accuracy_std"]),
                          ("Neighbor 1 Accuracy", "±1σ", n_neighbor_gt_series[0].x, n_neighbor_gt_series[0].ys["value_accuracy"], n_neighbor_gt_series[0].ys["value_accuracy_std"]),
                          ("Neighbor 2 Accuracy", "±1σ", n_neighbor_gt_series[1].x, n_neighbor_gt_series[1].ys["value_accuracy"], n_neighbor_gt_series[1].ys["value_accuracy_std"]),
                          ("Random Accuracy", "±1σ", random_gt_series.x, random_gt_series.ys["value_accuracy"], random_gt_series.ys["value_accuracy_std"]),
                          ("Training Data Accuracy", "±1σ", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Value Accuracy"], gd_accuracy_series.ys["Iteration Value Accuracy_std"]),
                          ("Training Data Overall Accuracy", "±1σ", gd_overall_acc_series.x, gd_overall_acc_series.ys["Overall Value Accuracy"], gd_overall_acc_series.ys["Overall Value Accuracy_std"]),
                      ], "Iterations", "Accuracy", "merged_full_accuracy_value_std")

    plot_shaded_error("Value Head Model Performance on Ground Truth of States and Training Data Accuracy",
                      [
                          ("Seen Accuracy", "±95% CI", train_gt_series.x, train_gt_series.ys["value_accuracy"], train_gt_series.ys["value_accuracy_ci"]),
                          ("Neighbor 1 Accuracy", "±95% CI", n_neighbor_gt_series[0].x, n_neighbor_gt_series[0].ys["value_accuracy"], n_neighbor_gt_series[0].ys["value_accuracy_ci"]),
                          ("Neighbor 2 Accuracy", "±95% CI", n_neighbor_gt_series[1].x, n_neighbor_gt_series[1].ys["value_accuracy"], n_neighbor_gt_series[1].ys["value_accuracy_ci"]),
                          ("Random Accuracy", "±95% CI", random_gt_series.x, random_gt_series.ys["value_accuracy"], random_gt_series.ys["value_accuracy_ci"]),
                          ("Training Data Accuracy", "±95% CI", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Value Accuracy"], gd_accuracy_series.ys["Iteration Value Accuracy_ci"]),
                          ("Training Data Overall Accuracy", "±95% CI", gd_overall_acc_series.x, gd_overall_acc_series.ys["Overall Value Accuracy"], gd_overall_acc_series.ys["Overall Value Accuracy_ci"]),
                          ("Training Data Accuracy ND", "±95% CI", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Value Accuracy ND"], gd_accuracy_series.ys["Iteration Value Accuracy ND_ci"]),
                          ("Training Data Overall Accuracy ND", "±95% CI", gd_overall_acc_series.x, gd_overall_acc_series.ys["Overall Value Accuracy ND"], gd_overall_acc_series.ys["Overall Value Accuracy ND_ci"]),
                      ], "Iterations", "Accuracy", "merged_full_accuracy_value_ci")
    
    plot_shaded_error("Value Head Model Performance on Ground Truth of Non-Trivial States and Training Data Accuracy",
                      [
                          ("Seen Accuracy", "±95% CI", train_nt_gt_series.x, train_nt_gt_series.ys["value_accuracy"], train_nt_gt_series.ys["value_accuracy_ci"]),
                          ("Neighbor 1 Accuracy", "±95% CI", n_neighbor_nt_gt_series[0].x, n_neighbor_nt_gt_series[0].ys["value_accuracy"], n_neighbor_nt_gt_series[0].ys["value_accuracy_ci"]),
                          ("Neighbor 2 Accuracy", "±95% CI", n_neighbor_nt_gt_series[1].x, n_neighbor_nt_gt_series[1].ys["value_accuracy"], n_neighbor_nt_gt_series[1].ys["value_accuracy_ci"]),
                          ("Random Accuracy", "±95% CI", random_nt_gt_series.x, random_nt_gt_series.ys["value_accuracy"], random_nt_gt_series.ys["value_accuracy_ci"]),
                          ("Training Data Accuracy", "±95% CI", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Value Accuracy"], gd_accuracy_series.ys["Iteration Value Accuracy_ci"]),
                          ("Training Data Overall Accuracy", "±95% CI", gd_overall_acc_series.x, gd_overall_acc_series.ys["Overall Value Accuracy"], gd_overall_acc_series.ys["Overall Value Accuracy_ci"]),
                      ], "Iterations", "Accuracy", "merged_full_accuracy_value_nt_ci")
    
    plot_shaded_error("Policy Head Model Performance on Ground Truth of States and Training Data Accuracy",
                      [
                          ("Seen Accuracy", "±95% CI", train_gt_series.x, train_gt_series.ys["policy_accuracy"], train_gt_series.ys["policy_accuracy_ci"]),
                          ("Neighbor 1 Accuracy", "±95% CI", n_neighbor_gt_series[0].x, n_neighbor_gt_series[0].ys["policy_accuracy"], n_neighbor_gt_series[0].ys["policy_accuracy_ci"]),
                          ("Neighbor 2 Accuracy", "±95% CI", n_neighbor_gt_series[1].x, n_neighbor_gt_series[1].ys["policy_accuracy"], n_neighbor_gt_series[1].ys["policy_accuracy_ci"]),
                          ("Random Accuracy", "±95% CI", random_gt_series.x, random_gt_series.ys["policy_accuracy"], random_gt_series.ys["policy_accuracy_ci"]),
                          ("Seen Accuracy NT", "±95% CI", train_nt_gt_series.x, train_nt_gt_series.ys["policy_accuracy"], train_nt_gt_series.ys["policy_accuracy_ci"]),
                          ("Neighbor 1 Accuracy NT", "±95% CI", n_neighbor_nt_gt_series[0].x, n_neighbor_nt_gt_series[0].ys["policy_accuracy"], n_neighbor_nt_gt_series[0].ys["policy_accuracy_ci"]),
                          ("Neighbor 2 Accuracy NT", "±95% CI", n_neighbor_nt_gt_series[1].x, n_neighbor_nt_gt_series[1].ys["policy_accuracy"], n_neighbor_nt_gt_series[1].ys["policy_accuracy_ci"]),
                          ("Random Accuracy NT", "±95% CI", random_nt_gt_series.x, random_nt_gt_series.ys["policy_accuracy"], random_nt_gt_series.ys["policy_accuracy_ci"]),
                          ("Training Data Accuracy", "±95% CI", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Policy Accuracy"], gd_accuracy_series.ys["Iteration Policy Accuracy_ci"]),
                          ("Training Data Overall Accuracy", "±95% CI", gd_overall_acc_series.x, gd_overall_acc_series.ys["Overall Policy Accuracy"], gd_overall_acc_series.ys["Overall Policy Accuracy_ci"]),
                          ("Training Data Accuracy NT", "±95% CI", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Policy Accuracy NT"], gd_accuracy_series.ys["Iteration Policy Accuracy NT_ci"]),
                          ("Training Data Overall Accuracy NT", "±95% CI", gd_overall_acc_series.x, gd_overall_acc_series.ys["Overall Policy Accuracy NT"], gd_overall_acc_series.ys["Overall Policy Accuracy NT_ci"]),
                      ], "Iterations", "Accuracy", "merged_full_full_accuracy_policy_ci")
    
    plot_shaded_error("Policy Head Model Performance on Ground Truth of States and Training Data Accuracy",
                      [
                          #("Seen Accuracy", "±95% CI", train_gt_series.x, train_gt_series.ys["policy_accuracy"], train_gt_series.ys["policy_accuracy_ci"]),
                          #("Neighbor 1 Accuracy", "±95% CI", n_neighbor_gt_series[0].x, n_neighbor_gt_series[0].ys["policy_accuracy"], n_neighbor_gt_series[0].ys["policy_accuracy_ci"]),
                          #("Neighbor 2 Accuracy", "±95% CI", n_neighbor_gt_series[1].x, n_neighbor_gt_series[1].ys["policy_accuracy"], n_neighbor_gt_series[1].ys["policy_accuracy_ci"]),
                          #("Random Accuracy", "±95% CI", random_gt_series.x, random_gt_series.ys["policy_accuracy"], random_gt_series.ys["policy_accuracy_ci"]),
                          ("Seen Accuracy NT", "±95% CI", train_nt_gt_series.x, train_nt_gt_series.ys["policy_accuracy"], train_nt_gt_series.ys["policy_accuracy_ci"]),
                          ("Neighbor 1 Accuracy NT", "±95% CI", n_neighbor_nt_gt_series[0].x, n_neighbor_nt_gt_series[0].ys["policy_accuracy"], n_neighbor_nt_gt_series[0].ys["policy_accuracy_ci"]),
                          ("Neighbor 2 Accuracy NT", "±95% CI", n_neighbor_nt_gt_series[1].x, n_neighbor_nt_gt_series[1].ys["policy_accuracy"], n_neighbor_nt_gt_series[1].ys["policy_accuracy_ci"]),
                          ("Random Accuracy NT", "±95% CI", random_nt_gt_series.x, random_nt_gt_series.ys["policy_accuracy"], random_nt_gt_series.ys["policy_accuracy_ci"]),
                          #("Training Data Accuracy", "±95% CI", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Policy Accuracy"], gd_accuracy_series.ys["Iteration Policy Accuracy_ci"]),
                          #("Training Data Overall Accuracy", "±95% CI", gd_overall_acc_series.x, gd_overall_acc_series.ys["Overall Policy Accuracy"], gd_overall_acc_series.ys["Overall Policy Accuracy_ci"]),
                          ("Training Data Accuracy NT", "±95% CI", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Policy Accuracy NT"], gd_accuracy_series.ys["Iteration Policy Accuracy NT_ci"]),
                          ("Training Data Overall Accuracy NT", "±95% CI", gd_overall_acc_series.x, gd_overall_acc_series.ys["Overall Policy Accuracy NT"], gd_overall_acc_series.ys["Overall Policy Accuracy NT_ci"]),
                      ], "Iterations", "Accuracy", "merged_full_accuracy_policy_ci")
    
    plot_shaded_error("Training Data Value Accuracy Over Game Progress (10 & 100 bins)",
                      [
                            ("Value Accuracy (100 bins)", "±95% CI", gd_prog_acc_100.x, gd_prog_acc_100.ys["Value Accuracy"], gd_prog_acc_100.ys["Value Accuracy_ci"]),
                            ("Value Accuracy (10 bins)", "±95% CI", gd_prog_acc_10.x, gd_prog_acc_10.ys["Value Accuracy"], gd_prog_acc_10.ys["Value Accuracy_ci"]),
                      ], "Game Progress (%)", "Accuracy", "merged_gamedata_progress_acc_ci")

    plot_shaded_error("Model Accuracy on Ground Truth Value by State Progress",
               [
                   ("Value Accuracy", "±95% CI", state_progress_gtv.x, state_progress_gtv.ys["value_accuracy"], state_progress_gtv.ys["value_accuracy_ci"]),
                   #("Policy Accuracy", state_progress_gtv.x, state_progress_gtv.ys["policy_accuracy"])
               ], "State Progress (%)", "Accuracy", "state_progress_gtv")
    
    plot_shaded_error("Number of States in Each Progress Bin",
            [
                ("Num States", "±95% CI", state_progress_state_nums.x, state_progress_state_nums.ys["Num States"], state_progress_state_nums.ys["Num States_ci"]),
                #("Num States NT", "±95% CI", state_progress_state_nums.x, state_progress_state_nums.ys["Num States NT"], state_progress_state_nums.ys["Num States NT_ci"]),
            ], "State Progress (%)", "Number of States", "state_progress_num_states")

    for i, num_bin in enumerate(num_bins_list):
        plot_shaded_error(f"Number of States and Unique States Over Game Progress (Num Bins: {num_bin})",
                   [
                       ("Num States", "±95% CI", n_num_states_series[i].x, n_num_states_series[i].ys["Num States"], n_num_states_series[i].ys["Num States_ci"]),
                       ("Num Unique States", "±95% CI", n_num_unique_states_series[i].x, n_num_unique_states_series[i].ys["Num Unique States"], n_num_unique_states_series[i].ys["Num Unique States_ci"])
                   ], "Game Progress (%)", "Number of States", f"gp_merged_num_states_over_game_progress_{num_bin}")
        
        plot_shaded_error(f"Baseline Accuracy Over Game Progress (Num Bins: {num_bin})",
                   [
                        ("Value Accuracy", "±95% CI", n_baseline_acc_over_gp_series[i].x, n_baseline_acc_over_gp_series[i].ys["Baseline Value Accuracy"], n_baseline_acc_over_gp_series[i].ys["Baseline Value Accuracy_ci"]),
                        ("Value Accuracy (ND)", "±95% CI", n_baseline_acc_over_gp_series[i].x, n_baseline_acc_over_gp_series[i].ys["Baseline Value Accuracy (ND)"], n_baseline_acc_over_gp_series[i].ys["Baseline Value Accuracy (ND)_ci"]),
                        ("Policy Accuracy", "±95% CI", n_baseline_acc_over_gp_series[i].x, n_baseline_acc_over_gp_series[i].ys["Baseline Policy Accuracy"], n_baseline_acc_over_gp_series[i].ys["Baseline Policy Accuracy_ci"]),
                        ("Policy Accuracy (NT)", "±95% CI", n_baseline_acc_over_gp_series[i].x, n_baseline_acc_over_gp_series[i].ys["Baseline Policy Accuracy (NT)"], n_baseline_acc_over_gp_series[i].ys["Baseline Policy Accuracy (NT)_ci"])
                   ], "Game Progress (%)", "Baseline Accuracy", f"gp_merged_baseline_accuracy_over_game_progress_{num_bin}")
        
        plot_shaded_error(f"Training Data Accuracy Over Game Progress (Num Bins: {num_bin})",
                   [
                        ("Value Accuracy", "±95% CI", n_acc_over_gp_series[i].x, n_acc_over_gp_series[i].ys["Value Accuracy"], n_acc_over_gp_series[i].ys["Value Accuracy_ci"]),
                        ("Value Accuracy (ND)", "±95% CI", n_acc_over_gp_series[i].x, n_acc_over_gp_series[i].ys["Value Accuracy (ND)"], n_acc_over_gp_series[i].ys["Value Accuracy (ND)_ci"]),
                        ("Policy Accuracy", "±95% CI", n_acc_over_gp_series[i].x, n_acc_over_gp_series[i].ys["Policy Accuracy"], n_acc_over_gp_series[i].ys["Policy Accuracy_ci"]),
                        ("Policy Accuracy (NT)", "±95% CI", n_acc_over_gp_series[i].x, n_acc_over_gp_series[i].ys["Policy Accuracy (NT)"], n_acc_over_gp_series[i].ys["Policy Accuracy (NT)_ci"]),
                   ], "Game Progress (%)", "Training Data Accuracy", f"gp_merged_training_data_accuracy_over_game_progress_{num_bin}")
        
        plot_shaded_error(f"Branching Factor Over Game Progress (Num Bins: {num_bin})",
                   [
                       ("Average Branching Factor", "±95% CI", n_branching_factor_series[i].x, n_branching_factor_series[i].ys["Average Branching Factor"], n_branching_factor_series[i].ys["Average Branching Factor_ci"])
                   ], "Game Progress (%)", "Branching Factor", f"gp_merged_branching_factor_over_game_progress_{num_bin}")

if __name__ == "__main__":
    main()