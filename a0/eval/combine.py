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
    If the series have different x-values, only the intersection of all the x's will be kept.
    The resultant series will contain the mean, standard deviation, and confidence interval for each y in the original series.
    '''
    # find the intersection of all x-values in all the Series objects
    common_x = set(series[0].x)
    for s in series[1:]:
        common_x &= set(s.x)
    
    # Sort to maintain order
    common_x = sorted(common_x)

    # log how many x-values are lost in each series.
    for i, s in enumerate(series):
        lost_x = set(s.x) - set(common_x)
        logger.info(f"Series {i} lost {len(lost_x)} x-values ({lost_x}).")

    # create a list of new, aligned Series objects
    aligned_series: list[Series] = []
    for s in series:
        # Create new Series with same y keys
        new_series = Series(list(s.ys.keys()))
        new_series.x = common_x.copy()
        
        # Create mapping from x value to index in original series
        x_to_index = {x_val: i for i, x_val in enumerate(s.x)}
        
        # Copy y values for common x values only
        for y_key in s.ys:
            new_series.ys[y_key] = [
                s.ys[y_key][x_to_index[x_val]] 
                for x_val in common_x
            ]
        
        aligned_series.append(new_series)
    
    # for each y, create a numpy array
    y_keys = list(aligned_series[0].ys.keys())
    merged_ys: dict[str, NDArray[np.float32]] = {}
    for y_key in y_keys:
        y_data = np.array([s.ys[y_key] for s in aligned_series])
        merged_ys[y_key] = y_data
    
    # create a merged series object, where each y in the original Series is replaced with
    # average, std, and CI
    merged_series = Series(y_keys + [y_key + "_std" for y_key in y_keys] + [y_key + "_ci" for y_key in y_keys])

    for y_key in y_keys:
        y_data = merged_ys[y_key]
        num_merged_series = y_data.shape[0]
        # calculate mean
        mean = np.mean(y_data, axis=0)

        # calculate standard deviation
        std = np.std(y_data, axis=0)

        # Calculate confidence interval
        alpha = 1 - confidence
        t_value = stats.t.ppf(1 - alpha/2, df=num_merged_series-1)
        confidence_interval = t_value * std / np.sqrt(num_merged_series)

        # put all these into the merged series object
        merged_series.x = common_x.copy()
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
    outputs = [f"output1{eval_dir}", f"output2{eval_dir}", f"output3{eval_dir}", f"output4{eval_dir}"]

    # choose the confidence level for all the plots
    confidence = 0.95

    # load and merge all the series (also save them)
    load_and_merge_series(outputs, "training_gtv_eval.pkl", confidence)
    load_and_merge_series(outputs, "random_gtv_eval.pkl", confidence)
    load_and_merge_series(outputs, "training_ev_eval.pkl", confidence)
    for i in range(2):
        load_and_merge_series(outputs, f"neighbor_{i+1}_gtv_eval.pkl", confidence)

    load_and_merge_series(outputs, "gamedata_stats.pkl", confidence)
    load_and_merge_series(outputs, "gamedata_acc.pkl", confidence)
    load_and_merge_series(outputs, "gamedata_overall_acc.pkl", confidence)
    load_and_merge_series(outputs, "training_metrics.pkl", confidence)

    # load the merged series from disk
    # this step is seperated in case I don't want to recalculate all the series
    train_gt_series = load_series(f"{config.eval_dir}/merged_training_gtv_eval.pkl")
    random_gt_series = load_series(f"{config.eval_dir}/merged_random_gtv_eval.pkl")
    train_ev_series = load_series(f"{config.eval_dir}/merged_training_ev_eval.pkl")
    n_neighbor_gt_series: list[Series] = []
    for i in range(2):
        neighbor_gt_series = load_series(f"{config.eval_dir}/merged_neighbor_{i+1}_gtv_eval.pkl")
        n_neighbor_gt_series.append(neighbor_gt_series)

    gamedata_series = load_series(f"{config.eval_dir}/merged_gamedata_stats.pkl")
    gd_accuracy_series = load_series(f"{config.eval_dir}/merged_gamedata_acc.pkl")
    gd_overall_acc_series = load_series(f"{config.eval_dir}/merged_gamedata_overall_acc.pkl")
    training_metrics = load_series(f"{config.eval_dir}/merged_training_metrics.pkl")

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
                      ], "Iterations", "Accuracy", "merged_full_accuracy_value_ci")
    
    plot_shaded_error("Policy Head Model Performance on Ground Truth of States and Training Data Accuracy",
                      [
                          ("Seen Accuracy", "±95% CI", train_gt_series.x, train_gt_series.ys["policy_accuracy"], train_gt_series.ys["policy_accuracy_ci"]),
                          ("Neighbor 1 Accuracy", "±95% CI", n_neighbor_gt_series[0].x, n_neighbor_gt_series[0].ys["policy_accuracy"], n_neighbor_gt_series[0].ys["policy_accuracy_ci"]),
                          ("Neighbor 2 Accuracy", "±95% CI", n_neighbor_gt_series[1].x, n_neighbor_gt_series[1].ys["policy_accuracy"], n_neighbor_gt_series[1].ys["policy_accuracy_ci"]),
                          ("Random Accuracy", "±95% CI", random_gt_series.x, random_gt_series.ys["policy_accuracy"], random_gt_series.ys["policy_accuracy_ci"]),
                          ("Training Data Accuracy", "±95% CI", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Policy Accuracy"], gd_accuracy_series.ys["Iteration Policy Accuracy_ci"]),
                          ("Training Data Overall Accuracy", "±95% CI", gd_overall_acc_series.x, gd_overall_acc_series.ys["Overall Policy Accuracy"], gd_overall_acc_series.ys["Overall Policy Accuracy_ci"]),
                          ("Training Data PM", "±95% CI", gd_accuracy_series.x, gd_accuracy_series.ys["Iteration Policy PM"], gd_accuracy_series.ys["Iteration Policy PM_ci"]),
                          ("Training Data Overall PM", "±95% CI", gd_overall_acc_series.x, gd_overall_acc_series.ys["Overall Policy PM"], gd_overall_acc_series.ys["Overall Policy PM_ci"])
                      ], "Iterations", "Accuracy", "merged_full_accuracy_policy_ci")

if __name__ == "__main__":
    main()