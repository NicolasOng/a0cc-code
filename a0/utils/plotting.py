import sys
import os

import pickle
from typing import Literal, Optional, overload

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.stats import gaussian_kde

from a0.utils.safe_load import safe_load_pickle

from config import config
from utils.log import get_logger
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

@overload
def load_series(series_path: str) -> Series: ...
@overload
def load_series(series_path: str, *, optional: Literal[False]) -> Series: ...
@overload
def load_series(series_path: str, *, optional: Literal[True]) -> Optional[Series]: ...
def load_series(series_path: str, *, optional: bool = False) -> Optional[Series]:
    '''
    Loads a Series object from the given path.
    By default exits the process if the file is missing (strict mode, used by
    the original eval pipeline). Pass optional=True to instead get None back.
    '''
    series: Optional[Series] = safe_load_pickle(series_path, "series")  # type: ignore[assignment]
    if series is None and not optional:
        logger.error(f"Series file not found at {series_path}. Please generate the series first.")
        sys.exit()
    return series

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

class DistributionSeries:
    '''
    A series of distributions over iterations, e.g. model value predictions
    or dataset value targets at each training iteration.

    Each "trial" is one independent run of the experiment, and contains a
    distribution (list of samples) at each x-value. A single eval run produces
    a series with one trial; merging across runs concatenates the trial dimension.
    '''
    x: list[int]
    trials: list[list[list[float]]]   # trials[t][i] = samples for trial t at x[i]
    name: str

    def __init__(self, name: str = "", num_trials: int = 1):
        self.x = []
        self.trials = [[] for _ in range(num_trials)]
        self.name = name

def save_distribution_series(series: DistributionSeries, series_path: str) -> None:
    '''Saves a DistributionSeries to the given path.'''
    logger.info(f"Saving distribution series to {series_path}...")
    os.makedirs(os.path.dirname(series_path), exist_ok=True)
    with open(series_path, 'wb') as f:
        pickle.dump(series, f)

@overload
def load_distribution_series(series_path: str) -> DistributionSeries: ...
@overload
def load_distribution_series(series_path: str, *, optional: Literal[False]) -> DistributionSeries: ...
@overload
def load_distribution_series(series_path: str, *, optional: Literal[True]) -> Optional[DistributionSeries]: ...
def load_distribution_series(series_path: str, *, optional: bool = False) -> Optional[DistributionSeries]:
    '''
    Loads a DistributionSeries from the given path.
    By default exits the process if the file is missing. Pass optional=True
    to get None back instead.
    '''
    series: Optional[DistributionSeries] = safe_load_pickle(series_path, "distribution series")  # type: ignore[assignment]
    if series is None and not optional:
        logger.error(f"Distribution series file not found at {series_path}. Please generate it first.")
        sys.exit()
    return series

def merge_distribution_series(series: list[DistributionSeries]) -> DistributionSeries:
    '''
    Combines multiple DistributionSeries into one. Each input series can have
    one or more trials; the output series concatenates all trials from all inputs.

    Trials are aligned to the union of x-values across all inputs. For x-values
    a trial doesn't have, an empty distribution [] is inserted as a placeholder
    (the plotting layer should skip empty distributions).
    '''
    all_x = sorted({x for s in series for x in s.x})
    merged = DistributionSeries(name=series[0].name, num_trials=0)
    merged.x = all_x

    for s_idx, s in enumerate(series):
        missing_x = set(all_x) - set(s.x)
        if missing_x:
            logger.info(f"DistributionSeries {s_idx} missing {len(missing_x)} x-values ({missing_x}).")
        x_to_idx = {x_val: i for i, x_val in enumerate(s.x)}
        for trial in s.trials:
            aligned = [trial[x_to_idx[x_val]] if x_val in x_to_idx else [] for x_val in all_x]
            merged.trials.append(aligned)

    return merged

def load_and_merge_distribution_series(dirs: list[str], series_fn: str, save: bool = True) -> DistributionSeries:
    '''Loads and merges distribution series from the specified directories.'''
    logger.info(f"Loading and merging distribution series {series_fn} from directories: {dirs}")
    series: list[DistributionSeries] = []
    for dir in dirs:
        series.append(load_distribution_series(f"{dir}{series_fn}"))

    merged_series = merge_distribution_series(series)

    if save:
        save_distribution_series(merged_series, f"{config.eval_dir}merged_{series_fn}")

    return merged_series

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
    plt.minorticks_on()
    plt.grid(True, which='major', linewidth=0.8)
    plt.grid(True, which='minor', linewidth=0.3, alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/{fn}.png")
    plt.close()

def plot_ridgeline(
    distributions: list[list[float]],
    labels: list[str],
    title: str,
    x_label: str,
    y_label: str,
    fn: str,
    value_range: tuple[float, float] = (-1, 1),
    overlap: float = 0.6,
    bw: float = 0.15,
    grid_points: int = 300,
) -> None:
    '''
    Plot a ridgeline chart: one KDE density line per distribution, stacked vertically.
    Args:
        distributions: list of value lists, one per ridge (bottom to top)
        labels: tick labels for each ridge (same length as distributions)
        title: plot title
        x_label: x-axis label
        y_label: y-axis label
        fn: filename (saved under config.plot_dir)
        value_range: (min, max) for the x-axis and KDE domain
        overlap: vertical spacing between ridges (lower = more overlap)
        bw: KDE bandwidth (passed to gaussian_kde bw_method)
        grid_points: number of points to evaluate the KDE on
    '''
    x_grid = np.linspace(value_range[0], value_range[1], grid_points)
    n = len(distributions)

    plt.figure(figsize=(16, max(6, n * 0.5)))
    for i, values in enumerate(distributions):
        arr = np.array(values)
        kde = gaussian_kde(arr, bw_method=bw)
        density = kde(x_grid)
        density = density / density.max()  # normalize peak to 1
        baseline = i * overlap
        plt.plot(x_grid, baseline + density, color='black', linewidth=1.0, alpha=0.7)

    plt.yticks(
        [i * overlap for i in range(n)],
        labels,
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(value_range[0], value_range[1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}{fn}.png")
    plt.close()

def plot_shaded_ridgeline(
    trials: list[list[list[float]]],
    labels: list[str],
    title: str,
    x_label: str,
    y_label: str,
    fn: str,
    confidence: float = 0.95,
    value_range: tuple[float, float] = (-1, 1),
    overlap: float = 0.6,
    bw: float = 0.15,
    grid_points: int = 300,
) -> None:
    '''
    Plot a ridgeline chart with cross-trial confidence interval bands.
    For each iteration:
      - Compute KDE for each trial (peak-normalized to 1)
      - Plot the mean KDE as a line, with a shaded Student-t CI band
        showing where trials disagree on the density
    Args:
        trials: nested list, trials[t][i] = list of samples for trial t at iteration i.
                A trial may contain [] for missing iterations (skipped).
        labels: tick labels for each iteration (same length as trials[0])
        confidence: confidence level for the CI band (default 0.95)
        value_range: (min, max) for the x-axis and KDE domain
        overlap: vertical spacing between ridges
        bw: KDE bandwidth (passed to gaussian_kde bw_method)
        grid_points: number of points to evaluate the KDE on
    '''
    if not trials or not trials[0]:
        logger.error(f"plot_shaded_ridgeline: empty trials, skipping {fn}")
        return

    num_iters = len(trials[0])
    x_grid = np.linspace(value_range[0], value_range[1], grid_points)

    plt.figure(figsize=(16, max(6, num_iters * 0.5)))

    for i in range(num_iters):
        # one raw KDE per (non-empty) trial at this iteration
        per_trial_density: list[NDArray[np.float64]] = []
        for trial in trials:
            samples = trial[i]
            if not samples:
                continue
            arr = np.array(samples)
            kde = gaussian_kde(arr, bw_method=bw)
            per_trial_density.append(kde(x_grid))

        if not per_trial_density:
            continue

        density_stack = np.stack(per_trial_density)
        # shared per-iteration normalization: divide all trials by the same scale
        # so that peak-height differences between trials are preserved within the ridge.
        shared_max = density_stack.max()
        if shared_max > 0:
            density_stack = density_stack / shared_max
        n = density_stack.shape[0]
        mean = density_stack.mean(axis=0)

        baseline = i * overlap

        # CI band only if we have at least 2 trials
        if n >= 2:
            std = density_stack.std(axis=0, ddof=1)
            alpha = 1 - confidence
            t_value = stats.t.ppf(1 - alpha / 2, df=n - 1)
            ci = t_value * std / np.sqrt(n)
            plt.fill_between(
                x_grid,
                baseline + mean - ci,
                baseline + mean + ci,
                alpha=0.30,
                color="C0",
                linewidth=0,
            )

        plt.plot(x_grid, baseline + mean, color="black", linewidth=1.0, alpha=0.85)

    plt.yticks([i * overlap for i in range(num_iters)], labels)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(value_range[0], value_range[1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}{fn}.png")
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
