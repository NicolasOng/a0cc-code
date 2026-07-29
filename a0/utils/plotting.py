import sys
import os

import pickle
import warnings
from typing import Literal, Optional, overload

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from scipy.stats import gaussian_kde

from a0.utils.safe_load import safe_load_pickle

from config import config
from utils.log import get_logger
logger = get_logger(__name__)


def _density_from_samples(
    samples: list[float] | NDArray[np.float64],
    x_grid: NDArray[np.float64],
    value_range: tuple[float, float],
    bw: float,
    num_buckets: int,
) -> NDArray[np.float64]:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return np.zeros_like(x_grid)
    if x_grid.size < 2:
        return np.ones_like(x_grid)

    hist, edges = np.histogram(arr, bins=num_buckets, range=value_range, density=False)
    bin_width = edges[1] - edges[0] if len(edges) > 1 else 1.0
    density = hist.astype(np.float64) / max(arr.size * bin_width, 1e-12)

    sigma_bins = max(0.25, bw * num_buckets * 0.2)
    smoothed = gaussian_filter1d(density, sigma=sigma_bins, mode="nearest")
    centers = 0.5 * (edges[:-1] + edges[1:])
    interpolated = np.interp(
        x_grid,
        centers,
        smoothed,
        left=float(smoothed[0]),
        right=float(smoothed[-1]),
    )
    if interpolated.max() <= 0:
        fallback_center = float(np.clip(np.median(arr), value_range[0], value_range[1]))
        fallback_idx = int(np.argmin(np.abs(x_grid - fallback_center)))
        interpolated[fallback_idx] = 1.0
    return interpolated

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
    By default raises FileNotFoundError if the file is missing (strict mode);
    pass optional=True to get None back instead.
    '''
    series: Optional[Series] = safe_load_pickle(series_path, "series")  # type: ignore[assignment]
    if series is None and not optional:
        raise FileNotFoundError(f"Series file not found at {series_path}.")
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

        # calculate mean and std ignoring NaNs. An all-NaN column is normal for
        # heatmap series (they emit a fixed D0..D_MAX key set, NaN where no run
        # reached that distance) and NaN is the correct answer there, so silence
        # numpy's "Mean of empty slice" rather than let it spam a real run.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
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
    Loads and merges series from the specified directories. Trial dirs that
    don't contain series_fn are skipped with a warning; raises FileNotFoundError
    only if no trial has the file.
    '''
    logger.info(f"Loading and merging series {series_fn} from directories: {dirs} with confidence level {confidence}")
    series: list[Series] = []
    missing: list[str] = []
    for dir in dirs:
        s = load_series(f"{dir}{series_fn}", optional=True)
        if s is None:
            missing.append(dir)
        else:
            series.append(s)
    if missing:
        logger.warning(f"load_and_merge_series: {series_fn} missing from {len(missing)}/{len(dirs)} trial dirs: {missing}")
    if not series:
        raise FileNotFoundError(f"Series file {series_fn} not found in any of {dirs}.")

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
    '''Saves a DistributionSeries to the given path.

    Each trial's per-iteration sample list is randomly subsampled to at most
    config.distribution_max_samples (0 = keep all). The distribution plots
    histogram the samples anyway, so a representative subsample preserves them
    while keeping these pkls small (raw targets are ~5k samples/iteration).
    '''
    cap = getattr(config, "distribution_max_samples", 0) or 0
    if cap > 0:
        for trial in series.trials:
            for i, samples in enumerate(trial):
                if samples is not None and len(samples) > cap:
                    arr = np.asarray(samples)
                    sel = np.random.choice(arr.shape[0], cap, replace=False)
                    trial[i] = arr[sel].tolist()
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
    By default raises FileNotFoundError if the file is missing; pass
    optional=True to get None back instead.
    '''
    series: Optional[DistributionSeries] = safe_load_pickle(series_path, "distribution series")  # type: ignore[assignment]
    if series is None and not optional:
        raise FileNotFoundError(f"Distribution series file not found at {series_path}.")
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
    '''Loads and merges distribution series from the specified directories.
    Trial dirs that don't contain series_fn are skipped with a warning; raises
    FileNotFoundError only if no trial has the file.'''
    logger.info(f"Loading and merging distribution series {series_fn} from directories: {dirs}")
    series: list[DistributionSeries] = []
    missing: list[str] = []
    for dir in dirs:
        s = load_distribution_series(f"{dir}{series_fn}", optional=True)
        if s is None:
            missing.append(dir)
        else:
            series.append(s)
    if missing:
        logger.warning(f"load_and_merge_distribution_series: {series_fn} missing from {len(missing)}/{len(dirs)} trial dirs: {missing}")
    if not series:
        raise FileNotFoundError(f"Distribution series file {series_fn} not found in any of {dirs}.")

    merged_series = merge_distribution_series(series)

    if save:
        save_distribution_series(merged_series, f"{config.eval_dir}merged_{series_fn}")

    return merged_series

def plot_given(title: str, series: list[tuple[str, list[int], list[float]]], x_label: str, y_label: str, fn: str, y_lim: tuple[float, float] | None = None) -> None:
    plt.figure(figsize=(16, 9))
    for label, x_values, y_values in series:
        plt.plot(x_values, y_values, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both')
    if y_lim is not None:
        plt.ylim(*y_lim)
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}{fn}.png")
    plt.close()

def plot_given_groups(title: str, groups: list[list[tuple[str, list[int], list[float]]]], x_label: str, y_label: str, fn: str, use_log_y: bool = False, y_lim: tuple[float, float] | None = None) -> None:
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
    if y_lim is not None:
        plt.ylim(*y_lim)
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

def plot_shaded_error_groups(title: str, groups: list[list[tuple[str, str, list[int], list[float], list[float]]]], x_label: str, y_label: str, fn: str, use_log_y: bool = False, y_lim: tuple[float, float] | None = None):
    '''
    Grouped variant of plot_shaded_error: each group shares a color, members
    within a group are differentiated by linestyle. Mirrors plot_given_groups
    but with shaded CI bands per series.
    Each tuple is (line_label, fill_label, x, y, fill).
    '''
    plt.figure(figsize=(16, 9))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    line_styles = ['-', '--', '-.', ':']
    for group_idx, group in enumerate(groups):
        color = colors[group_idx % len(colors)]
        for series_idx, (line_label, fill_label, x_values, y_values, fill_values) in enumerate(group):
            line_style = line_styles[series_idx % len(line_styles)]
            plt.plot(x_values, y_values, label=line_label, color=color, linestyle=line_style)
            plt.fill_between(
                x_values,
                np.array(y_values) - np.array(fill_values),
                np.array(y_values) + np.array(fill_values),
                alpha=0.2, color=color, label=fill_label,
            )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    if use_log_y:
        plt.yscale('log')
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.minorticks_on()
    plt.grid(True, which='major', linewidth=0.8)
    plt.grid(True, which='minor', linewidth=0.3, alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/{fn}.png")
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
    height: float = 1.8,
    bw: float = 0.15,
    grid_points: int = 300,
    density_method: Literal["kde", "buckets"] = "buckets",
    num_buckets: int = 20,
) -> None:
    '''
    Plot a ridgeline chart with cross-trial confidence interval bands.
    For each iteration:
            - Compute a smoothed histogram for each trial (peak-normalized to 1)
            - Plot the mean density as a line, with a shaded Student-t CI band
        showing where trials disagree on the density
    Args:
        trials: nested list, trials[t][i] = list of samples for trial t at iteration i.
                A trial may contain [] for missing iterations (skipped).
        labels: tick labels for each iteration (same length as trials[0])
        confidence: confidence level for the CI band (default 0.95)
        value_range: (min, max) for the x-axis and density domain
        overlap: vertical spacing between ridges
        bw: smoothing strength for the histogram blur
        grid_points: number of points to evaluate the density on
        density_method: "kde" for gaussian_kde, "buckets" for smoothed histogram
        num_buckets: number of histogram buckets for density_method="buckets"
    '''
    if not trials or not trials[0]:
        logger.error(f"plot_shaded_ridgeline: empty trials, skipping {fn}")
        return

    num_iters = len(trials[0])
    x_grid = np.linspace(value_range[0], value_range[1], grid_points)

    plt.figure(figsize=(8, max(6, num_iters * 0.5)))

    for i in range(num_iters):
        # one raw smoothed histogram per (non-empty) trial at this iteration
        per_trial_density: list[NDArray[np.float64]] = []
        for trial in trials:
            samples = trial[i]
            if not samples:
                continue
            if density_method == "kde":
                arr = np.asarray(samples, dtype=np.float64)
                kde = gaussian_kde(arr, bw_method=bw)
                per_trial_density.append(kde(x_grid))
            elif density_method == "buckets":
                per_trial_density.append(_density_from_samples(samples, x_grid, value_range, bw, num_buckets))
            else:
                raise ValueError(f"Unknown density_method={density_method}; expected 'kde' or 'buckets'")

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

        # subtle horizontal floor line at this iteration's baseline
        plt.plot([value_range[0], value_range[1]], [baseline, baseline],
                 color="grey", linewidth=0.5, alpha=0.4, zorder=0)

        # CI band only if we have at least 2 trials
        if n >= 2:
            std = density_stack.std(axis=0, ddof=1)
            alpha = 1 - confidence
            t_value = stats.t.ppf(1 - alpha / 2, df=n - 1)
            ci = t_value * std / np.sqrt(n)
            plt.fill_between(
                x_grid,
                baseline + (mean - ci) * height,
                baseline + (mean + ci) * height,
                alpha=0.30,
                color="C0",
                linewidth=0,
            )

        plt.plot(x_grid, baseline + mean * height, color="black", linewidth=1.0, alpha=0.85)

    plt.yticks([i * overlap for i in range(num_iters)], labels)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(value_range[0], value_range[1])
    plt.ylim(0, (num_iters - 1) * overlap + height)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}{fn}.png")
    plt.close()

def _pool_samples_per_iter(trials: list[list[list[float]]]) -> list[list[float]]:
    '''Pool samples across trials at each iteration. Empty trial slots skipped.'''
    num_iters = len(trials[0])
    pooled: list[list[float]] = [[] for _ in range(num_iters)]
    for trial in trials:
        for i in range(num_iters):
            if trial[i]:
                pooled[i].extend(trial[i])
    return pooled


def plot_percentile_bands(
    trials: list[list[list[float]]],
    labels: list[str],
    title: str,
    x_label: str,
    y_label: str,
    fn: str,
    value_range: tuple[float, float] = (-1, 1),
) -> None:
    '''
    Percentile-band + mean overlay summary of a DistributionSeries.
    For each iteration, pools samples across all trials and plots:
        - shaded 5-95% band (outer)
        - shaded 25-75% IQR band (inner)
        - median line (solid)
        - mean line (dashed)
    Robust to non-normal cases (bimodal, degenerate, discrete).
    '''
    if not trials or not trials[0]:
        logger.error(f"plot_percentile_bands: empty trials, skipping {fn}")
        return

    pooled = _pool_samples_per_iter(trials)
    num_iters = len(pooled)
    positions = np.arange(num_iters)

    pct = np.full((num_iters, 5), np.nan)
    mean = np.full(num_iters, np.nan)
    for i, samples in enumerate(pooled):
        if not samples:
            continue
        arr = np.asarray(samples, dtype=np.float64)
        pct[i] = np.percentile(arr, [5, 25, 50, 75, 95])
        mean[i] = arr.mean()

    valid = ~np.isnan(mean)
    if not valid.any():
        logger.error(f"plot_percentile_bands: no non-empty iterations, skipping {fn}")
        return

    x = positions[valid]
    p5, p25, p50, p75, p95 = (pct[valid, k] for k in range(5))

    plt.figure(figsize=(16, 9))
    plt.fill_between(x, p5, p95, color="C0", alpha=0.18, label="5-95%")
    plt.fill_between(x, p25, p75, color="C0", alpha=0.38, label="25-75% (IQR)")
    plt.plot(x, p50, color="black", linewidth=1.6, label="median")
    plt.plot(x, mean[valid], color="crimson", linewidth=1.4, linestyle="--", label="mean")

    plt.xticks(positions, labels, rotation=45 if len(labels) > 15 else 0)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    margin = 0.05 * (value_range[1] - value_range[0])
    plt.ylim(value_range[0] - margin, value_range[1] + margin)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}{fn}.png")
    plt.close()


def plot_value_proportions(
    trials: list[list[list[float]]],
    labels: list[str],
    title: str,
    x_label: str,
    y_label: str,
    fn: str,
    value_range: tuple[float, float] = (-1, 1),
    n_bins: int = 10,
) -> None:
    '''
    Stacked value-proportion summary of a DistributionSeries.
    For each iteration, pools samples across trials and shows the fraction
    falling into each of n_bins equal-width bins over value_range.
    Bins are colored by bin center (RdBu_r: blue = negative, red = positive),
    with a colorbar. Best view for discrete or bimodal distributions.
    '''
    if not trials or not trials[0]:
        logger.error(f"plot_value_proportions: empty trials, skipping {fn}")
        return

    pooled = _pool_samples_per_iter(trials)
    num_iters = len(pooled)
    positions = np.arange(num_iters)
    edges = np.linspace(value_range[0], value_range[1], n_bins + 1)

    props = np.zeros((n_bins, num_iters))
    valid = np.zeros(num_iters, dtype=bool)
    for i, samples in enumerate(pooled):
        if not samples:
            continue
        arr = np.clip(np.asarray(samples, dtype=np.float64), value_range[0], value_range[1])
        counts, _ = np.histogram(arr, bins=edges)
        props[:, i] = counts / arr.size
        valid[i] = True

    if not valid.any():
        logger.error(f"plot_value_proportions: no non-empty iterations, skipping {fn}")
        return

    cmap = plt.get_cmap("RdBu_r")
    norm = plt.Normalize(vmin=value_range[0], vmax=value_range[1])
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_colors = [cmap(norm(c)) for c in bin_centers]

    x = positions[valid]
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.stackplot(x, props[:, valid], colors=bin_colors, edgecolor="none")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, 1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45 if len(labels) > 15 else 0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("bin center (value)")
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


def plot_bar_with_error(title: str, series: tuple[str, list[int], list[float], list[float]], x_label: str, y_label: str, fn: str) -> None:
    '''Bar chart with symmetric error bars. The series tuple is (label, x, y, error).'''
    plt.figure(figsize=(16, 9))
    label, x_values, y_values, error_values = series
    plt.bar(x_values, y_values, yerr=error_values, label=label, capsize=5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/{fn}.png")
    plt.close()

# ---------- heatmaps ----------
# Series produced by AccuracyHeatmapCollector are 2D flattened into one key per
# (metric, bin): "<metric> D<distance>" / "<metric> P<progress bucket>", with x =
# iteration. That layout is deliberate — it keeps save_series/merge_series and the
# whole sweep-combine path working unchanged — so the pivot back to a matrix
# happens here, at plot time. Cap, bin width and count masking are applied here
# too, so re-cutting a heatmap never re-runs the GT-bound analysis pass.

# "further from the terminal than D_MAX" row, matching accuracy_heatmap.OVERFLOW
HEATMAP_OVERFLOW = "OVER"

def _heatmap_bins(series: "Series", metric: str, label: str, key_suffix: str = "") -> list[int | str]:
    '''
    The bins present for `metric`, numerically sorted, overflow last.

    `key_suffix` covers the merged series: merge_series appends "_std"/"_ci" to
    the WHOLE key, so a merged CI key is "Value Accuracy ND D5_ci" — the bin is
    in the middle, not at the end.
    '''
    prefix = f"{metric} {label}"
    suffixes = [
        key[len(prefix):-len(key_suffix)] if key_suffix else key[len(prefix):]
        for key in series.ys
        if key.startswith(prefix) and key.endswith(key_suffix)
    ]
    numeric = sorted(int(s) for s in suffixes if s.isdigit())
    bins: list[int | str] = list(numeric)
    if HEATMAP_OVERFLOW in suffixes:
        bins.append(HEATMAP_OVERFLOW)
    return bins

def heatmap_matrix(
    series: "Series",
    metric: str,
    label: str,
    min_count: int | None = None,
    count_metric: str | None = None,
    drop_empty_rows: bool = True,
    key_suffix: str = "",
) -> tuple[NDArray[np.float64], list[int | str]]:
    '''
    Pivots a flattened heatmap Series into (n_bins, n_iterations), y ascending.

    Cells with fewer than `min_count` samples are set to NaN so they render as
    missing rather than as a confident-looking value computed from a handful of
    samples. `drop_empty_rows` then trims trailing rows that no iteration
    populates — which is what implements the agreed cap: the last row kept is
    the furthest distance meeting `min_count` in EVERY iteration, so no row is
    ragged. Rows are never dropped from the middle.
    '''
    bins = _heatmap_bins(series, metric, label, key_suffix)
    if not bins:
        raise KeyError(f"no keys matching '{metric} {label}*{key_suffix}' in series")

    matrix = np.array([series.ys[f"{metric} {label}{b}{key_suffix}"] for b in bins], dtype=np.float64)

    if min_count is not None:
        key = count_metric or ("Count ND" if metric.endswith("ND") else "Count")
        # counts are always the plain merged mean, never the _std/_ci variant
        counts = np.array([series.ys[f"{key} {label}{b}"] for b in bins], dtype=np.float64)
        matrix = np.where(counts >= min_count, matrix, np.nan)

    if drop_empty_rows:
        populated = ~np.all(np.isnan(matrix), axis=1)
        if populated.any():
            # keep the overflow row only if it survived on its own merit
            last = int(np.max(np.flatnonzero(populated)))
            matrix = matrix[: last + 1]
            bins = bins[: last + 1]
    return matrix, bins

def plot_heatmap(
    title: str,
    series: "Series",
    metric: str,
    label: str,
    x_label: str,
    y_label: str,
    fn: str,
    min_count: int | None = None,
    diverging_center: float | None = None,
    v_lim: tuple[float, float] | None = None,
    cbar_label: str | None = None,
    key_suffix: str = "",
) -> None:
    '''
    Renders one flattened heatmap Series as a matrix.

    Colour follows the job the numbers do, not taste:
      diverging_center set -> polarity. Two opposite hues with a neutral grey
        midpoint pinned to the baseline (0.5 = chance for a binary win/loss sign
        call), so "better than a coin flip" is a hue flip rather than a shade the
        reader has to compare against a colourbar.
      otherwise         -> magnitude. One hue, light to dark. Never a rainbow.
    Masked (low-count) cells get their own light grey so "not enough data" never
    reads as a real value.
    '''
    matrix, bins = heatmap_matrix(series, metric, label, min_count=min_count,
                                  key_suffix=key_suffix)
    x = list(series.x)

    if diverging_center is not None:
        low, high = v_lim if v_lim is not None else (0.0, 1.0)
        cmap = plt.get_cmap("coolwarm").copy()
        # TwoSlopeNorm keeps the neutral midpoint pinned to the baseline even
        # when the data is lopsided
        norm = colors.TwoSlopeNorm(vmin=low, vcenter=diverging_center, vmax=high)
    else:
        cmap = plt.get_cmap("Blues").copy()
        low, high = v_lim if v_lim is not None else (
            0.0, float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0
        )
        norm = colors.Normalize(vmin=low, vmax=high)
    # Masked cells are distinguished by TEXTURE, not another shade: a flat grey
    # would sit right on top of the diverging ramp's neutral midpoint, making
    # "no data" and "exactly chance" look identical. Transparent bad cells let a
    # hatched axes background show through instead.
    cmap.set_bad(alpha=0.0)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_facecolor("#fbfbfb")
    ax.patch.set_hatch("///")
    ax.patch.set_edgecolor("#cccccc")
    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        cmap=cmap, norm=norm, aspect="auto",
        origin="lower", interpolation="nearest",
        extent=(min(x) - 0.5, max(x) + 0.5, -0.5, len(bins) - 0.5),
    )

    # label a readable subset of rows rather than all of them
    step = max(1, len(bins) // 20)
    ticks = list(range(0, len(bins), step))
    # always label the top row, but replace the previous tick rather than
    # appending when they would collide
    if ticks[-1] != len(bins) - 1:
        if len(bins) - 1 - ticks[-1] < step:
            ticks[-1] = len(bins) - 1
        else:
            ticks.append(len(bins) - 1)
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(bins[t]) for t in ticks])

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    # recessive: the cells are the data, the frame should not compete
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(cbar_label or metric)
    cbar.outline.set_visible(False)
    if diverging_center is not None:
        cbar.ax.axhline(diverging_center, color="#444444", linewidth=1)

    if min_count is not None:
        fig.text(0.01, 0.01, f"hatched = fewer than {min_count} samples",
                 fontsize=8, color="#666666")

    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/{fn}.png", dpi=120)
    plt.close()
