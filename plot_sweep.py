"""
Sweep-level plotting.

Reads `<sweep_dir>/sweep_results.json` (written by aggregate_sweep.py) and writes
auto-generated plots to `<sweep_dir>/sweep_plots/`:

  - bar_<metric>.png         always; one bar per HP point, error bars = 95% CI
  - line_<metric>.png        only for 1D sweeps (single swept HP key)
  - heatmap_<metric>.png     only for 2D sweeps (two swept HP keys)

For 3D+ sweeps and explicit-points sweeps with multiple keys, only bar charts
are produced. The sweep_results.json file is the public API for anything fancier.

Usage:
    python3 plot_sweep.py output/<sweep_name>/
"""

import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sanitize(name: str) -> str:
    """Make a metric name safe for use as a filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def fmt_value(v) -> str:
    """Compact value formatting for axis tick labels and bar labels."""
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
            return f"{v:.1e}"
        return f"{v:g}"
    return str(v)


def hp_label(overrides: dict, keys: list[str]) -> str:
    """Compact 'k1=v1 k2=v2' label for an HP point's bar tick."""
    return " ".join(f"{k}={fmt_value(overrides.get(k))}" for k in keys)


def metric_has_data(hp_points: list[dict], metric: str) -> bool:
    """True if at least one HP point has a non-None mean for this metric."""
    for hp in hp_points:
        m = hp.get("metrics", {}).get(metric)
        if m is not None and m.get("mean") is not None:
            return True
    return False


def extract_metric(hp_points: list[dict], metric: str) -> tuple[list[float | None], list[float | None]]:
    """Return (means, ci95s) aligned to hp_points."""
    means = []
    cis = []
    for hp in hp_points:
        m = hp.get("metrics", {}).get(metric)
        if m is None:
            means.append(None)
            cis.append(None)
            continue
        means.append(m.get("mean"))
        cis.append(m.get("ci95"))
    return means, cis


def plot_bar(out_path: str, metric: str, labels: list[str],
             means: list[float | None], cis: list[float | None]) -> None:
    """Bar chart with 95% CI error bars. Always-on, works for any sweep dim."""
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(labels) + 2), 4.5))

    xs = np.arange(len(labels))
    plot_means = [m if m is not None else np.nan for m in means]
    plot_errs = [c if c is not None else 0.0 for c in cis]

    ax.bar(xs, plot_means, yerr=plot_errs, capsize=4, color="#4477aa", edgecolor="black")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def values_are_numeric(values: list) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)


def plot_line_1d(out_path: str, metric: str, key: str,
                 x_values: list, means: list[float | None], cis: list[float | None]) -> None:
    """Line plot for a 1D sweep — x = swept HP value, y = metric mean ± 95% CI."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Sort by x if numeric, else preserve insertion order.
    numeric = values_are_numeric(x_values)
    if numeric:
        order = sorted(range(len(x_values)), key=lambda i: x_values[i])
    else:
        order = list(range(len(x_values)))
    xs_sorted = [x_values[i] for i in order]
    means_sorted = [means[i] for i in order]
    cis_sorted = [cis[i] for i in order]

    # Drop points with no data so lines don't have gaps from None.
    valid = [(x, m, c) for x, m, c in zip(xs_sorted, means_sorted, cis_sorted) if m is not None]
    if not valid:
        plt.close(fig)
        return

    if numeric:
        xs_plot = np.array([v[0] for v in valid], dtype=np.float64)
    else:
        xs_plot = np.arange(len(valid))
    ms = np.array([v[1] for v in valid], dtype=np.float64)
    cs = np.array([v[2] if v[2] is not None else 0.0 for v in valid], dtype=np.float64)

    ax.plot(xs_plot, ms, marker="o", color="#4477aa")
    ax.fill_between(xs_plot, ms - cs, ms + cs, alpha=0.25, color="#4477aa", label="±95% CI")

    if numeric and len(xs_plot) >= 2 and xs_plot.min() > 0:
        if xs_plot.max() / xs_plot.min() >= 20:
            ax.set_xscale("log")

    if not numeric:
        ax.set_xticks(xs_plot)
        ax.set_xticklabels([str(v[0]) for v in valid], rotation=45, ha="right")

    ax.set_xlabel(key)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs {key}")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_heatmap_2d(out_path: str, metric: str, key_x: str, key_y: str,
                    hp_points: list[dict]) -> None:
    """Heatmap for a 2D sweep. Cells are mean values; missing cells are NaN."""
    # Collect unique values for each axis
    x_vals_set: list = []
    y_vals_set: list = []
    seen_x: set = set()
    seen_y: set = set()
    for hp in hp_points:
        ov = hp.get("overrides", {})
        xv = ov.get(key_x)
        yv = ov.get(key_y)
        # Use repr-as-key so float comparison is reliable
        if repr(xv) not in seen_x:
            seen_x.add(repr(xv))
            x_vals_set.append(xv)
        if repr(yv) not in seen_y:
            seen_y.add(repr(yv))
            y_vals_set.append(yv)

    # Sort numerically if both axes are numeric
    if values_are_numeric(x_vals_set):
        x_vals_set.sort()
    if values_are_numeric(y_vals_set):
        y_vals_set.sort()

    x_idx = {repr(v): i for i, v in enumerate(x_vals_set)}
    y_idx = {repr(v): i for i, v in enumerate(y_vals_set)}

    grid = np.full((len(y_vals_set), len(x_vals_set)), np.nan, dtype=np.float64)
    for hp in hp_points:
        m = hp.get("metrics", {}).get(metric)
        if m is None or m.get("mean") is None:
            continue
        ov = hp.get("overrides", {})
        i = y_idx[repr(ov.get(key_y))]
        j = x_idx[repr(ov.get(key_x))]
        grid[i, j] = m["mean"]

    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(x_vals_set) + 2),
                                     max(4, 0.5 * len(y_vals_set) + 2)))

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="lightgray")
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, cmap=cmap, aspect="auto", origin="lower")
    fig.colorbar(im, ax=ax, label=metric)

    ax.set_xticks(np.arange(len(x_vals_set)))
    ax.set_xticklabels([fmt_value(v) for v in x_vals_set], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_vals_set)))
    ax.set_yticklabels([fmt_value(v) for v in y_vals_set])
    ax.set_xlabel(key_x)
    ax.set_ylabel(key_y)
    ax.set_title(metric)

    # Annotate cells
    for i in range(len(y_vals_set)):
        for j in range(len(x_vals_set)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=7, color="white" if v < (np.nanmin(grid) + np.nanmax(grid)) / 2 else "black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir")
    args = parser.parse_args()

    sweep_dir = args.sweep_dir.rstrip("/")
    results_path = os.path.join(sweep_dir, "sweep_results.json")
    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found (run aggregate_sweep.py first)", file=sys.stderr)
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    hp_points = results["hp_points"]
    metric_names = results["metric_names"]
    override_keys = results["override_keys"]

    plots_dir = os.path.join(sweep_dir, "sweep_plots")
    os.makedirs(plots_dir, exist_ok=True)

    n_dim = len(override_keys)
    print(f"Plotting sweep '{results['sweep_name']}': "
          f"{len(hp_points)} HP points, {len(metric_names)} metrics, {n_dim} HP dimensions")

    labels = [hp_label(hp.get("overrides", {}), override_keys) for hp in hp_points]

    n_plotted = 0
    n_skipped = 0
    for metric in metric_names:
        if not metric_has_data(hp_points, metric):
            print(f"  skipping {metric}: no data")
            n_skipped += 1
            continue

        means, cis = extract_metric(hp_points, metric)
        safe = sanitize(metric)

        # 1. Bar chart (always)
        try:
            plot_bar(os.path.join(plots_dir, f"bar_{safe}.png"), metric, labels, means, cis)
        except Exception as e:
            print(f"  WARNING: bar plot failed for {metric}: {e}")

        # 2. 1D line plot
        if n_dim == 1:
            key = override_keys[0]
            x_values = [hp.get("overrides", {}).get(key) for hp in hp_points]
            try:
                plot_line_1d(os.path.join(plots_dir, f"line_{safe}.png"),
                             metric, key, x_values, means, cis)
            except Exception as e:
                print(f"  WARNING: line plot failed for {metric}: {e}")

        # 3. 2D heatmap
        if n_dim == 2:
            key_x, key_y = override_keys[0], override_keys[1]
            try:
                plot_heatmap_2d(os.path.join(plots_dir, f"heatmap_{safe}.png"),
                                metric, key_x, key_y, hp_points)
            except Exception as e:
                print(f"  WARNING: heatmap failed for {metric}: {e}")

        n_plotted += 1

    if n_dim >= 3:
        print(f"  ({n_dim}D sweep — only bar plots produced; see sweep_results.json for full data)")

    print(f"Done. {n_plotted} metrics plotted, {n_skipped} skipped. Output: {plots_dir}/")


if __name__ == "__main__":
    main()
