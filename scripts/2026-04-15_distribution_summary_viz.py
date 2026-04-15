"""
Demo script for two alternative ways to summarize a DistributionSeries
(per-iteration distributions) as a line-plot, rather than a ridgeline:

    (A) Percentile-band + mean overlay
        - shaded 5–95% band (outer)
        - shaded 25–75% IQR band (inner)
        - median line (solid)
        - mean line (dashed)

    (B) Value-proportion stacked plot
        - samples binned into fixed value ranges from -1 to +1
        - stacked fraction per bin over iterations
        - colored by bin center (RdBu_r: blue = negative, red = positive)

Rendered for six scenarios to show how each plot reads under normal,
degenerate, bimodal, skewed, and mode-collapsing distributions.

Output → scripts/2026-04-15_distribution_summary_viz.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors


# ── Synthetic data ───────────────────────────────────────────────────────────

N_ITER = 20
N_SAMPLES = 1000
ITERS = np.arange(N_ITER)


def _scenario_gaussian_shrink(rng: np.random.Generator) -> list[np.ndarray]:
    """Gaussian centered near 0, drifts to +0.5, variance shrinks."""
    out = []
    for i in range(N_ITER):
        t = i / (N_ITER - 1)
        mu = 0.5 * t
        sigma = 0.5 * (1 - 0.6 * t)
        out.append(np.clip(rng.normal(mu, sigma, N_SAMPLES), -1, 1))
    return out


def _scenario_bimodal_polarize(rng: np.random.Generator) -> list[np.ndarray]:
    """Starts unimodal near 0, becomes sharply bimodal at ±1."""
    out = []
    for i in range(N_ITER):
        t = i / (N_ITER - 1)
        signal = t ** 1.5
        half = N_SAMPLES // 2
        pos = rng.uniform(0.6, 1.0, half)
        neg = rng.uniform(-1.0, -0.6, half)
        noise_scale = 0.6 * (1 - 0.7 * t)
        samples = np.concatenate([
            signal * pos + rng.normal(0, noise_scale, half),
            signal * neg + rng.normal(0, noise_scale, half),
        ])
        out.append(np.clip(samples, -1, 1))
    return out


def _scenario_all_ones(rng: np.random.Generator) -> list[np.ndarray]:
    """Degenerate: every sample is exactly +1 at every iteration."""
    return [np.ones(N_SAMPLES) for _ in range(N_ITER)]


def _scenario_skewed_to_one(rng: np.random.Generator) -> list[np.ndarray]:
    """Most mass near +1 with a shrinking tail toward 0 (right-skewed)."""
    out = []
    for i in range(N_ITER):
        t = i / (N_ITER - 1)
        n_body = int(N_SAMPLES * 0.85)
        body_lo = 1 - 0.5 * (1 - t)
        body = rng.uniform(body_lo, 1.0, n_body)
        tail = rng.uniform(0.0, body_lo, N_SAMPLES - n_body)
        out.append(np.concatenate([body, tail]))
    return out


def _scenario_discrete_shift(rng: np.random.Generator) -> list[np.ndarray]:
    """Three-way discrete {-1, 0, +1} with shifting proportions."""
    out = []
    for i in range(N_ITER):
        t = i / (N_ITER - 1)
        p_plus = 0.2 + 0.5 * t
        p_minus = 0.3 - 0.2 * t
        p_zero = 1 - p_plus - p_minus
        samples = rng.choice([-1.0, 0.0, 1.0], size=N_SAMPLES, p=[p_minus, p_zero, p_plus])
        out.append(samples)
    return out


def _scenario_mode_collapse(rng: np.random.Generator) -> list[np.ndarray]:
    """Uniform early, fraction of samples snap to {-1, +1} over time."""
    out = []
    for i in range(N_ITER):
        t = i / (N_ITER - 1)
        n_decided = int(N_SAMPLES * t)
        decided = rng.choice([-1.0, 1.0], size=n_decided)
        undecided = rng.uniform(-1, 1, N_SAMPLES - n_decided)
        out.append(np.concatenate([decided, undecided]))
    return out


SCENARIOS = [
    ("Gaussian drift + variance shrink", _scenario_gaussian_shrink),
    ("Unimodal → bimodal ±1 (polarizing)", _scenario_bimodal_polarize),
    ("Degenerate: all samples = +1", _scenario_all_ones),
    ("Right-skewed, mass piling at +1", _scenario_skewed_to_one),
    ("Discrete {-1, 0, +1}, shifting mix", _scenario_discrete_shift),
    ("Uniform → mode collapse to ±1", _scenario_mode_collapse),
]


# ── Summary stats & plotting helpers ─────────────────────────────────────────


def _summary_stats(samples_per_iter: list[np.ndarray]) -> dict[str, np.ndarray]:
    pct = np.stack([np.percentile(s, [5, 25, 50, 75, 95]) for s in samples_per_iter])
    means = np.array([s.mean() for s in samples_per_iter])
    stds = np.array([s.std() for s in samples_per_iter])
    return {
        "p5": pct[:, 0], "p25": pct[:, 1], "p50": pct[:, 2],
        "p75": pct[:, 3], "p95": pct[:, 4],
        "mean": means, "std": stds,
    }


def plot_percentile_band(ax, x, samples_per_iter, title: str) -> None:
    s = _summary_stats(samples_per_iter)
    ax.fill_between(x, s["p5"], s["p95"], color="C0", alpha=0.18, label="5–95%")
    ax.fill_between(x, s["p25"], s["p75"], color="C0", alpha=0.38, label="25–75% (IQR)")
    ax.plot(x, s["p50"], color="black", linewidth=1.6, label="median")
    ax.plot(x, s["mean"], color="crimson", linewidth=1.4, linestyle="--", label="mean")
    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel("value")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


def plot_value_proportions(ax, x, samples_per_iter, title: str,
                           n_bins: int = 10, value_range: tuple[float, float] = (-1, 1)) -> None:
    edges = np.linspace(value_range[0], value_range[1], n_bins + 1)
    props = np.zeros((n_bins, len(x)))
    for i, samples in enumerate(samples_per_iter):
        # include both endpoints in the extreme bins so exact -1 / +1 count
        s = np.clip(samples, value_range[0], value_range[1])
        counts, _ = np.histogram(s, bins=edges)
        total = max(len(s), 1)
        props[:, i] = counts / total

    cmap = plt.get_cmap("RdBu_r")
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    norm = mcolors.Normalize(vmin=value_range[0], vmax=value_range[1])
    bin_colors = [cmap(norm(c)) for c in bin_centers]

    ax.stackplot(x, props, colors=bin_colors, edgecolor="none")
    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel("proportion of samples")
    ax.set_ylim(0, 1)
    ax.set_xlim(x.min(), x.max())

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("bin center (value)")


# ── Build the figure ─────────────────────────────────────────────────────────


def main() -> None:
    rng = np.random.default_rng(0)

    n_rows = len(SCENARIOS)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3.2 * n_rows))

    for row, (title, gen) in enumerate(SCENARIOS):
        samples_per_iter = gen(rng)
        plot_percentile_band(axes[row, 0], ITERS, samples_per_iter,
                             f"{title}\n(A) percentile bands + mean")
        plot_value_proportions(axes[row, 1], ITERS, samples_per_iter,
                               f"{title}\n(B) value-bin proportions")

    fig.suptitle(
        "Distribution-series summary plots — two alternatives to ridgelines",
        fontsize=14, y=1.0,
    )
    fig.tight_layout()

    out = "scripts/2026-04-15_distribution_summary_viz.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
