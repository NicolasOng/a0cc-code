"""
Ridgeline styling options for value diagnostics.
All variations use the model_predictions distribution (the most interesting one).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ── Generate same dummy data as before ───────────────────────────────────────

N_ITERATIONS = 20
N_SAMPLES = 2000
rng = np.random.default_rng(42)

n_third = N_SAMPLES // 3
gt_values = np.concatenate([
    rng.uniform(-1, -0.3, n_third),
    rng.uniform(-0.1, 0.1, n_third),
    rng.uniform(0.3, 1.0, N_SAMPLES - 2 * n_third),
])

all_preds = []
for i in range(N_ITERATIONS):
    t = i / (N_ITERATIONS - 1)
    spread = 0.1 + 0.8 * t
    noise = rng.normal(0, 0.15, N_SAMPLES)
    preds = np.clip(gt_values * spread + noise, -1, 1)
    all_preds.append(preds)

x_grid = np.linspace(-1, 1, 300)

def kde_density(arr, bw=0.15):
    kde = gaussian_kde(arr, bw_method=bw)
    d = kde(x_grid)
    return d / d.max()

# ── Option A: baseline (what we had) ────────────────────────────────────────

def plot_a(ax):
    overlap = 0.7
    for i, preds in enumerate(all_preds):
        density = kde_density(preds)
        baseline = i * overlap
        ax.fill_between(x_grid, baseline, baseline + density,
                         alpha=0.6, color=plt.cm.viridis(i / N_ITERATIONS),
                         edgecolor='white', linewidth=0.5)
    ax.set_title("A: Baseline")
    ax.set_yticks([i * overlap for i in range(0, N_ITERATIONS, 5)])
    ax.set_yticklabels([str(i + 1) for i in range(0, N_ITERATIONS, 5)])

# ── Option B: higher overlap + dark edges + gradient fill ────────────────────

def plot_b(ax):
    overlap = 0.55
    for i, preds in enumerate(all_preds):
        density = kde_density(preds)
        baseline = i * overlap
        t = i / N_ITERATIONS
        ax.fill_between(x_grid, baseline, baseline + density,
                         alpha=0.7, color=plt.cm.coolwarm(t),
                         edgecolor='black', linewidth=0.6)
    ax.set_title("B: Coolwarm + dark edges")
    ax.set_yticks([i * overlap for i in range(0, N_ITERATIONS, 5)])
    ax.set_yticklabels([str(i + 1) for i in range(0, N_ITERATIONS, 5)])

# ── Option C: outline only (no fill), colored lines ──────────────────────────

def plot_c(ax):
    overlap = 0.6
    for i, preds in enumerate(all_preds):
        density = kde_density(preds)
        baseline = i * overlap
        t = i / N_ITERATIONS
        ax.plot(x_grid, baseline + density,
                color=plt.cm.plasma(t), linewidth=1.2, alpha=0.9)
        ax.axhline(baseline, color='grey', linewidth=0.2, alpha=0.3)
    ax.set_title("C: Lines only (no fill)")
    ax.set_yticks([i * overlap for i in range(0, N_ITERATIONS, 5)])
    ax.set_yticklabels([str(i + 1) for i in range(0, N_ITERATIONS, 5)])

# ── Option D: white fill with colored edge (classic ridgeline) ───────────────

def plot_d(ax):
    overlap = 0.65
    for i, preds in enumerate(all_preds):
        density = kde_density(preds)
        baseline = i * overlap
        t = i / N_ITERATIONS
        color = plt.cm.viridis(t)
        ax.fill_between(x_grid, baseline, baseline + density,
                         alpha=0.85, facecolor='white',
                         edgecolor=color, linewidth=0.8)
        # re-draw edge on top
        ax.plot(x_grid, baseline + density, color=color, linewidth=0.9)
    ax.set_title("D: White fill + colored edge")
    ax.set_yticks([i * overlap for i in range(0, N_ITERATIONS, 5)])
    ax.set_yticklabels([str(i + 1) for i in range(0, N_ITERATIONS, 5)])

# ── Option E: gradient fill per-ridge (light → saturated from base to peak) ─

def plot_e(ax):
    overlap = 0.6
    n_slices = 30  # vertical slices for gradient effect
    for i, preds in enumerate(all_preds):
        density = kde_density(preds)
        baseline = i * overlap
        t = i / N_ITERATIONS
        base_color = np.array(plt.cm.viridis(t))
        for s in range(n_slices):
            lo_frac = s / n_slices
            hi_frac = (s + 1) / n_slices
            y_lo = baseline + density * lo_frac
            y_hi = baseline + density * hi_frac
            alpha = 0.3 + 0.5 * hi_frac
            ax.fill_between(x_grid, y_lo, y_hi,
                             color=base_color, alpha=alpha, linewidth=0)
        ax.plot(x_grid, baseline + density,
                color=base_color * 0.7, linewidth=0.5, alpha=0.8)
    ax.set_title("E: Vertical gradient fill")
    ax.set_yticks([i * overlap for i in range(0, N_ITERATIONS, 5)])
    ax.set_yticklabels([str(i + 1) for i in range(0, N_ITERATIONS, 5)])

# ── Option F: filled + GT reference overlay ──────────────────────────────────

gt_density = kde_density(gt_values)

def plot_f(ax):
    overlap = 0.65
    for i, preds in enumerate(all_preds):
        density = kde_density(preds)
        baseline = i * overlap
        t = i / N_ITERATIONS
        ax.fill_between(x_grid, baseline, baseline + density,
                         alpha=0.6, color=plt.cm.viridis(t),
                         edgecolor='white', linewidth=0.4)
        # GT reference as dashed line
        ax.plot(x_grid, baseline + gt_density,
                color='red', linewidth=0.6, linestyle='--', alpha=0.5)
    ax.set_title("F: With GT reference (red dashed)")
    ax.set_yticks([i * overlap for i in range(0, N_ITERATIONS, 5)])
    ax.set_yticklabels([str(i + 1) for i in range(0, N_ITERATIONS, 5)])

# ── Option G: stepped histogram (no smoothing), line-only style like C ───────

def plot_g(ax, n_bins=10):
    overlap = 0.6
    edges = np.linspace(-1, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    for i, preds in enumerate(all_preds):
        counts = np.histogram(preds, bins=edges)[0].astype(float)
        counts /= counts.max()  # normalize peak to 1
        baseline = i * overlap
        # step function: duplicate each value at left and right bin edges
        step_x = np.repeat(edges, 2)[1:-1]
        step_y = np.repeat(counts, 2)
        t = i / N_ITERATIONS
        ax.plot(step_x, baseline + step_y,
                color=plt.cm.plasma(t), linewidth=1.0, alpha=0.9)
    ax.set_title(f"G: Stepped histogram ({n_bins} bins)")
    ax.set_yticks([i * overlap for i in range(0, N_ITERATIONS, 5)])
    ax.set_yticklabels([str(i + 1) for i in range(0, N_ITERATIONS, 5)])

# ── Option H: stepped histogram with fill ────────────────────────────────────

def plot_h(ax, n_bins=10):
    overlap = 0.6
    edges = np.linspace(-1, 1, n_bins + 1)
    for i, preds in enumerate(all_preds):
        counts = np.histogram(preds, bins=edges)[0].astype(float)
        counts /= counts.max()
        baseline = i * overlap
        step_x = np.repeat(edges, 2)[1:-1]
        step_y = np.repeat(counts, 2)
        t = i / N_ITERATIONS
        color = plt.cm.plasma(t)
        ax.fill_between(step_x, baseline, baseline + step_y,
                         alpha=0.4, color=color, linewidth=0)
        ax.plot(step_x, baseline + step_y,
                color=color, linewidth=0.8, alpha=0.9)
    ax.set_title(f"H: Stepped with fill ({n_bins} bins)")
    ax.set_yticks([i * overlap for i in range(0, N_ITERATIONS, 5)])
    ax.set_yticklabels([str(i + 1) for i in range(0, N_ITERATIONS, 5)])

# ── Option I: C style but all black lines ────────────────────────────────────

def plot_i(ax):
    overlap = 0.6
    for i, preds in enumerate(all_preds):
        density = kde_density(preds)
        baseline = i * overlap
        ax.plot(x_grid, baseline + density,
                color='black', linewidth=1.0, alpha=0.7)
        ax.axhline(baseline, color='grey', linewidth=0.2, alpha=0.3)
    ax.set_title("I: Black lines")
    ax.set_yticks([i * overlap for i in range(0, N_ITERATIONS, 5)])
    ax.set_yticklabels([str(i + 1) for i in range(0, N_ITERATIONS, 5)])

# ── Render all ───────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 3, figsize=(18, 16))
fig.suptitle("Ridgeline styling options (Model predictions)", fontsize=16, y=0.98)

all_plots = [plot_a, plot_b, plot_c, plot_d, plot_e, plot_f, plot_g, plot_h, plot_i]
for ax, plot_fn in zip(axes.flat, all_plots):
    plot_fn(ax)
    ax.set_xlabel("Value")
    ax.set_ylabel("Iteration")
    ax.set_xlim(-1, 1)

plt.tight_layout()
plt.savefig("scripts/ridgeline_options.png", dpi=150)
print("Saved → scripts/ridgeline_options.png")
plt.show()
