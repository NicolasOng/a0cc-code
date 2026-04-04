"""
Standalone script to demo visualization options for value diagnostics data.

The diagnostics dict (per iteration) has:
  - dataset_values_pre:  target values before balancing
  - dataset_values_post: target values after balancing
  - gt_eval_values:      ground truth values (balanced eval set)
  - model_predictions:   model value head output on eval set

We demo three approaches:
  (1) Per-iteration text histograms (like print_bucket_distribution)
  (2) Heatmap: iteration x value-bin, color = density
  (3) Summary statistics as line plots (mean, std per iteration)
  (4) Stacked area chart: bucket proportions over iterations
  (5) Ridgeline plot: overlapping density curves per iteration
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Generate dummy data ──────────────────────────────────────────────────────

N_ITERATIONS = 20
N_SAMPLES = 2000
N_BINS = 20
BIN_EDGES = np.linspace(-1, 1, N_BINS + 1)

rng = np.random.default_rng(42)


def make_dummy_diagnostics(n_iter: int) -> list[dict[str, list[float]]]:
    """
    Simulate diagnostics that evolve over iterations:
    - pre-balance: starts win-heavy, slowly becomes more balanced
    - post-balance: always roughly balanced (that's the point of balancing)
    - gt_eval_values: fixed balanced ground truth
    - model_predictions: starts near 0, gradually spreads toward GT
    """
    n_third = N_SAMPLES // 3
    gt_values = np.concatenate([
        rng.uniform(-1, -0.3, n_third),
        rng.uniform(-0.1, 0.1, n_third),
        rng.uniform(0.3, 1.0, N_SAMPLES - 2 * n_third),
    ])

    diagnostics = []
    for i in range(n_iter):
        t = i / max(n_iter - 1, 1)  # 0 → 1

        # pre-balance: skewed toward wins early, less so later
        win_frac = 0.7 - 0.3 * t
        n_win = int(N_SAMPLES * win_frac)
        n_loss = N_SAMPLES - n_win
        pre = np.concatenate([
            rng.uniform(0.0, 1.0, n_win),
            rng.uniform(-1.0, 0.0, n_loss),
        ])

        # post-balance: roughly symmetric
        half = N_SAMPLES // 2
        post = np.concatenate([
            rng.uniform(0.0, 1.0, half),
            rng.uniform(-1.0, 0.0, half),
        ])

        # model predictions: start clustered near 0, spread out toward GT over time
        spread = 0.1 + 0.8 * t
        noise = rng.normal(0, 0.15, N_SAMPLES)
        preds = np.clip(gt_values * spread + noise, -1, 1)

        diagnostics.append({
            'dataset_values_pre': pre.tolist(),
            'dataset_values_post': post.tolist(),
            'gt_eval_values': gt_values.tolist(),
            'model_predictions': preds.tolist(),
        })
    return diagnostics


all_diags = make_dummy_diagnostics(N_ITERATIONS)

# ── (1) Text histograms (like print_bucket_distribution) ────────────────────

def log_bucket_distribution(label: str, values: list[float], n_buckets: int = 10) -> None:
    arr = np.array(values)
    edges = np.linspace(-1, 1, n_buckets + 1)
    counts = np.histogram(arr, bins=edges)[0]
    total = len(arr)
    print(f"  {label} ({n_buckets} buckets, {total} samples):")
    for j in range(n_buckets):
        lo, hi = edges[j], edges[j + 1]
        c = int(counts[j])
        bar = "#" * int(40 * c / max(total, 1))
        bracket = "]" if j == n_buckets - 1 else ")"
        print(f"    [{lo:+.2f}, {hi:+.2f}{bracket} {c:6d} ({c/total:.2%}) {bar}")


print("=" * 80)
print("(1) PER-ITERATION TEXT HISTOGRAMS")
print("=" * 80)
for i, diag in enumerate(all_diags):
    print(f"\n--- Iteration {i + 1} ---")
    log_bucket_distribution("Dataset targets (pre-balance)", diag['dataset_values_pre'])
    log_bucket_distribution("Dataset targets (post-balance)", diag['dataset_values_post'])
    log_bucket_distribution("GT eval values", diag['gt_eval_values'])
    log_bucket_distribution("Model predictions", diag['model_predictions'])

# ── (2) Heatmaps ────────────────────────────────────────────────────────────

def build_heatmap(diagnostics: list[dict], key: str) -> np.ndarray:
    """Rows = iterations, columns = bins. Values = fraction of samples in each bin."""
    matrix = np.zeros((len(diagnostics), N_BINS))
    for i, diag in enumerate(diagnostics):
        counts = np.histogram(diag[key], bins=BIN_EDGES)[0]
        matrix[i] = counts / counts.sum()
    return matrix


fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("(2) Heatmaps: value distribution over iterations", fontsize=14)

keys_and_titles = [
    ('dataset_values_pre', 'Dataset targets (pre-balance)'),
    ('dataset_values_post', 'Dataset targets (post-balance)'),
    ('gt_eval_values', 'GT eval values'),
    ('model_predictions', 'Model predictions'),
]

for ax, (key, title) in zip(axes.flat, keys_and_titles):
    matrix = build_heatmap(all_diags, key)
    im = ax.imshow(
        matrix, aspect='auto', origin='lower',
        extent=[-1, 1, 1, N_ITERATIONS],
        cmap='YlOrRd', interpolation='nearest',
    )
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Iteration")
    fig.colorbar(im, ax=ax, label="Fraction")

plt.tight_layout()
plt.savefig("scripts/value_diagnostics_heatmaps.png", dpi=150)
print("\n\nSaved heatmaps → scripts/value_diagnostics_heatmaps.png")

# ── (3) Summary statistics as line plots ─────────────────────────────────────

def extract_stats(diagnostics: list[dict], key: str) -> dict[str, list[float]]:
    means, stds, medians = [], [], []
    q25s, q75s = [], []
    for diag in diagnostics:
        arr = np.array(diag[key])
        means.append(float(arr.mean()))
        stds.append(float(arr.std()))
        medians.append(float(np.median(arr)))
        q25s.append(float(np.percentile(arr, 25)))
        q75s.append(float(np.percentile(arr, 75)))
    return {'mean': means, 'std': stds, 'median': medians, 'q25': q25s, 'q75': q75s}


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("(3) Summary statistics over iterations", fontsize=14)
iterations = list(range(1, N_ITERATIONS + 1))

# Left: means
ax = axes[0]
for key, title in keys_and_titles:
    stats = extract_stats(all_diags, key)
    ax.plot(iterations, stats['mean'], label=title, marker='.')
ax.set_title("Mean value per iteration")
ax.set_xlabel("Iteration")
ax.set_ylabel("Mean")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Right: std
ax = axes[1]
for key, title in keys_and_titles:
    stats = extract_stats(all_diags, key)
    ax.plot(iterations, stats['std'], label=title, marker='.')
ax.set_title("Std of values per iteration")
ax.set_xlabel("Iteration")
ax.set_ylabel("Std")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("scripts/value_diagnostics_lineplots.png", dpi=150)
print("Saved line plots → scripts/value_diagnostics_lineplots.png")

# ── (4) Stacked area charts ─────────────────────────────────────────────────

N_AREA_BINS = 5
AREA_EDGES = np.linspace(-1, 1, N_AREA_BINS + 1)

def build_bucket_fractions(diagnostics: list[dict], key: str) -> np.ndarray:
    """Rows = iterations, columns = buckets. Values = fraction in each bucket."""
    matrix = np.zeros((len(diagnostics), N_AREA_BINS))
    for i, diag in enumerate(diagnostics):
        counts = np.histogram(diag[key], bins=AREA_EDGES)[0]
        matrix[i] = counts / counts.sum()
    return matrix

def bucket_labels() -> list[str]:
    labels = []
    for j in range(N_AREA_BINS):
        lo, hi = AREA_EDGES[j], AREA_EDGES[j + 1]
        bracket = "]" if j == N_AREA_BINS - 1 else ")"
        labels.append(f"[{lo:+.1f}, {hi:+.1f}{bracket}")
    return labels

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("(4) Stacked area: bucket proportions over iterations", fontsize=14)
labels = bucket_labels()
cmap = plt.cm.RdYlBu_r
colors = [cmap(j / (N_AREA_BINS - 1)) for j in range(N_AREA_BINS)]

for ax, (key, title) in zip(axes.flat, keys_and_titles):
    fracs = build_bucket_fractions(all_diags, key)
    ax.stackplot(iterations, fracs.T, labels=labels, colors=colors, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', fontsize=6)

plt.tight_layout()
plt.savefig("scripts/value_diagnostics_stacked_area.png", dpi=150)
print("Saved stacked area → scripts/value_diagnostics_stacked_area.png")

# ── (5) Ridgeline plots ─────────────────────────────────────────────────────

from scipy.stats import gaussian_kde

def ridgeline_subplot(ax, diagnostics: list[dict], key: str, title: str) -> None:
    """Draw overlapping KDE ridges, one per iteration, bottom to top."""
    x_grid = np.linspace(-1, 1, 200)
    overlap = 0.7  # how much adjacent ridges overlap

    for i, diag in enumerate(diagnostics):
        arr = np.array(diag[key])
        kde = gaussian_kde(arr, bw_method=0.15)
        density = kde(x_grid)
        # normalize so peak = 1 for consistent ridge height
        density = density / density.max()

        baseline = i * overlap
        ax.fill_between(x_grid, baseline, baseline + density,
                         alpha=0.6, color=plt.cm.viridis(i / len(diagnostics)),
                         edgecolor='white', linewidth=0.5)

    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_yticks([i * overlap for i in range(0, len(diagnostics), 5)])
    ax.set_yticklabels([str(i + 1) for i in range(0, len(diagnostics), 5)])
    ax.set_ylabel("Iteration")
    ax.set_xlim(-1, 1)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("(5) Ridgeline: density per iteration", fontsize=14)

for ax, (key, title) in zip(axes.flat, keys_and_titles):
    ridgeline_subplot(ax, all_diags, key, title)

plt.tight_layout()
plt.savefig("scripts/value_diagnostics_ridgeline.png", dpi=150)
print("Saved ridgeline → scripts/value_diagnostics_ridgeline.png")

plt.show()
