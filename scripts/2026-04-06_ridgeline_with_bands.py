"""
Ridgeline plot with cross-trial confidence interval bands.

Simulates value-prediction distributions across training iterations for
multiple independent trials, then plots a ridgeline where each ridge shows
the mean KDE across trials with a shaded Student-t confidence interval
band — matching how `merge_series` aggregates scalar metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, t as student_t

# ── Dummy data ───────────────────────────────────────────────────────────────
# Simulate model value predictions on a fixed eval set across training.
# Story:
#   - Iter 0 (untrained model): predictions are noisy near 0
#   - Mid iters: predictions slowly bifurcate toward {-1, +1}
#   - Late iters: predictions are sharply peaked at -1 and +1
#   - Trials disagree most in the middle (different training paths) and
#     converge in the late iterations.

N_TRIALS = 8
N_ITERATIONS = 20
N_SAMPLES = 1000

rng = np.random.default_rng(42)

def simulate_trial(trial_seed: int) -> list[np.ndarray]:
    '''Generate per-iteration prediction distributions for one trial.'''
    trial_rng = np.random.default_rng(trial_seed)
    iters = []
    # ground-truth-ish bimodal target the model is converging toward
    target_pos = trial_rng.uniform(0.6, 1.0, N_SAMPLES // 2)
    target_neg = trial_rng.uniform(-1.0, -0.6, N_SAMPLES // 2)
    targets = np.concatenate([target_pos, target_neg])

    for i in range(N_ITERATIONS):
        t = i / (N_ITERATIONS - 1)  # 0 → 1
        # how much of the target the model has internalized
        signal = t ** 1.5
        # cross-trial drift: each trial pulls in a slightly different direction
        # that shrinks as training converges
        trial_bias = trial_rng.normal(0, 0.25 * (1 - t))
        # in-trial noise: large early, shrinks late
        noise_scale = 0.5 * (1 - 0.7 * t)
        preds = signal * targets + trial_bias + trial_rng.normal(0, noise_scale, N_SAMPLES)
        preds = np.clip(preds, -1, 1)
        iters.append(preds)
    return iters

# trials shape: [trial][iteration] = ndarray of N_SAMPLES floats
trials = [simulate_trial(seed) for seed in range(N_TRIALS)]

# ── Plotting ─────────────────────────────────────────────────────────────────

x_grid = np.linspace(-1, 1, 300)
labels = [str(i) for i in range(N_ITERATIONS)]
overlap = 0.6
height = 1.8
bw = 0.15
confidence = 0.95

def per_trial_kdes(iter_idx: int) -> np.ndarray:
    '''
    Returns array of shape (N_TRIALS, len(x_grid)), with shared per-iteration
    normalization: all trials divided by the same max so peak-height differences
    between trials are preserved within the ridge.
    '''
    out = np.zeros((N_TRIALS, len(x_grid)))
    for t in range(N_TRIALS):
        kde = gaussian_kde(trials[t][iter_idx], bw_method=bw)
        out[t] = kde(x_grid)
    shared_max = out.max()
    if shared_max > 0:
        out = out / shared_max
    return out

# Student-t critical value matches the convention used in merge_series.
alpha = 1 - confidence
df = max(N_TRIALS - 1, 1)
t_value = student_t.ppf(1 - alpha / 2, df=df)

fig, ax = plt.subplots(figsize=(8, 12))

for i in range(N_ITERATIONS):
    per_trial = per_trial_kdes(i)         # shape (N_TRIALS, len(x_grid))
    mean = per_trial.mean(axis=0)
    std = per_trial.std(axis=0, ddof=1)
    ci = t_value * std / np.sqrt(N_TRIALS)
    lo = mean - ci
    hi = mean + ci

    baseline = i * overlap
    ax.plot([-1, 1], [baseline, baseline], color="grey", linewidth=0.5, alpha=0.4, zorder=0)
    ax.fill_between(
        x_grid,
        baseline + lo * height,
        baseline + hi * height,
        alpha=0.30,
        color="C0",
        linewidth=0,
    )
    ax.plot(x_grid, baseline + mean * height, color="black", linewidth=1.0, alpha=0.85)

ax.set_yticks([i * overlap for i in range(N_ITERATIONS)])
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted value")
ax.set_ylabel("Iteration")
ax.set_title(
    f"Mean KDE ± {int(confidence * 100)}% CI across {N_TRIALS} trials\n"
    "Band width = cross-trial uncertainty in the density at each value"
)
ax.set_xlim(-1.05, 1.05)

plt.tight_layout()

out_path = "scripts/ridgeline_with_bands.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
