"""
Visualise the weighting scheme of the λ-return:

  G_t^λ = (1 - λ) Σ_{n=1}^{T-t-1} λ^{n-1} G_{t:t+n}  +  λ^{T-t-1} G_t

For each (T, λ) combination we plot the weight assigned to every n-step
return G_{t:t+n} (n = 1 … T-t-1) and the final full-return term G_t.
All weights sum to 1.
"""

import matplotlib.pyplot as plt
import numpy as np


def lambda_return_weights(T, t, lam):
    """Return (labels, weights) for every component of G_t^λ."""
    horizon = T - t  # number of steps remaining
    labels = []
    weights = []

    for n in range(1, horizon):  # n = 1 … T-t-1
        w = (1 - lam) * lam ** (n - 1)
        labels.append(f"$G_{{t:t+{n}}}$")
        weights.append(w)

    # final full-return term
    w_full = lam ** (horizon - 1)
    labels.append(f"$G_t$ (full)")
    weights.append(w_full)

    return labels, np.array(weights)


def main():
    t = 0  # always measure from t=0 for clarity
    T_values = [5, 10, 20]
    lam_values = [0.0, 0.3, 0.5, 0.8, 0.9, 1.0]

    fig, axes = plt.subplots(
        len(T_values), len(lam_values),
        figsize=(3.2 * len(lam_values), 3 * len(T_values)),
        squeeze=False,
    )

    for row, T in enumerate(T_values):
        for col, lam in enumerate(lam_values):
            ax = axes[row][col]
            labels, weights = lambda_return_weights(T, t, lam)

            n_components = len(weights)
            xs = np.arange(n_components)
            colours = ["steelblue"] * (n_components - 1) + ["firebrick"]

            ax.bar(xs, weights, color=colours)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"T={T}, λ={lam}", fontsize=10)
            ax.set_ylabel("weight" if col == 0 else "")
            ax.set_xticks(xs)
            ax.set_xticklabels(
                [f"n={n}" for n in range(1, n_components)] + ["full"],
                fontsize=6, rotation=45, ha="right",
            )

            # annotate the weight sum as a sanity check
            ax.text(
                0.97, 0.93, f"Σ={weights.sum():.3f}",
                transform=ax.transAxes, fontsize=7,
                ha="right", va="top",
            )

    fig.suptitle(
        r"$\lambda$-return weights:  $(1-\lambda)\lambda^{n-1}$ per $G_{t:t+n}$"
        r"  +  $\lambda^{T-t-1}$ for $G_t$",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("lambda_return_weights.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
