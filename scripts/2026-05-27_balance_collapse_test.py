"""
Experiment: does the weighted_buckets balance code cause value-head collapse,
even though the weights are near 1.0?

Three conditions, many trials each:
  A) dataset_balance_method = "none"            (weights = exact 1.0)
  B) dataset_balance_method = "weighted_buckets" (weights ≈ 0.985–1.002)
  C) dataset_balance_method = "weighted_buckets" then force weights = 1.0
     (isolates the code path from the weight values)

Uses a synthetic dataset mimicking real td_lambda=1.0 data:
  ~77% draws (value 0), ~11.5% wins (+1), ~11.5% losses (-1)
with random board states, policies, and legal-move masks.

No self-play: just repeated supervised training from scratch with different
seeds, checking whether the value head collapses (value_pre_tanh > 5.0).

Usage:
    python scripts/2026-05-27_balance_collapse_test.py config/config-test.json
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from config import config
from a0.dataset import Dataset
from a0.model import create_model
from a0.train.dataset import train_model_epochs
from a0.eval.model_diagnostics import compute_iteration_diagnostics

NUM_TRIALS = 10
NUM_ITERATIONS = 2
DATASET_SIZE = 5000
COLLAPSE_THRESHOLD = 5.0
DRAW_FRAC = 0.77
WIN_FRAC = 0.115

def make_synthetic_dataset(seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    n = DATASET_SIZE
    bs = config.board_size
    policy_size = bs ** 4

    # values: ~77% draw (0), ~11.5% win (+1), ~11.5% loss (-1)
    n_draw = int(n * DRAW_FRAC)
    n_win = int(n * WIN_FRAC)
    n_loss = n - n_draw - n_win
    values = np.concatenate([
        np.zeros(n_draw),
        np.ones(n_win),
        -np.ones(n_loss),
    ]).astype(np.float32)
    rng.shuffle(values)
    values = values[:, None]  # (N, 1)

    # random board states (N, bs, bs, 2)
    states = rng.random((n, bs, bs, 2)).astype(np.float32)
    # random policies and masks
    policies = rng.random((n, policy_size)).astype(np.float32)
    masks = (rng.random((n, policy_size)) > 0.5).astype(np.float32)
    masks[:, 0] = 1.0  # at least one legal move

    ds = Dataset(config.training_batch_size)
    ds.set(states, values, policies, masks)
    return ds


def run_trial(seed: int, condition: str) -> dict:
    model = create_model(
        config.board_size,
        rngs=nnx.Rngs({'params': jax.random.PRNGKey(seed)})
    )

    # fixed probe batch for diagnostics (same across conditions for a given seed)
    probe_rng = np.random.default_rng(seed + 999999)
    bs = config.board_size
    probe_n = 256
    probe_ds = Dataset(probe_n)
    probe_ds.set(
        probe_rng.random((probe_n, bs, bs, 2)).astype(np.float32),
        probe_rng.uniform(-1, 1, (probe_n, 1)).astype(np.float32),
        probe_rng.random((probe_n, bs ** 4)).astype(np.float32),
        np.ones((probe_n, bs ** 4), dtype=np.float32),
    )

    collapsed_at = None
    pre_tanh_history = []

    for i in range(NUM_ITERATIONS):
        ds = make_synthetic_dataset(seed * 1000 + i)

        if condition == "weighted_buckets":
            ds.compute_value_weights_symmetric(
                n_buckets=config.num_buckets_for_balance,
                max_weight_ratio=config.max_weight_ratio,
            )
        elif condition == "weighted_then_ones":
            ds.compute_value_weights_symmetric(
                n_buckets=config.num_buckets_for_balance,
                max_weight_ratio=config.max_weight_ratio,
            )
            ds.weights = np.ones_like(ds.weights)

        model, _ = train_model_epochs(
            model=model, dataset=ds, num_epochs=1,
            save="none", plot=False, test_datasets={}
        )

        diag = compute_iteration_diagnostics(model, probe_ds)
        pre_tanh = diag['per_site_stats']['value_pre_tanh']['mean_abs']
        pre_tanh_history.append(pre_tanh)

        if pre_tanh > COLLAPSE_THRESHOLD and collapsed_at is None:
            collapsed_at = i + 1

    return {
        "seed": seed,
        "condition": condition,
        "collapsed": collapsed_at is not None,
        "collapsed_at": collapsed_at,
        "pre_tanh_history": pre_tanh_history,
    }


def main():
    out_dir = config.output_dir
    os.makedirs(out_dir, exist_ok=True)

    conditions = ["none", "weighted_buckets", "weighted_then_ones"]
    all_results = []

    for condition in conditions:
        n_collapsed = 0
        for trial in range(NUM_TRIALS):
            seed = trial * 7 + 42
            result = run_trial(seed, condition)
            all_results.append(result)
            if result["collapsed"]:
                n_collapsed += 1
            status = "COLLAPSED" if result["collapsed"] else "ok"
            pre_tanh_str = ", ".join(f"{v:.2f}" for v in result["pre_tanh_history"])
            print(f"  [{condition:25s}] trial {trial:2d} (seed={seed:5d}): {status:10s}  pre_tanh=[{pre_tanh_str}]", flush=True)
        print(f"  {condition}: {n_collapsed}/{NUM_TRIALS} collapsed ({n_collapsed/NUM_TRIALS:.0%})", flush=True)
        print()

    # save results
    results_path = os.path.join(out_dir, "balance_collapse_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")

    # summary
    print("\n=== SUMMARY ===")
    for condition in conditions:
        trials = [r for r in all_results if r["condition"] == condition]
        n_collapsed = sum(1 for r in trials if r["collapsed"])
        print(f"  {condition:25s}: {n_collapsed}/{len(trials)} collapsed ({n_collapsed/len(trials):.0%})")


if __name__ == "__main__":
    main()
