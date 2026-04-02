"""
Test script for Dataset.balance_values_symmetric().
Creates a synthetic dataset with an imbalanced distribution of soft targets
and demonstrates the symmetric balancing across different bucket counts.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.log import setup_logging
setup_logging(level=25)

from a0.dataset import Dataset

def make_synthetic_dataset(n: int = 2000) -> Dataset:
    """
    Create a dataset with an intentionally skewed distribution:
    - Many samples near +1 (positive bias)
    - Fewer samples near -1
    - A scattering in the middle
    """
    rng = np.random.default_rng(0)

    # 1000 samples clustered near +0.9 to +1.0
    pos_heavy = rng.uniform(0.85, 1.0, size=1000)
    # 300 samples clustered near -0.9 to -1.0
    neg_light = rng.uniform(-1.0, -0.85, size=300)
    # 400 samples spread across the middle
    middle = rng.uniform(-0.5, 0.5, size=400)
    # 200 moderate positive
    mod_pos = rng.uniform(0.3, 0.7, size=200)
    # 100 moderate negative
    mod_neg = rng.uniform(-0.7, -0.3, size=100)

    values = np.concatenate([pos_heavy, neg_light, middle, mod_pos, mod_neg]).astype(np.float32)
    n = len(values)

    # dummy states, policies, masks (not relevant for balancing)
    states = rng.random((n, 1, 5, 5)).astype(np.float32)
    policies = rng.random((n, 25)).astype(np.float32)
    masks = np.ones((n, 25), dtype=np.float32)

    ds = Dataset(batch_size=32)
    ds.set(states, values, policies, masks)
    return ds


def test_balance(n_buckets: int) -> None:
    ds = make_synthetic_dataset()
    print(f"\n{'='*60}")
    print(f"  n_buckets = {n_buckets}")
    print(f"{'='*60}")

    print(f"\nBefore balancing ({len(ds)} samples):")
    ds.print_bucket_distribution(n=n_buckets)

    ds.balance_values_symmetric(n_buckets=n_buckets)

    print(f"\nAfter balancing ({len(ds)} samples):")
    ds.print_bucket_distribution(n=n_buckets)


if __name__ == "__main__":
    for nb in [2, 10, 20, 40]:
        test_balance(nb)
