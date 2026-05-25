from __future__ import annotations

from typing import Generator, Optional

import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

class Dataset:
    states: NDArray[np.float32]
    values: NDArray[np.float32]
    policies: NDArray[np.float32]
    masks: NDArray[np.float32]
    weights: Optional[NDArray[np.float32]]

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.weights = None

    def __setstate__(self, state: dict) -> None:
        # Backward compat: pickled datasets predating `weights` don't have the field.
        self.__dict__.update(state)
        if 'weights' not in state:
            self.weights = None

    def _effective_weights(self) -> NDArray[np.float32]:
        # Returns self.weights if set, else a ones array of shape (N, 1). Used when
        # emitting batches so downstream code can always unpack a weights tensor.
        if self.weights is not None:
            return self.weights
        return np.ones((self.states.shape[0], 1), dtype=np.float32)

    def set(self, states: NDArray[np.float32], values: NDArray[np.float32], policies: NDArray[np.float32], masks: NDArray[np.float32], weights: Optional[NDArray[np.float32]] = None) -> None:
        self.states = states
        self.values = values
        self.policies = policies
        self.masks = masks
        self.weights = weights

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def trim(self, new_size: int, shuffle: bool) -> None:
        if shuffle:
            self.shuffle()

        if new_size < self.states.shape[0]:
            self.states = self.states[:new_size]
            self.values = self.values[:new_size]
            self.policies = self.policies[:new_size]
            self.masks = self.masks[:new_size]
            if self.weights is not None:
                self.weights = self.weights[:new_size]

    def split_off_test(self, test_size: int, shuffle: bool) -> Dataset:
        """Split off a test set of the specified size."""
        assert test_size < self.states.shape[0], "Test size must be less than the dataset size."

        # shuffle the dataset before splitting
        if shuffle:
            self.shuffle()

        # Create a new dataset for the test set
        test_dataset = Dataset(self.batch_size)

        # Split the data
        test_weights = self.weights[-test_size:] if self.weights is not None else None
        test_dataset.set(self.states[-test_size:], self.values[-test_size:], self.policies[-test_size:], self.masks[-test_size:], test_weights)

        # Trim the original dataset
        self.trim(len(self) - test_size, False)

        return test_dataset

    def split_off_first_n(self, n: int, shuffle: bool) -> Dataset:
        """Split off a dataset of the specified size."""
        assert n < self.states.shape[0], "Dataset size must be less than the dataset size."

        # shuffle the dataset before splitting
        if shuffle:
            self.shuffle()

        # Create a new dataset for the test set
        test_dataset = Dataset(self.batch_size)

        # Split the data
        first_weights = self.weights[:n] if self.weights is not None else None
        test_dataset.set(self.states[:n], self.values[:n], self.policies[:n], self.masks[:n], first_weights)

        return test_dataset

    def shuffle(self) -> None:
        # shuffle the data
        perm = np.random.permutation(self.states.shape[0])
        self.states = self.states[perm]
        self.values = self.values[perm]
        self.policies = self.policies[perm]
        self.masks = self.masks[perm]
        if self.weights is not None:
            self.weights = self.weights[perm]

    def batches(self) -> Generator[tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]], None, None]:
        data_len = self.states.shape[0]
        weights = self._effective_weights()
        for i in range(0, data_len, self.batch_size):
            if i + self.batch_size > data_len:
                break
            yield self.states[i:i + self.batch_size], self.values[i:i + self.batch_size], self.policies[i:i + self.batch_size], self.masks[i:i + self.batch_size], weights[i:i + self.batch_size]

    def jnp_batches(self) -> Generator[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], None, None]:
        jnp_states = jnp.asarray(self.states, dtype=jnp.float32)
        jnp_values = jnp.asarray(self.values, dtype=jnp.float32)
        jnp_policies = jnp.asarray(self.policies, dtype=jnp.float32)
        jnp_masks = jnp.asarray(self.masks, dtype=jnp.float32)
        jnp_weights = jnp.asarray(self._effective_weights(), dtype=jnp.float32)
        data_len = jnp_states.shape[0]
        for i in range(0, data_len, self.batch_size):
            if i + self.batch_size > data_len:
                break
            yield jnp_states[i:i + self.batch_size], jnp_values[i:i + self.batch_size], jnp_policies[i:i + self.batch_size], jnp_masks[i:i + self.batch_size], jnp_weights[i:i + self.batch_size]

    def num_batches(self) -> int:
        """Return the number of complete batches that will be output by the batches method."""
        data_len = len(self)
        return data_len // self.batch_size

    def __len__(self) -> int:
        return self.states.shape[0]
    
    def get_distribution(self) -> tuple[int, int, int]:
        """
        Get the distribution of values in the dataset.
        Returns a tuple of (win_count, draw_count, loss_count).
        """
        wins = self.values > 0
        draws = self.values == 0  
        losses = self.values < 0
        
        win_count = int(np.sum(wins))
        draw_count = int(np.sum(draws))
        loss_count = int(np.sum(losses))

        return win_count, draw_count, loss_count
    
    def print_distribution(self) -> None:
        """
        Print the distribution of values in the dataset.
        """
        win_count, draw_count, loss_count = self.get_distribution()

        print(f"Dataset distribution:")
        print(f"  Wins (>0):  {win_count:6d} ({win_count/len(self):.2%})")
        print(f"  Draws (=0): {draw_count:6d} ({draw_count/len(self):.2%})")
        print(f"  Losses (<0):{loss_count:6d} ({loss_count/len(self):.2%})")

    def print_bucket_distribution(self, n: int = 10) -> None:
        """
        Log the distribution of values across n equal-width buckets over [-1, 1].
        """
        total = len(self)
        edges = np.linspace(-1, 1, n + 1)
        counts = np.histogram(self.values, bins=edges)[0]

        logger.log(25, f"Dataset distribution ({n} buckets):")
        for i in range(n):
            lo, hi = edges[i], edges[i + 1]
            count = int(counts[i])
            bar = "#" * int(40 * count / max(total, 1))
            logger.log(25, f"  [{lo:+.2f}, {hi:+.2f}{']' if i == n - 1 else ')'} {count:6d} ({count/total:.2%}) {bar}")
    
    def balance_values(self) -> None:
        """
        Balance the dataset to have equal proportions of wins (>0) and losses (<0).
        Reports the distribution before and after balancing.
        """
        # Categorize values by sign
        wins = self.values > 0
        draws = self.values == 0
        losses = self.values < 0
        
        self.print_distribution()

        # Find the minimum count to balance to
        win_count = int(np.sum(wins))
        loss_count = int(np.sum(losses))
        target_count = min(win_count, loss_count)
        draw_target_count = target_count // 50  # keep some draws, but fewer

        if target_count == 0:
            print("Cannot balance: one category has no samples")
            return

        # Randomly sample indices for each category
        np.random.seed(42)
        indices_to_keep = []

        for mask, name in [(wins, "wins"), (draws, "draws"), (losses, "losses")]:
            category_indices = np.where(mask)[0]
            # Determine the target for this category
            category_target = draw_target_count if name == "draws" else target_count
            # for balancing, sample if there are more than needed (target different for draws)
            if len(category_indices) > category_target:
                selected = np.random.choice(category_indices, size=category_target, replace=False)
                indices_to_keep.extend(selected.tolist())
            # otherwise, keep all
            else:
                indices_to_keep.extend(category_indices.tolist())

        # Update dataset
        indices_to_keep = np.array(sorted(indices_to_keep))
        self.states = self.states[indices_to_keep]
        self.values = self.values[indices_to_keep]
        self.policies = self.policies[indices_to_keep]
        self.masks = self.masks[indices_to_keep]
        if self.weights is not None:
            self.weights = self.weights[indices_to_keep]

        print(f"Balanced to {target_count} samples each ({len(self)} total)")
        self.print_distribution()
    
    def balance_values_symmetric(self, n_buckets: int = 20) -> None:
        """
        Balance the dataset by pairing symmetric buckets around 0 and downsampling
        the larger of each pair to match the smaller.
        For example, with n_buckets=20, bucket [-1, -0.9) is paired with [0.9, 1],
        [-0.9, -0.8) with [0.8, 0.9), etc.
        If n_buckets is odd, the middle bucket straddling 0 is left untouched.
        """
        edges = np.linspace(-1, 1, n_buckets + 1)
        n_pairs = n_buckets // 2
        has_middle = n_buckets % 2 == 1

        indices_to_keep: list[int] = []

        for i in range(n_pairs):
            # negative bucket: edges[i] to edges[i+1]
            neg_mask = (self.values >= edges[i]) & (self.values < edges[i + 1])
            # positive bucket: edges[n_buckets - 1 - i] to edges[n_buckets - i]
            j = n_buckets - 1 - i
            pos_mask = (self.values >= edges[j]) & (self.values <= edges[j + 1] if j + 1 == n_buckets else self.values < edges[j + 1])

            neg_indices = np.where(neg_mask)[0]
            pos_indices = np.where(pos_mask)[0]
            target = min(len(neg_indices), len(pos_indices))

            if target == 0:
                # keep whatever exists in either bucket
                indices_to_keep.extend(neg_indices.tolist())
                indices_to_keep.extend(pos_indices.tolist())
                continue

            for idx_arr in [neg_indices, pos_indices]:
                if len(idx_arr) > target:
                    selected = np.random.choice(idx_arr, size=target, replace=False)
                    indices_to_keep.extend(selected.tolist())
                else:
                    indices_to_keep.extend(idx_arr.tolist())

        # middle bucket: leave untouched
        if has_middle:
            mid = n_pairs
            mid_mask = (self.values >= edges[mid]) & (self.values < edges[mid + 1])
            indices_to_keep.extend(np.where(mid_mask)[0].tolist())

        indices_to_keep = np.array(sorted(indices_to_keep))
        old_size = len(self)
        self.states = self.states[indices_to_keep]
        self.values = self.values[indices_to_keep]
        self.policies = self.policies[indices_to_keep]
        self.masks = self.masks[indices_to_keep]
        if self.weights is not None:
            self.weights = self.weights[indices_to_keep]

        logger.log(25, f"Symmetric balance ({n_buckets} buckets): {old_size} -> {len(self)} samples")

    def compute_value_weights_symmetric(self, n_buckets: int = 2, max_weight_ratio: float = 10.0, draw_eps: float = 1e-3) -> None:
        """
        Populate per-sample weights using symmetric-pair inverse-frequency weighting
        over the value range [-1, 1]. Bucket i is paired with bucket n_buckets-1-i;
        each pair's effective target is min(h[i], h[n-1-i]), and each sample in
        bucket b gets w = target / h[b]. Weights are normalized to mean 1.0, clipped
        to max_weight_ratio, then re-normalized.

        If n_buckets is even, a tiny [-draw_eps, draw_eps] band is carved out of
        the innermost pair to form an implicit "draws" bucket left at weight 1.0.
        Without this, exact-zero draws fall into the positive bucket and get
        weighted as wins.
        """
        n = len(self)
        weights = np.ones(n, dtype=np.float32)

        edges = np.linspace(-1, 1, n_buckets + 1)
        n_pairs = n_buckets // 2
        has_middle = n_buckets % 2 == 1
        has_draw_bucket = not has_middle
        flat_values = self.values.reshape(-1)

        # assign raw inverse-frequency weights per symmetric pair
        for i in range(n_pairs):
            j = n_buckets - 1 - i
            low_neg, high_neg = edges[i], edges[i + 1]
            low_pos, high_pos = edges[j], edges[j + 1]
            # innermost pair (touches 0) with even n_buckets: exclude the draw band
            if has_draw_bucket and i == n_pairs - 1:
                high_neg = -draw_eps
                low_pos = draw_eps

            neg_mask = (flat_values >= low_neg) & (flat_values < high_neg)
            if j + 1 == n_buckets:
                pos_mask = (flat_values >= low_pos) & (flat_values <= high_pos)
            else:
                pos_mask = (flat_values >= low_pos) & (flat_values < high_pos)

            neg_count = int(np.sum(neg_mask))
            pos_count = int(np.sum(pos_mask))
            target = min(neg_count, pos_count)

            # empty pair: leave weights at 1.0 for any samples that did land here
            if target == 0:
                continue

            weights[neg_mask] = target / neg_count
            weights[pos_mask] = target / pos_count

        # middle bucket (odd n_buckets) is untouched — weight 1.0.
        # For even n_buckets, the draw band (-draw_eps, draw_eps) is implicitly
        # untouched since neither inner-pair mask covers it.
        if has_middle:
            mid = n_pairs
            mid_mask = (flat_values >= edges[mid]) & (flat_values < edges[mid + 1])
            weights[mid_mask] = 1.0

        # normalize to mean 1.0, clip high end, re-normalize
        mean_w = float(np.mean(weights))
        if mean_w > 0:
            weights /= mean_w
        weights = np.minimum(weights, max_weight_ratio)
        mean_w = float(np.mean(weights))
        if mean_w > 0:
            weights /= mean_w

        self.weights = weights.reshape(-1, 1).astype(np.float32)

        logger.log(25, f"Value weights ({n_buckets} buckets, max_ratio={max_weight_ratio}): "
                      f"min={float(self.weights.min()):.3f}, "
                      f"max={float(self.weights.max()):.3f}, "
                      f"mean={float(self.weights.mean()):.3f}")

    def print_shapes(self) -> None:
        logger.info(f"states:   {self.states.shape}")
        logger.info(f"values:   {self.values.shape}")
        logger.info(f"policies: {self.policies.shape}")
        logger.info(f"masks:    {self.masks.shape}")

    def clear_values(self) -> None:
        """
        Clear the values in the dataset (set all to 0).
        """
        self.values = np.zeros_like(self.values)

    def clear_policies(self) -> None:
        """
        Clear the policies in the dataset (set all to 0).
        """
        self.policies = np.zeros_like(self.policies)
