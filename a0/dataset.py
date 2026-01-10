from __future__ import annotations

from typing import Generator

import jax.numpy as jnp
import jax.random as jrandom

class Dataset:
    states: jnp.ndarray
    values: jnp.ndarray
    policies: jnp.ndarray
    masks: jnp.ndarray

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
    
    def set(self, states: jnp.ndarray, values: jnp.ndarray, policies: jnp.ndarray, masks: jnp.ndarray) -> None:
        self.states = states
        self.values = values
        self.policies = policies
        self.masks = masks
    
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
    
    def split_off_test(self, test_size: int, shuffle: bool) -> Dataset:
        """Split off a test set of the specified size."""
        assert test_size < self.states.shape[0], "Test size must be less than the dataset size."

        # shuffle the dataset before splitting
        if shuffle:
            self.shuffle()
        
        # Create a new dataset for the test set
        test_dataset = Dataset(self.batch_size)
        
        # Split the data
        test_dataset.set(self.states[-test_size:], self.values[-test_size:], self.policies[-test_size:], self.masks[-test_size:])
        
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
        test_dataset.set(self.states[:n], self.values[:n], self.policies[:n], self.masks[:n])
        
        return test_dataset

    def shuffle(self) -> None:
        # shuffle the data
        key = jrandom.PRNGKey(0)
        perm = jrandom.permutation(key, self.states.shape[0])
        self.states = self.states[perm]
        self.values = self.values[perm]
        self.policies = self.policies[perm]
        self.masks = self.masks[perm]

    def batches(self) -> Generator[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], None, None]:
        data_len = self.states.shape[0]
        for i in range(0, data_len, self.batch_size):
            if i + self.batch_size > data_len:
                break
            yield self.states[i:i + self.batch_size], self.values[i:i + self.batch_size], self.policies[i:i + self.batch_size], self.masks[i:i + self.batch_size]

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
        
        win_count = int(jnp.sum(wins))
        draw_count = int(jnp.sum(draws))
        loss_count = int(jnp.sum(losses))
        
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
        win_count = int(jnp.sum(wins))
        loss_count = int(jnp.sum(losses))
        target_count = min(win_count, loss_count)
        draw_target_count = target_count // 50  # keep some draws, but fewer
        
        if target_count == 0:
            print("Cannot balance: one category has no samples")
            return
        
        # Randomly sample indices for each category
        key = jrandom.PRNGKey(42)
        indices_to_keep = []
        
        for mask, name in [(wins, "wins"), (draws, "draws"), (losses, "losses")]:
            category_indices = jnp.where(mask)[0]
            # Determine the target for this category
            category_target = draw_target_count if name == "draws" else target_count
            # for balancing, sample if there are more than needed (target different for draws)
            if len(category_indices) > category_target:
                key, subkey = jrandom.split(key)
                selected = jrandom.choice(subkey, category_indices, shape=(category_target,), replace=False)
                indices_to_keep.extend(selected.tolist())
            # otherwise, keep all
            else:
                indices_to_keep.extend(category_indices.tolist())
        
        # Update dataset
        indices_to_keep = jnp.array(sorted(indices_to_keep))
        self.states = self.states[indices_to_keep]
        self.values = self.values[indices_to_keep]
        self.policies = self.policies[indices_to_keep]
        self.masks = self.masks[indices_to_keep]
        
        print(f"Balanced to {target_count} samples each ({len(self)} total)")
        self.print_distribution()
    
    def clear_values(self) -> None:
        """
        Clear the values in the dataset (set all to 0).
        """
        self.values = jnp.zeros_like(self.values)
    
    def clear_policies(self) -> None:
        """
        Clear the policies in the dataset (set all to 0).
        """
        self.policies = jnp.zeros_like(self.policies)
