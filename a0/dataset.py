from __future__ import annotations

from typing import Generator

import jax.numpy as jnp
import jax.random as jrandom

class Dataset:
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
    
    def set(self, states: jnp.ndarray, values: jnp.ndarray, policies: jnp.ndarray) -> None:
        self.states = states
        self.values = values
        self.policies = policies
    
    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size
    
    def trim(self, new_size: int, shuffle: bool) -> None:
        if shuffle:
            self.shuffle()
        
        if new_size < self.states.shape[0]:
            self.states = self.states[:new_size]
            self.values = self.values[:new_size]
            self.policies = self.policies[:new_size]
    
    def split_off_test(self, test_size: int, shuffle: bool) -> Dataset:
        """Split off a test set of the specified size."""
        assert test_size < self.states.shape[0], "Test size must be less than the dataset size."

        # shuffle the dataset before splitting
        if shuffle:
            self.shuffle()
        
        # Create a new dataset for the test set
        test_dataset = Dataset(self.batch_size)
        
        # Split the data
        test_dataset.set(self.states[-test_size:], self.values[-test_size:], self.policies[-test_size:])
        
        # Trim the original dataset
        self.trim(len(self) - test_size, False)
        
        return test_dataset

    def shuffle(self) -> None:
        # shuffle the data
        key = jrandom.PRNGKey(0)
        perm = jrandom.permutation(key, self.states.shape[0])
        self.states = self.states[perm]
        self.values = self.values[perm]
        self.policies = self.policies[perm]

    def batches(self) -> Generator[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], None, None]:
        data_len = self.states.shape[0]
        for i in range(0, data_len, self.batch_size):
            if i + self.batch_size > data_len:
                break
            yield self.states[i:i + self.batch_size], self.values[i:i + self.batch_size], self.policies[i:i + self.batch_size]

    def num_batches(self) -> int:
        """Return the number of complete batches that will be output by the batches method."""
        data_len = len(self)
        return data_len // self.batch_size

    def __len__(self) -> int:
        return self.states.shape[0]
