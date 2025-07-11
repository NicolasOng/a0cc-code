from collections import deque
from typing import Generator

import jax.numpy as jnp
import jax.random as jrandom

class TrainingData:
    def __init__(self, board: jnp.ndarray, value: float, policy: jnp.ndarray):
        self.board: jnp.ndarray = board
        self.value: float = value
        self.policy: jnp.ndarray = policy

class Dataset:
    def __init__(self, max_size: int, batch_size: int, static: bool=False) -> None:
        self.batch_size = batch_size
        self.static = static
        self.data: deque[TrainingData] = deque(maxlen=max_size)
    
    def set(self, states: jnp.ndarray, values: jnp.ndarray, policies: jnp.ndarray) -> None:
        assert self.static, "Dataset is not static; cannot set new data."
        self.states = states
        self.values = values
        self.policies = policies

    def add(self, new_data: TrainingData) -> None:
        assert not self.static, "Dataset is static; cannot add new data."
        self.data.append(new_data)
    
    def trim(self, new_size: int) -> None:
        if not self.static and new_size < len(self.data):
            self.data = deque(list(self.data)[:new_size], maxlen=new_size)
        elif self.static and new_size < self.states.shape[0]:
            self.states = self.states[:new_size]
            self.values = self.values[:new_size]
            self.policies = self.policies[:new_size]
    
    def convert_to_static(self) -> None:
        assert not self.static, "Dataset is already static."
        # Convert the deque of training_data to 3 separate jnp.ndarrays
        self.states = jnp.stack([d.board for d in self.data]) # (board_size, board_size) -> (N, board_size, board_size)
        self.values = jnp.array([d.value for d in self.data]) [:, None]  # Add [:, None] to make its shape (N, 1)
        self.policies = jnp.stack([d.policy for d in self.data]) # (board_size ** 4) -> (N, board_size ** 4)
        self.data.clear()  # Clear the deque as we no longer need it
        self.static = True  # Mark the dataset as static

    def shuffle(self) -> None:
        if not self.static:
            # Convert the deque of training_data to 3 separate jnp.ndarrays
            self.states = jnp.stack([d.board for d in self.data]) # (board_size, board_size) -> (N, board_size, board_size)
            self.values = jnp.array([d.value for d in self.data]) [:, None]  # Add [:, None] to make its shape (N, 1)
            self.policies = jnp.stack([d.policy for d in self.data]) # (board_size ** 4) -> (N, board_size ** 4)
        
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
        return len(self.data) if not self.static else self.states.shape[0]
