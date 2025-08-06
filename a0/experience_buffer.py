from __future__ import annotations

from collections import deque

import jax.numpy as jnp

from a0.dataset import Dataset

class ExperienceData:
    def __init__(self, board: jnp.ndarray, value: float, policy: jnp.ndarray):
        self.board: jnp.ndarray = board
        self.value: float = value
        self.policy: jnp.ndarray = policy

class ExperienceBuffer:
    def __init__(self, max_size: int) -> None:
        self.data: deque[ExperienceData] = deque(maxlen=max_size)

    def add(self, new_data: ExperienceData) -> None:
        self.data.append(new_data)
    
    def get_dataset(self, batch_size: int) -> Dataset:
        # Create a new static dataset with the same maxlen and batch_size
        new_dataset = Dataset(batch_size)
        new_dataset.set(
            jnp.stack([d.board for d in self.data]), # (board_size, board_size) -> (N, board_size, board_size)
            jnp.array([d.value for d in self.data]) [:, None], # Add [:, None] to make its shape (N, 1)
            jnp.stack([d.policy for d in self.data]) # (board_size ** 4) -> (N, board_size ** 4)
        )
        return new_dataset

    def __len__(self) -> int:
        return len(self.data)
