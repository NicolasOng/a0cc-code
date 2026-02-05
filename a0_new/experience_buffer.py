from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from a0_new.dataset import Dataset

class ExperienceData:
    def __init__(
            self,
            state: NDArray[np.float32],
            value: float,
            policy: NDArray[np.float32],
            mask: NDArray[np.float32]
        ) -> None:
        self.state: NDArray[np.float32] = state
        self.value: float = value
        self.policy: NDArray[np.float32] = policy
        self.mask: NDArray[np.float32] = mask

class ExperienceBuffer:
    def __init__(self, max_size: int) -> None:
        self.data: deque[ExperienceData] = deque(maxlen=max_size)

    def add(self, new_data: ExperienceData) -> None:
        self.data.append(new_data)
    
    def get_dataset(self, batch_size: int) -> Dataset:
        # Create a new static dataset with the same maxlen and batch_size
        new_dataset = Dataset(batch_size)
        new_dataset.set(
            np.stack([d.state for d in self.data]), # (board_size, board_size) -> (N, board_size, board_size, 2)
            np.array([d.value for d in self.data]) [:, None], # Add [:, None] to make its shape (N, 1)
            np.stack([d.policy for d in self.data]), # (board_size ** 4) -> (N, board_size ** 4)
            np.stack([d.mask for d in self.data]) # (board_size ** 4) -> (N, board_size ** 4)
        )
        return new_dataset

    def __len__(self) -> int:
        return len(self.data)
