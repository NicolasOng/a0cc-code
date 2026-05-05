from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from cc.core import Player, Board

@dataclass
class TurnInfo:
    iteration: int
    board: Board
    winner: Player | None
    turn_index: int
    game_length: int
    gt_outcome: float
    experienced_outcome: float
    gt_policy: NDArray[np.float32]
    experienced_policy: NDArray[np.float32]
    is_trivial: bool
    progress: int    # 0-99, percentage through the game
    alternative_value_target: float | None
    alternative_policy_target: NDArray[np.float32] | None

@dataclass
class GameInfo:
    iteration: int
    winner: Player | None
    ended: bool
    game_length: int
    game_time: float

class Collector(Protocol):
    def on_game(self, gi: GameInfo) -> None: ...

    def on_turn(self, ti: TurnInfo) -> None: ...

    def on_iteration_end(self, iteration: int) -> None: ...

    def finalize(self) -> None: ...

class GameProgressCollector(Protocol):
    '''Protocol for collectors that track stats per game-progress bucket.'''
    def init_buckets(self, bucket_upper_bounds: list[int]) -> None: ...

    def on_turn(self, ti: TurnInfo, bucket: int) -> None: ...

    def on_iteration_end(self, iteration: int) -> None: ...

    def finalize(self) -> None: ...