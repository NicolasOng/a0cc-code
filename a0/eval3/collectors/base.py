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
    # target-refresh path: (iteration, refresh, target) tuples from the
    # generation iteration; None on classic runs and old pickles
    refresh_value_targets: list[tuple[int, int, float]] | None = None
    # game-level facts needed by trajectory-keyed collectors. `ended` separates
    # real terminal states from turn-limit timeouts (an unfinished game has no
    # terminal state, so distance-from-terminal is meaningless for it).
    # `game_index` is this game's position in its iteration's gamedata list —
    # the key the refresh sidecar rows join on.
    ended: bool = True
    game_index: int = -1

@dataclass
class GameInfo:
    iteration: int
    winner: Player | None
    ended: bool
    game_length: int
    game_time: float

def progress_bucket_upper_bounds(n_buckets: int) -> list[int]:
    '''Upper bounds of n equal game-progress buckets over 0-99, e.g. [10, 20, ... 100].'''
    size = 100 / n_buckets
    return [int((b + 1) * size) for b in range(n_buckets)]

def progress_bucket_for(progress: int, n_buckets: int) -> int:
    '''The bucket upper bound that `progress` (0-99) falls into.'''
    size = 100 / n_buckets
    idx = min(int(progress / size), n_buckets - 1)
    return int((idx + 1) * size)

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