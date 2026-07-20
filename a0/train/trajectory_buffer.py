'''
Trajectory replay buffer for the target-refresh training path.

Unlike ExperienceBuffer (flat list of samples with fixed targets), this buffer
stores whole games so value targets can be *recomputed* against the current
model between training epochs ("refreshes"). Each refresh re-runs the shared
TD(lambda) backward recursion (a0.train.targets.td_lambda_targets) over every
buffered game, so targets never go stale and each refresh propagates the
terminal signal one more ply backward without new self-play.

Bookkeeping per refresh:
  - games from the current (generation) iteration: (iteration, refresh, target)
    tuples are appended onto turn.refresh_value_targets so they end up inside
    the gamedata pkl (which is saved AFTER the refresh loop in this path).
    turn.alternative_value_target keeps refresh-0 semantics (target computed
    with the model at iteration start) for eval3 compatibility.
  - older buffered games (their GameData ref has been dropped): rows
    (generation_iteration, game_index, turn_idx, rank, iteration, refresh,
    target) accumulate for a per-iteration sidecar file.
  - delta stats vs the previous refresh (mean/max |target change|) are
    returned so the caller can log signal-propagation progress live.
'''
from collections import deque

import numpy as np
from numpy.typing import NDArray

from config import config

from a0.game import GameData
from a0.dataset import Dataset
from a0.model import AlphaZeroModel, MultiTrunkAlphaZeroModel
from a0.model_utils import board_to_input, get_legal_move_mask_from_state
from a0.train.targets import td_lambda_targets, terminal_value
from cc.ground_truth import RankUnrank

from utils.log import get_logger
logger = get_logger(__name__)

class TrajectoryData:
    '''
    One game, stored as flat per-turn arrays. Built once when the game enters
    the buffer; everything derivable is precomputed here (board inputs, masks,
    ranks) so refreshes only run model inference + the backward recursion.
    '''
    def __init__(self, game_data: GameData, generation_iteration: int, game_index: int, ranker: RankUnrank) -> None:
        turn_data = game_data.turn_data
        # (T, 1, B, B, 2): per-turn board_to_input outputs, stacked.
        # Kept in this shape so dataset assembly matches ExperienceBuffer
        # (np.stack of (1,B,B,2) boards); inference squeezes axis 1.
        self.boards: NDArray[np.float32] = np.stack([board_to_input(turn.board) for turn in turn_data])
        self.policies: NDArray[np.float32] = np.stack([turn.player_data for turn in turn_data])
        self.masks: NDArray[np.float32] = np.stack(
            [get_legal_move_mask_from_state(turn.board, for_model=True) for turn in turn_data]
        )
        # ranks cached once for sidecar rows (position key for GT lookups)
        self.ranks: NDArray[np.int64] = np.array([ranker.rank(turn.board) for turn in turn_data], dtype=np.int64)
        self.z: float = terminal_value(game_data)
        self.generation_iteration: int = generation_iteration
        self.game_index: int = game_index
        # kept ONLY until the generation iteration's gamedata pkl is saved
        # (drop_game_data_refs), so refresh targets can be written onto TurnData
        self.game_data: GameData | None = game_data
        # targets from the previous refresh, for delta stats
        self.last_targets: NDArray[np.float32] | None = None

    def __len__(self) -> int:
        return self.boards.shape[0]

class TrajectoryReplayBuffer:
    '''
    Deque of TrajectoryData capped by total position count (same semantics as
    the flat buffer's replay_buffer_size); eviction is whole-game FIFO.
    '''
    def __init__(self, max_positions: int) -> None:
        self.max_positions = max_positions
        self.games: deque[TrajectoryData] = deque()
        self.num_positions = 0
        self.ranker = RankUnrank()
        # pending sidecar rows: (generation_iteration, game_index, turn_idx,
        # rank, iteration, refresh, target)
        self._sidecar_rows: list[tuple[int, int, int, int, int, int, float]] = []
        self._sidecar_lam_by_refresh: dict[int, float] = {}

    def add_game(self, game_data: GameData, iteration: int, game_index: int) -> None:
        if not game_data.turn_data:
            return
        traj = TrajectoryData(game_data, iteration, game_index, self.ranker)
        self.games.append(traj)
        self.num_positions += len(traj)
        while self.num_positions > self.max_positions and len(self.games) > 1:
            evicted = self.games.popleft()
            self.num_positions -= len(evicted)

    def drop_game_data_refs(self) -> None:
        '''Call after the generation iteration's gamedata pkls are saved:
        refresh targets for these games now route to the sidecar instead.'''
        for traj in self.games:
            traj.game_data = None

    def refresh_and_build_dataset(
        self,
        model: AlphaZeroModel | MultiTrunkAlphaZeroModel,
        lam: float,
        iteration: int,
        refresh: int,
        batch_size: int,
    ) -> tuple[Dataset, dict[str, float | int | None]]:
        '''
        Recomputes all value targets against the current model and returns
        (dataset, delta_stats). delta_stats has mean/max |target - previous
        refresh target| over positions that existed at the previous refresh
        (None when no positions qualify, e.g. the very first refresh).
        '''
        # 1) value inference over every buffered position, chunked to a fixed
        #    shape (same reasoning as the self-play path: one JIT kernel,
        #    bounded peak memory). Skipped entirely for lam == 1: the MC
        #    recursion never reads the values.
        num_positions = self.num_positions
        if lam == 1.0:
            all_values = np.zeros((num_positions, 1), dtype=np.float32)
        else:
            chunk_size = getattr(config, "refresh_inference_batch_size", 256)
            boards = np.concatenate([traj.boards[:, 0] for traj in self.games], axis=0)  # (N, B, B, 2)
            value_chunks = []
            for start in range(0, num_positions, chunk_size):
                chunk = boards[start:start + chunk_size]
                chunk_len = chunk.shape[0]
                if chunk_len < chunk_size:
                    padding = np.zeros((chunk_size - chunk_len, *chunk.shape[1:]), dtype=chunk.dtype)
                    chunk = np.concatenate([chunk, padding], axis=0)
                chunk_values, _ = model.inference(chunk)
                value_chunks.append(np.asarray(chunk_values[:chunk_len]))
            all_values = np.concatenate(value_chunks, axis=0)

        # 2) per-game backward recursion + delta stats + target bookkeeping
        deltas_sum, deltas_max, deltas_n = 0.0, 0.0, 0
        target_arrays: list[NDArray[np.float32]] = []
        offset = 0
        for traj in self.games:
            t = len(traj)
            game_values = all_values[offset:offset + t]
            offset += t
            targets = np.array(td_lambda_targets(game_values, traj.z, lam), dtype=np.float32)
            target_arrays.append(targets)

            if traj.last_targets is not None:
                d = np.abs(targets - traj.last_targets)
                deltas_sum += float(d.sum())
                deltas_max = max(deltas_max, float(d.max()))
                deltas_n += t
            traj.last_targets = targets

            if traj.game_data is not None:
                # generation iteration: write onto TurnData (ends up in the pkl)
                for turn_idx, turn in enumerate(traj.game_data.turn_data):
                    if turn.refresh_value_targets is None:
                        turn.refresh_value_targets = []
                    turn.refresh_value_targets.append((iteration, refresh, float(targets[turn_idx])))
                    if refresh == 0:
                        # refresh-0 semantics: target computed with the model at
                        # iteration start — matches what the non-refresh TD path
                        # would have recorded, keeps eval3 comparable
                        turn.alternative_value_target = float(targets[turn_idx])
                        turn.alternative_policy_target = turn.player_data
            else:
                # older buffered game: sidecar rows
                for turn_idx in range(t):
                    self._sidecar_rows.append((
                        traj.generation_iteration, traj.game_index, turn_idx,
                        int(traj.ranks[turn_idx]), iteration, refresh,
                        float(targets[turn_idx]),
                    ))
        self._sidecar_lam_by_refresh[refresh] = lam

        # 3) flat dataset, matching ExperienceBuffer.get_dataset shapes exactly:
        #    states (N, 1, B, B, 2), values (N, 1), policies (N, P), masks (N, P)
        dataset = Dataset(batch_size)
        dataset.set(
            np.concatenate([traj.boards for traj in self.games], axis=0),
            np.concatenate(target_arrays)[:, None].astype(np.float32),
            np.concatenate([traj.policies for traj in self.games], axis=0),
            np.concatenate([traj.masks for traj in self.games], axis=0),
        )

        delta_stats: dict[str, float | int | None] = {
            "mean": (deltas_sum / deltas_n) if deltas_n else None,
            "max": deltas_max if deltas_n else None,
            "n": deltas_n,
        }
        return dataset, delta_stats

    def pop_sidecar(self) -> dict:
        '''Returns and clears the pending sidecar payload for this iteration.'''
        payload = {
            "columns": ["generation_iteration", "game_index", "turn_idx", "rank", "iteration", "refresh", "target"],
            "rows": self._sidecar_rows,
            "lam_by_refresh": dict(self._sidecar_lam_by_refresh),
        }
        self._sidecar_rows = []
        self._sidecar_lam_by_refresh = {}
        return payload

    def __len__(self) -> int:
        return self.num_positions
