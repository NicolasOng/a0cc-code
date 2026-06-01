from __future__ import annotations
from enum import Enum
from typing import Callable, Iterable, Optional, Protocol
import random

import numpy as np

from cc.core import Board, Game, Move, Player
from cc.ranking import CCState, CCPSRank12


def goal_corner_for(player: Player, board_size: int) -> tuple[int, int]:
    if player == Player.PLAYER_X:
        return (board_size - 1, board_size - 1)
    return (0, 0)


def goal_distance(positions, goal_corner: tuple[int, int], exclude_within: int = 0) -> int:
    gx, gy = goal_corner
    total = 0
    for p in positions:
        d = abs(p.x - gx) + abs(p.y - gy)
        if d < exclude_within:
            continue
        total += d
    return total


def filter_forward(state: Board, moves: list[Move]) -> list[Move]:
    if state.current_player == Player.PLAYER_X:
        return [m for m in moves if m.is_down()]
    return [m for m in moves if m.is_up()]


def forward_progress(move: Move, player: Player) -> int:
    _, d_anti = move.diagonal_distances()
    return d_anti if player == Player.PLAYER_X else -d_anti


def _pick_best_random_tie(items: Iterable[Move], score_fn: Callable[[Move], float]) -> Move:
    best_score: float | None = None
    best: list[Move] = []
    for m in items:
        s = score_fn(m)
        if best_score is None or s > best_score:
            best_score = s
            best = [m]
        elif s == best_score:
            best.append(m)
    return random.choice(best)


# --- Evaluators ---
#
# Each evaluator owns *both* the depth-cap value (`evaluate`) and the terminal
# value (`terminal_value`). Splitting these lets each strategy keep one
# internal value scale: ZeroEval returns ±1 / 0 at terminals (preserves the
# "terminal is ground truth" property when there's no real heuristic), while
# DistEval/LBDistEval reuse their own raw eval at terminals so we don't have
# to normalize raw eval values into the ±1 terminal range.

class StateEvaluator(Protocol):
    def evaluate(self, state: Board) -> float: ...
    def terminal_value(self, state: Board, winner: Optional[Player], root_player: Player) -> float: ...


class ZeroEval:
    def evaluate(self, state: Board) -> float:
        return 0.0

    def terminal_value(self, state: Board, winner: Optional[Player], root_player: Player) -> float:
        if winner is None:
            return 0.0
        return 1.0 if winner == root_player else -1.0


class DistEval:
    def __init__(self, root_player: Player, board_size: int, exclude_within: int = 0):
        self.root_player = root_player
        self.exclude_within = exclude_within
        opp_player = Player.PLAYER_O if root_player == Player.PLAYER_X else Player.PLAYER_X
        self.me_goal = goal_corner_for(root_player, board_size)
        self.opp_goal = goal_corner_for(opp_player, board_size)

    def evaluate(self, state: Board) -> float:
        x_pos, o_pos = state.get_player_positions()
        if self.root_player == Player.PLAYER_X:
            me_pos, opp_pos = x_pos, o_pos
        else:
            me_pos, opp_pos = o_pos, x_pos
        d_me = goal_distance(me_pos, self.me_goal, self.exclude_within)
        d_opp = goal_distance(opp_pos, self.opp_goal, self.exclude_within)
        turn_bonus = 1 if state.current_player == self.root_player else 0
        return float(-d_me + d_opp + turn_bonus)

    def terminal_value(self, state: Board, winner: Optional[Player], root_player: Player) -> float:
        # Terminals are extrema of DistEval (a winning state minimizes d_me — the
        # 6 winning pieces tile the goal triangle — and typically has a large d_opp),
        # so reusing the eval here keeps the value scale uniform.
        return self.evaluate(state)


class LBDistEval(DistEval):
    def __init__(self, root_player: Player, board_size: int, home_size: int):
        super().__init__(root_player, board_size, exclude_within=home_size)


class DBEval:
    """Single-agent database (BFS) distance-to-goal evaluator.

    Drop-in for DistEval with the same -d_me + d_opp + turn_bonus shape, but d_me/d_opp
    are *exact* minimum single-agent moves to each player's goal (opponent absent), read
    from a precomputed BFS depth array indexed by P1 rank (see
    scripts/2026-05-29_single_agent_bfs.py). This captures jump shortcuts that the
    Manhattan-sum DistEval misses.

    The BFS array is seeded at the P1 (PLAYER_X) goal, and rank_p1 only ranks PLAYER_X
    pieces with to_move == 0. PLAYER_O's distance to its own goal is obtained by the 180-deg
    rotation symmetry that swaps the two goal corners — reversing the CCState spot index
    (i -> num_spots-1-i) maps an O configuration into the equivalent X frame.
    """

    # bfs[rank] == -1 marks a P1 config the BFS never reached. Legal in-game configs are
    # all reachable, but treat any stray -1 as "infinitely far from goal" so it can never
    # masquerade as a near-win (small d) in -d_me + d_opp.
    _UNREACHABLE = 1 << 14

    def __init__(self, root_player: Player, board_size: int, num_pieces: int,
                 bfs: np.ndarray, ranker: Optional[CCPSRank12] = None):
        self.root_player = root_player
        self.num_spots = board_size * board_size
        self.num_pieces = num_pieces
        self.bfs = bfs
        self.ranker = ranker or CCPSRank12(self.num_spots, 2, num_pieces)

    def _depth(self, ccstate_list: list[int], target: int, flip: bool) -> int:
        '''Min single-agent moves to goal for the pieces marked `target` (1=X, 2=O) in the
        diagonal-ordered ccstate_list. flip rotates the O frame into the X frame.'''
        idxs = [i for i, v in enumerate(ccstate_list) if v == target]
        if flip:
            idxs = [self.num_spots - 1 - i for i in idxs]
        s = CCState(self.num_spots, self.num_pieces, 2)
        s.board = [0] * self.num_spots
        for i in idxs:
            s.board[i] = 1
        s.build_pieces_from_board()
        s.to_move = 0
        d = int(self.bfs[self.ranker.rank_p1(s)])
        return self._UNREACHABLE if d < 0 else d

    def evaluate(self, state: Board) -> float:
        # Diagonal-ordered list once (1=X, 2=O); each side is ranked toward its OWN goal,
        # so the O side (whichever player that is) gets the rotation flip.
        ccstate_list = CCState.grid_to_CCState_order(state.board)
        if self.root_player == Player.PLAYER_X:
            me_target, opp_target = 1, 2
        else:
            me_target, opp_target = 2, 1
        d_me = self._depth(ccstate_list, me_target, flip=(self.root_player != Player.PLAYER_X))
        d_opp = self._depth(ccstate_list, opp_target, flip=(self.root_player == Player.PLAYER_X))
        turn_bonus = 1 if state.current_player == self.root_player else 0
        return float(-d_me + d_opp + turn_bonus)

    def terminal_value(self, state: Board, winner: Optional[Player], root_player: Player) -> float:
        # The goal config has depth 0, so a win minimizes d_me — an extremum of evaluate,
        # mirroring DistEval; reuse evaluate to keep one value scale.
        return self.evaluate(state)


# --- Rollout policies ---

class RolloutPolicy(Protocol):
    def choose(self, state: Board, moves: list[Move], game: Game) -> Move: ...


class RandomPolicy:
    def choose(self, state: Board, moves: list[Move], game: Game) -> Move:
        return random.choice(moves)


class RandomForwardPolicy:
    def choose(self, state: Board, moves: list[Move], game: Game) -> Move:
        forward = filter_forward(state, moves)
        if forward:
            return random.choice(forward)
        return random.choice(moves)


class _ProgressPolicy:
    def __init__(self, epsilon: float = 0.0):
        assert 0.0 <= epsilon <= 1.0, f"epsilon must be in [0, 1], got {epsilon}"
        self.epsilon = epsilon
        self._fallback = RandomForwardPolicy()

    def _candidates(self, state: Board, forward: list[Move]) -> list[Move]:
        raise NotImplementedError

    def choose(self, state: Board, moves: list[Move], game: Game) -> Move:
        if self.epsilon > 0 and random.random() < self.epsilon:
            return self._fallback.choose(state, moves, game)
        forward = filter_forward(state, moves)
        # if no forward moves, score over all moves instead — picks the "least bad"
        # (largest forward_progress, even if zero or negative) rather than blind random.
        pool = forward if forward else moves
        candidates = self._candidates(state, pool)
        return _pick_best_random_tie(candidates, lambda m: forward_progress(m, state.current_player))


class BestPlayoutPolicy(_ProgressPolicy):
    def _candidates(self, state: Board, forward: list[Move]) -> list[Move]:
        return forward


class BackPlayoutPolicy(_ProgressPolicy):
    def _candidates(self, state: Board, forward: list[Move]) -> list[Move]:
        gx, gy = goal_corner_for(state.current_player, len(state.board))
        def piece_dist(m: Move) -> int:
            return abs(m.start.x - gx) + abs(m.start.y - gy)
        max_dist = max(piece_dist(m) for m in forward)
        return [m for m in forward if piece_dist(m) == max_dist]


# --- Enum types + factories ---

class EvaluatorType(Enum):
    NONE = "none"
    DIST = "dist"
    LBDIST = "lbdist"
    DB = "db"


class PolicyType(Enum):
    RANDOM = "random"
    FORWARD = "forward"
    BEST = "best"
    BACK = "back"


def make_evaluator(eval_type: EvaluatorType, root_player: Player, board_size: int, home_size: int,
                   num_pieces: Optional[int] = None, bfs: Optional[np.ndarray] = None) -> StateEvaluator:
    if eval_type == EvaluatorType.NONE:
        return ZeroEval()
    if eval_type == EvaluatorType.DIST:
        return DistEval(root_player, board_size)
    if eval_type == EvaluatorType.LBDIST:
        return LBDistEval(root_player, board_size, home_size)
    if eval_type == EvaluatorType.DB:
        if bfs is None or num_pieces is None:
            raise ValueError("DB evaluator requires both `bfs` and `num_pieces`.")
        return DBEval(root_player, board_size, num_pieces, bfs)
    raise ValueError(f"Unknown evaluator type: {eval_type!r}.")


def make_policy(policy_type: PolicyType, epsilon: float = 0.0) -> RolloutPolicy:
    if policy_type == PolicyType.RANDOM:
        return RandomPolicy()
    if policy_type == PolicyType.FORWARD:
        return RandomForwardPolicy()
    if policy_type == PolicyType.BEST:
        return BestPlayoutPolicy(epsilon)
    if policy_type == PolicyType.BACK:
        return BackPlayoutPolicy(epsilon)
    raise ValueError(f"Unknown policy type: {policy_type!r}.")
