from __future__ import annotations
from enum import Enum
from typing import Callable, Iterable, Protocol, TypeVar
import math
import random

from cc.core import Board, Game, Move, Player


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


T = TypeVar("T")


def _pick_best_random_tie(items: Iterable[T], score_fn: Callable[[T], float]) -> T:
    best_score: float | None = None
    best: list[T] = []
    for x in items:
        s = score_fn(x)
        if best_score is None or s > best_score:
            best_score = s
            best = [x]
        elif s == best_score:
            best.append(x)
    return random.choice(best)


# --- Evaluators ---

class StateEvaluator(Protocol):
    def evaluate(self, state: Board) -> float: ...


class ZeroEval:
    def evaluate(self, state: Board) -> float:
        return 0.0


class DistEval:
    def __init__(self, root_player: Player, board_size: int, exclude_within: int = 0):
        self.root_player = root_player
        self.exclude_within = exclude_within
        opp_player = Player.PLAYER_O if root_player == Player.PLAYER_X else Player.PLAYER_X
        self.me_goal = goal_corner_for(root_player, board_size)
        self.opp_goal = goal_corner_for(opp_player, board_size)
        self.max_dist = 2 * (board_size - 1)

    def evaluate(self, state: Board) -> float:
        x_pos, o_pos = state.get_player_positions()
        if self.root_player == Player.PLAYER_X:
            me_pos, opp_pos = x_pos, o_pos
        else:
            me_pos, opp_pos = o_pos, x_pos
        d_me = goal_distance(me_pos, self.me_goal, self.exclude_within)
        d_opp = goal_distance(opp_pos, self.opp_goal, self.exclude_within)
        turn_bonus = 1 if state.current_player == self.root_player else 0
        raw = -d_me + d_opp + turn_bonus
        # squash to (-1, 1); scale so a lopsided race lands near saturation
        scale = max(1, len(me_pos) * self.max_dist)
        return math.tanh(raw / scale)


class LBDistEval(DistEval):
    def __init__(self, root_player: Player, board_size: int, home_size: int):
        super().__init__(root_player, board_size, exclude_within=home_size)


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
        if not forward:
            return random.choice(moves)
        candidates = self._candidates(state, forward)
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


class PolicyType(Enum):
    RANDOM = "random"
    FORWARD = "forward"
    BEST = "best"
    BACK = "back"


def make_evaluator(eval_type: EvaluatorType, root_player: Player, board_size: int, home_size: int) -> StateEvaluator:
    if eval_type == EvaluatorType.NONE:
        return ZeroEval()
    if eval_type == EvaluatorType.DIST:
        return DistEval(root_player, board_size)
    if eval_type == EvaluatorType.LBDIST:
        return LBDistEval(root_player, board_size, home_size)
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
