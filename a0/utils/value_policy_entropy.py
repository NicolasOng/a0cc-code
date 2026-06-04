'''
Value-derived policy entropy.

For a board, the model's value head is run on every child board (one per legal
move). A child board is from the opponent's perspective, so a good move yields a
low child value; negated child values are softmaxed into a move distribution and
its normalized entropy is taken. This measures how peaked the value head's
implicit move preference is (sharp value fn -> low entropy, flat -> high).
'''
import copy

import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp
from scipy import stats

from cc.core import Board
from cc.ground_truth import RankUnrank
from a0.model import AlphaZeroModel
from a0.model_utils import board_to_input

from config import config


def normalized_policy_entropy(policy: NDArray[np.float32]) -> float:
    '''
    Normalized Shannon entropy of a target policy distribution.
    `policy` is a probability distribution over actions (illegal/unvisited
    moves are 0). Entropy is normalized by log(support size) so it lies in
    [0, 1]: 0 for a one-hot target, 1 for a target uniform over its support.
    Distributions with <= 1 nonzero entry return 0.
    '''
    p = np.asarray(policy, dtype=np.float64)
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    p = p / p.sum()
    ent = -np.sum(p * np.log(p))
    return float(ent / np.log(p.size))


def build_child_inputs(boards: list[Board], ranker: RankUnrank) -> tuple[NDArray[np.float32], list[int]]:
    '''
    Filters to non-terminal boards with more than one legal move, then encodes
    every child board (one per legal move) as a model input. Returns all
    children concatenated into one (M, board, board, 2) array plus a per-board
    count list, so flat model outputs can be split back per board.

    Terminal boards have no meaningful next-move distribution, and single-move
    boards (including forced passes) give a degenerate entropy of 0 that would
    only dilute the mean — so both are dropped. Children are model-independent,
    so this is computed once and reused across models.
    '''
    child_inputs: list[NDArray[np.float32]] = []
    counts: list[int] = []
    for board in boards:
        if ranker.is_terminal(board):
            continue
        moves = ranker.cc.generate_moves_for_given_board(board)
        if len(moves) <= 1:
            continue
        for move in moves:
            child = copy.deepcopy(board)
            child.apply_move(move)
            child_inputs.append(board_to_input(child))  # (1, board, board, 2)
        counts.append(len(moves))
    if child_inputs:
        concatenated = np.concatenate(child_inputs, axis=0)
    else:
        concatenated = np.zeros((0, config.board_size, config.board_size, 2), dtype=np.float32)
    return concatenated, counts


def _softmax(scores: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
    z = np.asarray(scores, dtype=np.float64) / temperature
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def child_value_entropies(
    model: AlphaZeroModel,
    child_inputs: NDArray[np.float32],
    counts: list[int],
    temperature: float = 1.0,
    batch_size: int = 256,
) -> list[float]:
    '''
    Runs the value head over all child inputs (in fixed-size chunks, keeping the
    final partial chunk), splits the flat values back per board via `counts`,
    and returns each board's normalized value-derived policy entropy. Child
    values are negated before softmax since they are from the opponent's
    perspective (lower child value = better move for us).
    '''
    values: list[NDArray[np.float32]] = []
    for i in range(0, child_inputs.shape[0], batch_size):
        batch = jnp.asarray(child_inputs[i:i + batch_size], dtype=jnp.float32)
        value, _ = model.inference(batch)
        values.append(np.array(value, dtype=np.float32).reshape(-1))
    flat = np.concatenate(values) if values else np.zeros(0, dtype=np.float32)

    entropies: list[float] = []
    offset = 0
    for c in counts:
        v = flat[offset:offset + c]
        offset += c
        entropies.append(normalized_policy_entropy(_softmax(-v, temperature)))
    return entropies


def mean_and_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    '''Mean and two-sided confidence-interval half-width across the given
    values, using the same t-based convention as the series-merging code.'''
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 0:
        return 0.0, 0.0
    mean = float(arr.mean())
    if n < 2:
        return mean, 0.0
    std = float(arr.std(ddof=1))
    t_value = float(stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1))
    ci = float(t_value * std / np.sqrt(n))
    return mean, ci
