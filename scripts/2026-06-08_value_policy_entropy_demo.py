'''
Demo: show a few "value-head policies" and their normalized entropies for the
model in the repo (model_49.pkl), using the same computation as
collect_value_policy_entropy / a0.utils.value_policy_entropy.

For a board, the value head is run on every child board (one per legal move).
A child board is from the opponent's perspective, so a good move yields a LOW
child value; negated child values are softmaxed into a move distribution (the
"value-head policy") and its normalized entropy is taken (0 = peaked on one
move, 1 = uniform over all moves).

Boards are sourced from ONE self-play game (model vs itself), so the positions
are in-distribution for the model, then a few boards spread across the game are
shown.

Run from repo root with the 25-6 config so board_size/encoding match the model:
    python scripts/2026-06-08_value_policy_entropy_demo.py config/config25-6.json
'''
import os
import sys
import random

import numpy as np

# Put the repo root on sys.path so `a0`/`config` import when run as a script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# config is the argv-driven singleton; override output_dir BEFORE any *_dir
# property is touched so we write under Downloads and don't disturb output/.
from config import config

OUT_DIR = os.path.expanduser("~/Downloads/2026-06-08_value_policy_entropy")
os.makedirs(OUT_DIR, exist_ok=True)
config.output_dir = OUT_DIR + "/"

from cc.core import Game
from a0.model import load_model
from a0.model_utils import board_to_input
from a0.players.a0 import A0Player
from a0.game import play
from a0.utils.value_policy_entropy import _softmax, normalized_policy_entropy

MODEL_PATH = "model_49.pkl"
NUM_BOARDS = 3              # how many positions from the game to inspect
DEMO_MCTS_SAMPLES = 64     # << config's 512, just to keep one game fast
TEMPERATURE = 1.0          # same default as collect_value_policy_entropy
SEED = 0


def play_one_selfplay_game(model):
    '''One model-vs-itself game; returns its GameData (turn_data has the boards
    actually visited, each with the legal moves available there).'''
    def make_player() -> A0Player:
        return A0Player(
            board_size=config.board_size,
            num_pieces=config.num_pieces,
            model=model,
            exploit=False,                 # sample w/ dirichlet noise -> a real game
            mcts_samples=DEMO_MCTS_SAMPLES,
            no_reverse_moves=not config.backwards_moves,
            no_illegal_moves=not config.illegal_moves,
            no_side_moves=not config.sideways_moves,
            rollout_type=config.rollout_type,
            rollout_depth=config.rollout_depth,
            policy_type=config.policy_type,
            epsilon=config.epsilon,
            dirichlet_epsilon=config.dirichlet_epsilon,
        )

    game = Game(
        config.board_size,
        config.num_pieces,
        no_reverse_moves=not config.backwards_moves,
        no_illegal_moves=not config.illegal_moves,
        no_side_moves=not config.sideways_moves,
    )
    return play(game, players=[make_player(), make_player()], turn_limit=config.turn_limit)


def pick_boards(turn_data, n: int):
    '''Boards with >1 legal move (build_child_inputs keeps these), spread evenly
    across the game so we see early / mid / late positions.'''
    usable = [(t.board, t.moves) for t in turn_data if len(t.moves) > 1]
    if len(usable) <= n:
        chosen = list(range(len(usable)))
    else:
        chosen = [round(i * (len(usable) - 1) / (n - 1)) for i in range(n)]
    return [(idx, usable[idx][0], usable[idx][1]) for idx in chosen]


def value_head_policy(model, board, moves):
    '''Returns (child_values, move_probs) for one board: raw value-head output on
    each child, and the softmax(-value) "value-head policy" over the moves.
    Mirrors child_value_entropies for a single board.'''
    import copy
    import jax.numpy as jnp

    child_inputs = []
    for move in moves:
        child = copy.deepcopy(board)
        child.apply_move(move)
        child_inputs.append(board_to_input(child))  # (1, B, B, 2)
    batch = jnp.asarray(np.concatenate(child_inputs, axis=0), dtype=jnp.float32)
    values, _ = model.inference(batch)
    values = np.array(values, dtype=np.float32).reshape(-1)
    probs = _softmax(-values, TEMPERATURE)
    return values, probs


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"Config: {config.path}  (board_size={config.board_size}, "
          f"num_pieces={config.num_pieces})")
    print(f"Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH, training=False)

    print(f"Playing one self-play game (mcts_samples={DEMO_MCTS_SAMPLES})...")
    data = play_one_selfplay_game(model)
    print(f"Game over: {len(data.turn_data)} turns, winner={data.winner}, "
          f"ended={data.ended}")

    boards = pick_boards(data.turn_data, NUM_BOARDS)

    lines: list[str] = []
    for shown, (turn_idx, board, moves) in enumerate(boards):
        values, probs = value_head_policy(model, board, moves)
        ent = normalized_policy_entropy(probs)

        header = (f"\n===== Board {shown + 1}/{len(boards)}  "
                  f"(game turn {turn_idx}, current_player={board.current_player}, "
                  f"{len(moves)} moves) =====")
        print(header)
        print(board.board_view())
        print(f"Normalized value-head-policy entropy: {ent:.4f}  "
              f"(0=peaked, 1=uniform)")

        # show moves sorted by the value-head policy probability
        order = np.argsort(-probs)
        print(f"{'move':>28}  {'child_val':>9}  {'prob':>6}")
        for j in order:
            print(f"{str(moves[j]):>28}  {values[j]:>9.4f}  {probs[j]:>6.3f}")

        lines.append(header.strip())
        lines.append(board.board_view())
        lines.append(f"entropy={ent:.4f}")
        for j in order:
            lines.append(f"  {str(moves[j])}  child_val={values[j]:.4f}  prob={probs[j]:.3f}")

    out_file = os.path.join(OUT_DIR, "value_policy_entropy_demo.txt")
    with open(out_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote summary to {out_file}")


if __name__ == "__main__":
    main()
