"""Play a game against a trained A0 model, as a human on the terminal.

Usage:
    python scripts/play_vs_trained_model.py <model.pkl> [board_size] [num_pieces]

Defaults to the 16-3 board (board_size 4, 3 pieces), the smallest one, which
is the quickest to play by hand. The board/piece counts must match the model's
-- a 25-6 checkpoint will not load into a 16-3 game.
"""
import os
import sys
# moved here from the repo root; keep repo-root imports working
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from a0.game import play
from a0.players.a0 import A0Player
from a0.players.human import HumanPlayer
from a0.model import load_model

from cc.core import Game

if len(sys.argv) < 2:
    sys.exit(__doc__)

model_path = sys.argv[1]
board_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4
num_pieces = int(sys.argv[3]) if len(sys.argv) > 3 else 3

# match the training-time rules: no backwards moves, sideways allowed,
# illegal moves filtered out (see config/config.json)
game = Game(
    board_size=board_size,
    num_pieces=num_pieces,
    repeats_for_draw=6,
    no_reverse_moves=True,
    no_illegal_moves=True,
    no_side_moves=False,
)
model = load_model(model_path)
player = A0Player(board_size, num_pieces, model, exploit=True)
human = HumanPlayer()
game_data = play(game, [player, human], turn_limit=100)
