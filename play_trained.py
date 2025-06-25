from a0.game import play
from a0.players.a0 import A0Player
from a0.players.human import HumanPlayer
from a0.model import load_model

from cc.core import Game

game = Game(4, 3, True, False, True)
model = load_model("data/training/model_30.pkl")
player = A0Player(4, 3, model)
human = HumanPlayer()
game_data = play(game, [player, human], turn_limit=100)
