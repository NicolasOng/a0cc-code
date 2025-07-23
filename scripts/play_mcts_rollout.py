from a0.players.mcts_rollout import MCTSRolloutPlayer
from a0.players.human import HumanPlayer

from a0.game import play

from cc.core import Game

from config import config

def main():
    play(
        Game(config.board_size, config.num_pieces, False, False, False),
        players=[
            MCTSRolloutPlayer(board_size=config.board_size, num_pieces=config.num_pieces),
            HumanPlayer()
        ],
        turn_limit=1000
    )

if __name__ == "__main__":
    main()
