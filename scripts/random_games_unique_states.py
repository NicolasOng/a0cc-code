from a0.players.human import HumanPlayer
from a0.players.mcts_rollout import MCTSRolloutPlayer
from a0.players.gt import GroundTruthPlayer
from a0.players.model import ModelPlayer
from a0.players.random import RandomPlayer
from a0.players.a0 import A0Player
from cc.core import Board

from a0.model import load_model

from a0.game import play

from cc.core import Game

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def generate_states_from_random_play(n: int = 100):
    player = RandomPlayer()

    unique_states: set[Board] = set()
    total_states = 0
    for i in range(n):
        game = Game(
            board_size=config.board_size,
            num_pieces=config.num_pieces,
            repeats_for_draw=-1,
            no_reverse_moves=True,
            no_illegal_moves=False,
            no_side_moves=False
        )
        gamedata = play(game, players=[player, player], turn_limit=80)
        print(f"Game {i+1} ended: {gamedata.ended}, Winner: {gamedata.winner}, Turns: {len(gamedata.turn_data)}")
        for turn in gamedata.turn_data:
            unique_states.add(turn.board)
            total_states += 1
        unique_states_percentage = (len(unique_states) / total_states) if total_states > 0 else 0
        print(f"Game {i+1}/{n} completed. Unique states percent: {unique_states_percentage:.2%} ({len(unique_states)}/{total_states})")

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="random_games_unique_states"
    )

    generate_states_from_random_play(1000)
