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

import matplotlib.pyplot as plt

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def generate_states_from_random_play(n: int = 100, random_percent: float = 1.0) -> tuple[float, int, int]:
    player = RandomPlayer(
        random_percent=random_percent
    )

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
        logger.info(f"Game {i+1} ended: {gamedata.ended}, Winner: {gamedata.winner}, Turns: {len(gamedata.turn_data)}")
        for turn in gamedata.turn_data:
            unique_states.add(turn.board)
            total_states += 1
        unique_states_percentage = (len(unique_states) / total_states) if total_states > 0 else 0.0
        logger.info(f"Game {i+1}/{n} completed. Unique states percent: {unique_states_percentage:.2%} ({len(unique_states)}/{total_states})")
    
    return unique_states_percentage, total_states, len(unique_states)

def get_unique_percent_vs_random_percent(n: int = 100, random_percents: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]) -> dict[float, float]:
    results: dict[float, float] = {}
    for rp in random_percents:
        logger.log(25, f"Generating states with random percent: {rp}")

        unique_states_percentage, total_states, unique_states_count = generate_states_from_random_play(n, random_percent=rp)

        results[rp] = unique_states_percentage
        logger.log(25, f"Random Percent: {rp}, Unique States Percent: {unique_states_percentage:.2%} ({unique_states_count}/{total_states})")
    
    return results

def plot_results(results: dict[float, float]) -> None:
    random_percents = list(results.keys())
    unique_percents = list(results.values())

    plt.figure(figsize=(10, 6))
    plt.plot(random_percents, unique_percents, marker='o')
    plt.title('Unique States Percentage vs Random Move Percentage')
    plt.xlabel('Random Move Percentage')
    plt.ylabel('Unique States Percentage')
    plt.xticks(random_percents)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    setup_logging(
        level=25,
        log_dir=config.log_dir,
        process_name="random_games_unique_states"
    )

    r = get_unique_percent_vs_random_percent(n=1000, random_percents=[0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0])
    plot_results(r)