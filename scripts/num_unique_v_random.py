from a0.players.random import RandomPlayer
from cc.core import Board

from a0.game import play

from cc.core import Game

import matplotlib.pyplot as plt

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def generate_data_from_random_play(num_states: int = 100000, random_percent: float = 1.0) -> tuple[list[int], list[float]]:
    player = RandomPlayer(
        random_percent=random_percent
    )

    state_ns: list[int] = []
    unique_percent: list[float] = []

    unique_states: set[Board] = set()
    total_states = 0
    while True:
        game = Game(
            board_size=config.board_size,
            num_pieces=config.num_pieces,
            repeats_for_draw=-1,
            no_reverse_moves=True,
            no_illegal_moves=False,
            no_side_moves=False
        )
        gamedata = play(game, players=[player, player], turn_limit=80)
        logger.info(f"Game ended: {gamedata.ended}, Winner: {gamedata.winner}, Turns: {len(gamedata.turn_data)}")
        for turn in gamedata.turn_data:
            unique_states.add(turn.board)
            total_states += 1
        unique_states_percentage = (len(unique_states) / total_states) if total_states > 0 else 0.0
        logger.info(f"Unique states percent: {unique_states_percentage:.2%} ({len(unique_states)}/{total_states})")
        state_ns.append(total_states)
        unique_percent.append(unique_states_percentage)
        if total_states >= num_states:
            break
    
    return state_ns, unique_percent

def plot_data(all_data: dict[float, tuple[list[int], list[float]]]) -> None:
    plt.figure(figsize=(10, 6))
    for rp, (state_ns, unique_percent) in all_data.items():
        plt.plot(state_ns, unique_percent, label=f'Random Percent: {rp}')
    plt.xlabel('Number of States Encountered')
    plt.ylabel('Percentage of Unique States')
    plt.title('Unique States vs. Total States Encountered for Different Randomness Levels')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    rps: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
    all_data: dict[float, tuple[list[int], list[float]]] = {}
    for rp in rps:
        logger.log(25, f"Generating data with random percent: {rp}")
        state_ns, unique_percent = generate_data_from_random_play(num_states=100000, random_percent=rp)
        all_data[rp] = (state_ns, unique_percent)
    plot_data(all_data)

if __name__ == "__main__":
    setup_logging(
        level=25,
        log_dir=config.log_dir,
        process_name="num_unique_v_random"
    )
    main()
