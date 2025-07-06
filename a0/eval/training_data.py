import pickle
from tqdm import tqdm
from itertools import zip_longest
from typing import Generator

import numpy as np
import matplotlib.pyplot as plt

from cc.core import Player
from cc.lookups import CCBaselineSolver
from a0.dataset import Dataset, TrainingData
from a0.game import GameData
from a0.train.alphazero import board_to_input

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

training_data_path = config.training_dir

def game_data_generator() -> Generator[list[GameData], None, None]:
    '''
    Load game data from the training data path.
    Returns a list of lists of game data.
    Each list corresponds to a single training iteration.
    '''
    # load all the game data objects from config.training_dir
    #game_data_lists: list[list[GameData]] = []
    for i in tqdm(range(config.training_iterations)):
        file_path = f"{config.training_dir}/gamedata_{i + 1}.pkl"
        with open(file_path, 'rb') as file:
            data: list[GameData] = pickle.load(file)
            #game_data_lists.append(data)
            yield data
    #logger.info(f"Loaded {len(game_data_lists)} gamedata lists from {config.training_dir}.")
    #return game_data_lists

def training_data_generator() -> Generator[list[TrainingData], None, None]:
    '''
    Load training data from the training data path.
    Returns a list of lists of training data.
    Each list corresponds to a single training iteration.
    '''
    # load all the training data objects from config.training_dir
    #training_data_lists: list[list[TrainingData]] = []
    for i in tqdm(range(config.training_iterations)):
        file_path = f"{config.training_dir}/training_set_{i + 1}.pkl"
        with open(file_path, 'rb') as file:
            data: list[TrainingData] = pickle.load(file)
            #training_data_lists.append(data)
            yield data
    #logger.info(f"Loaded {len(training_data_lists)} training data lists from {config.training_dir}.")
    #return training_data_lists

def check_gd_and_td_equivalence(game_data_lists: list[list[GameData]] | Generator[list[GameData], None, None], training_data_lists: list[list[TrainingData]] | Generator[list[TrainingData], None, None]) -> bool:
    '''
    Check if the game data and training data are equivalent.
    This is done by comparing the game data and training data for each iteration.

    Game data lists is a list of lists of GameData objects,
    where each inner list holds all the gamedata from a single training iteration.
    Training data lists is a list of lists of TrainingData objects,
    where each inner list holds all the training data objects from a single training iteration.
    '''
    # in each iteration,
    for i, (game_data_list, training_data_list) in enumerate(zip_longest(game_data_lists, training_data_lists)):
        if game_data_list is None:
            logger.error(f"Game data list for iteration {i} is None.")
            return False
        if training_data_list is None:
            logger.error(f"Training data list for iteration {i} is None.")
            return False
        state_no = 0
        # for each game in the iteration,
        for j, game_data in enumerate(game_data_list):
            # for each state in the game,
            for k, game_state in enumerate(game_data.turn_data):
                # get the corresponding training data
                training_data = training_data_list[state_no]
                state_no += 1
                # check if the board states match
                gs_policy = game_state.player_data
                td_policy = training_data.policy
                if not np.array_equal(gs_policy, td_policy):
                    logger.debug(f"Policy mismatch at iteration {i}, game {j}, turn {k}.")
                    return False
                gs_value = 0 if game_data.winner is None else 1 if game_data.winner == game_state.board.current_player else -1
                td_value = training_data.value
                if gs_value != td_value:
                    logger.error(f"Value mismatch at iteration {i}, game {j}, turn {k}: {gs_value} != {td_value}")
                    return False
                gs_state = board_to_input(game_state.board)
                td_state = training_data.board
                if not np.array_equal(gs_state, td_state):
                    logger.error(f"State mismatch at iteration {i}, game {j}, turn {k}.")
                    return False

    logger.info("Game data and training data equivalence check completed.")
    return True

def check_winners_match(game_data_lists: list[list[GameData]] | Generator[list[GameData], None, None]) -> bool:
    '''
    Check if the winners in the game data match the winners in the game objects.
    This is done by comparing the winner in each GameData object with the winner in the Game object
    and the winner in the last TurnData object.
    Returns True if all winners match, False otherwise.
    '''
    for i, game_data_list in enumerate(game_data_lists):
        for j, game_data in enumerate(game_data_list):
            game = game_data.game
            winner = game_data.winner
            game_winner = game.winner
            board_winner = game.get_winner(game_data.turn_data[-1].board)
            if winner != game_winner:
                logger.error(f"Game winner mismatch at iteration {i}, game {j}: {winner} != {game_winner}")
                return False
            # the final board state is not recorded yet, so this check doesn't work.
            # if winner != board_winner:
            #     logger.error(f"Board winner mismatch at iteration {i}, game {j}: {winner} != {board_winner}")
            #     logger.error(f"\n{game_data.turn_data[-1].board.board_view()}")
            #     logger.error("BOARD HISTORY:")
            #     for board in game.board_history:
            #         logger.error(f"\n{board.board_view()}")
            #     return False
    return True

def check_game_data_accuracy(game_data_lists: list[list[GameData]] | Generator[list[GameData], None, None]) -> None:

    solver = CCBaselineSolver(config.solve_data, config.num_spots, config.num_players, config.num_pieces)
    iteration_accuracies: list[float] = []
    for i, game_data_list in enumerate(game_data_lists):
        iteration_num_correct = 0
        iteration_total = 0
        for j, game_data in enumerate(game_data_list):
            winner = game_data.winner
            for k, game_state in enumerate(game_data.turn_data):
                board = game_state.board
                sd_outcome = solver.get_outcome(board)
                gd_outcome = 0 if winner is None else 1 if winner == board.current_player else -1
                if sd_outcome == gd_outcome:
                    iteration_num_correct += 1
                iteration_total += 1
        iteration_accuracy = iteration_num_correct / iteration_total if iteration_total > 0 else 0
        iteration_accuracies.append(iteration_accuracy)
        logger.info(f"Iteration {i} accuracy: {iteration_accuracy:.2%} ({iteration_num_correct}/{iteration_total})")
    
    output_path = f"{config.eval_dir}/gamedata_acc.pkl"
    with open(output_path, 'wb') as file:
        pickle.dump(iteration_accuracies, file)
    logger.info(f"Game data accuracies saved to {output_path}.")

    plt.plot(iteration_accuracies)
    plt.title("Training Data Accuracy by Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.savefig(f"{config.plot_dir}training_data_accuracy.png")
    plt.clf()

class GameDataStats:
    '''
    Class to hold statistics about game data for a single self-play iteration.
    '''
    def __init__(self):
        self.total_games = 0
        self.player_x_wins: list[int] = []
        self.player_o_wins: list[int] = []
        self.draws_repeat: list[int] = []
        self.draws_timeout: list[int] = []
        self.game_lengths: list[int] = []
        self.game_times: list[float] = []
    
    def get_line(self) -> str:
        '''
        Get a string representation of the statistics for this iteration.
        '''
        return (f"{self.total_games}, "
                f"{sum(self.player_x_wins)}, "
                f"{sum(self.player_o_wins)}, "
                f"{sum(self.draws_repeat)}, "
                f"{sum(self.draws_timeout)}, "
                f"{np.mean(self.game_lengths) if self.game_lengths else 0:.2f}, "
                f"{np.std(self.game_lengths) if self.game_lengths else 0:.2f}, "
                f"{np.mean(self.game_times) if self.game_times else 0:.2f} seconds, "
                f"{np.std(self.game_times) if self.game_times else 0:.2f} seconds")
    
    def get_header(self) -> str:
        '''
        Get a header string for the statistics.
        '''
        return ("Total Games, Player X Wins, Player O Wins, "
                "Draws (Repeat), Draws (Timeout), "
                "Avg Game Length, Std Game Length, "
                "Avg Game Time, Std Game Time")

def game_data_stats() -> None:
    '''
    Print statistics about the game data.
    This includes the number of games, turns, and players.
    '''
    logger.info("Calculating game data statistics...")
    iteration_stats: list[GameDataStats] = []
    # for each iteration,
    gd_gen = game_data_generator()
    for game_data_list in gd_gen:
        # get the stats for the game data generated in that iteration
        # wins/losses/draws, types of draws, num turns (mean, etc), avg length in time, who won
        logger.info(f"Processing {len(game_data_list)} games in this iteration.")
        stats = GameDataStats()
        for game_data in game_data_list:
            stats.total_games += 1
            # get the game length and time,
            stats.game_lengths.append(len(game_data.turn_data))
            stats.game_times.append(game_data.time)
            # check for a winner
            px_won = 1 if game_data.winner == Player.PLAYER_X else 0
            po_won = 1 if game_data.winner == Player.PLAYER_O else 0
            stats.player_x_wins.append(px_won)
            stats.player_o_wins.append(po_won)
            # check for draws
            draw_repeat = 1 if game_data.ended and game_data.winner is not None else 0
            draw_timeout = 1 if not game_data.ended else 0
            stats.draws_repeat.append(draw_repeat)
            stats.draws_timeout.append(draw_timeout)
        # add the stats for this iteration to the list
        iteration_stats.append(stats)
    
    # save the statistics to a file
    logger.info("Saving game data statistics...")
    output_path = f"{config.eval_dir}/gamedata_stats.pkl"
    with open(output_path, 'wb') as file:
        pickle.dump(iteration_stats, file)
    logger.info(f"Game data accuracies saved to {output_path}.")

    # log the statistics
    logger.info("Game Data Statistics:")
    logger.info(iteration_stats[0].get_header())
    for i, stats in enumerate(iteration_stats):
        logger.info(f"Iteration {i}: {stats.get_line()}")
    
    # make plots for the statistics
    logger.info("Plotting game data statistics...")

    # first, plot outcomes
    logger.info("Plotting game outcomes (stacked)...")
    iterations = list(range(len(iteration_stats)))
    player_x_wins = [sum(stats.player_x_wins) for stats in iteration_stats]
    player_o_wins = [sum(stats.player_o_wins) for stats in iteration_stats]
    draws_repeat = [sum(stats.draws_repeat) for stats in iteration_stats]
    draws_timeout = [sum(stats.draws_timeout) for stats in iteration_stats]

    plt.clf()
    plt.stackplot(iterations, player_x_wins, player_o_wins, draws_repeat, draws_timeout,
        labels=['Player X Wins', 'Player O Wins', 'Draws (Repeat)', 'Draws (Timeout)'],
    )
    plt.title('Game Outcomes by Training Iteration')
    plt.xlabel('Training Iteration')
    plt.ylabel('Number of Games')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{config.plot_dir}/gamedata_outcomes_stacked.png")
    plt.clf()

    logger.info("Plotting game outcomes (line)...")
    plt.plot(iterations, player_x_wins, label='Player X Wins')
    plt.plot(iterations, player_o_wins, label='Player O Wins')
    plt.plot(iterations, draws_repeat, label='Draws (Repeat)')
    plt.plot(iterations, draws_timeout, label='Draws (Timeout)')
    
    plt.title('Game Outcomes by Training Iteration')
    plt.xlabel('Training Iteration')
    plt.ylabel('Number of Games')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{config.plot_dir}/gamedata_outcomes_lines.png")
    plt.clf()

    # then, plot game lengths
    logger.info("Plotting game lengths...")
    avg_game_lengths = [np.mean(stats.game_lengths) for stats in iteration_stats]
    std_game_lengths = [np.std(stats.game_lengths) for stats in iteration_stats]
    plt.errorbar(iterations, avg_game_lengths, yerr=std_game_lengths, 
                 marker='o', capsize=5, capthick=2, linewidth=2)
    plt.title('Average Game Length by Training Iteration')
    plt.xlabel('Training Iteration')
    plt.ylabel('Average Number of Turns')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{config.plot_dir}/game_lengths.png")
    plt.clf()

def main():
    setup_logging(level=20, log_dir=config.log_dir, process_name='training_data_evals')
    
    logger.info("Starting training data evaluations...")

    #check_winners_match(game_data_generator())
    #check_gd_and_td_equivalence(game_data_generator(), training_data_generator())
    check_game_data_accuracy(game_data_generator())
    game_data_stats()

    logger.info("Training data evaluations completed.")

if __name__ == "__main__":
    main()
