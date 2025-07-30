import pickle
from tqdm import tqdm
from typing import Generator

import numpy as np
import matplotlib.pyplot as plt

from cc.core import Player
from cc.lookups import CCBaselineSolver
from a0.game import GameData
from a0.train.dataset import DatasetData, load_dataset_data, stats_from_dataset_data

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

def dataset_data_generator() -> Generator[DatasetData, None, None]:
    '''
    Load dataset data from the training data path.
    Returns a list of dataset data.
    '''
    for i in tqdm(range(config.training_iterations)):
        file_path = f"{config.training_dir}/iteration_stats_{i + 1}.pkl"
        with open(file_path, 'rb') as file:
            data: DatasetData = pickle.load(file)
            yield data

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

    plt.figure(figsize=(16, 9))
    plt.plot(iteration_accuracies)
    plt.title("Training Data Accuracy by Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/training_data_accuracy.png")
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
                f"{np.mean(self.game_times) if self.game_times else 0:.2f}s, "
                f"{np.std(self.game_times) if self.game_times else 0:.2f}s")
    
    @staticmethod
    def get_header() -> str:
        '''
        Get a header string for the statistics.
        '''
        return ("Total Games, Player X Wins, Player O Wins, "
                "Draws (Repeat), Draws (Timeout), "
                "Avg Game Length, Std Game Length, "
                "Avg Game Time, Std Game Time")

def game_data_list_stats(game_data_list: list[GameData]) -> GameDataStats:
    '''
    Collects stats about a list of GameData objects.
    This includes the number of games, turns, and players.
    '''
    logger.info(f"Processing {len(game_data_list)} games to get their stats.")
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
        draw_repeat = 1 if game_data.ended and game_data.winner is None else 0
        draw_timeout = 1 if not game_data.ended else 0
        stats.draws_repeat.append(draw_repeat)
        stats.draws_timeout.append(draw_timeout)
    # add the stats for this iteration to the list
    return stats

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
        stats = game_data_list_stats(game_data_list)
        # add the stats for this iteration to the list
        iteration_stats.append(stats)
    
    # save the statistics to a file
    logger.info("Saving game data statistics...")
    output_path = f"{config.eval_dir}/gamedata_stats.pkl"
    with open(output_path, 'wb') as file:
        pickle.dump(iteration_stats, file)
    logger.info(f"Game data stats saved to {output_path}.")

    # log the statistics
    logger.info("Game Data Statistics:")
    logger.info(GameDataStats.get_header())
    for i, stats in enumerate(iteration_stats):
        logger.info(f"Iteration {i}: {stats.get_line()}")
    
    # make plots for the statistics
    logger.info("Plotting game data statistics...")

    # first, plot outcomes
    # get the number of games won by each player, draws, etc.
    iterations = list(range(len(iteration_stats)))
    total_games = [stats.total_games for stats in iteration_stats]
    player_x_wins = [sum(stats.player_x_wins) for stats in iteration_stats]
    player_o_wins = [sum(stats.player_o_wins) for stats in iteration_stats]
    draws_repeat = [sum(stats.draws_repeat) for stats in iteration_stats]
    draws_timeout = [sum(stats.draws_timeout) for stats in iteration_stats]

    # plot the total number of games played in each iteration
    logger.info("Plotting total games played...")
    plt.clf()
    plt.figure(figsize=(16, 9))
    plt.plot(iterations, total_games, marker='o')
    plt.title('Total Games Played by Training Iteration')
    plt.xlabel('Training Iteration')
    plt.ylabel('Total Games')
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/gamedata_total_games.png")
    plt.clf()

    # plot them in a stacked plot
    logger.info("Plotting game outcomes (stacked)...")
    plt.figure(figsize=(16, 9))
    plt.stackplot(iterations, player_x_wins, player_o_wins, draws_repeat, draws_timeout,
        labels=['Player X Wins', 'Player O Wins', 'Draws (Repeat)', 'Draws (Timeout)'],
    )
    plt.title('Game Outcomes by Training Iteration')
    plt.xlabel('Training Iteration')
    plt.ylabel('Number of Games')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/gamedata_outcomes_stacked.png")
    plt.clf()

    # plot them in a stacked proportional plot
    logger.info("Plotting game outcomes (stacked, proportional)...")
    plt.figure(figsize=(16, 9))
    plt.stackplot(
        np.array(iterations),
        np.array(player_x_wins) / np.array(total_games),
        np.array(player_o_wins) / np.array(total_games),
        np.array(draws_repeat) / np.array(total_games),
        np.array(draws_timeout) / np.array(total_games),
        labels=['Player X Wins', 'Player O Wins', 'Draws (Repeat)', 'Draws (Timeout)'],
    )
    plt.title('Game Outcomes by Training Iteration')
    plt.xlabel('Training Iteration')
    plt.ylabel('Proportion of Games')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/gamedata_outcomes_stacked_proportional.png")
    plt.clf()

    # plot them in a line plot
    logger.info("Plotting game outcomes (line)...")
    plt.figure(figsize=(16, 9))
    plt.plot(iterations, player_x_wins, label='Player X Wins')
    plt.plot(iterations, player_o_wins, label='Player O Wins')
    plt.plot(iterations, draws_repeat, label='Draws (Repeat)')
    plt.plot(iterations, draws_timeout, label='Draws (Timeout)')
    
    plt.title('Game Outcomes by Training Iteration')
    plt.xlabel('Training Iteration')
    plt.ylabel('Number of Games')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/gamedata_outcomes_lines.png")
    plt.clf()

    # then, plot game lengths (both turns and time)
    logger.info("Plotting game lengths...")
    # turns
    avg_game_lengths = [float(np.mean(stats.game_lengths)) for stats in iteration_stats]
    std_game_lengths = [float(np.std(stats.game_lengths)) for stats in iteration_stats]
    plot_game_lengths(iterations, avg_game_lengths, std_game_lengths, "Turns")
    # time
    avg_game_times = [float(np.mean(stats.game_times)) for stats in iteration_stats]
    std_game_times = [float(np.std(stats.game_times)) for stats in iteration_stats]
    plot_game_lengths(iterations, avg_game_times, std_game_times, "Time (seconds)")

def plot_game_lengths(iterations: list[int], avg_lengths: list[float], std_length: list[float], l_type: str):
    plt.figure(figsize=(16, 9))
    plt.errorbar(iterations, avg_lengths, yerr=std_length, 
                 marker='o', capsize=5, capthick=1, linewidth=1)
    plt.title(f'Average Game Length ({l_type}) by Training Iteration')
    plt.xlabel('Training Iteration')
    plt.ylabel(f'Length ({l_type})')
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/game_lengths_{l_type.lower()}.png")
    plt.clf()

def plot_training_performance_metrics():
    '''
    Plot the training performance metrics from the training data.
    This includes the losses, accuracies, and other metrics.
    '''
    logger.info("Plotting training performance metrics...")
    
    # load the dataset data from the training data path
    dataset_data_gen = dataset_data_generator()
    
    # collect the losses and accuracies
    losses: list[float] = []
    value_losses: list[float] = []
    policy_losses: list[float] = []
    value_accuracies: list[float] = []
    policy_accuracies: list[float] = []

    for dataset_data in dataset_data_gen:
        loss, value_loss, policy_loss, value_accuracy, policy_accuracy = stats_from_dataset_data(dataset_data)
        losses.append(loss)
        value_losses.append(value_loss)
        policy_losses.append(policy_loss)
        value_accuracies.append(value_accuracy)
        policy_accuracies.append(policy_accuracy)

    # plot the performance metrics
    plt.figure(figsize=(16, 9))
    plt.plot(losses, label='Loss')
    plt.plot(value_losses, label='Value Loss')
    plt.plot(policy_losses, label='Policy Loss')
    plt.plot(value_accuracies, label='Value Accuracy')
    plt.plot(policy_accuracies, label='Policy Accuracy')
    plt.xlabel("Iteration")
    plt.ylabel("Performance")
    plt.title("Training Performance Metrics")
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/training_metrics.png")
    plt.clf()

def main():
    setup_logging(level=20, log_dir=config.log_dir, process_name='training_data_evals')
    
    logger.info("Starting training data evaluations...")

    check_game_data_accuracy(game_data_generator())
    game_data_stats()
    plot_training_performance_metrics()

    logger.info("Training data evaluations completed.")

if __name__ == "__main__":
    main()
