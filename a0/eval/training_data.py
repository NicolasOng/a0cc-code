import pickle
from tqdm import tqdm
from typing import Generator

import numpy as np
import matplotlib.pyplot as plt

from cc.core import Player
from cc.ground_truth import GroundTruth
from a0.game import GameData
from a0.train.dataset import DatasetData, stats_from_dataset_data
from a0.eval.dataset_evaluation import Series, save_series

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def game_data_generator(dir: str, n: int) -> Generator[tuple[int, list[GameData]], None, None]:
    '''
    Load game data from the given path, from 1 to n inclusive
    Returns a list of lists of game data, assuming each is named gamedata_<iteration>.pkl
    Each list corresponds to a single training iteration.
    If a file doesn't exist, it skips it.
    '''
    for i in range(n):
        file_path = f"{dir}/gamedata_{i + 1}.pkl"
        try:
            with open(file_path, 'rb') as file:
                data: list[GameData] = pickle.load(file)
                yield i + 1, data
        except Exception as e:
            logger.error(f"Failed to load game data {i} at {file_path}: {e}")

def dataset_data_generator(dir: str, n: int) -> Generator[tuple[int, DatasetData], None, None]:
    '''
    Load dataset data from the given path.
    Each dataset data object contains the training data for each iteration.
    Attempts to load each iteration's dataset data (iteration_stats_{i + 1}.pkl) from 1 to n inclusive
    Returns a list of dataset data.
    '''
    for i in range(n):
        file_path = f"{dir}/iteration_stats_{i + 1}.pkl"
        try:
            with open(file_path, 'rb') as file:
                data: DatasetData = pickle.load(file)
                yield i + 1, data
        except Exception as e:
            logger.error(f"Failed to load dataset data {i} at {file_path}: {e}")

def check_game_data_accuracy(game_data_lists: list[tuple[int, list[GameData]]]) -> None:
    '''
    Checks the accuracy of game data by comparing the outcomes from the solver
    with the ground truth outcomes.
    '''
    last_n_iterations = config.replay_buffer_size // config.training_samples
    gd_accuracy_series = Series(["Iteration Value Accuracy", "EB Value Accuracy"])
    gt = GroundTruth()
    iteration_accuracies: list[float] = []
    # for each game data list/iteration,
    for i, game_data_list in game_data_lists:
        iteration_num_correct = 0
        iteration_total = 0
        # go through each game,
        for _, game_data in enumerate(game_data_list):
            winner = game_data.winner
            # go through each turn
            for _, game_state in enumerate(game_data.turn_data):
                # and check if the experienced outcome matches the ground truth
                # TODO: policy?
                board = game_state.board
                sd_outcome = gt.get_outcome(board)
                gd_outcome = 0 if winner is None else 1 if winner == board.current_player else -1
                if sd_outcome == gd_outcome:
                    iteration_num_correct += 1
                iteration_total += 1
        iteration_accuracy = iteration_num_correct / iteration_total if iteration_total > 0 else 0
        iteration_accuracies.append(iteration_accuracy)
        # log the iteration's accuracy
        logger.info(f"Iteration {i} accuracy: {iteration_accuracy:.2%} ({iteration_num_correct}/{iteration_total})")
        # add this info to a series object
        gd_accuracy_series.x.append(i)
        gd_accuracy_series.ys["Iteration Value Accuracy"].append(iteration_accuracy)
        # also calculate the estimated experience buffer accuracy
        # basically the average accuracy over the last n iterations
        total_eb_acc = 0
        total_eb_its = 0
        for j in range(len(gd_accuracy_series.x) - 1, 0 - 1, -1):
            cur_iteration = gd_accuracy_series.x[j]
            cur_acc = gd_accuracy_series.ys["Iteration Value Accuracy"][j]
            if cur_iteration < i - last_n_iterations:
                break
            total_eb_acc += cur_acc
            total_eb_its += 1
        eb_acc = total_eb_acc / total_eb_its if total_eb_its > 0 else 0
        gd_accuracy_series.ys["EB Value Accuracy"].append(eb_acc)
    
    save_series(gd_accuracy_series, f"{config.eval_dir}/gamedata_acc.pkl")

class GameDataStats:
    '''
    Class to hold statistics about game data for a single self-play iteration.
    '''
    def __init__(self, iteration: int):
        self.iteration = iteration
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

def game_data_list_stats(iteration: int, game_data_list: list[GameData]) -> GameDataStats:
    '''
    Collects stats about a list of GameData objects.
    This includes the number of games, turns, and players.
    '''
    logger.info(f"Processing {len(game_data_list)} games to get their stats.")
    stats = GameDataStats(iteration)
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

def get_stats_of_each_iterations_game_data() -> None:
    '''
    Print statistics about the game data.
    This includes the number of games, turns, and players.
    '''
    logger.info("Calculating game data statistics...")
    iteration_stats: list[GameDataStats] = []
    # for each iteration,
    gd_gen = game_data_generator(config.training_dir, config.training_iterations)
    for i, game_data_list in gd_gen:
        # get the stats for the game data generated in that iteration
        # wins/losses/draws, types of draws, num turns (mean, etc), avg length in time, who won
        stats = game_data_list_stats(i, game_data_list)
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
    for stats in iteration_stats:
        logger.info(f"Iteration {stats.iteration}: {stats.get_line()}")
    
    # creating series for the game data stats
    logger.info("Creating series for game data statistics...")

    gamedata_series = Series(["Total Games", "Player X Wins", "Player O Wins", "Draws (Repeat)", "Draws (Timeout)", "Avg Game Length", "Avg Game Time", "Std Game Length", "Std Game Time"])
    gamedata_series.x = [stats.iteration for stats in iteration_stats]
    gamedata_series.ys["Total Games"] = [stats.total_games for stats in iteration_stats]
    gamedata_series.ys["Player X Wins"] = [sum(stats.player_x_wins) for stats in iteration_stats]
    gamedata_series.ys["Player O Wins"] = [sum(stats.player_o_wins) for stats in iteration_stats]
    gamedata_series.ys["Draws (Repeat)"] = [sum(stats.draws_repeat) for stats in iteration_stats]
    gamedata_series.ys["Draws (Timeout)"] = [sum(stats.draws_timeout) for stats in iteration_stats]
    gamedata_series.ys["Avg Game Length"] = [float(np.mean(stats.game_lengths)) if stats.game_lengths else 0 for stats in iteration_stats]
    gamedata_series.ys["Avg Game Time"] = [float(np.mean(stats.game_times)) if stats.game_times else 0 for stats in iteration_stats]
    gamedata_series.ys["Std Game Length"] = [float(np.std(stats.game_lengths)) if stats.game_lengths else 0 for stats in iteration_stats]
    gamedata_series.ys["Std Game Time"] = [float(np.std(stats.game_times)) if stats.game_times else 0 for stats in iteration_stats]

    # save the series
    save_series(gamedata_series, f"{config.eval_dir}/gamedata_stats.pkl")

def get_training_performance_metrics():
    '''
    Plot the training performance metrics from the training data.
    This includes the losses, accuracies, and other metrics.
    '''
    logger.info("Plotting training performance metrics...")
    
    # load the dataset data from the training data path
    dataset_data_gen = dataset_data_generator(config.training_dir, config.training_iterations)
    
    # collect the losses and accuracies
    training_metrics = Series(["Loss", "Value Loss", "Policy Loss", "Value Accuracy", "Policy Accuracy"])

    for i, dataset_data in dataset_data_gen:
        loss, value_loss, policy_loss, value_accuracy, policy_accuracy = stats_from_dataset_data(dataset_data)
        training_metrics.x.append(i)
        training_metrics.ys["Loss"].append(loss)
        training_metrics.ys["Value Loss"].append(value_loss)
        training_metrics.ys["Policy Loss"].append(policy_loss)
        training_metrics.ys["Value Accuracy"].append(value_accuracy)
        training_metrics.ys["Policy Accuracy"].append(policy_accuracy)
    
    # save the series
    save_series(training_metrics, f"{config.eval_dir}/training_metrics.pkl")

def main():
    setup_logging(level=20, log_dir=config.log_dir, process_name='training_data_evals')
    
    logger.info("Starting training data evaluations...")

    check_game_data_accuracy(list(game_data_generator(config.training_dir, config.training_iterations)))
    get_stats_of_each_iterations_game_data()
    get_training_performance_metrics()

    logger.info("Training data evaluations completed.")

if __name__ == "__main__":
    main()
