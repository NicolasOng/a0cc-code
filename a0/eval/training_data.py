import pickle
from tqdm import tqdm
from typing import Generator
from collections import defaultdict
import random

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from cc.core import Player, Board
from cc.ground_truth import GroundTruth
from a0.game import GameData
from a0.train.dataset import DatasetData, stats_from_dataset_data
from a0.eval.dataset_evaluation import Series, save_series, policy_accuracy_function, policy_probability_mass_function

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
    #last_n_iterations = config.replay_buffer_size // config.training_samples
    gd_accuracy_series = Series(["Iteration Value Accuracy", "Iteration Value Accuracy ND", "Iteration Policy Accuracy", "Iteration Policy PM", "Iteration Policy Accuracy NT", "Iteration Policy PM NT"])
    gd_overall_accuracy_series = Series(["Overall Value Accuracy", "Overall Value Accuracy ND", "Overall Policy Accuracy", "Overall Policy PM", "Overall Policy Accuracy NT", "Overall Policy PM NT"])
    gt = GroundTruth()
    total_num_correct_value = 0
    total_num_correct_value_nd = 0
    total_num_correct_policy = 0
    total_pm_policy = 0
    total_num_correct_policy_nt = 0
    total_pm_policy_nt = 0
    total = 0
    total_nt = 0
    total_nd = 0
    # for each game data list/iteration,
    for i, game_data_list in game_data_lists:
        iteration_num_correct_value = 0
        iteration_num_correct_value_nd = 0
        iteration_num_correct_policy = 0
        iteration_pm_policy = 0
        iteration_num_correct_policy_nt = 0
        iteration_pm_policy_nt = 0
        iteration_total = 0
        iteration_total_nt = 0
        iteration_total_nd = 0
        # go through each game,
        for _, game_data in enumerate(game_data_list):
            winner = game_data.winner
            # go through each turn
            for _, game_state in enumerate(game_data.turn_data):
                board = game_state.board
                is_trivial = gt.is_trivial(board)
                # check the value accuracy
                sd_outcome = gt.get_outcome(board)
                gd_outcome = 0 if winner is None else 1 if winner == board.current_player else -1
                accurate_outcome = sd_outcome == gd_outcome
                if accurate_outcome:
                    iteration_num_correct_value += 1
                    total_num_correct_value += 1
                # account for non-draws
                if not gd_outcome == 0:
                    iteration_total_nd += 1
                    total_nd += 1
                    if accurate_outcome:
                        iteration_num_correct_value_nd += 1
                        total_num_correct_value_nd += 1
                # check the policy accuracy
                sd_policy = np.array(gt.get_1ply_policy_prob_dist_list(board, for_model=True))
                gd_policy: NDArray[np.float32] = game_state.player_data
                pm_policy = policy_probability_mass_function(gd_policy, sd_policy)
                policy_is_acc = policy_accuracy_function(gd_policy, sd_policy)
                iteration_pm_policy += pm_policy
                total_pm_policy += pm_policy
                if policy_is_acc:
                    iteration_num_correct_policy += 1
                    total_num_correct_policy += 1
                # account for non-trivial boards
                if not is_trivial:
                    iteration_total_nt += 1
                    total_nt += 1
                    iteration_pm_policy_nt += pm_policy
                    total_pm_policy_nt += pm_policy
                    if policy_is_acc:
                        iteration_num_correct_policy_nt += 1
                        total_num_correct_policy_nt += 1
                iteration_total += 1
                total += 1
        iteration_value_accuracy = iteration_num_correct_value / iteration_total if iteration_total > 0 else 0
        iteration_value_accuracy_nd = iteration_num_correct_value_nd / iteration_total_nd if iteration_total_nd > 0 else 0
        iteration_policy_accuracy = iteration_num_correct_policy / iteration_total if iteration_total > 0 else 0
        iteration_policy_pm = iteration_pm_policy / iteration_total if iteration_total > 0 else 0
        iteration_policy_accuracy_nt = iteration_num_correct_policy_nt / iteration_total_nt if iteration_total_nt > 0 else 0
        iteration_policy_pm_nt = iteration_pm_policy_nt / iteration_total_nt if iteration_total_nt > 0 else 0
        # log the iteration's accuracy
        logger.info(f"Iteration {i} value accuracy: {iteration_value_accuracy:.2%} ({iteration_num_correct_value}/{iteration_total})")
        logger.info(f"Iteration {i} value accuracy (non-draws only): {iteration_value_accuracy_nd:.2%} ({iteration_num_correct_value_nd}/{iteration_total_nd})")
        logger.info(f"Iteration {i} policy accuracy: {iteration_policy_accuracy:.2%} ({iteration_num_correct_policy}/{iteration_total})")
        logger.info(f"Iteration {i} policy PM: {iteration_policy_pm:.2%}")
        logger.info(f"Iteration {i} policy accuracy (non-trivial only): {iteration_policy_accuracy_nt:.2%} ({iteration_num_correct_policy_nt}/{iteration_total_nt})")
        logger.info(f"Iteration {i} policy PM (non-trivial only): {iteration_policy_pm_nt:.2%}")
        # add this info to the series object
        gd_accuracy_series.x.append(i)
        gd_accuracy_series.ys["Iteration Value Accuracy"].append(iteration_value_accuracy)
        gd_accuracy_series.ys["Iteration Value Accuracy ND"].append(iteration_value_accuracy_nd)
        gd_accuracy_series.ys["Iteration Policy Accuracy"].append(iteration_policy_accuracy)
        gd_accuracy_series.ys["Iteration Policy PM"].append(iteration_policy_pm)
        gd_accuracy_series.ys["Iteration Policy Accuracy NT"].append(iteration_policy_accuracy_nt)
        gd_accuracy_series.ys["Iteration Policy PM NT"].append(iteration_policy_pm_nt)

    # save all the accuracy data to a file
    save_series(gd_accuracy_series, f"{config.eval_dir}/gamedata_acc.pkl")

    # Calculate, log, and save the overall accuracy
    total_value_accuracy = total_num_correct_value / total if total > 0 else 0
    total_value_accuracy_nd = total_num_correct_value_nd / total_nd if total_nd > 0 else 0
    total_policy_accuracy = total_num_correct_policy / total if total > 0 else 0
    total_policy_pm = total_pm_policy / total if total > 0 else 0
    total_policy_accuracy_nt = total_num_correct_policy_nt / total_nt if total_nt > 0 else 0
    total_policy_pm_nt = total_pm_policy_nt / total_nt if total_nt > 0 else 0
    logger.info(f"Overall Value Accuracy: {total_value_accuracy:.2%} ({total_num_correct_value}/{total})")
    logger.info(f"Overall Value Accuracy (Non-Draws Only): {total_value_accuracy_nd:.2%} ({total_num_correct_value_nd}/{total_nd})")
    logger.info(f"Overall Policy Accuracy: {total_policy_accuracy:.2%} ({total_num_correct_policy}/{total})")
    logger.info(f"Overall Policy PM: {total_policy_pm:.2%}")
    logger.info(f"Overall Policy Accuracy (Non-Trivial Only): {total_policy_accuracy_nt:.2%} ({total_num_correct_policy_nt}/{total_nt})")
    logger.info(f"Overall Policy PM (Non-Trivial Only): {total_policy_pm_nt:.2%}")
    # save the overall accuracy at every iteration for plotting
    for iter_num in range(len(gd_accuracy_series.x)):
        gd_overall_accuracy_series.x.append(gd_accuracy_series.x[iter_num])
        gd_overall_accuracy_series.ys["Overall Value Accuracy"].append(total_value_accuracy)
        gd_overall_accuracy_series.ys["Overall Value Accuracy ND"].append(total_value_accuracy_nd)
        gd_overall_accuracy_series.ys["Overall Policy Accuracy"].append(total_policy_accuracy)
        gd_overall_accuracy_series.ys["Overall Policy PM"].append(total_policy_pm)
        gd_overall_accuracy_series.ys["Overall Policy Accuracy NT"].append(total_policy_accuracy_nt)
        gd_overall_accuracy_series.ys["Overall Policy PM NT"].append(total_policy_pm_nt)
    save_series(gd_overall_accuracy_series, f"{config.eval_dir}/gamedata_overall_acc.pkl")

def check_game_data_bias(game_data_lists: list[tuple[int, list[GameData]]]) -> None:
    '''
    Checks the bias of game data by seeing how many wins/losses/draws there are.
    '''
    gd_bias_series = Series(["Iteration Win Percent", "Iteration Loss Percent", "Iteration Draw Percent"])
    gd_overall_bias_series = Series(["Overall Win Percent", "Overall Loss Percent", "Overall Draw Percent"])
    total_wins = 0
    total_losses = 0
    total_draws = 0
    total = 0
    # for each game data list/iteration,
    for i, game_data_list in game_data_lists:
        iteration_num_wins = 0
        iteration_num_losses = 0
        iteration_num_draws = 0
        iteration_total = 0
        # go through each game,
        for _, game_data in enumerate(game_data_list):
            winner = game_data.winner
            # go through each turn
            for _, game_state in enumerate(game_data.turn_data):
                # and check what the experienced outcome is
                board = game_state.board
                # get the outcome of the game for the current player
                current_player = board.current_player
                if winner is None:
                    # draws and unfinished games
                    value = 0.0
                else:
                    # 1 for win, -1 for loss
                    # based on the perspective of the current player
                    value = 1.0 if winner == current_player else -1.0
                # count wins/losses
                if value == 1.0:
                    iteration_num_wins += 1
                    total_wins += 1
                elif value == -1.0:
                    iteration_num_losses += 1
                    total_losses += 1
                else:
                    iteration_num_draws += 1
                    total_draws += 1
                iteration_total += 1
                total += 1
        iteration_win_percent = iteration_num_wins / iteration_total if iteration_total > 0 else 0
        iteration_loss_percent = iteration_num_losses / iteration_total if iteration_total > 0 else 0
        iteration_draw_percent = iteration_num_draws / iteration_total if iteration_total > 0 else 0
        # log the iteration's bias
        logger.info(f"Iteration {i} win percent: {iteration_win_percent:.2%} ({iteration_num_wins}/{iteration_total})")
        logger.info(f"Iteration {i} loss percent: {iteration_loss_percent:.2%} ({iteration_num_losses}/{iteration_total})")
        logger.info(f"Iteration {i} draw percent: {iteration_draw_percent:.2%} ({iteration_num_draws}/{iteration_total})")
        # add this info to the series object
        gd_bias_series.x.append(i)
        gd_bias_series.ys["Iteration Win Percent"].append(iteration_win_percent)
        gd_bias_series.ys["Iteration Loss Percent"].append(iteration_loss_percent)
        gd_bias_series.ys["Iteration Draw Percent"].append(iteration_draw_percent)

    # save all the bias data to a file
    save_series(gd_bias_series, f"{config.eval_dir}/gamedata_bias.pkl")

    # Calculate, log, and save the overall bias
    total_win_percent = total_wins / total if total > 0 else 0
    total_loss_percent = total_losses / total if total > 0 else 0
    total_draw_percent = total_draws / total if total > 0 else 0
    logger.info(f"Overall Win Percent: {total_win_percent:.2%} ({total_wins}/{total})")
    logger.info(f"Overall Loss Percent: {total_loss_percent:.2%} ({total_losses}/{total})")
    logger.info(f"Overall Draw Percent: {total_draw_percent:.2%} ({total_draws}/{total})")
    # saving the first and last iteration for easy plotting
    gd_overall_bias_series.x.append(gd_bias_series.x[0])
    gd_overall_bias_series.ys["Overall Win Percent"].append(total_win_percent)
    gd_overall_bias_series.ys["Overall Loss Percent"].append(total_loss_percent)
    gd_overall_bias_series.ys["Overall Draw Percent"].append(total_draw_percent)
    gd_overall_bias_series.x.append(gd_bias_series.x[-1])
    gd_overall_bias_series.ys["Overall Win Percent"].append(total_win_percent)
    gd_overall_bias_series.ys["Overall Loss Percent"].append(total_loss_percent)
    gd_overall_bias_series.ys["Overall Draw Percent"].append(total_draw_percent)
    save_series(gd_overall_bias_series, f"{config.eval_dir}/gamedata_overall_bias.pkl")

def get_all_games_generated_during_training() -> list[GameData]:
    '''
    Retrieves all game data generated during training.
    '''
    logger.info("Retrieving all games generated during training...")
    all_games: list[GameData] = []
    gd_gen = game_data_generator(config.training_dir, config.training_iterations)
    for _, game_data_list in gd_gen:
        all_games.extend(game_data_list)
    logger.info(f"Retrieved {len(all_games)} games.")
    return all_games

def get_state_accuracy_dict(game_data_list: list[GameData]) -> dict[Board, list[bool]]:
    '''
    Given a list of GameData, returns a dict in the form of:
    dict[Board] = list[bool],
    where the list represents the accuracy of the training data for that state,
    each time it is seen.
    '''
    gt = GroundTruth()
    state_accuracy: dict[Board, list[bool]] = dict()
    total_states_seen = 0
    for _, game_data in enumerate(game_data_list):
        winner = game_data.winner
        # go through each turn
        for _, game_state in enumerate(game_data.turn_data):
            total_states_seen += 1
            board = game_state.board
            if board not in state_accuracy:
                state_accuracy[board] = []
            # check if the experienced outcome matches the ground truth
            sd_outcome = gt.get_outcome(board)
            gd_outcome = 0 if winner is None else 1 if winner == board.current_player else -1
            state_accuracy[board].append(sd_outcome == gd_outcome)
    
    # now we can run analysis on this info.
    percent_unique = len(state_accuracy) / total_states_seen if total_states_seen > 0 else 0
    logger.info(f"Found {len(state_accuracy)} unique game states in the given game data list.")
    logger.info(f"Percentage of unique game states: {percent_unique:.2%} ({len(state_accuracy)}/{total_states_seen})")

    return state_accuracy

def get_state_accuracy_dicts(game_data_lists: list[tuple[int, list[GameData]]]) -> list[dict[Board, list[bool]]]:
    '''
    Given a list of lists of GameData objects,
    generates a state accuracy dict for each list.
    '''
    return [get_state_accuracy_dict(game_data_list) for _, game_data_list in game_data_lists]

def split_by_visited_seen_bins(state_accuracy: dict[Board, list[bool]], thresholds: list[int]) -> list[dict[Board, list[bool]]]:
    '''
    Splits the state accuracy data into bins based on thresholds for visits.
    '''
    # create a list to hold the binned data
    binned_data: list[dict[Board, list[bool]]] = []
    threshold_2_idx = {threshold: i for i, threshold in enumerate(thresholds)}
    for threshold in thresholds:
        binned_data.append(dict())
    # for each state in the state_acc dict,
    for state in state_accuracy:
        # get how many times it was seen
        accuracies = state_accuracy[state]
        times_seen = len(accuracies)

        # determine which bin to put it into
        bin = None
        for threshold in thresholds:
            if times_seen <= threshold:
                bin = threshold
                break
        if bin is None:
            assert False, f"Unbinned state with {times_seen} seen"

        # add it to the bin
        binned_data[threshold_2_idx[bin]][state] = accuracies

    return binned_data

def duplicate_states_analysis(state_acc_dicts: list[dict[Board, list[bool]]], thresholds: list[int] | None = None) -> None:
    # create a dict of all boards and their accuracies.
    logger.info("Analyzing duplicate game states...")

    # this block of code graphs how many states were seen a certain number of times.
    # and graphs the results
    # for example, how many states were seen once, twice, etc.
    #graph_state_seen_counts = len(state_acc_dicts) == 1
    graph_state_seen_counts = False
    if graph_state_seen_counts:
        # state_seen_count: dict[how many times state was seen] = number of states seen that many times
        state_seen_count: defaultdict[int, int] = defaultdict(int)
        # for each unique state,
        for accuracies in state_acc_dicts[0].values():
            # count how many times it was seen,
            # and add 1 to the number of states seen that many times.
            state_seen_count[len(accuracies)] += 1

        # log the state seen count
        i = 0
        for seen_count, num_states in sorted(state_seen_count.items()):
            logger.info(f"States seen {seen_count} times: {num_states}")
            i += 1
            if i >= 10: break

        seen_counts = sorted(state_seen_count.keys())
        num_states_counts = [state_seen_count[count] for count in seen_counts]
        plt.figure(figsize=(12, 6))
        #plt.yscale('log')
        plt.bar(seen_counts, num_states_counts, width=1.0)
        plt.xlabel('Number of Times State Was Seen')
        plt.ylabel('Number of States')
        plt.title('Distribution of State Repetition Frequency')
        plt.grid(axis='y', alpha=0.3)
        plt.show()
        plt.clf()

    # this block of code counts how many times each unique state was seen,
    # then graphs the results in descending order.
    #graph_unique_seen_counts_descending = len(state_acc_dicts) == 1
    graph_unique_seen_counts_descending = False
    if graph_unique_seen_counts_descending:
        # counts list: number of times each unique state was seen, descending order
        counts_list = sorted([len(accuracies) for accuracies in state_acc_dicts[0].values()], reverse=True)
        i = 0
        for count in counts_list:
            logger.info(f"State seen {count} times.")
            i += 1
            if i >= 10: break
        
        plt.figure(figsize=(12, 6))
        plt.yscale('log')
        plt.bar(list(range(len(counts_list))), counts_list, width=1.0)
        plt.ylabel('Number of Times State Was Seen (LOG SCALE)')
        plt.xlabel('States from Most to Least Seen')
        plt.title('Distribution of State Repetition Frequency')
        plt.grid(axis='y', alpha=0.3)
        plt.show()
        plt.clf()
    
    graph_binned_counts_vars_accs = True
    if graph_binned_counts_vars_accs:
        # bin counts, accuracies, and variance.
        acc_var_dict: dict[int, tuple[int, int, float, float, float]] = defaultdict(lambda: (0, 0, 0.0, 0.0, 0.0))

        for key, state_accuracy in enumerate(state_acc_dicts):
            for state in state_accuracy:
                accuracies = state_accuracy[state]
                times_seen = len(accuracies)

                #accuracies = [1 if random.random() < 0.5 else 0 for _ in range(times_seen)]

                acc = float(np.mean(accuracies))
                variance = 4 * (acc * (1 - acc))
                consensus_acc = 0.5 if acc == 0.5 else 1 if acc > 0.5 else 0
                new_tuple = (1, times_seen, acc * times_seen, variance, consensus_acc)

                old_tuple = acc_var_dict[key]
                acc_var_dict[key] = (
                    new_tuple[0] + old_tuple[0],
                    new_tuple[1] + old_tuple[1],
                    new_tuple[2] + old_tuple[2],
                    new_tuple[3] + old_tuple[3],
                    new_tuple[4] + old_tuple[4]
                )

        # average the accuracies and variances
        # overall_acc = 0
        # overall_var = 0
        # overall_consensus = 0
        # total_unique_count = 0
        # total_total_count = 0
        for key in acc_var_dict:
            unique_count, total_count, total_acc, total_var, total_consensus = acc_var_dict[key]
            # overall_acc += total_acc
            # overall_var += total_var
            # overall_consensus += total_consensus
            # total_unique_count += unique_count
            # total_total_count += total_count
            acc_var_dict[key] = (unique_count, total_count, total_acc / total_count if total_count > 0 else 0, total_var / unique_count if unique_count > 0 else 0, total_consensus / unique_count if unique_count > 0 else 0)
        # average the accuracies and variances
        # overall_acc = overall_acc / total_total_count if total_total_count > 0 else 0
        # overall_var = overall_var / total_unique_count if total_unique_count > 0 else 0
        # overall_consensus = overall_consensus / total_unique_count if total_unique_count > 0 else 0
        # overall_unique = total_unique_count / total_total_count if total_total_count > 0 else 0

        # graph this
        x_labels = []
        y_means = []
        y_vars = []
        y_tcounts = []
        y_ucounts = []
        y_consensus = []
        prev_key = 0
        for key in sorted(acc_var_dict.keys()):
            unique_count, total_count, mean_acc, mean_var, mean_consensus = acc_var_dict[key]
            xl = f"{key}"
            if thresholds is not None:
                cur_key = thresholds[key]
                xl = f"{prev_key + 1}-{cur_key}"
            x_labels.append(xl)
            y_means.append(mean_acc)
            y_vars.append(mean_var)
            y_ucounts.append(unique_count)
            y_tcounts.append(total_count - unique_count)
            y_consensus.append(mean_consensus)
            prev_key = key
        
        x_label = 'Iteration'
        by = "Iteration"
        if thresholds is not None:
            x_label = 'Number of Times State Was Seen Bins' 
            by = "State Seen Count" 

        # plt.figure(figsize=(12, 6))
        # plt.bar(x_labels, y_ucounts, label='Unique States')
        # plt.bar(x_labels, y_tcounts, bottom=y_ucounts, label='Duplicate States')
        # plt.xlabel(x_label)
        # plt.ylabel('Number of Unique States')
        # plt.title(f'Number of Unique States by {by}')
        # plt.legend()
        # plt.grid(axis='y', alpha=0.3)
        # plt.show()
        logger.info(f"Unique States: {x_labels}, {y_ucounts}, {y_tcounts}")

        # plt.figure(figsize=(12, 6))
        # plt.bar(x_labels, y_means, alpha=0.5, label='Mean Accuracy')
        # plt.xlabel(x_label)
        # plt.ylabel('Accuracy')
        # plt.title(f'Mean Accuracy by {by}')
        # plt.legend()
        # plt.grid(axis='y', alpha=0.3)
        # plt.ylim(0, 1)  # Set y-axis from 0 to 1
        # plt.show()
        # plt.clf()
        logger.info(f"Accuracy: {x_labels}, {y_means}")

        # plt.figure(figsize=(12, 6))
        # plt.bar(x_labels, y_vars, alpha=0.5, label='Mean Variance')
        # plt.xlabel(x_label)
        # plt.ylabel('Variance')
        # plt.title(f'Mean Variance by {by}')
        # plt.legend()
        # plt.grid(axis='y', alpha=0.3)
        # plt.ylim(0, 1)  # Set y-axis from 0 to 1
        # plt.show()
        # plt.clf()
        logger.info(f"Variance: {x_labels}, {y_vars}")

        # plt.figure(figsize=(12, 6))
        # plt.bar(x_labels, y_consensus, alpha=0.5, label='Mean Consensus')
        # plt.xlabel(x_label)
        # plt.ylabel('Consensus')
        # plt.title(f'Mean Consensus by {by}')
        # plt.legend()
        # plt.grid(axis='y', alpha=0.3)
        # plt.ylim(0, 1)  # Set y-axis from 0 to 1
        # plt.show()
        logger.info(f"Consensus: {x_labels}, {y_consensus}")
    
    acc_states_over_time = False
    if acc_states_over_time:
        # board: list[tuple(iteration, num seen, cum num seen, acc, cum acc)]
        board_stats_dict: dict[Board, list[tuple[int, int, int, float, float]]] = defaultdict(list)
        # generate the dict
        for iteration, state_accuracy in enumerate(state_acc_dicts):
            for state in state_accuracy:
                accuracies = state_accuracy[state]

                times_seen = len(accuracies)
                cum_times_seen = sum([t[1] for t in board_stats_dict[state]]) + times_seen

                acc = float(np.mean(accuracies))
                cum_acc = (sum([t[1] * t[3] for t in board_stats_dict[state]]) + (times_seen * acc)) / cum_times_seen

                board_stats_dict[state].append((iteration, times_seen, cum_times_seen, acc, cum_acc))

        # make the iterations all start at 0
        for state in board_stats_dict.keys():
            start_idx = board_stats_dict[state][0][0]
            board_stats_dict[state] = [(stats[0] - start_idx, stats[1], stats[2], stats[3], stats[4]) for stats in board_stats_dict[state]]
        
        # get an average for each measure at all iterations.
        average_stats: dict[str, list[float]] = defaultdict(list)
        its: list[int] = []
        for iteration, _ in enumerate(state_acc_dicts):
            total_seen = 0
            total_cum_seen = 0
            total_acc = 0
            total_cum_acc = 0
            total = 0
            for _, stats in board_stats_dict.items():
                for t in stats:
                    if t[0] == iteration:
                        total_seen += t[1]
                        total_cum_seen += t[2]
                        total_acc += t[3]
                        total_cum_acc += t[4]
                        total += 1
            average_stats["Times Seen"].append(total_seen / total)
            average_stats["Cumulative Times Seen"].append(total_cum_seen / total)
            average_stats["Accuracy"].append(total_acc / total)
            average_stats["Cumulative Accuracy"].append(total_cum_acc / total)
            its.append(iteration)

        # plot all the graphs
        # plot_state_stats(board_stats_dict, 'Times Seen', its, average_stats["Times Seen"])
        # plot_state_stats(board_stats_dict, 'Cumulative Times Seen', its, average_stats["Cumulative Times Seen"])
        # plot_state_stats(board_stats_dict, 'Accuracy', its, average_stats["Accuracy"])
        # plot_state_stats(board_stats_dict, 'Cumulative Accuracy', its, average_stats["Cumulative Accuracy"])

def plot_state_stats(board_stats_dict: dict[Board, list[tuple[int, int, int, float, float]]], mode: str, its: list[int], avg: list[float]):
    plt.figure(figsize=(12, 6))
    #plt.yscale('log')
    for _, stats in board_stats_dict.items():
        iterations = [s[0] for s in stats]
        times_seen = [s[1] for s in stats]
        cum_times_seen = [s[2] for s in stats]
        acc = [s[3] for s in stats]
        cum_acc = [s[4] for s in stats]
        if mode == 'Times Seen':
            plt.plot(iterations, times_seen, label='Times Seen', color='black', alpha=0.1)
        elif mode == 'Cumulative Times Seen':
            plt.plot(iterations, cum_times_seen, label='Cumulative Times Seen', color='black', alpha=0.1)
        elif mode == 'Accuracy':
            plt.plot(iterations, acc, label='Accuracy', color='black', alpha=0.1)
        elif mode == 'Cumulative Accuracy':
            plt.plot(iterations, cum_acc, label='Cumulative Accuracy', color='black', alpha=0.1)
    plt.plot(its, avg, label='Average', color='red')
    plt.xlabel('T+ Iteration')
    plt.ylabel(mode)
    plt.title(f"State {mode} over T+ Iterations")
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.clf()


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
    logger.info("Saving training performance metrics...")
    
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

def gamedata_accuracy_over_progress(game_data_lists: list[tuple[int, list[GameData]]]) -> None:
    gt = GroundTruth()

    bins_100: dict[int, tuple[int, int]] = {100 * i // 100: (0, 0) for i in range(100)}
    bins_10: dict[int, tuple[int, int]] = {100 * i // 10: (0, 0) for i in range(10)}

    # put all the boards from the game data into the appropriate bins
    logger.info("Extracting boards from game data into progress accuracy bins...")
    for _, game_data_list in tqdm(game_data_lists):
        for game_data in game_data_list:
            game_length = len(game_data.turn_data)
            winner = game_data.winner
            turn_data = game_data.turn_data
            for turn_no, turn in enumerate(turn_data):
                progress = (turn_no * 100) // game_length
                # find the appropriate bin for this progress
                bin_100_key = max([k for k in bins_100.keys() if k <= progress])
                bin_10_key = max([k for k in bins_10.keys() if k <= progress])
                # check the accuracy of the board
                # value acc
                board = turn.board
                sd_outcome = gt.get_outcome(board)
                gd_outcome = 0 if winner is None else 1 if winner == board.current_player else -1
                if gd_outcome == 0:
                    continue # skip draws for this analysis
                # increment the counts (correct, total) for the appropriate bins
                if sd_outcome == gd_outcome:
                    bins_100[bin_100_key] = (bins_100[bin_100_key][0] + 1, bins_100[bin_100_key][1] + 1)
                    bins_10[bin_10_key] = (bins_10[bin_10_key][0] + 1, bins_10[bin_10_key][1] + 1)
                else:
                    bins_100[bin_100_key] = (bins_100[bin_100_key][0], bins_100[bin_100_key][1] + 1)
                    bins_10[bin_10_key] = (bins_10[bin_10_key][0], bins_10[bin_10_key][1] + 1)
                # TODO: policy acc
    print(bins_100)
    print(bins_10)

    accs_100 = {bin_key: bin_tuples[0] / bin_tuples[1] for bin_key, bin_tuples in bins_100.items() if bin_tuples[1] > 0}
    accs_10 = {bin_key: bin_tuples[0] / bin_tuples[1] for bin_key, bin_tuples in bins_10.items() if bin_tuples[1] > 0}
    print(accs_100)
    print(accs_10)

    nums_100 = {bin_key: bin_tuples[1] for bin_key, bin_tuples in bins_100.items()}
    nums_10 = {bin_key: bin_tuples[1] for bin_key, bin_tuples in bins_10.items()}
    print(nums_100)
    print(nums_10)

    progress_acc_100 = Series(["Num States", "Value Accuracy"])
    progress_acc_10 = Series(["Num States", "Value Accuracy"])
    for bin_key in sorted(accs_100.keys()):
        progress_acc_100.x.append(bin_key)
        progress_acc_100.ys["Num States"].append(nums_100[bin_key])
        progress_acc_100.ys["Value Accuracy"].append(accs_100[bin_key])
    for bin_key in sorted(accs_10.keys()):
        progress_acc_10.x.append(bin_key)
        progress_acc_10.ys["Num States"].append(nums_10[bin_key])
        progress_acc_10.ys["Value Accuracy"].append(accs_10[bin_key])

    # save the series
    save_series(progress_acc_100, f"{config.eval_dir}/gamedata_progress_acc_100.pkl")
    save_series(progress_acc_10, f"{config.eval_dir}/gamedata_progress_acc_10.pkl")

def main():
    setup_logging(level=20, log_dir=config.log_dir, process_name='training_data_evals')
    
    logger.info("Starting training data evaluations...")

    # by iteration
    #state_acc_dicts = get_state_accuracy_dicts(list(game_data_generator(config.training_dir, config.training_iterations)))
    #duplicate_states_analysis(state_acc_dicts)
    # by seen bins
    #thresholds = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    #state_acc_dicts = split_by_visited_seen_bins(get_state_accuracy_dict(get_all_games_generated_during_training()), thresholds)
    #duplicate_states_analysis(state_acc_dicts, thresholds)
    # overall
    duplicate_states_analysis([get_state_accuracy_dict(get_all_games_generated_during_training())])

    check_game_data_accuracy(list(game_data_generator(config.training_dir, config.training_iterations)))
    check_game_data_bias(list(game_data_generator(config.training_dir, config.training_iterations)))
    get_stats_of_each_iterations_game_data()
    get_training_performance_metrics()
    gamedata_accuracy_over_progress(list(game_data_generator(config.training_dir, config.training_iterations)))

    logger.info("Training data evaluations completed.")

if __name__ == "__main__":
    main()
