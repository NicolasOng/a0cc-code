from a0.game import GameData
from a0.eval.plotting import Series, save_series, load_series, plot_given
from cc.ground_truth import GroundTruth
from cc.core import Board
from a0.eval.training_data import game_data_generator

from a0.eval.dataset_evaluation import policy_accuracy_function

import numpy as np
from numpy.typing import NDArray

from collections import Counter

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def get_majority_max_index(arrays: list[NDArray[np.float32]]) -> NDArray[np.float32]:
    # List of arrays, all same shape
    max_indices = [np.argmax(arr) for arr in arrays]  # np.argmax returns flat index for multi-dim
    # Find the most common index
    counter = Counter(max_indices)
    majority_index = counter.most_common(1)[0][0]  # Gets the most common flat index (the "value" here is the index number)
    # Create output array: zeros with 1 at majority_index (using flat indexing for multi-dim)
    output = np.zeros(arrays[0].shape, dtype=np.float32)
    output.flat[majority_index] = 1.0
    return output

def get_bpp_and_bppma_metrics(gt: GroundTruth, value_dict: dict[Board, list[int]], policy_dict: dict[Board, list[NDArray[np.float32]]]) -> tuple[float, float, float, float, float, float, float, float]:
    total_unique_boads = 0
    total_unique_nd_boads = 0
    total_unique_nt_boads = 0
    total_boards = 0
    total_nd_boards = 0
    total_nt_boards = 0
    value_bpp = 0
    value_bpp_nd = 0
    policy_bpp = 0
    policy_bpp_nt = 0
    value_bppma = 0
    value_bppma_nd = 0
    policy_bppma = 0
    policy_bppma_nt = 0
    for board in value_dict.keys():
        is_trivial = gt.is_trivial(board)

        values = value_dict[board]
        policies = policy_dict[board]

        assert len(values) == len(policies), "Different number of values and policies for the same board!"
        num_boards = len(values)

        values_nd = [v for v in values if v != 0]
        num_nd_boards = len(values_nd)

        most_common_value = Counter(values).most_common(1)[0][0]
        most_common_value_nd = Counter(values_nd).most_common(1)[0][0] if num_nd_boards > 0 else None
        most_common_policy = get_majority_max_index(policies)

        most_common_value_count = values.count(most_common_value)
        most_common_value_nd_count = None
        if most_common_value_nd:
            most_common_value_nd_count = values_nd.count(most_common_value_nd) if num_nd_boards > 0 else 0
        most_common_policy_count = sum(1 for p in policies if policy_accuracy_function(most_common_policy, p))

        gt_value = gt.get_outcome(board)
        gt_policy = np.array(gt.get_1ply_policy_prob_dist_list(board, for_model=True))

        most_common_value_is_acc = 1 if most_common_value == gt_value else 0
        most_common_value_nd_is_acc = None
        if most_common_value_nd:
            most_common_value_nd_is_acc = 1 if most_common_value_nd == gt_value else 0
        most_common_policy_is_acc = 1 if policy_accuracy_function(most_common_policy, gt_policy) else 0

        total_unique_boads += 1
        total_unique_nd_boads += 1 if num_nd_boards > 0 else 0
        if not is_trivial:
            total_unique_nt_boads += 1

        total_boards += num_boards
        if not is_trivial:
            total_nt_boards += num_boards
        total_nd_boards += num_nd_boards

        value_bpp += most_common_value_count
        if most_common_value_nd_count:
            value_bpp_nd += most_common_value_nd_count
        policy_bpp += most_common_policy_count
        if not is_trivial:
            policy_bpp_nt += most_common_policy_count
        
        value_bppma += most_common_value_is_acc
        if most_common_value_nd_is_acc:
            value_bppma_nd += most_common_value_nd_is_acc
        policy_bppma += most_common_policy_is_acc
        if not is_trivial:
            policy_bppma_nt += most_common_policy_is_acc
    
    value_bpp_rate = value_bpp / total_boards if total_boards > 0 else 0
    value_bpp_nd_rate = value_bpp_nd / total_nd_boards if total_nd_boards > 0 else 0
    policy_bpp_rate = policy_bpp / total_boards if total_boards > 0 else 0
    policy_bpp_nt_rate = policy_bpp_nt / total_nt_boards if total_nt_boards > 0 else 0

    value_bppma_rate = value_bppma / total_unique_boads if total_unique_boads > 0 else 0
    value_bppma_nd_rate = value_bppma_nd / total_unique_nd_boads if total_unique_nd_boads > 0 else 0
    policy_bppma_rate = policy_bppma / total_unique_boads if total_unique_boads > 0 else 0
    policy_bppma_nt_rate = policy_bppma_nt / total_unique_nt_boads if total_unique_nt_boads > 0 else 0

    return (value_bpp_rate, value_bpp_nd_rate, policy_bpp_rate, policy_bpp_nt_rate,
            value_bppma_rate, value_bppma_nd_rate, policy_bppma_rate, policy_bppma_nt_rate)
    

def get_best_possible_performance_overall(game_data_lists: list[tuple[int, list[GameData]]]) -> None:
    '''
    ...
    '''
    total_boards = 0
    value_dict: dict[Board, list[int]] = {}
    policy_dict: dict[Board, list[NDArray[np.float32]]] = {}
    x_list: list[int] = []
    # for each game data list/iteration,
    for i, game_data_list in game_data_lists:
        x_list.append(i)
        # go through each game,
        for _, game_data in enumerate(game_data_list):
            winner = game_data.winner
            # go through each turn
            for _, game_state in enumerate(game_data.turn_data):
                total_boards += 1
                board = game_state.board
                value = 0 if winner is None else 1 if winner == board.current_player else -1
                policy: NDArray[np.float32] = game_state.player_data
                # store the value
                if board not in value_dict:
                    value_dict[board] = []
                value_dict[board].append(value)
                # store the policy
                if board not in policy_dict:
                    policy_dict[board] = []
                policy_dict[board].append(policy)
    
    logger.info(f"Total boards processed: {total_boards}")
    logger.info(f"Unique boards encountered: {len(value_dict)}")
    assert len(value_dict) == len(policy_dict), "Value and policy dicts have different number of unique boards!"

    gt = GroundTruth()
    (value_bpp_rate, value_bpp_nd_rate, policy_bpp_rate, policy_bpp_nt_rate,
            value_bppma_rate, value_bppma_nd_rate, policy_bppma_rate, policy_bppma_nt_rate) = get_bpp_and_bppma_metrics(gt, value_dict, policy_dict)

    gd_overall_bpp_series = Series(["Overall Value BPP", "Overall Value BPP ND", "Overall Policy BPP", "Overall Policy BPP NT"])
    gd_overall_bppma_series = Series(["Overall Value BPPMA", "Overall Value BPPMA ND", "Overall Policy BPPMA", "Overall Policy BPPMA NT"])
    for i in x_list:
        gd_overall_bpp_series.x.append(i)
        gd_overall_bpp_series.ys["Overall Value BPP"].append(value_bpp_rate)
        gd_overall_bpp_series.ys["Overall Value BPP ND"].append(value_bpp_nd_rate)
        gd_overall_bpp_series.ys["Overall Policy BPP"].append(policy_bpp_rate)
        gd_overall_bpp_series.ys["Overall Policy BPP NT"].append(policy_bpp_nt_rate)
        gd_overall_bppma_series.x.append(i)
        gd_overall_bppma_series.ys["Overall Value BPPMA"].append(value_bppma_rate)
        gd_overall_bppma_series.ys["Overall Value BPPMA ND"].append(value_bppma_nd_rate)
        gd_overall_bppma_series.ys["Overall Policy BPPMA"].append(policy_bppma_rate)
        gd_overall_bppma_series.ys["Overall Policy BPPMA NT"].append(policy_bppma_nt_rate)
    save_series(gd_overall_bpp_series, f"{config.eval_dir}/gd_overall_bpp_series.pkl")
    save_series(gd_overall_bppma_series, f"{config.eval_dir}/gd_overall_bppma_series.pkl")
    
def get_best_possible_performance_per_iteration(game_data_lists: list[tuple[int, list[GameData]]]) -> None:
    '''
    ...
    '''
    gt = GroundTruth()
    gd_bpp_series = Series(["Iteration Value BPP", "Iteration Value BPP ND", "Iteration Policy BPP", "Iteration Policy BPP NT"])
    gd_bppma_series = Series(["Iteration Value BPPMA", "Iteration Value BPPMA ND", "Iteration Policy BPPMA", "Iteration Policy BPPMA NT"])
    total_boards = 0
    # for each game data list/iteration,
    for i, game_data_list in game_data_lists:
        # go through each game,
        gd_bpp_series.x.append(i)
        gd_bppma_series.x.append(i)
        itr_value_dict: dict[Board, list[int]] = {}
        itr_policy_dict: dict[Board, list[NDArray[np.float32]]] = {}
        for _, game_data in enumerate(game_data_list):
            winner = game_data.winner
            # go through each turn
            for _, game_state in enumerate(game_data.turn_data):
                total_boards += 1
                board = game_state.board
                value = 0 if winner is None else 1 if winner == board.current_player else -1
                policy: NDArray[np.float32] = game_state.player_data
                # store the value
                if board not in itr_value_dict:
                    itr_value_dict[board] = []
                itr_value_dict[board].append(value)
                # store the policy
                if board not in itr_policy_dict:
                    itr_policy_dict[board] = []
                itr_policy_dict[board].append(policy)
        (value_bpp_rate, value_bpp_nd_rate, policy_bpp_rate, policy_bpp_nt_rate,
            value_bppma_rate, value_bppma_nd_rate, policy_bppma_rate, policy_bppma_nt_rate) = get_bpp_and_bppma_metrics(gt, itr_value_dict, itr_policy_dict)
        gd_bpp_series.ys["Iteration Value BPP"].append(value_bpp_rate)
        gd_bpp_series.ys["Iteration Value BPP ND"].append(value_bpp_nd_rate)
        gd_bpp_series.ys["Iteration Policy BPP"].append(policy_bpp_rate)
        gd_bpp_series.ys["Iteration Policy BPP NT"].append(policy_bpp_nt_rate)
        gd_bppma_series.ys["Iteration Value BPPMA"].append(value_bppma_rate)
        gd_bppma_series.ys["Iteration Value BPPMA ND"].append(value_bppma_nd_rate)
        gd_bppma_series.ys["Iteration Policy BPPMA"].append(policy_bppma_rate)
        gd_bppma_series.ys["Iteration Policy BPPMA NT"].append(policy_bppma_nt_rate)
    save_series(gd_bpp_series, f"{config.eval_dir}/gd_bpp_series.pkl")
    save_series(gd_bppma_series, f"{config.eval_dir}/gd_bppma_series.pkl")

def main():
    game_data_lists = list(game_data_generator(config.training_dir, config.training_iterations))
    # Compute best possible performance per iteration
    get_best_possible_performance_per_iteration(game_data_lists)

    # Compute best possible performance overall
    get_best_possible_performance_overall(game_data_lists)

    # load the series and plot them
    gd_bpp_series = load_series(f"{config.eval_dir}/gd_bpp_series.pkl")
    gd_bppma_series = load_series(f"{config.eval_dir}/gd_bppma_series.pkl")
    gd_overall_bpp_series = load_series(f"{config.eval_dir}/gd_overall_bpp_series.pkl")
    gd_overall_bppma_series = load_series(f"{config.eval_dir}/gd_overall_bppma_series.pkl")

    plot_given("BPP Series",
    [
        ("Overall Value BPP", gd_overall_bpp_series.x, gd_overall_bpp_series.ys["Overall Value BPP"]),
        ("Overall Value BPP ND", gd_overall_bpp_series.x, gd_overall_bpp_series.ys["Overall Value BPP ND"]),
        ("Overall Policy BPP", gd_overall_bpp_series.x, gd_overall_bpp_series.ys["Overall Policy BPP"]),
        ("Overall Policy BPP NT", gd_overall_bpp_series.x, gd_overall_bpp_series.ys["Overall Policy BPP NT"]),
        ("Iteration Value BPP", gd_bpp_series.x, gd_bpp_series.ys["Iteration Value BPP"]),
        ("Iteration Value BPP ND", gd_bpp_series.x, gd_bpp_series.ys["Iteration Value BPP ND"]),
        ("Iteration Policy BPP", gd_bpp_series.x, gd_bpp_series.ys["Iteration Policy BPP"]),
        ("Iteration Policy BPP NT", gd_bpp_series.x, gd_bpp_series.ys["Iteration Policy BPP NT"]),
    ],
    "Iteration", "Accuracy", "bpp_series")

    plot_given("BPPMA Series",
    [
        ("Overall Value BPPMA", gd_overall_bppma_series.x, gd_overall_bppma_series.ys["Overall Value BPPMA"]),
        ("Overall Value BPPMA ND", gd_overall_bppma_series.x, gd_overall_bppma_series.ys["Overall Value BPPMA ND"]),
        ("Overall Policy BPPMA", gd_overall_bppma_series.x, gd_overall_bppma_series.ys["Overall Policy BPPMA"]),
        ("Overall Policy BPPMA NT", gd_overall_bppma_series.x, gd_overall_bppma_series.ys["Overall Policy BPPMA NT"]),
        ("Iteration Value BPPMA", gd_bppma_series.x, gd_bppma_series.ys["Iteration Value BPPMA"]),
        ("Iteration Value BPPMA ND", gd_bppma_series.x, gd_bppma_series.ys["Iteration Value BPPMA ND"]),
        ("Iteration Policy BPPMA", gd_bppma_series.x, gd_bppma_series.ys["Iteration Policy BPPMA"]),
        ("Iteration Policy BPPMA NT", gd_bppma_series.x, gd_bppma_series.ys["Iteration Policy BPPMA NT"]),
    ],
    "Iteration", "Accuracy", "bppma_series")

if __name__ == "__main__":
    setup_logging(
        level=25,
        log_dir=config.log_dir
    )
    
    main()
