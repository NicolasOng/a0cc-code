from a0.game import GameData, TurnData
from a0.eval.plotting import Series, save_series, load_series, plot_given, plot_given_groups
from cc.ground_truth import GroundTruth
from cc.core import Board
from a0.eval.training_data import game_data_generator
from scripts.sl_on_policy_head import get_n_random_states

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
    '''
    Computes BPP and BPPMA metrics given value and policy dictionaries.
    value and policy dictionaries map Board states to lists of values/policies experienced for those states during self-play.
    BPP (Best Possible Performance) measures the best achievable accuracy by selecting the most common value/policy for each board state.
    BPPMA (Best Possible Performance Model Accuracy) measures accuracy by checking if the most common value/policy matches the ground truth.
    ie what the model's accuracy would be if it always predicted the most common value/policy for each board state.
    It also accounts for non-draw (ND) and non-trivial (NT) cases separately.
    '''
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
    Creates overall BPP and BPPMA metrics across all provided game data lists.
    It does this by aggregating value and policy data for each unique board state across all game data,
    then computing the BPP and BPPMA metrics based on this aggregated data.
    Saves the resulting metrics as series for later analysis and plotting.
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
    Creates BPP and BPPMA metrics for each iteration/game data list separately.
    For each iteration, it aggregates value and policy data for each unique board state within that iteration,
    then computes the BPP and BPPMA metrics based on this aggregated data.
    Saves the resulting metrics as series for later analysis and plotting.
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

def get_unique_states_metrics(game_data_lists: list[tuple[int, list[GameData]]]) -> None:
    '''
    Computes and logs the number of unique board states encountered across all provided game data lists.
    '''
    gd_unique_series = Series(["Iteration Unique States", "Iteration Unique Percent"])
    gd_unique_overall_series = Series(["Overall Unique States", "Overall Unique Percent"])
    x_list: list[int] = []
    unique_boards_overall: set[Board] = set()
    total_boards_overall = 0
    for i, game_data_list in game_data_lists:
        x_list.append(i)
        unique_boards_iteration: set[Board] = set()
        total_boards_iteration = 0
        for _, game_data in enumerate(game_data_list):
            for _, turn_data in enumerate(game_data.turn_data):
                board = turn_data.board
                total_boards_overall += 1
                total_boards_iteration += 1
                unique_boards_overall.add(board)
                unique_boards_iteration.add(board)
        logger.info(f"Iteration unique boards: {len(unique_boards_iteration)}, total boards: {total_boards_iteration}, percentage: {len(unique_boards_iteration)/total_boards_iteration:.2%}")
        gd_unique_series.x.append(i)
        gd_unique_series.ys["Iteration Unique States"].append(len(unique_boards_iteration))
        gd_unique_series.ys["Iteration Unique Percent"].append(len(unique_boards_iteration)/total_boards_iteration)
    logger.info(f"Overall unique boards: {len(unique_boards_overall)}, total boards: {total_boards_overall}, percentage: {len(unique_boards_overall)/total_boards_overall:.2%}")
    for i in x_list:
        gd_unique_overall_series.x.append(i)
        gd_unique_overall_series.ys["Overall Unique States"].append(len(unique_boards_overall))
        gd_unique_overall_series.ys["Overall Unique Percent"].append(len(unique_boards_overall)/total_boards_overall)
    
    save_series(gd_unique_series, f"{config.eval_dir}/gd_unique_series.pkl")
    save_series(gd_unique_overall_series, f"{config.eval_dir}/gd_unique_overall_series.pkl")

def get_state_count_metrics(game_data_lists: list[tuple[int, list[GameData]]]) -> None:
    '''
    Computes and logs how many boards appeared 1 time, 2 times, etc., across all provided game data lists.
    '''
    max_count = 5
    y_list: list[str] = []
    for count in range(1, max_count + 1):
        y_list.append(f"Boards Appeared {count} Times")
    gd_state_count_series = Series(y_list)
    gd_state_count_series_overall = Series(y_list)
    board_list_overall: list[Board] = []
    total_boards_overall = 0
    x_list: list[int] = []
    for i, game_data_list in game_data_lists:
        x_list.append(i)
        board_list_iteration: list[Board] = []
        total_boards_iteration = 0
        for _, game_data in enumerate(game_data_list):
            for _, turn_data in enumerate(game_data.turn_data):
                total_boards_overall += 1
                total_boards_iteration += 1
                board = turn_data.board
                board_list_overall.append(board)
                board_list_iteration.append(board)
        gd_state_count_series.x.append(i)
        # prints how many boards appeared 1 time, 2 times, etc, up to max_count
        counter_iteration = Counter(board_list_iteration)
        logger.info(f"Iteration total boards: {total_boards_iteration}")
        logger.info(f"Iteration board occurrence counts:")
        for count in range(1, max_count + 1):
            num_boards_with_count = sum(1 for v in counter_iteration.values() if v == count)
            logger.info(f" Boards that appeared {count} times: {num_boards_with_count}, fraction: {num_boards_with_count / total_boards_iteration:.2%}")
            gd_state_count_series.ys[f"Boards Appeared {count} Times"].append(num_boards_with_count / total_boards_iteration)
    
    logger.info(f"Overall total boards: {total_boards_overall}")
    counter_overall = Counter(board_list_overall)
    for i in x_list:
        gd_state_count_series_overall.x.append(i)
        logger.info(f"Overall board occurrence counts:")
        for count in range(1, max_count + 1):
            num_boards_with_count = sum(1 for v in counter_overall.values() if v == count)
            logger.info(f" Boards that appeared {count} times: {num_boards_with_count}, fraction: {num_boards_with_count / total_boards_overall:.2%}")
            gd_state_count_series_overall.ys[f"Boards Appeared {count} Times"].append(num_boards_with_count / total_boards_overall)
    
    save_series(gd_state_count_series, f"{config.eval_dir}/gd_state_count_series.pkl")
    save_series(gd_state_count_series_overall, f"{config.eval_dir}/gd_state_count_series_overall.pkl")

def get_and_plot_bpp_metrics(game_data_lists: list[tuple[int, list[GameData]]]) -> None:
    # Compute best possible performance per iteration
    get_best_possible_performance_per_iteration(game_data_lists)

    # Compute best possible performance overall
    get_best_possible_performance_overall(game_data_lists)

    # load the series and plot them
    gd_bpp_series = load_series(f"{config.eval_dir}/gd_bpp_series.pkl")
    gd_bppma_series = load_series(f"{config.eval_dir}/gd_bppma_series.pkl")
    gd_overall_bpp_series = load_series(f"{config.eval_dir}/gd_overall_bpp_series.pkl")
    gd_overall_bppma_series = load_series(f"{config.eval_dir}/gd_overall_bppma_series.pkl")

    plot_given_groups("BPP Series",
    [
        [
            ("Overall Value BPP", gd_overall_bpp_series.x, gd_overall_bpp_series.ys["Overall Value BPP"]),
            ("Iteration Value BPP", gd_bpp_series.x, gd_bpp_series.ys["Iteration Value BPP"]),
        ],
        [
            ("Overall Value BPP ND", gd_overall_bpp_series.x, gd_overall_bpp_series.ys["Overall Value BPP ND"]),
            ("Iteration Value BPP ND", gd_bpp_series.x, gd_bpp_series.ys["Iteration Value BPP ND"]),
        ],
        [
            ("Overall Policy BPP", gd_overall_bpp_series.x, gd_overall_bpp_series.ys["Overall Policy BPP"]),
            ("Iteration Policy BPP", gd_bpp_series.x, gd_bpp_series.ys["Iteration Policy BPP"]),
        ],
        [
            ("Overall Policy BPP NT", gd_overall_bpp_series.x, gd_overall_bpp_series.ys["Overall Policy BPP NT"]),
            ("Iteration Policy BPP NT", gd_bpp_series.x, gd_bpp_series.ys["Iteration Policy BPP NT"]),
        ]
    ],
    "Iteration", "Accuracy", "bpp_series")

    plot_given_groups("BPPMA Series",
    [
        [
            ("Overall Value BPPMA", gd_overall_bppma_series.x, gd_overall_bppma_series.ys["Overall Value BPPMA"]),
            ("Iteration Value BPPMA", gd_bppma_series.x, gd_bppma_series.ys["Iteration Value BPPMA"]),
        ],
        [
            ("Overall Value BPPMA ND", gd_overall_bppma_series.x, gd_overall_bppma_series.ys["Overall Value BPPMA ND"]),
            ("Iteration Value BPPMA ND", gd_bppma_series.x, gd_bppma_series.ys["Iteration Value BPPMA ND"]),
        ],
        [
            ("Overall Policy BPPMA", gd_overall_bppma_series.x, gd_overall_bppma_series.ys["Overall Policy BPPMA"]),
            ("Iteration Policy BPPMA", gd_bppma_series.x, gd_bppma_series.ys["Iteration Policy BPPMA"]),
        ],
        [
            ("Overall Policy BPPMA NT", gd_overall_bppma_series.x, gd_overall_bppma_series.ys["Overall Policy BPPMA NT"]),
            ("Iteration Policy BPPMA NT", gd_bppma_series.x, gd_bppma_series.ys["Iteration Policy BPPMA NT"]),
        ]
    ],
    "Iteration", "Accuracy", "bppma_series")

def get_and_plot_unique_states_metrics(game_data_lists: list[tuple[int, list[GameData]]]) -> None:
    # Compute unique states metrics
    get_unique_states_metrics(game_data_lists)

    # load the series and plot them
    gd_unique_series = load_series(f"{config.eval_dir}/gd_unique_series.pkl")
    gd_unique_overall_series = load_series(f"{config.eval_dir}/gd_unique_overall_series.pkl")

    plot_given("Unique States Count Overall",
    [
        ("Overall Unique States", gd_unique_overall_series.x, gd_unique_overall_series.ys["Overall Unique States"]),
    ],
    "Iteration", "Count / Percent", "unique_states_count_overall")

    plot_given("Unique States Count Iteration",
    [
        ("Iteration Unique States", gd_unique_series.x, gd_unique_series.ys["Iteration Unique States"]),
    ],
    "Iteration", "Count / Percent", "unique_states_count_iteration")

    plot_given("Unique States Percentages",
    [
        ("Iteration Unique Percent", gd_unique_series.x, gd_unique_series.ys["Iteration Unique Percent"]),
        ("Overall Unique Percent", gd_unique_overall_series.x, gd_unique_overall_series.ys["Overall Unique Percent"]),
    ],
    "Iteration", "Count / Percent", "unique_states_percentages")

def get_and_plot_state_count_metrics(game_data_lists: list[tuple[int, list[GameData]]]) -> None:
    # Compute state count metrics
    #get_state_count_metrics(game_data_lists)

    # load the series and plot them
    gd_state_count_series = load_series(f"{config.eval_dir}/gd_state_count_series.pkl")
    gd_state_count_series_overall = load_series(f"{config.eval_dir}/gd_state_count_series_overall.pkl")

    max_count = 5
    plot_given_groups("State Count Series Iteration",
    [
        [
            (f"Boards Appeared {count} Times Percentage Overall", gd_state_count_series_overall.x, gd_state_count_series_overall.ys[f"Boards Appeared {count} Times"]),
            (f"Boards Appeared {count} Times Percentage Iteration", gd_state_count_series.x, gd_state_count_series.ys[f"Boards Appeared {count} Times"])
        ]
        for count in range(1, max_count + 1)
    ],
    "Iteration", "Count", "state_count_series_log", use_log_y=True)

    plot_given_groups("State Count Series Iteration",
    [
        [
            (f"Boards Appeared {count} Times Percentage Overall", gd_state_count_series_overall.x, gd_state_count_series_overall.ys[f"Boards Appeared {count} Times"]),
            (f"Boards Appeared {count} Times Percentage Iteration", gd_state_count_series.x, gd_state_count_series.ys[f"Boards Appeared {count} Times"])
        ]
        for count in range(1, max_count + 1)
    ],
    "Iteration", "Count", "state_count_series", use_log_y=False)

    plot_given_groups("State Count Series Iteration",
    [
        [
            (f"Boards Appeared {count} Times Percentage Overall", gd_state_count_series_overall.x, gd_state_count_series_overall.ys[f"Boards Appeared {count} Times"]),
            (f"Boards Appeared {count} Times Percentage Iteration", gd_state_count_series.x, gd_state_count_series.ys[f"Boards Appeared {count} Times"])
        ]
        for count in range(1, 1 + 1)
    ],
    "Iteration", "Count", "state_count_series_1", use_log_y=False)


def main():
    #game_data_lists = list(game_data_generator(config.training_dir, config.training_iterations))

    #get_and_plot_bpp_metrics(game_data_lists)
    #get_and_plot_unique_states_metrics(game_data_lists)
    get_and_plot_state_count_metrics(1)

if __name__ == "__main__":
    setup_logging(
        level=25,
        log_dir=config.log_dir
    )
    
    main()
