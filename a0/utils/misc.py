import pickle
from tqdm import tqdm
from typing import Generator, Any
from collections import defaultdict
import random

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from cc.core import Player, Board
from cc.ground_truth import GroundTruth, RankUnrank
from a0.game import GameData
from a0.train.dataset import DatasetData, stats_from_dataset_data
from a0.eval.dataset_evaluation import Series, save_series, policy_accuracy_function, policy_probability_mass_function
from a0.utils.load_training_data import game_data_generator, dataset_data_generator

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def get_baseline_accuracy(boards: list[Board], gt: GroundTruth) -> tuple[float, float, float, float]:
    '''
    Returns the baseline accuracy for random guessing.
    Could do this more efficiently by batching the boards together,
    but this is simpler to implement and should be fast enough for our purposes.
    returns:
    - value accuracy
    - policy accuracy
    - value accuracy (non-draws)
    - policy accuracy (non-trivial boards)
    '''
    total_boards = len(boards)
    assert total_boards > 0, "No boards provided for baseline accuracy calculation."
    total_nd = 0
    total_nt = 0
    total_acc_policy = 0.0
    total_acc_value = 0.0
    total_acc_policy_nt = 0.0
    total_acc_value_nd = 0.0
    for board in boards:
        # policy
        gt_policy = gt.get_1ply_policy_outcomes_list(board, for_model=False)
        random_policy = gt.get_random_valid_move_prob_dist_list(board, for_model=False)

        policy_acc = policy_accuracy_function(np.array(random_policy), np.array(gt_policy))
        # value
        gt_outcome = gt.get_outcome(board)
        random_outcome = random.choice([-1, 1])
        value_acc = 1.0 if random_outcome == gt_outcome else 0.0
        # adding to totals
        total_acc_policy += policy_acc
        total_acc_value += value_acc
        if not gt.is_trivial(board):
            total_acc_policy_nt += policy_acc
            total_nt += 1
        if not gt_outcome == 0:
            total_acc_value_nd += value_acc
            total_nd += 1

    return total_acc_value / total_boards, total_acc_policy / total_boards, total_acc_value_nd / total_nd if total_nd > 0 else 0, total_acc_policy_nt / total_nt if total_nt > 0 else 0

def get_branching_factor(boards: list[Board], r: RankUnrank) -> float:
    '''
    Average number of legal moves per board. Solve-data-free; accepts any
    RankUnrank (including GroundTruth).
    '''
    total_branching_factor = 0
    for board in boards:
        total_branching_factor += sum(r.get_valid_moves_list(board, for_model=False))
    return total_branching_factor / len(boards) if boards else 0.0
