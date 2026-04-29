from a0.eval.plotting import Series, save_series
from cc.ground_truth import GroundTruth
from cc.core import Board

from a0.eval.dataset_evaluation import policy_accuracy_function

import numpy as np
from numpy.typing import NDArray

from collections import Counter

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from cc.core import Board
from cc.ground_truth import GroundTruth
from a0.utils.plotting import Series, save_series
from a0.eval.dataset_evaluation import policy_accuracy_function
from a0.eval3.collectors.base import Collector, GameInfo, TurnInfo

from config import config
from utils.log import get_logger
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
   
class BPPCollector(Collector):
    '''
    Best Possible Performance (BPP) and BPP Model-Accuracy (BPPMA) metrics.
    BPP measures the best achievable accuracy when always predicting the most
    common value/policy seen for each unique board state. BPPMA measures the
    accuracy of those majority predictions against ground truth.
    Per-iteration aggregates within each iteration; overall aggregates across all.
    → {name}_bpp.pkl, {name}_bppma.pkl, {name}_overall_bpp.pkl, {name}_overall_bppma.pkl
    '''

    def __init__(
        self,
        gt: GroundTruth,
        name: str = "gamedata",
        get_outcome: Callable[[TurnInfo], float] = lambda ti: ti.experienced_outcome,
        get_policy: Callable[[TurnInfo], NDArray[np.float32]] = lambda ti: ti.experienced_policy,
    ):
        self._name = name
        self._get_outcome = get_outcome
        self._get_policy = get_policy
        self._gt = gt
        self.iteration_bpp_series = Series([
            "Iteration Value BPP", "Iteration Value BPP ND",
            "Iteration Policy BPP", "Iteration Policy BPP NT",
        ])
        self.iteration_bppma_series = Series([
            "Iteration Value BPPMA", "Iteration Value BPPMA ND",
            "Iteration Policy BPPMA", "Iteration Policy BPPMA NT",
        ])
        self.overall_bpp_series = Series([
            "Overall Value BPP", "Overall Value BPP ND",
            "Overall Policy BPP", "Overall Policy BPP NT",
        ])
        self.overall_bppma_series = Series([
            "Overall Value BPPMA", "Overall Value BPPMA ND",
            "Overall Policy BPPMA", "Overall Policy BPPMA NT",
        ])
        self._iter_value_dict: dict[Board, list[int]] = {}
        self._iter_policy_dict: dict[Board, list[NDArray[np.float32]]] = {}
        self._overall_value_dict: dict[Board, list[int]] = {}
        self._overall_policy_dict: dict[Board, list[NDArray[np.float32]]] = {}

    def on_game(self, gi: GameInfo) -> None:
        pass

    def on_turn(self, ti: TurnInfo) -> None:
        value = int(self._get_outcome(ti))
        policy = self._get_policy(ti)
        board = ti.board

        if board not in self._iter_value_dict:
            self._iter_value_dict[board] = []
            self._iter_policy_dict[board] = []
        self._iter_value_dict[board].append(value)
        self._iter_policy_dict[board].append(policy)

        if board not in self._overall_value_dict:
            self._overall_value_dict[board] = []
            self._overall_policy_dict[board] = []
        self._overall_value_dict[board].append(value)
        self._overall_policy_dict[board].append(policy)

    def on_iteration_end(self, iteration: int) -> None:
        (v_bpp, v_bpp_nd, p_bpp, p_bpp_nt,
         v_bppma, v_bppma_nd, p_bppma, p_bppma_nt) = get_bpp_and_bppma_metrics(
            self._gt, self._iter_value_dict, self._iter_policy_dict
        )
        self.iteration_bpp_series.x.append(iteration)
        self.iteration_bpp_series.ys["Iteration Value BPP"].append(v_bpp)
        self.iteration_bpp_series.ys["Iteration Value BPP ND"].append(v_bpp_nd)
        self.iteration_bpp_series.ys["Iteration Policy BPP"].append(p_bpp)
        self.iteration_bpp_series.ys["Iteration Policy BPP NT"].append(p_bpp_nt)
        self.iteration_bppma_series.x.append(iteration)
        self.iteration_bppma_series.ys["Iteration Value BPPMA"].append(v_bppma)
        self.iteration_bppma_series.ys["Iteration Value BPPMA ND"].append(v_bppma_nd)
        self.iteration_bppma_series.ys["Iteration Policy BPPMA"].append(p_bppma)
        self.iteration_bppma_series.ys["Iteration Policy BPPMA NT"].append(p_bppma_nt)
        self._iter_value_dict = {}
        self._iter_policy_dict = {}

    def finalize(self) -> None:
        save_series(self.iteration_bpp_series, f"{config.eval_dir}/{self._name}_bpp.pkl")
        save_series(self.iteration_bppma_series, f"{config.eval_dir}/{self._name}_bppma.pkl")

        (v_bpp, v_bpp_nd, p_bpp, p_bpp_nt,
         v_bppma, v_bppma_nd, p_bppma, p_bppma_nt) = get_bpp_and_bppma_metrics(
            self._gt, self._overall_value_dict, self._overall_policy_dict
        )
        for x in self.iteration_bpp_series.x:
            self.overall_bpp_series.x.append(x)
            self.overall_bpp_series.ys["Overall Value BPP"].append(v_bpp)
            self.overall_bpp_series.ys["Overall Value BPP ND"].append(v_bpp_nd)
            self.overall_bpp_series.ys["Overall Policy BPP"].append(p_bpp)
            self.overall_bpp_series.ys["Overall Policy BPP NT"].append(p_bpp_nt)
            self.overall_bppma_series.x.append(x)
            self.overall_bppma_series.ys["Overall Value BPPMA"].append(v_bppma)
            self.overall_bppma_series.ys["Overall Value BPPMA ND"].append(v_bppma_nd)
            self.overall_bppma_series.ys["Overall Policy BPPMA"].append(p_bppma)
            self.overall_bppma_series.ys["Overall Policy BPPMA NT"].append(p_bppma_nt)
        save_series(self.overall_bpp_series, f"{config.eval_dir}/{self._name}_overall_bpp.pkl")
        save_series(self.overall_bppma_series, f"{config.eval_dir}/{self._name}_overall_bppma.pkl")
        logger.info(
            f"BPPCollector '{self._name}': saved {self._name}_bpp.pkl + {self._name}_bppma.pkl "
            f"+ {self._name}_overall_bpp.pkl + {self._name}_overall_bppma.pkl. "
            f"Overall: v_bpp={v_bpp:.2%} v_bpp_nd={v_bpp_nd:.2%} "
            f"p_bpp={p_bpp:.2%} p_bpp_nt={p_bpp_nt:.2%} "
            f"v_bppma={v_bppma:.2%} v_bppma_nd={v_bppma_nd:.2%} "
            f"p_bppma={p_bppma:.2%} p_bppma_nt={p_bppma_nt:.2%} "
            f"(unique boards={len(self._overall_value_dict)})"
        )
