import pickle
import random
from dataclasses import dataclass
from typing import Callable, Generator, Any, Protocol

from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray

from cc.core import Player, Board
from cc.ground_truth import GroundTruth
from a0.game import GameData
from a0.train.dataset import DatasetData, stats_from_dataset_data, plot_model_performance
from a0.eval.plotting import Series, save_series
from a0.eval.dataset_evaluation import policy_accuracy_function, policy_probability_mass_function
from a0.eval.training_data import dataset_data_generator, game_data_generator

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def get_and_save_avg_training_metrics_per_iteration() -> None:
    '''
    Saves avg training loss/accuracy per iteration from DatasetData files.
    → training_metrics.pkl
    Also saves plots of training performance over iterations.
    '''
    logger.info("Saving training performance metrics...")
    dd_gen = dataset_data_generator(config.training_dir, config.training_iterations)
    train_datas: list[DatasetData] = []

    series = Series(["Loss", "Value Loss", "Policy Loss", "Value Accuracy", "Policy Accuracy"])
    for i, dataset_data in dd_gen:
        loss, v_loss, p_loss, v_acc, p_acc = stats_from_dataset_data(dataset_data)
        series.x.append(i)
        series.ys["Loss"].append(loss)
        series.ys["Value Loss"].append(v_loss)
        series.ys["Policy Loss"].append(p_loss)
        series.ys["Value Accuracy"].append(v_acc)
        series.ys["Policy Accuracy"].append(p_acc)
        train_datas.append(dataset_data)
    
    plot_model_performance("training_plots/full_a0", train_datas)

    save_series(series, f"{config.eval_dir}/training_metrics.pkl")

@dataclass
class TurnInfo:
    iteration: int
    board: Board
    winner: Player | None
    turn_index: int
    game_length: int
    gt_outcome: float
    experienced_outcome: float
    gt_policy: NDArray[np.float32]
    experienced_policy: NDArray[np.float32]
    is_trivial: bool
    progress: int    # 0-99, percentage through the game
    alternative_targets: dict[str, Any]

@dataclass
class GameInfo:
    iteration: int
    winner: Player | None
    ended: bool
    game_length: int
    game_time: float

class Collector(Protocol):
    def on_game(self, gi: GameInfo) -> None: ...

    def on_turn(self, ti: TurnInfo) -> None: ...

    def on_iteration_end(self, iteration: int) -> None: ...

    def finalize(self) -> None: ...

def traverse_game_data_with_collectors(collectors: list[Collector]) -> None:
    '''
    Walks all game data once, computing per-turn facts and handing them
    to each collector.
    '''
    gt = GroundTruth()
    gd_gen = game_data_generator(config.training_dir, config.training_iterations)

    for i, game_data_list in tqdm(gd_gen, desc="Iterations"):
        for game_data in game_data_list:
            game_length = len(game_data.turn_data)
            winner = game_data.winner

            game_info = GameInfo(
                iteration=i,
                winner=winner,
                ended=game_data.ended,
                game_length=game_length,
                game_time=game_data.time,
            )
            for c in collectors:
                c.on_game(game_info)

            for turn_idx, turn in enumerate(game_data.turn_data):
                board = turn.board
                gt_outcome = gt.get_outcome(board)
                experienced_outcome = 0.0 if winner is None else 1.0 if winner == board.current_player else -1.0
                gt_policy = np.array(gt.get_1ply_policy_prob_dist_list(board, for_model=True))
                progress = (turn_idx * 100) // game_length

                turn_info = TurnInfo(
                    iteration=i,
                    board=board,
                    winner=winner,
                    turn_index=turn_idx,
                    game_length=game_length,
                    gt_outcome=gt_outcome,
                    experienced_outcome=experienced_outcome,
                    gt_policy=gt_policy,
                    experienced_policy=turn.player_data,
                    is_trivial=gt.is_trivial(board),
                    progress=progress,
                    alternative_targets=turn.alternative_targets
                )
                for c in collectors:
                    c.on_turn(turn_info)

        for c in collectors:
            c.on_iteration_end(i)

    for c in collectors:
        c.finalize()

class GameStatsCollector(Collector):
    '''Collects per-iteration game outcome stats → gamedata_stats.pkl'''

    def __init__(self):
        self.series = Series([
            "Total Games", "Player X Wins", "Player O Wins",
            "Draws (Repeat)", "Draws (Timeout)",
            "Avg Game Length", "Avg Game Time",
            "Std Game Length", "Std Game Time",
        ])
        self._reset_iteration()

    def _reset_iteration(self):
        self._total_games = 0
        self._px_wins = 0
        self._po_wins = 0
        self._draws_repeat = 0
        self._draws_timeout = 0
        self._game_lengths: list[int] = []
        self._game_times: list[float] = []

    def on_game(self, gi: GameInfo) -> None:
        self._total_games += 1
        self._game_lengths.append(gi.game_length)
        self._game_times.append(gi.game_time)
        if gi.winner == Player.PLAYER_X:
            self._px_wins += 1
        elif gi.winner == Player.PLAYER_O:
            self._po_wins += 1
        elif gi.ended:
            self._draws_repeat += 1
        else:
            self._draws_timeout += 1

    def on_turn(self, ti: TurnInfo) -> None:
        pass

    def on_iteration_end(self, iteration: int) -> None:
        self.series.x.append(iteration)
        self.series.ys["Total Games"].append(self._total_games)
        self.series.ys["Player X Wins"].append(self._px_wins)
        self.series.ys["Player O Wins"].append(self._po_wins)
        self.series.ys["Draws (Repeat)"].append(self._draws_repeat)
        self.series.ys["Draws (Timeout)"].append(self._draws_timeout)
        gl = self._game_lengths
        gt_ = self._game_times
        self.series.ys["Avg Game Length"].append(float(np.mean(gl)) if gl else 0)
        self.series.ys["Avg Game Time"].append(float(np.mean(gt_)) if gt_ else 0)
        self.series.ys["Std Game Length"].append(float(np.std(gl)) if gl else 0)
        self.series.ys["Std Game Time"].append(float(np.std(gt_)) if gt_ else 0)
        self._reset_iteration()

    def finalize(self) -> None:
        save_series(self.series, f"{config.eval_dir}/gamedata_stats.pkl")

class AccuracyCollector(Collector):
    '''
    Collects per-iteration and overall value/policy accuracy vs ground truth.
    → {name}_acc.pkl, {name}_overall_acc.pkl
    '''

    def __init__(
        self,
        name: str = "gamedata",
        get_outcome: Callable[[TurnInfo], float] = lambda ti: ti.experienced_outcome,
        get_policy: Callable[[TurnInfo], NDArray[np.float32]] = lambda ti: ti.experienced_policy,
    ):
        self._name = name
        self._get_outcome = get_outcome
        self._get_policy = get_policy
        self.iteration_series = Series([
            "Iteration Value Accuracy", "Iteration Value Accuracy ND",
            "Iteration Policy Accuracy", "Iteration Policy PM",
            "Iteration Policy Accuracy NT", "Iteration Policy PM NT",
        ])
        self.overall_series = Series([
            "Overall Value Accuracy", "Overall Value Accuracy ND",
            "Overall Policy Accuracy", "Overall Policy PM",
            "Overall Policy Accuracy NT", "Overall Policy PM NT",
        ])
        self._reset_iteration()
        # overall accumulators
        self._total = 0
        self._total_nd = 0
        self._total_nt = 0
        self._total_correct_value = 0
        self._total_correct_value_nd = 0
        self._total_correct_policy = 0
        self._total_pm_policy = 0.0
        self._total_correct_policy_nt = 0
        self._total_pm_policy_nt = 0.0

    def _reset_iteration(self):
        self._it_total = 0
        self._it_total_nd = 0
        self._it_total_nt = 0
        self._it_correct_value = 0
        self._it_correct_value_nd = 0
        self._it_correct_policy = 0
        self._it_pm_policy = 0.0
        self._it_correct_policy_nt = 0
        self._it_pm_policy_nt = 0.0

    def on_game(self, gi: GameInfo) -> None:
        pass

    def on_turn(self, ti: TurnInfo) -> None:
        outcome = self._get_outcome(ti)
        policy = self._get_policy(ti)

        # value accuracy
        value_correct = (ti.gt_outcome == outcome)
        if value_correct:
            self._it_correct_value += 1
            self._total_correct_value += 1

        # value accuracy (no draws)
        is_draw = (outcome == 0.0)
        if not is_draw:
            self._it_total_nd += 1
            self._total_nd += 1
            if value_correct:
                self._it_correct_value_nd += 1
                self._total_correct_value_nd += 1

        # policy accuracy and probability mass
        pm = policy_probability_mass_function(policy, ti.gt_policy)
        policy_correct = policy_accuracy_function(policy, ti.gt_policy)
        self._it_pm_policy += pm
        self._total_pm_policy += pm
        if policy_correct:
            self._it_correct_policy += 1
            self._total_correct_policy += 1

        # policy accuracy (non-trivial)
        if not ti.is_trivial:
            self._it_total_nt += 1
            self._total_nt += 1
            self._it_pm_policy_nt += pm
            self._total_pm_policy_nt += pm
            if policy_correct:
                self._it_correct_policy_nt += 1
                self._total_correct_policy_nt += 1

        self._it_total += 1
        self._total += 1

    def on_iteration_end(self, iteration: int) -> None:
        t = self._it_total
        t_nd = self._it_total_nd
        t_nt = self._it_total_nt
        self.iteration_series.x.append(iteration)
        self.iteration_series.ys["Iteration Value Accuracy"].append(self._it_correct_value / t if t else 0)
        self.iteration_series.ys["Iteration Value Accuracy ND"].append(self._it_correct_value_nd / t_nd if t_nd else 0)
        self.iteration_series.ys["Iteration Policy Accuracy"].append(self._it_correct_policy / t if t else 0)
        self.iteration_series.ys["Iteration Policy PM"].append(self._it_pm_policy / t if t else 0)
        self.iteration_series.ys["Iteration Policy Accuracy NT"].append(self._it_correct_policy_nt / t_nt if t_nt else 0)
        self.iteration_series.ys["Iteration Policy PM NT"].append(self._it_pm_policy_nt / t_nt if t_nt else 0)
        self._reset_iteration()

    def finalize(self) -> None:
        save_series(self.iteration_series, f"{config.eval_dir}/{self._name}_acc.pkl")

        t = self._total
        t_nd = self._total_nd
        t_nt = self._total_nt

        for iter_num in range(len(self.iteration_series.x)):
            x = self.iteration_series.x[iter_num]
            self.overall_series.x.append(x)
            self.overall_series.ys["Overall Value Accuracy"].append(self._total_correct_value / t if t else 0)
            self.overall_series.ys["Overall Value Accuracy ND"].append(self._total_correct_value_nd / t_nd if t_nd else 0)
            self.overall_series.ys["Overall Policy Accuracy"].append(self._total_correct_policy / t if t else 0)
            self.overall_series.ys["Overall Policy PM"].append(self._total_pm_policy / t if t else 0)
            self.overall_series.ys["Overall Policy Accuracy NT"].append(self._total_correct_policy_nt / t_nt if t_nt else 0)
            self.overall_series.ys["Overall Policy PM NT"].append(self._total_pm_policy_nt / t_nt if t_nt else 0)
        save_series(self.overall_series, f"{config.eval_dir}/{self._name}_overall_acc.pkl")

class BiasCollector(Collector):
    '''
    Collects per-iteration and overall win/loss/draw percentages.
    → {name}_bias.pkl, {name}_overall_bias.pkl
    '''

    def __init__(
        self,
        name: str = "gamedata",
        get_outcome: Callable[[TurnInfo], float] = lambda ti: ti.experienced_outcome,
    ):
        self._name = name
        self._get_outcome = get_outcome
        self.iteration_series = Series(["Iteration Win Percent", "Iteration Loss Percent", "Iteration Draw Percent"])
        self.overall_series = Series(["Overall Win Percent", "Overall Loss Percent", "Overall Draw Percent"])
        self._reset_iteration()
        self._total_wins = 0
        self._total_losses = 0
        self._total_draws = 0
        self._total = 0

    def _reset_iteration(self):
        self._it_wins = 0
        self._it_losses = 0
        self._it_draws = 0
        self._it_total = 0

    def on_game(self, gi: GameInfo) -> None:
        pass

    def on_turn(self, ti: TurnInfo) -> None:
        outcome = self._get_outcome(ti)
        if outcome > 0:
            self._it_wins += 1
            self._total_wins += 1
        elif outcome < 0:
            self._it_losses += 1
            self._total_losses += 1
        else:
            self._it_draws += 1
            self._total_draws += 1
        self._it_total += 1
        self._total += 1

    def on_iteration_end(self, iteration: int) -> None:
        t = self._it_total
        self.iteration_series.x.append(iteration)
        self.iteration_series.ys["Iteration Win Percent"].append(self._it_wins / t if t else 0)
        self.iteration_series.ys["Iteration Loss Percent"].append(self._it_losses / t if t else 0)
        self.iteration_series.ys["Iteration Draw Percent"].append(self._it_draws / t if t else 0)
        self._reset_iteration()

    def finalize(self) -> None:
        save_series(self.iteration_series, f"{config.eval_dir}/{self._name}_bias.pkl")

        t = self._total
        for iter_num in range(len(self.iteration_series.x)):
            x = self.iteration_series.x[iter_num]
            self.overall_series.x.append(x)
            self.overall_series.ys["Overall Win Percent"].append(self._total_wins / t if t else 0)
            self.overall_series.ys["Overall Loss Percent"].append(self._total_losses / t if t else 0)
            self.overall_series.ys["Overall Draw Percent"].append(self._total_draws / t if t else 0)
        save_series(self.overall_series, f"{config.eval_dir}/{self._name}_overall_bias.pkl")

def run_collectors() -> None:
    '''
    Runs all training data collectors in a single pass over the game data
    '''
    logger.info("Starting training data analyses...")

    # TODO: hardcode this string
    alt_outcome: Callable[[TurnInfo], float] = lambda ti: ti.alternative_targets["td_lambda"]

    # TODO: create OverGameProgressMetaCollector.

    collectors: list[Collector] = [
        GameStatsCollector(),
        AccuracyCollector(),
        BiasCollector(),
        AccuracyCollector(
            name="gamedata_alt",
            get_outcome=alt_outcome
        ),
        BiasCollector(
            name="gamedata_alt",
            get_outcome=alt_outcome
        )
    ]
    traverse_game_data_with_collectors(collectors)

    logger.info("Training data analyses completed.")

def main():
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name='training_data'
    )

    get_and_save_avg_training_metrics_per_iteration()
    run_collectors()

if __name__ == "__main__":
    main()
