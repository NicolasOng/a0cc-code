import os
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
from a0.dataset import Dataset
from a0.train.dataset import DatasetData, stats_from_dataset_data, plot_model_performance
from a0.utils.plotting import Series, save_series, plot_ridgeline, DistributionSeries, save_distribution_series
from a0.eval.dataset_evaluation import policy_accuracy_function, policy_probability_mass_function
from a0.eval.training_data import dataset_data_generator, game_data_generator
from a0.utils.load_training_data import dataset_diagnostics_generator
from a0.experience_buffer import ExperienceData
from a0.model_utils import board_to_input, get_legal_move_mask_from_state
from a0.utils.states import convert_experience_list_to_dataset, get_gtd_from_states
from a0.eval.generate_datasets import save_dataset
from a0.utils.misc import get_baseline_accuracy, get_branching_factor
from a0.eval3.generate_datasets import get_nd_and_nt_datasets_from_state_list
from a0.eval3.collectors.base import Collector, GameInfo, TurnInfo, GameProgressCollector
from a0.eval3.collectors.bpp_and_bppma import BPPCollector

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

    logger.info(f"Loaded training metrics for {len(train_datas)} iterations.")

    if train_datas:
        # plot_model_performance writes to "{plot_dir}/training_plots/full_a0.png"
        # but doesn't create the training_plots/ subdir itself, so make it here.
        os.makedirs(f"{config.plot_dir}/training_plots", exist_ok=True)
        logger.info("Plotting full a0 training performance...")
        plot_model_performance("training_plots/full_a0", train_datas)
    else:
        logger.warning("No iteration_stats files found; skipping training performance plot.")

    save_series(series, f"{config.eval_dir}/training_metrics.pkl")
    logger.info(f"Saved training_metrics.pkl ({len(series.x)} points).")

def get_and_save_dataset_diagnostics_distributions() -> None:
    '''
    Loads per-iteration dataset diagnostics (pre/post-balance value targets) saved
    during training and stores them as DistributionSeries — one for the pre-balance
    distribution and one for the post-balance distribution.
    → dataset_pre_balance_distributions.pkl
    → dataset_post_balance_distributions.pkl
    '''
    logger.info("Saving dataset diagnostics distributions...")
    pre = DistributionSeries(name="dataset_values_pre")
    post = DistributionSeries(name="dataset_values_post")

    found_any = False
    for iteration, diag in dataset_diagnostics_generator(config.training_dir, config.training_iterations):
        found_any = True
        pre.x.append(iteration)
        pre.trials[0].append(diag['dataset_values_pre'])
        post.x.append(iteration)
        post.trials[0].append(diag['dataset_values_post'])

    if not found_any:
        logger.error("No dataset diagnostics found; skipping distribution series save.")
        return

    logger.info(f"Loaded dataset diagnostics for {len(pre.x)} iterations.")
    save_distribution_series(pre, f"{config.eval_dir}/dataset_pre_balance_distributions.pkl")
    save_distribution_series(post, f"{config.eval_dir}/dataset_post_balance_distributions.pkl")
    logger.info("Saved dataset_pre_balance_distributions.pkl and dataset_post_balance_distributions.pkl.")

def traverse_game_data_with_collectors(collectors: list[Collector]) -> None:
    '''
    Walks all game data once, computing per-turn facts and handing them
    to each collector.
    '''
    logger.info(f"traverse_game_data_with_collectors: {len(collectors)} collectors")
    for c in collectors:
        logger.info(f"  - {type(c).__name__}")

    gt = GroundTruth()
    gd_gen = game_data_generator(config.training_dir, config.training_iterations)

    total_iterations = 0
    total_games = 0
    total_turns = 0
    for i, game_data_list in tqdm(gd_gen, desc="Iterations"):
        total_iterations += 1
        for game_data in game_data_list:
            total_games += 1
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
                total_turns += 1
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
                    alternative_value_target=turn.alternative_value_target,
                    alternative_policy_target=turn.alternative_policy_target,
                )
                for c in collectors:
                    c.on_turn(turn_info)

        for c in collectors:
            c.on_iteration_end(i)

    logger.info(f"traverse_game_data_with_collectors: traversal done — {total_iterations} iterations, {total_games} games, {total_turns} turns")
    logger.info("Finalizing collectors...")
    for c in collectors:
        logger.info(f"  finalizing {type(c).__name__}...")
        c.finalize()
    logger.info("All collectors finalized.")

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
        logger.info(f"GameStatsCollector: saved gamedata_stats.pkl ({len(self.series.x)} iterations).")

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
        logger.info(
            f"AccuracyCollector '{self._name}': saved {self._name}_acc.pkl + {self._name}_overall_acc.pkl. "
            f"Overall: value_acc={self._total_correct_value / t if t else 0:.2%} "
            f"value_acc_nd={self._total_correct_value_nd / t_nd if t_nd else 0:.2%} "
            f"policy_acc={self._total_correct_policy / t if t else 0:.2%} "
            f"policy_acc_nt={self._total_correct_policy_nt / t_nt if t_nt else 0:.2%} "
            f"(n={t}, n_nd={t_nd}, n_nt={t_nt})"
        )

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
        logger.info(
            f"BiasCollector '{self._name}': saved {self._name}_bias.pkl + {self._name}_overall_bias.pkl. "
            f"Overall: win={self._total_wins / t if t else 0:.2%} "
            f"loss={self._total_losses / t if t else 0:.2%} "
            f"draw={self._total_draws / t if t else 0:.2%} "
            f"(n={t})"
        )

class ExperiencedDatasetCollector(Collector):
    '''
    Collects states with their value/policy into a Dataset.
    Keeps at most `n_per_iteration` random samples per iteration to bound memory,
    then selects at most `n_total` random samples in finalize to control dataset size.
    → {name}_dataset.pkl
    '''

    def __init__(
        self,
        name: str = "gamedata",
        batch_size: int = 256,
        n_per_iteration: int | None = None,
        n_total: int | None = None,
        get_outcome: Callable[[TurnInfo], float] = lambda ti: ti.experienced_outcome,
        get_policy: Callable[[TurnInfo], NDArray[np.float32]] = lambda ti: ti.experienced_policy,
    ):
        self._name = name
        self._batch_size = batch_size
        self._n_per_iteration = n_per_iteration
        self._n_total = n_total
        self._get_outcome = get_outcome
        self._get_policy = get_policy
        self._experiences: list[ExperienceData] = []
        self._iteration_buffer: list[ExperienceData] = []

    def on_game(self, gi: GameInfo) -> None:
        pass

    def on_turn(self, ti: TurnInfo) -> None:
        board_input = board_to_input(ti.board)  # (1, board_size, board_size, 2)
        value = self._get_outcome(ti)
        policy = self._get_policy(ti)
        mask = get_legal_move_mask_from_state(ti.board, for_model=True)
        self._iteration_buffer.append(ExperienceData(board_input, value, policy, mask))

    def on_iteration_end(self, iteration: int) -> None:
        buf = self._iteration_buffer
        if self._n_per_iteration is not None and len(buf) > self._n_per_iteration:
            indices = random.sample(range(len(buf)), self._n_per_iteration)
            buf = [buf[i] for i in indices]
        self._experiences.extend(buf)
        self._iteration_buffer = []

    def finalize(self) -> None:
        exps = self._experiences
        if self._n_total is not None and len(exps) > self._n_total:
            indices = random.sample(range(len(exps)), self._n_total)
            exps = [exps[i] for i in indices]
        
        dataset = convert_experience_list_to_dataset(exps, self._batch_size)

        save_dataset(f"{self._name}", dataset)

        logger.info(f"ExperiencedDatasetCollector '{self._name}': saved {len(dataset)} samples to {self._name}.pkl")

class GameProgressMetaCollector(Collector):
    '''
    Routes turns to game-progress-bucket-aware sub-collectors.
    Divides game progress (0-99) into n_buckets equal ranges.
    '''

    def __init__(self, n_buckets: int, collectors: list[GameProgressCollector]):
        self._n_buckets = n_buckets
        self._bucket_size = 100 / n_buckets
        self._collectors = collectors
        bounds = self.bucket_upper_bounds(n_buckets)
        for c in self._collectors:
            c.init_buckets(bounds)

    @staticmethod
    def bucket_upper_bounds(n_buckets: int) -> list[int]:
        size = 100 / n_buckets
        return [int((b + 1) * size) for b in range(n_buckets)]

    def _get_bucket_upper(self, progress: int) -> int:
        bucket_idx = min(int(progress / self._bucket_size), self._n_buckets - 1)
        return int((bucket_idx + 1) * self._bucket_size)

    def on_game(self, gi: GameInfo) -> None:
        pass

    def on_turn(self, ti: TurnInfo) -> None:
        bucket = self._get_bucket_upper(ti.progress)
        for c in self._collectors:
            c.on_turn(ti, bucket)

    def on_iteration_end(self, iteration: int) -> None:
        for c in self._collectors:
            c.on_iteration_end(iteration)

    def finalize(self) -> None:
        for c in self._collectors:
            c.finalize()

class StateCountProgressCollector(GameProgressCollector):
    '''Counts the number of states in each game progress bucket. → {name}_progress_count.pkl'''

    def __init__(self, name: str = "gamedata"):
        self._name = name
        self._buckets: list[int] = []
        self._counts: dict[int, int] = {}
        self._unique_states: dict[int, set[Board]] = {}

    def init_buckets(self, bucket_upper_bounds: list[int]) -> None:
        self._buckets = bucket_upper_bounds
        self._counts = {b: 0 for b in bucket_upper_bounds}
        self._unique_states = {b: set() for b in bucket_upper_bounds}

    def on_turn(self, ti: TurnInfo, bucket: int) -> None:
        self._counts[bucket] += 1
        self._unique_states[bucket].add(ti.board)

    def on_iteration_end(self, iteration: int) -> None:
        pass

    def finalize(self) -> None:
        series = Series(["Count", "Unique"])
        for b in self._buckets:
            series.x.append(b)
            series.ys["Count"].append(self._counts[b])
            series.ys["Unique"].append(len(self._unique_states[b]))
        save_series(series, f"{config.eval_dir}/{self._name}_{len(self._buckets)}_progress_count.pkl")
        logger.info(
            f"StateCountProgressCollector '{self._name}': saved {self._name}_{len(self._buckets)}_progress_count.pkl. "
            f"per-bucket counts: {[self._counts[b] for b in self._buckets]} "
            f"per-bucket unique: {[len(self._unique_states[b]) for b in self._buckets]}"
        )

class AccuracyProgressCollector(GameProgressCollector):
    '''
    Collects value/policy accuracy per game progress bucket.
    → {name}_{n_buckets}_progress_acc.pkl
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
        self._buckets: list[int] = []
        self._stats: dict[int, dict[str, int | float]] = {}

    @staticmethod
    def _empty_stats() -> dict[str, int | float]:
        return {
            'total': 0, 'total_nd': 0, 'total_nt': 0,
            'correct_value': 0, 'correct_value_nd': 0,
            'correct_policy': 0, 'pm_policy': 0.0,
            'correct_policy_nt': 0, 'pm_policy_nt': 0.0,
        }

    def init_buckets(self, bucket_upper_bounds: list[int]) -> None:
        self._buckets = bucket_upper_bounds
        self._stats = {b: self._empty_stats() for b in bucket_upper_bounds}

    def on_turn(self, ti: TurnInfo, bucket: int) -> None:
        outcome = self._get_outcome(ti)
        policy = self._get_policy(ti)
        s = self._stats[bucket]

        value_correct = (ti.gt_outcome == outcome)
        if value_correct:
            s['correct_value'] += 1

        is_draw = (outcome == 0.0)
        if not is_draw:
            s['total_nd'] += 1
            if value_correct:
                s['correct_value_nd'] += 1

        pm = policy_probability_mass_function(policy, ti.gt_policy)
        policy_correct = policy_accuracy_function(policy, ti.gt_policy)
        s['pm_policy'] += pm
        if policy_correct:
            s['correct_policy'] += 1

        if not ti.is_trivial:
            s['total_nt'] += 1
            s['pm_policy_nt'] += pm
            if policy_correct:
                s['correct_policy_nt'] += 1

        s['total'] += 1

    def on_iteration_end(self, iteration: int) -> None:
        pass

    def finalize(self) -> None:
        series = Series([
            "Value Accuracy", "Value Accuracy ND",
            "Policy Accuracy", "Policy PM",
            "Policy Accuracy NT", "Policy PM NT",
        ])
        for b in self._buckets:
            s = self._stats[b]
            t, t_nd, t_nt = s['total'], s['total_nd'], s['total_nt']
            series.x.append(b)
            series.ys["Value Accuracy"].append(s['correct_value'] / t if t else 0)
            series.ys["Value Accuracy ND"].append(s['correct_value_nd'] / t_nd if t_nd else 0)
            series.ys["Policy Accuracy"].append(s['correct_policy'] / t if t else 0)
            series.ys["Policy PM"].append(s['pm_policy'] / t if t else 0)
            series.ys["Policy Accuracy NT"].append(s['correct_policy_nt'] / t_nt if t_nt else 0)
            series.ys["Policy PM NT"].append(s['pm_policy_nt'] / t_nt if t_nt else 0)
        save_series(series, f"{config.eval_dir}/{self._name}_{len(self._buckets)}_progress_acc.pkl")
        logger.info(
            f"AccuracyProgressCollector '{self._name}': saved "
            f"{self._name}_{len(self._buckets)}_progress_acc.pkl "
            f"(n per bucket: {[self._stats[b]['total'] for b in self._buckets]})"
        )

class BoardFunctionProgressCollector(GameProgressCollector):
    '''
    Collects boards per game progress bucket, then applies functions
    to each bucket's board list in finalize. Each function returns a
    dict[str, float] mapping series labels to values.
    → {name}_progress_{fn_name}.pkl per function

    If `n` is given, each bucket is downsampled to at most `n` boards before
    being passed to the per-bucket functions. The same sample is reused across
    all functions in a single finalize call so their outputs are comparable.
    `finalize_functions` always receive the full (un-sampled) bucket map — they
    handle their own sampling if they need it.
    '''

    def __init__(
        self,
        name: str,
        functions: dict[str, Callable[[list[Board]], dict[str, float]]],
        finalize_functions: list[Callable[[dict[int, list[Board]], int], None]] | None = None,
        n: int | None = None,
    ):
        self._name = name
        self._functions = functions
        self._finalize_functions = finalize_functions
        self._n = n
        self._buckets: list[int] = []
        self._boards: dict[int, list[Board]] = {}

    def init_buckets(self, bucket_upper_bounds: list[int]) -> None:
        self._buckets = bucket_upper_bounds
        self._boards = {b: [] for b in bucket_upper_bounds}

    def on_turn(self, ti: TurnInfo, bucket: int) -> None:
        self._boards[bucket].append(ti.board)

    def _sampled_boards(self) -> dict[int, list[Board]]:
        '''
        Returns a per-bucket sample of at most self._n boards for the
        per-bucket functions. If self._n is None, returns the full lists.
        '''
        if self._n is None:
            return self._boards
        sampled: dict[int, list[Board]] = {}
        for b, boards in self._boards.items():
            if len(boards) > self._n:
                sampled[b] = random.sample(boards, self._n)
                logger.info(f"BoardFunctionProgressCollector '{self._name}': sampled bucket {b} down to {self._n}/{len(boards)} boards")
            else:
                sampled[b] = boards
        return sampled

    def on_iteration_end(self, iteration: int) -> None:
        pass

    def finalize(self) -> None:
        logger.info(f"BoardFunctionProgressCollector '{self._name}': finalizing ({len(self._functions)} functions, {len(self._buckets)} buckets)...")
        sampled = self._sampled_boards()

        empty: dict[str, float] = {}
        for fn_name, fn in self._functions.items():
            logger.info(f"BoardFunctionProgressCollector '{self._name}': running '{fn_name}' on {len(self._buckets)} buckets...")
            results: dict[int, dict[str, float]] = {}
            for b in self._buckets:
                boards = sampled[b]
                results[b] = fn(boards) if boards else empty
                logger.info(f"  bucket {b}: {len(boards)} boards → {results[b]}")
            labels = list(next((r for r in results.values() if r), empty).keys())
            series = Series(labels)
            for b in self._buckets:
                series.x.append(b)
                for label in labels:
                    series.ys[label].append(results[b].get(label, 0))
            save_series(series, f"{config.eval_dir}/{self._name}_progress_{fn_name}.pkl")

        for fn in self._finalize_functions or []:
            logger.info(f"BoardFunctionProgressCollector '{self._name}': running finalize function {getattr(fn, '__name__', repr(fn))}...")
            fn(self._boards, len(self._buckets))

def save_dataset_dict(dataset_dict: dict[int, Dataset], name: str) -> None:
    output_path = f"{config.dataset_out_dir}/{name}.pkl"
    with open(output_path, 'wb') as file:
        pickle.dump(dataset_dict, file)
    logger.info(f"Saved {name} dataset dict to {output_path}.")

def convert_and_save_state_buckets_to_datasets(state_buckets: dict[int, list[Board]], n_buckets: int, gt: GroundTruth, n: int, batch_size: int) -> None:
    logger.info(f"convert_and_save_state_buckets_to_datasets: {n_buckets} buckets, target n={n} per bucket")
    datasets_nd: dict[int, Dataset] = {}
    datasets_nt: dict[int, Dataset] = {}
    for bucket, boards in state_buckets.items():
        logger.info(f"  bucket {bucket}: building nd+nt datasets from {len(boards)} boards...")
        nd_dataset, nt_dataset = get_nd_and_nt_datasets_from_state_list(boards, gt, n, batch_size)
        datasets_nd[bucket] = nd_dataset
        datasets_nt[bucket] = nt_dataset
        logger.info(f"  bucket {bucket}: nd={len(nd_dataset)}, nt={len(nt_dataset)}")
    save_dataset_dict(datasets_nd, f"game_progress_{n_buckets}_nd")
    save_dataset_dict(datasets_nt, f"game_progress_{n_buckets}_nt")

def baseline_accuracy_fn(boards: list[Board], gt: GroundTruth) -> dict[str, float]:
    v_acc, p_acc, v_acc_nd, p_acc_nt = get_baseline_accuracy(boards, gt)
    return {
        "Value Accuracy": v_acc,
        "Policy Accuracy": p_acc,
        "Value Accuracy ND": v_acc_nd,
        "Policy Accuracy NT": p_acc_nt,
    }

def get_branching_factor_fn(boards: list[Board], gt: GroundTruth) -> dict[str, float]:
    bf = get_branching_factor(boards, gt)
    return {
        "Branching Factor": bf,
    }

def run_collectors(gt: GroundTruth) -> None:
    '''
    Runs all training data collectors in a single pass over the game data.
    Generates the following data:
    - gamedata_stats.pkl: per-iteration game outcome stats
    - overall and per iteration accuracy and bias files for
        both experienced and alternative targets (x8)
    - per-iteration and overall BPP/BPPMA files (x4)
    - dataset of states with the experienced value/policy (or alt targets)
    - gamedata_{n_buckets}_progress_count.pkl:
        count of states in each game progress bucket
    - gamedata_{n_buckets}_progress_acc.pkl:
        accuracy vs ground truth in each game progress bucket (experienced and alt targets)
    - gamedata_{n_buckets}_progress_{fn_name}.pkl:
        - baseline accuracy
        - branching factor
        - game_progress_{n_buckets}_nd.pkl (and nt)
    '''
    logger.info("run_collectors: building collector list...")

    alt_outcome: Callable[[TurnInfo], float] = lambda ti: float(np.sign(ti.alternative_value_target)) if ti.alternative_value_target is not None else 0.0
    # Raw continuous version for the saved Dataset, so value_loss against the
    # stored labels is meaningful (e.g. L2 to the actual TD-lambda target, not ±1).
    alt_outcome_raw: Callable[[TurnInfo], float] = lambda ti: float(ti.alternative_value_target) if ti.alternative_value_target is not None else 0.0

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
        ),
        BPPCollector(gt=gt),
        ExperiencedDatasetCollector(
            name="experienced_dataset",
            batch_size=256,
            n_per_iteration=500,
            n_total=2000
        ),
        ExperiencedDatasetCollector(
            name="alt_targets_dataset",
            batch_size=256,
            n_per_iteration=500,
            n_total=2000,
            get_outcome=alt_outcome_raw,
        ),
        GameProgressMetaCollector(
            n_buckets=10,
            collectors=[
                StateCountProgressCollector(),
                AccuracyProgressCollector(name="experienced"),
                AccuracyProgressCollector(name="alt_targets", get_outcome=alt_outcome),
                BoardFunctionProgressCollector(
                    name="gamedata",
                    functions={
                        "Baseline Accuracy": lambda boards: baseline_accuracy_fn(boards, gt),
                        "Branching Factor": lambda boards: get_branching_factor_fn(boards, gt)
                    },
                    finalize_functions=[
                        lambda boards, n_buckets: convert_and_save_state_buckets_to_datasets(boards, n_buckets, gt, n=1000, batch_size=256)
                    ],
                    n=2000,
                ),
            ]
        ),
    ]
    logger.info(f"run_collectors: built {len(collectors)} collectors, starting traversal...")
    traverse_game_data_with_collectors(collectors)

    logger.info("run_collectors: training data analyses completed.")

def main():
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name='training_data'
    )

    logger.info("=" * 60)
    logger.info("training_data.py: starting all training-data analyses")
    logger.info(f"  training_dir       = {config.training_dir}")
    logger.info(f"  eval_dir           = {config.eval_dir}")
    logger.info(f"  dataset_out_dir    = {config.dataset_out_dir}")
    logger.info(f"  training_iterations = {config.training_iterations}")
    logger.info("=" * 60)

    gt = GroundTruth()

    logger.info("[1/3] avg training metrics per iteration")
    get_and_save_avg_training_metrics_per_iteration()

    logger.info("[2/3] dataset diagnostics distributions")
    get_and_save_dataset_diagnostics_distributions()

    logger.info("[3/3] traversal-based collectors")
    run_collectors(gt)

    logger.info("training_data.py: all analyses complete")

if __name__ == "__main__":
    main()
