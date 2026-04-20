import multiprocessing
import concurrent.futures
from concurrent.futures import Future, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from enum import Enum

import dill
import numpy as np

from cc.core import Game, Player

from a0.game import PlayerClass, play
from a0.players.a0 import A0Player
from a0.players.random import RandomPlayer
from a0.players.mcts_rollout import MCTSRolloutPlayer
from a0.utils.load_training_data import load_models
from a0.utils.plotting import Series, save_series

from config import config

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)


NUM_GAMES = 100
BASELINE_MCTS_ITERATIONS = 64
TURN_LIMIT = 100


class GameResult(Enum):
    WIN = 0
    LOSS = 1
    DRAW_REPEAT = 2
    DRAW_TIMEOUT = 3


@dataclass
class GameStats:
    '''Stats from a single game, from P1's perspective.'''
    result: GameResult
    num_turns: int
    duration_seconds: float


@dataclass
class MatchupStats:
    '''Aggregated stats across N games of one matchup, from the focal player's perspective.'''
    wins: int = 0
    losses: int = 0
    draws_repeat: int = 0
    draws_timeout: int = 0
    num_turns: list[int] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)

    @property
    def num_games(self) -> int:
        return self.wins + self.losses + self.draws_repeat + self.draws_timeout

    @property
    def expected_value(self) -> float:
        n = self.num_games
        return (self.wins - self.losses) / n if n > 0 else 0.0


def _make_game() -> Game:
    return Game(
        config.board_size,
        config.num_pieces,
        repeats_for_draw=config.repeats_for_draw,
        no_reverse_moves=not config.backwards_moves,
        no_illegal_moves=not config.illegal_moves,
        no_side_moves=not config.sideways_moves
    )


def single_game(player1: PlayerClass, player2: PlayerClass) -> GameStats:
    '''Play one game; return per-game stats from P1's perspective.'''
    game_data = play(_make_game(), [player1, player2], turn_limit=TURN_LIMIT)
    if game_data.winner == Player.PLAYER_X:
        result = GameResult.WIN
    elif game_data.winner == Player.PLAYER_O:
        result = GameResult.LOSS
    elif game_data.ended:
        result = GameResult.DRAW_REPEAT
    else:
        result = GameResult.DRAW_TIMEOUT
    return GameStats(
        result=result,
        num_turns=len(game_data.turn_data),
        duration_seconds=game_data.time,
    )


def _play_single_game(serialized_p1: bytes, serialized_p2: bytes) -> GameStats:
    setup_logging(level=20, log_dir=config.log_dir, process_name='player_evaluation')
    p1 = dill.loads(serialized_p1)
    p2 = dill.loads(serialized_p2)
    return single_game(p1, p2)


def _flip_result(result: GameResult) -> GameResult:
    if result == GameResult.WIN:
        return GameResult.LOSS
    if result == GameResult.LOSS:
        return GameResult.WIN
    return result


def run_matchup(
    focal: PlayerClass,
    opponent: PlayerClass,
    num_games: int,
    focal_first: bool,
) -> MatchupStats:
    '''
    Run num_games of focal vs opponent in parallel; results are reported from focal's perspective.
    If focal_first, focal plays as P1; otherwise focal plays as P2 (W/L are flipped accordingly).
    '''
    p1, p2 = (focal, opponent) if focal_first else (opponent, focal)
    p1_serialized = dill.dumps(p1)
    p2_serialized = dill.dumps(p2)

    logger.info(f"Using {config.num_workers} workers; running {num_games} games (focal_first={focal_first}).")

    stats = MatchupStats()
    with concurrent.futures.ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        futures: list[Future[GameStats]] = [
            executor.submit(_play_single_game, p1_serialized, p2_serialized)
            for _ in range(num_games)
        ]
        num_done = 0
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                futures.remove(fut)
                game_stats = fut.result()
                num_done += 1
                result = game_stats.result if focal_first else _flip_result(game_stats.result)
                logger.info(
                    f"Game {num_done}/{num_games}: {result.name} "
                    f"(turns={game_stats.num_turns}, time={game_stats.duration_seconds:.1f}s)"
                )
                if result == GameResult.WIN:
                    stats.wins += 1
                elif result == GameResult.LOSS:
                    stats.losses += 1
                elif result == GameResult.DRAW_REPEAT:
                    stats.draws_repeat += 1
                elif result == GameResult.DRAW_TIMEOUT:
                    stats.draws_timeout += 1
                stats.num_turns.append(game_stats.num_turns)
                stats.durations.append(game_stats.duration_seconds)
    return stats


def get_trained_players() -> list[tuple[int, A0Player]]:
    '''Load every trained-model checkpoint and wrap it in an A0Player.'''
    return [
        (i, A0Player(
            config.board_size,
            config.num_pieces,
            model,
            exploit=True,
            mcts_samples=config.mcts_samples,
            no_reverse_moves=not config.backwards_moves,
            no_illegal_moves=not config.illegal_moves,
            no_side_moves=not config.sideways_moves,
            rollout_type=config.rollout_type,
            rollout_depth=config.rollout_depth,
            policy_type=config.policy_type,
            epsilon=config.epsilon,
            dirichlet_epsilon=config.dirichlet_epsilon
        ))
        for i, model in load_models(config.training_dir, config.training_iterations)
    ]


def make_baseline() -> MCTSRolloutPlayer:
    return MCTSRolloutPlayer(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        no_reverse_moves=not config.backwards_moves,
        no_illegal_moves=not config.illegal_moves,
        no_side_moves=not config.sideways_moves,
        mcts_iterations=BASELINE_MCTS_ITERATIONS,
    )


_STAT_NAMES = (
    'wins', 'losses', 'draws_repeat', 'draws_timeout', 'num_games',
    'ev', 'mean_turns', 'std_turns', 'mean_duration', 'std_duration',
)


def _matchup_stats_dict(stats: MatchupStats) -> dict[str, float]:
    '''Flatten a MatchupStats into scalar stats keyed by name.'''
    turns = np.asarray(stats.num_turns, dtype=float) if stats.num_turns else np.array([np.nan])
    durs = np.asarray(stats.durations, dtype=float) if stats.durations else np.array([np.nan])
    return {
        'wins': float(stats.wins),
        'losses': float(stats.losses),
        'draws_repeat': float(stats.draws_repeat),
        'draws_timeout': float(stats.draws_timeout),
        'num_games': float(stats.num_games),
        'ev': stats.expected_value,
        'mean_turns': float(np.mean(turns)),
        'std_turns': float(np.std(turns)),
        'mean_duration': float(np.mean(durs)),
        'std_duration': float(np.std(durs)),
    }


def _y_keys() -> list[str]:
    '''All y-keys produced by evaluate_all, in a stable order.'''
    keys = [f'{stat}_{side}' for stat in _STAT_NAMES for side in ('p1', 'p2')]
    keys += [f'random_vs_baseline_{stat}_{side}'
             for stat in _STAT_NAMES for side in ('p1', 'p2')]
    return keys


def evaluate_all(num_games: int, output_path: str) -> Series:
    baseline = make_baseline()
    random_player = RandomPlayer()

    trained = get_trained_players()
    logger.info(f"Loaded {len(trained)} trained players for evaluation.")

    series = Series(ys=_y_keys())
    series.x = [i for i, _ in trained]

    for i, model_player in trained:
        logger.info(f"=== Model iter {i}: as P1 vs baseline ===")
        s1 = run_matchup(model_player, baseline, num_games, focal_first=True)
        logger.info(
            f"  EV={s1.expected_value:+.3f} "
            f"(W={s1.wins} L={s1.losses} Dr={s1.draws_repeat} Dt={s1.draws_timeout})"
        )

        logger.info(f"=== Model iter {i}: as P2 vs baseline ===")
        s2 = run_matchup(model_player, baseline, num_games, focal_first=False)
        logger.info(
            f"  EV={s2.expected_value:+.3f} "
            f"(W={s2.wins} L={s2.losses} Dr={s2.draws_repeat} Dt={s2.draws_timeout})"
        )

        for stat, value in _matchup_stats_dict(s1).items():
            series.ys[f'{stat}_p1'].append(value)
        for stat, value in _matchup_stats_dict(s2).items():
            series.ys[f'{stat}_p2'].append(value)

    logger.info("=== Reference: random as P1 vs baseline ===")
    ref_p1 = run_matchup(random_player, baseline, num_games, focal_first=True)
    logger.info(f"  EV={ref_p1.expected_value:+.3f}")

    logger.info("=== Reference: random as P2 vs baseline ===")
    ref_p2 = run_matchup(random_player, baseline, num_games, focal_first=False)
    logger.info(f"  EV={ref_p2.expected_value:+.3f}")

    # broadcast random-vs-baseline reference values as constants across all
    # iterations so they survive merge_series and render as horizontal lines
    num_iters = len(series.x)
    ref_p1_stats = _matchup_stats_dict(ref_p1)
    ref_p2_stats = _matchup_stats_dict(ref_p2)
    for stat in _STAT_NAMES:
        series.ys[f'random_vs_baseline_{stat}_p1'] = [ref_p1_stats[stat]] * num_iters
        series.ys[f'random_vs_baseline_{stat}_p2'] = [ref_p2_stats[stat]] * num_iters

    save_series(series, output_path)
    return series


def main() -> None:
    setup_logging(level=20, log_dir=config.log_dir, process_name='player_evaluation')
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    out_path = f"{config.eval_dir}/player_evaluation_results.pkl"
    evaluate_all(NUM_GAMES, out_path)


if __name__ == "__main__":
    main()
