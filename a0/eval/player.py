import json
import multiprocessing
from datetime import datetime
import concurrent.futures
from concurrent.futures import Future, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from enum import Enum

import dill
import numpy as np

from cc.core import Game, Player

from a0.game import PlayerClass, GameData, play
from a0.players.a0 import A0Player
from a0.players.random import RandomPlayer
from a0.players.mcts_rollout import MCTSRolloutPlayer
from a0.utils.load_training_data import load_models
from a0.utils.plotting import Series, save_series

from config import config

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)


NUM_GAMES = 64
BASELINE_MCTS_ITERATIONS = 2048
TURN_LIMIT = 80


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


def _extract_stats(game_data: GameData) -> GameStats:
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


def _build_game_log(game_data: GameData) -> list[dict]:
    return [
        {
            'turn': i,
            'player': 'p1' if i % 2 == 0 else 'p2',
            'board': td.board.board_view(),
            'move': str(td.move),
        }
        for i, td in enumerate(game_data.turn_data)
    ]


def single_game(player1: PlayerClass, player2: PlayerClass) -> GameStats:
    '''Play one game; return per-game stats from P1's perspective.'''
    return _extract_stats(play(_make_game(), [player1, player2], turn_limit=TURN_LIMIT))


def _play_single_game(
    serialized_p1: bytes,
    serialized_p2: bytes,
    log_game: bool = False,
) -> tuple[GameStats, list[dict] | None]:
    setup_logging(level=20, log_dir=config.log_dir, process_name='player_evaluation')
    p1 = dill.loads(serialized_p1)
    p2 = dill.loads(serialized_p2)
    game_data = play(_make_game(), [p1, p2], turn_limit=TURN_LIMIT)
    return _extract_stats(game_data), (_build_game_log(game_data) if log_game else None)


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
    log_games: int = 0,
    game_log_path: str | None = None,
    label: str = '',
) -> MatchupStats:
    '''
    Run num_games of focal vs opponent in parallel; results are reported from focal's perspective.
    If focal_first, focal plays as P1; otherwise focal plays as P2 (W/L are flipped accordingly).
    The first log_games games are logged as JSONL to game_log_path (if provided).
    '''
    p1, p2 = (focal, opponent) if focal_first else (opponent, focal)
    p1_serialized = dill.dumps(p1)
    p2_serialized = dill.dumps(p2)

    logger.info(f"Using {config.num_workers} workers; running {num_games} games (focal_first={focal_first}).")

    stats = MatchupStats()
    with concurrent.futures.ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        futures: list[Future[tuple[GameStats, list[dict] | None]]] = [
            executor.submit(_play_single_game, p1_serialized, p2_serialized, i < log_games)
            for i in range(num_games)
        ]
        num_done = 0
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                futures.remove(fut)
                game_stats, game_log = fut.result()
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
                if game_log is not None and game_log_path is not None:
                    record = {
                        'label': label,
                        'focal_first': focal_first,
                        'result': result.name,
                        'num_turns': game_stats.num_turns,
                        'duration_seconds': game_stats.duration_seconds,
                        'turns': game_log,
                    }
                    with open(game_log_path, 'a') as f:
                        f.write(json.dumps(record) + '\n')
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


def _log_matchup_result(label: str, stats: MatchupStats) -> None:
    logger.info(
        f"  {label}: EV={stats.expected_value:+.3f} "
        f"(W={stats.wins} L={stats.losses} Dr={stats.draws_repeat} Dt={stats.draws_timeout})"
    )


def evaluate_players(
    players: list[tuple[int, PlayerClass]],
    opponent: PlayerClass,
    num_games: int,
    output_path: str,
    log_games: int = 0,
    game_log_path: str | None = None,
) -> Series:
    '''
    Evaluate a list of players against a fixed opponent.
    Each player produces one x-point; stats are stored per side (p1/p2).
    Saves and returns a Series with x = player indices and ys keyed {stat}_{side}.
    If log_games > 0, the first log_games games of each matchup are written as
    JSONL records to game_log_path.
    '''
    y_keys = [f'{stat}_{side}' for stat in _STAT_NAMES for side in ('p1', 'p2')]
    series = Series(ys=y_keys)
    series.x = [i for i, _ in players]

    for i, player in players:
        logger.info(f"=== Player {i}: as P1 vs opponent ===")
        s1 = run_matchup(player, opponent, num_games, focal_first=True,
                         log_games=log_games, game_log_path=game_log_path,
                         label=f'player_{i}_p1')
        _log_matchup_result("P1", s1)

        logger.info(f"=== Player {i}: as P2 vs opponent ===")
        s2 = run_matchup(player, opponent, num_games, focal_first=False,
                         log_games=log_games, game_log_path=game_log_path,
                         label=f'player_{i}_p2')
        _log_matchup_result("P2", s2)

        for stat, value in _matchup_stats_dict(s1).items():
            series.ys[f'{stat}_p1'].append(value)
        for stat, value in _matchup_stats_dict(s2).items():
            series.ys[f'{stat}_p2'].append(value)

    save_series(series, output_path)
    return series


def evaluate_references(
    references: dict[str, PlayerClass],
    opponent: PlayerClass,
    num_games: int,
    output_path: str,
    log_games: int = 0,
    game_log_path: str | None = None,
) -> Series:
    '''
    Evaluate a set of named reference players against a fixed opponent.
    Each reference player produces a single data point at x=0.
    Saves and returns a Series with x=[0] and ys keyed {name}_{stat}_{side}.
    If log_games > 0, the first log_games games of each matchup are written as
    JSONL records to game_log_path.
    '''
    y_keys = [
        f'{name}_{stat}_{side}'
        for name in references
        for stat in _STAT_NAMES
        for side in ('p1', 'p2')
    ]
    series = Series(ys=y_keys)
    series.x = [0]

    for name, player in references.items():
        logger.info(f"=== Reference '{name}': as P1 vs opponent ===")
        s1 = run_matchup(player, opponent, num_games, focal_first=True,
                         log_games=log_games, game_log_path=game_log_path,
                         label=f'{name}_p1')
        _log_matchup_result("P1", s1)

        logger.info(f"=== Reference '{name}': as P2 vs opponent ===")
        s2 = run_matchup(player, opponent, num_games, focal_first=False,
                         log_games=log_games, game_log_path=game_log_path,
                         label=f'{name}_p2')
        _log_matchup_result("P2", s2)

        for stat, value in _matchup_stats_dict(s1).items():
            series.ys[f'{name}_{stat}_p1'].append(value)
        for stat, value in _matchup_stats_dict(s2).items():
            series.ys[f'{name}_{stat}_p2'].append(value)

    save_series(series, output_path)
    return series

def mcts_test() -> None:
    setup_logging(level=20, log_dir=config.log_dir, process_name='player_evaluation')
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    baseline = make_baseline()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    mcts_players = [
        (iters, MCTSRolloutPlayer(
            board_size=config.board_size,
            num_pieces=config.num_pieces,
            no_reverse_moves=not config.backwards_moves,
            no_illegal_moves=not config.illegal_moves,
            no_side_moves=not config.sideways_moves,
            mcts_iterations=iters,
        ))
        for iters in [16, 64, 256]
    ]

    evaluate_players(
        players=mcts_players,
        opponent=baseline,
        num_games=NUM_GAMES,
        output_path=f"{config.eval_dir}/player_evaluation_results.pkl",
        log_games=1,
        game_log_path=f"{config.eval_dir}/player_game_logs_{timestamp}.jsonl",
    )

    evaluate_references(
        references={
            'random': RandomPlayer(),
            'mcts_rollout': make_baseline(),
        },
        opponent=baseline,
        num_games=NUM_GAMES,
        output_path=f"{config.eval_dir}/reference_evaluation_results.pkl",
        log_games=1,
        game_log_path=f"{config.eval_dir}/reference_game_logs_{timestamp}.jsonl",
    )


def main() -> None:
    setup_logging(level=20, log_dir=config.log_dir, process_name='player_evaluation')
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    baseline = make_baseline()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    trained_players = get_trained_players()

    # # remove all players except iteration 4
    # trained_players = [(i, p) for i, p in trained_players if i == 4]
    # print(f"Evaluating {len(trained_players)} trained players: {[i for i, _ in trained_players]}")

    evaluate_players(
        players=trained_players,
        opponent=baseline,
        num_games=NUM_GAMES,
        output_path=f"{config.eval_dir}/player_evaluation_results.pkl",
        log_games=1,
        game_log_path=f"{config.eval_dir}/player_game_logs_{timestamp}.jsonl",
    )

    # exit()

    evaluate_references(
        references={
            'random': RandomPlayer(),
            'mcts_rollout': make_baseline(),
        },
        opponent=baseline,
        num_games=NUM_GAMES,
        output_path=f"{config.eval_dir}/reference_evaluation_results.pkl",
        log_games=1,
        game_log_path=f"{config.eval_dir}/reference_game_logs_{timestamp}.jsonl",
    )


if __name__ == "__main__":
    #mcts_test()
    main()
