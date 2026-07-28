import os
import gc
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import json
import math
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
from a0.mcts.rollout_strategies import EvaluatorType, PolicyType
from a0.players.mcts_rollout import MCTSRolloutPlayer
from a0.model import load_model
from a0.utils.plotting import Series, save_series, load_series

from config import config

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)


NUM_GAMES = config.player_eval_num_games  # games/side per checkpoint for the win-rate curve
BASELINE_MCTS_ITERATIONS = 512
BASELINE_EPSILON = 0.1   # for the BEST rollout policy used by the baseline
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
            'board': td.board.board_view().split('\n'),
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
    The first log_games games are appended to a pretty-printed JSON array at game_log_path (if provided).
    '''
    p1, p2 = (focal, opponent) if focal_first else (opponent, focal)
    p1_serialized = dill.dumps(p1)
    p2_serialized = dill.dumps(p2)

    logger.info(f"Using {config.num_workers} workers; running {num_games} games (focal_first={focal_first}).")

    stats = MatchupStats()
    # Reuse each worker for several games to amortize per-process JAX/CUDA init,
    # but recycle periodically so per-worker memory stays bounded (de-risk showed
    # ~38 GB stable at this reuse level; a fresh process per game was the old =1).
    with concurrent.futures.ProcessPoolExecutor(max_workers=config.num_workers, max_tasks_per_child=8) as executor:
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
                    records = []
                    if os.path.exists(game_log_path):
                        with open(game_log_path) as f:
                            records = json.load(f)
                    records.append(record)
                    with open(game_log_path, 'w') as f:
                        json.dump(records, f, indent=2)
    return stats


def available_checkpoint_iters() -> list[int]:
    '''Iteration numbers i for which model_<i>.pkl exists in training_dir, ascending.'''
    d = config.training_dir
    return [
        i for i in range(config.training_iterations + 1)
        if os.path.exists(f"{d}/model_{i}.pkl")
    ]


def iters_to_evaluate(available: list[int], already_done: set[int]) -> list[int]:
    '''Pick which checkpoints to evaluate this run.

    Honors config.player_eval_stride (evaluate every Nth checkpoint), but always
    includes iteration 0 and the latest available checkpoint. Anything already
    present in the results series (already_done) is skipped so extending a run
    doesn't re-evaluate from 0.
    '''
    if not available:
        return []
    stride = max(1, int(getattr(config, "player_eval_stride", 1)))
    selected = {i for i in available if i % stride == 0}
    selected.add(available[0])   # iteration 0 (baseline / freshly-initialized model)
    selected.add(available[-1])  # the latest checkpoint
    return sorted(i for i in selected if i not in already_done)


def build_a0_player(model, mcts_samples: int) -> A0Player:
    '''Wrap a loaded model in an A0Player using the shared eval settings, at the
    given MCTS search budget. Shared by the win-rate curve (a0.eval.player) and
    the plateau sweep (a0.eval.player_sweep) so both build identical players.'''
    return A0Player(
        config.board_size,
        config.num_pieces,
        model,
        exploit=True,
        mcts_samples=mcts_samples,
        no_reverse_moves=not config.backwards_moves,
        no_illegal_moves=not config.illegal_moves,
        no_side_moves=not config.sideways_moves,
        rollout_type=config.rollout_type,
        rollout_depth=config.rollout_depth,
        policy_type=config.policy_type,
        epsilon=config.epsilon,
        dirichlet_epsilon=config.dirichlet_epsilon,
    )


def get_trained_players(iters: list[int]):
    '''Yield (iteration, A0Player) one checkpoint at a time.

    Loading lazily (instead of building the whole list up front) keeps the main
    process holding at most one model on the GPU at once. Preloading every
    checkpoint here exhausts GPU memory when combined with the per-worker CUDA
    contexts spawned during matchups, so this must be consumed lazily and each
    player released before advancing (see evaluate_players). Missing checkpoint
    files are skipped.'''
    for i in iters:
        model_path = f"{config.training_dir}/model_{i}.pkl"
        if not os.path.exists(model_path):
            logger.warning(f"model {i} not found at {model_path}; skipping.")
            continue
        model = load_model(model_path)
        yield (i, build_a0_player(model, config.player_eval_mcts_samples))


def _baseline_c() -> float:
    '''Anchor c for the DIST baseline, scaled to the current config: half-range / √2,
    where half-range = num_pieces * 2 * (board_size - 1). See the c-calibration
    script (scripts/2026-05-07_rollout_strategies_c_calibration.py) for the
    derivation.'''
    half_range = config.num_pieces * 2 * (config.board_size - 1)
    return half_range / math.sqrt(2)


def make_baseline() -> MCTSRolloutPlayer:
    return MCTSRolloutPlayer(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        no_reverse_moves=not config.backwards_moves,
        no_illegal_moves=True,
        no_side_moves=not config.sideways_moves,
        mcts_iterations=BASELINE_MCTS_ITERATIONS,
        rollout_depth=8,
        evaluator=EvaluatorType.DIST,
        policy=PolicyType.BEST,
        policy_epsilon=BASELINE_EPSILON,
        c=_baseline_c() * 0.25,
    )


def _bfs_path() -> str:
    '''Default location of the single-agent BFS depth array for the current config
    (scripts/2026-05-29_single_agent_bfs.py output, named bfs_db/bfs_<board_size>_<num_pieces>.npy).'''
    return f"{config.input_dir}bfs_db/bfs_{config.board_size}_{config.num_pieces}.npy"


def make_bfs_baseline(bfs_path: str | None = None) -> MCTSRolloutPlayer:
    '''Like make_baseline, but uses the exact single-agent BFS distance-to-goal (DB) as the
    evaluator instead of the Manhattan-sum DIST. c is anchored to the array's true value
    range (max BFS depth) rather than DIST's analytic half-range.'''
    bfs = np.load(bfs_path or _bfs_path())
    max_depth = int(bfs[bfs >= 0].max())
    return MCTSRolloutPlayer(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        no_reverse_moves=not config.backwards_moves,
        no_illegal_moves=True,
        no_side_moves=not config.sideways_moves,
        mcts_iterations=BASELINE_MCTS_ITERATIONS,
        rollout_depth=8,
        evaluator=EvaluatorType.DB,
        policy=PolicyType.BEST,
        policy_epsilon=BASELINE_EPSILON,
        c=(max_depth / math.sqrt(2)) * 0.25,
        bfs=bfs,
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
    existing_series: Series | None = None,
) -> Series:
    '''
    Evaluate a list of players against a fixed opponent.
    Each player produces one x-point; stats are stored per side (p1/p2).
    Saves and returns a Series with x = player indices and ys keyed {stat}_{side}.
    If log_games > 0, the first log_games games of each matchup are appended to
    a pretty-printed JSON array at game_log_path.
    If existing_series is given, new points are appended to it (rather than
    starting fresh) and the series is re-sorted by x before returning — so a run
    can be extended without re-evaluating checkpoints already present.
    '''
    y_keys = [f'{stat}_{side}' for stat in _STAT_NAMES for side in ('p1', 'p2')]
    if existing_series is not None:
        series = existing_series
        for k in y_keys:
            series.ys.setdefault(k, [])
    else:
        series = Series(ys=y_keys)

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

        series.x.append(i)
        for stat, value in _matchup_stats_dict(s1).items():
            series.ys[f'{stat}_p1'].append(value)
        for stat, value in _matchup_stats_dict(s2).items():
            series.ys[f'{stat}_p2'].append(value)
        save_series(series, output_path)

        # Release this checkpoint's model (and its GPU buffers) before the
        # generator loads the next one, so the main process holds at most one
        # model at a time and leaves room for the worker CUDA contexts.
        del player
        gc.collect()

    # Keep the series ordered by checkpoint index. Appends are usually already
    # ascending, but re-sorting keeps things correct even when extending a run
    # whose stride changed (so newly-added lower iterations slot into place).
    if series.x:
        order = sorted(range(len(series.x)), key=lambda k: series.x[k])
        series.x = [series.x[k] for k in order]
        for key in series.ys:
            series.ys[key] = [series.ys[key][k] for k in order]
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
    If log_games > 0, the first log_games games of each matchup are appended to
    a pretty-printed JSON array at game_log_path.
    '''
    series = Series()
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
            series.ys[f'{name}_{stat}_p1'] = [value]
        for stat, value in _matchup_stats_dict(s2).items():
            series.ys[f'{name}_{stat}_p2'] = [value]
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
        game_log_path=f"{config.eval_dir}/player_game_logs_{timestamp}.json",
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
        game_log_path=f"{config.eval_dir}/reference_game_logs_{timestamp}.json",
    )


def main() -> None:
    setup_logging(level=20, log_dir=config.log_dir, process_name='player_evaluation')

    if not config.do_player_eval:
        logger.info("config.do_player_eval=False; skipping player evaluation.")
        return

    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    baseline = make_baseline()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Resume-aware selection: load any prior results, then evaluate only the
    # checkpoints we haven't scored yet (honoring player_eval_stride). This lets
    # an extended run pick up new checkpoints instead of restarting at 0.
    player_output_path = f"{config.eval_dir}/player_evaluation_results.pkl"
    existing = load_series(player_output_path, optional=True)
    already_done = set(existing.x) if existing is not None else set()
    available = available_checkpoint_iters()
    iters = iters_to_evaluate(available, already_done)
    logger.info(
        f"Player eval (stride={getattr(config, 'player_eval_stride', 1)}): "
        f"available={available}, already_done={sorted(already_done)}, to_evaluate={iters}"
    )

    if iters:
        trained_players = get_trained_players(iters)
        evaluate_players(
            players=trained_players,
            opponent=baseline,
            num_games=NUM_GAMES,
            output_path=player_output_path,
            log_games=1,
            game_log_path=f"{config.eval_dir}/player_game_logs_{timestamp}.json",
            existing_series=existing,
        )
    else:
        logger.info("No new checkpoints to evaluate; leaving existing player results as-is.")

    # References (random / mcts_rollout) don't change with training, so compute
    # them once and reuse across resumed/extended runs.
    reference_output_path = f"{config.eval_dir}/reference_evaluation_results.pkl"
    if load_series(reference_output_path, optional=True) is not None:
        logger.info("Reference evaluation already present; skipping.")
        return

    evaluate_references(
        references={
            'random': RandomPlayer(),
            'mcts_rollout': make_baseline(),
        },
        opponent=baseline,
        num_games=NUM_GAMES,
        output_path=reference_output_path,
        log_games=1,
        game_log_path=f"{config.eval_dir}/reference_game_logs_{timestamp}.json",
    )


if __name__ == "__main__":
    #mcts_test()
    main()
