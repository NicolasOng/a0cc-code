from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import dill
import os
from concurrent.futures import Future, wait, FIRST_COMPLETED
import concurrent.futures
import multiprocessing

from a0.game import PlayerClass, play
from a0.model import load_model
from a0.players.a0 import A0Player
from a0.players.random import RandomPlayer
from a0.players.mcts_rollout import MCTSRolloutPlayer
from cc.core import Game, Player

from config import config

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

from enum import Enum

class GameResult(Enum):
    WIN = 0
    LOSS = 1
    DRAW_REPEAT = 2
    DRAW_TIMEOUT = 3

def single_game(player1: PlayerClass, player2: PlayerClass) -> GameResult:
    # play a game
    game_data = play(Game(config.board_size, config.num_pieces, True, False, False), [player1, player2], turn_limit=100)
    # get the winner of the game
    if game_data.winner == Player.PLAYER_X:
        return GameResult.WIN
    elif game_data.winner == Player.PLAYER_O:
        return GameResult.LOSS
    elif game_data.ended:
        return GameResult.DRAW_REPEAT
    else:
        return GameResult.DRAW_TIMEOUT

def _play_single_game(serialized_player1: bytes, serialized_player2: bytes) -> GameResult:
    '''
    Helper function to play a single game between two serialized players.
    '''
    player1 = dill.loads(serialized_player1)
    player2 = dill.loads(serialized_player2)
    return single_game(player1, player2)

def player_evaluation(player1: PlayerClass, player2: PlayerClass, num_games: int) -> tuple[int, int, int, int]:
    '''
    Evaluate two players by playing a series of games between them in parallel.
    Returns the number of wins, losses, and draws from the perspective of player1.
    Output: (wins, losses, draws_repeat, draws_timeout)
    '''
    # Initialize counters for wins, losses, and draws (for player 1)
    wins = 0
    losses = 0
    draws_repeat = 0
    draws_timeout = 0
    # serialize the players
    player1_serialized: bytes = dill.dumps(player1)
    player2_serialized: bytes = dill.dumps(player2)
    # use a process pool to play the games in parallel
    num_cores = os.cpu_count() or 4
    logger.info(f"Using {num_cores} cores for playing games.")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # create a list to hold the futures
        futures: list[Future[GameResult]] = []

        # create a function to start a game
        def start_game() -> None:
            future = executor.submit(
                _play_single_game,
                player1_serialized,
                player2_serialized
            )
            futures.append(future)
        
        # start all the games
        for _ in range(num_games):
            start_game()
        
        # process the results as they come in
        num_done = 0
        while futures:
            # when a game (or games) finish(es),
            done, _ = wait(futures, return_when=FIRST_COMPLETED)

            # for each finished game,
            for future in done:
                # remove it from the list of futures
                futures.remove(future)
                # get the result of the game
                result = future.result()
                num_done += 1
                logger.info(f"Game {num_done}/{num_games} finished: {result}")
                # update the counters based on the result
                if result == GameResult.WIN:
                    wins += 1
                elif result == GameResult.LOSS:
                    losses += 1
                elif result == GameResult.DRAW_REPEAT:
                    draws_repeat += 1
                elif result == GameResult.DRAW_TIMEOUT:
                    draws_timeout += 1

    return wins, losses, draws_repeat, draws_timeout

def get_trained_players_list() -> list[A0Player]:
    '''
    Load all trained players from the training directory.
    Returns a list of A0Player instances.
    The training directory and number of iterations are defined in the config.
    '''
    players: list[A0Player] = []
    for i in tqdm(range(config.training_iterations + 1)):
        model_path = f"{config.training_dir}/model_{i}.pkl"
        try:
            model = load_model(model_path)
            player = A0Player(config.board_size, config.num_pieces, model)
            players.append(player)
        except Exception as e:
            logger.error(f"Failed to load model or create player {i + 1} at {model_path}: {e}")
    return players

def evaluate_trained_players(num_games: int, player_baseline: PlayerClass, fn: str) -> None:
    '''
    Evaluate all trained players against a baseline player.
    Saves results to a pickle file.
    '''
    # load all trained players
    trained_players = get_trained_players_list()
    #trained_players = [MCTSRolloutPlayer(config.board_size, config.num_pieces, True, 64), MCTSRolloutPlayer(config.board_size, config.num_pieces, True, 128)]

    logger.info(f"Evaluating {len(trained_players)} trained players against a random player.")

    win_list: list[int] = []
    loss_list: list[int] = []
    draw_repeat_list: list[int] = []
    draw_timeout_list: list[int] = []

    # for each trained player,
    for i, player in enumerate(trained_players):
        logger.info(f"Evaluating player {i + 1} against the baseline player.")

        # evaluate the player against the baseline player
        wins, losses, draws_repeat, draws_timeout = player_evaluation(player, player_baseline, num_games)
        win_list.append(wins)
        loss_list.append(losses)
        draw_repeat_list.append(draws_repeat)
        draw_timeout_list.append(draws_timeout)

        logger.info(f"Player {i + 1} - Wins: {wins}, Losses: {losses}, Draws (Repeat): {draws_repeat}, Draws (Timeout): {draws_timeout}")

    # save the results to a file
    results = {
        "wins": win_list,
        "losses": loss_list,
        "draws_repeat": draw_repeat_list,
        "draws_timeout": draw_timeout_list
    }
    with open(fn, 'wb') as f:
        pickle.dump(results, f)

def graph_player_evaluation_results(fn: str) -> None:
    '''
    Load player evaluation results from a file and plot them.
    '''
    with open(fn, 'rb') as f:
        results = pickle.load(f)
    
    wins = results['wins']
    losses = results['losses']
    draws_repeat = results['draws_repeat']
    draws_timeout = results['draws_timeout']

    plt.figure(figsize=(16, 9))
    plt.plot(wins, label='Wins', color='green')
    plt.plot(losses, label='Losses', color='red')
    plt.plot(draws_repeat, label='Draws (Repeat)', color='blue')
    plt.plot(draws_timeout, label='Draws (Timeout)', color='orange')
    plt.xlabel('Game Number')
    plt.ylabel('Count')
    plt.title('Player Evaluation Results')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(config.plot_dir + '/player_evaluation_results.png')

def main():
    setup_logging(level=20, log_dir=config.log_dir, process_name='player_evaluation')

    # Set the multiprocessing start method to 'spawn' for compatibility with JAX
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    baseline_player = MCTSRolloutPlayer(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        no_reverse_moves=True,
        mcts_iterations=32
    )
    #baseline_player = RandomPlayer()

    evaluate_trained_players(100, baseline_player, f"{config.eval_dir}/player_evaluation_results.pkl")
    graph_player_evaluation_results(f"{config.eval_dir}/player_evaluation_results.pkl")

if __name__ == "__main__":
    main()
