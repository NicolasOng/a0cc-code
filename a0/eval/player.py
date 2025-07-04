from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

from a0.game import PlayerClass, play
from a0.model import load_model
from a0.players.a0 import A0Player
from a0.players.random import RandomPlayer
from cc.core import Game, Player

from config import config

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def player_evaluation(player1: PlayerClass, player2: PlayerClass, num_games: int) -> tuple[int, int, int]:
    '''
    Evaluate two players by playing a series of games between them.
    Returns the number of wins, losses, and draws for each player.
    '''
    # Initialize counters for wins, losses, and draws (for player 1)
    wins = 0
    losses = 0
    draws = 0
    for game_num in range(num_games):
        # play a game
        print(f"Game {game_num + 1}/{num_games}")
        game_data = play(Game(config.board_size, config.num_pieces, True, False, True), [player1, player2], turn_limit=100)

        # get the winner of the game
        if game_data.winner == Player.PLAYER_X:
            wins += 1
        elif game_data.winner == Player.PLAYER_O:
            losses += 1
        else:
            draws += 1
    
    return wins, losses, draws

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
            player = A0Player(config.board_size, config.num_pieces, model, False)
            players.append(player)
        except Exception as e:
            logger.error(f"Failed to load model or create player {i + 1} at {model_path}: {e}")
    return players

def evaluate_trained_players(num_games: int, player_baseline: PlayerClass, fn: str) -> None:
    '''
    Evaluate all trained players against a baseline player.
    Saves results to a pickle file.
    TODO: can make this parallel
    '''
    # load all trained players
    trained_players = get_trained_players_list()

    logger.info(f"Evaluating {len(trained_players)} trained players against a random player.")

    win_list: list[int] = []
    loss_list: list[int] = []
    draw_list: list[int] = []

    # for each trained player,
    for i, player in enumerate(trained_players):
        logger.info(f"Evaluating player {i + 1} against the baseline player.")

        # evaluate the player against the baseline player
        wins, losses, draws = player_evaluation(player, player_baseline, num_games)
        win_list.append(wins)
        loss_list.append(losses)
        draw_list.append(draws)

        logger.info(f"Player {i + 1} - Wins: {wins}, Losses: {losses}, Draws: {draws}")
    
    # save the results to a file
    results = {
        "wins": win_list,
        "losses": loss_list,
        "draws": draw_list
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
    draws = results['draws']

    plt.figure(figsize=(10, 5))
    plt.plot(wins, label='Wins', color='green')
    plt.plot(losses, label='Losses', color='red')
    plt.plot(draws, label='Draws', color='blue')
    plt.xlabel('Game Number')
    plt.ylabel('Count')
    plt.title('Player Evaluation Results')
    plt.legend()
    plt.savefig(config.plot_dir + '/player_evaluation_results.png')

def main():
    setup_logging(level=20, log_dir=config.log_dir, process_name='player_evaluation')
    random_player = RandomPlayer()
    evaluate_trained_players(100, random_player, f"{config.eval_dir}/player_evaluation_results.pkl")
    graph_player_evaluation_results(f"{config.eval_dir}/player_evaluation_results.pkl")

if __name__ == "__main__":
    main()
