import pickle
from tqdm import tqdm

import numpy as np

from a0.dataset import Dataset, TrainingData
from a0.game import GameData
from a0.train.alphazero import board_to_input

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

training_data_path = config.training_dir

def load_game_data() -> list[list[GameData]]:
    '''
    Load game data from the training data path.
    Returns a list of lists of game data.
    Each list corresponds to a single training iteration.
    '''
    # load all the game data objects from config.training_dir
    game_data_lists: list[list[GameData]] = []
    for i in tqdm(range(config.training_iterations)):
        file_path = f"{config.training_dir}/gamedata_{i + 1}.pkl"
        with open(file_path, 'rb') as file:
            data: list[GameData] = pickle.load(file)
            game_data_lists.append(data)
    logger.info(f"Loaded {len(game_data_lists)} gamedata lists from {config.training_dir}.")
    
    return game_data_lists

def load_training_data() -> list[list[TrainingData]]:
    '''
    Load training data from the training data path.
    Returns a list of lists of training data.
    Each list corresponds to a single training iteration.
    '''
    # load all the training data objects from config.training_dir
    training_data_lists: list[list[TrainingData]] = []
    for i in tqdm(range(config.training_iterations)):
        file_path = f"{config.training_dir}/training_set_{i + 1}.pkl"
        with open(file_path, 'rb') as file:
            data: list[TrainingData] = pickle.load(file)
            training_data_lists.append(data)
    logger.info(f"Loaded {len(training_data_lists)} training data lists from {config.training_dir}.")
    return training_data_lists

def check_gd_and_td_equivalence(game_data_lists: list[list[GameData]], training_data_lists: list[list[TrainingData]]) -> bool:
    '''
    Check if the game data and training data are equivalent.
    This is done by comparing the game data and training data for each iteration.

    Game data lists is a list of lists of GameData objects,
    where each inner list holds all the gamedata from a single training iteration.
    Training data lists is a list of lists of TrainingData objects,
    where each inner list holds all the training data objects from a single training iteration.
    '''
    # first check if the number of iterations match
    if len(game_data_lists) != len(training_data_lists):
        logger.error(f"Number of iterations in game data and training data do not match: {len(game_data_lists)} != {len(training_data_lists)}")
        return False
    
    # in each iteration,
    for i, (game_data_list, training_data_list) in enumerate(zip(game_data_lists, training_data_lists)):
        state_no = 0
        # for each game in the iteration,
        for j, game_data in enumerate(game_data_list):
            # for each state in the game,
            for k, game_state in enumerate(game_data.turn_data):
                # get the corresponding training data
                training_data = training_data_list[state_no]
                state_no += 1
                # check if the board states match
                gs_policy = game_state.player_data
                td_policy = training_data.policy
                if not np.array_equal(gs_policy, td_policy):
                    logger.debug(f"Policy nmismatch at iteration {i}, game {j}, turn {k}.")
                    return False
                gs_value = 0 if game_data.winner is None else 1 if game_data.winner == game_state.board.current_player else -1
                td_value = training_data.value
                if gs_value != td_value:
                    logger.error(f"Value mismatch at iteration {i}, game {j}, turn {k}: {gs_value} != {td_value}")
                    return False
                gs_state = board_to_input(game_state.board)
                td_state = training_data.board
                if not np.array_equal(gs_state, td_state):
                    logger.error(f"State mismatch at iteration {i}, game {j}, turn {k}.")
                    return False

    logger.info("Game data and training data equivalence check completed.")
    return True

def check_winners_match(game_data_lists: list[list[GameData]]) -> bool:
    '''
    Check if the winners in the game data match the winners in the game objects.
    This is done by comparing the winner in each GameData object with the winner in the Game object
    and the winner in the last TurnData object.
    Returns True if all winners match, False otherwise.
    '''
    for i, game_data_list in enumerate(game_data_lists):
        for j, game_data in enumerate(game_data_list):
            game = game_data.game
            winner = game_data.winner
            game_winner = game.winner
            board_winner = game.get_winner(game_data.turn_data[-1].board)
            if winner != game_winner:
                logger.error(f"Game winner mismatch at iteration {i}, game {j}: {winner} != {game_winner}")
                return False
            if winner != board_winner:
                logger.error(f"Board winner mismatch at iteration {i}, game {j}: {winner} != {board_winner}")
                return False
    return True

def main():
    setup_logging(level=20, log_dir='logs/', process_name='training_data_evals')
    
    gamedatas = load_game_data()
    training_datas = load_training_data()

    check_winners_match(gamedatas)
    check_gd_and_td_equivalence(gamedatas, training_datas)

if __name__ == "__main__":
    main()
