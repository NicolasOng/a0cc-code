from a0.game import PlayerClass
from cc.core import Board, Game, Player
from cc.ground_truth import GroundTruth
from a0.model_utils import Policy
from scripts.inspect_training_data import print_policy
from a0.players.model import ModelPlayer
from a0.model import load_model
from a0.eval.dataset_evaluation import policy_accuracy_function
from a0.players.random import RandomPlayer
from a0.eval.generate_datasets import get_unique_boards_from_training_data

import numpy as np
import random

from config import config

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def get_random_board(gt: GroundTruth, cc: Game) -> Board:
    while True:
        max_rank = gt.get_max_rank()
        rank = random.randint(0, max_rank - 1)
        #rank = 2175897167
        random_board = gt.unrank(rank)
        if not gt.is_trivial(random_board) and not cc.get_done(random_board):
            break
    #print(f"Generated random board with rank {rank}")
    return random_board

def remove_trivial_boards(boards: list[Board], gt: GroundTruth, cc: Game) -> list[Board]:
    non_trivial_boards: list[Board] = []
    for board in boards:
        if not gt.is_trivial(board) and not cc.get_done(board):
            non_trivial_boards.append(board)
    return non_trivial_boards

def inspect_player_policy(player: PlayerClass, board: Board, gt: GroundTruth, cc: Game, rotate_policy: bool = False) -> bool:
    '''
    Player should output a policy with the select_move function.
    The policy can be for the model or not. If so, it should be rotated.
    '''
    legal_moves = cc.generate_moves_for_given_board(board)
    move, player_data = player.select_move(board, legal_moves)

    # get the ground truth policy for the board
    gt_policy = Policy(config.board_size)
    gt_numpy_policy = np.array(gt.get_1ply_policy_outcomes_list(board, for_model=False))
    gt_policy.set_logits(gt_numpy_policy, rotate_180=False)
    #gt_move_probs = gt_policy.get_move_probabilities(legal_moves)

    p = Policy(config.board_size)
    p.set_logits(np.array(player_data), rotate_180=rotate_policy and board.current_player == Player.PLAYER_O)
    player_numpy_policy = np.array(p.get_policy_list())

    acc = policy_accuracy_function(player_numpy_policy, gt_numpy_policy)

    logger.info(f"Player selected move: {move}")
    print_policy(board, cc, p, gt_policy)

    return acc

def inspect_player_policy_on_random_boards(player: PlayerClass, gt: GroundTruth, cc: Game, num_boards: int = 5) -> None:
    num_total = 0
    num_acc = 0
    for i in range(num_boards):
        board = get_random_board(gt, cc)
        print(f"Inspecting player policy on random board {i+1}:")
        logger.info("\n" + board.board_view())
        logger.info(f"Current player: {board.current_player}")
        acc = inspect_player_policy(player, board, gt, cc, False)
        num_total += 1
        if acc:
            num_acc += 1
        logger.info(f"Player policy accuracy on {num_boards} random boards: {num_acc}/{num_total} = {num_acc/num_total:.2%}")

def inspect_player_policy_on_random_board(player: PlayerClass, gt: GroundTruth, cc: Game):
    board = get_random_board(gt, cc)
    print("Inspecting player policy on random board:")
    logger.info("\n" + board.board_view())
    logger.info(f"Current player: {board.current_player}")
    inspect_player_policy(player, board, gt, cc, False)

def inspect_player_policy_on_given_boards(player: PlayerClass, gt: GroundTruth, cc: Game, boards: list[Board]) -> None:
    num_boards = len(boards)
    num_total = 0
    num_acc = 0
    for i, board in enumerate(boards):
        print(f"Inspecting player policy on given board {i+1}:")
        logger.info("\n" + board.board_view())
        logger.info(f"Current player: {board.current_player}")
        acc = inspect_player_policy(player, board, gt, cc, False)
        num_total += 1
        if acc:
            num_acc += 1
        logger.info(f"Player policy accuracy on {num_boards} given boards: {num_acc}/{num_total} = {num_acc/num_total:.2%}")

def inspect_player_policy_on_seen_boards(player: PlayerClass, gt: GroundTruth, cc: Game, num_boards: int = 5) -> None:
    logger.info("Getting unique boards from training data...")
    seen_boards = list(get_unique_boards_from_training_data())

    unique_seen_total = len(seen_boards)
    logger.info(f"Found {unique_seen_total} unique boards in training data. Removing trivial boards...")
    random.shuffle(seen_boards)
    seen_boards = seen_boards[:num_boards * 3]
    seen_boards = remove_trivial_boards(seen_boards, gt, cc)

    seen_nt_total = len(seen_boards)
    logger.info(f"{seen_nt_total} non-trivial unique boards remain after removing trivial boards ({seen_nt_total/(num_boards * 3):.2%}).")
    logger.info(f"Sampling {num_boards} boards for inspection...")
    random.shuffle(seen_boards)
    seen_boards = seen_boards[:num_boards]

    inspect_player_policy_on_given_boards(player, gt, cc, seen_boards)

def main():
    gt = GroundTruth()
    cc = Game(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        no_reverse_moves=not config.backwards_moves
    )
    model = load_model(config.training_dir + f"model_50.pkl")
    model_player = ModelPlayer(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        model=model
    )
    random_player = RandomPlayer(
        random_percent=1.0
    )
    #inspect_player_policy_on_random_board(player, gt, cc)
    #inspect_player_policy_on_random_boards(random_player, gt, cc, num_boards=1000)
    inspect_player_policy_on_seen_boards(random_player, gt, cc, num_boards=1000)

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="ipp"
    )

    main()
