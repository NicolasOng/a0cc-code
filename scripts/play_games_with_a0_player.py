import jax
from flax import nnx

import numpy as np
from numpy.typing import NDArray

from config import config

from a0.players.a0 import A0Player
from a0.players.human import HumanPlayer
from a0.model import AlphaZeroModel, load_model, save_model
from cc.core import Game, Player
from a0.eval.dataset_evaluation import Series, save_series, policy_accuracy_function, policy_probability_mass_function

from cc.ground_truth import GroundTruth

from a0.model_utils import board_to_input, Policy

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def human_game(mcts_iterations: int = 64) -> tuple[int, int]:
    gt = GroundTruth()

    model = AlphaZeroModel(
        config.board_size,
        training=False,
        rngs=nnx.Rngs({'params': jax.random.PRNGKey(0)})
    )
    model = load_model(f"{config.training_dir}/model_{49}.pkl")

    player = A0Player(config.board_size, config.num_pieces, model, exploit=True)
    player.mcts_iterations = mcts_iterations

    human_player = HumanPlayer()

    players = [player, human_player]

    game = Game(config.board_size, config.num_pieces, False, False, False)

    turn = 0
    turn_limit = 80
    total_nt_acc = 0
    total_nt = 0
    while not game.end and (turn < turn_limit):
        #if turn == 10: exit()
        player = players[turn % len(players)]

        moves = game.start_turn()
        
        move, player_data = player.select_move(game.board, moves)

        if type(player) == A0Player:
            logger.info("\n" + game.board.visualize_move_ends([move]))
            logger.info(f"Current player: {game.board.current_player}")
            logger.info(f"Selected move: {move}")

            # visualize moves
            gt_moves, gt_outcomes = gt.get_1ply_policy_moves(game.board)
            player_p = Policy(config.board_size)
            player_p.set_logits(np.array(player_data), rotate_180=game.board.current_player == Player.PLAYER_O)
            player_probs = player_p.get_move_probabilities(gt_moves)
            for m, o, p in sorted(zip(gt_moves, gt_outcomes, player_probs), key=lambda x: x[2], reverse=True):
                logger.info(f"Move: {m}, Outcome: {o}, Player Prob: {p:.3f}")

        game.end_turn(move)

        turn += 1
    
    if game.winner is None:
        logger.info(f"It's a draw!")
    else:
        logger.info(f"Player {game.winner} wins!")
    logger.info(f"Total turns played: {turn}")

def game(mcts_iterations: int = 64) -> tuple[int, int]:
    gt = GroundTruth()

    model = AlphaZeroModel(
        config.board_size,
        training=False,
        rngs=nnx.Rngs({'params': jax.random.PRNGKey(0)})
    )

    player = A0Player(config.board_size, config.num_pieces, model, exploit=True)
    player.mcts_iterations = mcts_iterations

    players = [player, player]

    game = Game(config.board_size, config.num_pieces, True, False, False)

    turn = 0
    turn_limit = 80
    total_nt_acc = 0
    total_nt = 0
    while not game.end and (turn < turn_limit):
        player = players[turn % len(players)]

        moves = game.start_turn()
        
        move, player_data = player.select_move(game.board, moves)

        # logger.info("\n" + game.board.visualize_move_ends([move]))
        # logger.info(f"Current player: {game.board.current_player}")
        # logger.info(f"Selected move: {move}")

        # visualize moves
        # gt_moves, gt_outcomes = gt.get_1ply_policy_moves(game.board)
        # player_p = Policy(config.board_size)
        # player_p.set_logits(np.array(player_data), rotate_180=game.board.current_player == Player.PLAYER_O)
        # player_probs = player_p.get_move_probabilities(gt_moves)
        # for m, o, p in sorted(zip(gt_moves, gt_outcomes, player_probs), key=lambda x: x[2], reverse=True):
        #     logger.info(f"Move: {m}, Outcome: {o}, Player Prob: {p:.3f}")

        # TODO: look at and visualize policy created
        is_trivial = gt.is_trivial(game.board)
        if not is_trivial:
            total_nt += 1
            sd_policy = np.array(gt.get_1ply_policy_prob_dist_list(game.board, for_model=True))
            gd_policy: NDArray[np.float32] = player_data
            #pm_policy = policy_probability_mass_function(gd_policy, sd_policy)
            policy_is_acc = policy_accuracy_function(gd_policy, sd_policy)
            if policy_is_acc:
                total_nt_acc += 1
            #logger.info(f"Non-trivial turn. Player policy accuracy: {policy_is_acc}. Total accuracy: {total_nt_acc}/{total_nt} = {total_nt_acc/total_nt:.2%}")

        game.end_turn(move)

        turn += 1
    
    if game.winner is None:
        logger.info(f"It's a draw!")
    else:
        logger.info(f"Player {game.winner} wins!")
    logger.info(f"Total turns played: {turn}")
    logger.info(f"Trivial turns skipped: {turn - total_nt}/{turn} = {(turn - total_nt)/turn:.2%}")
    logger.info(f"Total non-trivial accuracy: {total_nt_acc}/{total_nt} = {total_nt_acc/total_nt:.2%}")
    
    return total_nt_acc, total_nt

def eval_acc(mcts_iterations: int = 64):
    logger.info(f"Evaluating accuracy over 10 games with {mcts_iterations} MCTS iterations...")
    total_total = 0
    total_acc = 0
    for i in range(10):
        acc, total = game(mcts_iterations=mcts_iterations)
        total_total += total
        total_acc += acc

    logger.info(f"Total non-trivial accuracy: {total_acc}/{total_total} = {total_acc/total_total:.2%}")

def main():
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="play_w_a0"
    )

    human_game(mcts_iterations=512)

    # eval_acc(mcts_iterations=64)
    # eval_acc(mcts_iterations=512)
    # eval_acc(mcts_iterations=1024)

if __name__ == "__main__":
    main()
