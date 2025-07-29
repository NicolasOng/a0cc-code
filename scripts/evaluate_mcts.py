import random

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from a0.train.generate_datasets import load_ground_truth_dataset
from a0.model import load_model
from a0.eval.dataset_evaluation import evaluate_model, value_accuracy_function, policy_accuracy_function
from a0.players.a0 import NNMCTSProblem
from cc.core import Game, Board, Player
from cc.ground_truth import GroundTruth
from a0.graph_search.mcts import MCTS
from a0.model_utils import input_to_board, Policy

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

from config import config

setup_logging(
    level=20,
    log_dir=config.log_dir,
    process_name="evaluate_mcts"
)

def policy_from_mcts(mcts: MCTS, board: Board) -> Policy:
    '''
    Creates a policy from the MCTS root's children.
    The policy is in the perspective of the current player.
    Uses softmax to create a probability distribution.
    (board_size ** 4,)
    '''
    # with the root's children, create a policy distribution logits
    visit_counts = [float(child.visits) for child in mcts.root.children]
    moves = [board.child_board_to_move(child.state) for child in mcts.root.children]

    # create a well-shaped policy distribution,
    p = Policy(config.board_size)
    p.set_logits_from_moves(moves, visit_counts, rotate_180=False)
    # mask non-legal moves,
    p.set_legal_moves(moves)
    # softmax it to get the policy distribution
    p.apply_softmax(temperature=1.0, mask=True)
    
    # get the numpy array of the policy distribution
    # np_policy = np.array(p.get_policy_list())

    return p

def main():
    # TODO: make this a parameter?
    model_filename = "..."
    num_boards_per_iteration = 500

    # load the given model
    model = load_model(model_filename)
    logger.info(f"Model loaded from {model_filename}")

    # create the game
    cc = Game(config.board_size, config.num_pieces, False, False, False)
    
    # get the states to evaluate the model on
    gt_dataset = load_ground_truth_dataset()
    logger.info("Ground truth dataset loaded successfully.")
    gt_dataset.trim(num_boards_per_iteration, shuffle=True)

    # first, evaluate the model without MCTS
    logger.info("Evaluating model without MCTS...")
    _, _, _, value_acc, policy_acc = evaluate_model(model, gt_dataset, 0)
    logger.info(f"Base value accuracy: {value_acc:.2%}, Base policy accuracy: {policy_acc:.2%}")

    # evaluate the model with MCTS over some number of iterations
    logger.info("Evaluating model with MCTS...")
    uct_x = [0]
    uct_value_acc = [value_acc]
    uct_policy_acc = [policy_acc]
    puct_x = [0]
    puct_value_acc = [value_acc]
    puct_policy_acc = [policy_acc]
    gt_dataset.set_batch_size(1)
    for i in range(25, 500, 25):
        logger.info(f"Evaluating model with {i} MCTS iterations...")
        total_uct_value_acc = 0
        total_puct_value_acc = 0
        total_uct_policy_acc = 0
        total_puct_policy_acc = 0
        num_evaled = 0
        batches = gt_dataset.batches()
        for j, batch in enumerate(batches):
            is_po = j % 2 == 1  # whether the current player is Player O (half the time)
            # (1, board_size, board_size, 2), (1, 1), (1, board_size ** 4)
            board_batch, value_batch, policy_batch = batch
            board_batch = np.array(board_batch, dtype=np.float32)
            value_batch = np.array(value_batch, dtype=np.float32)
            policy_batch = np.array(policy_batch, dtype=np.float32)
            board = input_to_board(board_batch, player_o=is_po)

            legal_moves = cc.generate_moves_for_given_board(board)
            is_done = cc.get_done(board)
            is_illegal = not cc.legal(board)
            if not legal_moves or is_done or is_illegal:
                logger.info(f"Skipping batch {j+1}/{len(gt_dataset)}: No legal moves, or done, or illegal.")
                continue

            # print(board.board_view())
            # print(board.current_player)
            # for move in legal_moves:
            #     print(move)

            nmp_uct = NNMCTSProblem(
                initial_state=board,
                game=cc,
                model=model
            )
            nmp_puct = NNMCTSProblem(
                initial_state=board,
                game=cc,
                model=model
            )
        
            # create the MCTS instances
            mcts_uct = MCTS(nmp_uct, 'uct')
            mcts_puct = MCTS(nmp_puct, 'puct')

            # run the MCTS for the specified number of iterations
            mcts_uct.run(iterations=i)
            mcts_puct.run(iterations=i)

            # get the value and policy from MCTS
            value_uct = mcts_uct.root.reward / mcts_uct.root.visits
            value_puct = mcts_puct.root.reward / mcts_puct.root.visits

            # with the root's children, create a policy distribution logits
            policy_uct = policy_from_mcts(mcts_uct, board)
            policy_puct = policy_from_mcts(mcts_puct, board)
            policy_uct_np = np.array(policy_uct.get_policy_list())
            policy_puct_np = np.array(policy_puct.get_policy_list())

            policy_batch_p = Policy(config.board_size)
            policy_batch_p.set_logits(policy_batch[0], rotate_180=is_po)
            policy_batch_p_np = np.array(policy_batch_p.get_policy_list())

            # print([f'{mp:.2f}' for mp in policy_batch_p.get_move_probabilities(legal_moves)])
            # print([f'{mp:.2f}' for mp in policy_uct.get_move_probabilities(legal_moves)])

            # evaluate the value and policy accuracy
            # (1,), (1, )
            value_acc_uct = value_accuracy_function(value_batch[0], np.array([value_uct]))
            value_acc_puct = value_accuracy_function(value_batch[0], np.array([value_puct]))
            # (board_size ** 4, ), (board_size ** 4, )
            policy_acc_uct = policy_accuracy_function(policy_uct_np, policy_batch_p_np)
            policy_acc_puct = policy_accuracy_function(policy_puct_np, policy_batch_p_np)

            total_uct_value_acc += value_acc_uct
            total_puct_value_acc += value_acc_puct
            total_uct_policy_acc += policy_acc_uct
            total_puct_policy_acc += policy_acc_puct
            num_evaled += 1
        
        avg_uct_value_acc = total_uct_value_acc / num_evaled
        avg_puct_value_acc = total_puct_value_acc / num_evaled
        avg_uct_policy_acc = total_uct_policy_acc / num_evaled
        avg_puct_policy_acc = total_puct_policy_acc / num_evaled
        uct_x.append(i)
        puct_x.append(i)
        uct_value_acc.append(avg_uct_value_acc)
        puct_value_acc.append(avg_puct_value_acc)
        uct_policy_acc.append(avg_uct_policy_acc)
        puct_policy_acc.append(avg_puct_policy_acc)

        logger.info(f"Metrics for {i} MCTS iterations:")
        logger.info(f"UCT Value Accuracy: {avg_uct_value_acc:.2%}")
        logger.info(f"PUCT Value Accuracy: {avg_puct_value_acc:.2%}")
        logger.info(f"UCT Policy Accuracy: {avg_uct_policy_acc:.2%}")
        logger.info(f"PUCT Policy Accuracy: {avg_puct_policy_acc:.2%}")
        logger.info("")
    
    # save the results to a file
    output_path = f"{config.eval_dir}/mcts_eval.pkl"
    with open(output_path, 'wb') as file:
        np.savez(file, uct_x=uct_x, uct_value_acc=uct_value_acc, uct_policy_acc=uct_policy_acc,
                 puct_x=puct_x, puct_value_acc=puct_value_acc, puct_policy_acc=puct_policy_acc)
    logger.info(f"Results saved to {output_path}")

    # plot the results
    plt.figure(figsize=(16 * 0.75, 9 * 0.75))
    plt.plot(uct_x, uct_value_acc, label="UCT Value Accuracy", color='blue', linestyle='-')
    plt.plot(uct_x, uct_policy_acc, label="UCT Policy Accuracy", color='blue', linestyle='--')
    plt.plot(puct_x, puct_value_acc, label="PUCT Value Accuracy", color='red', linestyle='-')
    plt.plot(puct_x, puct_policy_acc, label="PUCT Policy Accuracy", color='red', linestyle='--')
    plt.xlabel("MCTS Iterations")
    plt.ylabel("Value/Policy Accuracy")
    plt.title("MCTS Evaluation Results")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{config.eval_dir}/mcts_eval.png", dpi=300)
    plt.clf()

if __name__ == "__main__":
    main()
