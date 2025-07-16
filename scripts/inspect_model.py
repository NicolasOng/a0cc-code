import random

import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp

from a0.model import load_model
from a0.players.a0 import board_to_input, Policy
from cc.ground_truth import GroundTruth
from cc.core import Player, Game
from a0.eval.dataset_evaluation import policy_accuracy, policy_accuracy_batch
from a0.train.generate_datasets import load_ground_truth_dataset

from config import config

def value_accuracy(pred_values: NDArray[np.float32], ground_truth: NDArray[np.float32]) -> float:
    value_classification = np.where(pred_values <= 0, -1, 1)
    return float(np.mean(value_classification == ground_truth))

def create_batch(gt: GroundTruth, num_samples: int):
    """
    Creates a batch of random boards and their ground truth labels.
    """
    max_rank = gt.get_max_rank()
    batch: list[dict[str, NDArray[np.float32]]] = []
    for _ in range(num_samples):
        rank = random.randint(0, max_rank - 1)
        board = gt.unrank(rank)
        gt_policy = np.array(gt.get_1ply_policy_list(board, for_model=True))
        gt_outcome = np.array([gt.get_outcome(board)])
        batch.append({
            'board_input': board_to_input(board)[0],
            'gt_policy': gt_policy,
            'gt_outcome': gt_outcome})
    states = jnp.stack([b["board_input"] for b in batch])
    values = jnp.array([b["gt_outcome"] for b in batch])
    policies = jnp.stack([b["gt_policy"] for b in batch])
    return states, values, policies

def inspect_policy_batch_accuracy():
    model = load_model(config.training_dir + '/model_0.pkl')

    gt = GroundTruth()
    cc = Game(config.board_size, config.num_pieces, False, False, False)
    max_rank = gt.get_max_rank()

    states, gt_values, gt_policies = create_batch(gt, 256)

    pred_values, pred_policies = model(jnp.array(states))
    pred_values = np.array(pred_values)
    pred_policies = np.array(pred_policies)

    #print(f"Predicted values: {pred_values}")
    #print(f"Ground truth values: {gt_values}")
    model_value_accuracy = value_accuracy(pred_values, gt_values)
    model_policy_accuracy = policy_accuracy_batch(pred_policies, gt_policies)
    print(f"Model policy accuracy: {model_policy_accuracy}")
    print(f"Model value accuracy: {model_value_accuracy}")

def inspect_model():
    model = load_model(config.training_dir + '/model_0.pkl')

    gt = GroundTruth()
    cc = Game(config.board_size, config.num_pieces, False, False, False)
    max_rank = gt.get_max_rank()

    # choose a random rank to generate
    rank = random.randint(0, max_rank)
    #rank = 298093
    board = gt.unrank(rank)
    # get the legal moves for the board
    legal_moves = cc.generate_moves_for_given_board(board)
    # get the ground truth policy for the board
    gt_policy = Policy(config.board_size)
    gt_numpy_policy = np.array(gt.get_1ply_policy_list(board, False))
    gt_policy.set_logits(gt_numpy_policy, False)
    gt_move_probs = gt_policy.get_move_probabilities(legal_moves)

    print(f"Board rank: {rank}")
    print(board.board_view())
    print(f"Current player: {board.current_player}")
    print(f"Board state is trivial: {len(np.unique(np.array(gt_move_probs))) == 1}")
    print(f"Number of legal moves: {len(legal_moves)}")

    rotate_board = board.current_player == Player.PLAYER_O
    board_input = board_to_input(board)

    pred_value, pred_policy = model(board_input)
    pred_value, pred_policy = np.array(pred_value), np.array(pred_policy)

    p = Policy(config.board_size)
    p.set_logits(pred_policy[0], rotate_board)
    p.set_legal_moves(legal_moves)
    p.apply_softmax(1.0, True)
    move_probs = p.get_move_probabilities(legal_moves)

    nmp = Policy(config.board_size)
    nmp.set_logits(pred_policy[0], rotate_board)
    nmp.apply_softmax(1.0, False)

    #print([f'{x:.2f}' for x in nmp.policy])

    print(f"Number of non-zero elements in masked predicted policy: {np.count_nonzero(p.policy)}")
    #print(f"Number of non-near-zero elements in non-masked predicted policy: {np.sum(nmp.policy > 1e-2)}")

    print(f"Predicted value: {pred_value}")
    #print(f"Predicted policy: {p.policy}")
    #print(f"Move probabilities: {move_probs}")
    #print(f"Ground truth policy: {gt_move_probs}")
    print(f"Move probabilities: {[f'{x:.2f}' for x in move_probs]}")
    print(f"Ground truth policy: {[f'{x:.2f}' for x in gt_move_probs]}")
    print(f"Policy accuracy: {policy_accuracy(np.array(p.policy), gt_numpy_policy)}")

def main():
    inspect_model()
    #inspect_policy_batch_accuracy()

if __name__ == "__main__":
    main()
