import sys
import os
import random

from typing import Generator

from jax import numpy as jnp
import jax
import optax
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from a0.model import AlphaZeroModel, load_model
from a0.dataset import Dataset
from a0.eval.plotting import Series, save_series
from a0.players.a0 import A0Player
from a0.model_utils import input_to_board, board_to_input

from a0.eval.dataset_evaluation import load_models, load_dataset, value_loss_function, value_accuracy_function, get_policy_mask, policy_loss_function, policy_accuracy_batch

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def evaluate_model_mcts(player: A0Player, evaluation_dataset: Dataset):
    '''
    Evaluates the model on the given evaluation dataset with MCTS.
    The evaluation dataset is a Dataset object containing
    - game states
    - target values
    - target policies
    Returns the average loss, value loss, policy loss, and accuracy.
    And any other metrics I might want to add later.
    Mostly copied from dataset_evaluation.py's evaluate_model function.
    '''
    logger.info(f"Evaluating model on a dataset ({len(evaluation_dataset)})...")

    assert evaluation_dataset.batch_size == 1, "Evaluation dataset batch size must be 1 for MCTS evaluation."

    # shuffle the dataset and create batches generator
    evaluation_dataset.shuffle()
    batches = evaluation_dataset.batches()

    total_loss = 0.0
    total_value_loss, total_value_accuracy = 0.0, 0.0
    total_policy_loss, total_policy_accuracy = 0.0, 0.0

    num_batches = 0
    
    for ts, batch in tqdm(enumerate(batches)):
        # get the board input, value label, and policy label from the batch
        board_input, value_label, policy_label = batch

        # convert the board input to a Board object
        board_input = np.squeeze(board_input, axis=1)  # because the shape was (B, 1, 5, 5, 2) for some reason
        po = random.random() < 0.5
        board = input_to_board(board_input, player_o=po)

        # get the model's predictions, and convert them to numpy arrays
        value, policy = player.get_value_and_policy(board)
        pred_value = np.array(value, dtype=np.float32)
        pred_policy = np.array(policy, dtype=np.float32)
        # also convert the labels to numpy arrays
        value_label = np.array(value_label, dtype=np.float32)
        policy_label = np.array(policy_label, dtype=np.float32)

        # Calculate actual loss (L2) for value prediction
        value_loss = value_loss_function(pred_value, value_label)
        # Calculate accuracy for value prediction
        value_accuracy = value_accuracy_function(pred_value, value_label)

        # get the mask for valid moves in the policy
        mask_value = 0.0 # 0.0 for CE, -1.0 for BCE
        policy_mask = get_policy_mask(policy_label, mask_value=mask_value)
        # Apply the mask to the predicted policy and label policy
        # The mask value when using Softmax Cross Entropy loss should be -1e9
        # When using Binary Cross Entropy, it should be 0.0
        mask_value = -1e9 # -1e9 for CE, 0.0 for BCE
        masked_policy_pred = np.where(policy_mask, pred_policy, mask_value)
        #masked_policy_pred = pred_policy
        masked_policy_label = np.where(policy_mask, policy_label, 0.0)
        # Calculate policy loss
        policy_loss = policy_loss_function(masked_policy_pred, masked_policy_label) # for CE
        # policy_loss = policy_loss_function_binary(masked_policy_pred, masked_policy_label) # for BCE
        # Calculate policy accuracy
        # (for BCE) set mask to be very negative,
        # as the logits of illegal moves must be lower than those of legal moves.
        masked_policy_pred = np.where(policy_mask, pred_policy, -1e9)
        policy_accuracy = policy_accuracy_batch(masked_policy_pred, masked_policy_label)

        loss = value_loss + policy_loss

        total_loss += loss
        total_value_loss += value_loss
        total_value_accuracy += value_accuracy
        total_policy_loss += policy_loss
        total_policy_accuracy += policy_accuracy
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_value_loss = total_value_loss / num_batches
    avg_value_accuracy = total_value_accuracy / num_batches
    avg_policy_loss = total_policy_loss / num_batches
    avg_policy_accuracy = total_policy_accuracy / num_batches

    logger.info(
        f"Loss: {avg_loss:.4f}, "
        f"Value Loss: {avg_value_loss:.4f}, "
        f"Value Accuracy: {avg_value_accuracy:.2%}, "
        f"Policy Loss: {avg_policy_loss:.4f}, "
        f"Policy Accuracy: {avg_policy_accuracy:.2%}"
    )
    return avg_loss, avg_value_loss, avg_policy_loss, avg_value_accuracy, avg_policy_accuracy

def evaluate_all_models_mcts(players: list[tuple[int, A0Player]], evaluation_dataset: Dataset, fn: str):
    '''
    Evaluates all models in the training directory with MCTS.
    The training directory is defined in the config.
    They are all evaluated on the same evaluation dataset.
    '''
    logger.info(f"Evaluating all models with MCTS in the training directory (eval_type={fn})...")

    # Iterate through all model files in the training directory
    metrics = Series(["loss", "value_loss", "policy_loss", "value_accuracy", "policy_accuracy"])
    for i, player in tqdm(players):
        logger.info(f"Evaluating model {i + 1}")
        loss, value_loss, policy_loss, value_accuracy, policy_accuracy = evaluate_model_mcts(player, evaluation_dataset)
        metrics.x.append(i)
        metrics.ys["loss"].append(loss)
        metrics.ys["value_loss"].append(value_loss)
        metrics.ys["policy_loss"].append(policy_loss)
        metrics.ys["value_accuracy"].append(value_accuracy)
        metrics.ys["policy_accuracy"].append(policy_accuracy)
    
    # save the metrics to a file
    metrics_path = f"{config.eval_dir}/{fn}.pkl"
    save_series(metrics, metrics_path)

def load_players(training_dir: str, training_iterations: int) -> list[tuple[int, A0Player]]:
    '''
    Loads the A0 players for MCTS evaluation.
    Returns a list of A0Player objects.
    '''
    players: list[tuple[int, A0Player]] = []
    models = load_models(training_dir, training_iterations)
    for i, model in models:
        player = A0Player(
            board_size=config.board_size,
            num_pieces=config.num_pieces,
            model=model,
            mcts_samples=config.eval_mcts_samples[0],
            no_reverse_moves=not config.backwards_moves,
            no_side_moves=not config.sideways_moves
        )
        players.append((i, player))
    return players

def main():
    setup_logging(level=20, log_dir=config.log_dir, process_name='dataset_evaluation')

    logger.info("Starting dataset evaluation...")

    # load the players
    players = load_players(config.training_dir, config.training_iterations)

    # Load the datasets
    num_neighbors = config.eval_neighbors
    training_dataset = load_dataset(f"{config.dataset_out_dir}/training_gtv.pkl")
    random_dataset = load_dataset(f"{config.dataset_out_dir}/random_gtv.pkl")
    training_e_dataset = load_dataset(f"{config.dataset_out_dir}/training_ev.pkl")
    neighbor_datasets: list[Dataset] = []
    for i in range(num_neighbors):
        neighbor_dataset = load_dataset(f"{config.dataset_out_dir}/neighbor_{i+1}_gtv.pkl")
        neighbor_datasets.append(neighbor_dataset)
    
    # load the datasets (non-trivial)
    training_nt_dataset = load_dataset(f"{config.dataset_out_dir}/training_nt_gtv.pkl")
    random_nt_dataset = load_dataset(f"{config.dataset_out_dir}/random_nt_gtv.pkl")
    training_nt_e_dataset = load_dataset(f"{config.dataset_out_dir}/training_nt_ev.pkl")
    neighbor_nt_datasets: list[Dataset] = []
    for i in range(num_neighbors):
        neighbor_dataset = load_dataset(f"{config.dataset_out_dir}/neighbor_{i+1}_nt_gtv.pkl")
        neighbor_nt_datasets.append(neighbor_dataset)
    
    # create a list with all datasets for easy trimming
    all_datasets = [
        training_dataset,
        random_dataset,
        training_e_dataset,
        *neighbor_datasets,
        training_nt_dataset,
        random_nt_dataset,
        training_nt_e_dataset,
        *neighbor_nt_datasets
    ]

    # trim down the datasets to a smaller size for faster evaluation
    n = 100
    for dataset in all_datasets:
        dataset.trim(n, shuffle=False)
    
    # set the batch size for all datasets to 1
    # since MCTS is run on a single board at a time, unlike the model
    for dataset in all_datasets:
        dataset.batch_size = 1

    # for each mcts sample count, evaluate all models
    for mcts_samples in config.eval_mcts_samples:
        logger.info(f"Evaluating with MCTS samples: {mcts_samples}")

        # update the players' mcts_samples
        for i, player in players:
            player.mcts_iterations = mcts_samples
    
        # Evaluate all models
        evaluate_all_models_mcts(players, training_dataset, f"mcts_eval_{mcts_samples}/training_gtv_mcts_eval_{mcts_samples}")
        evaluate_all_models_mcts(players, random_dataset, f"mcts_eval_{mcts_samples}/random_gtv_mcts_eval_{mcts_samples}")
        evaluate_all_models_mcts(players, training_e_dataset, f"mcts_eval_{mcts_samples}/training_ev_mcts_eval_{mcts_samples}")
        for i in range(num_neighbors):
            evaluate_all_models_mcts(players, neighbor_datasets[i], f"mcts_eval_{mcts_samples}/neighbor_{i+1}_gtv_mcts_eval_{mcts_samples}")

        # Evaluate all models
        evaluate_all_models_mcts(players, training_nt_dataset, f"mcts_eval_{mcts_samples}/training_nt_gtv_mcts_eval_{mcts_samples}")
        evaluate_all_models_mcts(players, random_nt_dataset, f"mcts_eval_{mcts_samples}/random_nt_gtv_mcts_eval_{mcts_samples}")
        evaluate_all_models_mcts(players, training_nt_e_dataset, f"mcts_eval_{mcts_samples}/training_nt_ev_mcts_eval_{mcts_samples}")
        for i in range(num_neighbors):
            evaluate_all_models_mcts(players, neighbor_nt_datasets[i], f"mcts_eval_{mcts_samples}/neighbor_{i+1}_nt_gtv_mcts_eval_{mcts_samples}")

    logger.info("Dataset evaluation completed.")


if __name__ == "__main__":
    main()
