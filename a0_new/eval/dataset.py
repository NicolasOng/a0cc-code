import sys
import os

from typing import Generator

from jax import numpy as jnp
import jax
import optax
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from a0.eval.plotting import Series, save_series

from a0_new.protocols.model import TrainableModel
from a0_new.dataset import Dataset

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def value_loss_function(pred_outcome: NDArray[np.float32], label_outcome: NDArray[np.float32]) -> float:
    '''
    Function for outcome loss.
    It calculates the L2 loss between the predicted outcome and the label outcome.
    Both pred_outcome and label_outcome are expected to be 1D arrays of shape (1,) or (batch, 1).
    It handles both.
    '''
    # Calculate L2 loss
    return float(np.mean(np.array(optax.l2_loss(pred_outcome, label_outcome), dtype=np.float32)))

def value_accuracy_function(pred_outcome: NDArray[np.float32], label_outcome: NDArray[np.float32]) -> float:
    '''
    Function for outcome accuracy.
    It checks if the predicted outcome matches the label outcome.
    Both pred_outcome and label_outcome are expected to be 1D arrays of shape (1,) or (batch, 1).
    It handles both.
    '''
    # Convert outcomes to -1, 0, 1
    pred_classification = np.where(pred_outcome <= 0, -1, 1)
    label_classification = np.where(label_outcome <= 0, -1, 1)
    # Calculate accuracy
    return float(np.mean(pred_classification == label_classification, dtype=np.float32))

def get_policy_mask(label_policy: NDArray[np.float32], mask_value: float) -> NDArray[np.bool_]:
    '''
    Returns a mask for the label policy, when the label policy is a probability distribution.
    The mask is True for valid moves (where label_policy > mask_value).
    This is useful for filtering out invalid moves in the policy.
    If the policy is a policy distribution, illegal moves will be 0.
    If the policy is a for the binary CE loss, illegal moves will be -1.
    '''
    return label_policy > mask_value

def policy_loss_function(pred_policy: NDArray[np.float32], label_policy: NDArray[np.float32]) -> float:
    '''
    Function for policy loss.
    It calculates the cross-entropy loss between the predicted policy and the label policy.
    Both pred_policy and label_policy are expected to be 2D arrays of shape (batch_size, board_size ** 4),
    or 1D arrays of shape (board_size ** 4,).
    Both arrays should be "masked" first so that only valid moves are considered (all illegal moves to 0).
    The label policy is expected to be a probability distribution (sum to 1).
    The predicted policy is expected to be logits (not probabilities).
    '''
    # Calculate cross-entropy loss
    return float(np.mean(np.array(optax.softmax_cross_entropy(labels=label_policy, logits=pred_policy), dtype=np.float32)))

def policy_loss_function_binary(pred_policy: NDArray[np.float32], label_policy: NDArray[np.float32]) -> float:
    '''
    Function for policy loss using binary cross-entropy.
    It calculates the binary cross-entropy loss between the predicted policy and the label policy.
    Both pred_policy and label_policy are expected to be 2D arrays of shape (batch_size, board_size ** 4),
    or 1D arrays of shape (board_size ** 4,).
    Both arrays should be "masked" first so that only valid moves are considered (all illegal moves to 0).
    The label policy is expected to contain the probability each move is a win (0 to 1).
    The predicted policy is expected to be logits (not probabilities).
    '''
    # Calculate binary cross-entropy loss
    return float(np.mean(np.array(optax.sigmoid_binary_cross_entropy(logits=pred_policy, labels=label_policy), dtype=np.float32)))

def policy_accuracy_function(pred_policy: NDArray[np.float32], label_policy: NDArray[np.float32]):
    '''
    Function for policy accuracy.
    It checks if the predicted policy's top move is among the top moves in the label policy.
    This is useful for evaluating the model's policy predictions.
    Both pred_policy and label_policy are expected to be 1D arrays of shape (board_size ** 4,).
    Both arrays should be "masked" so that only valid moves are considered.
    '''
    # Find all moves with maximum predicted value
    pred_max = np.max(pred_policy)
    pred_top_moves = np.where(pred_policy == pred_max)[0]
    
    # Randomly select one if there are ties
    pred_top_move = np.random.choice(pred_top_moves)

    # Identify all top moves in the label policy (handle ties)
    label_max = np.max(label_policy)
    label_top_moves = np.where(label_policy == label_max)[0]

    # Check if predicted move is among the top label moves
    return pred_top_move in label_top_moves

def policy_accuracy_batch(pred_policy: NDArray[np.float32], label_policy: NDArray[np.float32]) -> float:
    '''
    Function for calculating the average policy accuracy over a batch.
    It checks if the predicted policy's top move is among the top moves in the label policy.
    Both arrays should be masked so that only valid moves are considered.
    pred_policy and label_policy are expected to be 2D arrays of shape (batch_size, board_size ** 4).
    '''
    accuracies = [policy_accuracy_function(p, l) for p, l in zip(pred_policy, label_policy)]
    mean_accuracy = np.mean(accuracies)
    return float(mean_accuracy)

def policy_probability_mass_function(pred_policy: NDArray[np.float32], label_policy: NDArray[np.float32]) -> float:
    '''
    Function for policy probability mass on optimal moves.
    It calculates how much probability mass the prediction assigns to moves 
    that are tied for the highest value in the ground truth.
    Both pred_policy and label_policy are expected to be 1D arrays of shape (board_size ** 4,).
    pred_policy should be a probability distribution (NOT LOGITS).
    label_policy should be a probability distribution.
    '''
    # Find the maximum value in the ground truth
    label_max = np.max(label_policy)
    
    # Create a mask for all moves tied for the best in ground truth
    optimal_moves_mask = (label_policy == label_max)
    
    # Sum the probability mass assigned to optimal moves
    prob_mass_on_optimal = np.sum(pred_policy * optimal_moves_mask)
    
    return float(prob_mass_on_optimal)

def policy_probability_mass_batch(pred_policy: NDArray[np.float32], label_policy: NDArray[np.float32]) -> float:
    '''
    Function for calculating the average probability mass on optimal moves over a batch.
    pred_policy and label_policy are expected to be 2D arrays of shape (batch_size, board_size ** 4).
    '''
    prob_masses = [policy_probability_mass_function(p, l) for p, l in zip(pred_policy, label_policy)]
    mean_prob_mass = np.mean(prob_masses)
    return float(mean_prob_mass)

def evaluate_model(model: TrainableModel, evaluation_dataset: Dataset) -> tuple[float, float, float, float, float]:
    '''
    Evaluates the model on the given evaluation dataset.
    The evaluation dataset is a Dataset object containing
    - game states
    - target values
    - target policies
    Returns the average loss, value loss, policy loss, and accuracy.
    And any other metrics I might want to add later.
    '''
    logger.info(f"Evaluating model on a dataset ({len(evaluation_dataset)})...")

    # shuffle the dataset and create batches generator
    evaluation_dataset.shuffle()
    batches = evaluation_dataset.jnp_batches()

    total_loss = 0.0
    total_value_loss, total_value_accuracy = 0.0, 0.0
    total_policy_loss, total_policy_accuracy = 0.0, 0.0

    num_batches = 0
    
    for ts, batch in tqdm(enumerate(batches)):
        # get the board input, value label, and policy label from the batch
        board_input, value_label, policy_label, policy_mask = batch

        # get the model's predictions, and convert them to numpy arrays
        value, policy = model.inference(board_input)
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
        #mask_value = 0.0 # 0.0 for CE, -1.0 for BCE
        #policy_mask = get_policy_mask(policy_label, mask_value=mask_value)
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
