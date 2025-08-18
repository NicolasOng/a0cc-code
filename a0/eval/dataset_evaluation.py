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

from a0.model import AlphaZeroModel, load_model
from a0.dataset import Dataset
from a0.eval.plotting import Series, save_series

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
    # Argmax of the predicted policy (model's choice)
    pred_top_move = np.argmax(pred_policy)

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

def evaluate_model(model: AlphaZeroModel, evaluation_dataset: Dataset) -> tuple[float, float, float, float, float]:
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
    batches = evaluation_dataset.batches()

    total_loss = 0.0
    total_value_loss, total_value_accuracy = 0.0, 0.0
    total_policy_loss, total_policy_accuracy = 0.0, 0.0

    num_batches = 0
    
    for ts, batch in tqdm(enumerate(batches)):
        # get the board input, value label, and policy label from the batch
        board_input, value_label, policy_label = batch

        # get the model's predictions, and convert them to numpy arrays
        value, policy = model(board_input)
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

def evaluate_all_models(models: list[tuple[int, AlphaZeroModel]], evaluation_dataset: Dataset, fn: str) -> None:
    '''
    Evaluates all models in the training directory.
    The training directory is defined in the config.
    They are all evaluated on the same evaluation dataset.
    '''
    logger.info(f"Evaluating all models in the training directory (eval_type={fn})...")

    # Iterate through all model files in the training directory
    metrics = Series(["loss", "value_loss", "policy_loss", "value_accuracy", "policy_accuracy"])
    for i, model in tqdm(models):
        logger.info(f"Evaluating model {i + 1}")
        loss, value_loss, policy_loss, value_accuracy, policy_accuracy = evaluate_model(model, evaluation_dataset)
        metrics.x.append(i)
        metrics.ys["loss"].append(loss)
        metrics.ys["value_loss"].append(value_loss)
        metrics.ys["policy_loss"].append(policy_loss)
        metrics.ys["value_accuracy"].append(value_accuracy)
        metrics.ys["policy_accuracy"].append(policy_accuracy)
    
    # save the metrics to a file
    metrics_path = f"{config.eval_dir}/{fn}.pkl"
    save_series(metrics, metrics_path)

def evaluate_all_models_progressive(models: list[AlphaZeroModel], datasets: list[Dataset], fn: str) -> None:
    '''
    Evaluates all models on a dataset,
    except each model has its own dataset.
    So model 1 is evaluated on dataset 1, model 2 on dataset 2, etc.
    '''
    logger.info(f"Evaluating all models ({fn})...")

    # Iterate through all model files in the training directory
    losses: list[float] = []
    value_losses: list[float] = []
    policy_losses: list[float] = []
    value_accuracies: list[float] = []
    policy_accuracies: list[float] = []
    for i, (model, dataset) in tqdm(enumerate(zip(models, datasets))):
        logger.info(f"Evaluating model {i + 1}")
        loss, value_loss, policy_loss, value_accuracy, policy_accuracy = evaluate_model(model, dataset)
        losses.append(loss)
        value_losses.append(value_loss)
        policy_losses.append(policy_loss)
        value_accuracies.append(value_accuracy)
        policy_accuracies.append(policy_accuracy)

    # save the losses to a file
    losses_path = f"{config.eval_dir}/{fn}.pkl"
    with open(losses_path, 'wb') as f:
        pickle.dump({
            'losses': losses,
            'value_losses': value_losses,
            'policy_losses': policy_losses,
            'value_accuracies': value_accuracies,
            'policy_accuracies': policy_accuracies
        }, f)

def load_dataset(dataset_path: str) -> Dataset:
    '''
    Loads a dataset from the given path.
    Returns a Dataset object.
    If the file does not exist, it will log an error and exit.
    '''
    logger.info(f"Loading dataset from {dataset_path}...")
    try:
        with open(dataset_path, 'rb') as file:
            dataset: Dataset = pickle.load(file)
        logger.info(f"Loaded dataset from {dataset_path}.")
    except FileNotFoundError:
        logger.error(f"Dataset file not found at {dataset_path}. Please generate the dataset first.")
        sys.exit()
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit()
    return dataset

def load_dataset_list(datasets_path: str) -> list[Dataset]:
    try:
        with open(datasets_path, 'rb') as file:
            datasets: list[Dataset] = pickle.load(file)
        logger.info(f"Loaded datasets from {datasets_path}.")
    except FileNotFoundError:
        logger.error(f"Dataset file not found at {datasets_path}. Please generate the dataset first.")
        sys.exit()
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit()
    return datasets

def models_generator_function(dir: str, n: int) -> Generator[tuple[int, AlphaZeroModel], None, None]:
    '''
    Attempts to load [0 to n] AlphaZeroModels from the given directory.
    If one doesn't exist, it just skips it.
    File names are expected to be in the format "model_{i}.pkl" where i is the model number.
    '''
    for i in range(n + 1):
        model_path = f"{dir}/model_{i}.pkl"
        try:
            model = load_model(model_path)
            yield i, model
        except Exception as e:
            logger.error(f"Failed to load model {i} at {model_path}: {e}")

def load_models(dir: str, n: int) -> list[tuple[int, AlphaZeroModel]]:
    '''
    Returns a list of tuples (model_id, AlphaZeroModel).
    Attempts to load models from 0 to n (inclusive) - skips missing models.
    '''
    logger.info(f"Loading {n} models from {dir}...")
    return list(models_generator_function(dir, n))

def main():
    setup_logging(level=20, log_dir=config.log_dir, process_name='dataset_evaluation')

    logger.info("Starting dataset evaluation...")

    # load the models
    models = load_models(config.training_dir, config.training_iterations)

    # Load the datasets
    training_dataset = load_dataset(f"{config.dataset_out_dir}/training_gtv.pkl")
    random_dataset = load_dataset(f"{config.dataset_out_dir}/random_gtv.pkl")
    training_e_dataset = load_dataset(f"{config.dataset_out_dir}/training_ev.pkl")
    neighbor_datasets: list[Dataset] = []
    for i in range(2):
        neighbor_dataset = load_dataset(f"{config.dataset_out_dir}/neighbor_{i+1}_gtv.pkl")
        neighbor_datasets.append(neighbor_dataset)

    # trim down the training dataset to a smaller size for faster evaluation
    n = 1000
    training_dataset.trim(n, shuffle=False)
    training_e_dataset.trim(n, shuffle=False)
    for i in range(2):
        neighbor_datasets[i].trim(n, shuffle=False)

    # Evaluate all models
    evaluate_all_models(models, training_dataset, "training_gtv_eval")
    evaluate_all_models(models, random_dataset, "random_gtv_eval")
    evaluate_all_models(models, training_e_dataset, "training_ev_eval")
    for i in range(2):
        evaluate_all_models(models, neighbor_datasets[i], f"neighbor_{i+1}_gtv_eval")

    logger.info("Dataset evaluation completed.")


if __name__ == "__main__":
    main()
