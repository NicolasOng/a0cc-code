import sys
import os

from typing import Generator, Literal, Optional, overload

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
from a0.utils.load_training_data import load_models, models_generator_function
from a0.utils.safe_load import safe_load_pickle

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

def policy_entropy_batch(pred_logits: NDArray[np.float32], policy_mask: NDArray[np.bool_]) -> float:
    '''
    Mean normalized Shannon entropy of the model's policy over legal moves.
    pred_logits: 2D (batch, board_size ** 4) raw policy-head logits.
    policy_mask: 2D (batch, board_size ** 4), truthy for legal moves.
    The logits are softmaxed over legal moves only, then the entropy of each
    sample is normalized by log(num legal moves) so it lies in [0, 1] and is
    comparable across positions with different branching factors. Samples with
    <= 1 legal move contribute 0.
    '''
    mask = np.asarray(policy_mask).astype(bool)
    masked_logits = np.where(mask, pred_logits, -1e9)
    probs = np.array(jax.nn.softmax(masked_logits, axis=-1), dtype=np.float32)
    probs = np.where(mask, probs, 0.0)
    # np.where evaluates BOTH branches, so log(0) -> -inf -> 0*-inf = nan fires a
    # RuntimeWarning on every call even though the result is discarded. Feeding
    # the log a safe 1.0 where probs == 0 gives the same answer (0 * log(1) = 0)
    # without the warning — this runs millions of times in the model heatmaps.
    safe_probs = np.where(probs > 0, probs, 1.0)
    per_move = np.where(probs > 0, probs * np.log(safe_probs), 0.0)
    ent = -np.sum(per_move, axis=-1)                       # nats, per sample
    num_legal = np.sum(mask, axis=-1).astype(np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        norm = np.where(num_legal > 1, ent / np.log(num_legal), 0.0)
    return float(np.mean(norm))

def evaluate_model(model: AlphaZeroModel, evaluation_dataset: Dataset) -> tuple[float, float, float, float, float, float]:
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
    total_policy_entropy = 0.0

    num_batches = 0
    
    for ts, batch in tqdm(enumerate(batches)):
        # get the board input, value label, and policy label from the batch
        board_input, value_label, policy_label, policy_mask, _ = batch

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

        # Normalized entropy of the model's (softmaxed) policy over legal moves
        policy_entropy = policy_entropy_batch(pred_policy, np.asarray(policy_mask))

        loss = value_loss + policy_loss

        total_loss += loss
        total_value_loss += value_loss
        total_value_accuracy += value_accuracy
        total_policy_loss += policy_loss
        total_policy_accuracy += policy_accuracy
        total_policy_entropy += policy_entropy
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_value_loss = total_value_loss / num_batches
    avg_value_accuracy = total_value_accuracy / num_batches
    avg_policy_loss = total_policy_loss / num_batches
    avg_policy_accuracy = total_policy_accuracy / num_batches
    avg_policy_entropy = total_policy_entropy / num_batches

    logger.info(
        f"Loss: {avg_loss:.4f}, "
        f"Value Loss: {avg_value_loss:.4f}, "
        f"Value Accuracy: {avg_value_accuracy:.2%}, "
        f"Policy Loss: {avg_policy_loss:.4f}, "
        f"Policy Accuracy: {avg_policy_accuracy:.2%}, "
        f"Policy Entropy (norm): {avg_policy_entropy:.4f}"
    )
    return avg_loss, avg_value_loss, avg_policy_loss, avg_value_accuracy, avg_policy_accuracy, avg_policy_entropy

def evaluate_all_models(models: list[tuple[int, AlphaZeroModel]], evaluation_dataset: Dataset, fn: str) -> None:
    '''
    Evaluates all models in the training directory.
    The training directory is defined in the config.
    They are all evaluated on the same evaluation dataset.
    '''
    logger.info(f"Evaluating all models in the training directory (eval_type={fn})...")

    # Iterate through all model files in the training directory
    metrics = Series(["loss", "value_loss", "policy_loss", "value_accuracy", "policy_accuracy", "policy_entropy"])
    for i, model in tqdm(models):
        logger.info(f"Evaluating model {i + 1}")
        loss, value_loss, policy_loss, value_accuracy, policy_accuracy, policy_entropy = evaluate_model(model, evaluation_dataset)
        metrics.x.append(i)
        metrics.ys["loss"].append(loss)
        metrics.ys["value_loss"].append(value_loss)
        metrics.ys["policy_loss"].append(policy_loss)
        metrics.ys["value_accuracy"].append(value_accuracy)
        metrics.ys["policy_accuracy"].append(policy_accuracy)
        metrics.ys["policy_entropy"].append(policy_entropy)

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
        loss, value_loss, policy_loss, value_accuracy, policy_accuracy, _ = evaluate_model(model, dataset)
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

def evaluate_on_all_datasets(model: AlphaZeroModel, datasets: dict[int, Dataset], fn: str) -> None:
    '''
    Evaluates the model on all datasets in the dictionary.
    The dictionary is expected to have keys that are integers that can be ordered,
    and values as Dataset objects.
    '''
    logger.info(f"Evaluating model on all datasets ({fn})...")

    # Iterate through all datasets in the dictionary
    metrics = Series(["loss", "value_loss", "policy_loss", "value_accuracy", "policy_accuracy", "policy_entropy"])
    for bin_key in sorted(datasets.keys()):
        dataset = datasets[bin_key]
        dataset.batch_size = 1
        if len(dataset) < dataset.batch_size + 1:
            logger.info(f"Skipping small dataset with progress bin {bin_key}")
            continue
        logger.info(f"Evaluating dataset with progress bin {bin_key}")
        loss, value_loss, policy_loss, value_accuracy, policy_accuracy, policy_entropy = evaluate_model(model, dataset)
        metrics.x.append(bin_key)
        metrics.ys["loss"].append(loss)
        metrics.ys["value_loss"].append(value_loss)
        metrics.ys["policy_loss"].append(policy_loss)
        metrics.ys["value_accuracy"].append(value_accuracy)
        metrics.ys["policy_accuracy"].append(policy_accuracy)
        metrics.ys["policy_entropy"].append(policy_entropy)
    
    # save the metrics to a file
    metrics_path = f"{config.eval_dir}/{fn}.pkl"
    save_series(metrics, metrics_path)

@overload
def load_dataset(dataset_path: str) -> Dataset: ...
@overload
def load_dataset(dataset_path: str, *, optional: Literal[False]) -> Dataset: ...
@overload
def load_dataset(dataset_path: str, *, optional: Literal[True]) -> Optional[Dataset]: ...
def load_dataset(dataset_path: str, *, optional: bool = False) -> Optional[Dataset]:
    '''
    Loads a dataset from the given path.
    By default raises FileNotFoundError if the file is missing. Pass
    optional=True to get None back instead.
    '''
    dataset: Optional[Dataset] = safe_load_pickle(dataset_path, "dataset")  # type: ignore[assignment]
    if dataset is None and not optional:
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}.")
    return dataset

@overload
def load_dataset_list(datasets_path: str) -> list[Dataset]: ...
@overload
def load_dataset_list(datasets_path: str, *, optional: Literal[False]) -> list[Dataset]: ...
@overload
def load_dataset_list(datasets_path: str, *, optional: Literal[True]) -> Optional[list[Dataset]]: ...
def load_dataset_list(datasets_path: str, *, optional: bool = False) -> Optional[list[Dataset]]:
    '''
    Loads a list of datasets from a single pickle file.
    Strict by default (raises FileNotFoundError); pass optional=True to get None on missing.
    '''
    datasets: Optional[list[Dataset]] = safe_load_pickle(datasets_path, "dataset list")  # type: ignore[assignment]
    if datasets is None and not optional:
        raise FileNotFoundError(f"Dataset file not found at {datasets_path}.")
    return datasets

def calculate_dataset_bias(dataset: Dataset) -> tuple[float, float, float]:
    '''
    Calculates the bias of the dataset.
    The bias is defined as the percentage of wins, losses, and draws in the dataset.
    Returns a tuple of (win_percent, loss_percent, draw_percent).
    '''
    total = len(dataset)
    if total == 0:
        return 0.0, 0.0, 0.0
    wins = sum(1 for value in dataset.values if value > 0)
    losses = sum(1 for value in dataset.values if value < 0)
    draws = sum(1 for value in dataset.values if value == 0)
    win_percent = wins / total
    loss_percent = losses / total
    draw_percent = draws / total
    #logger.info(f"Dataset bias - Wins: {win_percent:.2%}, Losses: {loss_percent:.2%}, Draws: {draw_percent:.2%}")
    return win_percent, loss_percent, draw_percent

@overload
def load_dataset_dict(dataset_path: str) -> dict[int, Dataset]: ...
@overload
def load_dataset_dict(dataset_path: str, *, optional: Literal[False]) -> dict[int, Dataset]: ...
@overload
def load_dataset_dict(dataset_path: str, *, optional: Literal[True]) -> Optional[dict[int, Dataset]]: ...
def load_dataset_dict(dataset_path: str, *, optional: bool = False) -> Optional[dict[int, Dataset]]:
    '''
    Loads a dict of bin-key → Dataset from a pickle file.
    Strict by default (raises FileNotFoundError); pass optional=True to get None on missing.
    '''
    datasets: Optional[dict[int, Dataset]] = safe_load_pickle(dataset_path, "dataset dict")  # type: ignore[assignment]
    if datasets is None and not optional:
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    return datasets

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
    
    # load the datasets (non-trivial)
    training_nt_dataset = load_dataset(f"{config.dataset_out_dir}/training_nt_gtv.pkl")
    random_nt_dataset = load_dataset(f"{config.dataset_out_dir}/random_nt_gtv.pkl")
    training_nt_e_dataset = load_dataset(f"{config.dataset_out_dir}/training_nt_ev.pkl")
    neighbor_nt_datasets: list[Dataset] = []
    for i in range(2):
        neighbor_dataset = load_dataset(f"{config.dataset_out_dir}/neighbor_{i+1}_nt_gtv.pkl")
        neighbor_nt_datasets.append(neighbor_dataset)

    # trim down the datasets to a smaller size for faster evaluation
    n = 1000
    training_dataset.trim(n, shuffle=False)
    training_e_dataset.trim(n, shuffle=False)
    for i in range(2):
        neighbor_datasets[i].trim(n, shuffle=False)
    
    training_nt_dataset.trim(n, shuffle=False)
    training_nt_e_dataset.trim(n, shuffle=False)
    for i in range(2):
        neighbor_nt_datasets[i].trim(n, shuffle=False)
    
    # Evaluate all models
    evaluate_all_models(models, training_dataset, "training_gtv_eval")
    evaluate_all_models(models, random_dataset, "random_gtv_eval")
    evaluate_all_models(models, training_e_dataset, "training_ev_eval")
    for i in range(2):
        evaluate_all_models(models, neighbor_datasets[i], f"neighbor_{i+1}_gtv_eval")
    
    # Evaluate all models
    evaluate_all_models(models, training_nt_dataset, "training_nt_gtv_eval")
    evaluate_all_models(models, random_nt_dataset, "random_nt_gtv_eval")
    evaluate_all_models(models, training_nt_e_dataset, "training_nt_ev_eval")
    for i in range(2):
        evaluate_all_models(models, neighbor_nt_datasets[i], f"neighbor_{i+1}_nt_gtv_eval")

    datasets_list = [(training_dataset, "training_gtv"), (random_dataset, "random_gtv"), (training_e_dataset, "training_ev")] + [(neighbor_datasets[i], f"neighbor_{i+1}_gtv") for i in range(2)]
    for dataset, name in datasets_list:
        win_percent, loss_percent, draw_percent = calculate_dataset_bias(dataset)
        logger.info(f"Dataset {name} bias - Wins: {win_percent:.2%}, Losses: {loss_percent:.2%}, Draws: {draw_percent:.2%}, Total: {len(dataset)}")

    evaluate_on_all_datasets(models[-1][1], load_dataset_dict(f"{config.dataset_out_dir}/state_progress_gtv_datasets.pkl"), "state_progress_gtv_datasets_eval")
    evaluate_on_all_datasets(models[-1][1], load_dataset_dict(f"{config.dataset_out_dir}/state_progress_nt_gtv_datasets.pkl"), "state_progress_nt_gtv_datasets_eval")

    logger.info("Dataset evaluation completed.")


if __name__ == "__main__":
    main()
