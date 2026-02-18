from typing import Optional, Any
import os
import pickle
import matplotlib.pyplot as plt
import math
import time

import jax
from flax import nnx
import jax.numpy as jnp
import optax
import numpy as np

from config import config

from a0.model import AlphaZeroModel, save_model
from a0.dataset import Dataset
from a0.eval.dataset_evaluation import load_dataset_list, evaluate_model

from a0.model import AlphaZeroModel, save_model
from a0.dataset import Dataset

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

class TestData:
    value_loss: float
    policy_loss: float
    total_loss: float
    value_accuracy: float
    policy_accuracy: float

class BatchData:
    batch_size: int
    value_loss: float
    policy_loss: float
    total_loss: float
    value_accuracy: float
    policy_accuracy: float
    model_no: int | None = None
    test_metrics: dict[str, TestData]

class EpochData:
    batch_data: list[BatchData]
    time: Optional[float] = None

    def __init__(self):
        self.batch_data = []

class DatasetData:
    epoch_data: list[EpochData]

    def __init__(self):
        self.epoch_data = []

def value_loss_function(pred_values: jnp.ndarray, values_label: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the L2 loss between predicted values and ground truth values.
    """
    return jnp.mean(optax.l2_loss(pred_values, values_label)).astype(jnp.float32)

def value_accuracy_function(pred_values: jnp.ndarray, values_label: jnp.ndarray) -> float:
    '''
    Computes the accuracy of the predicted values against the ground truth values.
    '''
    pred_classification = jnp.where(pred_values <= 0, -1, 1)
    labels_classification = jnp.where(values_label <= 0, -1, 1)
    return jnp.mean(pred_classification == labels_classification).astype(float)

def get_policy_mask(policy: jnp.ndarray, mask_value: float) -> jnp.ndarray:
    """
    Creates a mask for the policy distribution where valid moves are 1 and illegal moves are 0.
    Or for the binary cross-entropy loss, where illegal moves are -1.
    """
    return policy > mask_value

def policy_loss_function(policy_label: jnp.ndarray, pred_logits: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the policy loss using softmax cross-entropy.
    pred_logits should be the logits (unnormalized scores) for the policy.
    policy_label should be the ground truth policy distribution.
    When masking illegal moves, use a very low value (e.g., -1e9) in the logits,
    and use 0 for the policy probability distribution.
    """
    return jnp.mean(optax.softmax_cross_entropy(labels=policy_label, logits=pred_logits))

def policy_loss_function_binary(pred_policy: jnp.ndarray, label_policy: jnp.ndarray) -> jnp.ndarray:
    '''
    Computes the policy loss using binary cross-entropy.
    The label policy is expected to contain the probability each move is a win (0 to 1).
    The pred_policy is expected to be logits (not probabilities).
    When masking illegal moves, use 0 for both the predicted policy and the label policy.
    '''
    # Calculate binary cross-entropy loss
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits=pred_policy, labels=label_policy))

def policy_accuracy_function(pred_policy: jnp.ndarray, policy_label: jnp.ndarray) -> float:
    """
    Computes the accuracy of the predicted policy against the ground truth policy.
    """
    # calculate the accuracy of the policy
    max_value = jnp.max(policy_label, axis=1, keepdims=True)  # Keep batch dimension
    is_max = policy_label == max_value  # Boolean mask for max values
    max_pred_indices = jnp.argmax(pred_policy, axis=1)  # Shape: (batch_size,)
    
    # Create one-hot encoding of predicted max indices
    pred_one_hot = jax.nn.one_hot(max_pred_indices, num_classes=pred_policy.shape[1])
    
    # Check if the predicted max index corresponds to any of the true max indices
    policy_accuracy = jnp.mean(jnp.any(is_max * pred_one_hot, axis=1)).astype(float)

    return policy_accuracy

def loss_fn(model: AlphaZeroModel, batch: dict[str, Any]):
    total_loss, value_loss, policy_loss, value_accuracy, policy_accuracy = 0.0, 0.0, 0.0, 0.0, 0.0
    board_input: jnp.ndarray = batch['board']  # (N, board_size, board_size, 2)
    value_label: jnp.ndarray = batch['value']
    policy_label: jnp.ndarray = batch['policy']
    policy_mask: jnp.ndarray = batch['mask']

    # get the model's predictions
    value, policy = model.train_inference(board_input)

    # calculate the value loss and accuracy
    value_loss = value_loss_function(value, value_label)
    value_accuracy = value_accuracy_function(value, value_label)

    # get the mask for valid moves in the policy
    # mask_value = 0.0 # 0.0 for CE, -1.0 for BCE
    # policy_mask = get_policy_mask(policy_label, mask_value=mask_value)
    # mask both the predicted policy and the label policy
    # The mask value when using Softmax Cross Entropy loss should be -1e9
    # When using Binary Cross Entropy, it should be 0.0
    mask_value = -1e9 # -1e9 for CE, 0.0 for BCE
    masked_pred_logits = jnp.where(policy_mask, policy, mask_value)
    #masked_pred_logits = policy
    masked_label_policy = jnp.where(policy_mask, policy_label, 0.0)
    # Use the masked logits and label policy to calculate the policy loss
    policy_loss = policy_loss_function(masked_label_policy, masked_pred_logits) # for CE
    # policy_loss = policy_loss_function_binary(masked_pred_logits, masked_label_policy) # for BCE
    # (for BCE) set mask to be very negative,
    # as the logits of illegal moves must be lower than those of legal moves.
    masked_pred_logits = jnp.where(policy_mask, masked_pred_logits, -1e9)
    policy_accuracy = policy_accuracy_function(masked_pred_logits, masked_label_policy)
    
    # calculate the total loss
    total_loss = value_loss + policy_loss

    # JAX requires the loss function to return a tuple of (loss, aux)
    # where aux can be any additional information you want to return
    return total_loss, (value_loss, policy_loss, value_accuracy, policy_accuracy)

@nnx.jit
def train_step(model: AlphaZeroModel, optimizer: nnx.Optimizer, batch: dict[str, Any]):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (value_loss, policy_loss, value_accuracy, policy_accuracy)), grads = grad_fn(model, batch)
    optimizer.update(grads)
    return loss, value_loss, policy_loss, value_accuracy, policy_accuracy

def train_model_epoch(model: AlphaZeroModel, dataset: Dataset, save: str = "None", cur_model_no: int = 0, test_datasets: dict[str, Dataset] = {}) -> tuple[AlphaZeroModel, EpochData, int]:
    logger.info(f"Training model on the given dataset ({len(dataset)})...")
    start = time.perf_counter()

    epoch_data = EpochData()

    # shuffle the dataset and create batches generator
    dataset.shuffle()
    batches = dataset.batches()

    num_batches = dataset.num_batches()
    batches_per_save = math.ceil(num_batches / 50)
    # total_models_evaluated = num_batches // batches_per_save # (ts + 1) % batches_per_save == 0
    total_models_evaluated = math.ceil(num_batches / batches_per_save) # ts % batches_per_save == 0
    logger.info(f"Number of batches: {num_batches}")
    logger.info(f"Batches per save: {batches_per_save}")
    logger.info(f"Total models: {total_models_evaluated}")

    # value: 0.00005
    optimizer = nnx.Optimizer(model, optax.adamw(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay
    ))

    for ts, batch in enumerate(batches):
        # convert the batch to a dictionary
        board_batch, value_batch, policy_batch, mask_batch = batch
        batch = {
            'board': board_batch,  # (N, board_size, board_size)
            'value': value_batch,  # (N, 1)
            'policy': policy_batch,  # (N, board_size ** 4)
            'mask': mask_batch  # (N, board_size ** 4)
        }
        loss, value_loss, policy_loss, value_accuracy, policy_accuracy = train_step(model, optimizer, batch)
        logger.info(f"Training Step {ts}/{num_batches}, "
                    f"Loss: {loss:.4f}, "
                    f"Value Loss: {value_loss:.4f}, "
                    f"Value Accuracy: {value_accuracy:.2%}, "
                    f"Policy Loss: {policy_loss:.4f}, "
                    f"Policy Accuracy: {policy_accuracy:.2%}")

        # create batch data
        batch_data = BatchData()
        batch_data.batch_size = len(batch['board'])
        batch_data.value_loss = value_loss
        batch_data.policy_loss = policy_loss
        batch_data.total_loss = loss
        batch_data.value_accuracy = value_accuracy
        batch_data.policy_accuracy = policy_accuracy
        batch_data.test_metrics = {}

        if test_datasets and ts % batches_per_save == 0:
            # evaluate the model on the test dataset every ... batches
            for test_dataset_name, test_dataset in test_datasets.items():
                logger.info(f"Evaluating model on test dataset '{test_dataset_name}'...")
                avg_loss, avg_value_loss, avg_policy_loss, avg_value_accuracy, avg_policy_accuracy = evaluate_model(model, test_dataset)
                test_data = TestData()
                test_data.value_loss = avg_value_loss
                test_data.policy_loss = avg_policy_loss
                test_data.total_loss = avg_loss
                test_data.value_accuracy = avg_value_accuracy
                test_data.policy_accuracy = avg_policy_accuracy
                batch_data.test_metrics[test_dataset_name] = test_data

        if save == "batch" and ts % batches_per_save == 0:
            # Save the model after every save_batch_amount batches
            cur_model_no += 1
            logger.info(f"Saving model {cur_model_no} after batch {ts + 1}...")
            save_model(config.training_dir + f'/model_{cur_model_no}.pkl', model)
            batch_data.model_no = cur_model_no
        
        # add the batch data to the epoch data
        epoch_data.batch_data.append(batch_data)
    
    end = time.perf_counter()
    epoch_data.time = end - start

    return model, epoch_data, cur_model_no

def train_model_epochs(model: AlphaZeroModel, dataset: Dataset, num_epochs: int, save: str = "None", plot: bool = True, test_datasets: dict[str, Dataset] = {}) -> tuple[AlphaZeroModel, DatasetData]:
    """
    Train the model for a number of epochs on the given dataset.
    """
    dataset_data = DatasetData()
    cur_model_no = 0
    for epoch in range(num_epochs):
        logger.info(f"Training epoch {epoch + 1}/{num_epochs}...")
        model, epoch_data, cur_model_no = train_model_epoch(model, dataset, save, cur_model_no, test_datasets)
        if save == "epoch":
            # Save the model after each epoch
            logger.info(f"Saving model after epoch {epoch + 1}...")
            save_model(config.training_dir + f'/model_{epoch + 1}.pkl', model)
            epoch_data.batch_data[-1].model_no = epoch + 1
        dataset_data.epoch_data.append(epoch_data)
        if plot:
            save_epoch_data(f"{config.training_dir}/epoch_{epoch + 1}_data.pkl", epoch_data)
            temp_dd = DatasetData()
            temp_dd.epoch_data.append(epoch_data)
            plot_model_performance(f"model_performance_epoch_{epoch + 1}", [temp_dd])
    return model, dataset_data

def train_model_datasets(model: AlphaZeroModel, datasets: list[Dataset], num_epochs: int, save: str = "None") -> tuple[AlphaZeroModel, list[DatasetData]]:
    """
    Train the model for a number of epochs on each dataset in the list.
    The model is saved either after each epoch or after each dataset, or never.
    """
    dataset_data_list: list[DatasetData] = []
    for i, dataset in enumerate(datasets):
        logger.info(f"Training on dataset {i + 1}/{len(datasets)}...")
        model, dataset_data = train_model_epochs(model, dataset, num_epochs, save)
        dataset_data_list.append(dataset_data)
        if save == "dataset":
            # Save the model after each dataset
            logger.info(f"Saving model after dataset {i + 1}...")
            save_model(config.training_dir + f'/model_{i + 1}.pkl', model)
    return model, dataset_data_list

def plot_model_performance(fn: str, dataset_datas: list[DatasetData]):
    # Collect model performance data over all epochs
    batch_x: list[int] = []
    value_losses: list[float] = []
    policy_losses: list[float] = []
    total_losses: list[float] = []
    value_accuracies: list[float] = []
    policy_accuracies: list[float] = []
    batch_i = 0
    test_x: list[int] = []
    test_value_losses: dict[str, list[float]] = {}
    test_policy_losses: dict[str, list[float]] = {}
    test_total_losses: dict[str, list[float]] = {}
    test_value_accuracies: dict[str, list[float]] = {}
    test_policy_accuracies: dict[str, list[float]] = {}
    for _, dataset in enumerate(dataset_datas):
        for _, epoch in enumerate(dataset.epoch_data):
            for _, batch in enumerate(epoch.batch_data):
                batch_x.append(batch_i)
                value_losses.append(batch.value_loss)
                policy_losses.append(batch.policy_loss)
                total_losses.append(batch.total_loss)
                value_accuracies.append(batch.value_accuracy)
                policy_accuracies.append(batch.policy_accuracy)
                if batch.test_metrics:
                    test_x.append(batch_i)
                    for test_dataset_name, test_dataset_metrics in batch.test_metrics.items():
                        if test_dataset_name not in test_value_losses:
                            test_value_losses[test_dataset_name] = []
                            test_policy_losses[test_dataset_name] = []
                            test_total_losses[test_dataset_name] = []
                            test_value_accuracies[test_dataset_name] = []
                            test_policy_accuracies[test_dataset_name] = []
                        test_value_losses[test_dataset_name].append(test_dataset_metrics.value_loss)
                        test_policy_losses[test_dataset_name].append(test_dataset_metrics.policy_loss)
                        test_total_losses[test_dataset_name].append(test_dataset_metrics.total_loss)
                        test_value_accuracies[test_dataset_name].append(test_dataset_metrics.value_accuracy)
                        test_policy_accuracies[test_dataset_name].append(test_dataset_metrics.policy_accuracy)
                batch_i += 1
    
    # get dataset and epoch boundaries
    num_datasets = 0
    num_epochs = 0
    dataset_boundaries: list[int] = []
    epoch_boundaries: list[int] = []
    cur_boundary = 0
    for dataset in dataset_datas:
        num_datasets += 1
        for epoch in dataset.epoch_data:
            num_epochs += 1
            # Add the number of batches in the epoch to the current boundary
            cur_boundary += len(epoch.batch_data)
            epoch_boundaries.append(cur_boundary)
        # Remove the last epoch boundary in each dataset
        epoch_boundaries = epoch_boundaries[:-1]
        dataset_boundaries.append(cur_boundary)
    # remove the last dataset boundary
    dataset_boundaries = dataset_boundaries[:-1]

    # start plotting
    plt.figure(figsize=(16, 9))

    # Add vertical lines for epoch boundaries
    for _, boundary in enumerate(epoch_boundaries):
        x=boundary - 0.5
        plt.axvline(x=x, color='gray', linestyle='--', alpha=0.35,
                    label='Epoch Boundary' if boundary == epoch_boundaries[0] else "")
        #plt.text(x, 0 - 0.05, f"Epoch {i + 1}", rotation=90, va='top', ha='center', fontsize=9, color='gray')
    
    # Add vertical lines for dataset boundaries
    for _, boundary in enumerate(dataset_boundaries):
        x=boundary - 0.5
        plt.axvline(x=x, color='gray', linestyle='--', alpha=0.7,
                    label='Dataset Boundary' if boundary == dataset_boundaries[0] else "")
        #plt.text(x, 0 - 0.05, f"Dataset {i + 1}", rotation=90, va='top', ha='center', fontsize=9, color='gray')
    
    # Plot the metrics
    plt.plot(batch_x, value_accuracies, label="Value Accuracy")
    plt.plot(batch_x, policy_accuracies, label="Policy Accuracy")
    plt.plot(batch_x, value_losses, label="Value Loss")
    plt.plot(batch_x, policy_losses, label="Policy Loss")
    plt.plot(batch_x, total_losses, label="Total Loss")

    if len(test_x) > 0:
        for test_dataset_name in test_value_losses.keys():
            plt.plot(test_x, test_value_accuracies[test_dataset_name], label=f"Value Accuracy ({test_dataset_name})", linestyle='--')
            plt.plot(test_x, test_policy_accuracies[test_dataset_name], label=f"Policy Accuracy ({test_dataset_name})", linestyle='--')
            plt.plot(test_x, test_value_losses[test_dataset_name], label=f"Value Loss ({test_dataset_name})", linestyle='--')
            plt.plot(test_x, test_policy_losses[test_dataset_name], label=f"Policy Loss ({test_dataset_name})", linestyle='--')
            plt.plot(test_x, test_total_losses[test_dataset_name], label=f"Total Loss ({test_dataset_name})", linestyle='--')

    # Label and style
    #plt.ylim(0, 1)
    plt.xlabel("Batch")
    plt.ylabel("Performance")
    plt.title("Model Performance over Batches")
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/{fn}.png")
    plt.close()

    # plot each metric separately
    plot_single_metric(batch_x, total_losses, test_x, test_total_losses, dataset_boundaries, epoch_boundaries, "Total Loss", fn)
    plot_single_metric(batch_x, value_losses, test_x, test_value_losses, dataset_boundaries, epoch_boundaries, "Value Loss", fn)
    plot_single_metric(batch_x, policy_losses, test_x, test_policy_losses, dataset_boundaries, epoch_boundaries, "Policy Loss", fn)
    plot_single_metric(batch_x, value_accuracies, test_x, test_value_accuracies, dataset_boundaries, epoch_boundaries, "Value Accuracy", fn)
    plot_single_metric(batch_x, policy_accuracies, test_x, test_policy_accuracies, dataset_boundaries, epoch_boundaries, "Policy Accuracy", fn)

def plot_single_metric(x_train: list[int], metric_train: list[float],
                       x_test: list[int], metric_tests: dict[str, list[float]],
                       dataset_boundaries: list[int], epoch_boundaries: list[int],
                       label: str, fn: str):
    # set figure size
    plt.figure(figsize=(16, 9))

    # Add vertical lines for epoch boundaries
    for _, boundary in enumerate(epoch_boundaries):
        x=boundary - 0.5
        plt.axvline(x=x, color='gray', linestyle='--', alpha=0.35,
                    label='Epoch Boundary' if boundary == epoch_boundaries[0] else "")
        #plt.text(x, 0 - 0.05, f"Epoch {i + 1}", rotation=90, va='top', ha='center', fontsize=9, color='gray')
    
    # Add vertical lines for dataset boundaries
    for _, boundary in enumerate(dataset_boundaries):
        x=boundary - 0.5
        plt.axvline(x=x, color='gray', linestyle='--', alpha=0.7,
                    label='Dataset Boundary' if boundary == dataset_boundaries[0] else "")
        #plt.text(x, 0 - 0.05, f"Dataset {i + 1}", rotation=90, va='top', ha='center', fontsize=9, color='gray')

    plt.plot(x_train, metric_train, label=f"Train {label}")
    if metric_tests:
        for test_dataset_name, metric_test in metric_tests.items():
            plt.plot(x_test, metric_test, label=f"Test {label} ({test_dataset_name})", linestyle='--')

    # Label and style
    if "Accuracy" in label:
        plt.ylim(0, 1)
    plt.xlabel("Batch")
    plt.ylabel(label)
    plt.title(f"{label} over Batches")
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/{fn}_{label.replace(' ', '_').lower()}.png")
    plt.close()

def save_epoch_data(epoch_data_path: str, epoch_data: EpochData):
    with open(epoch_data_path, 'wb') as file:
        pickle.dump(epoch_data, file)
    logger.info(f"Saved epoch data to {epoch_data_path}.")

def load_epoch_data(epoch_data_path: str) -> EpochData:
    if not os.path.exists(epoch_data_path):
        raise FileNotFoundError(f"Epoch data file {epoch_data_path} does not exist.")
    with open(epoch_data_path, 'rb') as file:
        epoch_data: EpochData = pickle.load(file)
    logger.info(f"Loaded epoch data from {epoch_data_path}.")
    return epoch_data

def save_dataset_data(dataset_data_path: str, dataset_data: DatasetData):
    with open(dataset_data_path, 'wb') as file:
        pickle.dump(dataset_data, file)
    logger.info(f"Saved dataset data to {dataset_data_path}.")

def load_dataset_data(dataset_data_path: str) -> DatasetData:
    if not os.path.exists(dataset_data_path):
        raise FileNotFoundError(f"Dataset data file {dataset_data_path} does not exist.")
    with open(dataset_data_path, 'rb') as file:
        dataset_data: DatasetData = pickle.load(file)
    logger.info(f"Loaded dataset data from {dataset_data_path}.")
    return dataset_data

def stats_from_dataset_data(dataset_data: DatasetData) -> tuple[float, float, float, float, float]:
    """
    Extracts the average losses and accuracies from the dataset data.
    Returns a tuple of (avg_total_loss, avg_value_loss, avg_policy_loss, avg_value_accuracy, avg_policy_accuracy).
    """
    total_loss = 0.0
    total_value_loss = 0.0
    total_policy_loss = 0.0
    total_value_accuracy = 0.0
    total_policy_accuracy = 0.0
    num_batches = 0

    for epoch in dataset_data.epoch_data:
        for batch in epoch.batch_data:
            total_loss += batch.total_loss
            total_value_loss += batch.value_loss
            total_policy_loss += batch.policy_loss
            total_value_accuracy += batch.value_accuracy
            total_policy_accuracy += batch.policy_accuracy
            num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    avg_total_loss = total_loss / num_batches
    avg_value_loss = total_value_loss / num_batches
    avg_policy_loss = total_policy_loss / num_batches
    avg_value_accuracy = total_value_accuracy / num_batches
    avg_policy_accuracy = total_policy_accuracy / num_batches

    return avg_total_loss, avg_value_loss, avg_policy_loss, avg_value_accuracy, avg_policy_accuracy

def train_model_on_given_dataset(dataset: Dataset, num_epochs: int = 1, save_type: str = 'batch', test_dataset: Dataset | None = None) -> tuple[AlphaZeroModel, DatasetData]:
    # create a model
    model = AlphaZeroModel(
        config.board_size,
        training=True,
        rngs=nnx.Rngs({'params': jax.random.PRNGKey(0)})
    )
    if save_type != 'none':
        save_model(config.training_dir + '/model_0.pkl', model)
    
    model, dataset_data = train_model_epochs(
        model=model,
        dataset=dataset,
        num_epochs=num_epochs,
        save=save_type,
        test_dataset=test_dataset
    )

    plot_model_performance(f"model_performance_dataset_{1}", [dataset_data])
    save_dataset_data(f"{config.training_dir}/dataset_{1}_data.pkl", dataset_data)

    return model, dataset_data

def main():
    setup_logging(level=20, log_dir=config.log_dir, process_name="train_dataset")
    logger.info("Starting training with generated training data...")
    
    # Train the model with generated training data
    #train_model_with_generated_training_data(with_policy=True)

    #train_model_on_ground_truth_dataset(5)
    
    logger.info("Training completed successfully.")

if __name__ == "__main__":
    main()
