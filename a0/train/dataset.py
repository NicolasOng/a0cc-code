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
    test_metrics: TestData | None = None

class EpochData:
    batch_data: list[BatchData]
    time: Optional[float] = None

    def __init__(self):
        self.batch_data = []

class DatasetData:
    epoch_data: list[EpochData]

    def __init__(self):
        self.epoch_data = []

value_loss_function = optax.l2_loss
policy_loss_function = optax.softmax_cross_entropy

def loss_fn(model: AlphaZeroModel, batch: dict[str, Any]):
    # get the model's predictions
    value, policy = model(batch['board'])

    # calculate the value loss and accuracy
    value_loss = jnp.mean(value_loss_function(value, batch['value']))
    value_classification = jnp.where(value <= 0, -1, 1)
    value_accuracy = jnp.mean(value_classification == batch['value']).astype(float)

    # create legal move mask (non-zero entries in policy labels)
    legal_mask = batch['policy'] > 0

    # calculate the policy loss
    # note that the policy loss function (optax.softmax_cross_entropy) expects logits,
    # so we set illegal moves to a very low value (e.g., -1e9)
    masked_logits = jnp.where(legal_mask, policy, -1e9)
    masked_logits = policy
    policy_loss = jnp.mean(policy_loss_function(labels=batch['policy'], logits=masked_logits))

    # calculate the accuracy of the policy
    max_value = jnp.max(batch['policy'], axis=1, keepdims=True)  # Keep batch dimension
    is_max = batch['policy'] == max_value  # Boolean mask for max values
    max_pred_indices = jnp.argmax(policy, axis=1)  # Shape: (batch_size,)
    
    # Create one-hot encoding of predicted max indices
    pred_one_hot = jax.nn.one_hot(max_pred_indices, num_classes=batch['policy'].shape[1])
    
    # Check if the predicted max index corresponds to any of the true max indices
    policy_accuracy = jnp.mean(jnp.any(is_max * pred_one_hot, axis=1)).astype(float)

    # calculate the total loss
    total_loss = value_loss + policy_loss
    #policy_loss = 0
    # JAX requires the loss function to return a tuple of (loss, aux)
    # where aux can be any additional information you want to return
    return total_loss, (value_loss, policy_loss, value_accuracy, policy_accuracy)

@nnx.jit
def train_step(model: AlphaZeroModel, optimizer: nnx.Optimizer, batch: dict[str, Any]):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (value_loss, policy_loss, value_accuracy, policy_accuracy)), grads = grad_fn(model, batch)
    optimizer.update(grads)
    return loss, value_loss, policy_loss, value_accuracy, policy_accuracy

def train_model_epoch(model: AlphaZeroModel, dataset: Dataset, save: str = "None", cur_model_no: int = 0, test_dataset: Dataset | None = None) -> tuple[AlphaZeroModel, EpochData, int]:
    logger.info(f"Training model on the given dataset ({len(dataset)})...")
    start = time.perf_counter()

    epoch_data = EpochData()

    # shuffle the dataset and create batches generator
    dataset.shuffle()
    batches = dataset.batches()

    num_batches = dataset.num_batches()
    batches_per_save = math.ceil(num_batches / 50)
    logger.info(f"Number of batches: {num_batches}")
    logger.info(f"Batches per save: {batches_per_save}")
    logger.info(f"Total models: {num_batches // batches_per_save}")

    # value: 0.00005
    optimizer = nnx.Optimizer(model, optax.adamw(0.00005))

    for ts, batch in enumerate(batches):
        # convert the batch to a dictionary
        board_batch, value_batch, policy_batch = batch
        batch = {
            'board': board_batch,  # (N, board_size, board_size)
            'value': value_batch,  # (N, 1)
            'policy': policy_batch  # (N, board_size ** 4)
        }
        loss, value_loss, policy_loss, value_accuracy, policy_accuracy = train_step(model, optimizer, batch)
        logger.info(f"Training Step {ts}/{num_batches}, Loss: {loss}, Value Loss: {value_loss}, Policy Loss: {policy_loss}, Value Accuracy: {value_accuracy}, Policy Accuracy: {policy_accuracy}")
        
        # create batch data
        batch_data = BatchData()
        batch_data.batch_size = len(batch['board'])
        batch_data.value_loss = value_loss
        batch_data.policy_loss = policy_loss
        batch_data.total_loss = loss
        batch_data.value_accuracy = value_accuracy
        batch_data.policy_accuracy = policy_accuracy

        if test_dataset is not None and (ts + 1) % batches_per_save == 0:
            # evaluate the model on the test dataset every ... batches
            avg_loss, avg_value_loss, avg_policy_loss, avg_value_accuracy, avg_policy_accuracy = evaluate_model(model, test_dataset, cur_model_no)
            test_data = TestData()
            test_data.value_loss = avg_value_loss
            test_data.policy_loss = avg_policy_loss
            test_data.total_loss = avg_loss
            test_data.value_accuracy = avg_value_accuracy
            test_data.policy_accuracy = avg_policy_accuracy
            batch_data.test_metrics = test_data

        if save == "batch" and (ts + 1) % batches_per_save == 0:
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

def train_model_epochs(model: AlphaZeroModel, dataset: Dataset, num_epochs: int, save: str = "None", plot: bool = True, test_dataset: Dataset | None = None) -> tuple[AlphaZeroModel, DatasetData]:
    """
    Train the model for a number of epochs on the given dataset.
    """
    dataset_data = DatasetData()
    cur_model_no = 0
    for epoch in range(num_epochs):
        logger.info(f"Training epoch {epoch + 1}/{num_epochs}...")
        model, epoch_data, cur_model_no = train_model_epoch(model, dataset, save, cur_model_no, test_dataset)
        if save == "epoch":
            # Save the model after each epoch
            logger.info(f"Saving model after epoch {epoch + 1}...")
            save_model(config.training_dir + f'/model_{epoch + 1}.pkl', model)
            epoch_data.batch_data[-1].model_no = epoch + 1
        dataset_data.epoch_data.append(epoch_data)
        if plot:
            save_epoch_data(epoch + 1, epoch_data)
            temp_dd = DatasetData()
            temp_dd.epoch_data.append(epoch_data)
            plot_model_performance(f"epoch_{epoch + 1}", [temp_dd])
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
    test_value_losses: list[float] = []
    test_policy_losses: list[float] = []
    test_total_losses: list[float] = []
    test_value_accuracies: list[float] = []
    test_policy_accuracies: list[float] = []
    for _, dataset in enumerate(dataset_datas):
        for _, epoch in enumerate(dataset.epoch_data):
            for _, batch in enumerate(epoch.batch_data):
                batch_x.append(batch_i)
                value_losses.append(batch.value_loss)
                policy_losses.append(batch.policy_loss)
                total_losses.append(batch.total_loss)
                value_accuracies.append(batch.value_accuracy)
                policy_accuracies.append(batch.policy_accuracy)
                if batch.test_metrics is not None:
                    test_x.append(batch_i)
                    test_value_losses.append(batch.test_metrics.value_loss)
                    test_policy_losses.append(batch.test_metrics.policy_loss)
                    test_total_losses.append(batch.test_metrics.total_loss)
                    test_value_accuracies.append(batch.test_metrics.value_accuracy)
                    test_policy_accuracies.append(batch.test_metrics.policy_accuracy)
                batch_i += 1
    
    # get dataset and epoch boundaries
    dataset_boundaries: list[int] = []
    epoch_boundaries: list[int] = []
    cur_boundary = 0
    for dataset in dataset_datas:
        for epoch in dataset.epoch_data:
            # Add the number of batches in the epoch to the current boundary
            cur_boundary += len(epoch.batch_data)
            epoch_boundaries.append(cur_boundary)
        # Rrmove the last epoch boundary in each dataset
        epoch_boundaries = epoch_boundaries[:-1]
        dataset_boundaries.append(cur_boundary)

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
    plt.figure(figsize=(12, 6))
    plt.plot(batch_x, value_accuracies, label="Value Accuracy")
    plt.plot(batch_x, policy_accuracies, label="Policy Accuracy")
    plt.plot(batch_x, value_losses, label="Value Loss")
    plt.plot(batch_x, policy_losses, label="Policy Loss")
    plt.plot(batch_x, total_losses, label="Total Loss")

    if len(test_x) > 0:
        plt.plot(test_x, test_value_accuracies, label="Test Value Accuracy", linestyle='--')
        plt.plot(test_x, test_policy_accuracies, label="Test Policy Accuracy", linestyle='--')
        plt.plot(test_x, test_value_losses, label="Test Value Loss", linestyle='--')
        plt.plot(test_x, test_policy_losses, label="Test Policy Loss", linestyle='--')
        plt.plot(test_x, test_total_losses, label="Test Total Loss", linestyle='--')

    # Label and style
    plt.xlabel("Batch")
    plt.ylabel("Performance")
    plt.title("Model Performance over Batches")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/model_performance_{fn}.png")
    plt.clf()

def save_epoch_data(epoch_num: int, epoch_data: EpochData):
    epoch_data_path = f"{config.training_dir}/epoch_{epoch_num}_data.pkl"
    with open(epoch_data_path, 'wb') as file:
        pickle.dump(epoch_data, file)
    logger.info(f"Saved epoch data to {epoch_data_path}.")

def save_dataset_data(dataset_num: int, dataset_data: DatasetData):
    dataset_data_path = f"{config.training_dir}/dataset_{dataset_num}_data.pkl"
    with open(dataset_data_path, 'wb') as file:
        pickle.dump(dataset_data, file)
    logger.info(f"Saved dataset data to {dataset_data_path}.")

def load_dataset_data(dataset_num: int) -> DatasetData:
    dataset_data_path = f"{config.training_dir}/dataset_{dataset_num}_data.pkl"
    if not os.path.exists(dataset_data_path):
        raise FileNotFoundError(f"Dataset data file {dataset_data_path} does not exist.")
    with open(dataset_data_path, 'rb') as file:
        dataset_data: DatasetData = pickle.load(file)
    logger.info(f"Loaded dataset data from {dataset_data_path}.")
    return dataset_data

def train_model_with_generated_training_data(with_policy: bool = True):
    '''
    load the training datasets from the eval directory and train the model on them.
    The training datasets are generated by the self-play process and saved in the eval directory.
    '''
    # load the datasets
    logger.info("Loading training datasets...")
    training_datasets = load_dataset_list(f"{config.eval_dir}/training_datasets.pkl")

    # create a model
    model = AlphaZeroModel(
        config.board_size,
        training=True,
        rngs=nnx.Rngs({'params': jax.random.PRNGKey(0)})
    )
    save_model(config.training_dir + '/model_0.pkl', model)

    # train a new model on the training datasets
    train_model_datasets(
        model=model,
        datasets=training_datasets,
        num_epochs=1,
        save_epoch=False,
        save_dataset=True
    )

def train_model_on_given_dataset(dataset: Dataset, num_epochs: int = 1, save_type: str = 'batch', test_dataset: Dataset | None = None) -> tuple[AlphaZeroModel, DatasetData]:
    # create a model
    model = AlphaZeroModel(
        config.board_size,
        training=True,
        rngs=nnx.Rngs({'params': jax.random.PRNGKey(0)})
    )
    save_model(config.training_dir + '/model_0.pkl', model)
    
    model, dataset_data = train_model_epochs(
        model=model,
        dataset=dataset,
        num_epochs=num_epochs,
        save=save_type,
        test_dataset=test_dataset
    )

    plot_model_performance(f"dataset_{1}", [dataset_data])
    save_dataset_data(1, dataset_data)

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
