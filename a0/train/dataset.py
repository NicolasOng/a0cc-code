from typing import Optional, Any
import multiprocessing
import concurrent.futures
from concurrent.futures import Future, wait, FIRST_COMPLETED
import os
import pickle
import dill
import matplotlib.pyplot as plt
import math

import jax
from flax import nnx
import jax.numpy as jnp
import optax

from config import config

from a0.game import play, GameData
from a0.players.a0 import A0Player, board_to_input
from a0.model import AlphaZeroModel, load_model, save_model
from cc.core import Game
from a0.dataset import Dataset, TrainingData
from a0.eval.dataset_evaluation import load_dataset_list

from a0.model import AlphaZeroModel, load_model, save_model
from a0.dataset import Dataset
from a0.players.a0 import board_to_input

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

class BatchData:
    batch_size: int
    value_loss: float
    policy_loss: float
    total_loss: float
    model_no: int | None = None

class EpochData:
    batch_data: list[BatchData]
    model_no: int | None = None

    def __init__(self):
        self.batch_data = []

class DatasetData:
    epoch_data: list[EpochData]
    model_no: int | None = None

    def __init__(self):
        self.epoch_data = []

value_loss_function = optax.l2_loss
policy_loss_function = optax.softmax_cross_entropy

def loss_fn(model: AlphaZeroModel, batch: dict[str, Any]):
    value, policy = model(batch['board'])
    value_loss = jnp.mean(value_loss_function(value, batch['value']))
    #masked_policy = jnp.where(batch['policy'], policy, 0)
    #policy_loss = jnp.mean(policy_loss_function(labels=batch['policy'], logits=masked_policy))
    total_loss = value_loss# + policy_loss
    policy_loss = 0
    # JAX requires the loss function to return a tuple of (loss, aux)
    # where aux can be any additional information you want to return
    return total_loss, (value_loss, policy_loss)

@nnx.jit
def train_step(model: AlphaZeroModel, optimizer: nnx.Optimizer, batch: dict[str, Any]):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (value_loss, policy_loss)), grads = grad_fn(model, batch)
    optimizer.update(grads)
    return loss, value_loss, policy_loss

def plot_epoch_data(epoch_num: int, epoch_data: EpochData):
    # Extract losses
    value_losses = [bd.value_loss for bd in epoch_data.batch_data]
    policy_losses = [bd.policy_loss for bd in epoch_data.batch_data]
    total_losses = [bd.total_loss for bd in epoch_data.batch_data]
    steps = list(range(len(epoch_data.batch_data)))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(steps, value_losses, label='Value Loss')
    plt.plot(steps, policy_losses, label='Policy Loss')
    plt.plot(steps, total_losses, label='Total Loss')
    
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title(f'Losses Over Batches in Epoch {epoch_num}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/epoch_{epoch_num}_losses.png")
    plt.clf()

def save_epoch_data(epoch_num: int, epoch_data: EpochData):
    epoch_data_path = f"{config.training_dir}/epoch_{epoch_num}_data.pkl"
    with open(epoch_data_path, 'wb') as file:
        pickle.dump(epoch_data, file)
    logger.info(f"Saved epoch data to {epoch_data_path}.")

def train_model_epoch(model: AlphaZeroModel, dataset: Dataset, save: str = "None", cur_model_no: int = 0) -> tuple[AlphaZeroModel, EpochData, int]:
    logger.info(f"Training model on the given dataset ({len(dataset)})...")

    epoch_data = EpochData()

    # shuffle the dataset and create batches generator
    dataset.shuffle()
    batches = dataset.batches()

    num_batches = dataset.num_batches()
    batches_per_save = math.ceil(num_batches / 50)
    logger.info(f"Number of batches: {num_batches}")
    logger.info(f"Batches per save: {batches_per_save}")
    logger.info(f"Total models: {num_batches // batches_per_save}")

    optimizer = nnx.Optimizer(model, optax.adamw(0.00005))

    for ts, batch in enumerate(batches):
        # convert the batch to a dictionary
        board_batch, value_batch, policy_batch = batch
        batch = {
            'board': board_batch,  # (N, board_size, board_size)
            'value': value_batch,  # (N, 1)
            'policy': policy_batch  # (N, board_size ** 4)
        }
        loss, value_loss, policy_loss = train_step(model, optimizer, batch)
        logger.info(f"Training Step {ts}, Loss: {loss}, Value Loss: {value_loss}, Policy Loss: {policy_loss}")
        print(f"Training Step {ts}, Loss: {loss}, Value Loss: {value_loss}, Policy Loss: {policy_loss}")
        
        # create batch data
        batch_data = BatchData()
        batch_data.batch_size = len(batch['board'])
        batch_data.value_loss = value_loss
        batch_data.policy_loss = policy_loss
        batch_data.total_loss = loss

        if save == "batch" and (ts + 1) % batches_per_save == 0:
            # Save the model after every save_batch_amount batches
            cur_model_no += 1
            logger.info(f"Saving model {cur_model_no} after batch {ts + 1}...")
            save_model(config.training_dir + f'/model_{cur_model_no}.pkl', model)
            batch_data.model_no = cur_model_no
        
        # add the batch data to the epoch data
        epoch_data.batch_data.append(batch_data)

    return model, epoch_data, cur_model_no

def plot_dataset_data(dataset_num: int, dataset_data: DatasetData):
    all_epochs = dataset_data.epoch_data
    # Collect losses and labels
    value_losses: list[float] = []
    policy_losses: list[float] = []
    total_losses: list[float] = []
    batch_labels: list[str] = []

    for _, epoch in enumerate(all_epochs):
        for batch_idx, batch in enumerate(epoch.batch_data):
            value_losses.append(batch.value_loss)
            policy_losses.append(batch.policy_loss)
            total_losses.append(batch.total_loss)
            batch_labels.append(f"{batch_idx + 1}")

    # Epoch boundary positions (between last and first batch of adjacent epochs)
    epoch_boundaries: list[int] = []
    cur_boundary = 0
    for epoch in all_epochs[:-1]:  # Exclude the last epoch for boundaries
        cur_boundary += len(epoch.batch_data)
        epoch_boundaries.append(cur_boundary)

    # Plot losses
    plt.figure(figsize=(12, 6))
    plt.plot(value_losses, label="Value Loss")
    plt.plot(policy_losses, label="Policy Loss")
    plt.plot(total_losses, label="Total Loss")

    # Add vertical lines for epoch boundaries
    for i, boundary in enumerate(epoch_boundaries):
        x=boundary - 0.5
        plt.axvline(x=x, color='gray', linestyle='--', alpha=0.7,
                    label='Epoch Boundary' if boundary == epoch_boundaries[0] else "")
        plt.text(x, 0 - 0.05,
             f"Epoch {i + 1}", rotation=90, va='top', ha='center', fontsize=9, color='gray')

    # Label and style
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Losses over Batches with Epoch Boundaries")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/dataset_{dataset_num}_losses.png")
    plt.clf()

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

def train_model_epochs(model: AlphaZeroModel, dataset: Dataset, num_epochs: int, save: str = "None", plot: bool = True) -> tuple[AlphaZeroModel, DatasetData]:
    """
    Train the model for a number of epochs on the given dataset.
    """
    dataset_data = DatasetData()
    cur_model_no = 0
    for epoch in range(num_epochs):
        logger.info(f"Training epoch {epoch + 1}/{num_epochs}...")
        model, epoch_data, cur_model_no = train_model_epoch(model, dataset, save, cur_model_no)
        if save == "epoch":
            # Save the model after each epoch
            logger.info(f"Saving model after epoch {epoch + 1}...")
            save_model(config.training_dir + f'/model_{epoch + 1}.pkl', model)
            epoch_data.model_no = epoch + 1
        dataset_data.epoch_data.append(epoch_data)
        if plot:
            plot_epoch_data(epoch + 1, epoch_data)
            save_epoch_data(epoch + 1, epoch_data)
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

def train_model_on_given_dataset(dataset: Dataset, num_epochs: int = 1, save_type: str = 'batch') -> tuple[AlphaZeroModel, DatasetData]:
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
        save=save_type
    )

    plot_dataset_data(1, dataset_data)
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
