from jax import numpy as jnp
import optax
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from a0.model import AlphaZeroModel, load_model
from a0.dataset import Dataset

from config import config
from utils.log import get_logger, setup_logging
#     Plays a game with the given player and returns the training set and game data.
logger = get_logger(__name__)

def evaluate_model(model: AlphaZeroModel, evaluation_dataset: Dataset, model_no: int) -> tuple[float, float, float]:
    logger.info(f"Evaluating model on a dataset ({len(evaluation_dataset)})...")

    # shuffle the dataset and create batches generator
    evaluation_dataset.shuffle()
    batches = evaluation_dataset.batches()

    total_loss, total_value_loss, total_policy_loss = 0.0, 0.0, 0.0
    num_batches = 0
    
    for ts, batch in enumerate(batches):
        # convert the batch to a dictionary
        board_batch, value_batch, policy_batch = batch
        batch = {
            'board': board_batch,  # (N, board_size, board_size)
            'value': value_batch,  # (N, 1)
            'policy': policy_batch  # (N, board_size ** 4)
        }
        value, policy = model(batch['board'])
        #masked_policy = jnp.where(batch['policy'], policy, 0)
        value_classification = jnp.where(value <= 0, -1, 1)

        if model_no == 40:
            print(value[:10])  # predicted values
            print(batch['value'][:10])  # true values
            #exit()

        value_loss = jnp.mean(optax.l2_loss(value, batch['value']))
        value_loss = jnp.mean(value_classification == batch['value'])
        #policy_loss = jnp.mean(optax.softmax_cross_entropy(labels=batch['policy'], logits=masked_policy))
        
        loss = value_loss# + policy_loss

        total_loss += loss
        total_value_loss += value_loss
        #total_policy_loss += policy_loss
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_value_loss = total_value_loss / num_batches
    avg_policy_loss = total_policy_loss / num_batches

    logger.info(f"Validation Loss: {avg_loss}, Value Loss: {avg_value_loss}, Policy Loss: {avg_policy_loss}")
    return avg_loss, avg_value_loss, avg_policy_loss

def evaluate_all_models(evaluation_dataset: Dataset) -> None:
    logger.info("Evaluating all models in the training directory...")

    # Iterate through all model files in the training directory
    losses = []
    value_losses = []
    policy_losses = []
    for i in tqdm(range(config.training_iterations + 1)):
        model_path = f"{config.training_dir}/model_{i}.pkl"
        try:
            model = load_model(model_path)
            logger.info(f"Evaluating model {i + 1} at {model_path}")
            loss, value_loss, policy_loss = evaluate_model(model, evaluation_dataset, i + 1)
        except Exception as e:
            logger.error(f"Failed to load or evaluate model {i + 1} at {model_path}: {e}")
        
        losses.append(loss)
        value_losses.append(value_loss)
        policy_losses.append(policy_loss)
    
    # save the losses to a file
    losses_path = f"{config.data_folder}/evaluation_losses_accuracy.pkl"
    with open(losses_path, 'wb') as f:
        pickle.dump({
            'losses': losses,
            'value_losses': value_losses,
            'policy_losses': policy_losses
        }, f)

def plot_losses(losses: list[float], value_losses: list[float], policy_losses: list[float]) -> None:
    plt.plot(losses)
    plt.show()

def main():
    setup_logging(level=20, log_dir='logs/', process_name='dataset_evaluation')

    # Load the dataset
    dataset_path = f"{config.data_folder}/training_gtv.pkl"
    try:
        with open(dataset_path, 'rb') as file:
            evaluation_dataset: Dataset = pickle.load(file)
        logger.info(f"Loaded evaluation dataset from {dataset_path}.")
    except FileNotFoundError:
        logger.error(f"Dataset file not found at {dataset_path}. Please generate the dataset first.")
        return
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # Evaluate all models
    #evaluate_all_models(evaluation_dataset)

    # load the losses
    losses_path = f"{config.data_folder}/evaluation_losses_accuracy.pkl"
    try:
        with open(losses_path, 'rb') as f:
            losses_data = pickle.load(f)
        losses = losses_data['losses']
        value_losses = losses_data['value_losses']
        policy_losses = losses_data['policy_losses']
        logger.info(f"Loaded evaluation losses from {losses_path}.")
    except Exception as e:
        logger.error(f"Error loading losses: {e}")
        return

    # Plot the losses
    plot_losses(losses, value_losses, policy_losses)

if __name__ == "__main__":
    main()
