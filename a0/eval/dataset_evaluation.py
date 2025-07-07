import sys

from jax import numpy as jnp
import optax
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from a0.model import AlphaZeroModel, load_model
from a0.dataset import Dataset

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def evaluate_model(model: AlphaZeroModel, evaluation_dataset: Dataset, model_no: int) -> tuple[float, float, float, float]:
    '''
    Evaluates the model on the given evaluation dataset.
    The evaluation dataset is a Dataset object containing
    - game states
    - target values
    - target policies
    Returns the average loss, value loss, policy loss, and accuracy.
    And any other metrics I might want to add later.
    TODO: I could make this generic for any model that can predict values and policies.
    '''
    logger.info(f"Evaluating model on a dataset ({len(evaluation_dataset)})...")

    # shuffle the dataset and create batches generator
    evaluation_dataset.shuffle()
    batches = evaluation_dataset.batches()

    total_loss, total_value_loss, total_policy_loss = 0.0, 0.0, 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    for ts, batch in tqdm(enumerate(batches)):
        # convert the batch to a dictionary
        board_batch, value_batch, policy_batch = batch
        batch = {
            'board': board_batch,  # (N, board_size, board_size)
            'value': value_batch,  # (N, 1)
            'policy': policy_batch  # (N, board_size ** 4)
        }
        value, policy = model(batch['board'])
        masked_policy = jnp.where(batch['policy'], policy, 0)
        # TODO: what about boards what value=0? how to handle those? and how many are there?
        value_classification = jnp.where(value <= 0, -1, 1)

        # Calculate actual loss (L2)
        value_loss = float(jnp.mean(optax.l2_loss(value, batch['value'])))
        
        # Calculate accuracy (fraction of correct classifications)
        accuracy = float(jnp.mean(value_classification == batch['value']))
        policy_loss = float(jnp.mean(optax.softmax_cross_entropy(labels=batch['policy'], logits=masked_policy)))

        # if model_no > 15 and (ts - 1) % 10 == 0:
        #     logger.info("Predicted | Actual")
        #     preds = value_classification.flatten()
        #     labels = batch['value'].flatten()
        #     for pred, actual in zip(preds[:10], labels[:10]):
        #         logger.info(f"{int(pred):9} | {int(actual)}")
        
        loss = value_loss# + policy_loss

        total_loss += loss
        total_value_loss += value_loss
        total_accuracy += accuracy
        total_policy_loss += policy_loss
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_value_loss = total_value_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    avg_policy_loss = total_policy_loss / num_batches

    logger.info(f"Validation Loss: {avg_loss}, Value Loss: {avg_value_loss}, Policy Loss: {avg_policy_loss}, Accuracy: {avg_accuracy}")
    return avg_loss, avg_value_loss, avg_policy_loss, avg_accuracy

def evaluate_all_models(models: list[AlphaZeroModel], evaluation_dataset: Dataset, fn: str) -> None:
    '''
    Evaluates all models in the training directory.
    The training directory is defined in the config.
    They are all evaluated on the same evaluation dataset.
    '''
    logger.info(f"Evaluating all models in the training directory ({fn})...")

    # Iterate through all model files in the training directory
    losses: list[float] = []
    value_losses: list[float] = []
    policy_losses: list[float] = []
    value_accuracies: list[float] = []
    for i, model in tqdm(enumerate(models)):
        logger.info(f"Evaluating model {i + 1}")
        loss, value_loss, policy_loss, value_accuracy = evaluate_model(model, evaluation_dataset, i + 1)
        losses.append(loss)
        value_losses.append(value_loss)
        policy_losses.append(policy_loss)
        value_accuracies.append(value_accuracy)
    
    # save the losses to a file
    losses_path = f"{config.eval_dir}/{fn}.pkl"
    with open(losses_path, 'wb') as f:
        pickle.dump({
            'losses': losses,
            'value_losses': value_losses,
            'policy_losses': policy_losses,
            'value_accuracies': value_accuracies
        }, f)

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
    for i, (model, dataset) in tqdm(enumerate(zip(models, datasets))):
        logger.info(f"Evaluating model {i + 1}")
        loss, value_loss, policy_loss, value_accuracy = evaluate_model(model, dataset, i + 1)
        losses.append(loss)
        value_losses.append(value_loss)
        policy_losses.append(policy_loss)
        value_accuracies.append(value_accuracy)
    
    # save the losses to a file
    losses_path = f"{config.eval_dir}/{fn}.pkl"
    with open(losses_path, 'wb') as f:
        pickle.dump({
            'losses': losses,
            'value_losses': value_losses,
            'policy_losses': policy_losses,
            'value_accuracies': value_accuracies
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

def load_models(dir: str, n: int) -> list[AlphaZeroModel]:
    '''
    Loads n models from the given directory.
    Returns a list of AlphaZeroModel instances.
    File names are expected to be in the format "model_{i}.pkl" where i is the model number.
    Loads models from 0 to n (inclusive).
    '''
    logger.info(f"Loading {n} models from {dir}...")
    models: list[AlphaZeroModel] = []
    for i in range(n + 1):
        model_path = f"{dir}/model_{i}.pkl"
        try:
            model = load_model(model_path)
            models.append(model)
        except Exception as e:
            logger.error(f"Failed to load model {i + 1} at {model_path}: {e}")
    return models

def load_losses(losses_path: str) -> tuple[list[float], list[float], list[float], list[float]]:
    '''
    Loads losses from the given path.
    Returns a tuple of lists: (losses, value_losses, policy_losses, value_accuracies).
    If the file does not exist, it will log an error and exit.
    '''
    try:
        with open(losses_path, 'rb') as f:
            losses_data = pickle.load(f)
        logger.info(f"Loaded evaluation losses from {losses_path}.")
        return (losses_data['losses'], losses_data['value_losses'], 
                losses_data['policy_losses'], losses_data['value_accuracies'])
    except FileNotFoundError:
        logger.error(f"Losses file not found at {losses_path}. Please generate the losses first.")
        sys.exit()
    except Exception as e:
        logger.error(f"Error loading losses: {e}")
        sys.exit()

def plot_losses(losses: list[float], value_losses: list[float], policy_losses: list[float], value_accuracies: list[float], fn: str) -> None:
    plt.plot(losses, label='Loss')
    plt.plot(value_losses, label='Value Loss')
    plt.plot(policy_losses, label='Policy Loss')
    plt.plot(value_accuracies, label='Value Accuracy')
    plt.legend()
    plt.savefig(f"{config.plot_dir}{fn}.png")
    plt.clf()

def plot_two_accuracies(accuracies1: list[float], accuracies2: list[float], a1n: str, a2n: str, fn: str) -> None:
    plt.plot(accuracies1, label=a1n)
    plt.plot(accuracies2, label=a2n)
    plt.legend()
    plt.savefig(f"{config.plot_dir}{fn}.png")
    plt.clf()

def trim_dataset(dataset: Dataset, n: int) -> Dataset:
    '''
    Trims the dataset to the first n elements.
    Returns a new Dataset object with the trimmed data.
    '''
    logger.info(f"Trimming dataset to {n} elements...")
    dataset.states = dataset.states[:n]
    dataset.values = dataset.values[:n]
    dataset.policies = dataset.policies[:n]
    return dataset

def main():
    setup_logging(level=20, log_dir=config.log_dir, process_name='dataset_evaluation')

    logger.info("Starting dataset evaluation...")

    # load the models
    models = load_models(config.training_dir, config.training_iterations + 1)

    # Load the datasets
    training_dataset = load_dataset(f"{config.eval_dir}/training_gtv.pkl")
    random_dataset = load_dataset(f"{config.eval_dir}/random_gtv.pkl")
    training_e_dataset = load_dataset(f"{config.eval_dir}/training_ev.pkl")
    
    # trim down the training dataset to a smaller size for faster evaluation
    n = 1000
    training_dataset = trim_dataset(training_dataset, n)
    training_e_dataset = trim_dataset(training_e_dataset, n)
    
    # Evaluate all models
    evaluate_all_models(models, training_dataset, "training_eval")
    evaluate_all_models(models, random_dataset, "random_eval")
    evaluate_all_models(models, training_e_dataset, "training_e_eval")

    # load and plot the losses
    tlosses, tvalue_losses, tpolicy_losses, tvalue_accuracies = load_losses(f"{config.eval_dir}/training_eval.pkl")
    plot_losses(tlosses, tvalue_losses, tpolicy_losses, tvalue_accuracies, "training_eval")
    rlosses, rvalue_losses, rpolicy_losses, rvalue_accuracies = load_losses(f"{config.eval_dir}/random_eval.pkl")
    plot_losses(rlosses, rvalue_losses, rpolicy_losses, rvalue_accuracies, "random_eval")
    telosses, tevalue_losses, tepolicy_losses, tevalue_accuracies = load_losses(f"{config.eval_dir}/training_e_eval.pkl")
    plot_losses(telosses, tevalue_losses, tepolicy_losses, tevalue_accuracies, "training_e_eval")
    plot_two_accuracies(tvalue_accuracies, rvalue_accuracies, "Training Accuracy", "Random Accuracy", "training_vs_random_accuracy")

    # load the training data dataset
    training_datasets = load_dataset_list(f"{config.eval_dir}/training_datasets.pkl")
    evaluate_all_models_progressive(models, training_datasets, "training_perf")
    losses, value_losses, policy_losses, value_accuracies = load_losses(f"{config.eval_dir}/training_perf.pkl")
    plot_losses(losses, value_losses, policy_losses, value_accuracies, "training_perf")

    logger.info("Dataset evaluation completed.")


if __name__ == "__main__":
    main()
