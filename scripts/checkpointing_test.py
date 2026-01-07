import os
import time

from cc.core import Board, Game
from cc.ground_truth import GroundTruth
from a0.dataset import Dataset
from a0.train.dataset import DatasetData
from a0.train.dataset import train_model_epochs, plot_model_performance
from a0.model import AlphaZeroModel, load_model, save_model
from scripts.sl_on_policy_head import get_n_random_states, create_gtd_from_states

from flax import nnx
import jax

import pickle

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def coprime_stepping_bijective_function(index: int, a: int, b: int, n: int) -> int:
    '''
    A simple bijective function that maps integers from 0 to n-1 to integers from 0 to n-1 (loops)
    using coprime stepping. This is used to shuffle the order of training samples.
    index: the index to map
    a: coprime integer to n
    b: integer offset
    n: the size of the set to map to
    returns: the mapped index
    '''
    return (a * index + b) % n

def cpsbf_test():
    n = 100 # 9,610,154,400 for 25-6
    a = 7  # 6,406,769,569 for 25-6 (prime therefore coprime and ~2/3 of n)
    b = 3
    # n = 9610154400
    # a = 6406769569
    # b = 4397420414
    seen: set[int] = set()
    for i in range(n):
        mapped = coprime_stepping_bijective_function(i, a, b, n)
        assert 0 <= mapped < n
        assert mapped not in seen
        seen.add(mapped)
    assert len(seen) == n
    logger.info("Coprime stepping bijective function test passed.")
    logger.info(f"Mapping for n={n}, a={a}, b={b}: {[coprime_stepping_bijective_function(i, a, b, n) for i in range(n)]}")

def get_states_with_coprime_stepping(starting_index: int, number_of_boards: int, remove_trivial: bool = True, remove_terminal: bool = True) -> tuple[list[Board], int]:
    '''
    Gets n random states using the coprime stepping bijective function to shuffle the order.
    '''
    gt = GroundTruth()
    game = Game(board_size=config.board_size, num_pieces=config.num_pieces)
    max_rank = gt.get_max_rank()
    a = config.coprime_stepping_a
    b = config.coprime_stepping_b
    state_list: list[Board] = []
    for i in range(number_of_boards):
        index = (starting_index + i) % max_rank
        mapped_index = coprime_stepping_bijective_function(index, a, b, max_rank)
        board = gt.unrank(mapped_index)
        if remove_trivial and gt.is_trivial(board):
            continue
        if remove_terminal and game.get_done(board):
            continue
        state_list.append(board)
    return state_list, (starting_index + number_of_boards) % max_rank

def get_new_model() -> AlphaZeroModel:
    model = AlphaZeroModel(
        board_size=config.board_size,
        num_filters=256,
        training=True,
        rngs=nnx.Rngs({'params': jax.random.PRNGKey(1)}),
        num_resblocks=3
    )
    return model

def train_on_datasets(model: AlphaZeroModel, dataset: Dataset, eval_datasets: dict[str, Dataset], num_epochs: int = 1) -> tuple[AlphaZeroModel, DatasetData]:
    '''
    Trains the given model on the given dataset for num_epochs epochs,
    then evaluates it on the eval_dataset.
    Plots the training performance.
    '''
    logger.info(f"Training given model for {num_epochs} epochs...")
    trained_model, dsd = train_model_epochs(model, dataset, num_epochs, save="None", plot=False, test_datasets=eval_datasets)
    return trained_model, dsd

def train_with_checkpointing(epochs_per_checkpoint: int = 1, states_per_epoch: int = 1000) -> None:
    # load the model, training data, and current index from disk (if they exist)
    if os.path.exists(config.training_dir + "model_checkpoint.pkl"):
        logger.info("Loading model from checkpoint...")
        trained_model = load_model(config.training_dir + "model_checkpoint.pkl", training=True)
    else:
        logger.info("Creating new model...")
        trained_model = get_new_model()
    
    if os.path.exists(config.training_dir + "datasetdata.pkl"):
        logger.info("Loading dataset data from checkpoint...")
        with open(config.training_dir + "datasetdata.pkl", "rb") as f:
            datasetdata: DatasetData = pickle.load(f)
    else:
        logger.info("Creating new dataset data...")
        datasetdata = DatasetData()
    
    if os.path.exists(config.training_dir + "current_index.txt"):
        logger.info("Loading current index from checkpoint...")
        with open(config.training_dir + "current_index.txt", "r") as f:
            current_index = int(f.read())
    else:
        logger.info("Starting from index 0...")
        current_index = 0

    for e in range(epochs_per_checkpoint):
        logger.info(f"Starting checkpoint epoch {e + 1} of {epochs_per_checkpoint}...")
        # generate a new dataset to train on
        logger.info(f"Generating new training dataset with current index {current_index}...")
        randomish_states, current_index = get_states_with_coprime_stepping(current_index, states_per_epoch)
        dataset = create_gtd_from_states(randomish_states)

        # generate an evaluation dataset
        logger.info("Generating evaluation dataset...")
        eval_dataset = create_gtd_from_states(get_n_random_states(5000))

        # train on that dataset
        logger.info("Training on dataset...")
        trained_model, dsd = train_on_datasets(trained_model, dataset, {"eval": eval_dataset}, num_epochs=1)
        # add the training data to the checkpointed
        datasetdata.epoch_data.extend(dsd.epoch_data)

        # plot the model performance so far
        logger.info("Plotting model performance...")
        plot_model_performance("checkpointed_model", [datasetdata])
        # save the checkpointed data and model to disk
        logger.info("Saving checkpointed model and dataset data...")
        with open(config.training_dir + "datasetdata.pkl", "wb") as f:
            pickle.dump(datasetdata, f)
        
        save_model(config.training_dir + "model_checkpoint.pkl", trained_model)

        # save the current index to disk
        logger.info("Saving current index...")
        with open(config.training_dir + "current_index.txt", "w") as f:
            f.write(str(current_index))

def main(epochs_per_checkpoint: int = 1, states_per_epoch: int = 1000) -> None:
    start_time = time.time()

    train_with_checkpointing(epochs_per_checkpoint=epochs_per_checkpoint, states_per_epoch=states_per_epoch)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Checkpointed training completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="checkpointing_test"
    )

    #train_with_checkpointing(epochs_per_checkpoint=10, states_per_epoch=100000)
    #cpsbf_test()
    main(epochs_per_checkpoint=10, states_per_epoch=100000)
