import os
import pickle
import random
import time

from a0.dataset import Dataset

from scripts.sl_on_policy_head import check_random_policy_acc

from a0.eval3.generate_datasets import get_nd_and_nt_datasets_from_state_list, get_random_state_list, get_training_dataset
from cc.ground_truth import GroundTruth
from a0.utils.states import get_gtd_from_states

from a0.train.dataset import DatasetData, make_optimizer, train_model_epochs, plot_model_performance
from a0.model import AlphaZeroModel
from flax import nnx
import jax

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def train_and_plot_datasets(fn: str, dataset: Dataset, eval_datasets: dict[str, Dataset], num_epochs: int = 10, res_blocks: int = 3) -> tuple[AlphaZeroModel, DatasetData]:
    '''
    Trains a model on the given dataset for num_epochs epochs,
    then evaluates it on the eval_dataset.
    Plots the training performance.
    '''
    logger.info(f"Training model '{fn}' for {num_epochs} epochs...")

    model = AlphaZeroModel(
        board_size=config.board_size,
        num_filters=256,
        rngs=nnx.Rngs({'params': jax.random.PRNGKey(1)}),
        num_resblocks=res_blocks
    )

    optimizer = make_optimizer(model)

    trained_model, dsd = train_model_epochs(model, dataset, num_epochs, save="None", plot=False, test_datasets=eval_datasets, optimizer=optimizer)
    plot_model_performance(fn, [dsd])

    return trained_model, dsd

def training_run(dataset_type: str, dataset_size: int, model_size: int, training_dataset: Dataset, eval_datasets: dict[str, Dataset]) -> DatasetData:
    logger.info(f"Starting training run: dataset_type={dataset_type}, dataset_size={dataset_size}, model_size={model_size}")
    start_time = time.time()
    _, dsd = train_and_plot_datasets(
        fn=f"{dataset_type}_{dataset_size}_{model_size}",
        dataset=training_dataset,
        eval_datasets=eval_datasets,
        num_epochs=20,
        res_blocks=model_size
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Finished training run: dataset_type={dataset_type}, dataset_size={dataset_size}, model_size={model_size} in {elapsed_time:.2f} seconds")
    return dsd

def train_models_sequential() -> None:
    # create all the datasets:
    # training datasets of size n of both types (gt and rand_gt) (2 datasets)
    # test datasets for each type, of size 10K (2 datasets)
    # evaluation datasets: seen and random, of size 10K each (2 seen + 1 rand = 3 datasets)
    logger.info("Creating datasets...")
    # note: just one
    dataset_sizes = [1024, 8192, 65536, 524288]
    n = 5000
    #dataset_types = ['gt', 'rand_gt']
    dataset_types = ['gt']

    logger.info("Creating test datasets...")
    gt = GroundTruth()
    test_nd, test_nt = get_nd_and_nt_datasets_from_state_list(get_random_state_list(n, gt), gt, n, 256)
    
    logger.info("Creating training datasets...")
    logger.info(f"Dataset sizes: {dataset_sizes}, types: {dataset_types}")
    training_datasets: dict[str, tuple[Dataset, dict[str, Dataset]]] = {}
    max_dataset_size = max(dataset_sizes)
    _, training_states = get_training_dataset(max_dataset_size, gt)
    random_acc = check_random_policy_acc(training_states)
    logger.info(f"Random policy accuracy on {max_dataset_size} states: {random_acc:.2%}")

    for size in dataset_sizes:
        random_states = random.sample(training_states, min(size, len(training_states)))
        seen_states = random.sample(random_states, min(n, len(random_states)))
        for dtype in dataset_types:
            dataset_name = f"{dtype}_{size}"
            rgtd = get_gtd_from_states(random_states, gt, batch_size=256, shuffle=True)
            seen_nd, seen_nt = get_nd_and_nt_datasets_from_state_list(seen_states, gt, n, 256)
            training_datasets[dataset_name] = (
                rgtd,
                {
                    'test_nd': test_nd,
                    'test_nt': test_nt,
                    'seen_nd': seen_nd,
                    'seen_nt': seen_nt
                }
            )

    # set the model sizes to test
    model_sizes = [1, 2, 3, 4, 5]
    # model_sizes = [3]
    logger.info(f"Model sizes to test: {model_sizes}")

    # run all training runs sequentially
    for size in dataset_sizes:
        for dtype in dataset_types:
            for msize in model_sizes:
                training_dataset, eval_datasets = training_datasets[f"{dtype}_{size}"]
                dsd = training_run(dtype, size, msize, training_dataset, eval_datasets)
                # save the DatasetData for this run
                save_path = os.path.join(config.training_dir, f"{dtype}_{size}_{msize}_dsd.pkl")
                with open(save_path, "wb") as f:
                    pickle.dump(dsd, f)

def main() -> None:
    logger.info(f"Starting sequential model size tests with {config.training_samples} training samples...")
    train_models_sequential()   

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="model_size_test"
    )

    main()
