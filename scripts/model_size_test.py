import multiprocessing
import os
import time
import concurrent.futures
from concurrent.futures import Future, wait, FIRST_COMPLETED

from a0.dataset import Dataset

from scripts.sl_on_policy_head import get_n_random_states, create_gtd_from_states, create_random_gtd_from_states, check_random_policy_acc, train_and_plot_datasets

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def training_run(dataset_type: str, dataset_size: int, model_size: int, training_dataset: Dataset, eval_datasets: dict[str, Dataset]) -> None:
    logger.info(f"Starting training run: dataset_type={dataset_type}, dataset_size={dataset_size}, model_size={model_size}")
    start_time = time.time()
    train_and_plot_datasets(
        fn=f"{dataset_type}_{dataset_size}_{model_size}",
        dataset=training_dataset,
        eval_datasets=eval_datasets,
        num_epochs=10,
        res_blocks=model_size
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Finished training run: dataset_type={dataset_type}, dataset_size={dataset_size}, model_size={model_size} in {elapsed_time:.2f} seconds")


def _training_run(dataset_type: str, dataset_size: int, model_size: int, training_dataset: Dataset, eval_datasets: dict[str, Dataset]) -> None:
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="model_size_test"
    )
    logger.info(f"Starting training run: dataset_type={dataset_type}, dataset_size={dataset_size}, model_size={model_size}")
    start_time = time.time()
    train_and_plot_datasets(
        fn=f"{dataset_type}_{dataset_size}_{model_size}",
        dataset=training_dataset,
        eval_datasets=eval_datasets,
        num_epochs=10,
        res_blocks=model_size
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Finished training run: dataset_type={dataset_type}, dataset_size={dataset_size}, model_size={model_size} in {elapsed_time:.2f} seconds")

def train_models_parallel() -> None:
    # create all the datasets:
    # training datasets of sizes 1K, 10K, 100K, 1M, of both types (gt and rand_gt) (8 datasets)
    # test datasets for each type, of size 10K (2 datasets)
    # evaluation datasets: seen and random, of size 10K each (8 seen + 1 rand = 9 datasets)
    logger.info("Creating datasets...")
    dataset_sizes = [1000, 10000, 100000, 1000000]
    dataset_types = ['gt', 'rand_gt']

    logger.info("Creating evaluation datasets...")
    eval_states = get_n_random_states(10000)
    eval_dataset = create_gtd_from_states(eval_states)

    logger.info(f"Creating test datasets... ({dataset_types})")
    test_states = get_n_random_states(10000)
    test_datasets: dict[str, Dataset] = {}
    for dtype in dataset_types:
        if dtype == 'gt':
            rgtd = create_gtd_from_states(test_states)
        elif dtype == 'rand_gt':
            rgtd = create_random_gtd_from_states(test_states)
        else:
            raise ValueError(f"Unknown dataset type: {dtype}")
        test_datasets[dtype] = rgtd
    
    logger.info("Creating training datasets...")
    logger.info(f"Dataset sizes: {dataset_sizes}, types: {dataset_types}")
    training_datasets: dict[str, tuple[Dataset, dict[str, Dataset]]] = {}
    max_dataset_size = max(dataset_sizes)
    training_states = get_n_random_states(max_dataset_size)
    random_acc = check_random_policy_acc(training_states)
    logger.info(f"Random policy accuracy on {max_dataset_size} states: {random_acc:.2%}")

    for size in dataset_sizes:
        random_states = training_states[:size]
        seen_states = random_states[:10000]
        for dtype in dataset_types:
            dataset_name = f"{dtype}_{size}"
            if dtype == 'gt':
                rgtd = create_gtd_from_states(random_states)
                seen_rgtd = create_gtd_from_states(seen_states)
            elif dtype == 'rand_gt':
                rgtd = create_random_gtd_from_states(random_states)
                seen_rgtd = create_random_gtd_from_states(seen_states)
            else:
                raise ValueError(f"Unknown dataset type: {dtype}")
            if dtype == 'gt':
                training_datasets[dataset_name] = (
                    rgtd,
                    {
                        'test': test_datasets[dtype],
                        'seen_eval': seen_rgtd,
                    }
                )
            elif dtype == 'rand_gt':
                training_datasets[dataset_name] = (
                    rgtd,
                    {
                        'test': test_datasets[dtype],
                        'seen_eval': seen_rgtd,
                        'eval': eval_dataset,
                    }
                )

    # set the model sizes to test
    model_sizes = [1, 2, 3, 4, 5]
    logger.info(f"Model sizes to test: {model_sizes}")

    # run all training runs in parallel using multiprocessing
    num_cores = os.cpu_count() or 4
    logger.info(f"Using {num_cores} cores for training runs.")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # create a list to hold the futures
        futures: list[Future[None]] = []

        # create a function to start a process that trains a model
        def spawn_training_run(dataset_type: str, dataset_size: int, model_size: int, training_dataset: Dataset, eval_datasets: dict[str, Dataset]) -> None:
            future = executor.submit(
                _training_run,
                dataset_type,
                dataset_size,
                model_size,
                training_dataset,
                eval_datasets
            )
            futures.append(future)
        
        # start a process for each training run
        for size in dataset_sizes:
            for dtype in dataset_types:
                for msize in model_sizes:
                    training_dataset, eval_datasets = training_datasets[f"{dtype}_{size}"]
                    spawn_training_run(dtype, size, msize, training_dataset, eval_datasets)
        
        while True:
            # when a game (or games) finish(es),
            done, _ = wait(futures, return_when=FIRST_COMPLETED)

            # for each finished sample,
            for future in done:
                # remove it from the list of futures
                futures.remove(future)
            
            # stop when all samples are generated
            if not futures:
                break

def train_models_sequential(n: int) -> None:
    # create all the datasets:
    # training datasets of size n of both types (gt and rand_gt) (2 datasets)
    # test datasets for each type, of size 10K (2 datasets)
    # evaluation datasets: seen and random, of size 10K each (2 seen + 1 rand = 3 datasets)
    logger.info("Creating datasets...")
    # note: just one
    dataset_sizes = [n]
    dataset_types = ['gt', 'rand_gt']

    logger.info("Creating evaluation datasets...")
    eval_states = get_n_random_states(10000)
    eval_dataset = create_gtd_from_states(eval_states)

    logger.info(f"Creating test datasets... ({dataset_types})")
    test_states = get_n_random_states(10000)
    test_datasets: dict[str, Dataset] = {}
    for dtype in dataset_types:
        if dtype == 'gt':
            rgtd = create_gtd_from_states(test_states)
        elif dtype == 'rand_gt':
            rgtd = create_random_gtd_from_states(test_states)
        else:
            raise ValueError(f"Unknown dataset type: {dtype}")
        test_datasets[dtype] = rgtd
    
    logger.info("Creating training datasets...")
    logger.info(f"Dataset sizes: {dataset_sizes}, types: {dataset_types}")
    training_datasets: dict[str, tuple[Dataset, dict[str, Dataset]]] = {}
    max_dataset_size = max(dataset_sizes)
    training_states = get_n_random_states(max_dataset_size)
    random_acc = check_random_policy_acc(training_states)
    logger.info(f"Random policy accuracy on {max_dataset_size} states: {random_acc:.2%}")

    for size in dataset_sizes:
        random_states = training_states[:size]
        seen_states = random_states[:10000]
        for dtype in dataset_types:
            dataset_name = f"{dtype}_{size}"
            if dtype == 'gt':
                rgtd = create_gtd_from_states(random_states)
                seen_rgtd = create_gtd_from_states(seen_states)
            elif dtype == 'rand_gt':
                rgtd = create_random_gtd_from_states(random_states)
                seen_rgtd = create_random_gtd_from_states(seen_states)
            else:
                raise ValueError(f"Unknown dataset type: {dtype}")
            if dtype == 'gt':
                training_datasets[dataset_name] = (
                    rgtd,
                    {
                        'test': test_datasets[dtype],
                        'seen_eval': seen_rgtd,
                    }
                )
            elif dtype == 'rand_gt':
                training_datasets[dataset_name] = (
                    rgtd,
                    {
                        'test': test_datasets[dtype],
                        'seen_eval': seen_rgtd,
                        'eval': eval_dataset,
                    }
                )

    # set the model sizes to test
    model_sizes = [1, 2, 3, 4, 5]
    logger.info(f"Model sizes to test: {model_sizes}")

    # run all training runs sequentially
    # start a process for each training run
    for size in dataset_sizes:
        for dtype in dataset_types:
            for msize in model_sizes:
                training_dataset, eval_datasets = training_datasets[f"{dtype}_{size}"]
                training_run(dtype, size, msize, training_dataset, eval_datasets)

def main() -> None:
    # train_models_parallel()
    logger.info(f"Starting sequential model size tests with {config.training_samples} training samples...")
    train_models_sequential(config.training_samples)   

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="model_size_test"
    )

    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    main()
