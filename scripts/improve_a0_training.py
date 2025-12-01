from scripts.sl_on_policy_head import create_dataset_from_selfplay, train_and_plot_datasets, get_n_random_states, create_gtd_from_states

import multiprocessing

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def main():
    sp_dataset, unique_sp_boards, _ = create_dataset_from_selfplay()
    rand_states = get_n_random_states(1000)

    logger.log(25, f"Total number of non-trivial boards in self-play dataset: {len(sp_dataset)}")
    logger.log(25, f"Number of unique non-trivial boards in self-play dataset: {len(unique_sp_boards)} ({len(unique_sp_boards)/len(sp_dataset):.2%} unique)")

    test_sp_dataset = sp_dataset.split_off_test(len(sp_dataset) // 10, shuffle=True)
    gt_seen_dataset = create_gtd_from_states(unique_sp_boards)
    gt_rand_dataset = create_gtd_from_states(rand_states)

    model = train_and_plot_datasets(
        fn="improved_a0_training",
        dataset=sp_dataset,
        eval_datasets={
            "sp_test": test_sp_dataset,
            "seen_gt": gt_seen_dataset,
            "random_gt": gt_rand_dataset
        },
        num_epochs=10
    )

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="improve_a0_training",
    )

    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    main()
