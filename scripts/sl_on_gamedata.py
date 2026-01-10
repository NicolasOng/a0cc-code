from scripts.train_on_generated_gamedata import get_generated_gamedata_dataset
from scripts.sl_on_policy_head import train_and_plot_datasets, get_n_random_states, create_gtd_from_states

from config import config

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def main():
    ev_dataset, eboards = get_generated_gamedata_dataset(
        n=None,
        simulated=False,
    )
    
    random_states = get_n_random_states(5000)

    test_ds = ev_dataset.split_off_test(10000, shuffle=True)

    seen_gt = create_gtd_from_states(list(eboards)[:5000])
    random_gt = create_gtd_from_states(random_states)

    train_and_plot_datasets(
        fn="sl_on_gamedata",
        dataset=ev_dataset,
        eval_datasets={
            "test": test_ds,
            "seen_gt": seen_gt,
            "random_gt": random_gt
        },
        num_epochs=10,
        res_blocks=3
    )

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="sl_on_gamedata"
    )

    main()
