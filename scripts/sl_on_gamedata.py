from scripts.train_on_generated_gamedata import get_generated_gamedata_dataset
from scripts.sl_on_policy_head import train_and_plot_datasets, get_n_random_states, create_gtd_from_states
from a0.model import save_model

import random

from config import config

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def main():
    ev_dataset, eboards = get_generated_gamedata_dataset(
        n=None,
        simulated=False,
    )
    eboards = list(eboards)[:5000]
    random.shuffle(eboards)
    
    random_states = get_n_random_states(5000)

    test_ds = ev_dataset.split_off_test(10000, shuffle=True)

    seen_gt = create_gtd_from_states(eboards)
    random_gt = create_gtd_from_states(random_states)

    trained_model = train_and_plot_datasets(
        fn="sl_on_gamedata",
        dataset=ev_dataset,
        eval_datasets={
            "test": test_ds,
            "seen_gt": seen_gt,
            "random_gt": random_gt
        },
        num_epochs=1,
        res_blocks=3
    )

    save_model(
        config.training_dir + "/sl_on_gamedata_model",
        trained_model
    )

def main2():
    _, eboards = get_generated_gamedata_dataset(
        n=None,
        simulated=False,
    )
    eboards = list(eboards)[:100000]
    random.shuffle(eboards)

    train_eboards = eboards[:80000]
    test_eboards = eboards[80000:]
    
    random_states = get_n_random_states(5000)

    train_ds = create_gtd_from_states(train_eboards)
    test_ds = create_gtd_from_states(test_eboards)

    seen_gt = create_gtd_from_states(train_eboards[:5000])
    random_gt = create_gtd_from_states(random_states)

    trained_model = train_and_plot_datasets(
        fn="sl_on_gamedata_gt",
        dataset=train_ds,
        eval_datasets={
            "test_gt": test_ds,
            "seen_gt": seen_gt,
            "random_gt": random_gt
        },
        num_epochs=1,
        res_blocks=3
    )

    save_model(
        config.training_dir + "/sl_on_gamedata_gt_model",
        trained_model
    )

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="sl_on_gamedata"
    )

    main2()
