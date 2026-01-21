from scripts.train_on_generated_gamedata import get_generated_gamedata_dataset
from scripts.sl_on_policy_head import train_and_plot_datasets, get_n_random_states, create_gtd_from_states
from a0.model import save_model
from cc.ground_truth import GroundTruth
from cc.core import Game, Board

import random

from config import config

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def remove_trivial_boards(boards: list[Board], gt: GroundTruth) -> list[Board]:
    non_trivial_boards: list[Board] = []
    for board in boards:
        if not gt.is_trivial(board):
            non_trivial_boards.append(board)
    return non_trivial_boards

def sl_on_gamedata_with_experienced_values(fn: str, train_on: int | None = None, test_on: int | None = None, include_trivial: bool = True):
    ev_dataset, _, eboards_nt = get_generated_gamedata_dataset(
        n=None,
        simulated=False,
        include_trivial=include_trivial
    )

    if train_on is not None:
        assert test_on is not None, "If train_on is specified, test_on must also be specified."
        ev_dataset.trim(train_on + test_on, shuffle=True)
    
    eboards_nt = list(eboards_nt)[:5000]
    random.shuffle(eboards_nt)
    
    random_states = get_n_random_states(5000)

    test_size = 10000
    if test_on is not None:
        test_size = test_on
    test_ds = ev_dataset.split_off_test(test_size, shuffle=True)

    seen_gt = create_gtd_from_states(eboards_nt)
    random_gt = create_gtd_from_states(random_states)

    trained_model = train_and_plot_datasets(
        fn=fn,
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
        config.training_dir + f"/{fn}_model",
        trained_model
    )

def sl_on_gamedata_with_gt_values(fn: str, train_on: int | None = None, test_on: int | None = None, include_trivial: bool = True):
    _, eboards, _ = get_generated_gamedata_dataset(
        n=None,
        simulated=False,
        include_trivial=include_trivial
    )

    if train_on is None:
        train_on = len(eboards) - 5000
    if test_on is None:
        test_on = 5000
    
    eboards = list(eboards)[:train_on + test_on]
    random.shuffle(eboards)

    train_eboards = eboards[:train_on]
    test_eboards = eboards[train_on:]
    
    random_states = get_n_random_states(5000)

    train_ds = create_gtd_from_states(train_eboards)
    test_ds = create_gtd_from_states(test_eboards)

    seen_gt = create_gtd_from_states(remove_trivial_boards(train_eboards[:5000], GroundTruth()))
    random_gt = create_gtd_from_states(random_states)

    seen_gt.balance_values()
    test_ds.balance_values()
    random_gt.print_distribution()

    trained_model = train_and_plot_datasets(
        fn=fn,
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
        config.training_dir + f"/{fn}_model",
        trained_model
    )

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="sl_on_gamedata"
    )

    sl_on_gamedata_with_gt_values(fn="sl_on_gamedata_gt_full")
