from scripts.sl_on_policy_head import create_dataset_from_selfplay

import multiprocessing

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

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

    sp_dataset, _, _ = create_dataset_from_selfplay()
