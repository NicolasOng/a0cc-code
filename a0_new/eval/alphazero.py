from typing import Any

import random

from a0_new.gt import get_gt
from a0_new.protocols.ground_truth import GTProtocol
from a0_new.protocols.game import T_state, T_action, Player
from a0_new.protocols.model import FullModelOnRaw
from a0_new.utils.states import get_random_states, get_gtd_from_states, filter_state_list, get_state_info_for_states, log_states_info, get_states_info
from a0_new.dataset import Dataset
from a0_new.utils.training_data import gamedata_generator, get_full_experience_data_list_from_training_data
from a0_new.utils.full_experience_data import get_states_from_fed_list

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def get_random_states_for_evaluation(
        n: int,
        gt: GTProtocol[T_state, T_action],
        adapter: FullModelOnRaw[Any, T_state, T_action]
    ) -> tuple[Dataset, Dataset]:
    random_states = get_random_states(n * 10, gt)
    si_list = get_state_info_for_states(random_states, gt)
    random_states_nd = filter_state_list(
        random_states,
        si_list,
        remove_draws=True
    )
    random_states_nt = filter_state_list(
        random_states,
        si_list,
        remove_illegal=True,
        remove_trivial=True,
        remove_terminal=True
    )
    # trim the random states to n states, after filtering out the unwanted states
    random_states_nt = random_states_nt[:n]

    logger.info(f"Logging info for random nd states:")
    log_states_info(get_states_info(random_states_nd, get_state_info_for_states(random_states_nd, gt)))
    logger.info(f"Logging info for random nt states:")
    log_states_info(get_states_info(random_states_nt, get_state_info_for_states(random_states_nt, gt)))
    gtd_nd = get_gtd_from_states(random_states_nd, gt, adapter)
    gtd_nt = get_gtd_from_states(random_states_nt, gt, adapter)
    return gtd_nd, gtd_nt

def get_seen_states_for_evaluation(
        n: int,
        gt: GTProtocol[T_state, T_action],
        adapter: FullModelOnRaw[Any, T_state, T_action]
    ) -> tuple[Dataset, Dataset]:
    training_data = gamedata_generator(config.training_dir, config.training_iterations)
    full_experience_data_list = get_full_experience_data_list_from_training_data(training_data)
    seen_states = list(set(get_states_from_fed_list(full_experience_data_list)))

    # shuffle seen states to get a random sample of them, then take the first n states
    random.shuffle(seen_states)
    seen_states = seen_states[:n * 10] # take more than n states to account for filtering out some of them

    si_list = get_state_info_for_states(seen_states, gt)
    seen_states_nd = filter_state_list(
        seen_states,
        si_list,
        remove_draws=True
    )
    seen_states_nt = filter_state_list(
        seen_states,
        si_list,
        remove_illegal=True,
        remove_trivial=True,
        remove_terminal=True
    )

    # trim the seen states to n states, after filtering out the unwanted states
    seen_states_nd = seen_states_nd[:n]
    seen_states_nt = seen_states_nt[:n]

    logger.info(f"Logging info for seen nd states:")
    log_states_info(get_states_info(seen_states_nd, get_state_info_for_states(seen_states_nd, gt)))
    logger.info(f"Logging info for seen nt states:")
    log_states_info(get_states_info(seen_states_nt, get_state_info_for_states(seen_states_nt, gt)))
    gtd_nd = get_gtd_from_states(seen_states_nd, gt, adapter)
    gtd_nt = get_gtd_from_states(seen_states_nt, gt, adapter)
    return gtd_nd, gtd_nt

def plot_evaluation_results():
    pass
