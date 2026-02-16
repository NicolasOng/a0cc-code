from typing import Any

import random

from a0_new.eval.dataset import evaluate_all_models
from a0_new.gt import get_gt
from a0_new.protocols.ground_truth import GTProtocol
from a0_new.protocols.game import T_state, T_action, Player
from a0_new.protocols.model import FullModelOnRaw, TrainableModel
from a0_new.utils.states import get_random_states, get_gtd_from_states, filter_state_list, get_state_info_for_states, log_states_info, get_states_info, remove_bias
from a0_new.dataset import Dataset
from a0_new.utils.training_data import gamedata_generator, model_generator, get_full_experience_data_list_from_training_data
from a0_new.utils.full_experience_data import get_states_from_fed_list

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def get_random_states_for_evaluation(
        n: int,
        gt: GTProtocol[T_state, T_action],
        adapter: FullModelOnRaw[Any, T_state, T_action]
    ) -> tuple[Dataset, Dataset]:
    '''
    n: the number of states to return for evaluation.
    gt: the ground truth protocol to use for getting the state values and policies.
    adapter: a model that can be used to get the state values and policies from the gt
    '''
    random_states = get_random_states(n * 10, gt)
    si_list = get_state_info_for_states(random_states, gt)
    random_states_nd = filter_state_list(
        random_states,
        si_list,
        remove_draws=True
    )
    random_states_nd = remove_bias(random_states_nd, get_state_info_for_states(random_states_nd, gt))
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
    '''
    n: the number of states to return for evaluation.
    gt: the ground truth protocol to use for getting the state values and policies.
    adapter: a model that can be used to get the state values and policies from the gt
    '''
    training_data = gamedata_generator(config.training_dir, config.training_iterations)
    full_experience_data_list = get_full_experience_data_list_from_training_data(training_data)
    seen_states = list(set(get_states_from_fed_list(full_experience_data_list)))

    # shuffle seen states to get a random sample of them, then take the first n states
    random.shuffle(seen_states)
    seen_states = seen_states[:n * 10] # take more than n states to account for filtering out some of them

    # could put this filtering in its own function,
    # since it's the typical filtering I'll be doing for any evaluation dataset
    si_list = get_state_info_for_states(seen_states, gt)
    seen_states_nd = filter_state_list(
        seen_states,
        si_list,
        remove_draws=True
    )
    seen_states_nd = remove_bias(seen_states_nd, get_state_info_for_states(seen_states_nd, gt))
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

def eval_models(n: int, model: TrainableModel, gt: GTProtocol[T_state, T_action], adapter: FullModelOnRaw[Any, T_state, T_action]) -> None:
    models = list(model_generator(model, config.training_dir, config.training_iterations))
    
    random_gtd_nd, random_gtd_nt = get_random_states_for_evaluation(
        n=n,
        gt=gt,
        adapter=adapter
    )

    seen_gtd_nd, seen_gtd_nt = get_seen_states_for_evaluation(
        n=n,
        gt=gt,
        adapter=adapter
    )

    evaluate_all_models(models, random_gtd_nd, "eval_random_nd")
    evaluate_all_models(models, random_gtd_nt, "eval_random_nt")
    evaluate_all_models(models, seen_gtd_nd, "eval_seen_nd")
    evaluate_all_models(models, seen_gtd_nt, "eval_seen_nt")
