import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import multiprocessing

from a0_new.train.alphazero import alphazero

from a0_new.protocols.model import FullModel

from a0_new.cc.game import CCGame, CCState, CCAction
from a0_new.cc.models.nn import CCNNModel
from a0_new.cc.models.full import CCFullModel
from a0_new.cc.players.a0 import CCPlayer

from a0_new.models.rollout import RolloutModel
from a0_new.models.mcts import MCTSModel

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def train_from_zero():
    cc_game = CCGame(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        side_moves=config.sideways_moves,
        backwards_moves=config.backwards_moves,
        num_repeats_to_draw=config.repeats_for_draw,
        illegal_moves=config.illegal_moves
    )
    ccnn_model = CCNNModel()
    ccfull_model = CCFullModel(ccnn_model, cc_game)
    rollout_model = RolloutModel[FullModel[CCState, CCAction], CCState, CCAction](
        cc_game,
        ccfull_model,
        rollout_type=config.rollout_type,
        rollout_depth=config.rollout_depth,
        policy_type=config.policy_type
    )
    mcts_model = MCTSModel[FullModel[CCState, CCAction], CCState, CCAction](
        cc_game,
        rollout_model,
        iterations=config.mcts_samples,
        selection_policy='puct',
        temperature=1.0
    )
    cc_player = CCPlayer(cc_game, mcts_model)

    ccnn_model.get_nn_model().save_to_file(config.training_dir + f'model_{0}.pkl')

    alphazero(cc_game, cc_player, starting_iteration=0)

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="cc_a0_train"
    )

    logger.info(f"config: {config.path}")
    logger.info(f"output_dir: {config.output_dir}")

    # Set the multiprocessing start method to 'spawn' for compatibility with JAX
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    train_from_zero()
