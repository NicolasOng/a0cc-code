import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import multiprocessing

from a0_new.train.alphazero import alphazero

from a0_new.protocols.model import FullModel

from a0_new.cc.game import CCGame, CCState, CCAction
from a0_new.cc.gt import CCGT
from a0_new.cc.models.nn import CCNNModel
from a0_new.cc.models.full import CCFullModel
from a0_new.cc.players.a0 import CCPlayer
from a0_new.cc.model import A0CCModel

from a0_new.models.rollout import RolloutModel
from a0_new.models.mcts import MCTSModel

from a0_new.eval.alphazero import eval_models
from a0_new.plot.alphazero import plot_seen_random

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def eval():
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
    gt = CCGT()

    eval_models(
        n=1000,
        model=ccnn_model.model,
        gt=gt,
        adapter=ccfull_model,
    )
    plot_seen_random()


if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="cc_a0_eval"
    )

    logger.info(f"config: {config.path}")
    logger.info(f"output_dir: {config.output_dir}")

    # Set the multiprocessing start method to 'spawn' for compatibility with JAX
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    eval()
