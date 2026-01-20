import multiprocessing
import time

from cc.core import Board, Move, Player, Game
from a0.model import AlphaZeroModel, load_model
from a0.players.a0 import A0Player
from a0.model_utils import board_to_input, get_policy_head_policy, get_value_head_policy
from a0.eval.dataset_evaluation import evaluate_model

from scripts.sl_on_policy_head import get_n_random_states, create_gtd_from_states, train_and_plot_datasets

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def main_parallel() -> None:
    pass

def time_model_inference(n: int = 100000, model_no: int = 25) -> None:
    start_time = time.time()

    trained_model = load_model(config.training_dir + f"model_{model_no}.pkl", training=False)
    random_states = get_n_random_states(n)
    rgtd = create_gtd_from_states(random_states)

    end_time = time.time()
    elapsed_time_setup = end_time - start_time

    start_time = time.time()

    loss, value_loss, policy_loss, value_accuracy, policy_accuracy = evaluate_model(trained_model, rgtd)

    logger.info(f"Evaluation results for model {model_no} on {n} random states:")
    logger.info(f"  Total Loss: {loss:.4f}")
    logger.info(f"  Value Loss: {value_loss:.4f}")
    logger.info(f"  Policy Loss: {policy_loss:.4f}")
    logger.info(f"  Value Accuracy: {value_accuracy:.2%}")
    logger.info(f"  Policy Accuracy: {policy_accuracy:.2%}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    logger.info(f"Total time for setting up {n} states: {elapsed_time_setup:.2f} seconds")
    logger.info(f"Total time for evaluating {n} states: {elapsed_time:.2f} seconds")

def time_model_training(n: int = 100000) -> None:
    start_time = time.time()

    random_states = get_n_random_states(n)
    rgtd = create_gtd_from_states(random_states)

    end_time = time.time()
    elapsed_time_setup = end_time - start_time

    start_time = time.time()

    train_and_plot_datasets(
        fn=f"timing_model",
        dataset=rgtd,
        eval_datasets={},
        num_epochs=10,
        res_blocks=3
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    logger.info(f"Total time for setting up {n} states: {elapsed_time_setup:.2f} seconds")
    logger.info(f"Total time for training {n} states: {elapsed_time:.2f} seconds")

def main() -> None:
    time_model_training(n=100000)
    time_model_inference(n=100000, model_no=50)

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="cpu_v_gpu"
    )

    # try:
    #     multiprocessing.set_start_method('spawn')
    # except RuntimeError:
    #     pass

    main()
