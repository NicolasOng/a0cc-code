import multiprocessing
import time

from cc.core import Board, Move, Player, Game
from a0.model import AlphaZeroModel, load_model
from a0.players.a0 import A0Player
from a0.model_utils import board_to_input, get_policy_head_policy, get_value_head_policy
from a0.eval.dataset_evaluation import evaluate_model

from scripts.sl_on_policy_head import get_n_random_states, create_gtd_from_states, train_and_plot_datasets

import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

@nnx.jit
def inference(model: AlphaZeroModel, states: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    return model(states, train=False)

def print_mean_and_ci(data: list[float], description: str) -> None:
    times_array = np.array(data)
    mean_time = np.mean(times_array)
    std_time = np.std(times_array, ddof=1)  # sample std
    n = len(times_array)
    ci_half_width = 1.96 * std_time / np.sqrt(n)
    ci_lower = mean_time - ci_half_width
    ci_upper = mean_time + ci_half_width
    
    logger.info(f"{description} ({len(data)}):")
    logger.info(f"  Average time: {mean_time:.4f} seconds")
    logger.info(f"  95% CI: {mean_time:.4f} +- {ci_half_width:.4f} [{ci_lower:.4f}, {ci_upper:.4f}] seconds")

def print_memory_stats() -> None:
    try:
        device = jax.devices("gpu")[0]
    except Exception as e:
        logger.info("No GPU device found. Skipping memory stats.")
        return
    stats = device.memory_stats()
    logger.info(f"GPU Memory Stats:")
    logger.info(f"  Bytes in use: {stats['bytes_in_use'] / 1024**2:.2f} MB")
    logger.info(f"  Peak bytes in use: {stats['peak_bytes_in_use'] / 1024**2:.2f} MB")

def time_model_inference(n: int = 100000, batch_size: int = 1) -> None:
    setup_time: float = 0.0
    inference_time: float = 0.0
    first_call_time: float = 0.0
    compiled_call_times: list[float] = []
    data_to_gpu_times: list[float] = []

    start_time = time.time()

    model = AlphaZeroModel(
        board_size=config.board_size,
        num_filters=256,
        rngs=nnx.Rngs({'params': jax.random.PRNGKey(1)})
    )

    num_batches = (n + batch_size - 1) // batch_size
    num_samples = num_batches * batch_size
    random_states = np.random.rand(num_samples, config.board_size, config.board_size, 2).astype(np.float32)

    end_time = time.time()
    setup_time = end_time - start_time

    start_time_inference = time.time()

    for i in range(0, n, batch_size):
        start_time = time.time()
        batch_states = jnp.asarray(random_states[i:i+batch_size], dtype=jnp.float32)
        end_time = time.time()
        elapsed_time = end_time - start_time
        data_to_gpu_times.append(elapsed_time)

        start_time = time.time()
        _ = inference(model, batch_states)
        end_time = time.time()
        elapsed_time = end_time - start_time

        if i == 0:
            first_call_time = elapsed_time
        else:
            compiled_call_times.append(elapsed_time)
            

    end_time_inference = time.time()
    inference_time = end_time_inference - start_time_inference

    logger.info(f"Timing model inference for {n} samples with batch size {batch_size}...")
    logger.info(f"Total time for setting up {n} states: {setup_time:.2f} seconds")
    logger.info(f"Total time for evaluating {n} states: {inference_time:.2f} seconds")
    logger.info(f"First call time (includes compilation): {first_call_time:.4f} seconds")
    if compiled_call_times:
        print_mean_and_ci(compiled_call_times, "Compiled call times")
    else:
        logger.info("No compiled calls were made.")
    if data_to_gpu_times:
        print_mean_and_ci(data_to_gpu_times, "Data to GPU transfer times")
    else:
        logger.info("No data transfer times recorded.")
    print_memory_stats()

def main() -> None:
    jax.clear_caches()
    n = 100000
    print_memory_stats()
    time_model_inference(n=n, batch_size=1)
    time_model_inference(n=n, batch_size=8)
    time_model_inference(n=n, batch_size=64)
    time_model_inference(n=n, batch_size=512)
    time_model_inference(n=n, batch_size=4096)

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="jax_time"
    )

    main()
