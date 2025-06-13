from a0.train.alphazero import train_alphazero
import multiprocessing
from utils.log import setup_logging

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir="logs/",
        process_name="training_alphazero"
    )

    # Set the multiprocessing start method to 'spawn' for compatibility with JAX
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    # Example usage
    train_alphazero(
        model_path=None
    )