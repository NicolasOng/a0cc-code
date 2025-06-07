from a0.train.alphazero import train_alphazero

import logging

# set up the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn' for compatibility with JAX
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    # Example usage
    train_alphazero(
        iterations=50,
        board_size=4,
        num_pieces=3,
        model_path=None,
        train_set_len=32 * 10,
        data_path="a0_data"
    )
