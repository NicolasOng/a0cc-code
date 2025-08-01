from a0.train.generate_datasets import generate_ground_truth_dataset, load_ground_truth_dataset
from a0.train.dataset import train_model_on_given_dataset

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

from config import config

print(f"board size: {config.board_size}, num pieces: {config.num_pieces}")

setup_logging(
    level=20,
    log_dir=config.log_dir,
    process_name="train_on_ground_truth"
)

# try to load the ground truth dataset, if it exists
# if not, generate it
try:
    gt_dataset = load_ground_truth_dataset()
    logger.info("Ground truth dataset loaded successfully.")
except FileNotFoundError:
    logger.warning(f"Dataset file was not found. Generating ground truth dataset...")
    # If the dataset does not exist, generate it
    generate_ground_truth_dataset()
    gt_dataset = load_ground_truth_dataset()

# process the dataset (trim it, create train-test split, etc.)
gt_dataset.trim(100000, shuffle=False)
gt_dataset_test = gt_dataset.split_off_test(len(gt_dataset) // 10, shuffle=True)

# train the model on the ground truth dataset
logger.info("Training model on ground truth dataset...")
_, dataset_data = train_model_on_given_dataset(gt_dataset, num_epochs=10, save_type='epoch', test_dataset=gt_dataset_test)
