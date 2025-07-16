from a0.train.generate_datasets import generate_ground_truth_dataset, load_ground_truth_dataset
from a0.train.dataset import train_model_on_given_dataset
from a0.eval.dataset_evaluation import load_models, evaluate_all_models, load_losses, plot_losses

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

from config import config

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
gt_dataset.trim(25000, shuffle=True)
gt_dataset_test = gt_dataset.split_off_test(len(gt_dataset) // 10, shuffle=True)

# train the model on the ground truth dataset
logger.info("Training model on ground truth dataset...")
_, dataset_data = train_model_on_given_dataset(gt_dataset, num_epochs=1, save_type='epoch', test_dataset=gt_dataset_test)

# the test happens during training, so don't need the rest here.
exit()

# from the dataset data, we can extract the number of models saved
num_saved = 0
for epoch_data in dataset_data.epoch_data:
    for batch_data in epoch_data.batch_data:
        if batch_data.model_no is not None:
            num_saved = batch_data.model_no

logger.info(f"Training completed. {num_saved} models were saved during training.")

# load all these models
models = load_models(config.training_dir, num_saved)

# evaluate the models on the ground truth dataset
fn = "gt_eval"
evaluate_all_models(models, gt_dataset_test, fn)
total, value, policy, accuracy = load_losses(f"{config.eval_dir}/{fn}.pkl")

# plot the losses
plot_losses(total, value, policy, accuracy, fn)
