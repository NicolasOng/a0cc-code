from a0.dataset import Dataset
from a0.train.dataset import train_model_epoch
from a0.model import AlphaZeroModel, save_model
from a0.eval.dataset_evaluation import evaluate_model
from scripts.sl_on_policy_head import get_n_random_states, create_gtd_from_states

from flax import nnx
import jax

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def train_model(fn: str, dataset: Dataset, eval_datasets: dict[str, Dataset], final_dataset: Dataset, num_epochs: int = 1) -> tuple[AlphaZeroModel, float, float]:
    '''
    Trains a model on the given dataset for num_epochs epochs,
    then evaluates it on the eval_dataset.
    Plots the training performance.
    Returns both the trained model and its value and policy accuracy on the final dataset.
    '''
    logger.info(f"Training model '{fn}' for {num_epochs} epochs...")

    model = AlphaZeroModel(
        board_size=config.board_size,
        num_filters=256,
        training=True,
        rngs=nnx.Rngs({'params': jax.random.PRNGKey(1)})
    )

    trained_model, _, _ = train_model_epoch(model, dataset, test_datasets=eval_datasets)

    loss, value_loss, policy_loss, value_accuracy, policy_accuracy = evaluate_model(trained_model, final_dataset)
    logger.info(f"For model '{fn}':")
    logger.info(f"Evaluation on final dataset - Loss: {loss:.2%}, Value Loss: {value_loss:.2%}, Policy Loss: {policy_loss:.2%}, Value Accuracy: {value_accuracy:.2%}, Policy Accuracy: {policy_accuracy:.2%}")

    return trained_model, value_accuracy, policy_accuracy

def train_model_to_value_acc(fn: str, target_acc: float) -> AlphaZeroModel:
    '''
    Trains a model on the given dataset until it reaches the target value accuracy,
    evaluating it on the eval_dataset after each training session.
    Plots the training performance.
    '''
    train_n = 3907 * 256  # 1,000,192
    validation_n = 100000
    logger.info(f"Training model '{fn}' to reach value accuracy of {target_acc:.2%}...")
    training_boards = get_n_random_states(train_n, remove_trivial=False, remove_terminal=True)
    training_boards_256 = [training_boards[i:i+256] for i in range(0, len(training_boards), 256)]
    validation_dataset = create_gtd_from_states(get_n_random_states(validation_n, remove_trivial=False, remove_terminal=True))
    achieved_value_acc = 0.0
    achieved_policy_acc = 0.0
    trained_model = None

    i = 0
    for i, board_256 in enumerate(training_boards_256):
        training_dataset = create_gtd_from_states(board_256)
        trained_model, achieved_value_acc, achieved_policy_acc = train_model(
            fn,
            training_dataset,
            {},
            validation_dataset,
        )

        if achieved_value_acc > target_acc:
            logger.info(f"Value accuracy reached the target: {achieved_value_acc:.2%} > {target_acc:.2%}.")
            break

    logger.info(f"Final model '{fn}' achieved value accuracy: {achieved_value_acc:.2%}, policy accuracy: {achieved_policy_acc:.2%}.")
    logger.info(f"Training completed after {i+1} training sessions of 256 states each. Total states used: {(i+1)*256}.")

    return trained_model

def main():
    acc = 0.80
    trained_model = train_model_to_value_acc(
        fn="a0_model_to_value_acc",
        target_acc=acc
    )
    save_model(config.training_dir + f"/model_value_acc_{acc:.2f}.pkl", trained_model)

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="train_model_to_acc",
    )

    main()
