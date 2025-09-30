from a0.eval.training_data import game_data_generator

from cc.ground_truth import GroundTruth

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def main():
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="inspect_games"
    )

    gt = GroundTruth()

    total = 0
    correct = 0

    incorrect_due_to_ties = 0
    incorrect_due_to_not_ties = 0

    for iteration_no, games in game_data_generator(config.training_dir, config.training_iterations):
        for game in games:
            winner = game.winner
            last_turn = game.turn_data[-1]
            logger.info("\n" + last_turn.board.board_view())
            logger.info(f"Selected move: {last_turn.move}")
            #logger.info(f"Player data: {last_turn.player_data}")
            gto = gt.get_outcome(last_turn.board)
            oo = 0 if winner is None else 1 if winner == last_turn.board.current_player else -1
            logger.info(f"Ground truth outcome: {gto}, actual outcome: {oo}")
            logger.info(f"Game outcome: {winner}")

            total += 1
            if gto == oo:
                correct += 1
            elif oo == 0:
                incorrect_due_to_ties += 1
            else:
                logger.warning("Discrepancy found between ground truth and actual outcome, not due to tie.")
                incorrect_due_to_not_ties += 1
                if incorrect_due_to_not_ties > 7:
                    exit()


    logger.info(f"Total games: {total}, Correct predictions: {correct}, Accuracy: {correct / total if total > 0 else 0:.2%}")
    logger.info(f"Incorrect due to ties: {incorrect_due_to_ties}, Accuracy ignoring ties: {correct / (total - incorrect_due_to_ties) if total - incorrect_due_to_ties > 0 else 0:.2%}")

if __name__ == "__main__":
    main()