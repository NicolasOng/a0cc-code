from a0.eval.training_data import game_data_generator

from cc.ground_truth import GroundTruth

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def inspect_last_states():
    gt = GroundTruth()

    total = 0
    correct = 0

    incorrect_due_to_ties = 0
    incorrect_due_to_not_ties = 0

    for iteration_no, games in game_data_generator(config.training_dir, config.training_iterations):
        for g_no, game in enumerate(games):
            winner = game.winner
            last_turn = game.turn_data[-1]
            logger.info(f"Inspecting game {g_no} in iteration {iteration_no}")
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


    logger.info(f"Total games: {total}, Correct predictions: {correct}, Accuracy: {correct / total if total > 0 else 0:.2%}")
    logger.info(f"Incorrect due to ties: {incorrect_due_to_ties}, Accuracy ignoring ties: {correct / (total - incorrect_due_to_ties) if total - incorrect_due_to_ties > 0 else 0:.2%}")

def inspect_games():
    gt = GroundTruth()

    for iteration_no, games in game_data_generator(config.training_dir, config.training_iterations):
        if iteration_no != 50: continue
        for g_no, game in enumerate(games):
            if g_no != 2: continue
            winner = game.winner
            for t_no, turn in enumerate(game.turn_data):
                logger.info(f"Inspecting game {g_no}, turn {t_no} in iteration {iteration_no}")
                logger.info("\n" + turn.board.visualize_move_ends([turn.move]))
                logger.info(f"Current player: {turn.board.current_player}")
                logger.info(f"Selected move: {turn.move}")
                #logger.info(f"Player data: {turn.player_data}")
            exit()
def main():
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="inspect_games"
    )

    inspect_games()

    

if __name__ == "__main__":
    main()