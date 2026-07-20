

from a0.eval.training_data import game_data_generator
from cc.ground_truth import GroundTruth
from a0.game import GameData
from a0.eval.dataset_evaluation import Series, save_series
from a0.experience_buffer import ExperienceBuffer

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

from a0.train.targets import game_data_to_training_set, game_data_to_gt_training_set, game_data_to_gt_value_training_set, game_data_to_gt_next_value_training_set

def check_game_data_bias(game_data_lists: list[tuple[int, list[GameData]]]) -> None:
    '''
    Checks the bias of game data by seeing how many wins/losses/draws there are.
    '''
    gt = GroundTruth()
    gd_bias_series = Series(["Iteration Win Percent", "Iteration Loss Percent", "Iteration Draw Percent"])
    gd_overall_bias_series = Series(["Overall Win Percent", "Overall Loss Percent", "Overall Draw Percent"])
    total_wins = 0
    total_losses = 0
    total_draws = 0
    total = 0
    # for each game data list/iteration,
    for i, game_data_list in game_data_lists:
        iteration_num_wins = 0
        iteration_num_losses = 0
        iteration_num_draws = 0
        iteration_total = 0

        experience_buffer = ExperienceBuffer(
            config.training_samples
        )
        
        # go through each game,
        for _, game_data in enumerate(game_data_list):

            if config.experiment == "gt":
                training_set = game_data_to_gt_training_set(game_data, gt)
            elif config.experiment == "gt_value":
                training_set = game_data_to_gt_value_training_set(game_data, gt)
            elif config.experiment == "gt_next_value":
                training_set = game_data_to_gt_next_value_training_set(game_data, gt)
            else:
                training_set = game_data_to_training_set(game_data)
            
            for data in training_set:
                experience_buffer.add(data)

        ds = experience_buffer.get_dataset(config.training_batch_size)
        win, draw, loss = ds.get_distribution()

        iteration_num_wins = win
        iteration_num_draws = draw
        iteration_num_losses = loss
        iteration_total = win + draw + loss

        total_wins += iteration_num_wins
        total_losses += iteration_num_losses
        total_draws += iteration_num_draws
        total += iteration_total

        iteration_win_percent = iteration_num_wins / iteration_total if iteration_total > 0 else 0
        iteration_loss_percent = iteration_num_losses / iteration_total if iteration_total > 0 else 0
        iteration_draw_percent = iteration_num_draws / iteration_total if iteration_total > 0 else 0
        # log the iteration's bias
        logger.info(f"Iteration {i} win percent: {iteration_win_percent:.2%} ({iteration_num_wins}/{iteration_total})")
        logger.info(f"Iteration {i} loss percent: {iteration_loss_percent:.2%} ({iteration_num_losses}/{iteration_total})")
        logger.info(f"Iteration {i} draw percent: {iteration_draw_percent:.2%} ({iteration_num_draws}/{iteration_total})")
        # add this info to the series object
        gd_bias_series.x.append(i)
        gd_bias_series.ys["Iteration Win Percent"].append(iteration_win_percent)
        gd_bias_series.ys["Iteration Loss Percent"].append(iteration_loss_percent)
        gd_bias_series.ys["Iteration Draw Percent"].append(iteration_draw_percent)

    # save all the bias data to a file
    save_series(gd_bias_series, f"{config.eval_dir}/gamedata_bias.pkl")

    # Calculate, log, and save the overall bias
    total_win_percent = total_wins / total if total > 0 else 0
    total_loss_percent = total_losses / total if total > 0 else 0
    total_draw_percent = total_draws / total if total > 0 else 0
    logger.info(f"Overall Win Percent: {total_win_percent:.2%} ({total_wins}/{total})")
    logger.info(f"Overall Loss Percent: {total_loss_percent:.2%} ({total_losses}/{total})")
    logger.info(f"Overall Draw Percent: {total_draw_percent:.2%} ({total_draws}/{total})")

    # saving the first and last iteration for easy plotting
    gd_overall_bias_series.x.append(gd_bias_series.x[0])
    gd_overall_bias_series.ys["Overall Win Percent"].append(total_win_percent)
    gd_overall_bias_series.ys["Overall Loss Percent"].append(total_loss_percent)
    gd_overall_bias_series.ys["Overall Draw Percent"].append(total_draw_percent)
    gd_overall_bias_series.x.append(gd_bias_series.x[-1])
    gd_overall_bias_series.ys["Overall Win Percent"].append(total_win_percent)
    gd_overall_bias_series.ys["Overall Loss Percent"].append(total_loss_percent)
    gd_overall_bias_series.ys["Overall Draw Percent"].append(total_draw_percent)
    save_series(gd_overall_bias_series, f"{config.eval_dir}/gamedata_overall_bias.pkl")

def main():
    setup_logging(level=20, log_dir=config.log_dir, process_name='training_data_evals')
    
    logger.info("Starting training data evaluations...")

    check_game_data_bias(list(game_data_generator(config.training_dir, config.training_iterations)))

    logger.info("Training data evaluations completed.")

if __name__ == "__main__":
    main()
