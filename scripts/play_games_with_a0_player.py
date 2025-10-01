import jax
from flax import nnx

from config import config

from a0.players.a0 import A0Player
from a0.model import AlphaZeroModel, load_model, save_model
from cc.core import Game

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def main():
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="play_w_a0"
    )

    model = AlphaZeroModel(
        config.board_size,
        training=False,
        rngs=nnx.Rngs({'params': jax.random.PRNGKey(0)})
    )

    player = A0Player(config.board_size, config.num_pieces, model)

    players = [player, player]

    game = Game(config.board_size, config.num_pieces, True, False, False)

    turn = 0
    turn_limit = 80
    while not game.end and (turn_limit is None or turn < turn_limit):
        player = players[turn % len(players)]

        moves = game.start_turn()
        
        move, player_data = player.select_move(game.board, moves)

        logger.info("\n" + game.board.visualize_move_ends([move]))
        logger.info(f"Current player: {game.board.current_player}")
        logger.info(f"Selected move: {move}")

        # TODO: look at and visualize policy created

        game.end_turn(move)

        turn += 1
    
    if game.winner is None:
        logger.info("It's a draw!")
    else:
        logger.info(f"Player {game.winner} wins!")

if __name__ == "__main__":
    main()
