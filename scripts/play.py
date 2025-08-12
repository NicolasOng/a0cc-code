from a0.players.human import HumanPlayer
from a0.players.mcts_rollout import MCTSRolloutPlayer
from a0.players.gt import GroundTruthPlayer
from a0.players.model import ModelPlayer
from a0.players.random import RandomPlayer
from a0.players.a0 import A0Player

from a0.model import load_model

from a0.game import play

from cc.core import Game

from config import config

def main():
    human_player = HumanPlayer()
    #human_player = RandomPlayer()

    # other_player = MCTSRolloutPlayer(board_size=config.board_size, num_pieces=config.num_pieces, no_reverse_moves=False)
    
    # other_player = GroundTruthPlayer(print_info=True)

    model_filename = "..."
    model = load_model(model_filename)
    #other_player = ModelPlayer(board_size=config.board_size, num_pieces=config.num_pieces, model=model)
    other_player = A0Player(board_size=config.board_size, num_pieces=config.num_pieces, model=model, exploit=True)

    results = play(
        Game(config.board_size, config.num_pieces, True, False, True),
        players=[
            other_player,
            human_player
        ],
        turn_limit=1000
    )

    # print the results
    print(f"Game ended: {results.ended}, Winner: {results.winner}, Turns: {len(results.turn_data)}")
    if results.final_board:
        print(f"Final board:\n{results.final_board.board_view()}")

if __name__ == "__main__":
    main()
