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

    # other_player = GroundTruthPlayer(mistake_rate=0.1, print_info=True)

    model_filename = "/home/nicolas/Downloads/model_49.pkl"
    model = load_model(model_filename, training=False)
    # other_player = ModelPlayer(board_size=config.board_size, num_pieces=config.num_pieces, model=model)
    other_player = A0Player(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        model=model,
        exploit=True,
        mcts_samples=config.mcts_samples,
        no_reverse_moves=not config.backwards_moves,
        no_illegal_moves=not config.illegal_moves,
        no_side_moves=not config.sideways_moves,
        rollout_type=config.rollout_type,
        rollout_depth=config.rollout_depth,
        policy_type=config.policy_type,
        epsilon=config.epsilon,
        dirichlet_epsilon=config.dirichlet_epsilon)
    #other_player = RandomPlayer()

    #human_player = other_player

    results = play(
        Game(
            config.board_size,
            config.num_pieces,
            no_reverse_moves=not config.backwards_moves,
            no_illegal_moves=not config.illegal_moves,
            no_side_moves=not config.sideways_moves
        ),
        players=[
            human_player,
            other_player
        ],
        turn_limit=80
    )

    # print the results
    print(f"Game ended: {results.ended}, Winner: {results.winner}, Turns: {len(results.turn_data)}")
    if results.final_board:
        print(f"Final board:\n{results.final_board.board_view()}")
    
    return results.winner is not None, results.final_board

def main2():
    not_draws = 0
    for i in range(100):
        print(f"Starting game {i+1}")
        not_draw, final_board = main()
        if not_draw:
            not_draws += 1
            if final_board:
                print(f"Final board state:\n{final_board.board_view()}")
        print("-" * 40)
    
    print(f"Out of 100 games, {not_draws} were not draws.")

if __name__ == "__main__":
    main()
