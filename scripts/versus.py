import sys
from a0.players.human import HumanPlayer
from a0.players.mcts_rollout import MCTSRolloutPlayer
from a0.players.gt import GroundTruthPlayer
from a0.players.model import ModelPlayer
from a0.players.random import RandomPlayer
from a0.players.a0 import A0Player

from a0.model import load_model

from a0.game import play

from cc.core import Game, Player

from config import config

def play_game(player1, player2):
    results = play(
        Game(config.board_size, config.num_pieces, True, False, False),
        players=[
            player1,
            player2
        ],
        turn_limit=80
    )

    print(results.final_board.board_view())
    print(len(results.turn_data))
    
    return results.winner, results.ended

def compare_models(model_location1, model_location2, num_games=100, mcts_samples=64):
    model1 = load_model(model_location1)
    model2 = load_model(model_location2)
    player1 = A0Player(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        model=model1,
        exploit=True,
        mcts_samples=mcts_samples)
    player2 = A0Player(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        model=model2,
        exploit=True,
        mcts_samples=mcts_samples)
    
    model1_wins = 0
    model2_wins = 0
    draws_by_repeat = 0
    draws_by_timeout = 0

    for i in range(num_games):
        print(f"Starting game {i+1} between models.")
        winner, ended = play_game(player1, player2)
        if not ended:
            draws_by_timeout += 1
            print("Game ended in a draw due to turn limit.")
        elif winner is None:
            draws_by_repeat += 1
            print("Game ended in a draw due to repeated state.")
        elif winner == Player.PLAYER_X:
            model1_wins += 1
            print("Model 1 wins!")
        elif winner == Player.PLAYER_O:
            model2_wins += 1
            print("Model 2 wins!")

        print(f"Current score - Model 1: {model1_wins}, Model 2: {model2_wins}, Draws by timeout: {draws_by_timeout}, Draws by repeat: {draws_by_repeat}")
        print("-" * 40)

    print(f"Final results after {num_games} games:")
    print(f"Model 1 wins: {model1_wins} ({model1_wins / num_games:.2%})")
    print(f"Model 2 wins: {model2_wins} ({model2_wins / num_games:.2%})")
    print(f"Draws by timeout: {draws_by_timeout} ({draws_by_timeout / num_games:.2%})")
    print(f"Draws by repeated state: {draws_by_repeat} ({draws_by_repeat / num_games:.2%})")

def main():
    # usage: python scripts/versus.py <config.json> <model_1.pkl> <model_2.pkl>
    if len(sys.argv) < 4:
        sys.exit("usage: python scripts/versus.py <config.json> <model_1.pkl> <model_2.pkl>")
    model_location1, model_location2 = sys.argv[2], sys.argv[3]
    compare_models(model_location1, model_location2, num_games=1, mcts_samples=512)

if __name__ == "__main__":
    main()
