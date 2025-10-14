from a0.players.human import HumanPlayer
from a0.players.random import RandomPlayer

from a0.game import play, GameData, TurnData
import matplotlib.pyplot as plt

from cc.core import Game, Player

from config import config

def main(p):
    player1 = RandomPlayer()
    player2 = RandomPlayer()

    results = play(
        Game(config.board_size, config.num_pieces, True, False, False, False),
        players=[
            player1,
            player2
        ],
        turn_limit=10000
    )

    # print the results
    if p:
        print(f"Game ended: {results.ended}, Winner: {results.winner}, Turns: {len(results.turn_data)}")
        if results.final_board:
            print(f"Final board:\n{results.final_board.board_view()}")
    
    return results.winner, results.ended, results.final_board, results

def create_histogram(data, title="Histogram", xlabel="Value", ylabel="Frequency", bins=None):
    """
    Create a histogram from a list of integers.
    
    Args:
        data: List of integers
        title: Title for the histogram
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        bins: Number of bins or bin edges (None for auto)
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.show()

def print_final_results(results: GameData):
    final_board = results.final_board
    if final_board:
        print(f"Final board state:\n{final_board.board_view()}")
    print(f"Final Player: {results.turn_data[-1].board.current_player}")
    print(f"Final move: {results.turn_data[-1].move} ({results.turn_data[-1].move.diagonal_distances()})")
    print("moves this turn:")
    for move in results.turn_data[-1].moves:
        print(move, move.diagonal_distances())
    print("-" * 40)

def main2():
    n = 1000
    draws_repeat = 0
    draws_timeout = 0
    px_win = 0
    po_win = 0
    px_illegal_win = 0
    po_illegal_win = 0
    px_proper_win = 0
    po_proper_win = 0
    num_turns: list[int] = []

    for i in range(n):
        print(f"Starting game {i+1}")
        winner, ended, final_board, results = main(False)
        draw_by_repeat = False
        draw_by_timeout = False
        if winner is Player.PLAYER_X:
            px_win += 1
        elif winner is Player.PLAYER_O:
            po_win += 1
        elif ended:
            draws_repeat += 1
            draw_by_repeat = True
        else:
            draws_timeout += 1
            draw_by_timeout = True
        
        proper_win = False
        winner_exists = False
        if winner is not None:
            winner_exists = True
            if results.turn_data[-1].board.current_player == winner:
                proper_win = True
                if winner is Player.PLAYER_X:
                    px_proper_win += 1
                else:
                    po_proper_win += 1
            else:
                if winner is Player.PLAYER_X:
                    px_illegal_win += 1
                else:
                    po_illegal_win += 1
        
        print(len(results.turn_data))
        if winner_exists:
            num_turns.append(len(results.turn_data))
        
        if winner is not None and False:
            print_final_results(results)
        
        if draw_by_repeat and False:
            print_final_results(results)
        
        if draw_by_timeout and False:
            print_final_results(results)
        
        if proper_win and len(results.turn_data) < 250 and len(results.turn_data) > 150 and True:
            print_final_results(results)
    
    print(f"total games: {n}")
    print(f"Draws by repetition: {draws_repeat / n:.2%} ({draws_repeat}), Draws by timeout: {draws_timeout / n:.2%} ({draws_timeout})")
    print(f"Player X wins: {px_win / n:.2%} ({px_win}), Player O wins: {po_win / n:.2%} ({po_win})")
    print(f"Player X proper wins: {px_proper_win / n:.2%} ({px_proper_win}), Player O proper wins: {po_proper_win / n:.2%} ({po_proper_win})")
    print(f"Player X illegal wins: {px_illegal_win / n:.2%} ({px_illegal_win}), Player O illegal wins: {po_illegal_win / n:.2%} ({po_illegal_win})")

    create_histogram(num_turns, title="Histogram of Number of Turns per (Winning) Game", xlabel="Number of Turns", ylabel="Frequency", bins=20)

if __name__ == "__main__":
    main2()
