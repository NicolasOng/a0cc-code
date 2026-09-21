import os
import sys
# moved here from the repo root; keep repo-root imports working
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random

from cc.core import Game, Board, Move, Point, Player, Tile

def get_user_input(max_num):
    while True:
        user_input = input("Enter a number: ")
        try:
            number = int(user_input)
            assert 0 <= number < max_num
            break
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
    return number

def create_board(x_positions, o_positions, current_player):
    board = Board()
    board.apply_points(x_positions, Tile.PLAYER_X)
    board.apply_points(o_positions, Tile.PLAYER_O)
    board.current_player = current_player
    return board

def main():
    game = Game()
    '''
    board = create_board(
        [Point(6, 0), Point(6, 1), Point(6, 2), Point(6, 3), Point(4, 6), Point(5, 5)],
        [Point(0, 0), Point(0, 1), Point(6, 4), Point(6, 5), Point(6, 6), Point(5, 6)],
        Player.PLAYER_O
    )
    game.set_board(board)
    '''
    while not game.end:
        # Player's turn
        moves = game.start_turn()
        if game.end: break

        print(game.board.visualize_move_ends(moves))
        print(f"Available moves: {len(moves)}")
        for i, move in enumerate(moves):
            print(f'{i}: {move}')
        move_idx = get_user_input(len(moves))
        print(f"Player chooses move: {move_idx}")
        game.end_turn(moves[move_idx])

        if game.end: break

        # AI's turn
        moves = game.start_turn()
        if game.end: break

        print(game.board.visualize_move_ends(moves))
        print(f"Available moves for AI: {len(moves)}")
        #random_move = random.choice(moves)
        random_move = moves[0]
        print(f"AI chooses move: {random_move}")
        game.end_turn(random_move)
    
    if game.winner is None:
        print("It's a draw!")
    else:
        print(f"Player {game.winner} wins!")

if __name__ == "__main__":
    main()
