import copy

import numpy as np

from a0.players.a0 import board_to_input
from cc.core import Game, Board, Move, Point, Tile, Player

from config import config

def print_board_input(input_array: np.ndarray):
    """
    Prints the board input in a readable format.
    """
    print(input_array.shape)
    print("P1 pieces:")
    print(input_array[0, :, :, 0])
    print("\nP2 pieces:")
    print(input_array[0, :, :, 1])

def test_board_rotation():
    print("Testing board rotation...")
    g = Game(4, 3)
    b = g.board
    b.apply_points([Point(2, 0)], Tile.PLAYER_O)

    rb = Board(4, 0)
    rb.apply_points([Point(3, 3), Point(3, 2), Point(2, 3)], Tile.PLAYER_X)
    rb.apply_points([Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 3)], Tile.PLAYER_O)
    rb.current_player = Player.PLAYER_O
    # print(rb.board_view())
    # print(rb.current_player)

    nb = copy.deepcopy(b)
    # print(nb.board_view())
    # print(nb.current_player)

    b.rotate_board_180()
    b.switch_player()
    # print(b.board_view())
    # print(b.current_player)
    assert b == rb, "Board rotation failed to match expected state"

    b.rotate_board_180()
    b.switch_player()
    assert b == nb, "Board rotation failed to return to original state"

def test_board_to_input():
    print("Testing board to input conversion...")

    print("Trial 1")
    b1 = Board(4, 0)
    b1.init_by_num_pieces(3)
    print(b1.board_view())
    print(b1.current_player)

    b2 = copy.deepcopy(b1)
    b2.current_player = Player.PLAYER_O
    print(b2.board_view())
    print(b2.current_player)

    input1 = board_to_input(b1)
    input2 = board_to_input(b2)

    print_board_input(input1)
    print_board_input(input2)
    print(np.array_equal(input1, input2))

    print("Trial 2")
    b1 = Board(4, 0)
    b1.apply_points([Point(0, 0), Point(1, 0), Point(0, 2)], Tile.PLAYER_X)
    b1.apply_points([Point(3, 2), Point(2, 3), Point(1, 3)], Tile.PLAYER_O)
    b1.current_player = Player.PLAYER_X
    print(b1.board_view())
    print(b1.current_player)

    b2 = Board(4, 0)
    b2.apply_points([Point(1, 0), Point(2, 0), Point(0, 1)], Tile.PLAYER_X)
    b2.apply_points([Point(3, 1), Point(3, 3), Point(2, 3)], Tile.PLAYER_O)
    b2.current_player = Player.PLAYER_O
    print(b2.board_view())
    print(b2.current_player)

    input1 = board_to_input(b1)
    input2 = board_to_input(b2)

    print_board_input(input1)
    print_board_input(input2)
    print(np.array_equal(input1, input2))
    
if __name__ == "__main__":
    test_board_rotation()
    test_board_to_input()
    print("Test passed successfully.")