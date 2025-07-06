import copy

import numpy as np

from a0.players.a0 import board_to_input
from cc.core import Game, Board, Move, Point, Tile, Player

from config import config

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

    print(input1.shape)
    print("First 4x4 array:")
    print(input1[0, :, :, 0])
    print("\nSecond 4x4 array:")
    print(input1[0, :, :, 1])

    print("First 4x4 array:")
    print(input2[0, :, :, 0])
    print("\nSecond 4x4 array:")
    print(input2[0, :, :, 1])
    print(np.array_equal(input1, input2))

    # TODO: more complicated tests where pieces aren't symmetrical


    pass
    
if __name__ == "__main__":
    test_board_rotation()
    test_board_to_input()
    print("Test passed successfully.")