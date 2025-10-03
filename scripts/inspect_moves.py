from cc.core import Move, Board, Tile

def main():
    m = Move(4, 0, 0, 4)
    b = Board(5, 3)

    b.apply_points([m.start, m.end], Tile.PLAYER_X)

    print(b.board_view())
    print(m.diagonal_distances())
    print("up:", m.is_up())
    print("down:", m.is_down())

main()