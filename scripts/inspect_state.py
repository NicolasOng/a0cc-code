from cc.core import Game, Board, Tile, Point, Player

g = Game(5, 6, False, False, False)

b = Board(5, 1)

def print_board(board: Board):
    print(board.current_player)
    print(board.board_view())

b.apply_points([Point(0, 0), Point(0, 2), Point(1, 0), Point(1, 2), Point(2, 0), Point(2, 1)], Tile.PLAYER_X)
b.apply_points([Point(1, 4), Point(2, 4), Point(3, 3), Point(4, 0), Point(4, 1), Point(4, 3)], Tile.PLAYER_O)
b.current_player = Player.PLAYER_O

print_board(b)

print(g.get_done_and_winner(b))