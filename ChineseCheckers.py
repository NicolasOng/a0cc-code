from __future__ import annotations
from enum import Enum
from typing import Optional

class Tile(Enum):
    EMPTY = -1
    PLAYER_X = 0
    PLAYER_O = 1
    PLAYER_X_GHOST = 2
    PLAYER_O_GHOST = 3

class Player(Enum):
    PLAYER_X = 0
    PLAYER_O = 1

tile_id_to_symbol = {
    Tile.EMPTY: '-',
    Tile.PLAYER_X: 'X',
    Tile.PLAYER_O: 'O',
    Tile.PLAYER_X_GHOST: 'x',
    Tile.PLAYER_O_GHOST: 'o'
}

class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"
class Move:
    def __init__(self, start_x: int, start_y: int, end_x: int, end_y: int) -> None:
        self.start = Point(start_x, start_y)
        self.end = Point(end_x, end_y)
    
    def __str__(self) -> str:
        return f"Move from {self.start} to {self.end}"
    
    def diagonal_distances(self):
        '''
        Returns the distance of the move in the main diagonal and anti-diagonal directions.
        main diagonal is side-to-side in board view, anti-diagonal is up-down in board view.
        '''
        d_main = (self.end.x - self.end.y) - (self.start.x - self.start.y)
        d_anti = (self.end.x + self.end.y) - (self.start.x + self.start.y)
        return d_main, d_anti

    def is_reverse(self) -> bool:
        '''
        Returns True if the move is a reverse move
        '''
        _, forward = self.diagonal_distances()
        return forward < 0

class Board:
    def __init__(self, board_size: int = 7, home_size: int = 3) -> None:
        '''
        Creates a new sqaure board with the given size.
        '''
        assert 0 <= home_size < board_size, f"home_size must be between 0 and {board_size}"
        self.board = [[Tile.EMPTY for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = Player.PLAYER_X
        self.home_size = home_size
        self.use_four_corners_to_jump = True
        self.no_reverse_moves = True
    
    def set_board(self, board: list[list[Tile]]) -> None:
        self.board = board
    
    def init_corner_triangles(self, triangle_size: int) -> None:
        '''
        Populates the board's TL (player x) and BR (player o) corners with pieces.
        triangle_size determines the number of "rows" of pieces in the corners.
        '''
        board_size = len(self.board)
        cur_length = triangle_size
        for x in range(triangle_size):
            for y in range(cur_length):
                self.board[x][y] = Tile.PLAYER_X
                self.board[board_size - x - 1][board_size - y - 1] = Tile.PLAYER_O
            cur_length -= 1
    
    def init_corner_squares(self, square_size: int) -> None:
        '''
        Populates the board's TL (player x) and BR (player o) corners with pieces.
        square_size determines the side length of the squares in the corners.
        '''
        board_size = len(self.board)
        for x in range(square_size):
            for y in range(square_size):
                self.board[x][y] = Tile.PLAYER_X
                self.board[board_size - x - 1][board_size - y - 1] = Tile.PLAYER_O

    def init_by_num_pieces(self, num_pieces: int) -> None:
        '''
        Populates the board's TL (player x) and BR (player o) corners with pieces.
        Corner formation is determined by num_pieces.
        '''
        triangle_size = 0
        sqaure_size = 0
        remove_corners = False
        # determine the starting formation based on num_pieces
        match num_pieces:
            case 1:
                triangle_size = 1
            case 2:
                triangle_size = 2
                remove_corners = True
            case 3:
                triangle_size = 2
            case 4:
                sqaure_size = 2
            case 5:
                triangle_size = 3
                remove_corners = True
            case 6:
                triangle_size = 3
            case 9:
                triangle_size = 4
                remove_corners = True
            case 10:
                triangle_size = 4
            case _:
                raise ValueError(f"Invalid number of pieces: {num_pieces}. No formation defined.")
        # create the starting formation
        if triangle_size > 0:
            self.init_corner_triangles(triangle_size)
        if sqaure_size > 0:
            self.init_corner_squares(sqaure_size)
        if remove_corners:
            board_size = len(self.board)
            self.board[0][0] = Tile.EMPTY
            self.board[board_size - 1][board_size - 1] = Tile.EMPTY
    
    def grid_view(self) -> str:
        board_str = [[tile_id_to_symbol[Tile.EMPTY] for _ in range(len(self.board))] for _ in range(len(self.board))]
        for i, row in enumerate(self.board):
            row_str = board_str[i]
            for j in range(len(row)):
                row_str[j] = tile_id_to_symbol[row[j]]
        return '\n'.join([' '.join(row) for row in board_str])
    
    def board_view(self) -> str:
        n = len(self.board)
        lines = []

        # Top half (including middle)
        for diag in range(n):
            line = ' ' * (n - diag - 1)
            for i in range(diag + 1):
                row = diag - i
                col = i
                line += tile_id_to_symbol[self.board[row][col]] + ' '
            lines.append(line.rstrip())

        # Bottom half
        for diag in range(n - 2, -1, -1):
            line = ' ' * (n - diag - 1)
            for i in range(diag + 1):
                row = n - 1 - i
                col = n - diag - 1 + i
                line += tile_id_to_symbol[self.board[row][col]] + ' '
            lines.append(line.rstrip())

        return '\n'.join(lines)
    
    def __str__(self) -> str:
        return self.grid_view()
    
    def position_on_main_board(self, x: int, y: int) -> bool:
        '''
        Returns True if the given position is on the board.
        '''
        return 0 <= x < len(self.board) and 0 <= y < len(self.board[0])
    
    def position_in_four_non_main_triangles(self, x: int, y: int) -> bool:
        '''
        Returns True if the given position is in one of the four non-main triangles.
        '''
        if self.position_on_main_board(x, y): return False
        board_size = len(self.board)
        # check if the position is in the two non-main triangles that extend from the y-axis
        j_start = 0
        j_end = board_size + self.home_size
        for i in range(board_size):
            if x == i and j_start <= y < j_end:
                return True
            if i < self.home_size + 1:
                j_end -= 1
            if i >= board_size - self.home_size - 1:
                j_start -= 1
        # check if the position is in the two non-main triangles that extend from the x-axis
        i_start = 0
        i_end = board_size + self.home_size
        for j in range(board_size):
            if y == j and i_start <= x < i_end:
                return True
            if j < self.home_size + 1:
                i_end -= 1
            if j >= board_size - self.home_size - 1:
                i_start -= 1
        return False
    
    def position_is_blocked(self, x: int, y: int) -> bool:
        '''
        Returns True if the given position is blocked by a piece.
        '''
        return self.board[x][y] != Tile.EMPTY
    
    def get_steps(self, x: int, y: int) -> list[Move]:
        '''
        Returns a list of all non-blocked and on-board step moves from (x, y).
        First four are the four cardinal directions.
        Last two are the diagonals perpendicular to the main TL-BR diagonal.
        '''
        step_moves = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (1, -1), (-1, 1)]:
            if self.position_on_main_board(x + dx, y + dy) and not self.position_is_blocked(x + dx, y + dy):
                step_moves.append(Move(x, y, x + dx, y + dy))
        return step_moves
    
    def get_jumps(self, x, y, visited=None):
        '''
        Returns a set of all non-blocked and on-board jump moves from (x, y).
        Uses DFS to find multi-hop jump paths:
        - get all jump moves from (x, y)
        - for each jump move found, recursively repeat this process
        - add each jump start to the visited set to prevent infinite loops
        '''
        if visited is None:
            visited = set()
        visited.add((x, y))
        jumps = set()

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, -1), (-1, 1)]
        for dx, dy in directions:
            step_x, step_y = x + dx, y + dy
            jump_x, jump_y = x + 2*dx, y + 2*dy

            if (jump_x, jump_y) not in visited:
                can_jump_over = self.position_on_main_board(step_x, step_y) and self.position_is_blocked(step_x, step_y)
                if can_jump_over:
                    can_jump_to = self.position_on_main_board(jump_x, jump_y) and not self.position_is_blocked(jump_x, jump_y)
                    can_side_jump_on = self.position_in_four_non_main_triangles(jump_x, jump_y)
                    if can_jump_to:
                        jumps.add(Move(x, y, jump_x, jump_y))
                    if can_jump_to or (self.use_four_corners_to_jump and can_side_jump_on):
                        deeper_jumps = self.get_jump_moves(jump_x, jump_y, visited) # NOT visited.copy()
                        jumps.update(deeper_jumps)                    
        return jumps

    def get_valid_moves(self, x, y):
        '''
        Returns a list of all non-blocked and on-board step and jump moves from (x, y).
        Optionally, also filters out reverse moves.
        '''
        valid_moves = []
        step_moves = self.get_steps(x, y)
        valid_moves.extend(step_moves)
        jump_moves = self.get_jumps(x, y)
        valid_moves.extend(jump_moves)
        if self.no_reverse_moves:
            valid_moves = [move for move in valid_moves if not move.is_reverse()]
        return valid_moves
    
    def check_for_winner(self, starting_board: list[list[Tile]]) -> tuple[bool, bool]:
        '''
        Checks if there is a winner on the board.
        Returns the winning player if there is one, otherwise None.
        Determines winner by checking if the player's goal area is filled, with at least one piece being the player's piece.
        The goal area is defined by the opponent's home area.
        It shouldn't be possible to have both players win at the same time (illegal state).
        '''
        assert (len(starting_board) == len(self.board) and
        len(starting_board[0]) == len(self.board[0])), "Starting board must be the same size as the current board."
        
        player_x_goal_filled = True
        player_o_goal_filled = True
        player_x_goal_has_own_piece = False
        player_o_goal_has_own_piece = False
        
        n = len(self.board)
        for i in range(n):
            for j in range(n):
                # check player o's goal area
                if starting_board[i][j] == Tile.PLAYER_X:
                    if self.board[i][j] == Tile.EMPTY:
                        player_o_goal_filled = False
                    elif self.board[i][j] == Tile.PLAYER_O:
                        player_o_goal_has_own_piece = True
                # check player x's goal area
                elif starting_board[i][j] == Tile.PLAYER_O:
                    if self.board[i][j] == Tile.EMPTY:
                        player_x_goal_filled = False
                    elif self.board[i][j] == Tile.PLAYER_X:
                        player_x_goal_has_own_piece = True
        
        player_x_winner = player_x_goal_filled and player_x_goal_has_own_piece
        player_o_winner = player_o_goal_filled and player_o_goal_has_own_piece
        
        return player_x_winner, player_o_winner

    def check_for_draw(self, board_history: list[Board]) -> bool:
        '''
        Checks if the game is a draw.
        A draw occurs when a state is repeated.
        '''
        for board in board_history:
            if self.board == board.board:
                return True
        return False

def print_all_starting_boards():
    for i in range(6):
        board = Board()
        board.init_by_num_pieces(i + 1)
        print(board)
        print("----")
    for i in range(8,10):
        board = Board()
        board.init_by_num_pieces(i + 1)
        print(board)
        print("----")

def step_visualization(board_size=7):
    board = Board(board_size=board_size)
    mid = board_size // 2
    board.board[mid][mid] = Tile.PLAYER_X
    moves = board.get_steps(mid, mid)
    for move in moves:
        board.board[move.end.x][move.end.y] = Tile.PLAYER_X_GHOST
    print(board)

def jump_visualization(board_size=7):
    board = Board(board_size=board_size)

    # Place a test piece in the center
    n = len(board.board)
    mid = n // 2
    board.board[mid][mid] = Tile.PLAYER_X
    board.board[mid][mid - 1] = Tile.PLAYER_X
    board.board[mid - 1][mid - 2] = Tile.PLAYER_X

    # Get all valid moves from the center
    valid_moves = board.get_valid_moves(mid, mid)

    # Visualize the board: 0 = current player, 1 = valid move
    for move in valid_moves:
        board.board[move.end.x][move.end.y] = Tile.PLAYER_X_GHOST

    # Print the visualization
    print(board)

def full_board_viz(main_size, home_size):
    offset = home_size
    large_size = main_size + home_size * 2
    # create  the board for visualization (very big) and the hexagon board to visualize
    hexagon_board = Board(board_size=main_size, home_size=home_size)
    large_board = Board(board_size=large_size)
    # go over all the positions in the large board and set them if they are in the hexagon board
    for x in range(-home_size, large_size):
        for y in range(-home_size, large_size):
            if hexagon_board.position_in_four_non_main_triangles(x, y):
                large_board.board[x + offset][y + offset] = Tile.PLAYER_O
            elif hexagon_board.position_on_main_board(x, y):
                large_board.board[x + offset][y + offset] = Tile.PLAYER_X
    print(large_board.grid_view())
    print(large_board.board_view())

def visualize_move(sx, sy, ex, ey):
    move = Move(sx, sy, ex, ey)
    print(move)
    print(move.diagonal_distances())
    board = Board()
    board.board[sx][sy] = Tile.PLAYER_X
    board.board[ex][ey] = Tile.PLAYER_X_GHOST
    print(board.grid_view())
    print(board.board_view())

if __name__ == "__main__":
    board = Board()
    board.board[1][2] = Tile.PLAYER_X
    print(board)
    print("Current Player:", board.current_player)

    #full_board_viz(5, 2)
    
    visualize_move(0, 0, 0, 1)
    
    visualize_move(3, 3, 4, 2)
    visualize_move(3, 3, 4, 4)
