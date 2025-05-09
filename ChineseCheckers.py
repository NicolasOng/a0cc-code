from enum import Enum
from typing import List, Optional

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

class Board:
    def __init__(self, board_size: int = 7) -> None:
        '''
        Creates a new sqaure board with the given size.
        '''
        self.board = [[Tile.EMPTY for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = Player.PLAYER_X
    
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
    
    def grid_view(self, board: list[list[Tile]]) -> str:
        board_str = [[tile_id_to_symbol[Tile.EMPTY] for _ in range(len(board))] for _ in range(len(board))]
        for i, row in enumerate(board):
            row_str = board_str[i]
            for j in range(len(row)):
                row_str[j] = tile_id_to_symbol[row[j]]
        return '\n'.join([' '.join(row) for row in board_str])
    
    def board_view(self, board: list[list[Tile]]) -> str:
        n = len(board)
        lines = []

        # Top half (including middle)
        for diag in range(n):
            line = ' ' * (n - diag - 1)
            for i in range(diag + 1):
                row = diag - i
                col = i
                line += tile_id_to_symbol[board[row][col]] + ' '
            lines.append(line.rstrip())

        # Bottom half
        for diag in range(n - 2, -1, -1):
            line = ' ' * (n - diag - 1)
            for i in range(diag + 1):
                row = n - 1 - i
                col = n - diag - 1 + i
                line += tile_id_to_symbol[board[row][col]] + ' '
            lines.append(line.rstrip())

        return '\n'.join(lines)
    
    def __str__(self) -> str:
        return self.grid_view(self.board)
    
    def get_adjacent(self, x, y):
        '''
        Returns a list of adjacent coordinates.
        First four are the four cardinal directions.
        Last two are the diagonals perpendicular to the main TL-BR diagonal.
        '''
        adj = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (1, -1), (-1, 1)]:
            if 0 <= x + dx < len(self.board) and 0 <= y + dy < len(self.board[0]):
                adj.append((x + dx, y + dy))
        return adj
    
    def get_jump_moves(self, x, y, visited=None):
        '''
        Returns a set of all possible jump moves from (x, y).
        Uses DFS to find multi-hop jump paths.
        '''
        if visited is None:
            visited = set()
        visited.add((x, y))
        jumps = set()

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, -1), (-1, 1)]
        for dx, dy in directions:
            mid_x, mid_y = x + dx, y + dy
            jump_x, jump_y = x + 2*dx, y + 2*dy

            if (0 <= mid_x < len(self.board) and 0 <= mid_y < len(self.board[0]) and
                0 <= jump_x < len(self.board) and 0 <= jump_y < len(self.board[0])):
                if self.board[mid_x][mid_y] in [0, 1] and self.board[jump_x][jump_y] == -1:
                    if (jump_x, jump_y) not in visited:
                        jumps.add((jump_x, jump_y))
                        # Recurse to allow chained jumps
                        deeper_jumps = self.get_jump_moves(jump_x, jump_y, visited.copy())
                        jumps.update(deeper_jumps)
        return jumps

    def get_valid_moves(self, x, y):
        '''
        Returns a list of all valid single-step and jump moves from (x, y).
        '''
        if self.board[x][y] != self.current_player:
            return []

        valid_moves = []
        # Single-step adjacent moves
        for nx, ny in self.get_adjacent(x, y):
            if self.board[nx][ny] == -1:
                valid_moves.append((nx, ny))

        # Jump moves
        jump_moves = self.get_jump_moves(x, y)
        valid_moves.extend(jump_moves)

        return valid_moves

if __name__ == "__main__":
    import copy
    board = Board()
    print(board)
    print("Current Player:", board.current_player)

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

    exit()
    # adjacent test:
    adj_n = len(board.board)
    adj_n2 = adj_n // 2
    adj_board = [[Tile.EMPTY for _ in range(adj_n)] for _ in range(adj_n)]
    adj_board[adj_n2][adj_n2] = Tile.PLAYER_X
    adj_pos = board.get_adjacent(adj_n2, adj_n2)
    for pos in adj_pos:
        adj_board[pos[0]][pos[1]] = Tile.PLAYER_X_GHOST
    print(board.board_view(adj_board))

    board = Board(board_size=7)

    # Place a test piece in the center
    n = len(board.board)
    mid = n // 2
    board.board = [[-1 for _ in range(n)] for _ in range(n)]
    board.board[mid][mid] = board.current_player  # Place current player's piece
    board.board[mid][mid - 1] = board.current_player
    board.board[mid - 1][mid - 2] = board.current_player

    # Get all valid moves from the center
    valid_moves = board.get_valid_moves(mid, mid)

    # Visualize the board: 0 = current player, 1 = valid move
    test_board = copy.deepcopy(board.board)
    for x, y in valid_moves:
        test_board[x][y] = 1

    # Print the visualization
    print(board.board_view(test_board))


