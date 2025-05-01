class ChineseCheckersBoard:
    def __init__(self, board_size=7, corner_size=3):
        '''
        Assumes 2 players, starting in the TL and BR corners.
        '''
        self.board = self.init_board(board_size, corner_size)
        self.current_player = 0
        self.board_id_to_symbol = {-1: '-', 0: 'X', 1: 'O'}
    
    def init_board(self, board_size, corner_size):
        '''
        Creates a new board with 2 players,
        pieces in the TL and BR corners (starting positions).
        '''
        board = [[-1 for _ in range(board_size)] for _ in range(board_size)]

        cur_length = corner_size
        for x in range(corner_size):
            for y in range(cur_length):
                board[x][y] = 0
                board[board_size - x - 1][board_size - y - 1] = 1
            cur_length -= 1

        return board
    
    def grid_view(self, board):
        board_str = [['-' for _ in range(len(board))] for _ in range(len(board))]
        for i, row in enumerate(board):
            row_str = board_str[i]
            for j in range(len(row)):
                row_str[j] = self.board_id_to_symbol[row[j]]
        return '\n'.join([' '.join(row) for row in board_str])
    
    def board_view(self, board):
        n = len(board)
        lines = []

        # Top half (including middle)
        for diag in range(n):
            line = ' ' * (n - diag - 1)
            for i in range(diag + 1):
                row = diag - i
                col = i
                line += self.board_id_to_symbol[board[row][col]] + ' '
            lines.append(line.rstrip())

        # Bottom half
        for diag in range(n - 2, -1, -1):
            line = ' ' * (n - diag - 1)
            for i in range(diag + 1):
                row = n - 1 - i
                col = n - diag - 1 + i
                line += self.board_id_to_symbol[board[row][col]] + ' '
            lines.append(line.rstrip())

        return '\n'.join(lines)
    
    def __str__(self):
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
    board = ChineseCheckersBoard()
    print(board)
    print("Current Player:", board.current_player)

    # adjacent test:
    adj_n = len(board.board)
    adj_n2 = adj_n // 2
    adj_board = [[-1 for _ in range(adj_n)] for _ in range(adj_n)]
    adj_board[adj_n2][adj_n2] = 0
    adj_pos = board.get_adjacent(adj_n2, adj_n2)
    for pos in adj_pos:
        adj_board[pos[0]][pos[1]] = 1
    print(board.board_view(adj_board))

    board = ChineseCheckersBoard(board_size=7, corner_size=3)

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


