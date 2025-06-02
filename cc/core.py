from __future__ import annotations
from enum import Enum
from typing import Optional
import copy

import logging
#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.CRITICAL)

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
    Tile.PLAYER_X_GHOST: '*',
    Tile.PLAYER_O_GHOST: '°'
}

player_to_tile = {
    Player.PLAYER_X: Tile.PLAYER_X,
    Player.PLAYER_O: Tile.PLAYER_O
}

board_to_home_size = {
    2: 0,
    3: 0,
    4: 0,
    5: 1,
    6: 2,
    7: 3,
    8: 3,
    9: 4,
    10: 4,
    11: 5,
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

    def is_up(self) -> bool:
        '''
        Returns True if the move is an up move (moves negative on the anti-diagonal).
        '''
        _, forward = self.diagonal_distances()
        return forward < 0
    
    def is_down(self) -> bool:
        '''
        Returns True if the move is a down move (moves positive on the anti-diagonal).
        '''
        _, forward = self.diagonal_distances()
        return forward > 0

class Board:
    def __init__(self, board_size: int = 7, home_size: int = 3) -> None:
        '''
        Creates a new sqaure board with the given size.
        '''
        assert 0 <= home_size < board_size, f"home_size must be between 0 and {board_size}"
        self.board = [[Tile.EMPTY for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = Player.PLAYER_X
        self.home_size = home_size
    
    def clear_board(self) -> None:
        '''
        Clears the board, setting all tiles to Tile.EMPTY.
        '''
        self.board = [[Tile.EMPTY for _ in range(len(self.board))] for _ in range(len(self.board))]
    
    def set_board(self, board: list[list[Tile]]) -> None:
        self.board = copy.deepcopy(board)

    def copy_board(self, board: Board) -> None:
        self.board = copy.deepcopy(board.board)
        self.current_player = board.current_player
        self.home_size = board.home_size
    
    def apply_points(self, points: list[Point], tile: Tile) -> None:
        '''
        Applies the given tile to the given points on the board.
        '''
        for point in points:
            x, y = point.x, point.y
            if self.position_on_main_board(x, y):
                self.board[x][y] = tile
    
    def apply_move(self, move: Move) -> None:
        '''
        Applies the given move to the board.
        '''
        player_tile = player_to_tile[self.current_player]
        logging.debug(f"Applying move: {move} for player {self.current_player}")
        assert self.position_on_main_board(move.start.x, move.start.y) and self.board[move.start.x][move.start.y] == player_tile, f"Invalid move: {move.start} is not occupied by the current player."
        assert self.position_on_main_board(move.end.x, move.end.y) and self.board[move.end.x][move.end.y] == Tile.EMPTY, f"Invalid move: {move.end} is occupied."
        # move the piece
        self.board[move.start.x][move.start.y] = Tile.EMPTY
        self.board[move.end.x][move.end.y] = player_tile
        # update the current player
        self.current_player = Player.PLAYER_O if self.current_player == Player.PLAYER_X else Player.PLAYER_X
    
    def undo_move(self, move: Move) -> None:
        '''
        "Undoes" the given move on the board.
        Assumes that the move given is the last move made by the previous player.
        '''
        player_tile = player_to_tile[Player.PLAYER_O if self.current_player == Player.PLAYER_X else Player.PLAYER_X]
        logging.debug(f"Undoing move: {move} for player {self.current_player}")
        assert self.position_on_main_board(move.end.x, move.end.y) and self.board[move.end.x][move.end.y] == player_tile, f"Invalid undo: {move.end} is not occupied by the current player."
        assert self.position_on_main_board(move.start.x, move.start.y) and self.board[move.start.x][move.start.y] == Tile.EMPTY, f"Invalid undo: {move.start} is occupied."
        # undo the piece move
        self.board[move.end.x][move.end.y] = Tile.EMPTY
        self.board[move.start.x][move.start.y] = player_tile
        # reverse the current player
        self.current_player = Player.PLAYER_O if self.current_player == Player.PLAYER_X else Player.PLAYER_X
        

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
        remove_nn = False
        remove_0m_m0 = False
        n = 0
        m = 0
        # determine the starting formation based on num_pieces
        match num_pieces:
            case 1:
                triangle_size = 1
            case 2:
                triangle_size = 2
                remove_nn = True
            case 3:
                triangle_size = 2
            case 4:
                sqaure_size = 2
            case 5:
                triangle_size = 3
                remove_nn = True
                n = 1
            case 6:
                triangle_size = 3
            case 7:
                triangle_size = 4
                remove_nn = True
                remove_0m_m0 = True
                n = 1
                m = 3
            case 8:
                triangle_size = 4
                remove_0m_m0 = True
                m = 3
            case 9:
                triangle_size = 4
                remove_nn = True
                n = 1
            case 10:
                triangle_size = 4
            case _:
                raise ValueError(f"Invalid number of pieces: {num_pieces}. No formation defined.")
        # create the starting formation
        if triangle_size > 0:
            self.init_corner_triangles(triangle_size)
        if sqaure_size > 0:
            self.init_corner_squares(sqaure_size)
        if remove_nn:
            board_size = len(self.board)
            self.board[n][n] = Tile.EMPTY
            self.board[board_size - 1 - n][board_size - 1 - n] = Tile.EMPTY
        if remove_0m_m0:
            board_size = len(self.board)
            self.board[0][m] = Tile.EMPTY
            self.board[m][0] = Tile.EMPTY
            self.board[board_size - 1][board_size - 1 - m] = Tile.EMPTY
            self.board[board_size - 1 - m][board_size - 1] = Tile.EMPTY
    
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
    
    def get_jumps(self, sx: int, sy: int, x: int, y: int, movement_rules: dict[str, bool], visited: set[tuple[int, int]] = None) -> set[Move]:
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

        use_four_corners_to_jump = movement_rules['use_four_corners_to_jump']
        can_jump_out_of_home = movement_rules['can_jump_out_of_home']

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, -1), (-1, 1)]
        for dx, dy in directions:
            step_x, step_y = x + dx, y + dy
            jump_x, jump_y = x + 2*dx, y + 2*dy

            if (jump_x, jump_y) not in visited:
                can_jump_over = self.position_on_main_board(step_x, step_y) and self.position_is_blocked(step_x, step_y)
                if can_jump_over:
                    jump_to_pos_on_main_board = self.position_on_main_board(jump_x, jump_y)
                    can_jump_to = jump_to_pos_on_main_board and not self.position_is_blocked(jump_x, jump_y)
                    can_side_jump_on = self.position_in_four_non_main_triangles(jump_x, jump_y)
                    if can_jump_to:
                        jumps.add(Move(sx, sy, jump_x, jump_y))
                    if can_jump_to or (use_four_corners_to_jump and can_side_jump_on) or (can_jump_out_of_home and not jump_to_pos_on_main_board):
                        deeper_jumps = self.get_jumps(sx, sy, jump_x, jump_y, movement_rules, visited) # NOT visited.copy()
                        jumps.update(deeper_jumps)                    
        return jumps

    def get_moves(self, x: int, y: int, movement_rules: dict[str, bool]) -> list[Move]:
        '''
        Returns a list of all non-blocked and on-board step and jump moves from (x, y).
        '''
        valid_moves = []
        step_moves = self.get_steps(x, y)
        valid_moves.extend(step_moves)
        jump_moves = self.get_jumps(x, y, x, y, movement_rules)
        valid_moves.extend(jump_moves)
        return valid_moves
    
    def check_for_winner(self, starting_board: Board) -> tuple[bool, bool]:
        '''
        Checks if there is a winner on the board.
        Returns the winning player if there is one, otherwise None.
        Determines winner by checking if the player's goal area is filled, with at least one piece being the player's piece.
        The goal area is defined by the opponent's home area.
        It shouldn't be possible to have both players win at the same time (illegal state).
        '''
        player_x_goal_filled = True
        player_o_goal_filled = True
        player_x_goal_has_own_piece = False
        player_o_goal_has_own_piece = False

        o_goal_area, x_goal_area = starting_board.get_player_positions()

        # check player o's goal area
        for o_goal_pos in o_goal_area:
            x, y = o_goal_pos.x, o_goal_pos.y
            if self.board[x][y] == Tile.EMPTY:
                player_o_goal_filled = False
            elif self.board[x][y] == Tile.PLAYER_O:
                player_o_goal_has_own_piece = True
        
        # check player x's goal area
        for x_goal_pos in x_goal_area:
            x, y = x_goal_pos.x, x_goal_pos.y
            if self.board[x][y] == Tile.EMPTY:
                player_x_goal_filled = False
            elif self.board[x][y] == Tile.PLAYER_X:
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
            if self.board == board.board and self.current_player == board.current_player:
                return True
        return False
    
    def get_player_positions(self) -> tuple[list[Point], list[Point]]:
        '''
        Returns a list of all positions occupied by each player.
        '''
        x_posiitons = []
        o_posiitons = []
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] == Tile.PLAYER_X:
                    x_posiitons.append(Point(i, j))
                elif self.board[i][j] == Tile.PLAYER_O:
                    o_posiitons.append(Point(i, j))
        return x_posiitons, o_posiitons
    
    def num_pieces(self) -> tuple[int, int]:
        '''
        Returns the number of pieces on the board for each player.
        '''
        x_posiitons, o_posiitons = self.get_player_positions()
        num_x_pieces = len(x_posiitons)
        num_o_pieces = len(o_posiitons)
        return num_x_pieces, num_o_pieces
    
    def num_spots(self) -> int:
        '''
        Returns the number of spots on the board.
        '''
        return len(self.board) * len(self.board[0])
    
    def board_sizes(self) -> tuple[int, int]:
        '''
        Returns the size of the board.
        '''
        return len(self.board), len(self.board[0])
    
    def positions_are_filled_by_player(self, positions: list[Point], player: Player) -> bool:
        '''
        Returns True if all positions are filled by the given player.
        '''
        player_tile = player_to_tile[player]
        for pos in positions:
            x, y = pos.x, pos.y
            if self.position_on_main_board(x, y) and self.board[x][y] != player_tile:
                return False
        return True
    
    def is_illegal_state(self, starting_board: Board) -> bool:
        '''
        Checks if the game is in an illegal state:
        - both players have won at the same time
        - winning conditions for player n are met, and it is player n turn to move
        - one or more unoccupied positions in a goal area are unreachable due to the opponent's pieces
        '''
        # check if both players have won at the same time
        player_x_winner, player_o_winner = self.check_for_winner(starting_board)
        if player_x_winner and player_o_winner:
            return True
        
        # check if the current player has won
        if (player_x_winner and self.current_player == Player.PLAYER_X) or (player_o_winner and self.current_player == Player.PLAYER_O):
            return True
        
        # check if there are unreachable positions in the goal area
        # (only implemented for the set starting boards with 1-6 pieces)
        num_x_pieces, num_o_pieces = self.num_pieces()
        if num_x_pieces == 6 and num_o_pieces == 6:
            # check if there are unreachable positions in the goal area
            board_size = len(self.board)
            x_illegal = [Point(0, 1), Point(1, 0), Point(2, 0), Point(0, 2)]
            o_illegal = [Point(board_size - p.x, board_size - p.y) for p in x_illegal]
            x_blocking = not self.position_is_blocked(0, 0) and self.positions_are_filled_by_player(x_illegal, Player.PLAYER_X)
            o_blocking = not self.position_is_blocked(board_size - 1, board_size - 1) and self.positions_are_filled_by_player(o_illegal, Player.PLAYER_O)
            if x_blocking or o_blocking:
                return True

        return False

    def simple_hash(self):
        hashable_board = tuple(tuple(tile.value for tile in row) for row in self.board)
        return (hashable_board, self.current_player.value, self.home_size)

class Game:
    def __init__(self, board_size: int = 7, num_pieces: int = 6) -> None:
        self.board = Board(board_size=board_size, home_size=board_to_home_size[board_size])
        self.board_history = []
        self.end = False
        self.winner = None

        # game rules
        # if true, player can "jump" off the board during chained jumps
        self.can_jump_out_of_home = False
        # if true, player can use the four non-main corners during chained jumps
        self.use_four_corners_to_jump = True
        self.no_reverse_moves = False
        self.no_illegal_moves = False
        self.no_draw_moves = False
        self.pass_moves = True
        self.draw_on_no_moves = False
        # set to True in normal play, False in eg tree search
        # as DFS saves the board history
        self.draw_on_repeated_state = False

        self.save_board_history = self.draw_on_repeated_state or self.no_draw_moves

        self.movement_rules = {
            'can_jump_out_of_home': self.can_jump_out_of_home,
            'use_four_corners_to_jump': self.use_four_corners_to_jump
        }

        self.initialize_game(num_pieces)
        
    def generate_moves_for_current_player(self) -> list[Move]:
        '''
        Generates all possible moves for the current player.
        '''
        x_positions, o_positions = self.board.get_player_positions()
        player_positions = x_positions if self.board.current_player == Player.PLAYER_X else o_positions

        for pos in player_positions:
            logging.debug(f"Player {self.board.current_player} has piece at {pos}")

        # generate all on-board non-blocked moves for each piece
        moves = []
        for pos in player_positions:
            x, y = pos.x, pos.y
            valid_moves = self.board.get_moves(x, y, self.movement_rules)
            moves.extend(valid_moves)
        
        # remove reverse moves if no_reverse_moves is set
        if self.no_reverse_moves:
            if self.board.current_player == Player.PLAYER_X:
                moves = [move for move in moves if not move.is_up()]
            else:
                moves = [move for move in moves if not move.is_down()]
        
        # remove moves that lead to an illegal state
        valid_moves = []
        if self.no_illegal_moves:
            for move in moves:
                # create a copy of the board and apply the move
                logging.debug(f"Checking if move {move} leads to an illegal state. Current player: {self.board.current_player}")
                new_board = Board()
                new_board.copy_board(self.board)
                new_board.apply_move(move)
                # check if the move doesn't lead to an illegal state
                if not new_board.is_illegal_state(self.board_history[0]):
                    valid_moves.append(move)
        else:
            valid_moves = moves
        moves = valid_moves

        # remove moves that lead to a draw
        valid_moves = []
        if self.no_draw_moves:
            for move in moves:
                # create a copy of the board and apply the move
                new_board = Board()
                new_board.copy_board(self.board)
                new_board.apply_move(move)
                # check if the move doesn't lead to a draw
                if not new_board.check_for_draw(self.board_history):
                    valid_moves.append(move)
        else:
            valid_moves = moves
        moves = valid_moves

        # create a pass move if needed
        if self.pass_moves and len(moves) == 0:
            pos = player_positions[0]
            x, y = pos.x, pos.y
            moves.append(Move(x, y, x, y))
        
        return moves
    
    def game_end_check(self) -> None:
        '''
        Checks if the game has ended - either a player has won or the game is a draw.
        '''
        # 1. check if the board is in an illegal state
        # in this case, the game ends and the player who caused the illegal state loses
        if not self.no_illegal_moves and self.board.is_illegal_state(self.board_history[0]):
            self.end = True
            self.winner = Player.PLAYER_O if self.board.current_player == Player.PLAYER_X else Player.PLAYER_X
            return
        
        # 2. check if a player has won
        # note that I don't check that the last move was made by the player who won,
        # because that check is done in the is_illegal_state method
        player_x_winner, player_o_winner = self.board.check_for_winner(self.board_history[0])
        if player_x_winner:
            self.end = True
            self.winner = Player.PLAYER_X
            return
        if player_o_winner:
            self.end = True
            self.winner = Player.PLAYER_O
            return
        
        # 3. check if the game is a draw
        if self.draw_on_repeated_state and self.board.check_for_draw(self.board_history):
            logging.debug(f"Game ended in a draw.")
            self.end = True
            self.winner = None
            return
    
    def initialize_game(self, num_pieces: int) -> None:
        '''
        Initializes the game with the given number of pieces.
        The starting formation is determined by num_pieces.
        '''
        self.board.init_by_num_pieces(num_pieces)
        self.board_history.append(copy.deepcopy(self.board))
        self.end = False
        self.winner = None

    def set_board(self, board: Board) -> None:
        '''
        Sets the game to the given board and player.
        Board history is empty.
        '''
        self.board.copy_board(board)
        self.end = False
        self.winner = None
        self.board_history = []
        self.board_history.append(copy.deepcopy(self.board))
    
    def set_game(self, game: Game) -> None:
        '''
        Sets the game to the given game.
        '''
        self.__dict__ = copy.deepcopy(game.__dict__)
    
    def start_turn(self) -> list[Move]:
        '''
        Starts a turn for the current player.
        Generates all possible moves for the current player.
        '''
        if self.end: return []
        
        # generate all possible moves for the current player
        moves = self.generate_moves_for_current_player()
        logging.debug(f"Player {self.board.current_player} has {len(moves)} moves.")

        # if there are no moves, the game ends
        # and the winner is the other player (or its a draw)
        if len(moves) == 0:
            self.end = True
            if self.draw_on_no_moves:
                self.winner = None
            else:
                self.winner = Player.PLAYER_O if self.board.current_player == Player.PLAYER_X else Player.PLAYER_X
            return []
        
        return moves
    
    def end_turn(self, move: Move) -> None:
        '''
        Ends the turn for the current player.
        Applies the given move to the board and updates the current player.
        '''
        if self.end: return

        # apply the move to the board
        self.board.apply_move(move)

        # check if the game has ended
        self.game_end_check()

        # add the board to the history
        if self.save_board_history:
            self.board_history.append(copy.deepcopy(self.board))

    def visualize_move_ends(self, moves: list[Move]) -> None:
        '''
        Visualizes the end positions of the given moves on the board.
        Assumes the moves are valid and for the current player.
        '''
        temp_board = Board()
        temp_board.copy_board(self.board)
        for move in moves:
            temp_board.board[move.end.x][move.end.y] = Tile.PLAYER_X_GHOST if self.board.current_player == Player.PLAYER_X else Tile.PLAYER_O_GHOST
        print(temp_board.board_view())

    def simple_hash(self):
        board_hash = self.board.simple_hash()
        board_history_hash = tuple(board.simple_hash() for board in self.board_history)
        game_rules_hash = (self.can_jump_out_of_home, self.use_four_corners_to_jump,
                           self.no_reverse_moves, self.no_illegal_moves,
                           self.no_draw_moves, self.pass_moves,
                           self.draw_on_no_moves, self.draw_on_repeated_state)
        return (board_hash, board_history_hash, game_rules_hash, self.end, self.winner)
    
    def __hash__(self):
        return self.simple_hash().__hash__()
    
    def __eq__(self, other):
        if not isinstance(other, Game):
            return NotImplemented
        return self.__hash__() == other.__hash__()


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

def visualize_board(bs, x_pos, o_pos):
    board = Board(board_size=bs)
    board.apply_points(x_pos, Tile.PLAYER_X)
    board.apply_points(o_pos, Tile.PLAYER_O)
    print(board.grid_view())
    print(board.board_view())

if __name__ == "__main__":
    visualize_board(7, [Point(0, 6), Point(1, 6), Point(2, 6), Point(3, 6), Point(5, 4), Point(5, 5)], [Point(0, 0), Point(1, 0), Point(4, 6), Point(5, 6), Point(6, 6), Point(6, 5)])
    visualize_board(7, [Point(6, 0), Point(6, 1), Point(6, 2), Point(6, 3), Point(4, 6), Point(5, 5)], [Point(0, 0), Point(0, 1), Point(6, 4), Point(6, 5), Point(6, 6), Point(5, 6)])
