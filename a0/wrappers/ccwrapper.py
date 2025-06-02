from __future__ import annotations
from typing import Optional
from cc.core import Board, board_to_home_size, Player, Move
from cc.ranking import CCState as CCStateR, generate_rect_board_lists

BOARD_SIZE = 4
NUM_PIECES = 3

starting_board = Board(BOARD_SIZE, board_to_home_size[BOARD_SIZE])
starting_board.init_by_num_pieces(NUM_PIECES)

movement_rules = {
    'can_jump_out_of_home': False,
    'use_four_corners_to_jump': True
}

localRectToBoard, localBoardToRect = generate_rect_board_lists(BOARD_SIZE, BOARD_SIZE)

def board_pos2ccstate_pos(x: int, y: int) -> int:
    """
    Convert board position (x, y) to CCState position.
    """
    return localRectToBoard[x * BOARD_SIZE + y]

class CCheckers:
    @staticmethod
    def Reset(state: CCState):
        num_pieces, _ = state.board.num_pieces()
        state.board.current_player = Player.PLAYER_X
        state.board.init_by_num_pieces(num_pieces)

    @staticmethod
    def Done(state: CCState) -> bool:
        player_x_winner, player_o_winner = state.board.check_for_winner(starting_board)
        return (player_x_winner and state.board.current_player == Player.PLAYER_O) or (player_o_winner and state.board.current_player == Player.PLAYER_X)

    @staticmethod
    def getMovesHelper(state: CCState) -> list[Move]:
        x_positions, o_positions = state.board.get_player_positions()
        player_positions = x_positions if state.board.current_player == Player.PLAYER_X else o_positions
        moves: list[Move] = []
        for pos in player_positions:
            x, y = pos.x, pos.y
            valid_moves = state.board.get_moves(x, y, movement_rules)
            moves.extend(valid_moves)
        return moves
    
    @staticmethod
    def getMovesForward(state: CCState) -> Optional[CCMove]:
        moves = CCheckers.getMovesHelper(state)
        if state.board.current_player == Player.PLAYER_X:
            moves = [move for move in moves if not move.is_up()]
        else:
            moves = [move for move in moves if not move.is_down()]
        # convert moves to CCMove format (linked list)
        cc_moves: list[CCMove] = []
        for move in moves:
            from_pos = board_pos2ccstate_pos(move.start.x, move.start.y)
            to_pos = board_pos2ccstate_pos(move.end.x, move.end.y)
            cc_moves.append(CCMove(from_pos, to_pos, from_pos, None))
        # Link the moves as a linked list
        for i in range(len(cc_moves) - 1):
            cc_moves[i].next = cc_moves[i + 1]
        return cc_moves[0] if cc_moves else None

class CCState:
    def __init__(self):
        self.board = Board(BOARD_SIZE, board_to_home_size[BOARD_SIZE])
        self.board.init_by_num_pieces(NUM_PIECES)
    
    def getBoard(self) -> list[int]:
        new_board, _ = CCStateR.convert_Board_to_CCState(self.board)
        return new_board
    
    def getToMove(self) -> int:
        cp = self.board.current_player
        return 0 if cp == Player.PLAYER_X else 1


class CCMove:
    def __init__(self, from_pos: int, to_pos: int, which: int, next: Optional[CCMove] = None):
        self.from_pos = from_pos
        self.to_pos = to_pos
        # which is just the same as from_pos?
        self.which = which
        self.next = next
    
    def getNextMove(self) -> Optional[CCMove]:
        return self.next
    
    def setNextMove(self, next_move: Optional[CCMove]):
        self.next = next_move
    
    def getTo(self) -> int:
        return self.to_pos
    
    def getFrom(self) -> int:
        return self.from_pos
class CCLocalRank12:
    pass
