'''
A wrapper around the existing cc.core.Game to conform to the new A0Game protocol.
'''

from __future__ import annotations

from typing import Optional, Any

from a0_new.protocols import A0Game, A0State, A0Action, Player
from cc.core import Game, Board, Move, Player as CCPlayer

class CC_Action(A0Action):
    def __init__(self, start_x: int, start_y: int, end_x: int, end_y: int):
        self.move = Move(start_x, start_y, end_x, end_y)

class CC_State(A0State[CC_Action]):
    def __init__(self, board_size: int, home_size: int):
        self.board: Board = Board(board_size, home_size)
        self._action = CC_Action(0,0,0,0)
    
    def init_from_board(self, board: Board) -> None:
        self.board = board

    def clone(self) -> CC_State:
        new_state = CC_State(len(self.board.board), self.board.home_size)
        new_state.board.copy_board(self.board)
        return new_state
    
    def apply_action(self, action: CC_Action) -> None:
        self.board.apply_move(action.move)
    
    def undo_action(self, action: CC_Action) -> None:
        self.board.undo_move(action.move)

class CC_Game(A0Game[CC_State, CC_Action]):
    def __init__(self, board_size: int, num_pieces: int, side_moves: bool, backwards_moves: bool, num_repeats_to_draw: int):
        self.board_size = board_size
        self.num_pieces = num_pieces
        self.side_moves = side_moves
        self.backwards_moves = backwards_moves
        self.num_repeats_to_draw = num_repeats_to_draw

        self.game = Game(
            board_size=board_size,
            num_pieces=num_pieces,
            repeats_for_draw=num_repeats_to_draw,
            no_side_moves=not side_moves,
            no_reverse_moves=not backwards_moves,
            no_illegal_moves=True
        )
    
    def reset(self) -> None:
        self.game.clear_game()
        self.game.initialize_game(self.num_pieces)
    
    def step(self, action: CC_Action) -> tuple[CC_State, float, bool, bool, Any]:
        assert self.game.end == False, "Cannot take a step in a finished game."
        # take the action
        self.game.end_turn(action.move)
        # get the info
        next_state, reward, terminated, truncated, info = self.get_info()
        return next_state, reward, terminated, truncated, info
    
    def get_info(self) -> tuple[CC_State, float, bool, bool, Any]:
        '''
        Gets the same data as the step method, but without changing the state.
        '''
        # get the next state
        # clone so it's independent
        next_state = CC_State(self.board_size, self.game.board.home_size)
        next_state.init_from_board(self.game.board)
        next_state = next_state.clone()
        # get reward
        reward = 1 if self.game.winner is not None else 0
        # check if terminated
        terminated = self.game.end
        # truncated is always False for this game
        truncated = False
        # info is empty
        info = None
        return next_state, reward, terminated, truncated, info
    
    def get_actions(self, state: CC_State | None) -> list[CC_Action]:
        moves = self.game.generate_moves_for_given_board(state.board if state is not None else self.game.board)
        actions = [CC_Action(move.start.x, move.start.y, move.end.x, move.end.y) for move in moves]
        return actions
    
    def is_legal(self, state: CC_State) -> bool:
        return self.game.legal(state.board)
    
    def is_terminal(self, state: CC_State) -> bool:
        return self.game.get_done(state.board)
    
    def get_winner(self, state: CC_State) -> Optional[Player]:
        winner = self.game.get_winner(state.board)
        if winner is None:
            return None
        return Player.X if winner == CCPlayer.PLAYER_X else Player.O
    
    def get_current_player(self) -> Player:
        current = self.game.board.current_player
        return Player.X if current == CCPlayer.PLAYER_X else Player.O
