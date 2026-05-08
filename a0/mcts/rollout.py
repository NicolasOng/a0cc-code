from __future__ import annotations
from typing import Optional

from cc.core import Game, Board
from a0.mcts.rollout_strategies import StateEvaluator, RolloutPolicy, ZeroEval, RandomPolicy

class SearchMoves:
    def __init__(self, initial_state: Board, cc: Game, max_depth: int,
                 evaluator: Optional[StateEvaluator] = None,
                 policy: Optional[RolloutPolicy] = None):
        self._initial_state = initial_state
        self.cc = cc
        self.max_depth = max_depth
        self.evaluator = evaluator if evaluator is not None else ZeroEval()
        self.policy = policy if policy is not None else RandomPolicy()

        # get the starting board
        num_pieces, _ = initial_state.num_pieces()
        self.starting_board = Game(board_size=len(initial_state.board), num_pieces=num_pieces).board

    def initial_state(self) -> Board:
        '''
        Returns the initial state of the problem.
        '''
        return self._initial_state

    def is_terminal(self, state: Board) -> bool:
        '''
        Checks if the given state is a terminal state.
        '''
        return self.cc.get_done(state)

    def get_successors(self, state: Board, is_root: bool) -> tuple[list[Board], None]:
        '''
        Returns a list of successor states for the given state.
        '''
        # get all possible moves for the current player
        moves = self.cc.generate_moves_for_given_board(state)

        # create a list of successor states by applying each move
        successors: list[Board] = []
        for move in moves:
            # create a copy of the board and apply the move
            new_board = Board()
            new_board.copy_board(state)
            new_board.apply_move(move)

            # add the new board to the list of successors
            successors.append(new_board)

        return successors, None

    def get_reward(self, state: Board) -> float:
        '''
        Returns the reward for the given state, from the perspective of the
        initial-state player. Plays out via self.policy until terminal or
        max_depth. Terminals defer to evaluator.terminal_value (so each eval
        owns its own value scale); depth-caps defer to evaluator.evaluate.
        '''
        root_player = self._initial_state.current_player
        current_board = state
        for _ in range(self.max_depth):
            is_done, winner = self.cc.get_done_and_winner(current_board)
            if is_done:
                return self.evaluator.terminal_value(current_board, winner, root_player)

            # otherwise advance one step under the configured rollout policy
            moves = self.cc.generate_moves_for_given_board(current_board)
            if not moves:
                break
            move = self.policy.choose(current_board, moves, self.cc)
            new_board = Board()
            new_board.copy_board(current_board)
            new_board.apply_move(move)
            current_board = new_board
        # depth cap reached (or no moves and not flagged terminal) — defer to the eval
        return self.evaluator.evaluate(current_board)

    def is_maximizing(self, state: Board) -> bool:
        return state.current_player == self._initial_state.current_player
