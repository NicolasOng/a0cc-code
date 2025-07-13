from __future__ import annotations
from typing import Optional, Any
import random

from cc.core import Game, Board, Move, Player
from a0.graph_search.mcts import MCTS

class SearchMoves:
    def __init__(self, initial_state: Board, player: UCTPlayer, max_depth: int):
        self._initial_state = initial_state
        self.player = player
        self.max_depth = max_depth

        # get the starting board
        num_pieces, _ = initial_state.num_pieces()
        self.starting_board = Game(len(initial_state.board), num_pieces).board

    def initial_state(self) -> Board:
        '''
        Returns the initial state of the problem.
        '''
        return self._initial_state
    
    def is_terminal(self, state: Board) -> bool:
        '''
        Checks if the given state is a terminal state.
        '''
        return self.player.game.terminal_state(state)

    def get_successors(self, state: Board) -> tuple[list[Board], list[Optional[float]]]:
        '''
        Returns a list of successor states for the given state.
        '''
        # get all possible moves for the current player
        moves = self.player.game.generate_moves_for_given_board(state)

        # create a list of successor states by applying each move
        successors: list[Board] = []
        successor_priors: list[Optional[float]] = []
        for move in moves:
            # create a copy of the board and apply the move
            new_board = Board()
            new_board.copy_board(state)
            new_board.apply_move(move)

            # add the new board to the list of successors
            successors.append(new_board)
            successor_priors.append(None)

        return successors, successor_priors

    def get_reward(self, state: Board) -> float:
        '''
        Returns the reward for the given state,
        considering the current player's perspective.
        Uses a random rollout to determine the reward.
        '''
        current_player = state.current_player
        current_board = state
        for _ in range(self.max_depth):
            # if the state is terminal, return the reward
            if self.is_terminal(current_board):
                px, po = current_board.check_for_winner(self.starting_board)
                winner_is_current_player = (px and current_player == Player.PLAYER_X) or (po and current_player == Player.PLAYER_O)
                winner_is_opponent = (px and current_player == Player.PLAYER_O) or (po and current_player == Player.PLAYER_X)
                if winner_is_current_player:
                    return 1.0
                elif winner_is_opponent:
                    return -1.0
                else:
                    return 0.0
            
            # if the state is not terminal, perform a random rollout
            moves = self.player.game.generate_moves_for_given_board(current_board)
            move = random.choice(moves)
            # apply the move to the board
            new_board = Board()
            new_board.copy_board(current_board)
            new_board.apply_move(move)
            current_board = new_board
        return 0

class UCTPlayer:
    def __init__(self, board_size: int, num_pieces: int):
        self.game = Game(board_size=board_size, num_pieces=num_pieces)
        self.temperature = 1.0  # Temperature for exploration in MCTS
        self.mcts_iterations = 100
    
    def select_move(self, state: Board, moves: list[Move]) -> tuple[Move, Any]:
        # perform mcts and get the root's children
        mcts = MCTS(SearchMoves(state, self, 100))
        mcts.run(iterations=self.mcts_iterations)
        
        child = mcts.get_best_root_child()
        if child:
            move = state.child_board_to_move(child.state)
        else:
            # if no child is found, select a random move
            move = random.choice(moves)

        # return the selected move
        return move, None
