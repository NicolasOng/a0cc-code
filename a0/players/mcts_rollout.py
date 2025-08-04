from __future__ import annotations
from typing import Optional, Any
import random

from cc.core import Game, Board, Move
from a0.graph_search.mcts import MCTS

class SearchMoves:
    def __init__(self, initial_state: Board, player: MCTSRolloutPlayer, max_depth: int):
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
        return self.player.game.get_done(state)

    def get_successors(self, state: Board) -> tuple[list[Board], None]:
        '''
        Returns a list of successor states for the given state.
        '''
        # get all possible moves for the current player
        moves = self.player.game.generate_moves_for_given_board(state)

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
        Returns the reward for the given state,
        considering the perspective of the player at the initial state of the search.
        Uses a random rollout to determine the reward.
        '''
        current_board = state
        for _ in range(self.max_depth):
            # check if the current board is terminal, and get its winner
            is_done, winner = self.player.game.get_done_and_winner(current_board)
            if is_done:
                # if the game is done, return the value based on the winner
                if winner is None:
                    return 0.0
                return 1.0 if winner == self._initial_state.current_player else -1.0
            
            # if the state is not terminal, perform a random rollout
            moves = self.player.game.generate_moves_for_given_board(current_board)
            move = random.choice(moves)
            # apply the move to the board
            new_board = Board()
            new_board.copy_board(current_board)
            new_board.apply_move(move)
            current_board = new_board
        # if the maximum depth is reached, just return 0.
        return 0

class MCTSRolloutPlayer:
    def __init__(self, board_size: int, num_pieces: int, no_reverse_moves: bool = True, mcts_iterations: int = 10000):
        self.game = Game(board_size, num_pieces, False, no_reverse_moves, False)
        self.mcts_iterations = mcts_iterations
        self.max_depth = 1000
    
    def select_move(self, state: Board, moves: list[Move]) -> tuple[Move, Any]:
        # perform mcts and get the root's children
        mcts = MCTS(SearchMoves(state, self, self.max_depth), 'uct')
        mcts.run(iterations=self.mcts_iterations)
        
        child = mcts.get_best_root_child()
        if child:
            move = state.child_board_to_move(child.state)
        else:
            # if no child is found, select a random move
            move = random.choice(moves)
        
        mcts.print_children()
        mcts.remove_unvisited_nodes(None)
        mcts.print_metrics()
        #mcts.draw_graph()

        # return the selected move
        return move, None
