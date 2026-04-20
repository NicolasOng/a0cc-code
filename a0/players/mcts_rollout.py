from __future__ import annotations
from typing import Optional, Any
import random

from cc.core import Game, Board, Move
from a0.graph_search.mcts import MCTS
from a0.mcts.random_rollout import SearchMoves

class MCTSRolloutPlayer:
    def __init__(self, board_size: int, num_pieces: int, no_reverse_moves: bool = True, no_illegal_moves: bool = True, no_side_moves: bool = True, mcts_iterations: int = 10000):
        self.game = Game(board_size=board_size,
                        num_pieces=num_pieces,
                        repeats_for_draw=-1,
                        no_reverse_moves=no_reverse_moves,
                        no_illegal_moves=no_illegal_moves,
                        no_side_moves=no_side_moves)
        self.mcts_iterations = mcts_iterations
        self.max_depth = 1000
    
    def select_move(self, state: Board, moves: list[Move]) -> tuple[Move, Any]:
        # perform mcts and get the root's children
        mcts = MCTS(SearchMoves(state, self.game, self.max_depth), 'uct')
        mcts.run(iterations=self.mcts_iterations)
        
        child = mcts.get_best_root_child()
        if child:
            move = state.child_board_to_move(child.state)
        else:
            # if no child is found, select a random move
            move = random.choice(moves)
        
        if False:
            mcts.print_children()
            mcts.remove_unvisited_nodes(None)
            mcts.print_metrics()
            #mcts.draw_graph()

        # return the selected move
        return move, None
