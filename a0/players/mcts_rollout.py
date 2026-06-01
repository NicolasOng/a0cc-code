from __future__ import annotations
from typing import Optional, Any
import math
import random

from cc.core import Game, Board, Move
from a0.graph_search.mcts import MCTS
from a0.mcts.rollout import SearchMoves
from a0.mcts.rollout_strategies import EvaluatorType, PolicyType, make_evaluator, make_policy

class MCTSRolloutPlayer:
    def __init__(self, board_size: int, num_pieces: int,
                 no_reverse_moves: bool = True, no_illegal_moves: bool = True, no_side_moves: bool = True,
                 mcts_iterations: int = 10000,
                 rollout_depth: int = 1000,
                 evaluator: EvaluatorType = EvaluatorType.NONE,
                 policy: PolicyType = PolicyType.RANDOM,
                 policy_epsilon: float = 0.0,
                 c: float = math.sqrt(2),
                 bfs: Optional[Any] = None):
        self.game = Game(board_size=board_size,
                        num_pieces=num_pieces,
                        repeats_for_draw=-1,
                        no_reverse_moves=no_reverse_moves,
                        no_illegal_moves=no_illegal_moves,
                        no_side_moves=no_side_moves)
        self.num_pieces = num_pieces
        self.bfs = bfs
        self.mcts_iterations = mcts_iterations
        self.rollout_depth = rollout_depth
        self.evaluator_type = evaluator
        self.policy_type = policy
        self.policy_epsilon = policy_epsilon
        self.c = c

    def select_move(self, state: Board, moves: list[Move]) -> tuple[Move, Any]:
        # build per-call: evaluator depends on the root player (state.current_player)
        evaluator = make_evaluator(self.evaluator_type, state.current_player,
                                   board_size=len(state.board), home_size=state.home_size,
                                   num_pieces=self.num_pieces, bfs=self.bfs)
        policy = make_policy(self.policy_type, epsilon=self.policy_epsilon)
        search = SearchMoves(state, self.game, self.rollout_depth, evaluator=evaluator, policy=policy)

        # perform mcts and get the root's children
        mcts = MCTS(search, 'uct', c=self.c)
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
