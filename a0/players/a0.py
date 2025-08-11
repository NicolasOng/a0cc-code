from __future__ import annotations
from typing import Any
import random
import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp
import jax

from cc.core import Board, Move, Player, player_to_tile, Game
from a0.graph_search.mcts import MCTS
from a0.model import AlphaZeroModel
from a0.model_utils import board_to_input, Policy

class NNMCTSProblem:
    def __init__(self, initial_state: Board, game: Game, model: AlphaZeroModel):
        self._initial_state = initial_state
        self.game = game
        self.model = model
    
    def initial_state(self) -> Board:
        '''
        Returns the initial state of the problem.
        '''
        return self._initial_state
    
    def is_terminal(self, state: Board) -> bool:
        '''
        Checks if the given state is a terminal state.
        '''
        return self.game.get_done(state)

    def get_successors(self, state: Board) -> tuple[list[Board], list[float]]:
        '''
        Returns a list of successor states for the given state.
        '''
        # get all possible moves for the current player
        moves = self.game.generate_moves_for_given_board(state)

        # create a list of successor states by applying each move
        successors: list[Board] = []
        for move in moves:
            # create a copy of the board and apply the move
            new_board = Board()
            new_board.copy_board(state)
            new_board.apply_move(move)

            # add the new board to the list of successors
            successors.append(new_board)
        
        # get priors for the successors by using the model
        # useful if MCTS uses PUCT
        # these are in the perspective of the state's current player
        _, policy = self.model(jnp.array(board_to_input(state)))
        p = Policy(len(state.board))
        p.set_logits(np.array(policy[0]), rotate_180=state.current_player == Player.PLAYER_O)
        p.set_legal_moves(moves)
        p.apply_softmax(temperature=1.0, mask=True)
        successor_priors = p.get_move_probabilities(moves)

        return successors, successor_priors

    def get_reward(self, state: Board) -> float:
        '''
        Returns the reward for the given state,
        considering the root player's perspective.
        Uses the model's evaluation function to determine the reward.
        If the game is done, it returns the value based on the winner.
        (could implement a rollout in the future)
        '''
        is_done, winner = self.game.get_done_and_winner(state)
        if is_done:
            # if the game is done, return the value based on the winner
            if winner is None:
                return 0.0
            return 1.0 if winner == self._initial_state.current_player else -1.0

        board_input = board_to_input(state)
        value, _ = self.model(jnp.array(board_input))
        value = float(value[0][0])

        # if the current player is not the initial player,
        if state.current_player != self._initial_state.current_player:
            # we need to negate the value
            value = -value
        
        return value

class A0Player:
    def __init__(self, board_size: int, num_pieces: int, model: AlphaZeroModel, exploit: bool = False):
        self.model = model
        self.game = Game(board_size, num_pieces, False, True, False)
        self.temperature = 1.0  # Temperature for exploration in MCTS
        self.random_selection_prob = 0  # Probability of selecting a random move
        self.mcts_iterations = 64
        self.exploit = exploit
    
    def select_move(self, state: Board, moves: list[Move]) -> tuple[Move, Any]:
        '''
        Selects a move using MCTS and returns the selected move along with the policy distribution.
        For exploration, it also allows for random choices.
        '''
        p = Policy(len(state.board))

        # perform mcts and get the root's children
        mcts = MCTS(NNMCTSProblem(state, self.game, self.model))
        mcts.run(iterations=self.mcts_iterations)
        children = mcts.get_root_children()
        assert len(children) > 0, "No children found in MCTS root node."

        # with the root's children, create a policy distribution logits
        mcts_root_children_visit_counts = [float(child.visits) for child in children]
        mcts_root_children_moves = [state.child_board_to_move(child.state) for child in children]

        # create a well-shaped policy distribution,
        p.set_logits_from_moves(mcts_root_children_moves, mcts_root_children_visit_counts, rotate_180=False)
        # mask non-legal moves,
        p.set_legal_moves(moves)
        p.apply_mask(0.0)
        # softmax it to get the policy distribution
        p.apply_power_normalize(self.temperature)

        # get the policy to return later
        mcts_policy = p.policy.copy()

        # if exploiting (eg for testing/use),
        # don't add noise and select the best move
        if self.exploit:
            selected_move = p.get_best_move(random_ties=True, rng_seed=42)
            return selected_move, mcts_policy

        # apply dirichlet noise for exploration
        p.add_dirichlet_noise(alpha=None, epsilon=0.25)

        # select a move based on the policy distribution
        # sampling instead of argmax to allow exploration
        # or randomly select a move with a small probability
        if random.random() < self.random_selection_prob:
            selected_move = random.choice(moves)
        else:
            selected_move = p.sample_move(42)

        # return the selected move and the mcts policy distribution
        return selected_move, mcts_policy
