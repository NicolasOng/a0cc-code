import random

from cc.core import Game, Board, Player, Move
from a0.model import AlphaZeroModel
from a0.model_utils import board_to_input, Policy
#from cc.ground_truth import GroundTruth

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

def get_value_head_policy(model: AlphaZeroModel, state: Board, moves: list[Move], for_model: bool = False) -> tuple[NDArray[np.float32], Policy]:
    rotate = state.current_player == Player.PLAYER_O if for_model else False

    values: list[float] = []
    for move in moves:
        state.apply_move(move)
        value, _ = model(jnp.array(board_to_input(state)))
        values.append(float(value[0][0]))
        state.undo_move(move)

    p = Policy(len(state.board))
    p.set_logits_from_moves(moves, values, rotate_180=False)
    p.set_legal_moves(moves)
    p.apply_softmax(temperature=1.0, mask=True)
    if rotate:
        p.rotate_policy()
    
    return p.policy, p

def get_policy_head_policy(model: AlphaZeroModel, state: Board, moves: list[Move], for_model: bool = False) -> tuple[NDArray[np.float32], Policy]:
    '''
    Returns the policy distribution over the given moves for the given state using the model.
    If for_model is True, the policy is rotated according to the model's perspective.
    '''
    rotate = state.current_player == Player.PLAYER_O if for_model else False

    _, policy = model(jnp.array(board_to_input(state)))

    p = Policy(len(state.board))
    p.set_logits(np.array(policy[0]), rotate_180=False)
    p.set_legal_moves(moves)
    p.apply_softmax(temperature=1.0, mask=True)
    if rotate:
        p.rotate_policy()
    
    return p.policy, p

def random_rollout(state: Board, game: Game, max_depth: int) -> tuple[bool, Player | None, Board]:
    current_board = state
    for _ in range(max_depth):
        # check if the current board is terminal
        is_done, winner = game.get_done_and_winner(current_board)
        # if so, return the result
        if is_done:
            return is_done, winner, current_board
        
        # if not terminal, get all possible moves for the current player
        moves = game.generate_moves_for_given_board(current_board)
        assert len(moves) > 0, "No moves available in non-terminal state"

        # select a random move
        move = random.choice(moves)

        # apply the move to the current board
        current_board.apply_move(move)
    
    # if max depth reached without terminal state, return result as non-terminal
    return False, None, current_board

def policy_max_rollout(state: Board, game: Game, model: AlphaZeroModel, max_depth: int, policy_type: str = "policy") -> tuple[bool, Player | None, Board]:
    current_board = state
    for _ in range(max_depth):
        # check if the current board is terminal
        is_done, winner = game.get_done_and_winner(current_board)
        # if so, return the result
        if is_done:
            return is_done, winner, current_board
        
        # if not terminal, get all possible moves for the current player
        moves = game.generate_moves_for_given_board(current_board)
        assert len(moves) > 0, "No moves available in non-terminal state"

        # get policy from model
        if policy_type == "policy":
            _, p = get_policy_head_policy(model, current_board, moves, for_model=False)
        elif policy_type == "value":
            _, p = get_value_head_policy(model, current_board, moves, for_model=False)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

        # select the move with the highest probability
        move_probs = p.get_move_probabilities(moves)
        best_move_index = int(np.argmax(move_probs))
        move = moves[best_move_index]

        # apply the move to the current board
        current_board.apply_move(move)
    
    # if max depth reached without terminal state, return result as non-terminal
    return False, None, current_board

class MCTS_NN:
    def __init__(self, initial_state: Board, game: Game, model: AlphaZeroModel, initial_moves: list[Move] | None = None, rollout_type: str = 'none', rollout_depth: int = -1, policy_type: str = 'policy'):
        self._initial_state = initial_state
        self.game = game
        self.model = model
        self.initial_moves = initial_moves
        self.rollout_type = rollout_type
        self.rollout_depth = rollout_depth
        self.policy_type = policy_type
        #self.gt = GroundTruth()
    
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

    def get_successors(self, state: Board, is_root: bool) -> tuple[list[Board], list[float]]:
        '''
        Returns a list of successor states for the given state.
        '''
        # get all possible moves for the current player
        moves = self.game.generate_moves_for_given_board(state)

        if is_root and self.initial_moves is not None:
            moves = self.initial_moves

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

        # for testing purposes, we can also use the ground truth to get the policy
        #policy = [self.gt.get_1ply_policy_prob_dist_list(state, for_model=True)]

        p = Policy(len(state.board))
        p.set_logits(np.array(policy[0]), rotate_180=False)
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
        is_done = False
        winner = None
        if self.rollout_type == 'random':
            is_done, winner, state = random_rollout(
                state,
                self.game,
                max_depth=self.rollout_depth
            )
        elif self.rollout_type == 'policy':
            is_done, winner, state = policy_max_rollout(
                state,
                self.game,
                self.model,
                max_depth=self.rollout_depth,
                policy_type=self.policy_type
            )
        elif self.rollout_type == 'none':
            is_done, winner = self.game.get_done_and_winner(state)
        else:
            raise ValueError(f"Unknown rollout type: {self.rollout_type}")

        # print(state.board_view())
        # print(f"Is done: {is_done}, Winner: {winner}, root player: {self._initial_state.current_player}")
        # print(f"final value: {0.0 if winner is None else 1.0 if winner == self._initial_state.current_player else -1.0}")
        if is_done:
            # if the game is done, return the value based on the winner
            if winner is None:
                return 0.0
            return 1.0 if winner == self._initial_state.current_player else -1.0

        board_input = board_to_input(state)
        value, _ = self.model(jnp.array(board_input))
        value = float(value[0][0])

        # print("Value from model:", value)
        # print("State current player:", state.current_player)
        # print(f"final value: {value if state.current_player == self._initial_state.current_player else -value}")

        # if the current player is not the initial player,
        if state.current_player != self._initial_state.current_player:
            # we need to negate the value
            value = -value
        
        return value

    def is_maximizing(self, state: Board) -> bool:
        '''
        Returns True if the current player to move in the given state is the maximizing player.
        Basically, if it's the same player as the initial state.
        '''
        #return True
        return state.current_player == self._initial_state.current_player
