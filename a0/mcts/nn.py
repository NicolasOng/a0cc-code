from cc.core import Game, Board, Player
from a0.model import AlphaZeroModel
from a0.model_utils import board_to_input, Policy

import jax.numpy as jnp
import numpy as np

class MCTS_NN:
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
