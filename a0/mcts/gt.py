import random

from cc.core import Board, Game
from cc.ground_truth import GroundTruth

def random_number(n: int) -> float:
    random.seed(n)
    return random.random()

class MCTS_GT:
    def __init__(self, initial_state: Board, game: Game, error_rate: float = 0.0):
        self._initial_state = initial_state
        self.game = game
        self.gt = GroundTruth()
        self.error_rate = error_rate

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

    def get_successors(self, state: Board) -> tuple[list[Board], None]:
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

        return successors, None

    def get_reward(self, state: Board) -> float:
        '''
        Returns the reward for the given state,
        considering the root player's perspective.
        Uses the ground truth to determine the reward,
        plus a probability of making a mistake.
        '''
        value = self.gt.get_outcome(state)

        if random_number(self.gt.rank(state)) < self.error_rate:
            # simulate a mistake by flipping the value
            value = -value

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
        return state.current_player == self._initial_state.current_player