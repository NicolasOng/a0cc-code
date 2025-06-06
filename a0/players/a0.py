from __future__ import annotations
from typing import Any
import numpy as np
import jax.numpy as jnp

from cc.core import Board, Move, Player, player_to_tile, Game
from a0.graph_search.mcts import MCTS
from a0.model import AlphaZeroModel

class SearchMoves:
    def __init__(self, initial_state: Board, player: A0Player):
        self._initial_state = initial_state
        self.player = player

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

    def get_successors(self, state: Board) -> list[Board]:
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
        # TODO: get priors for the successors by using the model
        # useful if MCTS uses PUCT
        return successors

    def get_reward(self, state: Board) -> float:
        '''
        Returns the reward for the given state,
        considering the current player's perspective.
        Uses the model's evaluation function to determine the reward.
        (could implement a rollout in the future)
        '''
        board_input = board_to_input(state)
        value, _ = self.player.model(board_input)
        return float(value[0][0])

def board_to_input(board: Board) -> jnp.ndarray:
    '''
    Converts a Board object to a JAX input array.
    The input is a 4D array with shape (1, BOARD_SIZE, BOARD_SIZE, 2).
    The last dimension represents the two players' pieces.
    The first channel is for the current player, and the second channel is for the opponent.
    The NN is trained to predict the value of the current player.
    '''
    board_size, _ = board.board_sizes()
    
    # get the tiles for each player
    other_player = Player.PLAYER_X if board.current_player == Player.PLAYER_O else Player.PLAYER_O
    other_player_tile = player_to_tile[other_player]
    current_player_tile = player_to_tile[board.current_player]

    # get 2D arrays of tile positions for each player
    cp_tiles = [[tile == current_player_tile for tile in row] for row in board.board]
    cp_array = np.array(cp_tiles, dtype=bool)
    op_tiles = [[tile == other_player_tile for tile in row] for row in board.board]
    op_array = np.array(op_tiles, dtype=bool)

    # create the input array
    input_array = np.zeros((1, board_size, board_size, 2), dtype=np.float32)
    # (numpy casts the boolean arrays to float32)
    input_array[0, :, :, 0] = cp_array
    input_array[0, :, :, 1] = op_array

    return jnp.array(input_array)

class PolicyDistribution:
    pass

class A0Player:
    def __init__(self, board_size: int, num_pieces: int, model: AlphaZeroModel):
        self.model = model
        self.game = Game(board_size=board_size, num_pieces=num_pieces)
        self.temperature = 1.0  # Temperature for exploration in MCTS
        self.mcts_iterations = 1000
    
    def select_move(self, state: Board, moves: list[Move]) -> tuple[Move, Any]:
        # perform mcts and get the root's children
        mcts = MCTS(SearchMoves(state, self))
        mcts.run(iterations=self.mcts_iterations)
        children = mcts.get_root_children()
        assert len(children) > 0, "No children found in MCTS root node."

        # with the root's children, create a policy distribution logits
        visits = [child.visits for child in children]

        # create a well-shaped policy ditribution and mask non-legal moves
        # the legal moves were given

        # softmax it to get the policy distribution

        # select a move based on the policy distribution
        # (for now, just select the move with the highest probability)

        # return the selected move and the mcts policy distribution

        pass
