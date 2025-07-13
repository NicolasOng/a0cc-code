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
        return self.player.game.get_done(state)

    def get_successors(self, state: Board) -> tuple[list[Board], list[float]]:
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
        
        # get priors for the successors by using the model
        # useful if MCTS uses PUCT
        # these are in the perspective of the state's current player
        _, policy = self.player.model(board_to_input(state))
        p = Policy(len(state.board))
        p.set_logits(policy[0], rotate_180=state.current_player == Player.PLAYER_O)
        p.mask_non_legal_moves(moves)
        p.apply_softmax()
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
        is_done, winner = self.player.game.get_done_and_winner(state)
        if is_done:
            # if the game is done, return the value based on the winner
            if winner is None:
                return 0.0
            return 1.0 if winner == self._initial_state.current_player else -1.0

        board_input = board_to_input(state)
        value, _ = self.player.model(board_input)
        value = float(value[0][0])

        # if the current player is not the initial player,
        if state.current_player != self._initial_state.current_player:
            # we need to negate the value
            value = -value
        
        return value

def board_to_input(board: Board) -> jnp.ndarray:
    '''
    Converts a Board object to a JAX input array.
    The input is a 4D array with shape (1, BOARD_SIZE, BOARD_SIZE, 2).
    The last dimension represents the two players' pieces.
    The first channel is for the current player, and the second channel is for the opponent.
    The NN is trained to predict the value of the current player.
    If the current player is Player.PLAYER_O, the board is rotated 180 degrees.
    This way, the NN also knows the orientation of the board
    If we don't do this, eg if all the channel 1 pieces are on the top,
    the NN won't know if they are in its home or goal area.
    '''
    board_size, _ = board.board_sizes()

    # rotate the board if necessary
    rotate_board = board.current_player == Player.PLAYER_O
    if rotate_board:
        board.rotate_board_180()
        board.flip_pieces()
        board.switch_player()
    
    # get the tiles for each player
    # "current player" is always Player.PLAYER_X due to the rotation
    current_player_tile = player_to_tile[Player.PLAYER_X]
    other_player_tile = player_to_tile[Player.PLAYER_O]

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

    # un-rotate the board if necessary
    if rotate_board:
        board.switch_player()
        board.flip_pieces()
        board.rotate_board_180()

    return jnp.array(input_array)

def create_rotated_policy_mapping(board_size: int) -> list[int]:
    '''
    Creates a mapping for the policy distribution indices
    when the board is rotated 180 degrees.
    This is used to rotate the policy distribution logits.
    '''
    rotated_policy_mapping = [0] * board_size ** 4
    for start_x in range(board_size):
        for start_y in range(board_size):
            for end_x in range(board_size):
                for end_y in range(board_size):
                    # get the normal move index
                    start_pos = start_x * board_size + start_y
                    end_pos = end_x * board_size + end_y
                    move_index = start_pos * (board_size * board_size) + end_pos
                    # and the rotated move index
                    rstart_x, rstart_y = board_size - 1 - start_x, board_size - 1 - start_y
                    rstart_pos = rstart_x * board_size + rstart_y
                    rend_x, rend_y = board_size - 1 - end_x, board_size - 1 - end_y
                    rend_pos = rend_x * board_size + rend_y
                    rmove_index = rstart_pos * (board_size * board_size) + rend_pos
                    # add the mapping
                    rotated_policy_mapping[move_index] = rmove_index

    return rotated_policy_mapping

class Policy:
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.policy: NDArray[np.float32] = np.zeros((self.board_size ** 4), dtype=np.float32) # (board_size ** 4,)
        self.mask: NDArray[np.float32] = np.zeros((self.board_size ** 4), dtype=np.float32) # (board_size ** 4,)
        self.policy_rotation_mapping = create_rotated_policy_mapping(board_size)
    
    def get_policy_list(self) -> list[float]:
        return self.policy.tolist()
    
    def move_to_policy_index(self, move: Move) -> int:
        '''
        Converts a Move object to a policy index using row-major position encoding.
        '''
        board_size = self.board_size
        start_pos = move.start.x * board_size + move.start.y
        end_pos = move.end.x * board_size + move.end.y
        return start_pos * (board_size * board_size) + end_pos

    def policy_index_to_move(self, index: int) -> Move:
        '''
        Converts a policy index to a Move object using row-major position encoding.
        '''
        board_size = self.board_size
        start_pos = index // (board_size * board_size)
        end_pos = index % (board_size * board_size)
        start_x = start_pos // board_size
        start_y = start_pos % board_size
        end_x = end_pos // board_size
        end_y = end_pos % board_size
        return Move(start_x, start_y, end_x, end_y)

    def rotate_policy_list(self, logits: NDArray[np.float32]) -> NDArray[np.float32]:
        '''
        Rotates the policy logits by 180 degrees using the precomputed mapping.
        This is used to adjust the policy distribution when the board is rotated.
        Assumes the given logits has the correct length.
        '''
        rotated_logits: NDArray[np.float32] = np.zeros((self.board_size ** 4), dtype=np.float32)
        for index in range(len(logits)):
            rotated_index = self.policy_rotation_mapping[index]
            rotated_logits[rotated_index] = logits[index]
        return rotated_logits
    
    def rotate_policy(self) -> None:
        '''
        Rotates the policy distribution by 180 degrees in-place.
        This is used to adjust the policy distribution when the board is rotated.
        '''
        self.policy = self.rotate_policy_list(self.policy)
        self.mask = self.rotate_policy_list(self.mask)
    
    def set_logits(self, logits: NDArray[np.float32], rotate_180: bool) -> None:
        '''
        Initializes the policy distribution with the given logits
        The logits are expected to be a list of length BOARD_SIZE**4.
        rotate functionality is used if the board was rotated 180 degrees before being passed into the model.
        '''
        assert len(logits) == self.board_size ** 4, \
            f"Expected logits length {self.board_size ** 4}, but got {len(logits)}."
        self.policy = logits

        # rotate if necessary
        if rotate_180:
            self.rotate_policy()

    def set_logits_from_moves(self, moves: list[Move], values: list[float], rotate_180: bool) -> None:
        '''
        Initializes the policy distribution with the given moves and their corresponding values.
        The moves are expected to be a list of Move objects, and values is a list of probabilities.
        '''
        self.policy = np.zeros((self.board_size ** 4), dtype=np.float32)
        for move, value in zip(moves, values):
            index = self.move_to_policy_index(move)
            self.policy[index] = value
        
        # rotate if necessary
        if rotate_180:
            self.rotate_policy()
    
    def set_legal_moves(self, legal_moves: list[Move]) -> None:
        '''
        Sets the legal moves in the policy object.
        Creates a mask for future use.
        '''
        self.mask = np.zeros((self.board_size ** 4), dtype=np.float32)
        for move in legal_moves:
            idx = self.move_to_policy_index(move)
            self.mask[idx] = 1
    
    def apply_softmax(self, temperature: float, mask: bool) -> None:
        '''
        Applies the softmax function to the logits to get the policy distribution.
        Low temperature values make the distribution more deterministic,
        while high temperature values make it more uniform.
        Mask is used to set non-legal moves to zero probability.
        Non-legal moves need to be set before using this.
        '''
        # apply temperature scaling
        self.policy = self.policy / temperature

        # mask non-legal moves
        if mask:
            self.policy = np.where(self.mask, self.policy, -np.inf)
        
        # Apply softmax using numpy
        # Subtract max for numerical stability
        max_val = np.max(self.policy[self.policy != -np.inf])
        exp_values = np.exp(self.policy - max_val)
        self.policy = exp_values / np.sum(exp_values)

        # ensure non-legal moves have zero probability
        if mask:
            self.policy = np.where(self.mask, self.policy, 0)
    
    def apply_power_normalize(self, tau: float) -> None:
        '''
        Computes the AlphaZero-style softmax over visit counts.
        Args:
            tau (float): Temperature parameter (τ). Lower values → more deterministic.
        '''
        # remove -inf values (non-legal moves)
        # this normalization is necessary to avoid NaNs
        self.policy = jnp.where(self.policy == -jnp.inf, 0.0, self.policy)

        # Avoid divide-by-zero or NaNs if all counts are zero
        if jnp.sum(self.policy) == 0:
            return jnp.ones_like(self.policy) / self.policy.size

        # Apply the (1/τ) power transform
        powered = self.policy ** (1.0 / tau)

        # Normalize to form a probability distribution
        self.policy = powered / jnp.sum(powered)
    
    def get_move_probability(self, move: Move) -> float:
        '''
        Returns the probability of the given move based on the policy distribution.
        Assumes that the move is valid and exists in the policy distribution.
        '''
        index = self.move_to_policy_index(move)
        return float(self.policy[index])
    
    def get_move_probabilities(self, moves: list[Move]) -> list[float]:
        '''
        Returns the probabilities of the given moves based on the policy distribution.
        '''
        return [self.get_move_probability(move) for move in moves]
    
    def get_best_move(self) -> Move:
        '''
        Returns the move with the highest probability based on the policy distribution.
        '''
        index = np.argmax(self.policy)
        return self.policy_index_to_move(index)
    
    def sample_move(self, rng: int) -> Move:
        '''
        Samples a move based on the policy distribution.
        Uses JAX's random choice to sample a move according to the policy probabilities.
        Assumes that the policy has been normalized to sum to 1.
        TODO: Can also implement other sampling strategies, like epsilon-greedy or top-k
        '''
        index = jax.random.choice(jax.random.PRNGKey(rng), len(self.policy), p=self.policy)
        return self.policy_index_to_move(index)

class A0Player:
    def __init__(self, board_size: int, num_pieces: int, model: AlphaZeroModel, use_mcts: bool = True):
        self.model = model
        self.game = Game(board_size, num_pieces, False, True, False)
        self.temperature = 1.0  # Temperature for exploration in MCTS
        self.random_selection_prob = 0.01  # Probability of selecting a random move
        self.mcts_iterations = 64
        self.use_mcts = use_mcts
    
    def select_move(self, state: Board, moves: list[Move]) -> tuple[Move, Any]:
        '''
        Selects a move using MCTS and returns the selected move along with the policy distribution.
        For exploration, it also allows for random choices.
        '''
        p = Policy(len(state.board))

        if self.use_mcts:
            # perform mcts and get the root's children
            mcts = MCTS(SearchMoves(state, self))
            mcts.run(iterations=self.mcts_iterations)
            children = mcts.get_root_children()
            assert len(children) > 0, "No children found in MCTS root node."

            # with the root's children, create a policy distribution logits
            mcts_root_children_visit_counts = [float(child.visits) for child in children]
            mcts_root_children_moves = [state.child_board_to_move(child.state) for child in children]

            # create a well-shaped policy ditribution,
            p.set_logits_from_moves(mcts_root_children_moves, mcts_root_children_visit_counts)
            # mask non-legal moves,
            p.mask_non_legal_moves(moves)
            # softmax it to get the policy distribution
            p.apply_power_normalize(self.temperature)
        else:
            # if not using MCTS, just get the policy distribution from the model
            _, logits = self.model(board_to_input(state))
            p.set_logits(logits[0], rotate_180=state.current_player == Player.PLAYER_O)
            # mask non-legal moves
            p.mask_non_legal_moves(moves)
            # apply softmax to the policy distribution
            p.apply_softmax()

        # select a move based on the policy distribution
        # sampling instead of argmax to allow exploration
        # or randomly select a move with a small probability
        if random.random() < self.random_selection_prob:
            selected_move = random.choice(moves)
        else:
            selected_move = p.sample_move(42)

        # return the selected move and the mcts policy distribution
        return selected_move, p.policy
