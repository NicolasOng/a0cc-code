from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp
import jax

from cc.core import Board, Move, Player, player_to_tile, board_to_home_size

def board_to_input(board: Board) -> NDArray[np.float32]:
    '''
    Converts a Board object to a numpy array.
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

    return input_array

def input_to_board(input_array: NDArray[np.float32], player_o: bool = False) -> Board:
    '''
    Converts a numpy array back to a Board object.
    The input is expected to be a 4D array with shape (1, BOARD_SIZE, BOARD_SIZE, 2).
    The last dimension represents the two players' pieces.
    The first channel is for the current player, and the second channel is for the opponent.
    '''
    # create a board (default is Player X)
    board_size = input_array.shape[1]
    board = Board(board_size, board_to_home_size[board_size])

    # get the tiles for each player
    cp_tiles = input_array[0, :, :, 0] > 0.5
    op_tiles = input_array[0, :, :, 1] > 0.5

    # fill the board with the tiles
    for x in range(board_size):
        for y in range(board_size):
            if cp_tiles[x, y]:
                board.board[x][y] = player_to_tile[Player.PLAYER_X]
            elif op_tiles[x, y]:
                board.board[x][y] = player_to_tile[Player.PLAYER_O]
    
    if player_o:
        # if the board is for Player O, rotate it 180 degrees
        board.rotate_board_180()
        board.flip_pieces()
        board.switch_player()
    
    return board

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
        self.mask: NDArray[np.int8] = np.zeros((self.board_size ** 4), dtype=np.int8) # (board_size ** 4,)
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

    def rotate_policy_list(self, logits: NDArray[np.generic]) -> NDArray[np.generic]:
        '''
        Rotates the policy logits by 180 degrees using the precomputed mapping.
        This is used to adjust the policy distribution when the board is rotated.
        Assumes the given logits has the correct length.
        '''
        rotated_logits: NDArray[np.generic] = np.zeros((self.board_size ** 4), dtype=logits.dtype)
        for index in range(len(logits)):
            rotated_index = self.policy_rotation_mapping[index]
            rotated_logits[rotated_index] = logits[index]
        return rotated_logits
    
    def rotate_policy(self) -> None:
        '''
        Rotates the policy distribution by 180 degrees in-place.
        This is used to adjust the policy distribution when the board is rotated.
        '''
        self.policy = self.rotate_policy_list(self.policy).astype(np.float32)
        self.mask = self.rotate_policy_list(self.mask).astype(np.int8)

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
        self.mask = np.zeros((self.board_size ** 4), dtype=np.int8)
        for move in legal_moves:
            idx = self.move_to_policy_index(move)
            self.mask[idx] = 1
    
    def apply_mask(self, mask_value: float) -> None:
        '''
        Applies a mask to the policy distribution.
        Sets non-legal moves to the specified mask value.
        '''
        self.policy = np.where(self.mask, self.policy, mask_value)
    
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
        Ensure there are no -inf or NaN values in the policy before applying this.
        Args:
            tau (float): Temperature parameter (τ). Lower values → more deterministic.
        '''
        # Avoid divide-by-zero or NaNs if all counts are zero
        if np.sum(self.policy) == 0:
            self.policy = np.array(np.ones_like(self.policy) / self.policy.size, dtype=np.float32)

        # Apply the (1/τ) power transform
        powered = self.policy ** (1.0 / tau)

        # Normalize to form a probability distribution
        self.policy = np.array(powered / np.sum(powered), dtype=np.float32)
    
    def add_dirichlet_noise(self, alpha: float | None, epsilon: float) -> None:
        """
        Applies Dirichlet noise to the policy for exploration, only to legal moves.
        Legal moves should be set before calling this method,
        and non-legal moves should be masked before or after.
        
        Args:
            alpha: Dirichlet concentration parameter (lower = more concentrated)
            epsilon: Mixing ratio (0 = no noise, 1 = all noise)
        """
        # find how many legal moves there are
        num_legal_moves = int(np.sum(self.mask))
        
        if num_legal_moves == 0:
            return  # No legal moves, nothing to do
        
        # Generate Dirichlet noise only for legal moves
        if alpha is None:
            alpha = float(1 / num_legal_moves)  # Default alpha if not provided
        noise_values = np.random.dirichlet([alpha] * num_legal_moves)
        
        # Create a full noise array with zeros for masked positions
        noise = np.zeros_like(self.policy)
        noise[self.mask] = noise_values

        # Mix the original policy with the noise
        self.policy = np.array(((1 - epsilon) * self.policy) + (epsilon * noise), dtype=np.float32)

        # Renormalize only the legal moves to ensure valid probability distribution
        if np.sum(self.policy[self.mask]) > 0:
            self.policy[self.mask] = self.policy[self.mask] / np.sum(self.policy[self.mask])

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

    def get_best_move(self, random_ties: bool = False, rng_seed: int | None = None) -> Move:
        '''
        Returns the move with the highest probability based on the policy distribution.
        If random_ties=True, breaks ties randomly instead of choosing the first occurrence.
        '''
        if random_ties:
            if rng_seed is not None:
                np.random.seed(rng_seed)
            max_prob = np.max(self.policy)
            max_indices = np.where(self.policy == max_prob)[0]
            index = int(np.random.choice(max_indices))
        else:
            index = int(np.argmax(self.policy))
        return self.policy_index_to_move(index)
    
    def sample_move(self, rng: int) -> Move:
        '''
        Samples a move based on the policy distribution.
        Assumes that the policy has been normalized to sum to 1.
        TODO: Can also implement other sampling strategies, like epsilon-greedy or top-k
        '''
        np.random.seed(rng)
        index = np.random.choice(len(self.policy), p=self.policy)
        return self.policy_index_to_move(index)
