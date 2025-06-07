from __future__ import annotations
from typing import Any
import numpy as np
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
        return self.player.game.terminal_state(state)

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
        _, policy = self.player.model(board_to_input(state))
        p = Policy(len(state.board))
        p.set_logits(policy[0])
        p.mask_non_legal_moves(moves)
        p.apply_softmax()
        successor_priors = p.get_move_probabilities(moves)

        return successors, successor_priors

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

class Policy:
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.policy = jnp.zeros((board_size ** 4,), dtype=jnp.float32)
    
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
    
    def set_logits(self, logits: jnp.ndarray) -> None:
        '''
        Initializes the policy distribution with the given logits
        The logits are expected to be a 1D array of shape (BOARD_SIZE**4,).
        '''
        assert logits.shape[0] == self.board_size ** 4, \
            f"Expected logits shape ({self.board_size ** 4},), but got {logits.shape}."
        self.policy = logits

    def set_logits_from_moves(self, moves: list[Move], values: list[float]) -> None:
        '''
        Initializes the policy distribution with the given moves and their corresponding values.
        The moves are expected to be a list of Move objects, and values is a list of probabilities.
        '''
        arr = np.zeros((self.board_size ** 4,), dtype=np.float32)
        for move, value in zip(moves, values):
            index = self.move_to_policy_index(move)
            arr[index] = value
        self.policy = jnp.array(arr)
    
    def mask_non_legal_moves(self, legal_moves: list[Move]) -> None:
        '''
        Masks the non-legal moves in the policy distribution.
        '''
        # create a 1D mask for the legal moves
        mask = np.zeros_like(self.policy, dtype=bool)
        for move in legal_moves:
            idx = self.move_to_policy_index(move)
            mask[idx] = True
        # apply the mask to the logits
        self.policy = jnp.where(mask, self.policy, -jnp.inf)
    
    def apply_softmax(self) -> None:
        '''
        Applies the softmax function to the logits to get the policy distribution.
        '''
        self.policy = jax.nn.softmax(self.policy)
    
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
        index = jnp.argmax(self.policy)
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
        mcts_root_children_visit_counts = [float(child.visits) for child in children]
        mcts_root_children_moves = [state.child_board_to_move(child.state) for child in children]

        # if sum(mcts_root_children_visit_counts) == 0:
        #     mcts_root_children_visit_counts = [ 1 for _ in mcts_root_children_visit_counts ] 

        # create a well-shaped policy ditribution,
        p = Policy(len(state.board))
        p.set_logits_from_moves(mcts_root_children_moves, mcts_root_children_visit_counts)
        # mask non-legal moves,
        p.mask_non_legal_moves(moves)
        # softmax it to get the policy distribution
        p.apply_power_normalize(self.temperature)

        # select a move based on the policy distribution
        # (select the move with the highest probability)
        selected_move = p.get_best_move()

        if selected_move.start.x == 0 and selected_move.start.y == 0 and selected_move.end.x == 0 and selected_move.end.y == 0:
            print(mcts_root_children_visit_counts)
            print(p.policy)
            for move in mcts_root_children_moves:
                print(move)

        # return the selected move and the mcts policy distribution
        return selected_move, p.policy
