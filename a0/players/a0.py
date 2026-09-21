from __future__ import annotations
from typing import Any
import copy
import random
import numpy as np
from numpy.typing import NDArray

from cc.core import Board, Move, Player, player_to_tile, Game
from a0.graph_search.mcts import MCTS
from a0.model import AlphaZeroModel
from a0.model_utils import board_to_input, Policy
from a0.mcts.nn import MCTS_NN
from a0.mcts.gt import MCTS_GT

class A0Player:
    def __init__(self, board_size: int, num_pieces: int, model: AlphaZeroModel, exploit: bool = False, mcts_samples: int = 64, no_reverse_moves: bool = True, no_illegal_moves: bool = True, no_side_moves: bool = False, rollout_type: str = "none", rollout_depth: int = -1, policy_type: str = "policy", epsilon: float = 0.1, dirichlet_epsilon: float = 0.25, value_mode: str = "raw", value_sign_threshold: float = 0.05, c_puct: float | None = None, rollout_policy_epsilon: float = 0.0) -> None:
        self.model = model
        self.game = Game(
            board_size=board_size,
            num_pieces=num_pieces,
            repeats_for_draw=-1,
            no_reverse_moves=no_reverse_moves,
            no_illegal_moves=no_illegal_moves,
            no_side_moves=no_side_moves
        )
        self.temperature = 1.0  # Temperature for exploration in MCTS
        self.random_selection_prob = epsilon  # Probability of selecting a random move
        self.mcts_iterations = mcts_samples
        self.exploit = exploit
        self.rollout_type = rollout_type
        self.rollout_depth = rollout_depth
        self.policy_type = policy_type
        self.dirichlet_epsilon = dirichlet_epsilon
        self.value_mode = value_mode
        self.value_sign_threshold = value_sign_threshold
        # PUCT exploration constant for this player's search; None → the
        # training c_puct from the config (see MCTS.__init__).
        self.c_puct = c_puct
        # epsilon for the BEST/BACK playout policies when rollout_type names one
        self.rollout_policy_epsilon = rollout_policy_epsilon

    def select_move(self, state: Board, moves: list[Move]) -> tuple[Move, Any]:
        '''
        Selects a move using MCTS and returns the selected move along with the policy distribution.
        For exploration, it also allows for random choices.
        '''
        p = Policy(len(state.board))

        # perform mcts and get the root's children
        mcts = MCTS(MCTS_NN(
                state,
                self.game,
                self.model,
                initial_moves=moves,
                policy_type=self.policy_type,
                rollout_type=self.rollout_type,
                rollout_depth=self.rollout_depth,
                value_mode=self.value_mode,
                value_sign_threshold=self.value_sign_threshold,
                rollout_policy_epsilon=self.rollout_policy_epsilon
            ),
            selection_policy="puct",
            c_puct=self.c_puct
        )
        mcts.run(iterations=self.mcts_iterations)
        children = mcts.get_root_children()
        assert len(children) > 0, "No children found in MCTS root node."

        # mcts.print_children()
        # print(state.board_view())

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
        mcts_p = copy.deepcopy(p)
        # we rotate the policy if the current player is O,
        # since this is for training the model
        if state.current_player == Player.PLAYER_O:
            mcts_p.rotate_policy()
        mcts_policy = mcts_p.policy

        # if exploiting (eg for testing/use),
        # don't add noise and select the best move
        if self.exploit:
            selected_move = p.get_best_move(random_ties=True, rng_seed=42)
            return selected_move, mcts_policy

        # apply dirichlet noise for exploration
        p.add_dirichlet_noise(alpha=None, epsilon=self.dirichlet_epsilon)

        # select a move based on the policy distribution
        # sampling instead of argmax to allow exploration
        # or randomly select a move with a small probability
        if random.random() < self.random_selection_prob:
            selected_move = random.choice(moves)
        else:
            selected_move = p.sample_move(42)

        # return the selected move and the mcts policy distribution
        return selected_move, mcts_policy

    def get_value_and_policy(self, state: Board, legal_moves: list[Move] | None, mcts_type: str = "NN", normalize_type: str = "power normalize", mcts_key: str = "uct", error_rate: float = 0.2) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        '''
        Returns the value and policy for the given state using the model.
        The policy is rotated if the current player is O,
        to maintain consistency in training.
        value is a np array of shape (1, 1)
        policy is a np array of shape (1, board_size**4)
        '''
        p = Policy(len(state.board))

        # perform mcts and get the root's children
        mcts_problem_object = None
        if mcts_type == "NN":
            mcts_problem_object = MCTS_NN(state, self.game, self.model, initial_moves=legal_moves, rollout_type=self.rollout_type, rollout_depth=self.rollout_depth, policy_type=self.policy_type, value_mode=self.value_mode, value_sign_threshold=self.value_sign_threshold, rollout_policy_epsilon=self.rollout_policy_epsilon)
        elif mcts_type == "GT":
            mcts_problem_object = MCTS_GT(state, self.game, error_rate=error_rate, initial_moves=legal_moves)
        
        assert mcts_problem_object is not None, f"Invalid mcts_type '{mcts_type}'"

        mcts = MCTS(mcts_problem_object, selection_policy=mcts_key, c_puct=self.c_puct)
        mcts.run(iterations=self.mcts_iterations)
        children = mcts.get_root_children()
        assert len(children) > 0, "No children found in MCTS root node."

        # with the root's children, create a policy distribution logits
        mcts_root_children_visit_counts = [float(child.visits) for child in children]
        mcts_root_children_moves = [state.child_board_to_move(child.state) for child in children]

        # create a well-shaped policy distribution,
        p.set_logits_from_moves(mcts_root_children_moves, mcts_root_children_visit_counts, rotate_180=False)
        # mask non-legal moves,
        if legal_moves is not None:
            p.set_legal_moves(legal_moves)
        else:
            p.set_legal_moves(mcts_root_children_moves)
        p.apply_mask(0.0)
        # softmax it to get the policy distribution
        p.apply_power_normalize(self.temperature)
        #p.apply_softmax(temperature=4.5, mask=True)

        # we rotate the policy if the current player is O,
        # since this is for training/evaluating the model
        if state.current_player == Player.PLAYER_O:
            p.rotate_policy()
        mcts_policy = p.policy
        # convert to np array of shape (1, board_size**4)
        mcts_policy = np.expand_dims(mcts_policy, axis=0)

        # get MCTS's estimated value for the state
        mcts_value = mcts.root.reward / mcts.root.visits if mcts.root.visits > 0 else 0.0
        # and convert it to a numpy array of shape (1, 1)
        mcts_value = np.array([[mcts_value]])

        # return the mcts value and policy
        return mcts_value, mcts_policy
