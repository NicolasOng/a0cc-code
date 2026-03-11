from __future__ import annotations
from typing import Any

import numpy as np
import jax.numpy as jnp

from cc.core import Board, Move, Player, Game
from a0.model import AlphaZeroModel
from a0.model_utils import board_to_input, Policy

class ModelPlayer:
    def __init__(self, board_size: int, num_pieces: int, model: AlphaZeroModel, no_illegal_moves: bool = True):
        self.model = model
        self.game = Game(board_size=board_size,
                        num_pieces=num_pieces,
                        repeats_for_draw=-1,
                        no_reverse_moves=True,
                        no_illegal_moves=no_illegal_moves)
    
    def select_move(self, state: Board, moves: list[Move]) -> tuple[Move, Any]:
        '''
        Selects a move using the model's learned policy (without MCTS).
        TODO: Can also use the model's value prediction to select the best move.
        '''
        p = Policy(len(state.board))
        # get the policy distribution from the model
        _, logits = self.model(jnp.array(board_to_input(state)))
        # set the logits in the policy, and rotate if necessary
        # the model is trained with the perspective of Player X,
        # so we need to rotate the logits so the policy aligns with the current legal moves
        p.set_logits(np.array(logits[0], dtype=np.float32), rotate_180=state.current_player == Player.PLAYER_O)
        # mask non-legal moves
        p.set_legal_moves(moves)
        p.apply_mask(-1e9)
        # apply softmax to the policy distribution
        p.apply_softmax(temperature=1.0, mask=True)

        # select a move based on the policy distribution
        selected_move = p.get_best_move()
        #selected_move = p.sample_move(42)

        # return the selected move and the mcts policy distribution
        return selected_move, p.policy
