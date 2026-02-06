from a0_new.protocols.model import FullModelOnRaw, T_raw_model
from a0_new.cc.game import CCGame, CCState, CCAction, Player
from a0_new.policy import Policy

# utils from previous, should be moved to a0.cc.model_utils or similar
from a0.model_utils import board_to_input, Policy as PolicyUtil

import numpy as np
from numpy.typing import NDArray

class CCFullModel(FullModelOnRaw[T_raw_model, CCState, CCAction]):
    def __init__(self, model: T_raw_model, game: CCGame):
        self.model = model
        self.game = game
        self.policy_util = PolicyUtil(game.board_size)
    
    def get_raw_model(self) -> T_raw_model:
        return self.model
    
    def set_raw_model(self, model: T_raw_model) -> None:
        self.model = model

    def evaluate_state(self, state: CCState, actions: list[CCAction] | None = None) -> tuple[float, Policy[CCAction]]:
        return self.evaluate_states([state], [actions])[0]
    
    def evaluate_states(self, states: list[CCState], actions: list[list[CCAction] | None]) -> list[tuple[float, Policy[CCAction]]]:
        # convert states to raw representations
        # list of (1, board_size, board_size, 2) np arrays
        raw_states_list = [self.get_raw_state(state) for state in states]
        # concatenate along batch dimension. (batch, board_size, board_size, 2)
        raw_states = np.concatenate(raw_states_list, axis=0)
        
        # submit batch to raw model
        # (batch, 1) (batch, board_size ** 4) np arrays
        raw_values, raw_policies = self.model.evaluate(raw_states)

        # convert outputs to (value, Policy) tuples
        value_list: list[float] = raw_values.flatten().tolist()

        # (batch, board_size ** 4) -> list of (board_size ** 4, ) np arrays
        raw_policies_list: list[NDArray[np.float32]] = [p for p in raw_policies]
        policy_list: list[Policy[CCAction]] = []
        for state, action_list, raw_policy in zip(states, actions, raw_policies_list):
            self.policy_util.clear()

            if action_list is None:
                # if no actions provided, use all legal actions
                legal_actions = self.game.get_actions(state)
            else:
                legal_actions = action_list
            
            was_rotated = state.get_current_player() == Player.O
            self.policy_util.set_logits(raw_policy, rotate_180=was_rotated)
            action_values = self.policy_util.get_move_probabilities([action.move for action in legal_actions])

            policy = Policy[CCAction](legal_actions, action_values)
            policy.softmax(temperature=1.0)

            policy_list.append(policy)
        
        return list(zip(value_list, policy_list))
    
    def get_state_value(self, state: CCState) -> float:
        return self.evaluate_state(state)[0]
    
    def get_state_policy(self, state: CCState, actions: list[CCAction] | None = None) -> Policy[CCAction]:
        return self.evaluate_state(state, actions)[1]

    def get_raw_state(self, state: CCState) -> NDArray[np.float32]:
        return board_to_input(state.board)
    
    def get_legal_actions_mask(self, state: CCState, actions: list[CCAction] | None, for_model: bool = True) -> NDArray[np.float32]:
        '''
        output mask is of shape (board_size ** 4, )
        '''
        to_rotate = state.get_current_player() == Player.O if for_model else False
        self.policy_util.clear()
        
        if actions is None:
            legal_actions = self.game.get_actions(state)
        else:
            legal_actions = actions

        self.policy_util.set_legal_moves([action.move for action in legal_actions])
        if to_rotate:
            self.policy_util.rotate_policy()
        
        # casts the boolean mask to float32,
        # with legal moves as 1.0 and illegal moves as 0.0
        return np.array(self.policy_util.mask, dtype=np.float32)
    
    def get_raw_policy(self, state: CCState, policy: Policy[CCAction], for_model: bool = True) -> NDArray[np.float32]:
        '''
        output policy is of shape (board_size ** 4, )
        '''
        to_rotate = state.get_current_player() == Player.O if for_model else False
        self.policy_util.clear()

        self.policy_util.set_logits_from_moves(
            [action.move for action in policy.actions],
            policy.action_values,
            rotate_180=to_rotate
        )
        return self.policy_util.policy.copy()
