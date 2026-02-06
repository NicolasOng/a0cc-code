from a0_new.protocols.game import A0Game, Player, T_state, T_action
from a0_new.protocols.model import FullModel, FullModelOnFull, T_full_model

from a0_new.policy import Policy

import random

class RolloutModel(FullModelOnFull[T_full_model, T_state, T_action]):
    '''
    Model to be used with MCTS for:
    - state reward estimations (using e.g. rollouts)
    - policy priors for action selection (using e.g. policy/value head policies)
    To declutter MCTS code.
    Uses a FullModel to do evaluations.
    '''
    def __init__(self,
                 game: A0Game[T_state, T_action],
                 model: T_full_model,
                 rollout_type: str = 'none',
                 rollout_depth: int = -1,
                 policy_type: str = 'policy'
                ):
        self.game = game
        self.model = model
        self.rollout_type = rollout_type
        self.rollout_depth = rollout_depth
        self.policy_type = policy_type

    def evaluate_state(self, state: T_state, actions: list[T_action] | None = None) -> tuple[float, Policy[T_action]]:
        value = self.get_state_value(state)
        policy = self.get_state_policy(state, actions)
        return value, policy
    
    def evaluate_states(self, states: list[T_state], actions: list[list[T_action] | None]) -> list[tuple[float, Policy[T_action]]]:
        results: list[tuple[float, Policy[T_action]]] = []
        for state, action_list in zip(states, actions):
            value = self.get_state_value(state)
            policy = self.get_state_policy(state, action_list)
            results.append((value, policy))
        return results
    
    def get_state_value(self, state: T_state) -> float:
        '''
        estimates the value of the given state using the specified rollout method
        '''
        estimating_for = state.get_current_player()

        # perform the appropriate rollout to get the value estimate
        winner = None
        is_done = False
        if self.rollout_type == 'random':
            is_done, winner, state = self.random_rollout(
                state,
                self.game,
                max_depth=self.rollout_depth
            )
        elif self.rollout_type == 'policy':
            is_done, winner, state = self.policy_max_rollout(
                state,
                self.game,
                self.model,
                max_depth=self.rollout_depth
            )
        elif self.rollout_type == 'none':
            is_done = self.game.is_terminal(state)
            if is_done: winner = self.game.get_winner(state)
        else:
            raise ValueError(f"Unknown rollout type: {self.rollout_type}")
        
        # if the game is done after the rollout,
        # return the value based on if the player on the given state won
        if is_done:
            if winner is None:
                return 0.0
            return 1.0 if winner == estimating_for else -1.0
        
        # if the game is not done, use the model to evaluate
        # the state after the rollout
        value = self.model.get_state_value(state)
        # adjust value to be from the perspective of the initial player
        if estimating_for != state.get_current_player():
            value = -value
        
        return value

    def get_state_policy(self, state: T_state, actions: list[T_action] | None = None) -> Policy[T_action]:
        if self.policy_type == 'policy':
            return self.model.get_state_policy(state, actions)
        
        if self.policy_type == 'value':
            return self.value_head_policy(self.model, state, actions if actions is not None else self.game.get_actions(state))
    
        raise ValueError(f"Unknown policy type: {self.policy_type}")
    
    def get_full_model(self) -> T_full_model:
        return self.model
    
    def set_full_model(self, model: T_full_model) -> None:
        self.model = model

    @staticmethod
    def random_rollout(state: T_state, game: A0Game[T_state, T_action], max_depth: int) -> tuple[bool, Player | None, T_state]:
        # Make a copy of the board to avoid modifying the original state
        # could also do a series of undo moves afterwards, but this is simpler (unsure about performance impact)
        current_state = state.clone()
        for _ in range(max_depth):
            # check if the current state is terminal
            is_done = game.is_terminal(current_state)
            winner = game.get_winner(current_state) if is_done else None
                
            # if so, return the result
            if is_done:
                return is_done, winner, current_state
            
            # if not terminal, get all possible actions for the current player
            actions = game.get_actions(current_state)
            assert len(actions) > 0, "No actions available in non-terminal state"

            # select a random action
            action = random.choice(actions)

            # apply the action to the current state
            current_state.apply_action(action)
        
        # if max depth reached without terminal state, return result as non-terminal
        return False, None, current_state

    @staticmethod
    def policy_max_rollout(state: T_state, game: A0Game[T_state, T_action], model: FullModel[T_state, T_action], max_depth: int) -> tuple[bool, Player | None, T_state]:
        # Make a copy of the board to avoid modifying the original state
        current_state = state.clone()
        for _ in range(max_depth):
            # check if the current state is terminal
            is_done = game.is_terminal(current_state)
            winner = game.get_winner(current_state) if is_done else None
                
            # if so, return the result
            if is_done:
                return is_done, winner, current_state
            
            # if not terminal, get all possible actions for the current player
            actions = game.get_actions(current_state)
            assert len(actions) > 0, "No actions available in non-terminal state"

            # get policy from model
            _, policy = model.evaluate_state(current_state, actions)

            # select the action with the highest value
            action = policy.sample_best_action()
            # apply the action to the current state
            current_state.apply_action(action)
        
        # if max depth reached without terminal state, return result as non-terminal
        return False, None, current_state
    
    @staticmethod
    def value_head_policy(model: FullModel[T_state, T_action], state: T_state, actions: list[T_action]) -> Policy[T_action]:
        # create a list of child states by applying each action
        children: list[T_state] = []
        for action in actions:
            # create a copy of the state and apply the action
            new_state = state.clone()
            new_state.apply_action(action)

            # add the new state to the list of successors
            children.append(new_state)
        
        # evaluate all child states using the model's value head
        values_and_policies = model.evaluate_states(children, [None] * len(children))

        # create a policy based on the values
        action_values: list[float] = []
        for value, _ in values_and_policies:
            action_values.append(value)
        
        return Policy(actions, action_values)
