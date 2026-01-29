from __future__ import annotations

from typing import Sequence

import random

from a0_new.protocols.game import A0Action

class Policy:
    '''
    A simple policy class to hold action values,
    and perform operations on them.
    '''
    def __init__(self, actions: list[A0Action], action_values: list[float]) -> None:
        self.actions = actions
        self.action_values = action_values
    
    def get_action_value(self, action: A0Action) -> float:
        '''
        Returns the probability of the given action.
        If the action is not in the policy, returns 0.0.
        '''
        if action in self.actions:
            index = self.actions.index(action)
            return self.action_values[index]
        else:
            return 0.0
        
    def get_action_values(self, actions: Sequence[A0Action]) -> list[float]:
        '''
        Returns a list of values for the given actions.
        If an action is not in the policy, its value is 0.0.
        '''
        return [self.get_action_value(action) for action in actions]
    
    def sample_best_action(self, random_ties: bool = False) -> A0Action:
        '''
        Returns the action with the highest value based on the policy distribution.
        If random_ties=True, breaks ties randomly instead of choosing the first occurrence.
        '''
        max_value = max(self.action_values)
        best_actions = [action for action, value in zip(self.actions, self.action_values) if value == max_value]
        if random_ties:
            return random.choice(best_actions)
        else:
            return best_actions[0]
