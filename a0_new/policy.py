from __future__ import annotations

from typing import Generic, Sequence

import math
import random

import numpy as np

from a0_new.protocols.game import T_action

class Policy(Generic[T_action]):
    '''
    A simple policy class to hold action values,
    and perform operations on them.
    '''
    def __init__(self, actions: Sequence[T_action], action_values: list[float]) -> None:
        self.actions = actions
        self.action_values = action_values
    
    def get_action_list(self) -> Sequence[T_action]:
        '''
        Returns the list of actions in the policy.
        '''
        return self.actions
    
    def get_action_value(self, action: T_action) -> float:
        '''
        Returns the probability of the given action.
        If the action is not in the policy, returns 0.0.
        '''
        if action in self.actions:
            index = self.actions.index(action)
            return self.action_values[index]
        else:
            return 0.0
        
    def get_action_values(self, actions: Sequence[T_action]) -> list[float]:
        '''
        Returns a list of values for the given actions.
        If an action is not in the policy, its value is 0.0.
        '''
        return [self.get_action_value(action) for action in actions]

    def sample_action(self) -> T_action:
        '''
        Samples an action based on the policy distribution.
        '''
        assert math.isclose(sum(self.action_values), 1.0), f"Sum is {sum(self.action_values)}, expected 1.0"
        return random.choices(self.actions, weights=self.action_values, k=1)[0]

    def sample_random_action(self) -> T_action:
        '''
        Samples an action uniformly at random from the available actions.
        '''
        return random.choice(self.actions)
    
    def sample_best_action(self, random_ties: bool = False) -> T_action:
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
    
    def power_normalize(self, temperature: float) -> None:
        '''
        Normalizes the action values using a power normalization with the given temperature.
        This modifies the action values in place.
        '''
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0 for power normalization.")
        
        if any(not isinstance(value, float) or value != value or value == float('inf') or value == float('-inf') for value in self.action_values):
            raise ValueError("Policy contains NaN or infinite values, cannot normalize.")
        
        powered_values = [value ** (1.0 / temperature) for value in self.action_values]
        total = sum(powered_values)
        if total > 0:
            self.action_values = [value / total for value in powered_values]
        else:
            # If total is 0, assign uniform distribution
            uniform_value = 1.0 / len(self.action_values) if len(self.action_values) > 0 else 0.0
            self.action_values = [uniform_value for _ in self.action_values]
    
    def softmax(self, temperature: float) -> None:
        '''
        Applies softmax to the action values with the given temperature.
        This modifies the action values in place.
        '''
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0.")
        
        if not self.action_values:
            return

        # 1. Scale by temperature first
        # 2. Subtract max of scaled values for numerical stability
        scaled_values = [v / temperature for v in self.action_values]
        max_scaled = max(scaled_values)
        
        # Calculate e^(x - max)
        exp_values = [math.exp(v - max_scaled) for v in scaled_values]
        
        total = sum(exp_values)
        
        # With max subtraction, total will be at least 1.0, 
        # but we keep the check for absolute robustness.
        if total > 0:
            self.action_values = [v / total for v in exp_values]
        else:
            # Fallback to uniform distribution
            n = len(self.action_values)
            self.action_values = [1.0 / n] * n
    
    def dirichlet_noise(self, alpha: float | None = None, epsilon: float = 0.25) -> None:
        """
        Applies Dirichlet noise to the policy for exploration.
        
        Args:
            alpha: Dirichlet concentration parameter (lower = more concentrated = more exploration)
            epsilon: Mixing ratio (0 = no noise, 1 = all noise)
        """
        num_actions = len(self.action_values)
        if num_actions == 0:
            return  # No actions, nothing to do
        
        if alpha is None:
            alpha = float(1 / num_actions)  # Default alpha if not provided
        noise = np.random.dirichlet([alpha] * num_actions)

        # Mix the original policy with the noise
        np_action_values = np.array(self.action_values, dtype=np.float32)
        self.action_values = np.array(((1 - epsilon) * np_action_values) + (epsilon * noise), dtype=np.float32).tolist()

        # Renormalize to ensure valid probability distribution
        total = sum(self.action_values)
        if total > 0:
            self.action_values = [(value / total) for value in self.action_values]
