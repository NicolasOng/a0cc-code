from __future__ import annotations

from typing import Any, Self, TypeVar

from a0_new.protocols.model import FullModel, NNModel
from a0_new.protocols.player import FullModelPlayer

import random

from a0_new.cc.game import CCGame, CCState, CCAction
from a0_new.cc.models.full import CCFullModel

from a0_new.policy import Policy

T_full_cc_model = TypeVar("T_full_cc_model", bound=FullModel[CCState, CCAction])

class CCPlayer(FullModelPlayer[T_full_cc_model, CCState, CCAction]):
    def __init__(
            self,
            game: CCGame,
            model: T_full_cc_model,
            exploit: bool = False,
            epsilon: float = 0.0,
            dirichlet_epsilon: float = 0.25
        ) -> None:
        self._state: CCState
        self.game = game
        self.model = model
        self.exploit = exploit
        self.epsilon = epsilon
        self.dirichlet_epsilon = dirichlet_epsilon

    def process_state(self, state: CCState, legal_actions: list[CCAction]) -> tuple[CCAction, float, Policy[CCAction]]:
        value, policy = self.model.evaluate_state(state, legal_actions)

        sampling_policy = Policy[CCAction](policy.actions, policy.action_values.copy())
        if self.exploit:
            action = policy.sample_best_action(random_ties=True)
        else:
            sampling_policy.dirichlet_noise(alpha=None, epsilon=self.dirichlet_epsilon)

            if random.random() < self.epsilon:
                action = sampling_policy.sample_random_action()
            else:
                action = sampling_policy.sample_action()
        
        return action, value, policy

    def get_action(self, state: CCState, legal_actions: list[CCAction]) -> CCAction:
        action, _, _ = self.process_state(state, legal_actions)
        return action

    def get_model(self) -> T_full_cc_model:
        return self.model

    def set_model(self, model: T_full_cc_model) -> None:
        self.model = model

    def clone(self) -> Self:
        '''
        Returns a deep copy of the player.
        '''
        ...
    