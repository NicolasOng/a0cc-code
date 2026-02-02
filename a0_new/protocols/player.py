from __future__ import annotations

from typing import Any, Protocol, TypeVar

from a0_new.protocols.model import FullModel
from a0_new.protocols.game import T_state, T_action

from a0_new.policy import Policy

class PlayerProtocol(Protocol[T_state, T_action]):
    '''
    Protocol for a generic player object.
    Methods are needed for game playing.
    '''
    _state: T_state
    def get_action(self, state: T_state, legal_actions: list[T_action]) -> T_action:
        '''
        Given a state and a list of legal actions, selects an action to take.
        '''
        ...
    
    def process_state(self, state: T_state, legal_actions: list[T_action]) -> tuple[T_action, float, Policy]:
        '''
        Given a state and a list of legal actions, selects an action to take.
        Returns the selected action, value, and policy for the state.
        '''
        ...

T_model = TypeVar("T_model", bound=FullModel[Any, Any])
class A0Player(PlayerProtocol[T_state, T_action], Protocol[T_model, T_state, T_action]):
    '''
    Protocol for a player to be used with AlphaZero training.
    Methods are needed for self-play and training the model.
    '''
    model: T_model
    def get_model(self) -> T_model:
        '''
        Returns the player's model.
        '''
        ...

    def set_model(self, model: T_model) -> None:
        '''
        Sets the player's model.
        '''
        ...

    def clone(self) -> A0Player[T_model, T_state, T_action]:
        '''
        Returns a deep copy of the player.
        '''
        ...
    