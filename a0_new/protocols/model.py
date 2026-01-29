from __future__ import annotations

from typing import Protocol

from a0_new.protocols.game import T_state, T_action

import numpy as np
from numpy.typing import NDArray
from flax import nnx

from a0_new.policy import Policy

class RawModel(Protocol):
    '''
    Protocol for a raw model that can generate values/policies
    from raw state representations.
    '''
    def evaluate(self, states: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        '''
        Given a batch of raw states, returns the predicted values and policy distributions.
        '''
        ...

class NNModel(RawModel, Protocol):
    '''
    Protocol for a trainable model with weights.
    To train properly, it should inherit or contain a Flax nnx.Module.
    '''
    def get_nn_model(self, training: bool) -> nnx.Module:
        '''
        Returns the underlying neural network model.
        If training is True, it should return the model in training mode (e.g. batchnorm layers).
        If training is False, it should return the model in inference mode.
        '''
        ...
    
    def set_nn_model(self, model: nnx.Module, training: bool) -> None:
        '''
        Sets the underlying neural network model.
        If training is True, it should set the model in training mode (e.g. batchnorm layers).
        If training is False, it should set the model in inference mode.
        '''
        ...

class FullModel(Protocol[T_state, T_action]):
    '''
    Protocol for a model that can generate values/policies
    from full game states.
    '''
    def evaluate_state(self, state: T_state, actions: list[T_action] | None = None) -> tuple[float, Policy]:
        '''
        Given a full game state, returns the predicted value and policy distribution.
        '''
        ...
    
    def evaluate_states(self, states: list[T_state], actions: list[list[T_action] | None]) -> list[tuple[float, Policy]]:
        '''
        Given a full game states, returns the predicted values and policy distributions.
        '''
        ...

class A0Model(NNModel, FullModel[T_state, T_action], Protocol[T_state, T_action]):
    '''
    Protocol for a model to be used with AlphaZero training.
    Must implement both NNModel and FullModel protocols.
    Get/set model methods allow the NN model to be swapped with a dynamic batching model.
    '''
    ...

    def get_model(self) -> RawModel:
        '''
        Returns the underlying model.
        '''
        ...
    
    def set_model(self, model: RawModel) -> None:
        '''
        Sets the underlying model.
        '''
        ...
