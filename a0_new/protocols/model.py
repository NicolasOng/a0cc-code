from __future__ import annotations

from typing import Any, Protocol, TypeVar, TypeAlias, Union
from abc import abstractmethod

from a0_new.protocols.game import T_state, T_action

import numpy as np
from numpy.typing import NDArray
from flax import nnx
import jax.numpy as jnp

from a0_new.policy import Policy

class RawModel(Protocol):
    '''
    Protocol for a raw model that can generate values/policies
    from raw state representations.
    '''
    def evaluate(self, states: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        '''
        Given a batch of raw states, returns the predicted values and policy logits (not probabilities).
        First dimension of states must be the batch dimension.
        '''
        ...

class TrainableModel(nnx.Module):
    @abstractmethod
    def inference(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        '''
        non-training inference.
        Given a batch of raw states, returns the predicted values and policy logits (not probabilities).
        Format of arrays should match that of the evaluate method.
        Method should be JIT-compiled.
        '''
        ...
    
    @abstractmethod
    def train_inference(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        '''
        training inference.
        Given a batch of raw states, returns the predicted values and policy logits (not probabilities).
        Format of arrays should match that of the evaluate method.
        Method should not be JIT-compiled (training script will handle that).
        '''
        ...
    
    @abstractmethod
    def save_to_file(self, path: str) -> None:
        '''
        Saves the model parameters to the given path.
        '''
        ...
    
    @classmethod
    @abstractmethod
    def load_from_file(cls, file_path: str) -> TrainableModel:
        """Each implementation decides how to load itself."""
        ...

T_nnx = TypeVar("T_nnx", bound=TrainableModel)
class NNModel(RawModel, Protocol[T_nnx]):
    '''
    Protocol for a trainable model with weights.
    To train properly, it should inherit or contain a Flax nnx.Module.
    IMPORTANT: evaluate method is for inference, should be JIT-compiled, and not used for training.
    '''
    def get_nn_model(self) -> T_nnx:
        '''
        Returns the underlying neural network model.
        '''
        ...
    
    def set_nn_model(self, model: T_nnx) -> None:
        '''
        Sets the underlying neural network model.
        '''
        ...

class FullModel(Protocol[T_state, T_action]):
    '''
    Protocol for a model that can generate values/policies
    from full game states.
    '''
    def evaluate_state(self, state: T_state, actions: list[T_action] | None = None) -> tuple[float, Policy[T_action]]:
        '''
        Given a full game state, returns the predicted value and policy logits.
        '''
        ...
    
    def evaluate_states(self, states: list[T_state], actions: list[list[T_action] | None]) -> list[tuple[float, Policy[T_action]]]:
        '''
        Given a full game states, returns the predicted values and policy logits.
        '''
        ...
    
    def get_state_value(self, state: T_state) -> float:
        '''
        Given a full game state, returns the predicted value.
        '''
        ...
    
    def get_state_policy(self, state: T_state, actions: list[T_action] | None = None) -> Policy[T_action]:
        '''
        Given a full game state, returns the predicted policy logits.
        '''
        ...

T_full_model = TypeVar("T_full_model", bound=FullModel[Any, Any])
class FullModelOnFull(FullModel[T_state, T_action], Protocol[T_full_model, T_state, T_action]):
    '''
    Protocol for a FullModel that uses a FullModel for evaluations.
    '''
    def get_full_model(self) -> T_full_model:
        '''
        Returns the underlying full model.
        '''
        ...
    
    def set_full_model(self, model: T_full_model) -> None:
        '''
        Sets the underlying full model.
        '''
        ...

T_nn_model = TypeVar("T_nn_model", bound=NNModel[TrainableModel])
T_raw_model = TypeVar("T_raw_model", bound=RawModel)
class FullModelOnRaw(FullModel[T_state, T_action], Protocol[T_raw_model, T_state, T_action]):
    '''
    Protocol for a FullModel that uses a RawModel for evaluations.
    '''
    def get_raw_model(self) -> T_raw_model:
        '''
        Returns the underlying raw model.
        '''
        ...
    
    def set_raw_model(self, model: T_raw_model) -> None:
        '''
        Sets the underlying raw model.
        '''
        ...
    
    def get_raw_state(self, state: T_state) -> NDArray[np.float32]:
        '''
        Given a full game state, returns the corresponding raw state representation.
        '''
        ...
    
    def get_legal_actions_mask(self, state: T_state, actions: list[T_action] | None, for_model: bool = True) -> NDArray[np.float32]:
        '''
        Given a full game state, returns a mask of legal actions.
        If for_model is True, the mask is formatted for use with the model's policy output.
        '''
        ...
    
    def get_raw_policy(self, state: T_state, policy: Policy[T_action], for_model: bool = True) -> NDArray[np.float32]:
        '''
        Given a full game state and a policy, returns the raw policy logits as an array.
        '''
        ...

RecursiveFullOnRawModel: TypeAlias = Union[
    FullModelOnRaw[T_raw_model, Any, Any],
    FullModelOnFull['RecursiveFullOnRawModel', Any, Any]
]
