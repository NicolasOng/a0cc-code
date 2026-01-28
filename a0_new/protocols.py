from __future__ import annotations

from typing import Optional, Any, Protocol, TypeVar, Self, TYPE_CHECKING
from enum import Enum

import numpy as np
from numpy.typing import NDArray

# This block is ONLY seen by type checkers, never executed at runtime
# We do this to avoid importing JAX/Flax at runtime if not needed
if TYPE_CHECKING:
    from flax import nnx
    # We define a dummy _nnxModule that looks like nnx.Module to the type checker
    _nnxModule = nnx.Module
else:
    # At runtime, we use 'object' so we don't have to import flax
    _nnxModule = object

class Player(Enum):
    X = 0
    O = 1

class A0Action(Protocol):
    ...

T_action = TypeVar("T_action", bound=A0Action)
class A0State(Protocol[T_action]):
    '''
    Protocol for a game state to be used with AlphaZero training.
    Methods are needed for MCTS.
    '''
    _action: T_action # allow invariance...
    def clone(self) -> Self:
        '''
        Returns a deep copy of the state.
        '''
        ...
    
    def apply_action(self, action: T_action) -> None:
        '''
        Applies the given action to the state.
        '''
        ...
    
    def undo_action(self, action: T_action) -> None:
        '''
        Undoes the given action from the state.
        '''
        ...

T_state = TypeVar("T_state", bound=A0State[Any])
class A0Game(Protocol[T_state, T_action]):
    '''
    Protocol for a game to be used with AlphaZero training.
    Assumes 2-Player Sequential Perfect Information games.
    Methods are needed for self-play and MCTS.
    '''
    def reset(self) -> None:
        '''
        Resets the game to the initial state.
        '''
        ...
    
    def step(self, action: T_action) -> tuple[T_state, float, bool, bool, Any]:
        '''
        Takes an action in the game and returns the next state, reward, terminated, truncated, and info.
        Note:
            Because this game is for AlphaZero training, each step alternates the current player,
            rather than a single agent playing against an environment.
        '''
        ...
    
    def get_info(self) -> tuple[T_state, float, bool, bool, Any]:
        '''
        Gets the same data as the step method, but without changing the state.
        '''
        ...
    
    def get_actions(self, state: T_state | None) -> list[T_action]:
        '''
        Given a state, returns a list of legal actions.
        If state is None, returns legal actions for the current game state.
        '''
        ...
    
    def is_legal(self, state: T_state) -> bool:
        '''
        Given a state, returns True if it's legal, False otherwise.
        '''
        ...
    
    def is_terminal(self, state: T_state) -> bool:
        '''
        Given a state, returns True if it's terminal, False otherwise.
        '''
        ...
    
    def get_winner(self, state: T_state) -> Optional[Player]:
        '''
        Given a terminal state, returns the winner (player index), or None for a draw.
        '''
        ...
    
    def get_current_player(self) -> Player:
        '''
        Returns the current player to move.
        '''
        ...

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

class Model(Protocol):
    '''
    Protocol for a generic model used for evaluating states.
    Methods are needed for evaluating states.
    '''
    def evaluate(self, states: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        '''
        Given a batch of states, returns the predicted values and policy distributions.
        '''
        ...

class A0Model(Model, Protocol):
    '''
    Protocol for a trainable model with weights.
    To train properly, it should inherit or contain a Flax nnx.Module.
    Extends the Model protocol.
    '''
    def get_nn_model(self) -> _nnxModule:
        '''
        Returns the underlying neural network model.
        '''
        ...
    
    def get_state(self) -> Any:
        '''
        Returns the model's state (e.g., weights).
        '''
        ...
    
    def set_state(self, state: Any) -> None:
        '''
        Sets the model's state (e.g., weights).
        '''
        ...

T_model = TypeVar("T_model", bound=Model)
class A0Player(PlayerProtocol[Any, Any], Protocol[T_model, T_state, T_action]):
    '''
    Protocol for a player to be used with AlphaZero training.
    Methods are needed for self-play and training the model.
    '''
    _state: T_state
    def process_state(self, state: T_state, legal_actions: list[T_action]) -> tuple[T_action, float, Any]:
        '''
        Given a state and a list of legal actions, selects an action to take.
        Returns the selected action, value, and policy for the state.
        '''
        ...
    
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
