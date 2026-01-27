from __future__ import annotations

from typing import Optional, Any, Protocol, TypeVar
import multiprocessing
import concurrent.futures
from concurrent.futures import Future, wait, FIRST_COMPLETED
import os
import pickle
import dill

import numpy as np
from numpy.typing import NDArray

from config import config

from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

class A0Action(Protocol):
    pass

class A0State(Protocol):
    '''
    Protocol for a game state to be used with AlphaZero training.
    Methods are needed for MCTS.
    '''
    def clone(self) -> A0State:
        '''
        Returns a deep copy of the state.
        '''
        ...
    
    def apply_action(self, action: A0Action) -> None:
        '''
        Applies the given action to the state.
        '''
        ...
    
    def undo_action(self, action: A0Action) -> None:
        '''
        Undoes the given action from the state.
        '''
        ...

class A0Game(Protocol):
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
    
    def step(self, action: A0Action) -> tuple[A0State, float, bool, bool, Any]:
        '''
        Takes an action in the game and returns the next state, reward, terminated, truncated, and info.
        Note:
            Because this game is for AlphaZero training, each step alternates the current player,
            rather than a single agent playing against an environment.
        '''
        ...
    
    def get_info(self) -> tuple[A0State, float, bool, bool, Any]:
        '''
        Gets the same data as the step method, but without changing the state.
        '''
        ...
    
    def get_actions(self, state: A0State | None) -> list[A0Action]:
        '''
        Given a state, returns a list of legal actions.
        If state is None, returns legal actions for the current game state.
        '''
        ...
    
    def is_legal(self, state: A0State) -> bool:
        '''
        Given a state, returns True if it's legal, False otherwise.
        '''
        ...
    
    def is_terminal(self, state: A0State) -> bool:
        '''
        Given a state, returns True if it's terminal, False otherwise.
        '''
        ...
    
    def get_winner(self, state: A0State) -> Optional[int]:
        '''
        Given a terminal state, returns the winner (player index), or None for a draw.
        '''
        ...

class Player(Protocol):
    '''
    Protocol for a generic player object.
    Methods are needed for game playing.
    '''
    def get_action(self, state: A0State, legal_actions: list[A0Action]) -> A0Action:
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
    Extends the Model protocol.
    '''
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
class A0Player(Player, Protocol[T_model]):
    '''
    Protocol for a player to be used with AlphaZero training.
    Methods are needed for self-play and training the model.
    '''
    def process_state(self, state: A0State, legal_actions: list[A0Action]) -> tuple[A0Action, float, Any]:
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

def alphazero(starting_iteration: int) -> None:
    pass

if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="training_alphazero_new"
    )

    logger.info(f"config: {config.path}")
    logger.info(f"output_dir: {config.output_dir}")

    # Set the multiprocessing start method to 'spawn' for compatibility with JAX
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
