from __future__ import annotations

from typing import Optional, Any, Protocol, TypeVar, Self
from enum import Enum

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
    def __str__(self) -> str:
        '''
        Returns a string representation of the state.
        Used for printing the state in MCTS debugging.
        '''
        ...
    
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
    
    def get_current_player(self) -> Player:
        '''
        Returns the player to move in this state.
        '''
        ...
    
    def child_board_to_action(self, child_state: Self) -> T_action:
        '''
        Given a child board representation, returns the corresponding action.
        Used in MCTS to map child states to actions.
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
    
    def get_actions(self, state: T_state | None = None) -> list[T_action]:
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
    
    def get_winner(self, state: T_state | None = None) -> Optional[Player]:
        '''
        Given a terminal state, returns the winner (player index), or None for a draw.
        If state is None, uses the results of the current game.
        '''
        ...
    
    def get_current_player(self) -> Player:
        '''
        Returns the player to move in the current state.
        '''
        ...
    
    def get_current_state(self) -> T_state:
        '''
        Returns the current state of the game.
        '''
        ...

T_game = TypeVar("T_game", bound=A0Game[Any, Any])
