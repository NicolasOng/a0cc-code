from typing import Optional, Protocol

from a0_new.protocols.game import T_state, T_action, Player

from a0_new.policy import Policy

class GTProtocol(Protocol[T_state, T_action]):
    def rank(self, state: T_state) -> int:
        '''
        Returns the rank of the given state.
        '''
        ...
    
    def unrank(self, rank: int) -> T_state:
        '''
        Returns a state from the given rank.
        '''
        ...

    def get_max_rank(self) -> int:
        '''
        Returns the maximum rank for the game.
        This is the total number of unique states.
        '''
        ...
    
    def get_outcome(self, state: T_state) -> float:
        '''
        Given a state, returns the ground truth outcome/reward for that state.
        '''
        ...

    def get_winner(self, state: T_state) -> Optional[Player]:
        '''
        Returns the winning player for the given state.
        If there is no winner (draw), returns None.
        '''
        ...
    
    def is_illegal(self, state: T_state) -> bool:
        '''
        Checks if the given state is illegal.
        Returns True if illegal, False otherwise.
        '''
        ...
    
    def is_draw(self, state: T_state) -> bool:
        '''
        Checks if the given state results in a draw.
        Returns True if draw, False otherwise.
        '''
        ...
    
    def get_policy(self, state: T_state) -> Policy[T_action]:
        '''
        Returns the ground truth policy for the given state.
        '''
        ...
    
    def is_trivial(self, state: T_state) -> bool:
        '''
        Checks if the given board is trivial.
        A trivial board is one where all the moves lead to the same outcome.
        '''
        ...
