from typing import Optional, Protocol

from a0_new.protocols.game import T_state

class GTProtocol(Protocol[T_state]):
    _state: T_state
    def get_outcome(self, state: T_state) -> float:
        '''
        Given a state, returns the ground truth outcome/reward for that state.
        Useful for debugging and evaluating MCTS performance.
        '''
        ...

class MCTSProblem(Protocol[T_state]):
    def initial_state(self) -> T_state:
        '''
        Returns the initial state of the problem.
        '''
        ...
    
    def is_terminal(self, state: T_state) -> bool:
        '''
        Checks if the given state is a terminal state.
        '''
        ...

    def get_successors(self, state: T_state, is_root: bool) -> tuple[list[T_state], Optional[list[float]]]:
        '''
        Returns a list of successor states for the given state.
        Optionally, returns a list of prior probabilities for each successor.
        '''
        ...

    def get_reward(self, state: T_state) -> float:
        '''
        Returns the reward for the given state.
        May perform a heuristic evaluation or a rollout to determine the reward.
        Useful for MCTS.
        '''
        ...
    
    def is_maximizing(self, state: T_state) -> bool:
        '''
        Returns True if the current player to move in the given state is the maximizing player.
        I.E. the player is trying to maximize the reward given in get_reward.
        This is used in the selection policy in the MCTS.
        To be specific, if the current node is a maximizing node, all its children will set the value estimate as is,
        while if it's a minimizing node, the value estimate will be negated.
        '''
        ...
