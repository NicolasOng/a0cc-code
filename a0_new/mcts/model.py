from typing import Generic

from a0_new.protocols.game import A0Game, Player, T_state, T_action
from a0_new.protocols.model import FullModel

import random

def random_rollout(state: T_state, game: A0Game[T_state, T_action], max_depth: int) -> tuple[bool, Player | None, T_state]:
    # Make a copy of the board to avoid modifying the original state
    # could also do a series of undo moves afterwards, but this is simpler (unsure about performance impact)
    current_state = state.clone()
    for _ in range(max_depth):
        # check if the current state is terminal
        is_done = game.is_terminal(current_state)
        winner = game.get_winner(current_state) if is_done else None
            
        # if so, return the result
        if is_done:
            return is_done, winner, current_state
        
        # if not terminal, get all possible actions for the current player
        actions = game.get_actions(current_state)
        assert len(actions) > 0, "No actions available in non-terminal state"

        # select a random action
        action = random.choice(actions)

        # apply the action to the current state
        current_state.apply_action(action)
    
    # if max depth reached without terminal state, return result as non-terminal
    return False, None, current_state

def policy_max_rollout(state: T_state, game: A0Game[T_state, T_action], model: FullModel[T_state, T_action], max_depth: int) -> tuple[bool, Player | None, T_state]:
    # Make a copy of the board to avoid modifying the original state
    current_state = state.clone()
    for _ in range(max_depth):
        # check if the current state is terminal
        is_done = game.is_terminal(current_state)
        winner = game.get_winner(current_state) if is_done else None
            
        # if so, return the result
        if is_done:
            return is_done, winner, current_state
        
        # if not terminal, get all possible actions for the current player
        actions = game.get_actions(current_state)
        assert len(actions) > 0, "No actions available in non-terminal state"

        # get policy from model
        _, policy = model.evaluate_state(current_state, actions)

        # select the action with the highest value
        action = policy.sample_best_action()
        # apply the action to the current state
        current_state.apply_action(action)
    
    # if max depth reached without terminal state, return result as non-terminal
    return False, None, current_state

class MCTS_Model(Generic[T_state, T_action]):
    def __init__(self,
                 initial_state: T_state,
                 game: A0Game[T_state, T_action],
                 model: FullModel[T_state, T_action],
                 initial_actions: list[T_action] | None = None,
                 rollout_type: str = 'none',
                 rollout_depth: int = -1,
                 calculate_priors: bool = True
                ):
        self._initial_state = initial_state
        self.game = game
        self.model = model
        self.initial_actions = initial_actions
        self.rollout_type = rollout_type
        self.rollout_depth = rollout_depth
        self.calculate_priors = calculate_priors
    
    def initial_state(self) -> T_state:
        '''
        Returns the initial state of the problem.
        '''
        return self._initial_state
    
    def is_terminal(self, state: T_state) -> bool:
        '''
        Checks if the given state is a terminal state.
        '''
        return self.game.is_terminal(state)

    def get_successors(self, state: T_state, is_root: bool) -> tuple[list[T_state], list[float]]:
        '''
        Returns a list of successor states for the given state.
        Optionally, returns a list of prior probabilities for each successor.
        If calculate_priors is False, returns a uniform+normalized list.
        '''
        if is_root and self.initial_actions is not None:
            actions = self.initial_actions
        else:
            # get all possible actions for the current player
            actions = self.game.get_actions(state)

        # create a list of successor states by applying each action
        successors: list[T_state] = []
        for action in actions:
            # create a copy of the state and apply the action
            new_state = state.clone()
            new_state.apply_action(action)

            # add the new state to the list of successors
            successors.append(new_state)
        
        # get priors for the successors by using the model
        # useful if MCTS uses PUCT
        # these are in the perspective of the state's current player
        if not self.calculate_priors:
            uniform_prior = 1.0 / len(successors) if len(successors) > 0 else 0.0
            successor_priors = [uniform_prior for _ in successors]
            return successors, successor_priors

        _, policy = self.model.evaluate_state(state, actions)

        successor_priors = policy.get_action_values(actions)

        return successors, successor_priors

    def get_reward(self, state: T_state) -> float:
        '''
        Returns the reward for the given state,
        considering the root player's perspective.
        If the game is done, it returns the value based on the winner.
        If the game is not done, it uses a rollout or the model to evaluate the state.
        
        I don't want too many if statements here.
        could give this class two models,
        one for this reward method and one for successor priors.
        Both FullModel types, just the get_reward one implements value only.
        Same with getting a policy head or value head policy.
        With a FullModel that uses policy max rollouts, it would itself need a FullModel.
        Refactor for the future?
        '''
        is_done = False
        winner = None
        if self.rollout_type == 'random':
            is_done, winner, state = random_rollout(
                state,
                self.game,
                max_depth=self.rollout_depth
            )
        elif self.rollout_type == 'policy':
            is_done, winner, state = policy_max_rollout(
                state,
                self.game,
                self.model,
                max_depth=self.rollout_depth
            )
        elif self.rollout_type == 'none':
            is_done = self.game.is_terminal(state)
            if is_done: winner = self.game.get_winner(state)
        else:
            raise ValueError(f"Unknown rollout type: {self.rollout_type}")
        
        # if the game is done, return the value based on the winner
        if is_done:
            if winner is None:
                return 0.0
            return 1.0 if winner == self._initial_state.get_current_player() else -1.0
        
        value, _ = self.model.evaluate_state(state)
        # if the current player is not the initial player,
        if state.get_current_player() != self._initial_state.get_current_player():
            # we need to negate the value
            value = -value
        
        return value

    def is_maximizing(self, state: T_state) -> bool:
        '''
        Returns True if the current player to move in the given state is the maximizing player.
        Basically, if it's the same player as the initial state.
        '''
        return state.get_current_player() == self._initial_state.get_current_player()
