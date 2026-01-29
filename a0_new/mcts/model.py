from typing import Generic

from a0_new.protocols.game import A0Game, T_state, T_action
from a0_new.protocols.model import FullModel

class MCTS_Model(Generic[T_state, T_action]):
    def __init__(self,
                 initial_state: T_state,
                 game: A0Game[T_state, T_action],
                 model: FullModel[T_state, T_action],
                 initial_actions: list[T_action] | None = None,
                 calculate_priors: bool = True
                ):
        self._initial_state = initial_state
        self._initial_state_player = initial_state.get_current_player()
        self.game = game
        self.model = model
        self.initial_actions = initial_actions
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

        policy = self.model.get_state_policy(state, actions)

        successor_priors = policy.get_action_values(actions)

        return successors, successor_priors

    def get_reward(self, state: T_state) -> float:
        '''
        Returns the reward for the given state,
        considering the root player's perspective.
        If the game is done, it returns the value based on the winner.
        If the game is not done, it uses a rollout or the model to evaluate the state.
        '''
        value_est_for = state.get_current_player()
        # estimate the value for the current state
        value = self.model.get_state_value(state)

        # adjust value to be from the perspective of the initial player
        if value_est_for != self._initial_state_player:
            value = -value
        
        return value

    def is_maximizing(self, state: T_state) -> bool:
        '''
        Returns True if the current player to move in the given state is the maximizing player.
        Basically, if it's the same player as the initial state.
        '''
        return state.get_current_player() == self._initial_state_player
