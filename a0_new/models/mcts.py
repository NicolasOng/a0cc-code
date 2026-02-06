from a0_new.protocols.game import A0Game, T_state, T_action
from a0_new.protocols.model import FullModel, FullModelOnFull, T_full_model

from a0_new.policy import Policy
from a0_new.mcts import MCTS
from a0_new.mcts_problems.generic import GenericMCTSProblem

class MCTSModel(FullModelOnFull[T_full_model, T_state, T_action]):
    '''
    Model that uses MCTS to evaluate states for a given game and model.
    '''
    def __init__(self,
                 game: A0Game[T_state, T_action],
                 model: T_full_model,
                 iterations: int,
                 selection_policy: str = 'puct',
                 temperature: float = 1.0
                ):
        self.game = game
        self.model = model
        self.iterations = iterations
        self.selection_policy = selection_policy
        self.temperature = temperature

    def evaluate_state(self, state: T_state, actions: list[T_action] | None = None) -> tuple[float, Policy[T_action]]:
        mcts_problem = GenericMCTSProblem(
            initial_state=state,
            game=self.game,
            model=self.model,
            initial_actions=actions
        )
        mcts = MCTS(
            problem=mcts_problem,
            selection_policy=self.selection_policy
        )
        mcts.run(self.iterations)

        children = mcts.get_root_children()
        assert len(children) > 0, "No children found in MCTS root node."

        # get MCTS's estimated value for the state
        mcts_value = mcts.root.reward / mcts.root.visits if mcts.root.visits > 0 else 0.0

        # with the root's children, create policy distribution logits
        mcts_root_children_visit_counts = [float(child.visits) for child in children]
        mcts_root_children_actions: list[T_action] = [state.child_board_to_action(child.state) for child in children]
        mcts_policy = Policy(mcts_root_children_actions, mcts_root_children_visit_counts)
        
        # normalize it to get the policy distribution
        mcts_policy.power_normalize(self.temperature)

        # return the mcts value and policy
        return mcts_value, mcts_policy
    
    def evaluate_states(self, states: list[T_state], actions: list[list[T_action] | None]) -> list[tuple[float, Policy[T_action]]]:
        results: list[tuple[float, Policy[T_action]]] = []
        for state, action_list in zip(states, actions):
            value, policy = self.evaluate_state(state, action_list)
            results.append((value, policy))
        return results
    
    def get_state_value(self, state: T_state) -> float:
        value, _ =self.evaluate_state(state)
        return value

    def get_state_policy(self, state: T_state, actions: list[T_action] | None = None) -> Policy[T_action]:
        _, policy = self.evaluate_state(state, actions)
        return policy
    
    def get_full_model(self) -> T_full_model:
        return self.model
    
    def set_full_model(self, model: T_full_model) -> None:
        self.model = model
