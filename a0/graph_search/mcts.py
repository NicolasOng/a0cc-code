from __future__ import annotations
import random
import math

from typing import Optional, Protocol, Any

class MCTSProblem(Protocol):
    def initial_state(self) -> Any:
        '''
        Returns the initial state of the problem.
        '''
        ...
    
    def is_terminal(self, state: Any) -> bool:
        '''
        Checks if the given state is a terminal state.
        '''
        ...

    def get_successors(self, state: Any) -> list[Any]:
        '''
        Returns a list of successor states for the given state.
        '''
        ...

    def get_reward(self, state: Any) -> float:
        '''
        Returns the reward for the given state.
        May perform a heuristic evaluation or a rollout to determine the reward.
        Useful for MCTS.
        '''
        ...

class MCTSNode:
    def __init__(self, state: Any, parent: Optional[MCTSNode]=None):
        self.state = state
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.visits = 0
        self.reward = 0.0
    
    def fully_expanded(self) -> bool:
        '''
        a node is fully expanded if it has children and all of its children have been visited at least once.
        '''
        return bool(self.children) and all(child.visits > 0 for child in self.children)

class MCTS:
    '''
    Monte Carlo Tree Search (MCTS) implementation for a generic graph problem.
    Based on https://int8.io/monte-carlo-tree-search-beginners-guide/#Policy_network_training_in_Alpha_Go_and_Alpha_Zero
    '''
    def __init__(self, problem: MCTSProblem):
        self.problem = problem
        self.root = MCTSNode(problem.initial_state())

    def run(self, iterations: int=1000) -> None:
        for _ in range(iterations):
            # 1. Traversal/Expansion/Selection
            # starting at the root, traverse the tree using a selection policy (UCT, PUCT, etc.),
            # until reaching a node that is
            # - not fully expanded
            #     - node is not a terminal state, but has no child nodes -> expand it, then select one of its children
            #     - node has unvisited child nodes -> select one of them
            # - a terminal state
            node = self.root
            while node.fully_expanded() and not self.problem.is_terminal(node.state):
                node = max(node.children, key=self.uct)
            
            # 1.1
            # if the node is not terminal, yet has no children,
            # expand it by generating its successors
            if not self.problem.is_terminal(node.state) and not node.children:
                successors = self.problem.get_successors(node.state)
                for succ in successors:
                    child = MCTSNode(succ, parent=node)
                    node.children.append(child)
            
            # 1.2
            # we now have a node that is either terminal or has unvisited children.
            # if it has children, select one of the unvisited ones randomly
            if node.children:
                node = random.choice([child for child in node.children if child.visits == 0])
            
            # 2. Simulation/Rollout/Heuristic Evaluation
            # evaluate the node's state.
            # how this is done depends on the problem implementation.
            reward = self.problem.get_reward(node.state)

            # 3. Backpropagation
            while node:
                node.visits += 1
                node.reward += reward
                node = node.parent
    
    @staticmethod
    def uct(node: MCTSNode) -> float:
        if node.visits == 0:
            return float('inf')
        parent_visits = node.parent.visits if node.parent else 1
        return (node.reward / node.visits) + math.sqrt(2 * math.log(parent_visits) / node.visits)

    def get_best_root_child(self) -> Optional[MCTSNode]:
        '''
        Returns the child of the root node with the highest visit count.
        If there are no children, returns None.
        '''
        return max(self.root.children, key=lambda c: c.visits) if self.root.children else None
    
    def get_root_children(self) -> list[MCTSNode]:
        return self.root.children
