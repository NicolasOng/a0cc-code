from __future__ import annotations
import random
import math
from collections import deque

from typing import Optional, Protocol, Any, TypeVar, Generic
BFSState = TypeVar('BFSState')

class BFSProblem(Protocol[BFSState]):
    def initial_state(self) -> BFSState:
        '''
        Returns the initial state of the problem.
        '''
        ...
    
    def is_terminal(self, state: BFSState) -> bool:
        '''
        Checks if the given state is a terminal state.
        '''
        ...
    
    def is_goal(self, state: BFSState) -> bool:
        '''
        Checks if the given state is a goal state.
        '''
        ...

    def get_successors(self, state: BFSState) -> list[BFSState]:
        '''
        Returns a list of successor states for the given state.
        '''
        ...
    
    def get_reward(self, state: BFSState) -> float:
        '''
        Returns the reward for the given state.
        May perform a heuristic evaluation or a rollout to determine the reward.
        Or use ground truth solve data.
        '''
        ...

class BFSNode(Generic[BFSState]):
    def __init__(self, state: BFSState, parent: Optional[BFSNode[BFSState]]=None):
        self.state = state
        self.parent = parent
        self.children: list[BFSNode[BFSState]] = []
        self.depth = 0 if parent is None else parent.depth + 1
        self.visits = 0
        self.reward = 0.0
    
    def fully_expanded(self) -> bool:
        '''
        a node is fully expanded if it has children and all of its children have been visited at least once.
        '''
        return bool(self.children) and all(child.visits > 0 for child in self.children)

class BFS(Generic[BFSState]):
    '''
    Breadth First Search (BFS) implementation for a generic graph problem.
    '''
    def __init__(self, problem: BFSProblem[BFSState], depth: int=1):
        self.problem = problem
        self.root = BFSNode(problem.initial_state())
        self.depth = depth
        self.queue = deque([self.root])
        self.visited: set[BFSState] = set()

    def run(self, iterations: int | None=None) -> None:
        # while the queue is not empty,
        # and we have not reached the maximum number of iterations
        current_iteration = 0
        while self.queue and (iterations is None or current_iteration < iterations):
            current_iteration += 1

            # pop the first node from the queue
            current = self.queue.popleft()
            # add the current node to the visited set
            self.visited.add(current.state)

            # if this node's depth is greater than the maximum depth,
            if current.depth > self.depth:
                # skip it
                continue

            # get the current node's reward and update its parents
            node = current
            reward = self.problem.get_reward(current.state)
            while node:
                node.visits += 1
                node.reward += reward
                node = node.parent

            # if the current node is a goal state,
            if self.problem.is_goal(current.state):
                # TODO: handle this when needed
                pass

            # if the current node is a terminal state,
            if self.problem.is_terminal(current.state):
                # TODO: handle this when needed
                pass
            
            # if the current node is not a terminal state,
            # get its successors and add them (the ones not visited yet) to the queue
            for child in self.problem.get_successors(current.state):
                if child not in self.visited:
                    # create a new node for the child
                    child_node = BFSNode(child, current)
                    # add the child node to the current node's children
                    current.children.append(child_node)
                    # add the child node to the queue
                    self.queue.append(child_node)
    
    def get_root_children(self) -> list[BFSNode[BFSState]]:
        return self.root.children
