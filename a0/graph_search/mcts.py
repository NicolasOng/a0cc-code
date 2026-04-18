from __future__ import annotations
import random
import math

from typing import Optional, Protocol, Any

import networkx as nx
import matplotlib.pyplot as plt

from collections import deque

from cc.ground_truth import GroundTruth
from config import config

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

    def get_successors(self, state: Any, is_root: bool) -> tuple[list[Any], Optional[list[float]]]:
        '''
        Returns a list of successor states for the given state.
        Optionally, returns a list of prior probabilities for each successor.
        '''
        ...

    def get_reward(self, state: Any) -> float:
        '''
        Returns the reward for the given state.
        May perform a heuristic evaluation or a rollout to determine the reward.
        Useful for MCTS.
        '''
        ...
    
    def is_maximizing(self, state: Any) -> bool:
        '''
        Returns True if the current player to move in the given state is the maximizing player.
        I.E. the player is trying to maximize the reward given in get_reward.
        This is used in the selection policy in the MCTS.
        To be specific, if the current node is a maximizing node, all its children will set the value estimate as is,
        while if it's a minimizing node, the value estimate will be negated.
        '''
        ...

class MCTSNode:
    def __init__(self, state: Any, prior: float, parent: Optional[MCTSNode]=None, is_maximizing: bool=True):
        self.state = state
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.visits = 0
        self.reward = 0.0
        self.prior = prior
        self.is_maximizing = is_maximizing
    
    def fully_expanded(self) -> bool:
        '''
        a node is fully expanded if it has children and all of its children have been visited at least once.
        a visit: a simulation or rollout that has been performed on the node.
        '''
        return bool(self.children) and all(child.visits > 0 for child in self.children)

class MCTS:
    '''
    Monte Carlo Tree Search (MCTS) implementation for a generic graph problem.
    Based on https://int8.io/monte-carlo-tree-search-beginners-guide/#Policy_network_training_in_Alpha_Go_and_Alpha_Zero
    '''
    def __init__(self, problem: MCTSProblem, selection_policy: str = 'uct'):
        self.problem = problem
        self.root = MCTSNode(problem.initial_state(), 1.0)

        # set the selection policy
        if selection_policy == 'uct':
            self.key = self.uct
        elif selection_policy == 'puct':
            self.key = self.puct
        else:
            raise ValueError(f"Invalid selection policy: {selection_policy}. Choose 'uct' or 'puct'.")

    def run(self, iterations: int=64) -> None:
        for _ in range(iterations):
            # 1. Traversal/Expansion/Selection
            # starting at the root, traverse the tree using a selection policy (UCT, PUCT, etc.),
            # until reaching a node that is
            # - not fully expanded
            #     - has no child nodes
            #     - or has unvisited child nodes
            # - or a terminal state
            node = self.root
            while node.fully_expanded() and not self.problem.is_terminal(node.state):
                node = max(node.children, key=self.key)
            
            # 1.1
            # if the node is not terminal, yet has no children,
            # expand it by generating its successors
            if not self.problem.is_terminal(node.state) and not node.children:
                successors, priors = self.problem.get_successors(node.state, node is self.root)
                maximizing = self.problem.is_maximizing(node.state)
                if priors is None:
                    default_prior = 1.0 / len(successors) if successors else 0.0
                    priors = [default_prior] * len(successors)
                for succ, prior in zip(successors, priors):
                    child = MCTSNode(succ, prior, parent=node, is_maximizing=maximizing)
                    node.children.append(child)
            
            # 1.2
            # we now have a node that is either terminal or has unvisited children.
            # if it has children, select one of the unvisited ones with the highest prior.
            if node.children:
                unvisited_children = [child for child in node.children if child.visits == 0]
                max_prior = max(child.prior for child in unvisited_children)
                highest_prior_unvisited_children = [child for child in unvisited_children if child.prior == max_prior]
                node = random.choice(highest_prior_unvisited_children)

            # 2. Simulation/Rollout/Heuristic Evaluation
            # evaluate the node's state.
            # how this is done depends on the problem implementation.
            reward = self.problem.get_reward(node.state)

            # 3. Backpropagation
            while node:
                node.visits += 1
                node.reward += reward
                node = node.parent
                # reward = 0.9 * reward
    
    @staticmethod
    def uct(node: MCTSNode) -> float:
        assert node.parent is not None, "UCT called on root node"
        # prioritize unvisited nodes
        if node.visits == 0:
            return float('inf')
        # exploitation factor (reward / visits)
        exploit = node.reward / node.visits
        if not node.is_maximizing:
            exploit = -exploit
        # exploration factor (c * sqrt(ln(N) / n))
        explore = math.sqrt(2) * math.sqrt(math.log(node.parent.visits) / node.visits)
        return exploit + explore
    
    @staticmethod
    def puct(node: MCTSNode) -> float:
        assert node.parent is not None, "PUCT called on root node"
        # prioritize unvisited nodes
        if node.visits == 0:
            return float('inf')
        # exploitation factor (reward / visits)
        exploit = node.reward / node.visits
        if not node.is_maximizing:
            exploit = -exploit
        # exploration/prior factor (c_puct * P * (sqrt(N) / (1 + n)))
        explore = config.c_puct * node.prior * (math.sqrt(node.parent.visits) / (1 + node.visits))
        return exploit + explore

    def get_best_root_child(self) -> Optional[MCTSNode]:
        '''
        Returns the child of the root node with the highest visit count.
        If there are no children, returns None.
        '''
        # best_child = max(self.root.children, key=lambda c: c.reward / c.visits if c.visits > 0 else float('-inf'))
        return max(self.root.children, key=lambda c: c.visits) if self.root.children else None
    
    def get_root_children(self) -> list[MCTSNode]:
        return self.root.children

    def draw_graph(self) -> None:
        '''
        Draws the MCTS tree as a directed graph using NetworkX and Matplotlib.
        '''
        # create a dict to hold the nodes and their IDs
        node_ids: dict[int, int] = {}
        # use a queue to perform a breadth-first traversal of the tree
        queue = deque([self.root])
        # create a directed graph
        G = nx.DiGraph()
        # simple BFS implementation to traverse the tree
        while queue:
            # pop the first node from the queue
            node = queue.popleft()
            # create its (custom) ID 
            node_id = len(node_ids)
            node_ids[id(node)] = node_id

            # add the node to the graph with its ID and attributes
            G.add_node(node_id, label=f"V={node.visits}\nR={node.reward:.2f}\nP={node.prior:.2f}\nAR={node.reward/node.visits if node.visits > 0 else 0.0:.2f}")

            # if the node has a parent, add an edge from the parent to this node
            if node.parent is not None:
                parent_id = node_ids[id(node.parent)]
                G.add_edge(parent_id, node_id)
            
            # add all children to the queue for further processing
            for child in node.children:
                queue.append(child)
        # draw the graph using NetworkX and Matplotlib
        #pos = nx.spring_layout(G, seed=42)  # positions for all nodes
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')  # Best for trees
        except Exception as _:
            # Fallback if pygraphviz not available
            pos = nx.kamada_kawai_layout(G)
        labels = nx.get_node_attributes(G, 'label')
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=1500, font_size=8)
        plt.show()
        #plt.savefig("mcts_tree.png")
        plt.clf()
    
    def print_children(self) -> None:
        '''
        Prints the children of the root node.
        '''
        print(f"Root has {len(self.root.children)} children:")
        print("Visits, Reward, Average Reward, Prior")
        print(f"{self.root.visits}, {self.root.reward:.2f}, {self.root.reward / self.root.visits if self.root.visits > 0 else 0.0:.2f}, {self.root.prior:.2f}")
        for child in self.root.children:
            print(f"\t{child.visits}, {child.reward:.2f}, {child.reward / child.visits if child.visits > 0 else 0.0:.2f}, {child.prior:.2f}")

    def remove_unvisited_nodes(self, node: MCTSNode | None) -> None:
        '''
        Removes all nodes from the tree that have not been visited.
        This is useful to clean up the tree after a search.
        '''
        # If no node is provided, start from the root
        if node is None:
            self.remove_unvisited_nodes(self.root)
            return
        
        # This code is run if a node is provided
        # first, remove all children that have not been visited
        node.children = [child for child in node.children if child.visits > 0]
        # then, recursively remove unvisited nodes from the children
        for child in node.children:
            self.remove_unvisited_nodes(child)
    
    def print_metrics(self) -> None:
        '''
        Gets and prints the following metrics:
        - Number of nodes in the tree
        - number of leaves
        - max/min/avg depth of the tree
        - max/min/avg branching factor
        '''
        if not self.root:
            print("Empty tree")
            return
        
        # Initialize metrics
        total_nodes = 0
        total_leaves = 0
        depths = []
        branching_factors = []
        
        # Use BFS to traverse the tree and collect metrics
        queue = deque([(self.root, 0)])  # (node, depth)
        
        while queue:
            node, depth = queue.popleft()
            total_nodes += 1
            
            # Check if it's a leaf node
            if not node.children:
                total_leaves += 1
                # record the leaf node depths
                depths.append(depth)
            else:
                # record the internal branching factors
                branching_factors.append(len(node.children))
                # Add children to queue
                for child in node.children:
                    queue.append((child, depth + 1))
        
        # Calculate statistics
        max_depth = max(depths) if depths else 0
        min_depth = min(depths) if depths else 0
        avg_depth = sum(depths) / len(depths) if depths else 0
        
        max_branching = max(branching_factors) if branching_factors else 0
        min_branching = min(branching_factors) if branching_factors else 0
        avg_branching = sum(branching_factors) / len(branching_factors) if branching_factors else 0
        
        # Print metrics
        print(f"MCTS Tree Metrics:")
        print(f"  Number of nodes: {total_nodes}")
        print(f"  Number of leaves: {total_leaves}")
        print(f"  Depth - Max: {max_depth}, Min: {min_depth}, Avg: {avg_depth:.2f}")
        print(f"  Branching factor - Max: {max_branching}, Min: {min_branching}, Avg: {avg_branching:.2f}")

    @staticmethod
    def print_tree_compact(node: MCTSNode, to_depth: int | None = None, prefix: str = "", is_last: bool = True) -> None:
        '''
        Prints the tree with multi-line states formatted compactly.
        '''
        if to_depth is not None and to_depth < 0:
            return

        # Print current node
        connector = "└── " if is_last else "├── "
        node_info = f"({'+' if node.is_maximizing else '-'}) V:{node.visits} R:{node.reward:.2f} P:{node.prior:.2f}, AR:{node.reward / node.visits if node.visits > 0 else 0.0:.2f}, PUCT:{MCTS.puct(node) if node.parent else 0:.2f}, UCT:{MCTS.uct(node) if node.parent else 0:.2f}"
        print(f"{prefix}{connector}{node_info}")
        
        # Prepare prefix for children  
        child_prefix = prefix + ("    " if is_last else "│   ")
        
        # Print children
        for i, child in enumerate(sorted(node.children, key=lambda c: c.visits, reverse=True)):
            is_child_last = (i == len(node.children) - 1)
            MCTS.print_tree_compact(child, to_depth - 1 if to_depth is not None else None, child_prefix, is_child_last)

    @staticmethod 
    def print_tree_full(node: MCTSNode, to_depth: int | None = None, prefix: str = "", is_last: bool = True, gt: GroundTruth | None = None) -> None:
        '''
        Prints the tree with multi-line states formatted compactly.
        '''
        if to_depth is not None and to_depth < 0:
            return
        
        gt_value = "N/A" if gt is None else f"{gt.get_outcome(node.state)}"
        
        # Print the state (assuming it's a Board object)
        state_string = node.state.board_view()
        state_string += "\n" + f"Current player: {node.state.current_player}"
        state_lines = state_string.split('\n')
        for line in [""] + state_lines:
            print(f"{prefix}{'│   '} {line}")

        # Print current node
        connector = "└── " if is_last else "├── "
        node_info = f"({'+' if node.is_maximizing else '-'}) V:{node.visits} R:{node.reward:.2f} P:{node.prior:.2f}, AR:{node.reward / node.visits if node.visits > 0 else 0.0:.2f}, PUCT:{MCTS.puct(node) if node.parent else 0:.2f}, UCT:{MCTS.uct(node) if node.parent else 0:.2f}, GT:{gt_value}"
        print(f"{prefix}{connector}{node_info}")
        
        # Prepare prefix for children  
        child_prefix = prefix + ("    " if is_last else "│   ")

        # Print children
        for i, child in enumerate(sorted(node.children, key=lambda c: c.visits, reverse=True)):
            is_child_last = (i == len(node.children) - 1)
            MCTS.print_tree_full(child, None if to_depth is None else to_depth - 1, child_prefix, is_child_last, gt)