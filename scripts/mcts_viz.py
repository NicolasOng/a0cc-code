from cc.core import Game, Move
from a0.mcts.nn import MCTS_NN
from a0.graph_search.mcts import MCTS
from a0.model import load_model
from cc.ground_truth import GroundTruth

from config import config

g = Game(config.board_size, config.num_pieces, False, False, False)
gt = GroundTruth()
#g.end_turn(Move(0, 2, 1, 2))

model = load_model(f"{config.training_dir}/model_{450}.pkl")

mcts = MCTS(MCTS_NN(g.board, g, model), selection_policy='puct')
mcts.run(iterations=64)

mcts.print_children()

mcts.remove_unvisited_nodes(mcts.root)
#mcts.print_metrics()
#mcts.draw_graph()

#MCTS.print_tree(mcts.root)
#MCTS.print_tree_compact(mcts.root, 1)
MCTS.print_tree_full(mcts.root, 1, gt=gt)