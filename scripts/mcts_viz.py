from cc.core import Game, Move
from a0.players.a0 import NNMCTSProblem
from a0.graph_search.mcts import MCTS
from a0.model import load_model

from config import config

g = Game(config.board_size, config.num_pieces, False, False, False)
g.end_turn(Move(0, 2, 1, 2))

model = load_model(f"{config.training_dir}/model_{49}.pkl")

mcts = MCTS(NNMCTSProblem(g.board, g, model))
mcts.run(iterations=512)

mcts.print_children()

mcts.remove_unvisited_nodes(mcts.root)
#mcts.print_metrics()
#mcts.draw_graph()

#MCTS.print_tree(mcts.root)
#MCTS.print_tree_compact(mcts.root)
MCTS.print_tree_full(mcts.root, 1)