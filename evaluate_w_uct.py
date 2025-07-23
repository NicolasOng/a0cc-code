import jax
from flax import nnx

from a0.eval.player import player_evaluation
from a0.players.mcts_rollout import MCTSRolloutPlayer
from a0.players.a0 import A0Player
from a0.model import AlphaZeroModel, load_model

board_size = 4
num_pieces = 3

# load the models to evaluate
model = AlphaZeroModel(
    board_size=board_size,
    num_filters=256,
    training=False,
    rngs=nnx.Rngs({'params': jax.random.PRNGKey(0)})
)
model00 = load_model("a0_data/model_0.pkl", model)
model49 = load_model("a0_data/model_49.pkl", model)

# Create players
player00 = A0Player(model=model00, board_size=board_size, num_pieces=num_pieces)
player49 = A0Player(model=model49, board_size=board_size, num_pieces=num_pieces)
player_uct = MCTSRolloutPlayer(board_size=board_size, num_pieces=num_pieces)

# evaluate the players
print("Evaluating A0 Player 0 against UCT Player...")
wins, losses, draws = player_evaluation(board_size, num_pieces, player00, player_uct, num_games=100)
print(f"A0 Player 0 vs UCT Player: {wins} wins, {losses} losses, {draws} draws")

print("Evaluating A0 Player 49 against UCT Player...")
wins, losses, draws = player_evaluation(board_size, num_pieces, player49, player_uct, num_games=100)
print(f"A0 Player 49 vs UCT Player: {wins} wins, {losses} losses, {draws} draws")
