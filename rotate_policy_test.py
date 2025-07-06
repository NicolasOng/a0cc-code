from a0.players.a0 import Policy
from cc.core import Move, Board, player_to_tile, Player

import jax.numpy as jnp

import copy

moves_list = [
    Move(0, 0, 0, 1),
    Move(1, 1, 2, 0),
    Move(2, 3, 3, 3)
]

moves_list_rotated = [Move(3 - move.start.x, 3 - move.start.y, 3 - move.end.x, 3 - move.end.y) for move in moves_list]

p = Policy(4)
p.set_logits_from_moves(moves_list, [0.1, 0.2, 0.3])

print("Original moves:")
board = Board(4)
board.apply_points([move.start for move in moves_list], player_to_tile[Player.PLAYER_X])
print(board.visualize_move_ends(moves_list))

print("Rotated moves:")
board.rotate_board_180()
print(board.visualize_move_ends(moves_list_rotated))

board_size = 4
rotated_policy_mapping = [0] * board_size ** 4
for start_x in range(board_size):
    for start_y in range(board_size):
        for end_x in range(board_size):
            for end_y in range(board_size):
                # get the normal move index
                start_pos = start_x * board_size + start_y
                end_pos = end_x * board_size + end_y
                move_index = start_pos * (board_size * board_size) + end_pos
                # and the rotated move index
                rstart_x, rstart_y = board_size - 1 - start_x, board_size - 1 - start_y
                rstart_pos = rstart_x * board_size + rstart_y
                rend_x, rend_y = board_size - 1 - end_x, board_size - 1 - end_y
                rend_pos = rend_x * board_size + rend_y
                rmove_index = rstart_pos * (board_size * board_size) + rend_pos
                # add the mapping
                rotated_policy_mapping[move_index] = rmove_index

print("setting moves in policy to be some arbitrary value, and getting them based on moves:")
probs = p.get_move_probabilities(moves_list)
#print(p.policy)
print(probs)

logits = copy.deepcopy(p.policy)
# create a new logits with the same shape as the original logits
rotated_logits = jnp.zeros_like(logits)
# fill the new logits with the rotated probabilities
for index in range(logits.shape[0]):
    rotated_index = rotated_policy_mapping[index]
    rotated_logits = rotated_logits.at[rotated_index].set(logits[index])
p.set_logits(rotated_logits, rotate_180=False)

rprobs = p.get_move_probabilities(moves_list_rotated)
print("rotated logits, and getting the values based on rotated moves (should be same as above):")
#print(p.policy)
print(rprobs)

logits = copy.deepcopy(p.policy)
# create a new logits with the same shape as the original logits
rotated_logits = jnp.zeros_like(logits)
# fill the new logits with the rotated probabilities
for index in range(logits.shape[0]):
    rotated_index = rotated_policy_mapping[index]
    rotated_logits = rotated_logits.at[rotated_index].set(logits[index])
p.set_logits(rotated_logits, rotate_180=False)

probs = p.get_move_probabilities(moves_list)

print("rotating again, and getting the values based on original moves (should be same as above):")
print(probs)