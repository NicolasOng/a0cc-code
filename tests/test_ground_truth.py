from cc.ground_truth import GroundTruth

from cc.core import Game
from a0.players.a0 import Policy

from config import config

def test_ground_truth():
    print("Testing GroundTruth class...")
    gt = GroundTruth()
    cc = Game(config.board_size, config.num_pieces, False, False, False)
    
    
    # get a board with 6/14 winning children
    board = gt.unrank(297232)
    
    p = gt.get_1ply_policy(board)

    print(p)

    new_p = Policy(config.board_size)
    new_p.set_logits(p, False)

    moves = cc.generate_moves_for_given_board(board)
    print("Current player:", board.current_player)
    for move in moves:
        board.apply_move(move)
        print(move, new_p.get_move_probability(move), gt.get_winner(board))
        board.undo_move(move)
    

if __name__ == "__main__":
    test_ground_truth()
