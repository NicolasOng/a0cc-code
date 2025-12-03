import random

from cc.core import Board, Move
#from cc.ranking import CCDefaultRank, CCState

class RandomPlayer:
    def __init__(self, random_percent: float = 1.0):
        # num_spots = board_size * board_size
        # self.r = CCDefaultRank(num_spots, 2, num_pieces)
        # self.s = CCState(num_spots, num_pieces, 2)
        self.random_percent = random_percent
    
    # def rank(self, board: Board) -> int:
    #     self.s.initialize_from_board(board)
    #     return self.r.rank(self.s)
    
    def select_move(self, state: Board, moves: list[Move]) -> tuple[Move, None]:
        '''
        selects a random move from the available moves.
        '''
        if random.random() < self.random_percent:
            selected_move = random.choice(moves)
        else:
            selected_move = moves[0]
        
        return selected_move, None
