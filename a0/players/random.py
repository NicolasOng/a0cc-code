import random

from cc.core import Board, Move

class RandomPlayer:
    def select_move(self, state: Board, moves: list[Move]) -> Move:
        '''
        selects a random move from the available moves.
        '''
        return random.choice(moves)
        #return moves[0]
