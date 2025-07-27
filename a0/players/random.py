import random

from cc.core import Board, Move

class RandomPlayer:
    def select_move(self, state: Board, moves: list[Move]) -> tuple[Move, None]:
        '''
        selects a random move from the available moves.
        '''
        return random.choice(moves), None
        #return moves[0]
