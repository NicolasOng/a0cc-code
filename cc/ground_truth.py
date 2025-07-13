from cc.core import Game, Board, Player, Move
from cc.lookups import CCBaselineSolver
from cc.ranking import CCDefaultRank, CCState
from a0.players.a0 import Policy

from config import config

class GroundTruth:
    '''
    Provides methods to rank, unrank, and get outcomes for boards.
    This class uses the CCDefaultRank and CCBaselineSolver to perform these operations.
    '''

    def __init__(self):
        self.r = CCDefaultRank(config.num_spots, config.num_players, config.num_pieces)
        self.s = CCState(config.num_spots, config.num_pieces, config.num_players)
        self.cc = Game(config.board_size, config.num_pieces, False, False, False)
        self.l = CCBaselineSolver(config.solve_data, config.num_spots, config.num_players, config.num_pieces)
    
    def rank(self, board: Board) -> int:
        '''
        Returns the rank of the given board.
        '''
        self.s.initialize_from_board(board)
        return self.r.rank(self.s)
    
    def unrank(self, rank: int) -> Board:
        '''
        Returns a board from the given rank.
        '''
        self.r.unrank(rank, self.s)
        return self.s.get_board()
    
    def get_outcome(self, board: Board) -> float:
        '''
        Returns the outcome of the game for the given board,
        from the perspective of the current player.
        1 for win, -1 for lose, 0 for draw.
        '''
        return self.l.get_outcome(board)

    def get_winner(self, board: Board) -> Player | None:
        '''
        Returns the winning player for the given board.
        If there is no winner (draw), returns None.
        '''
        return self.l.get_winner(board)
    
    def get_1ply_policy_moves(self, board: Board) -> tuple[list[Move], list[float]]:
        '''
        Returns the 1ply policy for the given board.
        It is based on the outcomes of the moves from the current board.
        The policy is a list of probabilities for each legal move,
        along with the moves themselves.
        '''
        # get the current player
        current_player = board.current_player
        # get all the moves for the given board
        moves = self.cc.generate_moves_for_given_board(board)

        # create a list to hold the move outcomes
        move_outcomes: list[float] = []

        # check the result of each move
        for move in moves:
            # apply the move to the board
            board.apply_move(move)

            # get the winner of the game for the new board
            winner = self.get_winner(board)

            # set the move probability based on the winner
            if winner is None:
                move_outcomes.append(0.0)
            elif winner == current_player:
                move_outcomes.append(1.0)
            elif winner != current_player:
                move_outcomes.append(-1.0)
            
            # undo the move to restore the board state
            # this is necessary to check the next move
            board.undo_move(move)
        
        return moves, move_outcomes

    def get_1ply_policy_list(self, board: Board, for_model: bool) -> list[float]:
        '''
        Returns the 1ply policy for the given board.
        It is based on the outcomes of the moves from the current board.
        The policy is a list of probabilities for each action.
        Its length is board_size ** 4.
        If this will be used for a model,
        it will be rotated 180 degrees when the current player is Player.PLAYER_O.
        '''
        rotate_board = board.current_player == Player.PLAYER_O
        moves, move_outcomes = self.get_1ply_policy_moves(board)
        
        # create a Policy object and set the logits from the moves and outcomes
        p = Policy(config.board_size)
        p.set_logits_from_moves(moves, move_outcomes, rotate_180=rotate_board and for_model)
        return p.get_policy_list()

