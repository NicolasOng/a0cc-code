from cc.core import Board, Player
from cc.solvedata import SolveData
from cc.ranking import CCDefaultRank, CCPSRank12, CCState
from typing import Any

result_dict = {
    2: 'Win',
    1: 'Loss',
    0: 'Draw',
    3: 'Illegal'
}

class Group:
    def __init__(self):
        self.symmetricRank = -1
        self.changed = 0
        self.handled = False
        self.assignedCount = 0
        self.memoryOffset = 0
        self.symmetryRedundant = False

class LookupBase:
    def init_metadata(self, r: CCPSRank12, cc: CCState) -> None:
        """
        Initializes the groups list with symmetry and memory offset info.
        """
        self.groups = [Group() for _ in range(r.get_max_p1_rank())]
        memoryOffset = 0
        symmetricStates = 0
        totalStates = 0
        for x in range(len(self.groups)):
            self.groups[x].symmetricRank = -1
        for x in range(len(self.groups)):
            self.groups[x].changed = 0
            self.groups[x].handled = False
            self.groups[x].assignedCount = 0
            s = CCState()
            r.unrank(x, 0, s)
            sym = CCState()
            cc.flip_player(s, sym, 0)
            otherRank = int(r.rank_p1(sym))
            if otherRank < x:
                self.groups[x].symmetryRedundant = True
                self.groups[x].symmetricRank = otherRank
                self.groups[otherRank].symmetricRank = x
            else:
                self.groups[x].memoryOffset = memoryOffset
                memoryOffset += r.get_max_p2_rank()
                self.groups[x].symmetryRedundant = False
                symmetricStates += r.get_max_p2_rank()
            totalStates += r.get_max_p2Rank()
        # Only group creation logic is implemented as requested.

class CCBaselineSolver:
    def __init__(self, filename: str, num_spots: int, num_players: int, num_pieces: int):
        self.solve_data = SolveData(filename)
        self.r = CCDefaultRank(num_spots, num_players, num_pieces)
        self.ccstate = CCState(num_spots, num_pieces, num_players)
    
    def lookup(self, s: CCState) -> int:
        '''
        Equivalent to tResult BaselineSolver::Lookup(const CCState &s) const in BaselineSolver.cpp.
        '''
        rank = self.r.rank(s)
        outcome = self.solve_data.get(rank)
        return outcome

    def board_lookup(self, board: Board) -> int:
        '''
        Looks up the outcome of a board state.
        This is a convenience method that converts a Board to CCState and calls lookup.
        '''
        self.ccstate.initialize_from_board(board)
        return self.lookup(self.ccstate)

    def get_outcome(self, board: Board) -> float:
        '''
        Returns the outcome of a board.
        '''
        # get the raw result from the solve data
        solve_data_outcome = self.board_lookup(board)

        # convert the result to an outcome in Player X's perspective
        if solve_data_outcome == 0 or solve_data_outcome == 3: # Draw or Illegal
            sd_winner = None
        elif solve_data_outcome == 1: # Loss
            sd_winner = Player.PLAYER_O
        elif solve_data_outcome == 2: # Win
            sd_winner = Player.PLAYER_X
        else:
            sd_winner = None  # Handle unexpected result
        
        # convert the outcome to a float from the current player's perspective
        if sd_winner is None:
            return 0.0
        elif sd_winner == board.current_player:
            return 1.0
        elif sd_winner != board.current_player:
            return -1.0
        else:
            return 0.0

class FullSymmetry:
    def __init__(self, filename: str, num_spots: int, num_players: int, num_pieces: int):
        self.solve_data = SolveData(filename)
        self.r = CCPSRank12(num_spots, num_players, num_pieces)
    
    def lookup(self, s: CCState) -> int:
        '''
        Equivalent to tResult Solver::Lookup(const CCState &s) const in FullSymmetrySolver.cpp.
        '''
        pass

class FasterSymmetry:
    pass

class ParallelBitCCSolver:
    pass
