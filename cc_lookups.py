from solve_file_loader import SolveData
from cc_ranking import CCDefaultRank, CCState

class CCBaselineSolver:
    def __init__(self, filename: str, num_spots: int, num_players: int, num_pieces: int):
        self.solve_data = SolveData(filename)
        self.r = CCDefaultRank(num_spots, num_players, num_pieces)
    
    def lookup(self, s: CCState) -> int:
        rank = self.r.rank(s)
        outcome = self.solve_data.get(rank)
        return outcome
