import json

class Config:
    def __init__(self, fn: str):
        with open(fn) as f:
            cf = json.load(f)
        self.board_size = cf.get("board_size", 4)
        self.num_players = cf.get("num_players", 2)
        self.num_pieces = cf.get("num_pieces", 3)
        self.num_spots = self.board_size * self.board_size

        self.solve_data = cf.get("solve_data", "solve-data/CC-SOLVE-BASELINE-16-3.dat")

        self.data_folder = cf.get("data_folder", "data/")

config = Config("config/config.json")