import json

class Config:
    def __init__(self, fn: str):
        with open(fn) as f:
            cf = json.load(f)
        #TODO: self.__dict__.update(dictionary)?
        self.board_size = cf.get("board_size", 4)
        self.num_players = cf.get("num_players", 2)
        self.num_pieces = cf.get("num_pieces", 3)
        self.num_spots = self.board_size * self.board_size

        self.solve_data = cf.get("solve_data", "solve-data/CC-SOLVE-BASELINE-16-3.dat")

        self.data_folder = cf.get("data_folder", "data/")

        self.training_iterations = cf.get("training_iterations", 50)
        self.training_samples = cf.get("training_samples", 320)
        self.training_dir = cf.get("training_dir", "data/training/")
        self.turn_limit: int | None = cf.get("turn_limit", 80)
        self.replay_buffer_size = cf.get("replay_buffer_size", 10000)
        self.training_batch_size = cf.get("training_batch_size", 64)

config = Config("config/config.json")