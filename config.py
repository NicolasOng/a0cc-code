import os
import sys
import json

class Config:
    board_size: int
    num_players: int
    num_pieces: int
    num_spots: int

    solve_data: str
    output_dir: str

    training_iterations: int
    training_samples: int
    turn_limit: int | None
    replay_buffer_size: int
    training_batch_size: int

    def __init__(self, config_fn: str, default_config_fn: str = "config/config.json"):
        '''
        Initialize the configuration from a JSON file.
        - config_fn: Path to the configuration file.
        - default_config_fn: Path to the default configuration file.
        The configuration file should contain the following
        - board_size: int, size of the board
        - num_players: int, number of players
        - num_pieces: int, number of pieces per player
        - solve_data: str, path to the solve data file
        - output_dir: str, directory for output data
        - training_iterations: int, number of training iterations
        - training_samples: int, number of training samples
        - turn_limit: int or None, limit on the number of turns in a game
        - replay_buffer_size: int, size of the replay buffer
        - training_batch_size: int, batch size for training
        '''
        # Load the default configuration first
        with open(default_config_fn) as f:
            cf = json.load(f)
        self.__dict__.update(cf)

        # Then load the specific configuration file
        # this overrides the default values with the specific ones
        with open(config_fn) as f:
            cf = json.load(f)
        self.__dict__.update(cf)

        # calculate derived attributes
        self.num_spots = self.board_size * self.board_size

        # create output and other directories if they do not exist
        os.makedirs(self.output_dir, exist_ok=True)
        self.training_dir = self.output_dir + "training/"
        os.makedirs(self.training_dir, exist_ok=True)
        self.log_dir = self.output_dir + "logs/"
        os.makedirs(self.log_dir, exist_ok=True)
        self.plot_dir = self.output_dir + "plots/"
        os.makedirs(self.plot_dir, exist_ok=True)
        self.eval_dir = self.output_dir + "eval/"
        os.makedirs(self.eval_dir, exist_ok=True)
        self.validation_dir = self.output_dir + "validations/"
        os.makedirs(self.validation_dir, exist_ok=True)

config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.json"
config = Config(config_path, "config/config.json")
