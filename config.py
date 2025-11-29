import os
import sys
import json

class Config:
    path: str

    board_size: int
    num_players: int
    num_pieces: int
    num_spots: int

    solve_data: str
    output_dir: str
    input_dir: str

    training_iterations: int
    training_samples: int
    turn_limit: int | None
    replay_buffer_size: int
    training_batch_size: int
    mcts_samples: int
    
    backwards_moves: bool
    sideways_moves: bool
    repeats_for_draw: int

    root_game_has_all_moves: bool
    rollout_type: str
    rollout_depth: int
    policy_type: str

    num_trials: int

    learning_rate: float
    weight_decay: float

    eval_neighbors: int
    eval_mcts_samples: list[int]

    model_training: bool

    def __init__(self, config_fn: str, default_config_fn: str = "config/config.json", trial_num: str = ""):
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
        self.path = config_fn

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

        if trial_num != "":
            self.output_dir = self.output_dir.rstrip("/") + str(trial_num) + "/"

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
        self.stats_dir = self.output_dir + "stats/"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.dataset_out_dir = self.output_dir + "datasets/"
        os.makedirs(self.dataset_out_dir, exist_ok=True)

        os.makedirs(self.input_dir, exist_ok=True)
        self.dataset_dir = self.input_dir + "datasets/"
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.solvedata_dir = self.input_dir + "solvedata/"
        os.makedirs(self.solvedata_dir, exist_ok=True)

config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.json"
trial_num = sys.argv[2] if len(sys.argv) > 2 else ""
config = Config(config_path, "config/config.json", trial_num)
