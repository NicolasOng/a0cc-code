import os
import sys
import json


def _lazy_dir(suffix: str, base_attr: str = "output_dir") -> property:
    """Property factory: returns `<self.base_attr>/<suffix>/`, creating it on first access."""
    def getter(self: "Config") -> str:
        path: str = getattr(self, base_attr) + suffix
        os.makedirs(path, exist_ok=True)
        return path
    return property(getter)


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
    epsilon: float
    dirichlet_epsilon: float

    num_trials: int

    learning_rate: float
    weight_decay: float
    value_loss_weight: float
    grad_clip_norm: float | None  # global-norm gradient clipping; null/None = off

    num_filters: int     # conv filters throughout the model trunk
    num_resblocks: int   # residual blocks in the trunk
    value_head_zero_init: bool  # zero-init the final value layer so pre-tanh starts at 0
    persist_optimizer_state: bool  # keep Adam moments across training iterations instead of re-creating the optimizer

    eval_neighbors: int
    eval_mcts_samples: list[int]

    model_training: bool

    coprime_stepping_a: int
    coprime_stepping_b: int

    use_gt: bool

    num_workers: int
    self_play_batch_size: int
    use_inference_server: bool  # route self-play inference through one shared server process (vs a model per worker)
    inference_req_timeout: float  # client->server request put timeout (seconds); default 60
    inference_res_timeout: float  # server->client response get timeout (seconds); default 60

    illegal_moves: bool
    experiment: str
    td_lambda: float

    alternative_target: str  # "gt" | "gt_value" | "gt_next_value" | "td_0" | "td_lambda" | "interpolated_td_lambda" | "normal"
    dataset_balance_method: str  # "none" | "subsample_buckets" | "weighted_buckets"
    num_buckets_for_balance: int
    max_weight_ratio: float

    detect_collapse: bool
    collapse_detection_iteration: int  # check iterations 1..N (1-indexed)
    collapse_threshold_pre_tanh: float  # value_pre_tanh mean_abs above this → collapse
    max_collapse_retries: int

    log_system_metrics: bool
    system_metrics_interval_seconds: float

    c_puct: float

    do_player_eval: bool  # gates a0.eval.player
    do_gt_evals: bool     # gates eval3 analyses that need the solve-data file

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
        - num_workers: int, number of worker threads
        - self_play_batch_size: int, batch size for self-play
        '''
        self.path = config_fn
        self.trial_num = trial_num

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
            self.output_dir = self.output_dir.rstrip("/") + "/trial_" + str(trial_num) + "/"

        # NOTE: directories are NOT created here. Each *_dir attribute below is
        # a property that lazily mkdir's its path on first access. This avoids
        # creating empty directories that the current invocation never writes to
        # (e.g., a per-HP combine job doesn't need training/, validations/, etc.).

    # ----- lazy directory properties -----
    # Each *_dir creates its directory on first access (and any missing parent
    # dirs, since os.makedirs uses exist_ok=True). Subsequent reads are
    # essentially free — exist_ok=True makes makedirs a no-op when the
    # directory already exists.
    training_dir   = _lazy_dir("training/")
    log_dir        = _lazy_dir("logs/")
    plot_dir       = _lazy_dir("plots/")
    eval_dir       = _lazy_dir("eval/")
    validation_dir = _lazy_dir("validations/")
    stats_dir      = _lazy_dir("stats/")
    dataset_out_dir = _lazy_dir("datasets/")
    dataset_dir    = _lazy_dir("datasets/", base_attr="input_dir")
    solvedata_dir  = _lazy_dir("solvedata/", base_attr="input_dir")

config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.json"
trial_num = sys.argv[2] if len(sys.argv) > 2 else ""
config = Config(config_path, "config/config.json", trial_num)
