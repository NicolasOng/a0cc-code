import random
import multiprocessing
import concurrent.futures
from concurrent.futures import Future, wait, FIRST_COMPLETED
import os

from cc.core import Board, Game, Player
from cc.ground_truth import GroundTruth
from a0.dataset import Dataset
from a0.model_utils import board_to_input, Policy, get_legal_move_mask_from_state
from a0.eval.training_data import game_data_generator
from a0.train.dataset import train_model_epochs, plot_model_performance
from a0.model import AlphaZeroModel, load_model
from a0.eval.dataset_evaluation import evaluate_model, policy_accuracy_function
from a0.graph_search.mcts import MCTS
from a0.mcts.gt import MCTS_GT
from a0.players.a0 import A0Player

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from flax import nnx
import jax

import dill

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def get_n_random_states(n: int, remove_trivial: bool = True, remove_terminal: bool = True) -> list[Board]:
    '''
    Returns a list of n random unique board states.
    '''
    gt = GroundTruth()
    game = Game(board_size=config.board_size, num_pieces=config.num_pieces)
    max_rank = gt.get_max_rank()
    state_set: set[Board] = set()
    while len(state_set) < n:
        rank = random.randint(0, max_rank - 1)
        board = gt.unrank(rank)
        if remove_trivial and gt.is_trivial(board):
            continue
        if remove_terminal and game.get_done(board):
            continue
        state_set.add(board)
    return list(state_set)

def create_gtd_from_states(boards: list[Board]) -> Dataset:
    '''
    Creates a ground truth dataset from a list of board states.
    This uses the 1ply prob dist over winning moves as the policy target.
    '''
    gt = GroundTruth()
    states: list[jnp.ndarray] = []
    values: list[float] = []
    policies: list[jnp.ndarray] = []
    masks: list[jnp.ndarray] = []
    for board in tqdm(boards):
        states.append(jnp.array(board_to_input(board))) # (1, board_size, board_size, 2)
        values.append(gt.get_outcome(board)) # float
        policies.append(jnp.array(gt.get_1ply_policy_prob_dist_list(board, for_model=True))) # (board_size ** 4,)
        masks.append(jnp.array(get_legal_move_mask_from_state(board, for_model=True), dtype=jnp.float32)) # (board_size ** 4,)
    # load all this into a Dataset object
    logger.info("Creating Dataset object with ground truth values...")
    jnp_states = jnp.concatenate(states, axis=0) # (N, board_size, board_size, 2)
    jnp_values = jnp.array(values).reshape(-1, 1)  # (N, 1)
    jnp_policies = jnp.stack(policies) # (N, board_size ** 4)
    jnp_masks = jnp.stack(masks) # (N, board_size ** 4)
    logger.info(f"states.shape: {jnp_states.shape}, values.shape: {jnp_values.shape}, policies.shape: {jnp_policies.shape}, masks.shape: {jnp_masks.shape}")
    gtv_dataset = Dataset(batch_size=256)
    gtv_dataset.set(jnp_states, jnp_values, jnp_policies, jnp_masks)
    return gtv_dataset

def create_random_gtd_from_states(boards: list[Board]) -> Dataset:
    '''
    Creates a ground truth dataset from n random unique board states.
    This chooses a random winning move as the policy target.
    '''
    gt = GroundTruth()
    states: list[jnp.ndarray] = []
    values: list[float] = []
    policies: list[jnp.ndarray] = []
    gt = GroundTruth()
    for board in tqdm(boards):
        states.append(jnp.array(board_to_input(board))) # (1, board_size, board_size, 2)
        values.append(gt.get_outcome(board)) # float
        policies.append(jnp.array(gt.get_random_best_move_prob_dist_list(board, for_model=True))) # (board_size ** 4,)
    # load all this into a Dataset object
    logger.info("Creating Dataset object with ground truth values...")
    jnp_states = jnp.concatenate(states, axis=0) # (N, board_size, board_size, 2)
    jnp_values = jnp.array(values).reshape(-1, 1)  # (N, 1)
    jnp_policies = jnp.stack(policies) # (N, board_size ** 4)
    logger.info(f"states.shape: {jnp_states.shape}, values.shape: {jnp_values.shape}, policies.shape: {jnp_policies.shape}")
    gtv_dataset = Dataset(batch_size=256)
    gtv_dataset.set(jnp_states, jnp_values, jnp_policies)
    return gtv_dataset

def create_random_dataset_from_states(boards: list[Board]) -> Dataset:
    '''
    Creates a dataset from n random unique board states.
    This chooses a random valid move as the policy target.
    '''
    gt = GroundTruth()
    states: list[jnp.ndarray] = []
    values: list[float] = []
    policies: list[jnp.ndarray] = []
    gt = GroundTruth()
    for board in tqdm(boards):
        states.append(jnp.array(board_to_input(board))) # (1, board_size, board_size, 2)
        values.append(random.choice([0, 1])) # float
        policies.append(jnp.array(gt.get_random_valid_move_prob_dist_list(board, for_model=True))) # (board_size ** 4,)
    # load all this into a Dataset object
    logger.info("Creating Dataset object with ground truth values...")
    jnp_states = jnp.concatenate(states, axis=0) # (N, board_size, board_size, 2)
    jnp_values = jnp.array(values).reshape(-1, 1)  # (N, 1)
    jnp_policies = jnp.stack(policies) # (N, board_size ** 4)
    logger.info(f"states.shape: {jnp_states.shape}, values.shape: {jnp_values.shape}, policies.shape: {jnp_policies.shape}")
    gtv_dataset = Dataset(batch_size=256)
    gtv_dataset.set(jnp_states, jnp_values, jnp_policies)
    return gtv_dataset

def create_dataset_from_selfplay(remove_trivial: bool = True) -> tuple[Dataset, list[Board], list[Board]]:
    # get the unique boards (with their outcome/policy) from the training data
    gt = GroundTruth()
    total_boards = 0
    acc_count = 0
    random_policy_acc = 0
    value_acc_count = 0
    value_total_boards = 0
    states: list[jnp.ndarray] = []
    values: list[float] = []
    policies: list[jnp.ndarray] = []
    masks: list[jnp.ndarray] = []
    boards: list[Board] = []
    logger.info("Getting unique boards from training data, with their experienced outcome...")
    game_data_lists = game_data_generator(config.training_dir, config.training_iterations)
    for _, game_data_list in tqdm(game_data_lists):
        for game_data in game_data_list:
            game_winner = game_data.winner
            turn_data = game_data.turn_data
            for turn in turn_data:
                # get the board and its experienced outcome (current player perspective)
                board = turn.board
                experienced_board = jnp.array(board_to_input(board)) # (1, board_size, board_size, 2)
                experienced_outcome = 0.0 if game_winner is None else 1.0 if game_winner == board.current_player else -1.0
                experienced_policy = turn.player_data

                # for value acc, ignore draws
                if experienced_outcome != 0.0:
                    gt_value = gt.get_outcome(board)
                    value_total_boards += 1
                    if experienced_outcome == gt_value:
                        value_acc_count += 1
                
                if remove_trivial and gt.is_trivial(board):
                    continue

                gt_policy = gt.get_1ply_policy_prob_dist_list(board, for_model=True)
                gt_random_policy = gt.get_random_valid_move_prob_dist_list(board, for_model=True)
                acc = policy_accuracy_function(np.array(experienced_policy), np.array(gt_policy))
                random_acc = policy_accuracy_function(np.array(gt_random_policy), np.array(gt_policy))
                total_boards += 1
                if acc:
                    acc_count += 1
                if random_acc:
                    random_policy_acc += 1
                
                mask = jnp.array(get_legal_move_mask_from_state(board, for_model=True)) # (board_size ** 4,)

                states.append(experienced_board)
                values.append(experienced_outcome)
                policies.append(experienced_policy)
                masks.append(mask)
                boards.append(board)
            
    # load all this into a Dataset object
    jnp_states = jnp.concatenate(states, axis=0) # (N, board_size, board_size, 2)
    jnp_values = jnp.array(values).reshape(-1, 1)  # (N, 1)
    jnp_policies = jnp.stack(policies) # (N, board_size ** 4)
    jnp_masks = jnp.stack(masks) # (N, board_size ** 4)
    logger.info(f"states.shape: {jnp_states.shape}, values.shape: {jnp_values.shape}, policies.shape: {jnp_policies.shape}, masks.shape: {jnp_masks.shape}")
    sp_dataset = Dataset(batch_size=256)
    sp_dataset.set(jnp_states, jnp_values, jnp_policies, jnp_masks)

    unique_boards = list(set(boards))

    logger.log(25, f"Policy accuracy against ground truth on self-play data (NT Boards: {remove_trivial}): {acc_count / total_boards if total_boards > 0 else 0.0:.2%} ({acc_count} / {total_boards})")
    logger.log(25, f"Random Policy accuracy against ground truth on self-play data (NT Boards: {remove_trivial}): {random_policy_acc / total_boards if total_boards > 0 else 0.0:.2%} ({random_policy_acc} / {total_boards})")
    logger.log(25, f"Value accuracy against ground truth on self-play data (ND Boards): {value_acc_count / value_total_boards if value_total_boards > 0 else 0.0:.2%} ({value_acc_count} / {value_total_boards})")
    logger.log(25, f"Total unique boards from self-play data (NT Boards: {remove_trivial}): {len(unique_boards) / len(boards) if len(boards) > 0 else 0.0:.2%} ({len(unique_boards)} / {len(boards)})")

    return sp_dataset, unique_boards, boards

def check_random_policy_acc(boards: list[Board]) -> float:
    '''
    Checks the policy accuracy of a random policy on the given boards.
    '''
    gt = GroundTruth()
    correct = 0
    total = 0
    for board in tqdm(boards):
        true_policy = gt.get_1ply_policy_prob_dist_list(board, for_model=False)
        random_policy = gt.get_random_valid_move_prob_dist_list(board, for_model=False)
        acc: bool = policy_accuracy_function(np.array(random_policy), np.array(true_policy))
        if acc:
            correct += 1
        total += 1
    logger.info(f"Random policy accuracy: {correct / total if total > 0 else 0.0:.2%} ({correct} / {total})")
    return correct / total if total > 0 else 0.0

def train_and_plot(fn: str, dataset: Dataset, eval_dataset: Dataset, num_epochs: int = 10) -> AlphaZeroModel:
    '''
    Trains a model on the given dataset for num_epochs epochs,
    then evaluates it on the eval_dataset.
    Plots the training performance.
    The plots assume that the given eval_dataset is the GTD.
    '''
    logger.info(f"Training model '{fn}' for {num_epochs} epochs...")

    model = AlphaZeroModel(
        board_size=config.board_size,
        num_filters=256,
        training=True,
        rngs=nnx.Rngs({'params': jax.random.PRNGKey(1)})
    )

    test_dataset = dataset.split_off_test(len(dataset) // 10, shuffle=True)
    eval_dataset_split = eval_dataset.split_off_test(len(eval_dataset) // 10, shuffle=True)

    datasets = {
        "test": test_dataset,
        "gt": eval_dataset_split
    }

    trained_model, dsd = train_model_epochs(model, dataset, num_epochs, save="None", plot=False, test_datasets=datasets)
    plot_model_performance(fn, [dsd])

    loss, value_loss, policy_loss, value_accuracy, policy_accuracy = evaluate_model(trained_model, eval_dataset)
    logger.info(f"For model '{fn}':")
    logger.info(f"Evaluation on eval dataset - Loss: {loss}, Value Loss: {value_loss}, Policy Loss: {policy_loss}, Value Accuracy: {value_accuracy}, Policy Accuracy: {policy_accuracy}")

    return trained_model


def train_and_plot_datasets(fn: str, dataset: Dataset, eval_datasets: dict[str, Dataset], num_epochs: int = 10, res_blocks: int = 3) -> AlphaZeroModel:
    '''
    Trains a model on the given dataset for num_epochs epochs,
    then evaluates it on the eval_dataset.
    Plots the training performance.
    '''
    logger.info(f"Training model '{fn}' for {num_epochs} epochs...")

    model = AlphaZeroModel(
        board_size=config.board_size,
        num_filters=256,
        training=True,
        rngs=nnx.Rngs({'params': jax.random.PRNGKey(1)}),
        num_resblocks=res_blocks
    )

    trained_model, dsd = train_model_epochs(model, dataset, num_epochs, save="None", plot=False, test_datasets=eval_datasets)
    plot_model_performance(fn, [dsd])

    return trained_model

def generate_mcts_policy(state: Board, game: Game, error_rate: float, mcts_iterations: int) -> tuple[NDArray[np.float32], float]:
    '''
    Generates a policy using MCTS with ground truth evaluations.
    Returns the policy distribution and the root reward.
    '''
    moves = game.generate_moves_for_given_board(state)

    mcts = MCTS(
        problem=MCTS_GT(state, game, error_rate),
        selection_policy="uct"
        )
    mcts.run(iterations=mcts_iterations)
    children = mcts.get_root_children()

    if len(children) == 0:
        print("MCTS generated no children!")
        print(state.board_view())
        print(state.current_player)
        moves = game.generate_moves_for_given_board(state)
        print("moves:")
        for move in moves:
            print(move)

    assert len(children) > 0, "No children found in MCTS root node."

    # with the root's children, create a policy distribution logits
    mcts_root_children_visit_counts = [float(child.visits) for child in children]
    mcts_root_children_moves = [state.child_board_to_move(child.state) for child in children]

    p = Policy(len(state.board))
    # create a well-shaped policy distribution,
    p.set_logits_from_moves(mcts_root_children_moves, mcts_root_children_visit_counts, rotate_180=False)
    # mask non-legal moves,
    p.set_legal_moves(moves)
    p.apply_mask(0.0)
    # softmax it to get the policy distribution
    p.apply_power_normalize(1)
    # we rotate the policy if the current player is O,
    # since this is for training the model
    if state.current_player == Player.PLAYER_O:
        p.rotate_policy()
    return p.policy, mcts.root.reward

def generate_mcts_dataset(boards: list[Board], game: Game, error_rate: float, mcts_iterations: int) -> Dataset:
    '''
    Generates a dataset using MCTS with ground truth evaluations on the given boards.
    '''
    gt = GroundTruth()

    states: list[jnp.ndarray] = []
    values: list[float] = []
    policies: list[jnp.ndarray] = []

    total_boards = 0
    acc_count = 0
    for board in tqdm(boards):
        policy, value = generate_mcts_policy(board, game, error_rate, mcts_iterations)
        states.append(jnp.array(board_to_input(board))) # (1, board_size, board_size, 2)
        values.append(value) # float
        policies.append(jnp.array(policy)) # (board_size ** 4,)

        gt_policy = gt.get_1ply_policy_prob_dist_list(board, for_model=True)
        acc = policy_accuracy_function(policy, np.array(gt_policy))
        total_boards += 1
        if acc:
            acc_count += 1

    logger.log(25, f"MCTS dataset generation complete for error rate {error_rate} and {mcts_iterations} iterations.")
    logger.log(25, f"MCTS Policy accuracy against ground truth on generated data (NT Boards): {acc_count / total_boards if total_boards > 0 else 0.0:.2%} ({acc_count} / {total_boards})")

    # load all this into a Dataset object
    logger.info("Creating Dataset object with MCTS-generated values...")
    jnp_states = jnp.concatenate(states, axis=0) # (N, board_size, board_size, 2)
    jnp_values = jnp.array(values).reshape(-1, 1)  # (N, 1)
    jnp_policies = jnp.stack(policies) # (N, board_size ** 4)
    logger.info(f"states.shape: {jnp_states.shape}, values.shape: {jnp_values.shape}, policies.shape: {jnp_policies.shape}")
    mcts_dataset = Dataset(batch_size=256)
    mcts_dataset.set(jnp_states, jnp_values, jnp_policies)
    return mcts_dataset

def mcts_function(error_rate: float, mcts_iterations: int, validation_dataset: Dataset):
    '''
    Generates a dataset using MCTS with ground truth evaluations,
    then trains/tests a model on it with a 90/10 split.
    Uses a validation dataset for evaluation after training.
    '''
    # get n random unique board states
    n = 10000
    logger.info(f"Generating {n} random unique board states...")
    boards = get_n_random_states(n, remove_trivial=True)

    # create a ground truth dataset from these states
    g = Game(board_size=config.board_size, num_pieces=config.num_pieces)
    mcts_dataset = generate_mcts_dataset(boards, g, error_rate, mcts_iterations)
    logger.info("MCTS dataset created.")

    # train a model on the mcts dataset + evaluate
    train_and_plot(f"sl_on_policy_head_mcts_er{error_rate}_it{mcts_iterations}", mcts_dataset, validation_dataset, num_epochs=10)

def generate_nn_dataset(boards: list[Board], player: A0Player, mcts_type: str = "NN", error_rate: float = 0.2, selection: str = "uct") -> Dataset:
    '''
    Generates a dataset using the given A0Player on the given boards.
    '''
    gt = GroundTruth()

    states: list[jnp.ndarray] = []
    values: list[float] = []
    policies: list[jnp.ndarray] = []

    total_boards = 0
    acc_count = 0
    for board in tqdm(boards):
        legal_moves = gt.cc.generate_moves_for_given_board(board)
        value, policy = player.get_value_and_policy(
            board,
            legal_moves,
            mcts_type=mcts_type,
            error_rate=error_rate,
            mcts_key=selection
        )

        states.append(jnp.array(board_to_input(board))) # (1, board_size, board_size, 2)
        values.append(value) # (1, 1)
        policies.append(jnp.array(policy)) # (1, board_size ** 4)

        gt_policy = gt.get_1ply_policy_prob_dist_list(board, for_model=True)
        acc = policy_accuracy_function(policy[0], np.array(gt_policy))
        total_boards += 1
        if acc:
            acc_count += 1

    logger.info(f"NN dataset generation complete.")
    logger.log(25, f"NN Policy accuracy against ground truth on generated data (NT Boards): {acc_count / total_boards if total_boards > 0 else 0.0:.2%} ({acc_count} / {total_boards})")

    # load all this into a Dataset object
    logger.info("Creating Dataset object with NN-generated values...")
    jnp_states = jnp.concatenate(states, axis=0) # (N, board_size, board_size, 2)
    jnp_values = jnp.concatenate(values, axis=0)  # (N, 1)
    jnp_policies = jnp.concatenate(policies, axis=0) # (N, board_size ** 4)
    logger.info(f"states.shape: {jnp_states.shape}, values.shape: {jnp_values.shape}, policies.shape: {jnp_policies.shape}")
    nn_dataset = Dataset(batch_size=256)
    nn_dataset.set(jnp_states, jnp_values, jnp_policies)
    return nn_dataset

def _generate_sample(serialized_player: bytes, game: Game, board: Board, mcts_type: str, error_rate: float, selection: str) -> tuple[Board, NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    '''
    Uses the given player to generate a value/policy sample for the given state.
    '''
    # first, set up logging (since this is a separate process)
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="sl_on_policy_head"
    )
    # convert the board
    state = np.array(board_to_input(board))
    # then generate the sample
    player: A0Player = dill.loads(serialized_player)
    legal_moves = game.generate_moves_for_given_board(board)
    value, policy = player.get_value_and_policy(
        board,
        legal_moves,
        mcts_type=mcts_type,
        error_rate=error_rate,
        mcts_key=selection
    )
    return board, state, value, policy

def generate_nn_dataset_parallel(boards: list[Board], player: A0Player, mcts_type: str = "NN", error_rate: float = 0.2, selection: str = "uct") -> Dataset:
    '''
    Generates a dataset using the given A0Player on the given boards.
    '''
    gt = GroundTruth()
    gt_game = gt.cc

    # serialize the player model (JAX models cannot be pickled directly + serialization is needed for multiprocessing)
    player_serialized: bytes = dill.dumps(player)

    states: list[jnp.ndarray] = []
    values: list[float] = []
    policies: list[jnp.ndarray] = []

    total_boards = 0
    acc_count = 0

    num_cores = os.cpu_count() or 4
    logger.info(f"Using {num_cores} cores for mcts sample generation.")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # create a list to hold the futures
        futures: list[Future[tuple[Board, NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]]] = []

        # create a function to start a process that generates a sample
        def spawn_gen_sample_process(board: Board) -> None:
            future = executor.submit(
                _generate_sample,
                serialized_player=player_serialized,
                game=gt_game,
                board=board,
                mcts_type=mcts_type,
                error_rate=error_rate,
                selection=selection
            )
            futures.append(future)
        
        # start a game for each board
        for board in boards:
            spawn_gen_sample_process(board)

        while True:
            # when a game (or games) finish(es),
            done, _ = wait(futures, return_when=FIRST_COMPLETED)

            # for each finished sample,
            for future in done:
                # remove it from the list of futures
                futures.remove(future)
                # get the value and policy for the board
                board, state, value, policy = future.result()
                # add these to the dataset lists
                states.append(jnp.array(state)) # (1, board_size, board_size, 2)
                values.append(jnp.array(value)) # (1, 1)
                policies.append(jnp.array(policy)) # (1, board_size ** 4)
                # check the accuracy
                gt_policy = gt.get_1ply_policy_prob_dist_list(board, for_model=True)
                acc = policy_accuracy_function(policy[0], np.array(gt_policy))
                total_boards += 1
                if acc:
                    acc_count += 1
            
            # stop when all samples are generated
            if not futures:
                break

    logger.info(f"NN dataset generation complete.")
    logger.log(25, f"mcts_type: {mcts_type}, error_rate: {error_rate}")
    logger.log(25, f"NN Policy accuracy against ground truth on generated data (NT Boards): {acc_count / total_boards if total_boards > 0 else 0.0:.2%} ({acc_count} / {total_boards})")

    # load all this into a Dataset object
    logger.info("Creating Dataset object with NN-generated values...")
    jnp_states = jnp.concatenate(states, axis=0) # (N, board_size, board_size, 2)
    jnp_values = jnp.concatenate(values, axis=0)  # (N, 1)
    jnp_policies = jnp.concatenate(policies, axis=0) # (N, board_size ** 4)
    logger.info(f"states.shape: {jnp_states.shape}, values.shape: {jnp_values.shape}, policies.shape: {jnp_policies.shape}")
    nn_dataset = Dataset(batch_size=256)
    nn_dataset.set(jnp_states, jnp_values, jnp_policies)
    return nn_dataset

def train_model_on_selfplay_states_and_mcts(fn: str, model_name: str, mcts_type: str = "NN", error_rate: float = 0.2):
    '''
    Gets the states from existing self-play data,
    then generates a dataset using MCTS (either with a trained model or ground truth model),
    then trains/tests a model on it with a 90/10 split.
    Uses a validation dataset and seen dataset for evaluation after training.
    '''
    trained_model = load_model(config.training_dir + model_name)
    player = A0Player(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        model=trained_model,
        exploit=True,
        mcts_samples=64,
        no_reverse_moves=True,
        no_side_moves=False
    )

    # create a dataset from self-play data.
    _, sp_boards, _ = create_dataset_from_selfplay(remove_trivial=True)
    #sp_boards = get_n_random_states(10, remove_trivial=True, remove_terminal=True)

    # create the nn dataset from the self-play boards with the trained model
    nn_dataset = generate_nn_dataset(sp_boards, player, mcts_type=mcts_type, error_rate=error_rate)
    # train/test split
    nn_dataset_test = nn_dataset.split_off_test(len(nn_dataset) // 10, shuffle=True)

    # create a ground truth dataset with random states for validation
    gtv_dataset = create_gtd_from_states(get_n_random_states(10000, remove_trivial=True))
    logger.info("Validation dataset created.")

    # create a seen gt dataset from the sp boards
    seen_gt_dataset = create_gtd_from_states(sp_boards)

    # train a model on the nn dataset + plot metrics
    train_and_plot_datasets(
        fn,
        nn_dataset,
        {
            "nn_test": nn_dataset_test,
            "seen_gt": seen_gt_dataset,
            "random_gt": gtv_dataset
        },
        num_epochs=10
    )

def train_model_on_random_states_and_mcts(fn: str, n: int, model_name: str, mcts_type: str = "NN", error_rate: float = 0.2, mcts_key: str = "puct", rollout_type: str = "none", policy_type: str = "policy"):
    '''
    Gets random states,
    then generates a dataset using MCTS (either with a trained model or ground truth model),
    then trains/tests a model on it with a 90/10 split.
    Uses a validation dataset and seen dataset for evaluation after training.
    '''
    logger.log(25, f"{fn}:")
    logger.log(25, f"n: {n}, model_name: {model_name}, mcts_type: {mcts_type}, error_rate: {error_rate}, mcts_key: {mcts_key}, rollout_type: {rollout_type}, policy_type: {policy_type}")
    trained_model = load_model(config.training_dir + model_name)
    player = A0Player(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        model=trained_model,
        exploit=True,
        mcts_samples=64,
        no_reverse_moves=True,
        no_side_moves=False,
        rollout_type=rollout_type,
        rollout_depth=20,
        policy_type=policy_type
    )

    # create a dataset from random data.
    random_boards = get_n_random_states(n, remove_trivial=True, remove_terminal=True)

    # create the nn dataset from the random boards with the trained model
    nn_dataset = generate_nn_dataset_parallel(random_boards, player, mcts_type=mcts_type, error_rate=error_rate, selection=mcts_key)
    # train/test split
    nn_dataset_test = nn_dataset.split_off_test(len(nn_dataset) // 10, shuffle=True)

    # create a ground truth dataset with random states for validation
    gtv_dataset = create_gtd_from_states(get_n_random_states(10000, remove_trivial=True))
    logger.info("Validation dataset created.")

    # create a seen gt dataset from the sp boards
    seen_gt_dataset = create_gtd_from_states(random_boards)

    # train a model on the nn dataset + plot metrics
    train_and_plot_datasets(
        fn,
        nn_dataset,
        {
            "nn_test": nn_dataset_test,
            "seen_gt": seen_gt_dataset,
            "random_gt": gtv_dataset
        },
        num_epochs=10
    )

def main():
    '''
    Generates the following datasets, then trains/tests models on them with a 90/10 split:
    - Ground truth dataset from n random unique board states
    - Random ground truth dataset from n random unique board states
    - Random dataset from n random unique board states
    - Self-play dataset from existing self-play data
    Uses a validation dataset for evaluation of all models after training.
    Additionally, evaluates the self-play trained model on both the self-play dataset and its ground truth counterpart.
    This is to see if the model can learn the ground truth dataset and the random ground truth dataset.
    Also the self-play dataset, and if learning from self play helps generalization to ground truth.
    Random dataset serves as an interesting comparison.
    '''
    # TODO: do with larger n
    # create a dataset from self-play data
    sp_dataset, sp_boards, _ = create_dataset_from_selfplay(remove_trivial=True)
    logger.info("Self-play dataset created.")

    # create a ground truth dataset from the self-play boards
    spgtv_dataset = create_gtd_from_states(sp_boards)
    logger.info("selfplay Ground truth dataset created.")

    sp_n = len(sp_dataset)
    logger.info(f"Self-play dataset size: {sp_n}")
    # get n random unique board states
    n = sp_n
    logger.info(f"Generating {n} random unique board states...")
    boards = get_n_random_states(n, remove_trivial=True)

    # create a ground truth dataset from these states
    gtv_dataset = create_gtd_from_states(boards)
    logger.info("Ground truth dataset created.")

    # create a random ground truth dataset from these states
    random_gtv_dataset = create_random_gtd_from_states(boards)
    logger.info("Random ground truth dataset created.")

    # create a random dataset from these states
    random_dataset = create_random_dataset_from_states(boards)
    logger.info("Random dataset created.")

    # create a ground truth dataset with different states for validation
    gtv_dataset_validation = create_gtd_from_states(get_n_random_states(10000, remove_trivial=True))
    logger.info("Validation dataset created.")

    # check the random policy accuracy on the gtv boards
    check_random_policy_acc(boards)

    # train a model on the gtv dataset + evaluate
    train_and_plot("sl_on_policy_head_gtv", gtv_dataset, gtv_dataset_validation, num_epochs=10)

    # train a model on the random gtv dataset + evaluate
    train_and_plot("sl_on_policy_head_random_gtv", random_gtv_dataset, gtv_dataset_validation, num_epochs=10)

    # train a model on the random dataset + evaluate
    train_and_plot("sl_on_policy_head_random", random_dataset, gtv_dataset_validation, num_epochs=10)

    # train a model on the self-play dataset + evaluate
    m = train_and_plot("sl_on_policy_head_selfplay", sp_dataset, gtv_dataset_validation, num_epochs=10)
    # also eval on itself
    loss, value_loss, policy_loss, value_accuracy, policy_accuracy = evaluate_model(m, sp_dataset)
    logger.info(f"Evaluation on sp dataset - Loss: {loss}, Value Loss: {value_loss}, Policy Loss: {policy_loss}, Value Accuracy: {value_accuracy}, Policy Accuracy: {policy_accuracy}")
    loss, value_loss, policy_loss, value_accuracy, policy_accuracy = evaluate_model(m, spgtv_dataset)
    logger.info(f"Evaluation on spgtv dataset - Loss: {loss}, Value Loss: {value_loss}, Policy Loss: {policy_loss}, Value Accuracy: {value_accuracy}, Policy Accuracy: {policy_accuracy}")

def main2():
    '''
    Generates datasets using MCTS with ground truth evaluations,
    then trains/tests models on them with a 90/10 split.
    Uses a validation dataset for evaluation of all models after training.
    Tests different error rates and MCTS sample sizes.
    '''
    # create a ground truth dataset with different states for validation
    gtv_dataset_validation = create_gtd_from_states(get_n_random_states(10000, remove_trivial=True))
    logger.info("Validation dataset created.")
    
    mcts_samples_list = [64, 512]
    errors_rate_list = [0.0, 0.2, 0.4, 0.5]

    for mcts_samples in mcts_samples_list:
        for error_rate in errors_rate_list:
            mcts_function(error_rate, mcts_samples, gtv_dataset_validation)


def main3(n: int):
    '''
    Generates the following datasets, then trains/tests models on them with a 90/10 split:
    - Ground truth dataset from n random unique board states
    - Random ground truth dataset from n random unique board states
    Uses a validation dataset for evaluation of all models after training.
    n sets how many random unique board states to generate.
    This is to mainly see how well the model can learn from n ground truth states.
    '''
    # get n random unique board states
    logger.info(f"Generating {n} random unique board states...")
    boards = get_n_random_states(n, remove_trivial=True)

    # create a ground truth dataset from these states
    gtv_dataset = create_gtd_from_states(boards)
    logger.info("Ground truth dataset created.")

    # create a random ground truth dataset from these states
    random_gtv_dataset = create_random_gtd_from_states(boards)
    logger.info("Random ground truth dataset created.")

    # create a ground truth dataset with different states for validation
    gtv_dataset_validation = create_gtd_from_states(get_n_random_states(10000, remove_trivial=True))
    logger.info("Validation dataset created.")

    # train a model on the gtv dataset + evaluate
    train_and_plot("sl_on_policy_head_gtv", gtv_dataset, gtv_dataset_validation, num_epochs=10)

    # train a model on the random gtv dataset + evaluate
    train_and_plot("sl_on_policy_head_random_gtv", random_gtv_dataset, gtv_dataset_validation, num_epochs=10)

def main4():
    '''
    Runs two experiments:
    - trains a model on self-play data with multiple evaluations (self-play test, seen ground truth, random ground truth)
    - trains a model on self-play ground truth data with multiple evaluations (self-play ground truth test, random ground truth)
    Each for 10 epochs.
    I suspect that the first experiment will do worse, while the second will do better.
    Then I just have to figure out the differences between the two.
    '''
    # create a dataset from self-play data.
    sp_dataset_train, sp_boards, _ = create_dataset_from_selfplay(remove_trivial=True)
    # do train/test split
    sp_dataset_test = sp_dataset_train.split_off_test(len(sp_dataset_train) // 10, shuffle=True)
    logger.info("Self-play dataset created.")
    # create a ground truth dataset from the self-play boards
    sp_gtv_dataset = create_gtd_from_states(sp_boards)
    logger.info("Ground truth dataset created.")

    # create a ground truth dataset with random states for validation
    gtv_dataset = create_gtd_from_states(get_n_random_states(10000, remove_trivial=True))
    logger.info("Validation dataset created.")

    # train a model on the self-play dataset + plot metrics
    train_and_plot_datasets(
        "sl_on_policy_head_selfplay_multiple_eval",
        sp_dataset_train,
        {
            "selfplay_test": sp_dataset_test,
            "seen_gt": sp_gtv_dataset,
            "random_gt": gtv_dataset
        },
        num_epochs=10
    )

    # train a model on the sp ground truth dataset + plot metrics
    sp_gtv_dataset_test = sp_gtv_dataset.split_off_test(len(sp_gtv_dataset) // 10, shuffle=True)
    train_and_plot_datasets(
        "sl_on_policy_head_selfplay_gtv_multiple_eval",
        sp_gtv_dataset,
        {
            "selfplay_gtv_test": sp_gtv_dataset_test,
            "random_gt": gtv_dataset
        },
        num_epochs=10
    )

def main5():
    '''
    Re-runs the following experiment:
    - trains a model on self-play ground truth data with multiple evaluations (self-play ground truth test, random ground truth)
    Except this time, keeps duplicate boards from the self-play data.
    '''
    # create a dataset from self-play data.
    _, _, sp_boards_duped = create_dataset_from_selfplay(remove_trivial=True)
    # create a ground truth dataset from the self-play boards
    sp_gtv_dataset = create_gtd_from_states(sp_boards_duped)
    logger.info("Ground truth dataset created.")

    # create a ground truth dataset with random states for validation
    gtv_dataset = create_gtd_from_states(get_n_random_states(10000, remove_trivial=True))
    logger.info("Validation dataset created.")

    # train a model on the sp ground truth dataset + plot metrics
    sp_gtv_dataset_test = sp_gtv_dataset.split_off_test(len(sp_gtv_dataset) // 10, shuffle=True)
    train_and_plot_datasets(
        "sl_on_policy_head_selfplay_gtv_multiple_eval",
        sp_gtv_dataset,
        {
            "selfplay_gtv_test": sp_gtv_dataset_test,
            "random_gt": gtv_dataset
        },
        num_epochs=10
    )

def main6():
    '''
    Performs the following:
    - generates policy/value targets with MCTS using the GT+err model on self-play boards
    - generates policy/value targets with MCTS using the trained model (~85% value acc) on self-play boards
    - trains a model on both to see how well they do
    I expect the GT+err MCTS to do better. Then it'll be confirmed that MCTS/model is the issue for some reason.
    '''
    mno = 450
    trained_model = load_model(config.training_dir + f"model_{mno}.pkl")
    player = A0Player(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        model=trained_model,
        exploit=True,
        mcts_samples=64,
        no_reverse_moves=True,
        no_side_moves=False
    )

    # create a dataset from self-play data.
    _, sp_boards, _ = create_dataset_from_selfplay(remove_trivial=True)
    # sp_boards = get_n_random_states(100, remove_trivial=True, remove_terminal=True)
    
    # create the mcts dataset from the self-play boards with the gt+err model
    g = Game(board_size=config.board_size, num_pieces=config.num_pieces)
    mcts_dataset = generate_mcts_dataset(sp_boards, g, 0.2, 64)
    # train/test split
    mcts_dataset_test = mcts_dataset.split_off_test(len(mcts_dataset) // 10, shuffle=True)

    # create the nn dataset from the self-play boards with the trained model
    nn_dataset = generate_nn_dataset(sp_boards, player)
    # train/test split
    nn_dataset_test = nn_dataset.split_off_test(len(nn_dataset) // 10, shuffle=True)

    # create a ground truth dataset with random states for validation
    gtv_dataset = create_gtd_from_states(get_n_random_states(10000, remove_trivial=True))
    logger.info("Validation dataset created.")

    # create a seen gt dataset from the sp boards
    seen_gt_dataset = create_gtd_from_states(sp_boards)

    # train a model on the mcts dataset + plot metrics
    train_and_plot_datasets(
        "sl_with_mcts",
        mcts_dataset,
        {
            "mcts_test": mcts_dataset_test,
            "seen_gt": seen_gt_dataset,
            "random_gt": gtv_dataset
        },
        num_epochs=10
    )

    # train a model on the nn dataset + plot metrics
    train_and_plot_datasets(
        "sl_with_nn",
        nn_dataset,
        {
            "nn_test": nn_dataset_test,
            "seen_gt": seen_gt_dataset,
            "random_gt": gtv_dataset
        },
        num_epochs=10
    )

def main7():
    '''
    Same as main6, but without the MCTS dataset.
    '''
    mno = 450
    trained_model = load_model(config.training_dir + f"model_{mno}.pkl")
    player = A0Player(
        board_size=config.board_size,
        num_pieces=config.num_pieces,
        model=trained_model,
        exploit=True,
        mcts_samples=64,
        no_reverse_moves=True,
        no_side_moves=False
    )

    # create a dataset from self-play data.
    _, sp_boards, _ = create_dataset_from_selfplay(remove_trivial=True)
    #sp_boards = get_n_random_states(10, remove_trivial=True, remove_terminal=True)

    # create the nn dataset from the self-play boards with the trained model
    nn_dataset = generate_nn_dataset(sp_boards, player)
    # train/test split
    nn_dataset_test = nn_dataset.split_off_test(len(nn_dataset) // 10, shuffle=True)

    # create a ground truth dataset with random states for validation
    gtv_dataset = create_gtd_from_states(get_n_random_states(10000, remove_trivial=True))
    logger.info("Validation dataset created.")

    # create a seen gt dataset from the sp boards
    seen_gt_dataset = create_gtd_from_states(sp_boards)

    # train a model on the nn dataset + plot metrics
    train_and_plot_datasets(
        "sl_with_nn",
        nn_dataset,
        {
            "nn_test": nn_dataset_test,
            "seen_gt": seen_gt_dataset,
            "random_gt": gtv_dataset
        },
        num_epochs=10
    )

def main8():
    '''
    Four tests with different datasets:
    - selfplay states + a0 model mcts
    - selfplay states + gt 0.2 err mcts
    - random states + a0 model mcts
    - random states + gt 0.2 err mcts
    Each trained for 10 epochs.
    '''
    train_model_on_selfplay_states_and_mcts(
        "selfplay_states_w_a0_model_450_mcts",
        "model_450.pkl",
        mcts_type="NN"
    )

    train_model_on_selfplay_states_and_mcts(
        "selfplay_states_w_gt02err_mcts",
        "model_450.pkl",
        mcts_type="GT",
        error_rate=0.2
    )

    train_model_on_random_states_and_mcts(
        "random_states_50k_w_a0_model_450_mcts",
        50000,
        "model_450.pkl",
        mcts_type="NN"
    )

    train_model_on_random_states_and_mcts(
        "random_states_50k_w_gt02err_mcts",
        50000,
        "model_450.pkl",
        mcts_type="GT",
        error_rate=0.2
    )

def main9():
    '''
    Two tests with different datasets:
    - selfplay states + sl 80% acc model mcts
    - random states + sl 80% acc model mcts
    Each trained for 10 epochs.
    '''
    train_model_on_selfplay_states_and_mcts(
        "selfplay_states_w_sl_80acc_model_mcts",
        f"model_value_acc_{0.80:.2f}.pkl",
        mcts_type="NN"
    )

    train_model_on_random_states_and_mcts(
        "random_states_50k_w_sl_80acc_model_mcts",
        50000,
        f"model_value_acc_{0.80:.2f}.pkl",
        mcts_type="NN"
    )

def main10():
    '''
    Two tests:
    - random states + a0 model mcts + puct
    - random states + gt 0.2 err mcts + puct
    Each trained for 10 epochs.
    '''
    train_model_on_random_states_and_mcts(
        "random_states_50k_w_a0_model_450_mcts_puct",
        50000,
        "model_450.pkl",
        mcts_type="NN",
        mcts_key="puct"
    )

    train_model_on_random_states_and_mcts(
        "random_states_50k_w_gt02err_mcts_puct",
        50000,
        "model_450.pkl",
        mcts_type="GT",
        error_rate=0.2,
        mcts_key="puct"
    )

def main11():
    '''
    local tests
    '''
    train_model_on_random_states_and_mcts(
        "random_states_10_w_a0_model_450_mcts_puct",
        10,
        "model_450.pkl",
        mcts_type="NN",
        mcts_key="puct"
    )

def main12():
    '''
    Three tests with random states + a0 model mcts + puct
    - rollout type: random
    - rollout type: policy, policy_type: policy
    - rollout type: policy, policy_type: value
    Each trained for 10 epochs.
    '''
    train_model_on_random_states_and_mcts(
        "random_states_50k_w_a0_model_450_puct_random_rollout",
        50000,
        "model_450.pkl",
        mcts_type="NN",
        mcts_key="puct",
        rollout_type='random'
    )

    train_model_on_random_states_and_mcts(
        "random_states_50k_w_a0_model_450_puct_policy_rollout",
        50000,
        "model_450.pkl",
        mcts_type="NN",
        mcts_key="puct",
        rollout_type='policy',
        policy_type='policy'
    )

    train_model_on_random_states_and_mcts(
        "random_states_50k_w_a0_model_450_puct_value_rollout",
        50000,
        "model_450.pkl",
        mcts_type="NN",
        mcts_key="puct",
        rollout_type='policy',
        policy_type='value'
    )


if __name__ == "__main__":
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="sl_on_policy_head"
    )

    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    main12()
