from typing import Any

from a0_new.policy import Policy
from a0_new.protocols.model import TrainableModel
from a0_new.protocols.game import Player
from a0_new.play import GameData
from a0_new.cc.model import A0CCModel

from a0.model import AlphaZeroModel, save_model
from a0.game import GameData as OldGameData, TurnData
from a0.model_utils import Policy as OldPolicy
from cc.core import Player as CCPlayer

from a0_new.utils.training_data import model_generator, gamedata_generator, datasetdata_generator

import jax
from flax import nnx
import numpy as np
from numpy.typing import NDArray

import pickle

from config import config

import os

def convert_policy(p: Policy[Any], rotate_180: bool) -> NDArray[np.float32]:
    op = OldPolicy(board_size=5)
    op.set_logits_from_moves([action.move for action in p.actions], p.action_values, rotate_180=rotate_180)
    return np.array(op.get_policy_list(), dtype=np.float32)

def convert_game_data(old_data: GameData[Any, Any]) -> OldGameData:
    # Create a new OldGameData instance
    new_data = OldGameData(game=old_data.game.game, turn_limit=old_data.turn_limit)

    # Convert each TurnData
    for turn in old_data.turn_data:
        player: CCPlayer = turn.state.board.current_player
        if player == CCPlayer.PLAYER_X:
            rotate_180 = False
        else:
            rotate_180 = True
            
        old_turn = TurnData(
            board=turn.state.board,
            moves=[action.move for action in turn.actions],
            move=turn.action.move,
            player_data=convert_policy(turn.policy, rotate_180=rotate_180)
        )
        new_data.turn_data.append(old_turn)
    
    # Copy other attributes
    new_data.ended = old_data.ended
    new_data.winner = None if old_data.winner is None else CCPlayer.PLAYER_X if old_data.winner == Player.X else CCPlayer.PLAYER_O
    new_data.time = old_data.time
    # note - should be final_state, but the current set has final_board.
    new_data.final_board = None if old_data.final_board is None else old_data.final_board.board
    
    return new_data

def convert_model(old_model: A0CCModel) -> AlphaZeroModel:
    # Extract parameters from the old model
    num_filters = old_model.conv.out_features
    num_resblocks = len(old_model.resblocks)
    # Infer board_size from value_head input features: board_size**2 * num_filters
    board_size = int((old_model.value_head.dense1.in_features / num_filters) ** 0.5)
    
    # Create a new AlphaZeroModel with the same parameters
    new_model = AlphaZeroModel(
        board_size=board_size,
        rngs=nnx.Rngs({'params': jax.random.PRNGKey(0)}),  # Dummy rngs, state will be overwritten
        num_filters=num_filters,
        num_resblocks=num_resblocks
    )
    
    # Split the old model to get its state
    _, state_old = nnx.split(old_model)
    
    # Get the graphdef from the new model
    graphdef, _ = nnx.split(new_model)
    
    # Merge the old state into the new model
    new_model = nnx.merge(graphdef, state_old)
    
    return new_model

def convert_gamedata(old_dir: str, new_dir: str, n: int) -> None:
    '''
    Converts gamedata files from the old format to the new format.
    Loads gamedata from old_dir, converts it, and saves it to new_dir.
    '''
    for iteration, gamedata in gamedata_generator(old_dir, n):
        # Convert gamedata to the new format (this is a placeholder, replace with actual conversion logic)
        converted_gamedata = [convert_game_data(data) for data in gamedata]
        
        # Save the converted gamedata to the new directory
        if new_dir:
            with open(new_dir + f"gamedata_{iteration}.pkl", 'wb') as f:
                pickle.dump(converted_gamedata, f)

def convert_models(old_dir: str, new_dir: str, n: int) -> None:
    '''
    Converts model files from the old format to the new format.
    Loads models from old_dir, converts them, and saves them to new_dir
    '''
    a0cc_model = A0CCModel(board_size=5)
    for iteration, model in model_generator(a0cc_model, old_dir, n):
        # Convert model to the new format (this is a placeholder, replace with actual conversion logic)
        converted_model = convert_model(model)
        
        # Save the converted model to the new directory
        save_model(new_dir + f'model_{iteration}.pkl', converted_model)

def main():
    old_gamedata_dir = config.training_dir
    new_gamedata_dir = f"output-converted{config.trial_num}/training/"
    old_model_dir = config.training_dir
    new_model_dir = f"output-converted{config.trial_num}/training/"
    n_iterations = config.training_iterations

    # Create the new directories if they don't exist
    os.makedirs(new_gamedata_dir, exist_ok=True)
    os.makedirs(new_model_dir, exist_ok=True)

    convert_gamedata(old_gamedata_dir, new_gamedata_dir, n_iterations)
    convert_models(old_model_dir, new_model_dir, n_iterations)
    # TODO: Should also just copy over the "iteration_stats_*.pkl" files.

if __name__ == "__main__":
    main()
