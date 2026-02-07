from __future__ import annotations

from typing import Any, cast

from a0_new.protocols.model import RecursiveFullOnRawModel, T_nn_model, RawModel, FullModelOnRaw, FullModelOnFull
from a0_new.protocols.player import FullModelPlayer

def get_full_on_raw_from_player(
        player: FullModelPlayer[RecursiveFullOnRawModel[Any], Any, Any]
    ) -> FullModelOnRaw[Any, Any, Any]:
    # get the full model from the player
    model: RecursiveFullOnRawModel[Any] = player.get_model()
    # Navigate through FullModelOnFull wrappers to reach the FullModelOnRaw
    while hasattr(model, 'get_full_model'):
        model = cast(FullModelOnFull[RecursiveFullOnRawModel[Any], Any, Any], model).get_full_model()
    # Now model is a FullModelOnRaw, return it
    return cast(FullModelOnRaw[Any, Any, Any], model)

def get_nn_model_from_player(
        player: FullModelPlayer[RecursiveFullOnRawModel[T_nn_model], Any, Any]
    ) -> T_nn_model:
    # get the full model from the player
    model: RecursiveFullOnRawModel[T_nn_model] = player.get_model()
    # Navigate through FullModelOnFull wrappers to reach the FullModelOnRaw
    while hasattr(model, 'get_full_model'):
        model = cast(FullModelOnFull[RecursiveFullOnRawModel[T_nn_model], Any, Any], model).get_full_model()
    # Now model is a FullModelOnRaw, return its raw model (the NNModel)
    return cast(FullModelOnRaw[T_nn_model, Any, Any], model).get_raw_model()

def set_raw_model_to_player(
        player: FullModelPlayer[RecursiveFullOnRawModel[Any], Any, Any],
        raw_model: RawModel
    ) -> None:
    # get the full model from the player
    model: RecursiveFullOnRawModel[Any] = player.get_model()
    # Navigate through FullModelOnFull wrappers to reach the FullModelOnRaw
    while hasattr(model, 'get_full_model'):
        model = cast(FullModelOnFull[RecursiveFullOnRawModel[Any], Any, Any], model).get_full_model()
    # Now model is a FullModelOnRaw, set its raw model
    cast(FullModelOnRaw[Any, Any, Any], model).set_raw_model(raw_model)
