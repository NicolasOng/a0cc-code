from a0_new.protocols.model import NNModel

from a0.model import AlphaZeroModel

import numpy as np
from numpy.typing import NDArray
import jax
import jax.numpy as jnp
from flax import nnx

from config import config

class CCNNModel(NNModel):
    def __init__(self):
        self.model = AlphaZeroModel(
            board_size=config.board_size,
            num_filters=256,
            training=False,
            rngs=nnx.Rngs({'params': jax.random.PRNGKey(1)})
        )

    def evaluate(self, states: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        # converts to and from jax arrays
        # more efficient methods should just use the Flax model directly
        values, policies = self.model(jnp.asarray(states, dtype=jnp.float32))
        return np.array(values, dtype=np.float32), np.array(policies, dtype=np.float32)
    
    def get_nn_model(self, training: bool) -> nnx.Module:
        # get the structure of the model with the correct mode
        model = AlphaZeroModel(
            board_size=config.board_size,
            num_filters=256,
            training=training,
            rngs=nnx.Rngs({'params': jax.random.PRNGKey(1)})
        )
        graphdef, _ = nnx.split(model)

        # get the state of the current model
        _, state = nnx.split(self.model)

        # merge to get the model in the correct mode
        return nnx.merge(graphdef, state)
    
    def set_nn_model(self, model: nnx.Module, training: bool) -> None:
        # get the structure of the model with the correct mode
        new_model = AlphaZeroModel(
            board_size=config.board_size,
            num_filters=256,
            training=training,
            rngs=nnx.Rngs({'params': jax.random.PRNGKey(1)})
        )
        graphdef, _ = nnx.split(new_model)

        # get the state from the provided model
        _, state = nnx.split(model)

        # merge to set the model in the correct mode
        self.model = nnx.merge(graphdef, state)
