from a0_new.protocols.model import NNModel

from a0.model import AlphaZeroModel

import numpy as np
from numpy.typing import NDArray
import jax
import jax.numpy as jnp
from flax import nnx

from config import config

@nnx.jit
def inference_step(model: AlphaZeroModel, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    return model(x, train=False)

class CCNNModel(NNModel[AlphaZeroModel]):
    def __init__(self):
        self.model: AlphaZeroModel = AlphaZeroModel(
            board_size=config.board_size,
            num_filters=256,
            rngs=nnx.Rngs({'params': jax.random.PRNGKey(1)})
        )

    def evaluate(self, states: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        # converts to and from jax arrays
        # more efficient methods should just use the Flax model directly
        jnp_states = jnp.asarray(states, dtype=jnp.float32)
        values, policies = inference_step(self.model, jnp_states)
        return np.array(values, dtype=np.float32), np.array(policies, dtype=np.float32)
    
    def get_nn_model(self) -> AlphaZeroModel:
        return self.model
    
    def set_nn_model(self, model: AlphaZeroModel) -> None:
        self.model = model
