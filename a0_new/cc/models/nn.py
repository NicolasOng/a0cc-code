from a0_new.protocols.model import NNModel

from a0_new.cc.model import A0CCModel

import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp

from config import config

class CCNNModel(NNModel[A0CCModel]):
    def __init__(self):
        self.model: A0CCModel = A0CCModel(
            board_size=config.board_size
        )

    def evaluate(self, states: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        # converts to and from jax arrays
        # more efficient methods should just use the Flax model directly
        jnp_states = jnp.asarray(states, dtype=jnp.float32)
        values, policies = self.model.inference(jnp_states)
        return np.array(values, dtype=np.float32), np.array(policies, dtype=np.float32)
    
    def get_nn_model(self) -> A0CCModel:
        return self.model
    
    def set_nn_model(self, model: A0CCModel) -> None:
        self.model = model
