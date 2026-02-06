from a0.model import AlphaZeroModel

from flax import nnx
import jax

import pickle
import dill

model = AlphaZeroModel(5, nnx.Rngs({'params': jax.random.PRNGKey(1)}))

if False:
    # this doesn't work
    # because the model contains a graphdef which can't be pickled
    pickled_model = pickle.dumps(model)
    unpickled_model = pickle.loads(pickled_model)

if False:
    # similarly, this doesn't work because the graphdef can't be pickled
    graphdef, state = nnx.split(model)
    pickled_graphdef = pickle.dumps(graphdef)
    pickled_state = pickle.dumps(state)
    unpickled_graphdef = pickle.loads(pickled_graphdef)
    unpickled_state = pickle.loads(pickled_state)

if True:
    # this works, because dill can handle the graphdef
    pickled_model = dill.dumps(model)
    unpickled_model = dill.loads(pickled_model)