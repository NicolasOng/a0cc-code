import jax
import jax.numpy as jnp

from a0.model import AlphaZeroModel
from flax import nnx

from flax import nnx
import jax
import jax.numpy as jnp
from flax import traverse_util

def print_model_summary(model):
    # 1. Get the state and pull the underlying dict
    state = nnx.state(model)
    
    # 2. Flatten the nested dictionary structure
    # This turns { 'layer': { 'kernel': arr } } into { ('layer', 'kernel'): arr }
    flat_state = traverse_util.flatten_dict(state.raw_mapping)
    
    header = f"{'Layer Path':<45} | {'Shape':<20} | {'Size':<12}"
    print(header)
    print("-" * len(header))
    
    total_params = 0
    for path_tuple, value in flat_state.items():
        # Join the path tuple into a string
        path_str = ".".join(map(str, path_tuple))
        
        # Access the array (in NNX, values are often Variable objects, so we get .value)
        arr = value.value if hasattr(value, 'value') else value
        
        if hasattr(arr, 'shape'):
            shape_str = str(arr.shape)
            size = arr.size
            total_params += size
            print(f"{path_str:<45} | {shape_str:<20} | {size:<12,}")

    print("-" * len(header))
    print(f"{'TOTAL PARAMETERS':<45} | {'':<20} | {total_params:<12,}")
    
    # Calculate memory
    total_bytes = sum(x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(state) if hasattr(x, 'size'))
    print(f"Total Memory Usage: {total_bytes / (1024**2):.2f} MB")


model = AlphaZeroModel(
        board_size=5,
        num_filters=256,
        rngs=nnx.Rngs({'params': jax.random.PRNGKey(1)})
    )

# Extract the state from your model
state = nnx.state(model)

# This 'state' contains everything: weights, biases, and even RNG states.
# You can filter it to just get the parameters if you like:
params = nnx.state(model, nnx.Param)

param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))

print(f"Total Parameters: {param_count:,}")

# Calculate size in bytes
param_bytes = sum(x.nbytes for x in jax.tree_util.tree_leaves(params))

# Convert to Megabytes
size_mb = param_bytes / (1024**2)
print(f"Model Size: {size_mb:.2f} MB")

print_model_summary(model)