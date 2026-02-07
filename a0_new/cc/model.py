import jax
import jax.numpy as jnp
from flax import nnx

from a0_new.protocols.model import TrainableModel

import pickle

from config import config

# class for the Residual Block (ResNet)
class ResidualBlock(nnx.Module):
    def __init__(self, features: int, rngs: nnx.Rngs):
        super().__init__()
        '''
        Initializes the Residual Block.
        Args:
        - features: the number of channels (depth) in the input tensor x, i.e., x.shape[-1].
        - rngs: a nnx.Rngs object containing random number generators for the model.
        '''
        self.conv1 = nnx.Conv(in_features=features, out_features=features, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False, rngs=rngs)
        self.conv2 = nnx.Conv(in_features=features, out_features=features, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(num_features=features, momentum=0.9, epsilon=1e-5, rngs=rngs)
        self.bn2 = nnx.BatchNorm(num_features=features, momentum=0.9, epsilon=1e-5, rngs=rngs)

    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        '''
        Applies the residual block to the input tensor x.
        The input tensor x is expected to have shape (batch_size, height, width, channels).
        The output tensor will have the same shape as the input tensor.
        '''
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not train)
        x = x + shortcut
        x = jax.nn.relu(x)
        return x

# class for the Policy Head
class PolicyHead(nnx.Module):
    def __init__(self, board_size: int, in_channels: int, num_filters: int, rngs: nnx.Rngs):
        '''
        Initializes the Policy Head.
        Args:
        - in_channels: the number of channels in the input x.
        - num_filters: the number of filters in the convolutional layer.
        - rngs: a nnx.Rngs object containing random number generators for the model.
        '''
        super().__init__()
        self.conv = nnx.Conv(in_features=in_channels, out_features=num_filters, kernel_size=(1, 1), strides=(1, 1), padding='SAME', use_bias=False, rngs=rngs)
        self.bn = nnx.BatchNorm(num_features=num_filters, momentum=0.9, epsilon=1e-5, rngs=rngs)
        self.dense = nnx.Linear(in_features=board_size*board_size*num_filters, out_features=board_size**4, rngs=rngs)

    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        '''
        Applies the policy head to the input tensor x.
        The input tensor x is expected to have shape (batch_size, height, width, channels).
        The output tensor will have shape (batch_size, BOARD_SIZE**4).
        '''
        # (batch_size, height, width, channels)
        x = self.conv(x)
        x = self.bn(x, use_running_average=not train)
        x = jax.nn.relu(x)
        # (batch_size, height, width, num_filters)
        x = x.reshape((x.shape[0], -1))
        # (batch_size, height * width * num_filters)
        x = self.dense(x)
        # (batch_size, BOARD_SIZE**4)
        return x

# class for the Value Head
class ValueHead(nnx.Module):
    def __init__(self, in_features: int, rngs: nnx.Rngs):
        '''
        Initializes the Value Head.
        Args:
        - in_features: the number of input features (height * width * channels).
        - rngs: a nnx.Rngs object containing random number generators for the model.
        '''
        super().__init__()
        self.dense1 = nnx.Linear(in_features=in_features, out_features=256, rngs=rngs)
        self.dense2 = nnx.Linear(in_features=256, out_features=1, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        '''
        Applies the value head to the input tensor x.
        The input tensor x is expected to have shape (batch_size, height, width, in_features).
        The output tensor will have shape (batch_size, 1).
        '''
        # x: (batch_size, height, width, in_features)
        x = x.reshape((x.shape[0], -1))  # flatten
        # x: (batch_size, height * width * in_features)
        x = self.dense1(x)
        # x: (batch_size, 256)
        x = jax.nn.relu(x)
        # x: (batch_size, 256)
        x = self.dense2(x)
        # x: (batch_size, 1)
        x = jnp.tanh(x)
        # x: (batch_size, 1)
        return x

# class for the AlphaZero model
class A0CCModel(TrainableModel):
    def __init__(
            self,
            board_size: int,
            rngs: nnx.Rngs = nnx.Rngs({'params': jax.random.PRNGKey(1)}),
            num_filters: int = 256,
            num_resblocks: int = 3
        ):
        '''
        Initializes the A0CC model.
        Args:
        - board_size: the size of the board (e.g., 7 for a 7x7 board).
        - rngs: a nnx.Rngs object containing random number generators for the model.
        - num_filters: the number of filters in the convolutional layers throughout the model.
        '''
        super().__init__()
        self.conv = nnx.Conv(in_features=2, out_features=num_filters, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False, rngs=rngs)
        self.bn = nnx.BatchNorm(num_features=num_filters, momentum=0.9, epsilon=1e-5, rngs=rngs)
        self.resblocks = [ResidualBlock(num_filters, rngs=rngs) for _ in range(num_resblocks)]
        self.policy_head = PolicyHead(board_size, num_filters, num_filters, rngs=rngs)
        self.value_head = ValueHead(board_size * board_size * num_filters, rngs=rngs)

    def __call__(self, x: jnp.ndarray, train: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
        '''
        Applies the A0CC model to the input tensor x.
        The input tensor x is expected to have shape (batch_size, board_size, board_size, 2).
        The output will be a tuple of two tensors:
        - value: a tensor of shape (batch_size, 1) representing the value of the position.
        - policy: a tensor of shape (batch_size, BOARD_SIZE**4) representing the (unmasked) policy distribution.
        '''
        # (batch_size, height, width, 2)
        x = self.conv(x)
        x = self.bn(x, use_running_average=not train)
        x = jax.nn.relu(x)
        # (batch_size, height, width, num_filters)
        for block in self.resblocks:
            x = block(x, train=train)
        policy = self.policy_head(x, train=train)
        # policy: (batch_size, BOARD_SIZE**4)
        value = self.value_head(x)
        # value: (batch_size, 1)
        return value, policy
    
    @nnx.jit
    def inference(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self(x, train=False)
    
    def train_inference(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self(x, train=True)

    def save_to_file(self, path: str) -> None:
        # get the state of the model
        _, state = nnx.split(self)

        # use pickle to save the state
        with open(path, 'wb') as f:
            pickle.dump(state, f)

def load_model(filepath: str) -> A0CCModel:
    # create a new model instance with the same parameters
    model = A0CCModel(
        board_size=config.board_size
    )
    
    # load the state from the pickle file
    with open(filepath, 'rb') as f:
        state = pickle.load(f)
    
    # get the structure of the model
    graphdef, _ = nnx.split(model)

    # restore the state
    model = nnx.merge(graphdef, state)

    return model
