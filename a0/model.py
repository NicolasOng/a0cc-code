import jax
import jax.numpy as jnp
from flax import nnx

from wrappers import ccwrapper as cw

# class for the Residual Block (ResNet)
class ResidualBlock(nnx.Module):
    def __init__(self, features: int, training: bool, rngs: nnx.Rngs):
        super().__init__()
        # features is the number of input channels (depth) in the input tensor x, i.e., x.shape[-1].
        # due to the skip connection, the output tensor will have the same number of channels.
        self.conv1 = nnx.Conv(in_features=features, out_features=features, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False, rngs=rngs)
        self.conv2 = nnx.Conv(in_features=features, out_features=features, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(num_features=features, use_running_average=not training, momentum=0.9, epsilon=1e-5, rngs=rngs)
        self.bn2 = nnx.BatchNorm(num_features=features, use_running_average=not training, momentum=0.9, epsilon=1e-5, rngs=rngs)
        self.training = training

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        '''
        Applies the residual block to the input tensor x.
        The input tensor x is expected to have shape (batch_size, height, width, channels).
        The output tensor will have the same shape as the input tensor.
        '''
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + shortcut
        x = jax.nn.relu(x)
        return x

# class for the Policy Head
class PolicyHead(nnx.Module):
    def __init__(self, in_features: int, num_filters: int, training: bool, *, rngs: nnx.Rngs):
        super().__init__()
        self.conv = nnx.Conv(in_features=in_features, out_features=num_filters, kernel_size=(1, 1), strides=(1, 1), padding='SAME', use_bias=False, rngs=rngs)
        self.bn = nnx.BatchNorm(num_features=num_filters, use_running_average=not training, momentum=0.9, epsilon=1e-5, rngs=rngs)
        self.dense = nnx.Linear(in_features=cw.BOARD_SIZE*cw.BOARD_SIZE*num_filters, out_features=cw.BOARD_SIZE**4, rngs=rngs)
        self.training = training

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        '''
        Applies the policy head to the input tensor x.
        The input tensor x is expected to have shape (batch_size, height, width, in_features).
        The output tensor will have shape (batch_size, BOARD_SIZE**4).
        '''
        # (batch_size, height, width, in_features)
        x = self.conv(x)
        x = self.bn(x)
        x = jax.nn.relu(x)
        # (batch_size, height, width, num_filters)
        x = x.reshape((x.shape[0], -1))
        # (batch_size, height * width * num_filters)
        x = self.dense(x)
        # (batch_size, BOARD_SIZE**4)
        return x

# class for the Value Head
class ValueHead(nnx.Module):
    def __init__(self, in_features: int, training: bool, *, rngs: nnx.Rngs):
        super().__init__()
        self.dense1 = nnx.Linear(in_features=in_features, out_features=256, rngs=rngs)
        self.dense2 = nnx.Linear(in_features=256, out_features=1, rngs=rngs)
        self.training = training

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
class AlphaZeroModel(nnx.Module):
    def __init__(self, in_features: int, num_filters: int, training: bool, *, rngs: nnx.Rngs):
        super().__init__()
        self.conv = nnx.Conv(in_features=in_features, out_features=num_filters, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False, rngs=rngs)
        self.bn = nnx.BatchNorm(num_features=num_filters, use_running_average=not training, momentum=0.9, epsilon=1e-5, rngs=rngs)
        self.resblocks = [ResidualBlock(num_filters, training, rngs=rngs) for _ in range(3)]
        self.policy_head = PolicyHead(num_filters, training, rngs=rngs)
        self.value_head = ValueHead(num_filters, training, rngs=rngs)
        self.training = training

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        '''
        Applies the AlphaZero model to the input tensor x.
        The input tensor x is expected to have shape (batch_size, height, width, in_features).
        The output will be a tuple of two tensors:
        - value: a tensor of shape (batch_size, 1) representing the value of the position.
        - policy: a tensor of shape (batch_size, BOARD_SIZE**4) representing the (unmasked) policy distribution.
        '''
        # (batch_size, height, width, in_features)
        x = self.conv(x)
        x = self.bn(x)
        x = jax.nn.relu(x)
        # (batch_size, height, width, num_filters)
        for block in self.resblocks:
            x = block(x)
        policy = self.policy_head(x)
        # policy: (batch_size, BOARD_SIZE**4)
        value = self.value_head(x)
        # value: (batch_size, 1)
        return value, policy

def create_model(rng: jax.Array, input_shape: tuple[int, ...], num_filters: int = 256, training: bool = True) -> AlphaZeroModel:
    rngs = nnx.Rngs({'params': rng})
    model = AlphaZeroModel(num_filters, training, rngs=rngs)
    return model

def forward_pass(model: AlphaZeroModel, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    return model(x)

def main() -> None:
    rng = jax.random.PRNGKey(42)
    input_shape = (1, 5, 5, 1)
    model = create_model(rng, input_shape)
    x = jax.random.normal(rng, input_shape)
    for i in range(10):
        v, p = forward_pass(model, x)
        print(v.shape, p.shape)

if __name__ == "__main__":
    main()

