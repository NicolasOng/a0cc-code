"""
Empirical comparison of MSE vs MAE on duplicate states with conflicting labels.

Setup: A tiny network learns to map a single input state to a scalar value.
The same state appears multiple times with different binary labels {+1, -1},
with a controllable majority ratio. We compare what the model converges to
under MSE vs MAE loss.
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx


class Model(nnx.Module):
    def __init__(self, rngs):
        self.l1 = nnx.Linear(4, 8, rngs=rngs)
        self.l2 = nnx.Linear(8, 1, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.l1(x))
        return self.l2(x).squeeze(-1)


def make_dataset(n_pos=3, n_neg=1):
    x = jnp.ones((n_pos + n_neg, 4))
    y = jnp.concatenate([jnp.ones(n_pos), -jnp.ones(n_neg)])
    return x, y


def train(x, y, loss_name, steps=5000, lr=0.01, seed=0):
    model = Model(rngs=nnx.Rngs(seed))
    opt = nnx.Optimizer(model, optax.sgd(lr))

    if loss_name == "mse":
        def loss_fn(model, x, y):
            return jnp.mean((model(x) - y) ** 2)
    else:
        def loss_fn(model, x, y):
            return jnp.mean(jnp.abs(model(x) - y))

    @nnx.jit
    def step(model, opt, x, y):
        loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
        opt.update(grads)
        return loss

    for _ in range(steps):
        step(model, opt, x, y)

    return float(model(x[:1])[0])


def main():
    configs = [
        (3, 1),  # 75% positive
        (5, 1),  # 83% positive
        (9, 1),  # 90% positive
        (1, 1),  # 50-50
    ]

    print(f"{'ratio':>10} {'mean':>8} {'median':>8} {'MSE pred':>10} {'MAE pred':>10}")
    print("-" * 52)

    for n_pos, n_neg in configs:
        x, y = make_dataset(n_pos, n_neg)
        labels = y.tolist()
        mean = sum(labels) / len(labels)
        median = sorted(labels)[len(labels) // 2]

        mse_pred = train(x, y, "mse", steps=5000)
        mae_pred = train(x, y, "mae", steps=5000)

        ratio = f"{n_pos}:{n_neg}"
        print(f"{ratio:>10} {mean:>8.2f} {median:>8.2f} {mse_pred:>10.4f} {mae_pred:>10.4f}")


if __name__ == "__main__":
    main()
