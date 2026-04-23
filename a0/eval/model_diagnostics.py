"""Per-iteration model diagnostics for characterizing training collapse.

The public entry point is `compute_iteration_diagnostics`, which runs a probe
batch through the model and returns a dict containing:
  - per-site pre-activation statistics (mean, max, dead-neuron fraction) at
    every ReLU site plus the value head's pre-tanh output and policy logits
  - prediction-distribution stats (mean/std/min/max) for value and policy
  - per-parameter + total gradient norms under a training-equivalent loss

The probe batch is expected to be a Dataset of uniformly-sampled states with
fixed random targets (see `a0.utils.states.get_random_states_no_gt` +
`get_rd_from_states`), so diagnostics are comparable across iterations.
"""

from typing import Any
import gc
import json
import time

import jax
import jax.numpy as jnp
from flax import nnx

from config import config

from a0.model import AlphaZeroModel, MultiTrunkAlphaZeroModel
from a0.train.dataset import value_loss_function, policy_loss_function

from utils.log import get_logger
logger = get_logger(__name__)


def _path_to_str(path: Any) -> str:
    try:
        return jax.tree_util.keystr(path)
    except Exception:
        parts: list[str] = []
        for k in path:
            if hasattr(k, 'key'):
                parts.append(str(k.key))
            elif hasattr(k, 'name'):
                parts.append(str(k.name))
            elif hasattr(k, 'idx'):
                parts.append(str(k.idx))
            else:
                parts.append(str(k))
        return '.'.join(parts)


def _site_stats(pre_act: jnp.ndarray) -> dict[str, float]:
    """Summary stats for a pre-activation tensor. `dead_frac` only makes sense
    at pre-ReLU sites: a unit is dead if its pre-activation is <= 0 across the
    entire probe batch (so the ReLU after it outputs 0 everywhere)."""
    a = pre_act.astype(jnp.float32)
    reduce_dims = tuple(range(a.ndim - 1))
    max_per_unit = jnp.max(a, axis=reduce_dims)
    dead_frac = jnp.mean((max_per_unit <= 0).astype(jnp.float32))
    return {
        'mean_abs': float(jnp.mean(jnp.abs(a))),
        'max_abs': float(jnp.max(jnp.abs(a))),
        'mean': float(jnp.mean(a)),
        'dead_frac': float(dead_frac),
    }


def _resblock_with_intermediates(
    block: Any,
    x: jnp.ndarray,
    train: bool,
    intermediates: dict[str, jnp.ndarray],
    prefix: str,
) -> jnp.ndarray:
    shortcut = x
    x = block.conv1(x)
    x = block.bn1(x, use_running_average=not train)
    intermediates[f'{prefix}_inner'] = x
    x = jax.nn.relu(x)
    x = block.conv2(x)
    x = block.bn2(x, use_running_average=not train)
    x = x + shortcut
    intermediates[f'{prefix}_outer'] = x
    x = jax.nn.relu(x)
    return x


def _forward_with_intermediates(
    model: AlphaZeroModel | MultiTrunkAlphaZeroModel,
    x: jnp.ndarray,
    train: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    """Replicates the model forward pass while capturing pre-activations at
    every ReLU site plus the value head's pre-tanh output. Handles both
    single-trunk (AlphaZeroModel) and separate-backbone (MultiTrunk) variants."""
    intermediates: dict[str, jnp.ndarray] = {}

    if isinstance(model, MultiTrunkAlphaZeroModel):
        v = model.value_conv(x)
        v = model.value_bn(v, use_running_average=not train)
        intermediates['value_trunk_stem'] = v
        v = jax.nn.relu(v)
        for i, block in enumerate(model.value_resblocks):
            v = _resblock_with_intermediates(block, v, train, intermediates, f'value_trunk_resblock_{i}')

        p = model.policy_conv(x)
        p = model.policy_bn(p, use_running_average=not train)
        intermediates['policy_trunk_stem'] = p
        p = jax.nn.relu(p)
        for i, block in enumerate(model.policy_resblocks):
            p = _resblock_with_intermediates(block, p, train, intermediates, f'policy_trunk_resblock_{i}')

        value_trunk_out, policy_trunk_out = v, p
    else:
        h = model.conv(x)
        h = model.bn(h, use_running_average=not train)
        intermediates['stem'] = h
        h = jax.nn.relu(h)
        for i, block in enumerate(model.resblocks):
            h = _resblock_with_intermediates(block, h, train, intermediates, f'resblock_{i}')
        value_trunk_out = policy_trunk_out = h

    # policy head
    p = model.policy_head.conv(policy_trunk_out)
    p = model.policy_head.bn(p, use_running_average=not train)
    intermediates['policy_head_conv'] = p
    p = jax.nn.relu(p)
    p = p.reshape((p.shape[0], -1))
    policy_logits = model.policy_head.dense(p)
    intermediates['policy_output'] = policy_logits

    # value head
    v = value_trunk_out.reshape((value_trunk_out.shape[0], -1))
    v = model.value_head.dense1(v)
    intermediates['value_head_dense1'] = v
    v = jax.nn.relu(v)
    pre_tanh = model.value_head.dense2(v)
    intermediates['value_pre_tanh'] = pre_tanh
    value = jnp.tanh(pre_tanh)

    return value, policy_logits, intermediates


def _diag_loss_fn(model: Any, batch: dict[str, Any]) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
    """Eval-mode loss for the diagnostic backward pass. Uses inference-mode BN
    (running stats) so grads can be computed without advancing the model's
    BatchNorm running statistics — those should only be touched by the real
    training loop."""
    board_input = batch['board']
    value_label = batch['value']
    policy_label = batch['policy']
    policy_mask = batch['mask']

    value, policy = model(board_input, train=False)

    value_loss = value_loss_function(value, value_label, None)
    masked_pred_logits = jnp.where(policy_mask, policy, -1e9)
    masked_label_policy = jnp.where(policy_mask, policy_label, 0.0)
    policy_loss = policy_loss_function(masked_label_policy, masked_pred_logits)
    total_loss = config.value_loss_weight * value_loss + policy_loss
    return total_loss, (value_loss, policy_loss)


def compute_iteration_diagnostics(
    model: AlphaZeroModel | MultiTrunkAlphaZeroModel,
    rsrd: Any,
) -> dict[str, Any]:
    """Runs the rsrd probe batch through the model and collects diagnostic
    statistics: per-site pre-activation magnitudes, dead-neuron fractions,
    prediction distribution, and per-parameter + total gradient norms."""
    boards = jnp.asarray(rsrd.states, dtype=jnp.float32)
    values = jnp.asarray(rsrd.values, dtype=jnp.float32)
    policies = jnp.asarray(rsrd.policies, dtype=jnp.float32)
    masks = jnp.asarray(rsrd.masks)

    # self-play workers have exited by this point; reclaim before the probe
    jax.clear_caches()
    gc.collect()

    value_pred, policy_logits, intermediates = _forward_with_intermediates(model, boards, train=False)
    per_site_stats = {name: _site_stats(a) for name, a in intermediates.items()}

    v = value_pred.astype(jnp.float32)
    p = policy_logits.astype(jnp.float32)
    prediction_dist = {
        'value': {
            'mean': float(jnp.mean(v)),
            'std': float(jnp.std(v)),
            'min': float(jnp.min(v)),
            'max': float(jnp.max(v)),
        },
        'policy_logits': {
            'mean': float(jnp.mean(p)),
            'std': float(jnp.std(p)),
            'min': float(jnp.min(p)),
            'max': float(jnp.max(p)),
        },
    }

    batch = {'board': boards, 'value': values, 'policy': policies, 'mask': masks}
    grad_fn = nnx.value_and_grad(_diag_loss_fn, has_aux=True)
    (loss, (value_loss, policy_loss)), grads = grad_fn(model, batch)

    per_param_grad_norm: dict[str, float] = {}
    total_sq = 0.0
    for path, leaf in jax.tree_util.tree_leaves_with_path(grads):
        name = _path_to_str(path)
        sq = float(jnp.sum(leaf.astype(jnp.float32) ** 2))
        per_param_grad_norm[name] = sq ** 0.5
        total_sq += sq
    total_grad_norm = total_sq ** 0.5

    return {
        'pred_std': prediction_dist['value']['std'],
        'total_grad_norm': total_grad_norm,
        'per_param_grad_norm': per_param_grad_norm,
        'per_site_stats': per_site_stats,
        'prediction_dist': prediction_dist,
        'loss': float(loss),
        'value_loss': float(value_loss),
        'policy_loss': float(policy_loss),
    }


def log_iteration_diagnostics(
    model: AlphaZeroModel | MultiTrunkAlphaZeroModel,
    rsrd: Any,
    log_path: str,
    attempt: int,
    iteration: int,
) -> dict[str, Any]:
    """Compute per-iteration diagnostics, log a human-readable summary, and
    append the full record (plus `attempt`, `iteration`, `timestamp`) to the
    jsonl at `log_path`. Returns the diagnostics dict so the caller can use
    fields like `pred_std` for early-abort decisions."""
    diagnostics = compute_iteration_diagnostics(model, rsrd)
    pred_std = diagnostics['pred_std']
    logger.log(25,
        f"Iter {iteration} diagnostics: pred_std={pred_std:.4f}, "
        f"total_grad_norm={diagnostics['total_grad_norm']:.4f}, "
        f"value_pre_tanh_max_abs={diagnostics['per_site_stats']['value_pre_tanh']['max_abs']:.2f}"
    )
    entry = {
        "attempt": attempt,
        "iteration": iteration,
        "timestamp": time.time(),
        **diagnostics,
    }
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        logger.log(30, f"Failed to write iteration diagnostics: {e}")
    return diagnostics
