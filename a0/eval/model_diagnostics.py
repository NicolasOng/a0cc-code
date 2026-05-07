"""Per-iteration model diagnostics for characterizing training collapse.

The public entry point is `compute_iteration_diagnostics`, which runs a probe
batch through the model and returns a dict containing:
  - per-site pre-activation statistics (mean, max, dead-neuron fraction) at
    every ReLU site plus the value head's pre-tanh output and policy logits
  - prediction-distribution stats (mean/std/min/max) for value and policy
  - per-parameter + total gradient norms under a training-equivalent loss

The probe batch is expected to be a Dataset of uniformly-sampled states with
fixed random targets (see `a0.utils.states.get_random_states` with a
`RankUnrank()` + `get_rd_from_states`), so diagnostics are comparable
across iterations.
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


def _build_site_grad_norms(
    model: AlphaZeroModel | MultiTrunkAlphaZeroModel,
    per_param_grad_norm: dict[str, float],
) -> dict[str, float]:
    """Map each activation site to the gradient norm of the kernel that produces it.
    Uses exact pytree key strings from jax.tree_util.keystr — format is stable
    across the nnx versions used in this project."""
    g = per_param_grad_norm
    result: dict[str, float] = {}

    if isinstance(model, MultiTrunkAlphaZeroModel):
        result['value_trunk_stem'] = g.get("['value_conv']['kernel'].value", 0.0)
        result['policy_trunk_stem'] = g.get("['policy_conv']['kernel'].value", 0.0)
        for i in range(len(model.value_resblocks)):
            result[f'value_trunk_resblock_{i}_inner'] = g.get(f"['value_resblocks'][{i}]['conv1']['kernel'].value", 0.0)
            result[f'value_trunk_resblock_{i}_outer'] = g.get(f"['value_resblocks'][{i}]['conv2']['kernel'].value", 0.0)
        for i in range(len(model.policy_resblocks)):
            result[f'policy_trunk_resblock_{i}_inner'] = g.get(f"['policy_resblocks'][{i}]['conv1']['kernel'].value", 0.0)
            result[f'policy_trunk_resblock_{i}_outer'] = g.get(f"['policy_resblocks'][{i}]['conv2']['kernel'].value", 0.0)
    else:
        result['stem'] = g.get("['conv']['kernel'].value", 0.0)
        for i in range(len(model.resblocks)):
            result[f'resblock_{i}_inner'] = g.get(f"['resblocks'][{i}]['conv1']['kernel'].value", 0.0)
            result[f'resblock_{i}_outer'] = g.get(f"['resblocks'][{i}]['conv2']['kernel'].value", 0.0)

    result['policy_head_conv'] = g.get("['policy_head']['conv']['kernel'].value", 0.0)
    result['policy_output']    = g.get("['policy_head']['dense']['kernel'].value", 0.0)
    result['value_head_dense1'] = g.get("['value_head']['dense1']['kernel'].value", 0.0)
    result['value_pre_tanh']   = g.get("['value_head']['dense2']['kernel'].value", 0.0)

    return result


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
    per_site_grad_norm = _build_site_grad_norms(model, per_param_grad_norm)

    return {
        'pred_std': prediction_dist['value']['std'],
        'total_grad_norm': total_grad_norm,
        'per_param_grad_norm': per_param_grad_norm,
        'per_site_grad_norm': per_site_grad_norm,
        'per_site_stats': per_site_stats,
        'prediction_dist': prediction_dist,
        'loss': float(loss),
        'value_loss': float(value_loss),
        'policy_loss': float(policy_loss),
    }


def _format_diagnostics_table(diagnostics: dict[str, Any]) -> list[str]:
    """Returns lines for the human-readable diagnostics table, without a header."""
    pv = diagnostics['prediction_dist']['value']
    site_stats = diagnostics['per_site_stats']
    site_grads = diagnostics['per_site_grad_norm']
    site_w = max(len(s) for s in site_stats)
    lines = [
        f"  pred   mean={pv['mean']:+.3f}  std={pv['std']:.4f}  min={pv['min']:+.3f}  max={pv['max']:+.3f}",
        f"  grad   total_norm={diagnostics['total_grad_norm']:.4e}",
        f"  {'site':<{site_w}}  mean_abs   max_abs   dead_frac   kern_grad",
    ]
    for site, stats in site_stats.items():
        grad = site_grads.get(site, float('nan'))
        lines.append(
            f"  {site:<{site_w}}  {stats['mean_abs']:8.4f}   "
            f"{stats['max_abs']:7.4f}   {stats['dead_frac']:9.2%}   {grad:.3e}"
        )
    return lines


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
    lines = [f"Iter {iteration} diagnostics (attempt {attempt}):"] + _format_diagnostics_table(diagnostics)
    logger.log(25, "\n".join(lines))
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


if __name__ == "__main__":
    import os
    import sys
    from a0.utils.load_training_data import load_models
    from a0.utils.states import get_random_states, remove_duplicates, get_rd_from_states
    from cc.ground_truth import RankUnrank

    training_dir = config.training_dir
    out_path = config.log_dir + "model_diagnostics.jsonl"

    model_entries = load_models(training_dir, config.training_iterations)

    if not model_entries:
        print("No models loaded.", file=sys.stderr)
        sys.exit(1)

    print("Building probe dataset...")
    n = 512
    _states = get_random_states(n, RankUnrank())
    _states, _ = remove_duplicates(_states)
    rsrd = get_rd_from_states(_states, n, shuffle=True)

    for iteration, model in model_entries:
        print(f"\n=== iter {iteration} ===")

        d = compute_iteration_diagnostics(model, rsrd)

        for line in _format_diagnostics_table(d):
            print(line)

        if out_path:
            entry = {
                "iteration": iteration,
                "timestamp": time.time(),
                **d,
            }
            with open(out_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    if out_path:
        print(f"\nDiagnostics written to {out_path}")
