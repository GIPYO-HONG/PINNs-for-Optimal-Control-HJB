import jax.numpy as jnp
import jax.random as jr

from .config import ExperimentConfig, TrainingConfig


def _sample_states(key_raw, key_total, size: int, config: TrainingConfig):
    totals = jr.uniform(
        key_total,
        (size,),
        minval=config.state_min_total,
        maxval=config.state_max,
    )
    raw = -jnp.log(
        jr.uniform(
            key_raw,
            (size, config.state_dim),
            minval=1e-6,
            maxval=1.0,
        )
    )
    return raw / raw.sum(axis=1, keepdims=True) * totals[:, None]


def sample_batch(
    key,
    training: TrainingConfig,
    experiment: ExperimentConfig,
):
    key, kt, kx, kc, kxt, kct, kn, knt = jr.split(key, 8)
    t_collocation = jr.uniform(
        kt,
        (training.collocation_size,),
        minval=0.0,
        maxval=training.horizon,
    )
    x_collocation = _sample_states(
        kx, kn, training.collocation_size, training
    )
    x_terminal = _sample_states(
        kxt, knt, training.terminal_size, training
    )

    if experiment.conditioned:
        c_collocation = jr.uniform(
            kc,
            (training.collocation_size,),
            minval=experiment.c_min,
            maxval=experiment.c_max,
        )
        c_terminal = jr.uniform(
            kct,
            (training.terminal_size,),
            minval=experiment.c_min,
            maxval=experiment.c_max,
        )
    else:
        c_collocation = jnp.full(
            (training.collocation_size,), experiment.fixed_c
        )
        c_terminal = jnp.full(
            (training.terminal_size,), experiment.fixed_c
        )

    batch = (
        t_collocation,
        x_collocation,
        c_collocation,
        x_terminal,
        c_terminal,
    )
    return key, batch
