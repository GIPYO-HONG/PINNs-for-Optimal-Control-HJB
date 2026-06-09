import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp

from .config import ExperimentConfig, TrainingConfig


class ValueNet(eqx.Module):
    mlp: eqx.nn.MLP
    conditioned: bool = eqx.field(static=True)
    horizon: float = eqx.field(static=True)
    state_max: float = eqx.field(static=True)
    c_min: float = eqx.field(static=True)
    c_range: float = eqx.field(static=True)

    def __init__(
        self,
        training: TrainingConfig,
        experiment: ExperimentConfig,
        *,
        key,
        width_size: int = 128,
        depth: int = 8,
    ):
        self.conditioned = experiment.conditioned
        self.horizon = training.horizon
        self.state_max = training.state_max
        self.c_min = experiment.c_min
        self.c_range = experiment.c_range
        self.mlp = eqx.nn.MLP(
            in_size=1 + training.state_dim + int(experiment.conditioned),
            out_size=1,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            final_activation=lambda x: x,
            key=key,
        )

    def __call__(self, t, x, c_val):
        time = jnp.asarray([t / self.horizon])
        state = x / self.state_max

        if not self.conditioned:
            return self.mlp(jnp.concatenate([time, state])).squeeze()

        normalized_c = jnp.asarray([(c_val - self.c_min) / self.c_range])
        return self.mlp(jnp.concatenate([time, normalized_c, state])).squeeze()


def make_model(key, training: TrainingConfig, experiment: ExperimentConfig):
    return ValueNet(training, experiment, key=key)
