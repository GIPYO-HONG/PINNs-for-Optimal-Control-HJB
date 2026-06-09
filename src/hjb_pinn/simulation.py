import diffrax
import jax
import jax.numpy as jnp

from .config import SEIRConfig, TrainingConfig
from .problem import (
    controlled_dynamics,
    optimal_control,
    uncontrolled_dynamics,
)


def simulate(
    model,
    c_val: float,
    seir: SEIRConfig,
    training: TrainingConfig,
    points: int = 300,
):
    times = jnp.linspace(0.0, training.horizon, points)
    initial_state = jnp.asarray(seir.initial_state)
    controller = diffrax.PIDController(rtol=1e-5, atol=1e-7)
    save_at = diffrax.SaveAt(ts=times)

    def controlled_field(t, state, args):
        current_model, current_c = args
        clipped_state = jnp.clip(state, 0.0, training.state_max)
        return controlled_dynamics(
            current_model,
            t,
            clipped_state,
            current_c,
            seir,
            training,
        )

    def uncontrolled_field(t, state, current_c):
        clipped_state = jnp.clip(state, 0.0, training.state_max)
        return uncontrolled_dynamics(clipped_state, current_c, seir)

    controlled = diffrax.diffeqsolve(
        diffrax.ODETerm(controlled_field),
        diffrax.Dopri5(),
        t0=0.0,
        t1=training.horizon,
        dt0=0.1,
        y0=initial_state,
        args=(model, c_val),
        stepsize_controller=controller,
        saveat=save_at,
    ).ys
    uncontrolled = diffrax.diffeqsolve(
        diffrax.ODETerm(uncontrolled_field),
        diffrax.Dopri5(),
        t0=0.0,
        t1=training.horizon,
        dt0=0.1,
        y0=initial_state,
        args=c_val,
        stepsize_controller=controller,
        saveat=save_at,
    ).ys
    controls = jax.vmap(
        lambda t, x: optimal_control(
            model, t, x, c_val, seir, training
        )
    )(times, controlled)
    return times, controlled, uncontrolled, controls
