import jax
import jax.numpy as jnp

from .config import SEIRConfig, TrainingConfig


def value(model, t, x, c_val, training: TrainingConfig):
    return model(t, x, c_val) * training.value_scale


def optimal_control(
    model,
    t,
    x,
    c_val,
    seir: SEIRConfig,
    training: TrainingConfig,
):
    grad_x_norm = jax.grad(lambda state: model(t, state, c_val))(x)
    grad_x = grad_x_norm * training.value_scale
    susceptible = x[0]
    control = 0.5 * susceptible * (grad_x[0] - grad_x[3])
    return jnp.clip(control, 0.0, seir.max_control)


def uncontrolled_dynamics(x, c_val, seir: SEIRConfig):
    susceptible, exposed, infected, recovered = x
    population = x.sum()

    d_susceptible = (
        seir.birth_rate * population
        - seir.death_rate * susceptible
        - c_val * susceptible * infected
    )
    d_exposed = (
        c_val * susceptible * infected
        - (seir.exposed_rate + seir.death_rate) * exposed
    )
    d_infected = (
        seir.exposed_rate * exposed
        - (
            seir.recovery_rate
            + seir.disease_death_rate
            + seir.death_rate
        )
        * infected
    )
    d_recovered = (
        seir.recovery_rate * infected - seir.death_rate * recovered
    )
    return jnp.asarray(
        [d_susceptible, d_exposed, d_infected, d_recovered]
    )


def controlled_dynamics(
    model,
    t,
    x,
    c_val,
    seir: SEIRConfig,
    training: TrainingConfig,
):
    control = optimal_control(model, t, x, c_val, seir, training)
    derivative = uncontrolled_dynamics(x, c_val, seir)
    vaccinated = control * x[0]
    return derivative + jnp.asarray([-vaccinated, 0.0, 0.0, vaccinated])


def running_cost(
    model,
    t,
    x,
    c_val,
    seir: SEIRConfig,
    training: TrainingConfig,
):
    control = optimal_control(model, t, x, c_val, seir, training)
    return seir.infection_cost * x[2] + control**2


def pde_residual(
    model,
    t,
    x,
    c_val,
    seir: SEIRConfig,
    training: TrainingConfig,
):
    value_t = jax.grad(lambda time: model(time, x, c_val))(t)
    value_x = jax.grad(lambda state: model(t, state, c_val))(x)
    dynamics = controlled_dynamics(
        model, t, x, c_val, seir, training
    )
    cost = running_cost(model, t, x, c_val, seir, training)
    return value_t + jnp.dot(value_x, dynamics) + cost / training.value_scale


def loss_fn(model, batch, seir: SEIRConfig, training: TrainingConfig):
    t_collocation, x_collocation, c_collocation, x_terminal, c_terminal = batch
    pde_values = jax.vmap(
        lambda t, x, c: pde_residual(
            model, t, x, c, seir, training
        )
    )(t_collocation, x_collocation, c_collocation)
    terminal_values = jax.vmap(
        lambda x, c: model(training.horizon, x, c)
    )(x_terminal, c_terminal)

    pde_loss = jnp.mean(pde_values**2)
    terminal_loss = jnp.mean(terminal_values**2)
    total_loss = pde_loss + training.terminal_weight * terminal_loss
    return total_loss, (pde_loss, terminal_loss)
