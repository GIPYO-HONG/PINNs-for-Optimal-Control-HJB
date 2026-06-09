import equinox as eqx
import jaxopt
import optax

from .config import ExperimentConfig, SEIRConfig, TrainingConfig
from .problem import loss_fn
from .sampling import sample_batch


def train(
    model,
    key,
    seir: SEIRConfig,
    training: TrainingConfig,
    experiment: ExperimentConfig,
):
    optimizer = optax.adam(training.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    history = {"pde": [], "terminal": [], "total": []}

    @eqx.filter_value_and_grad(has_aux=True)
    def compute_loss(current_model, batch):
        return loss_fn(current_model, batch, seir, training)

    @eqx.filter_jit
    def train_step(current_model, current_opt_state, batch):
        (loss, aux), grads = compute_loss(current_model, batch)
        updates, next_opt_state = optimizer.update(
            grads, current_opt_state, current_model
        )
        next_model = eqx.apply_updates(current_model, updates)
        return next_model, next_opt_state, loss, aux

    print("=" * 70)
    print("[ Phase 1 ] Adam Training")
    print("=" * 70)

    for epoch in range(training.adam_epochs + 1):
        key, batch = sample_batch(key, training, experiment)
        model, opt_state, loss, losses = train_step(model, opt_state, batch)

        if epoch % training.log_every == 0:
            pde_loss, terminal_loss = losses
            values = {
                "pde": float(pde_loss),
                "terminal": float(terminal_loss),
                "total": float(loss),
            }
            for name, value in values.items():
                history[name].append(value)
            print(
                f"Epoch {epoch:5d} | "
                f"PDE = {values['pde']:.6e} | "
                f"Terminal = {values['terminal']:.6e} | "
                f"Total = {values['total']:.6e}"
            )

    print("\n" + "=" * 70)
    print("[ Phase 2 ] L-BFGS Fine-Tuning")
    print("=" * 70)

    lbfgs_batches = []
    for _ in range(training.lbfgs_batches):
        key, batch = sample_batch(key, training, experiment)
        lbfgs_batches.append(batch)

    params, static = eqx.partition(model, eqx.is_array)

    def lbfgs_loss(current_params):
        current_model = eqx.combine(current_params, static)
        total = sum(
            loss_fn(current_model, batch, seir, training)[0]
            for batch in lbfgs_batches
        )
        return total / training.lbfgs_batches

    solution = jaxopt.LBFGS(
        fun=lbfgs_loss,
        maxiter=training.lbfgs_maxiter,
        tol=1e-7,
        history_size=50,
    ).run(params)
    model = eqx.combine(solution.params, static)
    print(f"L-BFGS complete. Final Loss = {float(solution.state.value):.8e}")
    return model, history
