import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn

import equinox as eqx
import optax
import jaxopt

import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

T_MAX = 2.0

LR = 1e-4
ADAM_EPOCHS = 10000
LBFGS_MAXITER = 1000

N_COLLOC = 512
N_TERM = 128

key = jr.PRNGKey(0)


# ============================================================
# VALUE NETWORK
# ============================================================

class ValueNet(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, width_size, depth, *, key):

        self.mlp = eqx.nn.MLP(
            in_size=2,
            out_size=1,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            final_activation=lambda x: x,
            key=key
        )

    def __call__(self, t, x):

        inputs = jnp.array([t, x])
        return self.mlp(inputs).squeeze()


model = ValueNet(
    width_size=64,
    depth=4,
    key=jr.PRNGKey(42)
)


# ============================================================
# DERIVATIVES
# ============================================================

def V(model, t, x):
    return model(t, x)


V_t = jax.grad(V, argnums=1)
V_x = jax.grad(V, argnums=2)


# ============================================================
# HJB PDE
#
# Example:
#
# V_t + x + xV_x - 1/2 (V_x)^2 = 0
#
# terminal:
# V(T,x)=0
# ============================================================

def pde_residual(model, t, x):

    vt = V_t(model, t, x)
    vx = V_x(model, t, x)

    return vt + x + x * vx - 0.5 * vx**2


def terminal_residual(model, x):

    return model(T_MAX, x)


v_pde_residual = jax.vmap(
    pde_residual,
    in_axes=(None, 0, 0)
)

v_terminal_residual = jax.vmap(
    terminal_residual,
    in_axes=(None, 0)
)


# ============================================================
# LOSS
# ============================================================

def loss_fn(model, t_colloc, x_colloc, x_term):

    pde_loss = jnp.mean(
        v_pde_residual(model, t_colloc, x_colloc)**2
    )

    terminal_loss = jnp.mean(
        v_terminal_residual(model, x_term)**2
    )

    return pde_loss + terminal_loss


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optax.adam(LR)

opt_state = optimizer.init(
    eqx.filter(model, eqx.is_array)
)


@eqx.filter_value_and_grad
def compute_loss(model, t_c, x_c, x_t):

    return loss_fn(model, t_c, x_c, x_t)


@eqx.filter_jit
def train_step(model, opt_state, t_c, x_c, x_t):

    loss, grads = compute_loss(
        model,
        t_c,
        x_c,
        x_t
    )

    updates, opt_state = optimizer.update(
        grads,
        opt_state,
        model
    )

    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


# ============================================================
# ADAM TRAINING
# ============================================================

loss_history = []

for epoch in range(ADAM_EPOCHS):

    key, kt, kx, ktm = jr.split(key, 4)

    t_c = jr.uniform(
        kt,
        (N_COLLOC,),
        minval=0.0,
        maxval=T_MAX
    )

    x_c = jr.uniform(
        kx,
        (N_COLLOC,),
        minval=-2.0,
        maxval=4.0
    )

    x_t = jr.uniform(
        ktm,
        (N_TERM,),
        minval=-2.0,
        maxval=4.0
    )

    model, opt_state, loss = train_step(
        model,
        opt_state,
        t_c,
        x_c,
        x_t
    )

    loss_history.append(loss)

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Loss = {loss:.8f}")

# ============================================================
# L-BFGS
# ============================================================

params, static = eqx.partition(model, eqx.is_array)

t_c_fixed = t_c
x_c_fixed = x_c
x_t_fixed = x_t


def lbfgs_loss(params):

    model = eqx.combine(params, static)

    return loss_fn(
        model,
        t_c_fixed,
        x_c_fixed,
        x_t_fixed
    )


lbfgs = jaxopt.LBFGS(
    fun=lbfgs_loss,
    maxiter=LBFGS_MAXITER
)

res = lbfgs.run(params)

params = res.params

model = eqx.combine(params, static)

# ============================================================
# ANALYTICAL CONTROL
# ============================================================

t_test = jnp.linspace(0.0, T_MAX, 200)

u_true = 1.0 - jnp.exp(2.0 - t_test)

x_star = 0.5 * jnp.exp(2.0 - t_test) - 1.0


# ============================================================
# PINN CONTROL
#
# u* = -V_x
# ============================================================

def get_u_pinn(t, x):

    return -V_x(model, t, x)


u_pinn = jax.vmap(
    get_u_pinn
)(t_test, x_star)


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(
    t_test,
    u_true,
    'r--',
    linewidth=2.5,
    label="Analytical"
)

plt.plot(
    t_test,
    u_pinn,
    'b-',
    linewidth=2.0,
    label="PINN"
)

plt.xlabel("t")
plt.ylabel("u(t)")
plt.title("Optimal Control Comparison")

plt.legend()
plt.grid(alpha=0.3)

plt.show()