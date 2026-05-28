# ============================================================
# IMPORT
# ============================================================

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn

import equinox as eqx
import optax
import jaxopt

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint


# ============================================================
# SETTINGS
# ============================================================

T = 20.0

LR = 5e-4

ADAM_EPOCHS = 1000
LBFGS_MAXITER = 500

N_COLLOC = 3000
N_TERM = 3000

STATE_DIM = 5

STATE_MAX = 2000.0

TERM_WEIGHT = 10.0

key = jr.PRNGKey(42)


# ============================================================
# FIXED PARAMETERS
# ============================================================

d = 0.5
c = 0.002
e = 0.5

g = 0.1
a = 0.2

A = 1.0

# ============================================================
# BETA RANGE
#
# beta = infection parameter
# ============================================================

BETA_MIN = 0.2
BETA_MAX = 0.8


# ============================================================
# INITIAL CONDITION
# ============================================================

S0 = 1000.0
E0 = 100.0
I0 = 50.0
R0 = 15.0

N0 = S0 + E0 + I0 + R0


# ============================================================
# VALUE NETWORK
#
# input:
# [t, S,E,I,R,N, beta]
# ============================================================

class ValueNet(eqx.Module):

    mlp: eqx.nn.MLP

    def __init__(self, width_size, depth, *, key):

        self.mlp = eqx.nn.MLP(
            in_size=1 + STATE_DIM + 1,
            out_size=1,
            width_size=128,
            depth=4,
            activation=jnn.tanh,
            final_activation=lambda x: x,
            key=key
        )

    def __call__(self, t, x, beta):

        inputs = jnp.concatenate([

            jnp.array([
                t / T,
                beta
            ]),

            x / STATE_MAX

        ])

        return self.mlp(inputs).squeeze()


model = ValueNet(
    width_size=128,
    depth=4,
    key=key
)


# ============================================================
# VALUE FUNCTION
# ============================================================

def V(model, t, x, beta):

    return model(t, x, beta)


# ============================================================
# DERIVATIVES
# ============================================================

V_t = jax.grad(V, argnums=1)

V_x = jax.grad(V, argnums=2)


# ============================================================
# OPTIMAL CONTROL
# ============================================================

def optimal_u(model, t, x, beta):

    grad_x = V_x(model, t, x, beta)

    S = x[0]

    V_S = grad_x[0]
    V_R = grad_x[3]

    u = 0.5 * S * (V_S - V_R)

    return jnp.clip(u, 0.0, 0.9)


# ============================================================
# DYNAMICS
# ============================================================

def dynamics(model, t, x, beta):

    S, E, I, R, N = x

    u = optimal_u(model, t, x, beta)

    dS = beta * N - d * S - c * S * I - u * S

    dE = c * S * I - (e + d) * E

    dI = e * E - (g + a + d) * I

    dR = g * I - d * R + u * S

    dN = (beta - d) * N - a * I

    return jnp.array([
        dS,
        dE,
        dI,
        dR,
        dN
    ])


# ============================================================
# RUNNING COST
# ============================================================

def running_cost(model, t, x, beta):

    I = x[2]

    u = optimal_u(model, t, x, beta)

    return A * I + u**2


# ============================================================
# HJB PDE
# ============================================================

def pde_residual(model, t, x, beta):

    vt = V_t(model, t, x, beta)

    grad_x = V_x(model, t, x, beta)

    f = dynamics(model, t, x, beta)

    L = running_cost(model, t, x, beta)

    residual = (
        vt
        + jnp.dot(grad_x, f)
        + L
    )

    return residual / STATE_MAX


# ============================================================
# TERMINAL
# ============================================================

def terminal_residual(model, x, beta):

    return model(T, x, beta)


# ============================================================
# VMAP
# ============================================================

v_pde_residual = jax.vmap(
    pde_residual,
    in_axes=(None, 0, 0, 0)
)

v_terminal_residual = jax.vmap(
    terminal_residual,
    in_axes=(None, 0, 0)
)


# ============================================================
# LOSS
# ============================================================

def loss_fn(model, t_c, x_c, beta_c, x_t, beta_t):

    pde_loss = jnp.mean(
        v_pde_residual(
            model,
            t_c,
            x_c,
            beta_c
        )**2
    )

    terminal_loss = jnp.mean(
        v_terminal_residual(
            model,
            x_t,
            beta_t
        )**2
    )

    total_loss = (
        pde_loss
        + TERM_WEIGHT * terminal_loss
    )

    return total_loss, (pde_loss, terminal_loss)


# ============================================================
# SAMPLE BATCH
# ============================================================

def sample_batch(key):

    key, kt, kx, kb, kxt, kbt = jr.split(key, 6)

    t_c = jr.uniform(
        kt,
        (N_COLLOC,),
        minval=0.0,
        maxval=T
    )

    x_c = jr.uniform(
        kx,
        (N_COLLOC, STATE_DIM),
        minval=0.0,
        maxval=STATE_MAX
    )

    beta_c = jr.uniform(
        kb,
        (N_COLLOC,),
        minval=BETA_MIN,
        maxval=BETA_MAX
    )

    x_t = jr.uniform(
        kxt,
        (N_TERM, STATE_DIM),
        minval=0.0,
        maxval=STATE_MAX
    )

    beta_t = jr.uniform(
        kbt,
        (N_TERM,),
        minval=BETA_MIN,
        maxval=BETA_MAX
    )

    return (
        key,
        t_c,
        x_c,
        beta_c,
        x_t,
        beta_t
    )


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optax.adam(LR)

opt_state = optimizer.init(
    eqx.filter(model, eqx.is_array)
)


# ============================================================
# TRAIN STEP
# ============================================================

@eqx.filter_value_and_grad(has_aux=True)
def compute_loss(
    model,
    t_c,
    x_c,
    beta_c,
    x_t,
    beta_t
):

    return loss_fn(
        model,
        t_c,
        x_c,
        beta_c,
        x_t,
        beta_t
    )


@eqx.filter_jit
def train_step(
    model,
    opt_state,
    t_c,
    x_c,
    beta_c,
    x_t,
    beta_t
):

    (loss, aux), grads = compute_loss(
        model,
        t_c,
        x_c,
        beta_c,
        x_t,
        beta_t
    )

    updates, opt_state = optimizer.update(
        grads,
        opt_state,
        model
    )

    model = eqx.apply_updates(
        model,
        updates
    )

    return model, opt_state, loss, aux


# ============================================================
# TRAINING
# ============================================================

print("=" * 70)
print("Training parameter-conditioned HJB-PINN")
print("=" * 70)

for epoch in range(ADAM_EPOCHS + 1):

    (
        key,
        t_c,
        x_c,
        beta_c,
        x_t,
        beta_t
    ) = sample_batch(key)

    model, opt_state, loss, aux = train_step(
        model,
        opt_state,
        t_c,
        x_c,
        beta_c,
        x_t,
        beta_t
    )

    pde_loss, terminal_loss = aux

    if epoch % 500 == 0:

        print(
            f"Epoch {epoch:5d} | "
            f"PDE = {float(pde_loss):.6f} | "
            f"Terminal = {float(terminal_loss):.6f} | "
            f"Total = {float(loss):.6f}"
        )


# ============================================================
# L-BFGS
# ============================================================

params, static = eqx.partition(
    model,
    eqx.is_array
)

(
    key,
    t_c_fixed,
    x_c_fixed,
    beta_c_fixed,
    x_t_fixed,
    beta_t_fixed
) = sample_batch(key)


def lbfgs_loss(params):

    model = eqx.combine(params, static)

    loss, _ = loss_fn(
        model,
        t_c_fixed,
        x_c_fixed,
        beta_c_fixed,
        x_t_fixed,
        beta_t_fixed
    )

    return loss


lbfgs = jaxopt.LBFGS(
    fun=lbfgs_loss,
    maxiter=LBFGS_MAXITER
)

res = lbfgs.run(params)

params = res.params

model = eqx.combine(params, static)

print("L-BFGS complete.")


# ============================================================
# SIMULATION
# ============================================================

def controlled_ode(y, t, beta):

    x = jnp.array(y)

    dx = dynamics(model, t, x, beta)

    return np.array(dx)


# ============================================================
# TEST BETAS
# ============================================================

beta_list = [
    0.25,
    0.45,
    0.70
]

t_span = np.linspace(0.0, T, 300)

y0 = np.array([
    S0,
    E0,
    I0,
    R0,
    N0
])


# ============================================================
# VISUALIZATION
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10)
)

colors = [
    "#1E88E5",
    "#FB8C00",
    "#E53935"
]

for beta, color in zip(beta_list, colors):

    sol = odeint(
        controlled_ode,
        y0,
        t_span,
        args=(beta,)
    )

    S = sol[:, 0]
    E = sol[:, 1]
    I = sol[:, 2]
    R = sol[:, 3]

    u_values = []

    for i, t in enumerate(t_span):

        x = jnp.array(sol[i])

        u = optimal_u(
            model,
            t,
            x,
            beta
        )

        u_values.append(float(u))

    u_values = np.array(u_values)

    axes[0,0].plot(
        t_span,
        S,
        color=color,
        label=f"beta={beta}"
    )

    axes[0,1].plot(
        t_span,
        I,
        color=color,
        label=f"beta={beta}"
    )

    axes[1,0].plot(
        t_span,
        R,
        color=color,
        label=f"beta={beta}"
    )

    axes[1,1].plot(
        t_span,
        u_values,
        color=color,
        label=f"beta={beta}"
    )


axes[0,0].set_title("S(t)")
axes[0,1].set_title("I(t)")
axes[1,0].set_title("R(t)")
axes[1,1].set_title("u*(t)")

for ax in axes.flatten():

    ax.grid(alpha=0.3)

    ax.legend()

    ax.set_xlabel("Time")


plt.tight_layout()

plt.savefig(
    "parameter_conditioned_seir.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# VALUE FUNCTION SLICE
# ============================================================

S_grid = np.linspace(
    0.0,
    STATE_MAX,
    200
)

plt.figure(figsize=(8,5))

for beta, color in zip(beta_list, colors):

    vals = []

    for S in S_grid:

        x = jnp.array([
            S,
            E0,
            I0,
            R0,
            N0
        ])

        val = V(
            model,
            0.0,
            x,
            beta
        )

        vals.append(float(val))

    vals = np.array(vals)

    plt.plot(
        S_grid,
        vals,
        color=color,
        linewidth=2.5,
        label=f"beta={beta}"
    )

plt.title("Value Function Slice for Different beta")

plt.xlabel("S")

plt.ylabel("V(x,t,beta)")

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    "value_function_beta.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


print("\nSaved:")
print(" - parameter_conditioned_seir.png")
print(" - value_function_beta.png")