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

ADAM_EPOCHS = 10000
LBFGS_MAXITER = 5000

N_COLLOC = 3000
N_TERM = 3000

STATE_DIM = 5

STATE_MAX = 2000.0

TERM_WEIGHT = 10.0

key = jr.PRNGKey(42)


# ============================================================
# MODEL PARAMETERS
# ============================================================

b = 0.525
d = 0.5
c = 0.002
e = 0.5

g = 0.1
a = 0.2

A = 1.0

S0 = 1000.0
E0 = 100.0
I0 = 50.0
R0 = 15.0

N0 = S0 + E0 + I0 + R0


# ============================================================
# VALUE NETWORK
# ============================================================

class ValueNet(eqx.Module):

    mlp: eqx.nn.MLP

    def __init__(self, width_size, depth, *, key):

        self.mlp = eqx.nn.MLP(
            in_size=1 + STATE_DIM,
            out_size=1,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            final_activation=lambda x: x,
            key=key
        )

    def __call__(self, t, x):

        inputs = jnp.concatenate([
            jnp.array([t / T]),
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

def V(model, t, x):

    return model(t, x)


# ============================================================
# DERIVATIVES
# ============================================================

V_t = jax.grad(V, argnums=1)

V_x = jax.grad(V, argnums=2)


# ============================================================
# OPTIMAL CONTROL
#
# u* = 0.5 * S * (V_S - V_R)
# ============================================================

def optimal_u(model, t, x):

    grad_x = V_x(model, t, x)

    S = x[0]

    V_S = grad_x[0]
    V_R = grad_x[3]

    u = 0.5 * S * (V_S - V_R)

    return jnp.clip(u, 0.0, 0.9)


# ============================================================
# DYNAMICS
# ============================================================

def dynamics(model, t, x):

    S, E, I, R, N = x

    u = optimal_u(model, t, x)

    dS = b * N - d * S - c * S * I - u * S

    dE = c * S * I - (e + d) * E

    dI = e * E - (g + a + d) * I

    dR = g * I - d * R + u * S

    dN = (b - d) * N - a * I

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

def running_cost(model, t, x):

    I = x[2]

    u = optimal_u(model, t, x)

    return A * I + u**2


# ============================================================
# HJB PDE RESIDUAL
# ============================================================

def pde_residual(model, t, x):

    vt = V_t(model, t, x)

    grad_x = V_x(model, t, x)

    f = dynamics(model, t, x)

    L = running_cost(model, t, x)

    residual = (
        vt
        + jnp.dot(grad_x, f)
        + L
    )

    return residual / STATE_MAX


# ============================================================
# TERMINAL CONDITION
#
# V(T,x)=0
# ============================================================

def terminal_residual(model, x):

    return model(T, x)


# ============================================================
# VMAP
# ============================================================

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

def loss_fn(model, t_c, x_c, x_t):

    pde_loss = jnp.mean(
        v_pde_residual(model, t_c, x_c)**2
    )

    terminal_loss = jnp.mean(
        v_terminal_residual(model, x_t)**2
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

    key, kt, kx, kxt = jr.split(key, 4)

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

    x_t = jr.uniform(
        kxt,
        (N_TERM, STATE_DIM),
        minval=0.0,
        maxval=STATE_MAX
    )

    return key, t_c, x_c, x_t


# ============================================================
# OPTIMIZER
# ============================================================

lr_schedule = optax.cosine_decay_schedule(
    init_value=LR,
    decay_steps=ADAM_EPOCHS,
    alpha=1e-4
)

optimizer = optax.adam(lr_schedule)

opt_state = optimizer.init(
    eqx.filter(model, eqx.is_array)
)


# ============================================================
# TRAIN STEP
# ============================================================

@eqx.filter_value_and_grad(has_aux=True)
def compute_loss(model, t_c, x_c, x_t):

    return loss_fn(model, t_c, x_c, x_t)


@eqx.filter_jit
def train_step(model, opt_state, t_c, x_c, x_t):

    (loss, aux), grads = compute_loss(
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

    model = eqx.apply_updates(
        model,
        updates
    )

    return model, opt_state, loss, aux


# ============================================================
# TRAINING
# ============================================================

loss_history = {
    "pde": [],
    "terminal": [],
    "total": []
}

print("=" * 70)
print("[ Phase 1 ] Adam Training")
print("=" * 70)

for epoch in range(ADAM_EPOCHS + 1):

    key, t_c, x_c, x_t = sample_batch(key)

    model, opt_state, loss, aux = train_step(
        model,
        opt_state,
        t_c,
        x_c,
        x_t
    )

    pde_loss, terminal_loss = aux

    if epoch % 500 == 0:

        print(
            f"Epoch {epoch:5d} | "
            f"PDE = {float(pde_loss):.6f} | "
            f"Terminal = {float(terminal_loss):.6f} | "
            f"Total = {float(loss):.6f}"
        )

        loss_history["pde"].append(float(pde_loss))
        loss_history["terminal"].append(float(terminal_loss))
        loss_history["total"].append(float(loss))


# ============================================================
# L-BFGS
# ============================================================

print("\n" + "=" * 70)
print("[ Phase 2 ] L-BFGS Fine-Tuning")
print("=" * 70)

params, static = eqx.partition(
    model,
    eqx.is_array
)

key, t_c_fixed, x_c_fixed, x_t_fixed = sample_batch(key)


def lbfgs_loss(params):

    model = eqx.combine(params, static)

    loss, _ = loss_fn(
        model,
        t_c_fixed,
        x_c_fixed,
        x_t_fixed
    )

    return loss


lbfgs = jaxopt.LBFGS(
    fun=lbfgs_loss,
    maxiter=LBFGS_MAXITER,
    tol=1e-7,
    history_size=50
)

res = lbfgs.run(params)

params = res.params

model = eqx.combine(params, static)

print("L-BFGS complete.")


# ============================================================
# CONTROLLED DYNAMICS
# ============================================================

def controlled_ode(y, t):

    x = jnp.array(y)

    dx = dynamics(model, t, x)

    return np.array(dx)


def uncontrolled_ode(y, t):

    S, E, I, R, N = y

    dS = b * N - d * S - c * S * I

    dE = c * S * I - (e + d) * E

    dI = e * E - (g + a + d) * I

    dR = g * I - d * R

    dN = (b - d) * N - a * I

    return [
        dS,
        dE,
        dI,
        dR,
        dN
    ]


# ============================================================
# SIMULATION
# ============================================================

t_span = np.linspace(0.0, T, 300)

y0 = np.array([
    S0,
    E0,
    I0,
    R0,
    N0
])

print("\nRunning simulation...")

sol_ctrl = odeint(
    controlled_ode,
    y0,
    t_span
)

sol_unctrl = odeint(
    uncontrolled_ode,
    y0,
    t_span
)

print("Simulation complete.")


# ============================================================
# CONTROL TRAJECTORY
# ============================================================

u_values = []

for i, t in enumerate(t_span):

    x = jnp.array(sol_ctrl[i])

    u = optimal_u(model, t, x)

    u_values.append(float(u))


u_values = np.array(u_values)


# ============================================================
# PLOT 1
#
# STATE TRAJECTORIES
# ============================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(14, 14)
)

fig.suptitle(
    "SEIR Optimal Control via HJB-PINN",
    fontsize=16,
    fontweight="bold"
)

state_info = [
    ("S", 0, "#1E88E5"),
    ("E", 1, "#FB8C00"),
    ("I", 2, "#E53935"),
    ("R", 3, "#43A047"),
    ("N", 4, "#6D4C41"),
]

positions = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (2, 0),
]

for (name, idx, color), (r, c_) in zip(state_info, positions):

    ax = axes[r, c_]

    ax.plot(
        t_span,
        sol_ctrl[:, idx],
        color=color,
        linewidth=2.5,
        label=f"{name} Controlled"
    )

    ax.plot(
        t_span,
        sol_unctrl[:, idx],
        linestyle="--",
        color=color,
        alpha=0.5,
        linewidth=2.0,
        label=f"{name} Uncontrolled"
    )

    ax.fill_between(
        t_span,
        sol_ctrl[:, idx],
        sol_unctrl[:, idx],
        alpha=0.12,
        color=color
    )

    ax.set_title(f"{name}(t)")

    ax.set_xlabel("Time")

    ax.set_ylabel(name)

    ax.grid(alpha=0.3)

    ax.legend()


# ============================================================
# CONTROL PLOT
# ============================================================

axes[2, 1].remove()

ax_u = fig.add_subplot(3, 2, 6)

ax_u.plot(
    t_span,
    u_values,
    color="#8E24AA",
    linewidth=2.5,
    label="Optimal Control"
)

ax_u.fill_between(
    t_span,
    0.0,
    u_values,
    alpha=0.15,
    color="#8E24AA"
)

ax_u.axhline(
    0.9,
    color="gray",
    linestyle=":"
)

ax_u.set_ylim(-0.05, 1.0)

ax_u.set_title("Optimal Vaccination Control")

ax_u.set_xlabel("Time")

ax_u.set_ylabel("u(t)")

ax_u.grid(alpha=0.3)

ax_u.legend()

plt.tight_layout()

plt.savefig(
    "seir_states.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PLOT 2
#
# LOSS HISTORY
# ============================================================

fig2, ax2 = plt.subplots(
    figsize=(8, 4)
)

xs = np.arange(
    len(loss_history["total"])
)

ax2.semilogy(
    xs,
    loss_history["pde"],
    linewidth=2,
    label="PDE Loss"
)

ax2.semilogy(
    xs,
    loss_history["terminal"],
    linewidth=2,
    label="Terminal Loss"
)

ax2.semilogy(
    xs,
    loss_history["total"],
    linewidth=3,
    label="Total Loss"
)

ax2.set_title("Training Loss")

ax2.set_xlabel("Checkpoint")

ax2.set_ylabel("Loss")

ax2.grid(alpha=0.3)

ax2.legend()

plt.tight_layout()

plt.savefig(
    "loss_history.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# VALUE FUNCTION VISUALIZATION
#
# Fix:
# E,I,R,N
#
# visualize:
# V(t,S)
# ============================================================

S_grid = np.linspace(
    0.0,
    STATE_MAX,
    200
)

t_fixed = 0.0

E_fixed = E0
I_fixed = I0
R_fixed = R0
N_fixed = N0

V_values = []

for S in S_grid:

    x = jnp.array([
        S,
        E_fixed,
        I_fixed,
        R_fixed,
        N_fixed
    ])

    val = V(
        model,
        t_fixed,
        x
    )

    V_values.append(float(val))

V_values = np.array(V_values)

plt.figure(figsize=(8,5))

plt.plot(
    S_grid,
    V_values,
    linewidth=2.5
)

plt.title("Value Function Slice")

plt.xlabel("Susceptible Population S")

plt.ylabel("V(t,x)")

plt.grid(alpha=0.3)

plt.tight_layout()

# plt.savefig(
#     "value_function_slice.png",
#     dpi=150,
#     bbox_inches="tight"
# )

plt.show()


# print("\nSaved:")
# print(" - seir_states.png")
# print(" - loss_history.png")
# print(" - value_function_slice.png")