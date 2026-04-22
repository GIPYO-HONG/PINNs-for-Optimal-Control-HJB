import jax
import jax.numpy as jnp
import optax
import jaxopt
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np

# ==========================================
# 파라미터 & 설정
# ==========================================
LAYER_SIZES  = [6, 256, 256, 256, 256, 1]
LR_ADAM      = 5e-4
EPOCHS_ADAM  = 10000
EPOCHS_LBFGS = 10000
N_COLLOC     = 3000
N_TERM       = 3000

b = 0.525;  d = 0.5;   c = 0.002;  e = 0.5
g = 0.1;    a = 0.2;   A = 1.0;     T = 20.0

S0 = 1000;  E0 = 100;  I0 = 50;  R0 = 15
N0 = S0 + E0 + I0 + R0

STATE_MAX   = 2000.0
T_SCALE     = T           # 시간 정규화 상수
X_SCALE     = STATE_MAX   # 상태 정규화 상수
TERM_WEIGHT = 10.0       # terminal loss 가중치

# ==========================================
# 네트워크
# ==========================================

def init_network_params(sizes, key):
    keys = jax.random.split(key, len(sizes) - 1)
    return [
        (jax.random.normal(k, (m, n)) * jnp.sqrt(2.0 / m), jnp.zeros(n))
        for k, m, n in zip(keys, sizes[:-1], sizes[1:])
    ]

def value_net(params, t, S, E, I, R, N):
    # ✅ 입력 정규화: 모든 입력을 [0,1] 범위로 맞춤
    # → tanh gradient vanishing 방지, 학습 안정화
    x = jnp.array([
        t / T_SCALE,
        S / X_SCALE,
        E / X_SCALE,
        I / X_SCALE,
        R / X_SCALE,
        N / X_SCALE,
    ])
    for w, b_bias in params[:-1]:
        x = jnp.tanh(jnp.dot(x, w) + b_bias)
    fw, fb = params[-1]
    return (jnp.dot(x, fw) + fb).squeeze()

# ==========================================
# 자동미분
# ==========================================
V_t = jax.grad(value_net, argnums=1)
V_S = jax.grad(value_net, argnums=2)
V_E = jax.grad(value_net, argnums=3)
V_I = jax.grad(value_net, argnums=4)
V_R = jax.grad(value_net, argnums=5)
V_N = jax.grad(value_net, argnums=6)

# ==========================================
# HJB PDE 잔차
# ==========================================

def pde_residual(params, t, S, E, I, R, N):
    vt = V_t(params, t, S, E, I, R, N)
    vs = V_S(params, t, S, E, I, R, N)
    ve = V_E(params, t, S, E, I, R, N)
    vi = V_I(params, t, S, E, I, R, N)
    vr = V_R(params, t, S, E, I, R, N)
    vn = V_N(params, t, S, E, I, R, N)

    # u* = S*(V_S - V_R) / 2  (HJB 최적화 조건)
    u_star = jnp.clip(0.5 * S * (vs - vr), 0.0, 0.9)

    dsdt = b*N - d*S - c*S*I - u_star*S
    dedt = c*S*I - (e+d)*E
    didt = e*E - (g+a+d)*I
    drdt = g*I - d*R + u_star*S
    dndt = (b-d)*N - a*I

    running_cost = A*I + u_star**2

    residual = vt + vs*dsdt + ve*dedt + vi*didt + vr*drdt + vn*dndt + running_cost

    # ✅ 잔차 정규화: 상태값이 크면 잔차도 커지므로 X_SCALE로 나눔
    # → PDE loss가 수천 단위 → 수 단위로 감소
    return residual / X_SCALE

def terminal_residual(params, S, E, I, R, N):
    # V(T, x) = 0
    return value_net(params, T, S, E, I, R, N)

v_pde_res  = jax.vmap(pde_residual,      in_axes=(None,0,0,0,0,0,0))
v_term_res = jax.vmap(terminal_residual, in_axes=(None,0,0,0,0,0))

# ==========================================
# 손실 함수
# ==========================================

def total_loss(params, tc, sc, ec, ic, rc, nc, st, et, it, rt, nt):
    pde_l  = jnp.mean(v_pde_res (params, tc, sc, ec, ic, rc, nc) ** 2)
    term_l = jnp.mean(v_term_res(params, st, et, it, rt, nt) ** 2)
    return pde_l + TERM_WEIGHT * term_l

def split_losses(params, tc, sc, ec, ic, rc, nc, st, et, it, rt, nt):
    pde_l  = jnp.mean(v_pde_res (params, tc, sc, ec, ic, rc, nc) ** 2)
    term_l = jnp.mean(v_term_res(params, st, et, it, rt, nt) ** 2)
    return pde_l, term_l, pde_l + TERM_WEIGHT * term_l

# ==========================================
# 샘플링 — S, E, I, R, N 모두 독립 샘플링
# N은 독립 동역학 dN/dt = (b-d)N - aI 를 가짐
# ==========================================

def sample_points(key):
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, 11)

    tc = jax.random.uniform(keys[0],  (N_COLLOC,), minval=0.0, maxval=T)
    sc = jax.random.uniform(keys[1],  (N_COLLOC,), minval=0.0, maxval=STATE_MAX)
    ec = jax.random.uniform(keys[2],  (N_COLLOC,), minval=0.0, maxval=STATE_MAX)
    ic = jax.random.uniform(keys[3],  (N_COLLOC,), minval=0.0, maxval=STATE_MAX)
    rc = jax.random.uniform(keys[4],  (N_COLLOC,), minval=0.0, maxval=STATE_MAX)
    nc = jax.random.uniform(keys[5],  (N_COLLOC,), minval=0.0, maxval=STATE_MAX)

    st = jax.random.uniform(keys[6],  (N_TERM,), minval=0.0, maxval=STATE_MAX)
    et = jax.random.uniform(keys[7],  (N_TERM,), minval=0.0, maxval=STATE_MAX)
    it = jax.random.uniform(keys[8],  (N_TERM,), minval=0.0, maxval=STATE_MAX)
    rt = jax.random.uniform(keys[9],  (N_TERM,), minval=0.0, maxval=STATE_MAX)
    nt = jax.random.uniform(keys[10], (N_TERM,), minval=0.0, maxval=STATE_MAX)

    return key, tc, sc, ec, ic, rc, nc, st, et, it, rt, nt

# ==========================================
# 1단계: Adam + Cosine Decay
# ==========================================
# ✅ Cosine decay 재활성화 — 후반 fine-tuning 효과
lr_schedule = optax.cosine_decay_schedule(
    init_value  = LR_ADAM,
    decay_steps = EPOCHS_ADAM,
    alpha       = 1e-4
)
optimizer = optax.adam(lr_schedule)

key    = jax.random.PRNGKey(42)
params = init_network_params(LAYER_SIZES, key)
opt_state = optimizer.init(params)

@jax.jit
def adam_step(params, opt_state, tc, sc, ec, ic, rc, nc, st, et, it, rt, nt):
    loss_val, grads = jax.value_and_grad(total_loss)(
        params, tc, sc, ec, ic, rc, nc, st, et, it, rt, nt
    )
    updates, new_opt_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(params, updates), new_opt_state, loss_val

loss_hist = {"pde": [], "term": [], "total": []}

print("=" * 62)
print("[ Phase 1 ]  Adam  (Cosine Decay LR,  lr=5e-4)")
print("=" * 62)
print(f"{'Epoch':>6}  {'PDE Loss':>12}  {'Term Loss':>12}  {'Total':>12}")
print("-" * 62)

for epoch in range(EPOCHS_ADAM + 1):
    key, tc, sc, ec, ic, rc, nc, st, et, it, rt, nt = sample_points(key)
    params, opt_state, _ = adam_step(
        params, opt_state, tc, sc, ec, ic, rc, nc, st, et, it, rt, nt
    )

    if epoch % 500 == 0:
        pde_l, term_l, total_l = split_losses(
            params, tc, sc, ec, ic, rc, nc, st, et, it, rt, nt
        )
        loss_hist["pde"].append(float(pde_l))
        loss_hist["term"].append(float(term_l))
        loss_hist["total"].append(float(total_l))
        print(f"{epoch:>6}  {float(pde_l):>12.6f}  {float(term_l):>12.6f}  {float(total_l):>12.6f}")

adam_log_len = len(loss_hist["total"])
print("=" * 62)

# ==========================================
# 2단계: L-BFGS fine-tuning
# ==========================================
print("\n[ Phase 2 ]  L-BFGS  (Fine-tuning)")
print("=" * 62)
print(f"{'Step':>6}  {'PDE Loss':>12}  {'Term Loss':>12}  {'Total':>12}")
print("-" * 62)

key, tc_f, sc_f, ec_f, ic_f, rc_f, nc_f, st_f, et_f, it_f, rt_f, nt_f = sample_points(key)

def lbfgs_loss(params):
    return total_loss(params, tc_f, sc_f, ec_f, ic_f, rc_f, nc_f,
                               st_f, et_f, it_f, rt_f, nt_f)

lbfgs_solver = jaxopt.LBFGS(
    fun          = lbfgs_loss,
    maxiter      = EPOCHS_LBFGS,
    tol          = 1e-7,
    history_size = 50,
)
lbfgs_state  = lbfgs_solver.init_state(params)
lbfgs_update = jax.jit(lbfgs_solver.update)

for step in range(EPOCHS_LBFGS):
    params, lbfgs_state = lbfgs_update(params, lbfgs_state)
    if step % 200 == 0:
        pde_l, term_l, total_l = split_losses(
            params, tc_f, sc_f, ec_f, ic_f, rc_f, nc_f,
            st_f, et_f, it_f, rt_f, nt_f
        )
        loss_hist["pde"].append(float(pde_l))
        loss_hist["term"].append(float(term_l))
        loss_hist["total"].append(float(total_l))
        print(f"{step:>6}  {float(pde_l):>12.6f}  {float(term_l):>12.6f}  {float(total_l):>12.6f}")

print("=" * 62)
print("Training complete.\n")

# ==========================================
# ODE 시뮬레이션
# ==========================================

def compute_u(params_nn, t, S, E, I, R, N):
    """u*(t) 계산 — JAX 스칼라를 Python float으로 안전하게 변환"""
    t_j = jnp.float32(t)
    Sj  = jnp.float32(S);  Ej = jnp.float32(E)
    Ij  = jnp.float32(I);  Rj = jnp.float32(R);  Nj = jnp.float32(N)
    vs  = V_S(params_nn, t_j, Sj, Ej, Ij, Rj, Nj)
    vr  = V_R(params_nn, t_j, Sj, Ej, Ij, Rj, Nj)
    u_jax = jnp.clip(0.5 * Sj * (vs - vr), 0.0, 0.9)
    return float(jax.device_get(u_jax).item())

def seir_controlled(y, t, params_nn):
    S, E, I, R, N = y
    u = compute_u(params_nn, t, S, E, I, R, N)
    return [
        b*N - d*S - c*S*I - u*S,
        c*S*I - (e+d)*E,
        e*E - (g+a+d)*I,
        g*I - d*R + u*S,
        (b-d)*N - a*I,
    ]

def seir_uncontrolled(y, t):
    S, E, I, R, N = y
    return [
        b*N - d*S - c*S*I,
        c*S*I - (e+d)*E,
        e*E - (g+a+d)*I,
        g*I - d*R,
        (b-d)*N - a*I,
    ]

t_span = np.linspace(0, T, 300)
y0     = [S0, E0, I0, R0, N0]

print("Running ODE simulation...")
sol_ctrl   = odeint(seir_controlled,   y0, t_span, args=(params,))
sol_unctrl = odeint(seir_uncontrolled, y0, t_span)

u_opt = [compute_u(params, t_val, *sol_ctrl[i]) for i, t_val in enumerate(t_span)]
print("Simulation done.\n")

# ==========================================
# 시각화 1: 상태변수별 controlled vs uncontrolled
# ==========================================
fig, axes = plt.subplots(3, 2, figsize=(14, 14))
fig.suptitle("SEIR Optimal Control via PINN (HJB)  |  Adam → L-BFGS",
             fontsize=14, fontweight='bold', y=1.01)

state_cfg = [
    (0, 0, 'S', '#2196F3'),
    (0, 1, 'E', '#FF9800'),
    (1, 0, 'I', '#F44336'),
    (1, 1, 'R', '#4CAF50'),
    (2, 0, 'N', '#795548'),
]

for row, col, name, color in state_cfg:
    ax  = axes[row, col]
    idx = ['S','E','I','R','N'].index(name)
    ax.plot(t_span, sol_ctrl[:,idx],   color=color, lw=2.5,
            label=f'{name} (controlled)')
    ax.plot(t_span, sol_unctrl[:,idx], color=color, lw=2.5, ls='--', alpha=0.55,
            label=f'{name} (no control)')
    ax.fill_between(t_span, sol_ctrl[:,idx], sol_unctrl[:,idx],
                    alpha=0.12, color=color)
    ax.set_title(f"State  {name}:  Controlled vs Uncontrolled", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlabel("Time"); ax.set_ylabel(name)

# [2,1]: 최적 제어 u*(t)
axes[2, 1].remove()
ax_u = fig.add_subplot(3, 2, 6)
ax_u.plot(t_span, u_opt, color='#9C27B0', lw=2.5, label='u*(t)')
ax_u.axhline(0.9, color='gray', ls=':', lw=1.5, label='u_max = 0.9')
ax_u.axhline(0.0, color='gray', ls=':', lw=1.0)
ax_u.fill_between(t_span, 0, u_opt, alpha=0.15, color='#9C27B0')
ax_u.set_title("Optimal Vaccination Control  u*(t)", fontsize=11)
ax_u.set_xlabel("Time"); ax_u.set_ylabel("u*(t)")
ax_u.set_ylim(-0.05, 1.0); ax_u.legend(fontsize=9); ax_u.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("seir_pinn_states.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved → seir_pinn_states.png")

# ==========================================
# 시각화 2: Loss history
# ==========================================
fig2, ax_l = plt.subplots(figsize=(8, 4))
xs = np.arange(len(loss_hist["total"]))
ax_l.semilogy(xs, loss_hist["pde"],   color='#2196F3', lw=1.8, alpha=0.85, label='PDE Loss')
ax_l.semilogy(xs, loss_hist["term"],  color='#FF9800', lw=1.8, alpha=0.85, label='Terminal Loss')
ax_l.semilogy(xs, loss_hist["total"], color='#212121', lw=2.5,             label='Total Loss')
ax_l.axvline(adam_log_len - 0.5, color='red', ls='--', lw=1.5, label='→ L-BFGS start')
ax_l.set_title("Training Loss  (Adam → L-BFGS)", fontsize=12)
ax_l.legend(fontsize=9); ax_l.grid(alpha=0.3, which='both')
ax_l.set_xlabel("Log checkpoint"); ax_l.set_ylabel("Loss (log scale)")
plt.tight_layout()
plt.savefig("seir_pinn_loss.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved → seir_pinn_loss.png")