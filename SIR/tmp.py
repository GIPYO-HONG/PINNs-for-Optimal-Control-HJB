import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import jaxopt

# 1. 설정 (입력 차원: t, S, I -> 3)
LAYER_SIZES = [3, 64, 64, 64, 1]
LEARNING_RATE = 1e-3
EPOCHS = 10000
T_MAX = 1.0
BETA = 2.0  # 감염률
GAMMA = 0.5 # 회복률

def init_network_params(sizes, key):
    keys = jax.random.split(key, len(sizes) - 1)
    return [(jax.random.normal(k, (m, n)) * jnp.sqrt(1.0 / m), jnp.zeros(n))
            for k, m, n in zip(keys, sizes[:-1], sizes[1:])]

def value_net(params, t, s, i):
    inputs = jnp.array([t, s, i])
    activations = inputs
    for w, b in params[:-1]:
        activations = jnp.tanh(jnp.dot(activations, w) + b)
    final_w, final_b = params[-1]
    return (jnp.dot(activations, final_w) + final_b).squeeze()

# 2. Physics-Informed Loss
# 각 입력 변수에 대한 편미분 정의
V_t = jax.grad(value_net, argnums=1)
V_S = jax.grad(value_net, argnums=2)
V_I = jax.grad(value_net, argnums=3)

def pde_residual(params, t, s, i):
    vt = V_t(params, t, s, i)
    vs = V_S(params, t, s, i)
    vi = V_I(params, t, s, i)
    
    # 최적 제어 u* 도출 (Hamiltonian 최소화 조건)
    # u* = beta * S * I * (V_I - V_S)
    u_star = jnp.clip(BETA * s * i * (vi - vs), 0.0, 1.0)
    
    running_cost = i**2 + 0.01 * u_star**2
    ds_dt = -BETA * (1 - u_star) * s * i
    di_dt = BETA * (1 - u_star) * s * i - GAMMA * i
    
    # HJB: V_t + L + V_S*S_dot + V_I*I_dot = 0
    return vt + running_cost + vs * ds_dt + vi * di_dt

def terminal_residual(params, s, i):
    # 터미널 코스트: 최종 시점의 감염자 수 최소화
    return value_net(params, T_MAX, s, i)

v_pde_residual = jax.vmap(pde_residual, in_axes=(None, 0, 0, 0))
v_terminal_residual = jax.vmap(terminal_residual, in_axes=(None, 0, 0))

@jax.jit
def loss_fn(params, tc, sc, ic, st, it):
    pde_loss = jnp.mean(v_pde_residual(params, tc, sc, ic) ** 2)
    term_loss = jnp.mean(v_terminal_residual(params, st, it) ** 2)
    return pde_loss + 100*term_loss

# 3. Training
optimizer = optax.adam(LEARNING_RATE)
key = jax.random.PRNGKey(42)
params = init_network_params(LAYER_SIZES, key)
opt_state = optimizer.init(params)

@jax.jit
def update(params, opt_state, tc, sc, ic, st, it):
    loss, grads = jax.value_and_grad(loss_fn)(params, tc, sc, ic, st, it)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

print("SIR (S, I) Optimal Control 학습 시작...")
for epoch in range(EPOCHS + 1):
    # key를 분할하여 리스트로 저장
    key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)

    tc = jax.random.uniform(k1, (1000,), minval=0.0, maxval=T_MAX)
    sc = jax.random.uniform(k2, (1000,), minval=0.0, maxval=1.0)
    ic = jax.random.uniform(k3, (1000,), minval=0.0, maxval=1.0)
    st = jax.random.uniform(k4, (500,), minval=0.0, maxval=1.0)
    it = jax.random.uniform(k5, (500,), minval=0.0, maxval=1.0)
    
    params, opt_state, loss = update(params, opt_state, tc, sc, ic, st, it)
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

print("2단계: L-BFGS 정밀 학습으로 Control(u) 살리기...")
# 고정 데이터 생성
key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
tc_f = jax.random.uniform(k1, (2000,), minval=0.0, maxval=T_MAX)
sc_f = jax.random.uniform(k2, (2000,), minval=0.0, maxval=1.0)
ic_f = jax.random.uniform(k3, (2000,), minval=0.0, maxval=1.0)
st_f = jax.random.uniform(k4, (1000,), minval=0.0, maxval=1.0)
it_f = jax.random.uniform(k5, (1000,), minval=0.0, maxval=1.0)

def lbfgs_loss(p):
    return loss_fn(p, tc_f, sc_f, ic_f, st_f, it_f)

lbfgs = jaxopt.LBFGS(fun=lbfgs_loss, maxiter=1000)
res = lbfgs.run(params)
params = res.params

print(f"학습 종료! 최종 Loss: {res.state.error:.6f}")

# # 4. 결과 시각화 (S=0.8일 때, I 변화에 따른 거리두기 강도)
# i_range = jnp.linspace(0, 0.5, 100)
# s_fixed = 0.8
# u_policy = jax.vmap(lambda i: jnp.clip(BETA * s_fixed * i * (V_I(params, 0.0, s_fixed, i) - V_S(params, 0.0, s_fixed, i)), 0.0, 1.0))(i_range)

# plt.plot(i_range, u_policy)
# plt.title(f"Optimal Control Policy (at S={s_fixed}, t=0)")
# plt.xlabel("Infected Ratio (I)")
# plt.ylabel("Distancing Intensity (u)")
# plt.grid(True)
# plt.show()

from scipy.integrate import odeint

# 최적 제어 함수 (학습된 params 사용)
def optimal_u(s, i, t):
    vi = V_I(params, t, s, i)
    vs = V_S(params, t, s, i)
    return jnp.clip(BETA * s * i * (vi - vs), 0.0, 1.0)

# 최적 제어가 적용된 시스템 미분 방정식
def system_dynamics(state, t):
    s, i = state
    u = optimal_u(s, i, t)
    dsdt = -BETA * (1 - u) * s * i
    didt = BETA * (1 - u) * s * i - GAMMA * i
    return [dsdt, didt]

# # 초기값에서부터 시뮬레이션 수행
# t_points = jnp.linspace(0, T_MAX, 100)
# results = odeint(system_dynamics, y0=[0.99, 0.01], t=t_points)

# # 결과 시각화
# plt.plot(t_points, results[:, 1], label="Infected (I)")
# plt.plot(t_points, results[:, 0], label="Susceptible (S)")
# plt.legend()
# plt.show()

# # ==========================================
# # 4. 결과 시각화 및 비교 분석
# # ==========================================

# # --- [준비] 비교를 위한 일반 SIR 모델 (방역 없음, u=0) ---
def vanilla_sir(state, t):
    s, i = state
    dsdt = -BETA * s * i
    didt = BETA * s * i - GAMMA * i
    return [dsdt, didt]

t_points = jnp.linspace(0, T_MAX, 100)
initial_state = [0.99, 0.01]  # S0=0.99, I0=0.01

# # ODE 시뮬레이션 (최적 제어 적용 vs 미적용)
results_pinn = odeint(system_dynamics, y0=initial_state, t=t_points)
results_vanilla = odeint(vanilla_sir, y0=initial_state, t=t_points)

# # 시뮬레이션 경로 상의 최적 제어량(u) 계산
u_history = [optimal_u(s, i, t) for s, i, t in zip(results_pinn[:,0], results_pinn[:,1], t_points)]

# # --- [그래프 1] 시간에 따른 상태 변화 비교 (S, I) ---
# plt.figure(figsize=(14, 5))

# plt.subplot(1, 2, 1)
# plt.plot(t_points, results_vanilla[:, 1], 'r--', alpha=0.5, label="Infected (No Control)")
# plt.plot(t_points, results_pinn[:, 1], 'r-', linewidth=2, label="Infected (PINN Control)")
# plt.plot(t_points, results_vanilla[:, 0], 'b--', alpha=0.5, label="Susceptible (No Control)")
# plt.plot(t_points, results_pinn[:, 0], 'b-', linewidth=2, label="Susceptible (PINN Control)")
# plt.title("SIR Dynamics: Controlled vs Uncontrolled", fontsize=12)
# plt.xlabel("Time (t)")
# plt.ylabel("Population Ratio")
# plt.legend()
# plt.grid(True, linestyle=':', alpha=0.7)

# # --- [그래프 2] 시간에 따른 최적 방역 강도 (u) ---
# plt.subplot(1, 2, 2)
# plt.fill_between(t_points, u_history, color='green', alpha=0.2)
# plt.plot(t_points, u_history, 'g-', linewidth=2, label="Optimal Policy $u^*(t)$")
# plt.title("Optimal Control Intensity over Time", fontsize=12)
# plt.xlabel("Time (t)")
# plt.ylabel("Distancing Intensity (u)")
# plt.ylim(-0.05, 1.05)
# plt.legend()
# plt.grid(True, linestyle=':', alpha=0.7)

# plt.tight_layout()
# plt.show()

# # --- [그래프 3] 정책 맵 (Policy Map): I 변화에 따른 u의 민감도 ---
# plt.figure(figsize=(8, 5))
# s_samples = [0.9, 0.7, 0.5]  # 다양한 S 상황 가정
# i_range = jnp.linspace(0, 0.5, 100)

# for s_val in s_samples:
#     u_vals = jax.vmap(lambda i: optimal_u(s_val, i, 0.0))(i_range)
#     plt.plot(i_range, u_vals, label=f"at S={s_val}")

# plt.title("Optimal Control Policy $u^*$ by Infection Level", fontsize=13)
# plt.xlabel("Current Infected Ratio (I)")
# plt.ylabel("Distancing Intensity (u)")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

# ==========================================
# 4. 결과 시각화 (S와 I를 각각 분리하여 비교)
# ==========================================

plt.figure(figsize=(15, 10))

# --- [그래프 1] 감염자 수 변화 (Infected, I) ---
# 방역의 핵심 목적인 '피크 감소(Flattening the Curve)'를 확인
plt.subplot(2, 2, 1)
plt.plot(t_points, results_vanilla[:, 1], 'r--', alpha=0.6, label="I (No Control)")
plt.plot(t_points, results_pinn[:, 1], 'r-', linewidth=2.5, label="I (PINN Control)")
plt.fill_between(t_points, results_vanilla[:, 1], results_pinn[:, 1], color='red', alpha=0.1, label="Reduction")
plt.title("Infected Population (I): Flattening the Curve", fontsize=13)
plt.xlabel("Time (t)")
plt.ylabel("Ratio of Population")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# --- [그래프 2] 감수성자 수 변화 (Susceptible, S) ---
# 방역을 통해 감염 속도가 늦춰지면서 S의 감소가 완만해지는지 확인
plt.subplot(2, 2, 2)
plt.plot(t_points, results_vanilla[:, 0], 'b--', alpha=0.6, label="S (No Control)")
plt.plot(t_points, results_pinn[:, 0], 'b-', linewidth=2.5, label="S (PINN Control)")
plt.title("Susceptible Population (S): Slowing Infection", fontsize=13)
plt.xlabel("Time (t)")
plt.ylabel("Ratio of Population")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# --- [그래프 3] 최적 방역 강도 (Optimal Control, u) ---
# 시간에 따라 시스템이 제안하는 거리두기 강도
plt.subplot(2, 2, 3)
plt.fill_between(t_points, u_history, color='green', alpha=0.2)
plt.plot(t_points, u_history, 'g-', linewidth=2, label="Optimal Policy $u^*(t)$")
plt.title("Optimal Distancing Intensity ($u$)", fontsize=13)
plt.xlabel("Time (t)")
plt.ylabel("Intensity")
plt.ylim(-0.05, 1.05)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# --- [그래프 4] 위상 평면 (Phase Plane: S-I Plane) ---
plt.subplot(2, 2, 4)
plt.plot(results_vanilla[:, 0], results_vanilla[:, 1], 'k--', alpha=0.5, label="Trajectory (No Control)")
plt.plot(results_pinn[:, 0], results_pinn[:, 1], 'm-', linewidth=2, label="Trajectory (PINN Control)")
plt.title("Phase Plane (S vs I)", fontsize=13)
plt.xlabel("Susceptible (S)")
plt.ylabel("Infected (I)")

# 수정된 부분: plt 대신 plt.gca() 사용
# plt.gca().invert_xaxis()  # S는 시간이 갈수록 감소하므로 축을 뒤집어 시간 흐름 시각화

plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()