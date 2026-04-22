# import jax
# import jax.numpy as jnp
# import optax
# import numpy as np
# import matplotlib.pyplot as plt

# # ==========================================
# # 1. 하이퍼파라미터 및 파라미터 설정
# # ==========================================
# # 입력: (t, T, Ti, V) -> 4개 / 출력: 가치 함수 Value -> 1개
# LAYER_SIZES = [4, 64, 64, 64, 1]
# LEARNING_RATE = 1e-3
# EPOCHS = 100000
# T_FINAL = 20.0  # 치료 기간 (days)

# # HIV 모델 상수 (Lab 8 기준)
# s = 10.0; m1 = 0.02; m2 = 0.5; m3 = 4.4; r = 0.03; Tmax = 1500.0; k = 0.000024; N = 300.0; A = 0.05

# # Xavier 초기화
# def init_network_params(sizes, key):
#     keys = jax.random.split(key, len(sizes) - 1)
#     return [
#         (jax.random.normal(k, (m, n)) * jnp.sqrt(1.0 / m), jnp.zeros(n))
#         for k, m, n in zip(keys, sizes[:-1], sizes[1:])
#     ]

# # MLP forward (Value Function V(t, T, Ti, V))
# def value_net(params, t, T, Ti, V):
#     inputs = jnp.array([t, T, Ti, V])
#     activations = inputs
#     for w, b in params[:-1]:
#         activations = jnp.tanh(jnp.dot(activations, w) + b)
#     final_w, final_b = params[-1]
#     return (jnp.dot(activations, final_w) + final_b).squeeze()

# # ==========================================
# # 2. Physics-Informed HJB Loss
# # ==========================================
# # 각 상태 변수에 대한 편미분
# V_t  = jax.grad(value_net, argnums=1)
# V_T  = jax.grad(value_net, argnums=2)
# V_Ti = jax.grad(value_net, argnums=3)
# V_V  = jax.grad(value_net, argnums=4)

# def hjb_residual(params, t, T, Ti, V):
#     vt = V_t(params, t, T, Ti, V)
#     vT = V_T(params, t, T, Ti, V)
#     vTi = V_Ti(params, t, T, Ti, V)
#     vv = V_V(params, t, T, Ti, V)
    
#     # 1. 최적 제어 u* 도출 (Hamiltonian을 u에 대해 미분하여 0이 되는 지점)
#     # H = T - A(1-u)^2 + vT*(f_T) + vTi*(f_Ti) + vv*(f_V)
#     # dH/du = 2A(1-u) - vT*k*V*T + vTi*k*V*T = 0
#     # u* = 1 - ( (vTi - vT)*k*V*T ) / (2*A)
    
#     u_star = 1.0 - ((vTi - vT) * k * V * T) / (2.0 * A)
#     u_star = jnp.clip(u_star, 0.0, 1.0) # 제어 범위 제한
    
#     # 2. 시스템 동역학 (Dynamics)
#     f_T  = (s / (1.0 + V)) - m1 * T + r * T * (1.0 - (T + Ti) / Tmax) - u_star * k * V * T
#     f_Ti = u_star * k * V * T - m2 * Ti
#     f_V  = N * m2 * Ti - m3 * V
    
#     # 3. HJB 방정식: Vt + max_u( Running_Cost + Grad_V * Dynamics ) = 0
#     # 여기서는 Max 문제이므로 Vt + [T - A(1-u)^2] + vT*fT + vTi*fTi + vv*fV = 0
#     running_cost = T - A * (1.0 - u_star)**2
#     return vt + running_cost + vT * f_T + vTi * f_Ti + vv * f_V

# # 터미널 조건: V(T_final, T, Ti, V) = 0 (수업 모델상 끝나는 시점의 가치는 0으로 가정)
# def terminal_residual(params, T, Ti, V):
#     return value_net(params, T_FINAL, T, Ti, V) - 0.0

# v_hjb_residual = jax.vmap(hjb_residual, in_axes=(None, 0, 0, 0, 0))
# v_terminal_residual = jax.vmap(terminal_residual, in_axes=(None, 0, 0, 0))

# @jax.jit
# def loss_fn(params, t_col, T_col, Ti_col, V_col, T_ter, Ti_ter, V_ter):
#     hjb_loss = jnp.mean(v_hjb_residual(params, t_col, T_col, Ti_col, V_col) ** 2)
#     term_loss = jnp.mean(v_terminal_residual(params, T_ter, Ti_ter, V_ter) ** 2)
#     return hjb_loss + term_loss

# # ==========================================
# # 3. Training
# # ==========================================
# optimizer = optax.adam(LEARNING_RATE)

# @jax.jit
# def update(params, opt_state, t_col, T_col, Ti_col, V_col, T_ter, Ti_ter, V_ter):
#     loss, grads = jax.value_and_grad(loss_fn)(params, t_col, T_col, Ti_col, V_col, T_ter, Ti_ter, V_ter)
#     updates, opt_state = optimizer.update(grads, opt_state)
#     params = optax.apply_updates(params, updates)
#     return params, opt_state, loss

# # 초기화
# key = jax.random.PRNGKey(42)
# params = init_network_params(LAYER_SIZES, key)
# opt_state = optimizer.init(params)

# print("HIV 최적 제어 학습 시작...")

# for epoch in range(EPOCHS):
#     k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
    
#     # 데이터 샘플링 (정규화된 범위 고려)
#     t_c  = jax.random.uniform(k1, (1000,), minval=0.0, maxval=T_FINAL)
#     T_c  = jax.random.uniform(k2, (1000,), minval=200.0, maxval=1200.0)
#     Ti_c = jax.random.uniform(k3, (1000,), minval=0.0, maxval=50.0)
#     V_c  = jax.random.uniform(k4, (1000,), minval=0.0, maxval=100.0)
    
#     T_t  = jax.random.uniform(k5, (200,), minval=200.0, maxval=1200.0)
#     Ti_t = jax.random.uniform(k6, (200,), minval=0.0, maxval=50.0)
#     V_t_ = jax.random.uniform(k7, (200,), minval=0.0, maxval=100.0)

#     params, opt_state, loss = update(params, opt_state, t_c, T_c, Ti_c, V_c, T_t, Ti_t, V_t_)

#     if epoch % 500 == 0:
#         print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

# # ==========================================
# # 4. 결과 시뮬레이션 (간단한 예시)
# # ==========================================
# # 특정 T세포 농도에서 바이러스 양에 따른 최적 제어 u* 확인
# v_range = jnp.linspace(0.0, 0.05, 100)
# T_fixed, Ti_fixed = 800.0, 0.04
# t_fixed = 5.0

# def get_u_star(V):
#     vT = V_T(params, t_fixed, T_fixed, Ti_fixed, V)
#     vTi = V_Ti(params, t_fixed, T_fixed, Ti_fixed, V)
#     u = 1.0 - ((vTi - vT) * k * V * T_fixed) / (2.0 * A)
#     return jnp.clip(u, 0.0, 1.0)

# u_values = jax.vmap(get_u_star)(v_range)

# plt.figure(figsize=(8,5))
# plt.plot(v_range, u_values, label=f"Optimal Control $u^*$ at t={t_fixed}")
# plt.axhline(y=1, color='r', linestyle='--', label="No Treatment (u=1)")
# plt.axhline(y=0, color='g', linestyle='--', label="Max Treatment (u=0)")
# plt.xlabel("Virus Concentration (V)")
# plt.ylabel("Control $u$ (1=Off, 0=Max)")
# plt.title("Optimal Chemotherapy Strategy vs Virus Load")
# plt.legend()
# plt.grid(True)
# plt.show()

import jax
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 파라미터 및 하이퍼파라미터 설정
# ==========================================
LAYER_SIZES = [4, 64, 64, 64, 1]
LEARNING_RATE = 1e-3 # 안정적인 학습을 위해 조금 낮춤
EPOCHS = 100000 
T_FINAL = 20.0

# Lab 8 지정 파라미터
s = 10.0; m1 = 0.02; m2 = 0.5; m3 = 4.4; r = 0.03
Tmax = 1500.0; k = 0.000024; N = 300.0; A = 100

# 초기 조건 (시뮬레이션 및 초기 손실용)
T0_val, Ti0_val, V0_val = 800.0, 0.04, 1.5

def init_network_params(sizes, key):
    keys = jax.random.split(key, len(sizes) - 1)
    return [
        (jax.random.normal(k, (m, n)) * jnp.sqrt(2.0 / (m + n)), jnp.zeros(n))
        for k, m, n in zip(keys, sizes[:-1], sizes[1:])
    ]

def value_net(params, t, T, Ti, V):
    # 입력 정규화 (학습 효율 증진)
    inputs = jnp.array([t/T_FINAL, T/Tmax, Ti/50.0, V/100.0])
    activations = inputs
    for w, b in params[:-1]:
        activations = jnp.tanh(jnp.dot(activations, w) + b)
    final_w, final_b = params[-1]
    return (jnp.dot(activations, final_w) + final_b).squeeze()

# ==========================================
# 2. Physics-Informed HJB Loss
# ==========================================
V_t  = jax.grad(value_net, argnums=1)
V_T  = jax.grad(value_net, argnums=2)
V_Ti = jax.grad(value_net, argnums=3)
V_V  = jax.grad(value_net, argnums=4)

def get_optimal_u(params, t, T, Ti, V):
    vT = V_T(params, t, T, Ti, V)
    vTi = V_Ti(params, t, T, Ti, V)
    # HJB 유도식: u* = 1 - ((vTi - vT)*k*V*T)/(2A)
    u_star = 1.0 - ((vTi - vT) * k * V * T) / (2.0 * A)
    return jnp.clip(u_star, 0.0, 1.0)

def hjb_residual(params, t, T, Ti, V):
    vt = V_t(params, t, T, Ti, V)
    vT = V_T(params, t, T, Ti, V)
    vTi = V_Ti(params, t, T, Ti, V)
    vv = V_V(params, t, T, Ti, V)
    
    u = get_optimal_u(params, t, T, Ti, V)
    
    f_T  = (s / (1.0 + V)) - m1 * T + r * T * (1.0 - (T + Ti) / Tmax) - u * k * V * T
    f_Ti = u * k * V * T - m2 * Ti
    f_V  = N * m2 * Ti - m3 * V
    
    running_cost = T - A * (1.0 - u)**2
    # HJB: Vt + RunningCost + GradV * Dynamics = 0
    return vt + running_cost + vT * f_T + vTi * f_Ti + vv * f_V

@jax.jit
def loss_fn(params, t_c, T_c, Ti_c, V_c, T_ter, Ti_ter, V_ter):
    # 1. HJB 잔차 손실
    hjb_res = jax.vmap(hjb_residual, in_axes=(None, 0, 0, 0, 0))(params, t_c, T_c, Ti_c, V_c)
    hjb_loss = jnp.mean(hjb_res ** 2)
    
    # 2. 터미널 조건 손실 (t = T_FINAL 일 때 가치는 0)
    v_term = jax.vmap(value_net, in_axes=(None, None, 0, 0, 0))(params, T_FINAL, T_ter, Ti_ter, V_ter)
    term_loss = jnp.mean(v_term ** 2)
    
    return hjb_loss + term_loss

# ==========================================
# 3. Training Loop
# ==========================================
optimizer = optax.adam(LEARNING_RATE)
key = jax.random.PRNGKey(42)
params = init_network_params(LAYER_SIZES, key)
opt_state = optimizer.init(params)

@jax.jit
def update(params, opt_state, t_c, T_c, Ti_c, V_c, T_ter, Ti_ter, V_ter):
    loss, grads = jax.value_and_grad(loss_fn)(params, t_c, T_c, Ti_c, V_c, T_ter, Ti_ter, V_ter)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

print(f"HIV 최적 제어 학습 시작 (A={A}, N={N})...")

for epoch in range(EPOCHS + 1):
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
    
    # 데이터 샘플링 (정규화된 범위 고려)
    t_c  = jax.random.uniform(k1, (1000,), minval=0.0, maxval=T_FINAL)
    T_c  = jax.random.uniform(k2, (1000,), minval=200.0, maxval=1200.0)
    Ti_c = jax.random.uniform(k3, (1000,), minval=0.0, maxval=50.0)
    V_c  = jax.random.uniform(k4, (1000,), minval=0.0, maxval=100.0)
    
    T_ter  = jax.random.uniform(k5, (200,), minval=200.0, maxval=1200.0)
    Ti_ter = jax.random.uniform(k6, (200,), minval=0.0, maxval=50.0)
    V_ter = jax.random.uniform(k7, (200,), minval=0.0, maxval=100.0)

    params, opt_state, loss = update(params, opt_state, t_c, T_c, Ti_c, V_c, T_ter, Ti_ter, V_ter)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

# ==========================================
# 4. 결과 시각화 (바이러스 농도에 따른 제어 전략)
# ==========================================
# 고정된 상태에서 바이러스 농도 V가 변할 때 u*가 어떻게 변하는지 확인
v_test = jnp.linspace(0.0, 10.0, 100)
t_test = 0.0 # 초기 시점
T_test = 800.0
Ti_test = 0.04

u_opt_vals = jax.vmap(lambda v: get_optimal_u(params, t_test, T_test, Ti_test, v))(v_test)

plt.figure(figsize=(10, 5))
plt.plot(v_test, u_opt_vals, lw=2, label="Optimal $u^*$ (Control)")
plt.fill_between(v_test, 0, u_opt_vals, alpha=0.1, color='blue')
plt.axhline(y=1, color='r', ls='--', label="No Drug (u=1)")
plt.axhline(y=0, color='g', ls='--', label="Max Drug (u=0)")
plt.title(f"Optimal Control Policy (A={A}, T={T_test})")
plt.xlabel("Virus Concentration (V)")
plt.ylabel("Control $u$ (Lower is stronger treatment)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("분석: A값이 매우 낮으므로 바이러스가 조금만 존재해도 모델은 u=0(최대 치료)을 선택할 가능성이 높습니다.")