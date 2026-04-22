import jax
import jax.numpy as jnp
import optax
import jaxopt
import matplotlib.pyplot as plt

# ==========================================
# 1. 하이퍼파라미터 및 MLP 네트워크 정의
# ==========================================
LAYER_SIZES = [2, 64, 64, 64, 1] # (t, x) → hidden → V
LEARNING_RATE = 1e-3
ADAM_EPOCHS = 5000     # Adam 학습 횟수
LBFGS_MAXITER = 2000   # L-BFGS 최대 반복 횟수
T_MAX = 2.0

N_colloc = 4000
N_term = 1000

# Xavier 초기화
def init_network_params(sizes, key):
    keys = jax.random.split(key, len(sizes) - 1)
    return [
        (jax.random.normal(k, (m, n)) * jnp.sqrt(1.0 / m), jnp.zeros(n))
        for k, m, n in zip(keys, sizes[:-1], sizes[1:])
    ]

# MLP forward
def value_net(params, t, x):
    inputs = jnp.array([t, x])
    activations = inputs
    for w, b in params[:-1]:
        activations = jnp.tanh(jnp.dot(activations, w) + b)
    final_w, final_b = params[-1]
    return (jnp.dot(activations, final_w) + final_b).squeeze()

# ==========================================
# 2. Physics-Informed Loss
# ==========================================
V_t = jax.grad(value_net, argnums=1)
V_x = jax.grad(value_net, argnums=2)

def pde_residual(params, t, x):
    vt = V_t(params, t, x)
    vx = V_x(params, t, x)
    return vt + x + x*vx - 0.5*vx**2

def terminal_residual(params, x):
    return value_net(params, T_MAX, x)

v_pde_residual = jax.vmap(pde_residual, in_axes=(None, 0, 0))
v_terminal_residual = jax.vmap(terminal_residual, in_axes=(None, 0))

@jax.jit
def loss_fn(params, t_colloc, x_colloc, x_term):
    pde_loss = jnp.mean(v_pde_residual(params, t_colloc, x_colloc) ** 2)
    # 터미널 조건에 가중치를 주어 뒷부분이 흔들리지 않게 고정
    term_loss = jnp.mean(v_terminal_residual(params, x_term) ** 2)
    return pde_loss + term_loss

# ==========================================
# 3. Training (Adam + L-BFGS Hybrid)
# ==========================================
# 네트워크 파라미터 초기화
key = jax.random.PRNGKey(42)
params = init_network_params(LAYER_SIZES, key)

# --- 1단계: Adam Optimizer ---
optimizer_adam = optax.adam(LEARNING_RATE)
opt_state_adam = optimizer_adam.init(params)

@jax.jit
def adam_update(params, opt_state, t_colloc, x_colloc, x_term):
    loss, grads = jax.value_and_grad(loss_fn)(params, t_colloc, x_colloc, x_term)
    updates, opt_state = optimizer_adam.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

print("1단계: Adam 학습 시작...")
for epoch in range(ADAM_EPOCHS):
    # Adam은 매번 새로운 샘플을 뽑아서 범용적인 윤곽을 학습합니다.
    key, kt, kx, ktm = jax.random.split(key, 4)
    t_c = jax.random.uniform(kt, (N_colloc,), minval=0.0, maxval=T_MAX)
    # x(0) ≈ 2.69를 포함하기 위해 공간 범위를 -2.0 ~ 4.0으로 유지
    x_c = jax.random.uniform(kx, (N_colloc,), minval=-2.0, maxval=4.0)
    x_t = jax.random.uniform(ktm, (N_term,), minval=-2.0, maxval=4.0)

    params, opt_state_adam, loss = adam_update(params, opt_state_adam, t_c, x_c, x_t)

    if epoch % 500 == 0:
        print(f"Adam Epoch {epoch:4d} | Loss: {loss:.6f}")

# --- 2단계: L-BFGS Optimizer ---
print("\n2단계: L-BFGS 정밀 학습 시작...")
# 2차 미분 최적화는 '고정된' 데이터셋에 대해 수렴시키는 것이 효과적입니다.
key, kt, kx, ktm = jax.random.split(key, 4)
t_c_fixed = jax.random.uniform(kt, (N_colloc,), minval=0.0, maxval=T_MAX)
x_c_fixed = jax.random.uniform(kx, (N_colloc,), minval=-2.0, maxval=4.0)
x_t_fixed = jax.random.uniform(ktm, (N_term,), minval=-2.0, maxval=4.0)

def lbfgs_loss(p):
    return loss_fn(p, t_c_fixed, x_c_fixed, x_t_fixed)

# jaxopt를 이용한 L-BFGS 실행
lbfgs = jaxopt.LBFGS(fun=lbfgs_loss, maxiter=LBFGS_MAXITER)
res = lbfgs.run(params)
params = res.params  # 최종 정밀 튜닝된 파라미터 업데이트

print(f"L-BFGS 완료! 최종 Loss: {res.state.error:.6f}")


# ==========================================
# 4. 결과 확인 (Control u 비교)
# ==========================================
t_test = jnp.linspace(0.0, T_MAX, 100)

# 1. 실제 해석적 해 (Analytical Control)
u_analytical = 1.0 - jnp.exp(2.0 - t_test)

# 2. PINN 예측 해 계산을 위한 x 궤적
# x(0) = 0.5 * exp(2) - 1 약 2.69
x_star_path = 0.5 * jnp.exp(2.0 - t_test) - 1.0

def get_pinn_u(t, x):
    return -V_x(params, t, x)

u_pinn = jax.vmap(get_pinn_u)(t_test, x_star_path)

# 3. 시각화
plt.figure(figsize=(8, 5))

plt.plot(t_test, u_analytical, 'r--', label="Analytical $u^*(t)$", linewidth=2.5)
plt.plot(t_test, u_pinn, 'b-', label="PINN Predicted $u^*(t)$ ($-V_x$)", alpha=0.8)

plt.title("Comparison of Optimal Control $u^*$", fontsize=14)
plt.xlabel("Time $t$", fontsize=12)
plt.ylabel("Control $u$", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()