import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

# 1. 설정 (입력 차원: t, S, I -> 3)
LAYER_SIZES = [3, 64, 64, 64, 1]
LEARNING_RATE = 1e-3
EPOCHS = 5000
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
    
    running_cost = i**2 + 0.5 * u_star**2
    ds_dt = -BETA * (1 - u_star) * s * i
    di_dt = BETA * (1 - u_star) * s * i - GAMMA * i
    
    # HJB: V_t + L + V_S*S_dot + V_I*I_dot = 0
    return vt + running_cost + vs * ds_dt + vi * di_dt

def terminal_residual(params, s, i):
    # 터미널 코스트: 최종 시점의 감염자 수 최소화
    return value_net(params, T_MAX, s, i) - 0.5 * i**2

v_pde_residual = jax.vmap(pde_residual, in_axes=(None, 0, 0, 0))
v_terminal_residual = jax.vmap(terminal_residual, in_axes=(None, 0, 0))

@jax.jit
def loss_fn(params, tc, sc, ic, st, it):
    pde_loss = jnp.mean(v_pde_residual(params, tc, sc, ic) ** 2)
    term_loss = jnp.mean(v_terminal_residual(params, st, it) ** 2)
    return pde_loss + term_loss

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

# 4. 결과 시각화 (S=0.8일 때, I 변화에 따른 거리두기 강도)
i_range = jnp.linspace(0, 0.5, 100)
s_fixed = 0.8
u_policy = jax.vmap(lambda i: jnp.clip(BETA * s_fixed * i * (V_I(params, 0.0, s_fixed, i) - V_S(params, 0.0, s_fixed, i)), 0.0, 1.0))(i_range)

plt.plot(i_range, u_policy)
plt.title(f"Optimal Control Policy (at S={s_fixed}, t=0)")
plt.xlabel("Infected Ratio (I)")
plt.ylabel("Distancing Intensity (u)")
plt.grid(True)
plt.show()