import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

# 1. 설정 (입력 차원: t, S, I, beta -> 4)
LAYER_SIZES = [4, 64, 64, 64, 1]
LEARNING_RATE = 1e-3
EPOCHS = 5000
T_MAX = 1.0
GAMMA = 0.5 # 회복률은 고정 (필요시 이것도 변수화 가능)

def init_network_params(sizes, key):
    keys = jax.random.split(key, len(sizes) - 1)
    return [(jax.random.normal(k, (m, n)) * jnp.sqrt(1.0 / m), jnp.zeros(n))
            for k, m, n in zip(keys, sizes[:-1], sizes[1:])]

def value_net(params, t, s, i, beta):
    # beta를 포함하여 4개의 입력을 하나의 벡터로 결합
    inputs = jnp.array([t, s, i, beta])
    activations = inputs
    for w, b in params[:-1]:
        activations = jnp.tanh(jnp.dot(activations, w) + b)
    final_w, final_b = params[-1]
    return (jnp.dot(activations, final_w) + final_b).squeeze()

# 2. Physics-Informed Loss
V_t = jax.grad(value_net, argnums=1)
V_S = jax.grad(value_net, argnums=2)
V_I = jax.grad(value_net, argnums=3)

def pde_residual(params, t, s, i, beta):
    vt = V_t(params, t, s, i, beta)
    vs = V_S(params, t, s, i, beta)
    vi = V_I(params, t, s, i, beta)
    
    # 최적 제어 u* (beta가 입력 변수이므로 식 안에서 그대로 사용)
    u_star = jnp.clip(beta * s * i * (vi - vs), 0.0, 1.0)
    
    running_cost = i**2 + 0.5 * u_star**2
    ds_dt = -beta * (1 - u_star) * s * i
    di_dt = beta * (1 - u_star) * s * i - GAMMA * i
    
    return vt + running_cost + vs * ds_dt + vi * di_dt

def terminal_residual(params, s, i, beta):
    return value_net(params, T_MAX, s, i, beta) - 0.5 * i**2

v_pde_residual = jax.vmap(pde_residual, in_axes=(None, 0, 0, 0, 0))
v_terminal_residual = jax.vmap(terminal_residual, in_axes=(None, 0, 0, 0))

@jax.jit
def loss_fn(params, tc, sc, ic, bc, st, it, bt):
    pde_loss = jnp.mean(v_pde_residual(params, tc, sc, ic, bc) ** 2)
    term_loss = jnp.mean(v_terminal_residual(params, st, it, bt) ** 2)
    return pde_loss + term_loss

# 3. Training
optimizer = optax.adam(LEARNING_RATE)
key = jax.random.PRNGKey(42)
params = init_network_params(LAYER_SIZES, key)
opt_state = optimizer.init(params)

@jax.jit
def update(params, opt_state, tc, sc, ic, bc, st, it, bt):
    loss, grads = jax.value_and_grad(loss_fn)(params, tc, sc, ic, bc, st, it, bt)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

print("Parametric SIR HJB 학습 시작...")
for epoch in range(EPOCHS + 1):
    key, k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 8)

    tc = jax.random.uniform(k1, shape=(1000,), minval=0.0, maxval=T_MAX)
    sc = jax.random.uniform(k2, shape=(1000,), minval=0.1, maxval=1.0)
    ic = jax.random.uniform(k3, shape=(1000,), minval=0.0, maxval=0.5)
    bc = jax.random.uniform(k4, shape=(1000,), minval=1.0, maxval=3.0)

    st = jax.random.uniform(k5, shape=(500,), minval=0.1, maxval=1.0)
    it = jax.random.uniform(k6, shape=(500,), minval=0.0, maxval=0.5)
    bt = jax.random.uniform(k7, shape=(500,), minval=1.0, maxval=3.0)
    
    params, opt_state, loss = update(params, opt_state, tc, sc, ic, bc, st, it, bt)
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

# 4. 결과 시각화 (서로 다른 감염률 beta에 따른 정책 비교)
i_range = jnp.linspace(0, 0.5, 100)
s_fixed = 0.8

plt.figure(figsize=(10, 6))
for test_beta in [1.5, 2.0, 2.5]:
    u_policy = jax.vmap(lambda i: jnp.clip(
        test_beta * s_fixed * i * (V_I(params, 0.0, s_fixed, i, test_beta) - V_S(params, 0.0, s_fixed, i, test_beta)), 
        0.0, 1.0))(i_range)
    plt.plot(i_range, u_policy, label=f"Beta = {test_beta}")

plt.title(f"Optimal Policy for Different Infection Rates (S={s_fixed}, t=0)")
plt.xlabel("Infected Ratio (I)")
plt.ylabel("Distancing Intensity (u)")
plt.legend()
plt.grid(True)
plt.show()