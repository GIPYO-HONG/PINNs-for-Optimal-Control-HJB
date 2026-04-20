import jax
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 하이퍼파라미터 및 MLP 네트워크 정의
# ==========================================
LAYER_SIZES = [2, 64, 64, 64, 1]  # (t, x) → hidden → V
LEARNING_RATE = 1e-3
EPOCHS = 5000
T_MAX = 1.0

# Xavier 초기화 (tanh에 더 적합)
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
    # .squeeze()를 붙여서 (1,) 배열을 스칼라로 만듭니다.
    return (jnp.dot(activations, final_w) + final_b).squeeze()

# ==========================================
# 2. Physics-Informed Loss
# ==========================================
V_t = jax.grad(value_net, argnums=1)
V_x = jax.grad(value_net, argnums=2)

def pde_residual(params, t, x):
    vt = V_t(params, t, x)
    vx = V_x(params, t, x)
    return vt + 0.5 * x**2 - 0.5 * vx**2 + x * vx

def terminal_residual(params, x):
    return value_net(params, T_MAX, x) - 0.5 * x**2

v_pde_residual = jax.vmap(pde_residual, in_axes=(None, 0, 0))
v_terminal_residual = jax.vmap(terminal_residual, in_axes=(None, 0))

@jax.jit
def loss_fn(params, t_colloc, x_colloc, x_term):
    pde_loss = jnp.mean(v_pde_residual(params, t_colloc, x_colloc) ** 2)
    term_loss = jnp.mean(v_terminal_residual(params, x_term) ** 2)
    return pde_loss + term_loss

# ==========================================
# 3. Training
# ==========================================
optimizer = optax.adam(LEARNING_RATE)

@jax.jit
def update(params, opt_state, t_colloc, x_colloc, x_term):
    loss, grads = jax.value_and_grad(loss_fn)(
        params, t_colloc, x_colloc, x_term
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# 초기화
key = jax.random.PRNGKey(42)
params = init_network_params(LAYER_SIZES, key)
opt_state = optimizer.init(params)

N_colloc = 2000
N_term = 500

print("학습 시작...")

for epoch in range(EPOCHS):
    # 매 epoch마다 샘플 새로 생성 (중요!)
    key, kt, kx, ktm = jax.random.split(key, 4)

    # minval=, maxval= 을 붙여주세요.
    t_colloc = jax.random.uniform(kt, (N_colloc,), minval=0.0, maxval=T_MAX)
    x_colloc = jax.random.uniform(kx, (N_colloc,), minval=-2.0, maxval=2.0)
    x_term = jax.random.uniform(ktm, (N_term,), minval=-2.0, maxval=2.0)

    params, opt_state, loss = update(
        params, opt_state, t_colloc, x_colloc, x_term
    )

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

print("학습 완료!")

# ==========================================
# 4. 결과 확인
# ==========================================
x_test = jnp.linspace(-2.0, 2.0, 100)

u_optimal = jax.vmap(lambda x: -V_x(params, 0.5, x))(x_test)

plt.plot(x_test, u_optimal, label="Optimal Control $u^*$ at t=0.5")
plt.xlabel("State x")
plt.ylabel("Control u")
plt.legend()
plt.grid(True)
plt.show()