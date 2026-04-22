import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np

# ==========================================
# 파라미터
# ==========================================
b = 0.525; d = 0.5; c = 0.0001; e = 0.5
g = 0.1; a = 0.2; T = 20.0

S0 = 1000; E0 = 100; I0 = 50; R0 = 15
N0 = S0 + E0 + I0 + R0

t_span = np.linspace(0, T, 300)
y0 = [S0, E0, I0, R0, N0]

# ==========================================
# ODE
# ==========================================
def seir_uncontrolled(y, t):
    S, E, I, R, N = y
    return [
        b*N - d*S - c*S*I,
        c*S*I - (e+d)*E,
        e*E - (g+a+d)*I,
        g*I - d*R,
        (b-d)*N - a*I,
    ]

sol = odeint(seir_uncontrolled, y0, t_span)

# ==========================================
# 시각화
# ==========================================
state_cfg = [
    (0, 0, 'S', '#2196F3'),
    (0, 1, 'E', '#FF9800'),
    (1, 0, 'I', '#F44336'),
    (1, 1, 'R', '#4CAF50'),
    (2, 0, 'N', '#795548'),
]

fig, axes = plt.subplots(3, 2, figsize=(12, 12))
fig.suptitle("SEIR Uncontrolled", fontsize=14, fontweight='bold')

for row, col, name, color in state_cfg:
    ax  = axes[row, col]
    idx = ['S','E','I','R','N'].index(name)
    ax.plot(t_span, sol[:, idx], color=color, lw=2.5, label=name)
    ax.set_title(name, fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlabel("Time"); ax.set_ylabel(name)

axes[2, 1].axis('off')  # 빈 칸

plt.tight_layout()
# plt.savefig("seir_plot.png", dpi=150, bbox_inches='tight')
plt.show()
# print("Saved → seir_plot.png")