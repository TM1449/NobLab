import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from tqdm import tqdm

# --- 論文パラメータ（Table 1）---
tau_E, tau_I = 0.125, 0.25      # 時定数 [s]
theta_E, theta_I = 2.0, 8.0     # 閾値
a_E, a_I = 0.8, 0.8             # シグモイド傾き
c_EE, c_EI = 8.0, 16.0          # 重み
c_IE, c_II = 8.0, 0.4           # 重み

# --- シミュレーション設定 ---
dt      = 1e-3   # 時間刻み [s]
T_total = 200.0  # 総時間 [s]
steps   = int(T_total / dt)
time    = np.arange(steps) * dt

# --- ネットワーク構築（Watts–Strogatz 小世界）---
N       = 80     # ノード数
K_WS    = 20     # 各ノードの近傍数（両側で計 K_WS/2）
p       = 0.2    # リワイヤ確率
G       = nx.watts_strogatz_graph(n=N, k=K_WS, p=p)
M       = nx.to_numpy_array(G)  # 隣接行列 Mij ∈ {0,1} :contentReference[oaicite:1]{index=1}
np.fill_diagonal(M, 0)          # 自己結合を除去

# --- シグモイド（閾値は入力 u で差し引く）---
def S_E(x):
    return 1.0 / (1.0 + np.exp(-a_E * x))
def S_I(x):
    return 1.0 / (1.0 + np.exp(-a_I * x))

# --- 外部入力：ランダムノイズ例 ---
P = np.random.normal(loc=1.0, scale=0.3, size=steps)  # E への入力
Q = np.random.normal(loc=0.0, scale=0.1, size=steps)  # I への入力

# --- WC 微分方程式（Eq.3）---
def wc_deriv(E, I, P_t, Q_t, K):
    # ネットワーク間入力：Σ_j M_ij E_j
    net_in = K * (M @ E)  # K はグローバル結合強度 :contentReference[oaicite:2]{index=2}
    uE = c_EE*E - c_EI*I - theta_E + P_t + net_in
    uI = c_IE*E - c_II*I - theta_I + Q_t
    dE = (-E + S_E(uE)) / tau_E
    dI = (-I + S_I(uI)) / tau_I
    return dE, dI

# --- 結合強度スキャン + RK4 シミュレーション + ヒートマップ表示 ---
Ks = np.linspace(0.05, 0.2, 6)  # 例：0.05–0.20 を 6 点で
last_sec   = 2.0               # 描画に使う末尾秒数
last_steps = int(last_sec / dt)

K = 0.1

# 初期化
E = np.random.rand(N) * 0.1
I = np.random.rand(N) * 0.1
E_hist = np.zeros((N, steps))
E_hist[:,0] = E

# RK4 ループ
for t in tqdm(range(steps-1)):
    dE1, dI1 = wc_deriv(E, I, P[t],   Q[t],   K)
    dE2, dI2 = wc_deriv(E+0.5*dt*dE1, I+0.5*dt*dI1, P[t], Q[t], K)
    dE3, dI3 = wc_deriv(E+0.5*dt*dE2, I+0.5*dt*dI2, P[t], Q[t], K)
    dE4, dI4 = wc_deriv(E+dt*dE3,     I+dt*dI3,     P[t], Q[t], K)
    E += (dt/6)*(dE1 + 2*(dE2 + dE3) + dE4)
    I += (dt/6)*(dI1 + 2*(dI2 + dI3) + dI4)
    E_hist[:,t+1] = E

# ヒートマップ
plt.figure(figsize=(8,4))
sns.heatmap(E_hist[:, -last_steps:], 
            cbar_kws={'label':'E activity'},
            yticklabels=False, xticklabels=last_steps//1000)
plt.title(f'Excitatory activity heatmap (K={K:.3f})')
plt.xlabel('Time steps (last 2s)')
plt.ylabel('Node index')
plt.tight_layout()
plt.show()
