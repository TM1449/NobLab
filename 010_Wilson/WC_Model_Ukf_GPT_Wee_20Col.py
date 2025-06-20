"""
Wilson–Cowan 20‑Column Network + Unscented Kalman Filter
=======================================================
• 各コラムの興奮性（E1–E20）を観測し，抑制性（I1–I20）と共通パラメータ W_EE を UKF で推定する
• 参考コード: WC_Model_Ukf_GPT_Wee_2Col.py
• 可視化:  (i) E1–E4 真値/推定,  (ii) E 全コラムのヒートマップ,  (iii) W_EE 収束

実行方法::
    pip install numpy matplotlib seaborn tqdm filterpy
    python WC_Model_Ukf_GPT_Wee_20Col.py
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

# -------------------------------
# Wilson‑Cowan 微分方程式
# -------------------------------
NC = 20                   # number of columns
DT = 0.1                  # integration step [ms]
T_TOTAL = 50.0            # total simulation time [ms]
N_STEP = int(T_TOTAL/DT)

# 接続と入力パラメータ
TRUE_WEE = 12.0           # 真の自己 E→E 結合
W_EI = 10.0               # E→I
W_IE = 10.0               # I→E
W_II = 0.0                # I→I (未使用)
W_INTER = 0.1             # 列間 E→E
P_E = 1.3                 # 外部入力 (E)
P_I = 2.0                 # 外部入力 (I)

# シグモイド関数
def S(x: np.ndarray, a: float = 1.3, theta: float = 4.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-a * (x - theta)))

# ベクトル化微分 dX/dt
def wc_multi_deriv(t: float, X: np.ndarray, params: dict) -> np.ndarray:
    E = X[:NC]
    I = X[NC:2*NC]
    W_EE = params["W_EE"]

    # 列間結合 (全結合・自己結合なし)
    E_inter = (np.sum(E) - E) / (NC - 1)

    dE = -E + S(W_EE * E - W_EI * I + W_INTER * E_inter + P_E)
    dI = -I + S(W_IE * E + P_I)
    return np.concatenate([dE, dI])

# 4次ルンゲクッタで 1 ステップ進める
def rk4_step(x: np.ndarray, dt: float, params: dict) -> np.ndarray:
    k1 = wc_multi_deriv(0, x, params)
    k2 = wc_multi_deriv(0, x + 0.5*dt*k1, params)
    k3 = wc_multi_deriv(0, x + 0.5*dt*k2, params)
    k4 = wc_multi_deriv(0, x + dt*k3, params)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# -------------------------------
# Ground‑truth simulation & observations
# -------------------------------

def simulate_truth():
    x = np.zeros(2*NC)
    x[:NC] = 0.1 + 0.05*np.random.randn(NC)  # E
    x[NC:] = 0.1 + 0.05*np.random.randn(NC)  # I

    xs_true = np.zeros((N_STEP, 2*NC))
    for k in range(N_STEP):
        xs_true[k] = x
        x = rk4_step(x, DT, {"W_EE": TRUE_WEE})
    return xs_true

np.random.seed(0)
xs_true = simulate_truth()

# 観測 (E のみ) にノイズを付加
OBS_VAR = 0.02**2
ys = xs_true[:, :NC] + np.random.randn(N_STEP, NC)*np.sqrt(OBS_VAR)

# -------------------------------
# UKF セットアップ
# -------------------------------
DIM_X = 2*NC + 1        # +1 for W_EE
DIM_Z = NC

# 初期平均
x0 = np.zeros(DIM_X)
# E,I は観測の初回値でバラつかせ，W_EE は誤った推定値で開始
x0[:NC] = ys[0]
x0[NC:2*NC] = 0.2
x0[-1] = 6.0            # 初期推定 = 半分の誤差

# 共分散初期値
P0 = np.eye(DIM_X)*0.1
P0[-1,-1] = 5.0

# プロセスノイズ
q_state = 1e-4
q_param = 5e-4
Q = np.eye(DIM_X)*q_state
Q[-1,-1] = q_param

# 観測ノイズ
R = np.eye(DIM_Z)*OBS_VAR

# Σ点
points = MerweScaledSigmaPoints(n=DIM_X, alpha=0.1, beta=2.0, kappa=0.0)

# 状態遷移関数 (UKF 用): 決定論的なので W_EE は変えない

def fx(x: np.ndarray, dt: float) -> np.ndarray:
    params = {"W_EE": x[-1]}
    next_state = rk4_step(x[:-1], dt, params)
    return np.concatenate([next_state, [x[-1]]])  # W_EE はそのまま

# 観測関数: E 部分のみ
hx = lambda x: x[:NC]

ukf = UnscentedKalmanFilter(dim_x=DIM_X, dim_z=DIM_Z, dt=DT,
                            fx=fx, hx=hx, points=points)
ukf.x = x0.copy()
ukf.P = P0.copy()
ukf.Q = Q.copy()
ukf.R = R.copy()

# Filtering loop
xs_est = np.zeros((N_STEP, DIM_X))
for k in tqdm(range(N_STEP)):
    ukf.predict()
    ukf.update(ys[k])
    xs_est[k] = ukf.x.copy()

# -------------------------------
# 可視化
# -------------------------------

def plot_results(xs_true: np.ndarray, xs_est: np.ndarray):
    time = np.arange(N_STEP)*DT

    # (i) E1–E4 time‑series
    plt.figure(figsize=(10,6))
    for idx in range(4):
        plt.plot(time, xs_true[:, idx], label=f"E{idx+1} true", linewidth=1)
        plt.plot(time, xs_est[:, idx], '--', label=f"E{idx+1} est", linewidth=1)
    plt.xlabel("time [ms]"); plt.ylabel("E activity"); plt.legend(); plt.title("E1–E4 time‑series (true vs estimate)");

    # (ii) Heatmaps for all columns (true & est)
    fig, axes = plt.subplots(1, 2, figsize=(14,4), sharey=True)
    sns.heatmap(xs_true[:, :NC].T, ax=axes[0], cbar=False)
    axes[0].set_title("True E (all columns)")
    axes[0].set_ylabel("Column #")
    sns.heatmap(xs_est[:, :NC].T, ax=axes[1], cbar=True)
    axes[1].set_title("Estimated E (all columns)")
    axes[1].set_xlabel("Time step")

    # (iii) Parameter trajectory W_EE
    plt.figure(figsize=(8,3))
    plt.plot(time, np.full_like(time, TRUE_WEE), label="True W_EE", linewidth=2)
    plt.plot(time, xs_est[:, -1], label="Estimate W_EE", linewidth=1)
    plt.xlabel("time [ms]"); plt.ylabel("W_EE"); plt.title("Parameter convergence"); plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_results(xs_true, xs_est)
