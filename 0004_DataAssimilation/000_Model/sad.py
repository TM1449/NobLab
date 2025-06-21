import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

# ---------- シミュレーション設定 ----------
dt      = 0.1             # [ms]
T_total = 500.0           # [ms]
N_step  = int(T_total / dt)
TIME    = np.arange(N_step) * dt

# ---------- コラム数 ----------
Nc = 20  # 全コラム数

# ---------- 観測対象コラムのインデックス ----------
# ここでは最初の10コラム(0–9)を観測
obs_indices = np.arange(10)  
n_obs = len(obs_indices)

# ---------- Wilson–Cowan パラメータ ----------
tau_E, tau_I = 20.0, 10.0
W_EE_true    = 16.0       # 真の W_EE
w_EI = 26.0
w_IE = 20.0
w_II = 1.0
C_inter = 0.7
r_E, r_I = 0.0, 0.0

# シグモイド
a_E, theta_E = 1.0, 5.0
a_I, theta_I = 1.0, 20.0
def sigmoid(x, a, th):
    return 1.0 / (1.0 + np.exp(-a * (x - th)))
S_E = lambda x: sigmoid(x, a_E, theta_E)
S_I = lambda x: sigmoid(x, a_I, theta_I)

# ---------- 入力信号 ----------
def input_signal(idx, dt, amp, period, width, base=0.0):
    t = idx * dt
    return np.where((t % period) < width, amp, 0.0) + base

IDX = np.arange(N_step)
P_E = input_signal(IDX, dt, amp=3.0, period=50.0, width=10.0, base=2.0)
Q_I = np.ones(N_step) * 7.0

# ---------- 真の軌道生成 (RK4) ----------
np.random.seed(0)
E_true = np.zeros((Nc, N_step))
I_true = np.zeros((Nc, N_step))
E_true[:, 0] = 0.5 + 0.05 * np.random.randn(Nc)
I_true[:, 0] = 0.4 + 0.05 * np.random.randn(Nc)

W_inter = np.ones((Nc, Nc)) - np.eye(Nc)
def wc_deriv(E, I, P_t, Q_t, W_EE):
    inter = C_inter * (W_inter @ E)
    dE = (-E + (1 - r_E*E) * S_E(W_EE*E - w_EI*I + P_t + inter)) / tau_E
    dI = (-I + (1 - r_I*I) * S_I(w_IE*E - w_II*I + Q_t)) / tau_I
    return dE, dI

for k in tqdm(range(N_step-1), desc='Truth RK4 Multi-column'):
    E, I = E_true[:, k], I_true[:, k]
    P_t, Q_t = P_E[k], Q_I[k]
    k1_E, k1_I = wc_deriv(E,    I,    P_t, Q_t, W_EE_true)
    k2_E, k2_I = wc_deriv(E+0.5*dt*k1_E, I+0.5*dt*k1_I, P_t, Q_t, W_EE_true)
    k3_E, k3_I = wc_deriv(E+0.5*dt*k2_E, I+0.5*dt*k2_I, P_t, Q_t, W_EE_true)
    k4_E, k4_I = wc_deriv(E+dt*k3_E,     I+dt*k3_I,     P_t, Q_t, W_EE_true)
    E_true[:, k+1] = E + dt/6*(k1_E + 2*k2_E + 2*k3_E + k4_E)
    I_true[:, k+1] = I + dt/6*(k1_I + 2*k2_I + 2*k3_I + k4_I)

# ---------- 観測データ (一部の E) ----------
OBS_NOISE_STD = 0.01
rng = np.random.default_rng(42)
# 全コラムの観測値をつくってから必要な列を切り出す
Z_full = E_true.T + rng.normal(0.0, OBS_NOISE_STD, size=(N_step, Nc))
Z_obs  = Z_full[:, obs_indices]  # shape=(N_step, n_obs)

# ---------- UKF セットアップ ----------
current_k = 0  # 時刻インデックス管理

dim_x = 2 * Nc + 1   # 状態次元: E_i, I_i, W_EE
dim_z = n_obs        # 観測次元: 観測する E のみ
pts = MerweScaledSigmaPoints(n=dim_x, alpha=1e-3, beta=2., kappa=0.)

def fx(x, _dt):
    """状態遷移 (RK4)"""
    global current_k
    E = x[:Nc]
    I = x[Nc:2*Nc]
    W = x[-1]
    P_t, Q_t = P_E[current_k], Q_I[current_k]

    k1_E, k1_I = wc_deriv(E, I, P_t, Q_t, W)
    k2_E, k2_I = wc_deriv(E+0.5*dt*k1_E, I+0.5*dt*k1_I, P_t, Q_t, W)
    k3_E, k3_I = wc_deriv(E+0.5*dt*k2_E, I+0.5*dt*k2_I, P_t, Q_t, W)
    k4_E, k4_I = wc_deriv(E+dt*k3_E,     I+dt*k3_I,     P_t, Q_t, W)
    E_next = E + dt/6*(k1_E + 2*k2_E + 2*k3_E + k4_E)
    I_next = I + dt/6*(k1_I + 2*k2_I + 2*k3_I + k4_I)

    return np.hstack([E_next, I_next, W])

def hx(x):
    """観測モデル: 選択した E のみを返す"""
    return x[obs_indices]

ukf = UnscentedKalmanFilter(dim_x=dim_x,
                            dim_z=dim_z,
                            dt=dt,
                            fx=fx,
                            hx=hx,
                            points=pts)

# 初期推定値
# E: 観測値を埋め、未観測は 0
x0_E = np.zeros(Nc)
x0_E[obs_indices] = Z_obs[0]
ukf.x = np.hstack([x0_E, np.zeros(Nc), 10.0])

# 共分散
P_E0 = 0.1; P_I0 = 0.1; P_W0 = 1.0
ukf.P = np.diag(np.hstack([
    np.ones(Nc)*P_E0,
    np.ones(Nc)*P_I0,
    [P_W0]
]))

# プロセス・観測ノイズ
ukf.Q = np.diag(np.hstack([
    np.ones(Nc)*1e-5,
    np.ones(Nc)*1e-5,
    [5e-4]
]))
ukf.R = np.eye(dim_z) * OBS_NOISE_STD**2

# ---------- UKF 実行 ----------
E_est = np.zeros((Nc, N_step))
I_est = np.zeros((Nc, N_step))
W_est = np.zeros(N_step)

for k in tqdm(range(N_step), desc='UKF Multi-column'):
    current_k = k
    ukf.predict()
    ukf.update(Z_obs[k])
    xk = ukf.x
    E_est[:, k] = xk[:Nc]
    I_est[:, k] = xk[Nc:2*Nc]
    W_est[k]    = xk[-1]

# ---------- 可視化 ----------

# 1. 推定パラメータ W_EE
plt.figure(figsize=(8,4))
plt.plot(TIME, W_est, 'm', label='Estimated $W_{EE}$')
plt.hlines(W_EE_true, 0, T_total, colors='k', linestyles='dashed', label='True $W_{EE}$')
plt.xlabel('Time [ms]')
plt.ylabel('$W_{EE}$')
plt.title('パラメータ $W_{EE}$ の推定結果')
plt.legend()
plt.grid(True)
plt.show()

# 2. ヒートマップ: 真値 vs 推定値の I dynamics
fig, ax = plt.subplots(1,2, figsize=(12,5))
im1 = ax[0].imshow(I_true, aspect='auto')
ax[0].set_title('True $I_i$ dynamics')
fig.colorbar(im1, ax=ax[0])
im2 = ax[1].imshow(I_est, aspect='auto')
ax[1].set_title('UKF Estimated $I_i$ dynamics')
fig.colorbar(im2, ax=ax[1])
plt.suptitle(f'抑制性活動 $I_i$ の比較 (Nc={Nc})')
plt.show()

# 3. 真値 E の時系列データ（4コラム分）
cols = [0, 1, 2, 3]  # プロットするコラム番号（0始まり）
plt.figure(figsize=(10,6))
for idx in cols:
    plt.plot(TIME, E_true[idx], label=f"True E_{idx+1}")
plt.xlabel("Time [ms]")
plt.ylabel("Excitation E")
plt.title("True E time series for 4 columns")
plt.legend()
plt.grid(True)
plt.show()

# 4. 真値 I と推定 I の時系列データ（4組）
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
for ax, idx in zip(axes.flatten(), cols):
    ax.plot(TIME, I_true[idx], label=f"True I_{idx+1}")
    ax.plot(TIME, I_est[idx], '--', label=f"Est I_{idx+1}")
    ax.set_title(f"Column {idx+1}")
    ax.set_ylabel("Inhibition I")
    ax.legend()
    ax.grid(True)
axes[-1, -1].set_xlabel("Time [ms]")
plt.suptitle("True vs Estimated I time series for 4 columns")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
