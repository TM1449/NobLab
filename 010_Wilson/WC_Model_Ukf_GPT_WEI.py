"""
Wilson–Cowan + UKF  ── W_EE 単一パラメータ推定版
================================================
filterpy.UnscentedKalmanFilter を用いて
    * (E, I) の神経活動状態
    * 結合強度パラメータ **W_EE**
を同時推定します。

観測は興奮性集団 E のみ。
パラメータはランダムウォーク (定数 + 白色ノイズ) と仮定。
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

# ====================== シミュレーション設定 ======================
DT      = 0.01   # [ms]
T_TOTAL = 500.0   # [ms]

N_STEP  = int(T_TOTAL / DT)
TIME    = np.arange(N_STEP) * DT

# ====================== Wilson–Cowan パラメータ (真値) ======================
TAU_E = 20.0          # 時定数 [ms]
TAU_I = 10.0          # 時定数 [ms]

W_EE_TRUE = 16.0  # 固定値
W_EI_TRUE = 26.0  # <-- 今回はこれを推定する

W_IE = 20.0
W_II = 1.0

R_E  = 0.0
R_I  = 0.0              # 不応期係数

# --- 推定対象パラメータ ---
PARAM_NAMES = ["W_EI"]
PARAM_TRUE  = np.array([W_EI_TRUE])
PARAM_INIT  = np.array([20.0])   # 初期値 (W_EI_TRUE とずらし)


# ----- シグモイド関数係数 -----
A_E = 1.0  
TH_E = 5.0

A_I = 1.0
TH_I = 20.0

# ====================== ユーティリティ関数 ======================

def logistic(x: float | np.ndarray, a: float, theta: float):
    """ロジスティック関数"""
    return 1.0 / (1.0 + np.exp(-a * (x - theta)))

S_E = lambda x: logistic(x, A_E, TH_E)
S_I = lambda x: logistic(x, A_I, TH_I)

# ====================== 外部入力刺激 ======================
def Input_Signal(T_idx, Dt, Amp, Period, Width = None, Base = 0):
    
    tt = np.asarray(T_idx) * Dt
    Sig = np.where((tt % Period) < Width, Amp, 0.0) + Base

    return Sig

IDX = np.arange(N_STEP)

#P_E = Input_Signal(IDX, DT, Amp=3, Period=50, Width=3000, Base=3)
#Q_I = Input_Signal(IDX, DT, Amp=7, Period=1, Width=3000, Base=7)

P_E = np.ones(N_STEP) * 3
Q_I = np.ones(N_STEP) * 7
"""
P_E = input_signal(IDX, DT, amp=3.0, period=50.0, width=2000.0)
Q_I = input_signal(IDX, DT, amp=7.0, period=50.0, width=5000.0)  # 定常入力
"""
# ====================== Wilson–Cowan 微分方程式 ======================

def wilson_cowan_deriv(E, I, P_t, Q_t, W_EI_est, W_IE):
    dE = (-E + (1.0 - R_E * E) * S_E(W_EE_TRUE * E - W_EI_est * I + P_t)) / TAU_E
    dI = (-I + (1.0 - R_I * I) * S_I(W_IE * E - W_II * I + Q_t)) / TAU_I
    return dE, dI

# ====================== 真の軌道生成 (RK4) ======================
E_true = np.zeros(N_STEP)
I_true = np.zeros(N_STEP)

E_true[0], I_true[0] = 0.5, 0.4  # 初期値

for k in tqdm(range(N_STEP - 1), desc="Truth (RK4)"):
    k1_E, k1_I = wilson_cowan_deriv(E_true[k],              I_true[k],              P_E[k], Q_I[k], W_EI_TRUE, W_IE)
    k2_E, k2_I = wilson_cowan_deriv(E_true[k]+ 0.5*DT*k1_E, I_true[k]+ 0.5*DT*k1_I, P_E[k], Q_I[k], W_EI_TRUE, W_IE)
    k3_E, k3_I = wilson_cowan_deriv(E_true[k]+ 0.5*DT*k2_E, I_true[k]+ 0.5*DT*k2_I, P_E[k], Q_I[k], W_EI_TRUE, W_IE)
    k4_E, k4_I = wilson_cowan_deriv(E_true[k]+     DT*k3_E, I_true[k]+     DT*k3_I, P_E[k], Q_I[k], W_EI_TRUE, W_IE)
    E_true[k+1] = E_true[k] + (DT/6.0)*(k1_E + 2*k2_E + 2*k3_E + k4_E)
    I_true[k+1] = I_true[k] + (DT/6.0)*(k1_I + 2*k2_I + 2*k3_I + k4_I)

# ====================== 観測データ ======================
OBS_NOISE_STD = 0.01
rng = np.random.default_rng(100)
Z_obs = E_true + rng.normal(0.0, OBS_NOISE_STD, size = N_STEP)

# ====================== UKF セットアップ ======================
DIM_X = 2 + len(PARAM_NAMES)   # E, I + 推定パラメータ
DIM_Z = 1                      # 観測 (E のみ)
SIGMA_PTS = MerweScaledSigmaPoints(n=DIM_X, alpha=1e-3, beta=2.0, kappa=0.0)

# --- fx: 状態遷移 (E, I は RK4 / パラメータ θ は定数) ---
step_counter = {"k": 0}

def fx(x: np.ndarray, _dt):
    k = step_counter["k"]
    E, I, W_EI_est = x

    k1_E, k1_I = wilson_cowan_deriv(E, I, P_E[k], Q_I[k], W_EI_est, W_IE)
    k2_E, k2_I = wilson_cowan_deriv(E+0.5*DT*k1_E, I+0.5*DT*k1_I, P_E[k], Q_I[k], W_EI_est, W_IE)
    k3_E, k3_I = wilson_cowan_deriv(E+0.5*DT*k2_E, I+0.5*DT*k2_I, P_E[k], Q_I[k], W_EI_est, W_IE)
    k4_E, k4_I = wilson_cowan_deriv(E+    DT*k3_E, I+    DT*k3_I, P_E[k], Q_I[k], W_EI_est, W_IE)

    E_next = E + (DT/6.0)*(k1_E + 2*k2_E + 2*k3_E + k4_E)
    I_next = I + (DT/6.0)*(k1_I + 2*k2_I + 2*k3_I + k4_I)

    return np.array([E_next, I_next, W_EI_est])


# --- hx: 観測モデル (E のみ) ---
hx = lambda x: np.array([x[0]])

ukf = UnscentedKalmanFilter(dim_x=DIM_X, dim_z=DIM_Z, dt=DT, fx=fx, hx=hx, points=SIGMA_PTS)

# 初期推定値
ukf.x = np.concatenate([[0.1, 0.3], PARAM_INIT])
ukf.P = np.diag([0.1, 0.1, 1])

# プロセス / 観測ノイズ (パラメータのランダムウォークを極小に設定)
PROCESS_NOISE_W_EE = 1e-3            # W_EE のランダムウォーク強さ 2e-3

ukf.Q = np.diag([1e-5, 1e-5, PROCESS_NOISE_W_EE])
ukf.R = np.array([[OBS_NOISE_STD**2]])

# ====================== UKF ループ ======================
E_est = np.zeros(N_STEP)
I_est = np.zeros(N_STEP)
W_EI_est_s = np.zeros(N_STEP)

for k in tqdm(range(N_STEP), desc="UKF with params"):
    ukf.predict()
    ukf.update(Z_obs[k])

    E_est[k], I_est[k] = ukf.x[:2]
    W_EI_est_s[k] = ukf.x[2]
    step_counter["k"] += 1

# ====================== 結果可視化 ======================
plt.figure(figsize=(10, 8))

# --- E (True / Obs / Est) ---
plt.subplot(3, 1, 1)
plt.plot(TIME, E_true, 'k', lw=1.3, label='True E')
plt.plot(TIME, Z_obs,  'gx', ms=2,  label='Obs E')
plt.plot(TIME, E_est,  'r',  lw=1.0, label='UKF est E')
plt.ylabel('E firing rate')
plt.legend()
plt.grid()

# --- I (True / Est) ---
plt.subplot(3, 1, 2)
plt.plot(TIME, I_true, 'k', lw=1.3, label='True I')
plt.plot(TIME, I_est,  'b',  lw=1.0, label='UKF est I')
plt.ylabel('I firing rate')
plt.legend()
plt.grid()

# --- 推定パラメータ ---
plt.subplot(3, 1, 3)
plt.plot(TIME, W_EI_est_s, 'm', lw=1.0, label='est W_EI')
plt.hlines(W_EI_TRUE, 0, T_TOTAL, colors='k', linestyles='dashed', label='true W_EI')
plt.ylabel('W_EI')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# ====================== 使い方メモ ======================
# * PARAM_NAMES に推定したいパラメータ名を追加し、wilson_cowan_deriv に渡せば拡張可能。
# * ukf.Q のパラメータ部を大きくするとランダムウォークが強まり、早く収束/振動しやすくなります。
# * 観測を I も含める場合は hx を変更し DIM_Z や R を合わせてください。
