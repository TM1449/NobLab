# Wilson_Cowan_UKF_RK4_fixed.py
# ------------------------------------------------------------
#  1) 「Wilson_Cowan_Original.py」のパラメータ／関数を踏襲
#  2) 4 次 RK でダイナミクスを離散化して UKF の fx に渡す
#  3) 観測は E 成分のみ（hx を編集すれば I も観測可）
#  4) step_counter をループ外で管理して IndexError を防止 ←★修正点
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

# ---------- シミュレーション設定 ----------
dt   = 0.01       # 時間刻み幅 (ms)
T    = 300.0      # 総シミュレーション時間 (ms)
Step = int(T / dt)  # ステップ数
time = np.arange(Step) * dt

# ---------- Wilson–Cowan パラメータ ----------
tau_E, tau_I = 20.0, 10.0
w_EE, w_EI   = 16.0, 26.0
w_IE, w_II   = 20.0,  1.0
r_E, r_I     =  0.0,  0.0
I_i          =  7.0                    # I への外部入力（定数）

# ----- シグモイド関数 -----
a_E, theta_E = 1.0,  5.0
a_I, theta_I = 1.0, 20.0

def logistic(x, a, theta): 
    """ロジスティック関数"""
    return 1.0 / (1.0 + np.exp(-a * (x - theta)))

def S_E(x): 
    """E ニューロンのシグモイド関数"""
    return logistic(x, a_E, theta_E)

def S_I(x): 
    """I ニューロンのシグモイド関数"""
    return logistic(x, a_I, theta_I)

# ---------- 時変パルス入力 ----------
P_base, pulseA = 2.0, 3.0
pulseWidth, period = 10.0, 20.0
def Pulse(t_ms):
    phase    = np.mod(t_ms, period)
    pulse_on = phase < pulseWidth
    return P_base + pulseA * pulse_on

# ---------- 事前に入力列を作成（外部刺激は既知と仮定） ----------
I_E = Pulse(time)
I_I = np.full_like(time, I_i)

# ---------- 真の軌道を 4 次 RK で生成（グラウンドトゥルース） ----------
E_true = np.zeros(Step)
I_true = np.zeros(Step)
E_true[0], I_true[0] = 0.5, 0.4

def wilson_cowan_deriv(E, I, I_E_t, I_I_t):
    dE = (-E + (1 - r_E * E) * S_E(w_EE * E - w_EI * I + I_E_t)) / tau_E
    dI = (-I + (1 - r_I * I) * S_I(w_IE * E - w_II * I + I_I_t)) / tau_I
    return dE, dI

for t in tqdm(range(Step - 1), desc='Truth (RK4)'):
    k1_E, k1_I = wilson_cowan_deriv(E_true[t], I_true[t], I_E[t], I_I[t])
    k2_E, k2_I = wilson_cowan_deriv(E_true[t] + 0.5*dt*k1_E, I_true[t] + 0.5*dt*k1_I, I_E[t], I_I[t])
    k3_E, k3_I = wilson_cowan_deriv(E_true[t] + 0.5*dt*k2_E, I_true[t] + 0.5*dt*k2_I, I_E[t], I_I[t])
    k4_E, k4_I = wilson_cowan_deriv(E_true[t] +     dt*k3_E, I_true[t] +     dt*k3_I, I_E[t], I_I[t])
    E_true[t+1] = E_true[t] + (dt/6)*(k1_E + 2*k2_E + 2*k3_E + k4_E)
    I_true[t+1] = I_true[t] + (dt/6)*(k1_I + 2*k2_I + 2*k3_I + k4_I)

# ---------- 観測（E のみ） + 観測ノイズ ----------
obs_noise_std = 0.05          # ←★ 観測ノイズ強度をここで調整
rng = np.random.default_rng(0)
z_obs = E_true + rng.normal(0, obs_noise_std, size=Step)

# ==================== UKF の設定 ====================
dim_x, dim_z = 2, 1  # 状態次元 (E,I) / 観測次元 (E)

points = MerweScaledSigmaPoints(n=dim_x, alpha=1e-3, beta=2.0, kappa=0.0)

# ------- fx: 4 次 RK1 ステップ (インデックスは進めない) --------
step_counter = {'k': 0}         # 時刻インデックスを保持

def fx(x, dt_unused):
    k = step_counter['k']       # 参照のみ
    E, I = x
    I_E_t, I_I_t = I_E[k], I_I[k]

    k1_E, k1_I = wilson_cowan_deriv(E, I, I_E_t, I_I_t)
    k2_E, k2_I = wilson_cowan_deriv(E + 0.5*dt*k1_E, I + 0.5*dt*k1_I, I_E_t, I_I_t)
    k3_E, k3_I = wilson_cowan_deriv(E + 0.5*dt*k2_E, I + 0.5*dt*k2_I, I_E_t, I_I_t)
    k4_E, k4_I = wilson_cowan_deriv(E +     dt*k3_E, I +     dt*k3_I, I_E_t, I_I_t)
    E_next = E + (dt/6)*(k1_E + 2*k2_E + 2*k3_E + k4_E)
    I_next = I + (dt/6)*(k1_I + 2*k2_I + 2*k3_I + k4_I)
    return np.array([E_next, I_next])

def hx(x):                      # 観測モデル（E のみ）
    return np.array([x[0]])

ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z,
                            dt=dt, fx=fx, hx=hx, points=points)

ukf.x = np.array([0.1, 0.3])    # 初期推定
ukf.P = np.diag([0.1, 0.1])

process_noise_E = 1e-5          # ←★ 状態ノイズ強度（E）
process_noise_I = 1e-5          # ←★ 状態ノイズ強度（I）
ukf.Q = np.diag([process_noise_E, process_noise_I])
ukf.R = np.array([[obs_noise_std**2]])

# ---------- UKF フィルタリング ----------
E_est, I_est = np.zeros(Step), np.zeros(Step)

for k in tqdm(range(Step), desc='UKF'):
    ukf.predict()               # fx が内部で 5 回呼ばれる
    ukf.update(z_obs[k])
    E_est[k], I_est[k] = ukf.x
    step_counter['k'] += 1      # ★ 1 ステップにつき 1 だけ進める

# ==================== 可視化 ====================
plt.figure(figsize=(9, 6))

plt.subplot(3,1,1)
plt.plot(time, E_true, 'k', lw=1.5, label='True E')
plt.plot(time, z_obs,  'gx', ms=2, label='Obs E')
plt.plot(time, E_est,  'r',  lw=1, label='UKF est E')
plt.ylabel('E firing rate')
plt.legend(); plt.grid()

plt.subplot(3,1,2)
plt.plot(time, I_true, 'k', lw=1.5, label='True I')
plt.plot(time, I_est,  'b',  lw=1, label='UKF est I')
plt.ylabel('I firing rate')
plt.legend(); plt.grid()

plt.subplot(3,1,3)
plt.plot(time, I_E, 'g', lw=2, label='External Input (I_E)')
plt.xlabel('Time (ms)')
plt.ylabel('Input')
plt.legend(); plt.grid()

plt.tight_layout()
plt.show()
