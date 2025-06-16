"""
Wilson–Cowan + Unscented Kalman Filter (UKF)
--------------------------------------------
* 時変パルス刺激 P_ext(t)
* 高活動初期値 (E=0.5, I=0.4)
* 未知パラメータ: w_EE  (対数空間で推定)
* 4 次ルンゲクッタ (RK4) 離散化
* 仕上げは散布図で可視化
依存: numpy, matplotlib, filterpy, tqdm
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

# ------------------ 1. 定数とユーティリティ ------------------
dt   = 0.02           # [ms]
Tend = 1000.0
tvec = np.arange(0, Tend + dt, dt)      # 時間軸
N    = len(tvec)

tau_E, tau_I = 20.0, 10.0
w_EI, w_IE, w_II = 26.0, 20.0, 1.0
r_E, r_I = 0.0, 0.0
Qext = 7.0

w_EE_true = 16.0                        # 真の結合重み

# シグモイド関数
def S(x, a, theta):
    return 1.0 / (1.0 + np.exp(-a * (x - theta)))

a_E, theta_E = 1.0, 5.0
a_I, theta_I = 1.0, 20.0
def S_E(x): return S(x, a_E, theta_E)
def S_I(x): return S(x, a_I, theta_I)

# ------------------ 2. 時変入力 ------------------------------
P_base = 2.0
pulseA = 3.0
pulseWidth, period = 20.0, 200.0         # [ms]
def P_ext(t_ms):
    """パルス付き入力: 200 ms 周期で 20 ms 幅のパルス (+3)"""
    return P_base + pulseA * ((t_ms % period) < pulseWidth)

P_vec = P_ext(tvec)                      # あらかじめ配列化

# ------------------ 3. 真値軌道を生成 ------------------------
E_true = np.zeros(N)
I_true = np.zeros(N)
E_true[0], I_true[0] = 0.5, 0.4          # 高活動初期値

def wc_deriv(E, I, wEE, Pnow):
    dE = (-E + (1 - r_E * E) * S_E(wEE * E - w_EI * I + Pnow)) / tau_E
    dI = (-I + (1 - r_I * I) * S_I(w_IE * E - w_II * I + Qext)) / tau_I
    return dE, dI

for k in tqdm(range(N - 1), desc='Generate truth (RK4)'):
    Pnow = P_vec[k]
    k1_E, k1_I = wc_deriv(E_true[k],               I_true[k],               w_EE_true, Pnow)
    k2_E, k2_I = wc_deriv(E_true[k] + 0.5*dt*k1_E, I_true[k] + 0.5*dt*k1_I, w_EE_true, Pnow)
    k3_E, k3_I = wc_deriv(E_true[k] + 0.5*dt*k2_E, I_true[k] + 0.5*dt*k2_I, w_EE_true, Pnow)
    k4_E, k4_I = wc_deriv(E_true[k] +     dt*k3_E, I_true[k] +     dt*k3_I, w_EE_true, Pnow)
    E_true[k+1] = E_true[k] + (dt/6)*(k1_E + 2*k2_E + 2*k3_E + k4_E)
    I_true[k+1] = I_true[k] + (dt/6)*(k1_I + 2*k2_I + 2*k3_I + k4_I)

# ------------------ 4. 観測生成 -------------------------------
R = 1e-2 * np.eye(2)                    # 観測ノイズ共分散
rng = np.random.default_rng(0)
y_obs = np.vstack([E_true, I_true]) + rng.multivariate_normal(
            mean=np.zeros(2), cov=R, size=N
        ).T

# ------------------ 5. UKF セットアップ -----------------------
dim_x, dim_z = 3, 2                     # [E, I, u], 観測 [E, I]
points = MerweScaledSigmaPoints(n=dim_x, alpha=1e-3, beta=2.0, kappa=0.0)

# インデックス用の辞書（fx から参照）
step_idx = {'k': 0}

def fx(state, _dt):
    """1 ステップ先を RK4 で予測。u=log w_EE はランダムウォーク"""
    k = step_idx['k']
    E, I, u = state
    wEE = np.exp(u)
    Pnow = P_vec[k]

    # RK4
    def deriv(e, i):
        return wc_deriv(e, i, wEE, Pnow)

    k1_E, k1_I = deriv(E,               I)
    k2_E, k2_I = deriv(E + 0.5*dt*k1_E, I + 0.5*dt*k1_I)
    k3_E, k3_I = deriv(E + 0.5*dt*k2_E, I + 0.5*dt*k2_I)
    k4_E, k4_I = deriv(E +     dt*k3_E, I +     dt*k3_I)
    E_next = E + (dt/6)*(k1_E + 2*k2_E + 2*k3_E + k4_E)
    I_next = I + (dt/6)*(k1_I + 2*k2_I + 2*k3_I + k4_I)

    return np.array([E_next, I_next, u])   # u はそのまま

def hx(state):
    """観測関数: [E, I]"""
    return state[:2]

ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z,
                            dt=dt, fx=fx, hx=hx, points=points)

# 初期値と共分散
ukf.x = np.array([0.5, 0.4, np.log(12.0)])      # E, I, u
ukf.P = np.diag([0.05**2, 0.05**2, 1.0])

# プロセス・観測ノイズ
Q_state = 1e-4
Q_param = 1e-6
ukf.Q = np.diag([Q_state, Q_state, Q_param])
ukf.R = R

# ------------------ 6. フィルタリング -------------------------
history = np.zeros((dim_x, N))
for k in tqdm(range(N), desc='UKF'):
    ukf.predict()
    ukf.update(y_obs[:, k])
    # パラメータ安全域 (log 5 ～ log 40)
    ukf.x[2] = np.clip(ukf.x[2], np.log(5.0), np.log(40.0))
    history[:, k] = ukf.x
    step_idx['k'] += 1

E_est, I_est = history[0], history[1]
wEE_est      = np.exp(history[2])

# ------------------ 7. 散布図プロット --------------------------
skip = 10        # プロット点を間引く間隔（表示負荷対策）

fig, axs = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

# (a) Excitatory population E
axs[0].scatter(tvec[::skip], E_true[::skip],
               c='black', s=10, label='True E')
axs[0].scatter(tvec[::skip], E_est[::skip],
               c='red',   s=10, label='UKF E_hat', alpha=0.6)
axs[0].set_ylabel('E')
axs[0].legend(); axs[0].grid(True)

# (b) Inhibitory population I
axs[1].scatter(tvec[::skip], I_true[::skip],
               c='black', s=10, label='True I')
axs[1].scatter(tvec[::skip], I_est[::skip],
               c='blue',  s=10, label='UKF I_hat', alpha=0.6)
axs[1].set_ylabel('I')
axs[1].legend(); axs[1].grid(True)

# (c) Estimated w_EE
axs[2].scatter(tvec[::skip],
               np.full_like(tvec[::skip], w_EE_true),
               c='black', s=10, label='True w_EE')
axs[2].scatter(tvec[::skip], wEE_est[::skip],
               c='green', s=10, label='UKF w_EE_hat', alpha=0.6)
axs[2].set_ylabel('w_EE')
axs[2].set_xlabel('Time [ms]')
axs[2].legend(); axs[2].grid(True)

fig.suptitle('UKF (scatter view): Wilson–Cowan, time-varying stimulus, w_EE estimation')
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
