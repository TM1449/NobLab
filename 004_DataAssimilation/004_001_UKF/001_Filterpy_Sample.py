"""
Wilson–Cowan model: joint state + parameter estimation with Unscented Kalman Filter
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

# ------------------------------------------------------------------
# 1.  真モデル設定（シミュレーション用）
# ------------------------------------------------------------------
dt       = 0.01         # s
steps    = 3000         # assimilation length
t_grid   = np.linspace(0, dt*steps, steps+1)

# Wilson–Cowan parameters (true, hidden from filter)
w_EE, w_EI =  10.0, 12.0
w_IE, w_II =  10.0, 10.0
a_E, a_I   =  1.3, 2.0
th_E, th_I =  4.0, 3.7
P_E_true, P_I =  1.25, 0.5          # P_E is unknown to the filter

def S(x, a, th):
    return 1.0 / (1.0 + np.exp(-a*(x - th)))

def wilson_cowan(state, t, P_E):
    E, I = state
    dE = -E + S(w_EE*E - w_EI*I + P_E, a_E, th_E)
    dI = -I + S(w_IE*E - w_II*I + P_I, a_I, th_I)
    return [dE, dI]

# integrate “nature run”
x0_true = [0.2, 0.2]
traj = odeint(wilson_cowan, x0_true, t_grid, args=(P_E_true,))

# noisy observation (E only)
obs_std = 0.05
rng = np.random.default_rng(0)
z_obs = traj[:,0] + rng.normal(0, obs_std, size=steps+1)

# ------------------------------------------------------------------
# 2.  UKF セットアップ
# ------------------------------------------------------------------
n_state   = 2   # E,I
n_param   = 1   # P_E
n_aug     = n_state + n_param
dt_filter = dt

# σ点
points = MerweScaledSigmaPoints(n_aug, alpha=1e-3, beta=2.0, kappa=0.0)

# --- 状態遷移 fx（1 ステップ分を 4 次 Runge–Kutta 離散化） ---
def fx(x_aug, dt):
    E, I, P_E = x_aug

    def f(vec):
        return wilson_cowan(vec, 0.0, P_E)

    k1 = np.array(f([E, I]))
    k2 = np.array(f([E + 0.5*dt*k1[0], I + 0.5*dt*k1[1]]))
    k3 = np.array(f([E + 0.5*dt*k2[0], I + 0.5*dt*k2[1]]))
    k4 = np.array(f([E +     dt*k3[0], I +     dt*k3[1]]))

    En = E + (dt/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    In = I + (dt/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    return np.array([En, In, P_E])  # P_E: random walk (≈const)

# --- 観測関数 hx ---
def hx(x_aug):
    return np.array([x_aug[0]])  # observe E only

ukf = UnscentedKalmanFilter(dim_x=n_aug, dim_z=1, dt=dt_filter,
                            fx=fx, hx=hx, points=points)

# 初期値（適当な guess）
ukf.x = np.array([0.1, 0.3, 0.8])   # [E0_guess, I0_guess, P_E_guess]
ukf.P = np.diag([0.1, 0.1, 1.0])    # パラメータは不確実性大きめ

# 雑音共分散
process_var_state  = 1e-4
process_var_param  = 1e-6           # P_E をほぼ定数扱い
ukf.Q = np.diag([process_var_state]*n_state + [process_var_param])
ukf.R = np.array([[obs_std**2]])

# ------------------------------------------------------------------
# 3.  フィルタリングループ
# ------------------------------------------------------------------
hist = np.zeros((steps+1, n_aug))
hist[0] = ukf.x
for k in range(steps+1):
    ukf.predict()
    ukf.update(z_obs[k])
    hist[k] = ukf.x

# ------------------------------------------------------------------
# 4.  可視化
# ------------------------------------------------------------------
t = t_grid
E_true, I_true = traj[:,0], traj[:,1]
E_hat,  I_hat  = hist[:,0], hist[:,1]
P_E_hat        = hist[:,2]

plt.figure(figsize=(10,7))

plt.subplot(3,1,1)
plt.plot(t, E_true, 'k', label='True E')
plt.plot(t, z_obs, 'gx', ms=2, label='Obs E')
plt.plot(t, E_hat, 'r', label='UKF E_hat')
plt.ylabel('E')
plt.legend(); plt.grid()

plt.subplot(3,1,2)
plt.plot(t, I_true, 'k', label='True I')
plt.plot(t, I_hat, 'r', label='UKF I_hat')
plt.ylabel('I')
plt.legend(); plt.grid()

plt.subplot(3,1,3)
plt.plot(t, [P_E_true]*len(t), 'k-', label='True P_E')
plt.plot(t, P_E_hat, 'r--',  label='UKF P_E_hat')
plt.ylabel('P_E'); plt.xlabel('time (s)')
plt.legend(); plt.grid()

plt.tight_layout()
plt.show()
