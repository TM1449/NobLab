"""
Wilson–Cowan (2 columns) + UKF
==============================
観測       : 各コラムの興奮性集団 E1, E2
推定状態   : (E1, I1, E2, I2, W_EE)
未知パラメータ: W_EE（2 コラムで共通，ランダムウォーク）
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

# ============== シミュレーション設定 ==============
DT      = 0.01      # [ms]
T_TOTAL = 500.0     # [ms]
N_STEP  = int(T_TOTAL / DT)
TIME    = np.arange(N_STEP) * DT

# ============== Wilson–Cowan パラメータ (真値) ==============
TAU_E, TAU_I = 20.0, 10.0      # 時定数 [ms]

W_EE_TRUE = 16.0               # **推定対象**
W_EI = 26.0
W_IE = 20.0
W_II = 1.0
C_INTER = 0.5                  # コラム間 Exc→Exc 結合強度 (既知とする)

R_E = 0.0                      # 不応期係数
R_I = 0.0

# ----- シグモイド関数係数 -----
A_E, TH_E = 1.0, 5.0
A_I, TH_I = 1.0, 20.0
def logistic(x, a, th): return 1.0 / (1.0 + np.exp(-a * (x - th)))
S_E = lambda x: logistic(x, A_E, TH_E)
S_I = lambda x: logistic(x, A_I, TH_I)

# ============== 外部入力刺激 (左右同じにしておく) ==============
def input_signal(idx, dt, amp, period, width=None, base=0.):
    t = np.asarray(idx) * dt
    sig = np.where((t % period) < width, amp, 0.0) + base
    return sig

IDX = np.arange(N_STEP)
P_E = input_signal(IDX, DT, amp=3, period=50, width=10, base=2)  # (N_STEP,)
Q_I = np.ones(N_STEP) * 7

# ============== 2 コラム Wilson–Cowan 微分方程式 ==============
def wc_2col_deriv(E1, I1, E2, I2, P_t, Q_t, W_EE, W_IE, C):
    """d/dt for (E1,I1,E2,I2)"""
    dE1 = (-E1 + (1 - R_E*E1)*S_E(W_EE*E1 - W_EI*I1 + P_t + C*E2)) / TAU_E
    dI1 = (-I1 + (1 - R_I*I1)*S_I(W_IE*E1 - W_II*I1 + Q_t)) / TAU_I
    dE2 = (-E2 + (1 - R_E*E2)*S_E(W_EE*E2 - W_EI*I2 + P_t + C*E1)) / TAU_E
    dI2 = (-I2 + (1 - R_I*I2)*S_I(W_IE*E2 - W_II*I2 + Q_t)) / TAU_I
    return dE1, dI1, dE2, dI2

# ============== 真の軌道生成 (RK4) ==============
E1_true = np.zeros(N_STEP); I1_true = np.zeros(N_STEP)
E2_true = np.zeros(N_STEP); I2_true = np.zeros(N_STEP)
E1_true[0], I1_true[0] = 0.5, 0.4
E2_true[0], I2_true[0] = 0.4, 0.5

for k in tqdm(range(N_STEP-1), desc='Truth (RK4)'):
    k1 = wc_2col_deriv(E1_true[k], I1_true[k], E2_true[k], I2_true[k],
                       P_E[k], Q_I[k], W_EE_TRUE, W_IE, C_INTER)
    k2 = wc_2col_deriv(E1_true[k]+0.5*DT*k1[0], I1_true[k]+0.5*DT*k1[1],
                       E2_true[k]+0.5*DT*k1[2], I2_true[k]+0.5*DT*k1[3],
                       P_E[k], Q_I[k], W_EE_TRUE, W_IE, C_INTER)
    k3 = wc_2col_deriv(E1_true[k]+0.5*DT*k2[0], I1_true[k]+0.5*DT*k2[1],
                       E2_true[k]+0.5*DT*k2[2], I2_true[k]+0.5*DT*k2[3],
                       P_E[k], Q_I[k], W_EE_TRUE, W_IE, C_INTER)
    k4 = wc_2col_deriv(E1_true[k]+DT*k3[0], I1_true[k]+DT*k3[1],
                       E2_true[k]+DT*k3[2], I2_true[k]+DT*k3[3],
                       P_E[k], Q_I[k], W_EE_TRUE, W_IE, C_INTER)
    E1_true[k+1] = E1_true[k] + DT/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
    I1_true[k+1] = I1_true[k] + DT/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
    E2_true[k+1] = E2_true[k] + DT/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
    I2_true[k+1] = I2_true[k] + DT/6*(k1[3]+2*k2[3]+2*k3[3]+k4[3])

# ============== 観測データ (E1,E2) ==============
OBS_NOISE_STD = 0.01
rng = np.random.default_rng(100)
Z_obs = np.vstack([E1_true, E2_true]).T + rng.normal(0.0, OBS_NOISE_STD,
                                                     size=(N_STEP,2))

# ============== UKF セットアップ ==============
DIM_X = 5      # E1,I1,E2,I2,W_EE
DIM_Z = 2      # E1,E2
SIGMA_PTS = MerweScaledSigmaPoints(n=DIM_X, alpha=1e-3, beta=2., kappa=0.)

step_counter = {"k": 0}
def fx(x: np.ndarray, _dt):
    k = step_counter["k"]
    E1,I1,E2,I2,W = x
    # 1 ステップ先へ RK4
    k1 = wc_2col_deriv(E1,I1,E2,I2,P_E[k],Q_I[k],W,W_IE,C_INTER)
    k2 = wc_2col_deriv(E1+0.5*DT*k1[0], I1+0.5*DT*k1[1],
                       E2+0.5*DT*k1[2], I2+0.5*DT*k1[3],
                       P_E[k],Q_I[k],W,W_IE,C_INTER)
    k3 = wc_2col_deriv(E1+0.5*DT*k2[0], I1+0.5*DT*k2[1],
                       E2+0.5*DT*k2[2], I2+0.5*DT*k2[3],
                       P_E[k],Q_I[k],W,W_IE,C_INTER)
    k4 = wc_2col_deriv(E1+DT*k3[0], I1+DT*k3[1],
                       E2+DT*k3[2], I2+DT*k3[3],
                       P_E[k],Q_I[k],W,W_IE,C_INTER)
    E1n = E1 + DT/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
    I1n = I1 + DT/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
    E2n = E2 + DT/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
    I2n = I2 + DT/6*(k1[3]+2*k2[3]+2*k3[3]+k4[3])
    return np.array([E1n,I1n,E2n,I2n,W])  # W は定数 (ランダムウォーク)

hx = lambda x: x[[0,2]]          # 観測：E1, E2

ukf = UnscentedKalmanFilter(dim_x=DIM_X, dim_z=DIM_Z,
                            dt=DT, fx=fx, hx=hx, points=SIGMA_PTS)

# 初期推定値
ukf.x = np.array([0.1,0.3,0.1,0.3,10.0])   # [E1,I1,E2,I2,W_EE]
ukf.P = np.diag([0.1,0.1,0.1,0.1,1.0])

# プロセス / 観測ノイズ
ukf.Q = np.diag([1e-5,1e-5,1e-5,1e-5,5e-4])   # 最後の要素が W_EE 用
ukf.R = np.eye(2) * OBS_NOISE_STD**2

# ============== UKF ループ ==============
E1_est = np.zeros(N_STEP); I1_est = np.zeros(N_STEP)
E2_est = np.zeros(N_STEP); I2_est = np.zeros(N_STEP)
W_est  = np.zeros(N_STEP)

for k in tqdm(range(N_STEP), desc='UKF 2-col'):
    ukf.predict()
    ukf.update(Z_obs[k])
    E1_est[k],I1_est[k],E2_est[k],I2_est[k],W_est[k] = ukf.x
    step_counter["k"] += 1

# ============== 可視化 ==============
plt.figure(figsize=(10,10))

plt.subplot(4,1,1)
plt.plot(TIME,E1_true,'k',lw=1.3,label='True E1')
plt.plot(TIME,E1_est ,'r',lw=1.0,label='UKF est E1')
plt.legend(); plt.ylabel('E1'); plt.grid()

plt.subplot(4,1,2)
plt.plot(TIME,E2_true,'k',lw=1.3,label='True E2')
plt.plot(TIME,E2_est ,'r',lw=1.0,label='UKF est E2')
plt.legend(); plt.ylabel('E2'); plt.grid()

plt.subplot(4,1,3)
plt.plot(TIME,I1_true,'k',lw=1.3,label='True I1')
plt.plot(TIME,I1_est ,'b',lw=1.0,label='UKF est I1')
plt.plot(TIME,I2_true,'k--',lw=1.3,label='True I2')
plt.plot(TIME,I2_est ,'b--',lw=1.0,label='UKF est I2')
plt.legend(); plt.ylabel('I firing'); plt.grid()

plt.subplot(4,1,4)
plt.plot(TIME,W_est,'m',lw=1.0,label='est W_EE')
plt.hlines(W_EE_TRUE,0,T_TOTAL,colors='k',linestyles='dashed',label='true W_EE')
plt.legend(); plt.ylabel('W_EE'); plt.grid()

plt.tight_layout() 
plt.show()



plt.figure(figsize=(10,10))

plt.subplot(4,1,1)
plt.plot(TIME,E1_true,'k',lw=1.5,label='True E1')
plt.plot(TIME,E1_est ,'r',lw=1.0,label='UKF est E1')
plt.legend(); plt.ylabel('E1'); plt.grid()

plt.subplot(4,1,2)
plt.plot(TIME,E2_true,'k',lw=1.5,label='True E2')
plt.plot(TIME,E2_est ,'r',lw=1.0,label='UKF est E2')
plt.legend(); plt.ylabel('E2'); plt.grid()

plt.subplot(4,1,3)
plt.plot(TIME,I1_true,'k',lw=1.5,label='True I1')
plt.plot(TIME,I1_est ,'b',lw=1.0,label='UKF est I1')
plt.legend(); plt.ylabel('I1'); plt.grid()

plt.subplot(4,1,4)
plt.plot(TIME,I2_true,'k--',lw=1.5,label='True I2')
plt.plot(TIME,I2_est ,'b--',lw=1.0,label='UKF est I2')
plt.legend(); plt.ylabel('I2'); plt.grid()

plt.tight_layout() 
plt.show()