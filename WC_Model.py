# 2025_0617 作成中
# このモデルが本体となるようにする 

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

# ---------- シミュレーション設定 ----------
dt = 0.001  # シミュレーション刻み幅 (ms)
T_Total = 300.0   # 総シミュレーション時間 (ms)

Step = int(T_Total / dt)  # ステップ数
time = np.arange(Step) * dt # 時間配列


# ----- Wilson–Cowan パラメータ -----
tau_E = 20.0  # 興奮性ニューロンの時間定数 (ms)
tau_I = 10.0  # 抑制性ニューロンの時間定数 (ms)

w_EE = 16.0  # E-E 接続の重み
w_EI = 26.0  # E-I 接続の重み

w_IE = 20.0  # I-E 接続の重み
w_II = 1.0   # I-I 接続の重み

r_E = 0.0  # E ニューロンの回復率
r_I = 0.0  # I ニューロンの回復率

I_i = 7.0  # 外部入力の強さ


# ----- シグモイド関数 -----
a_E = 1.0
theta_E = 5.0

a_I = 1.0
theta_I = 20.0

def logistic(x, a, theta):
    """ロジスティック関数"""
    return 1.0 / (1.0 + np.exp(-a * (x - theta)))

def S_E(x):
    """E ニューロンのシグモイド関数"""
    return logistic(x, a_E, theta_E)

def S_I(x):
    """I ニューロンのシグモイド関数"""
    return logistic(x, a_I, theta_I)

# ----- 入力信号作成 -----
def Input_Signal(T_idx, Dt, Amp, Period, Type = None, Width = None, Noise = None):
    """
    任意の時間軸 t * dt 上に 任意の信号を生成する。

    Parameters
    ----------
    t : np.ndarray
        時間軸（ms など）。dtype は float。
    dt : float
        時間刻み幅 dt。
    Amplitude : float, optional
        波の振幅 A。
    Period : float, optional
        周期 T（t と同じ単位）。
    Type : 
        信号の種類。None（恒等関数）, np.sinなどで指定。
    width : （None, np.sin, np.cos, 'pulse'）
        （パルス波を指定したとき必須）パルス幅を指定。
    Noise :
        ノイズを指定した場合、加算する。
    Returns
    -------
    np.ndarray
        Type で指定した信号を返す。
    """
    tt = np.asarray(T_idx) * dt

    if Type == None:
        Sig = Amp * np.ones_like(tt)

    elif Type in (np.sin, np.cos):
        Sig = Amp * Type(2 * np.pi * tt / Period)

    elif Type == 'pulse':
        if Width is None:
            raise ValueError("Width must be set for pulse input")
        Sig = np.where((tt % Period) < Width, Amp, 00)

    else:
        raise ValueError("Unsupported Type")
    
    if Noise is not None:
        sig += np.random.normal(0.0, Noise, size=tt.shape)

    return Sig

# ------外部刺激 P(t) ----
def P(x):
    return x

# ------外部刺激 Q(t) ----
def Q(x):
    return x


# ----- 外部刺激信号生成 -----
idx = np.arange(Step)
P_E = Input_Signal(T_idx = idx, Dt = dt, Amp = 3.0, Period = 50, Type = 'pulse', Width = 20)
Q_I = Input_Signal(T_idx = idx, Dt = dt, Amp = 7, Period = 50, Type = None, Width = None)

# ----- 真の軌道を 4次 RKで生成 -----
E_True = np.zeros(Step)
I_True = np.zeros(Step)

E_True[0] = 0.5
I_True[0] = 0.4

def Wilson_Cowan(E, I, Pt, Qt):
    dE = (-E + (1 - r_E * E) * S_E(w_EE * E - w_EI * I + Pt)) / tau_E
    dI = (-I + (1 - r_I * I) * S_I(w_IE * E - w_II * I + Qt)) / tau_I
    return dE, dI

for i in tqdm(range(Step -1), desc = 'Truth Wilson-Cowan Model (RK4)'):
    k1_E, k1_I = Wilson_Cowan(E_True[i]                  , I_True[i]                  , P_E[i], Q_I[i])
    k2_E, k2_I = Wilson_Cowan(E_True[i] + 0.5 * dt * k1_E, I_True[i] + 0.5 * dt * k1_I, P_E[i], Q_I[i])
    k3_E, k3_I = Wilson_Cowan(E_True[i] + 0.5 * dt * k2_E, I_True[i] + 0.5 * dt * k2_I, P_E[i], Q_I[i])
    k4_E, k4_I = Wilson_Cowan(E_True[i] +       dt * k3_E, I_True[i] +       dt * k3_I, P_E[i], Q_I[i])

    E_True[i + 1] = E_True[i] + (dt / 6) * (k1_E + 2 * k2_E + 2 * k3_E + k4_E)
    I_True[i + 1] = I_True[i] + (dt / 6) * (k1_I + 2 * k2_I + 2 * k3_I + k4_I)
    
plt.figure(figsize = (8, 4))
plt.plot(time, E_True, 'r', lw=2, label='Excitatory (E)')
plt.plot(time, I_True, 'b', lw=2, label='Inhibitory (I)')
plt.xlabel('Time (ms)')
plt.ylabel('Firing Rate')
plt.title('Wilson–Cowan Simulation')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize = (8, 4))
plt.plot(time, E_True, 'r', lw=2, label='Excitatory (E)')
plt.plot(time, P_E, 'b', lw=2, label='Input Signal')
plt.xlabel('Time (ms)')
plt.ylabel('Firing Rate / Input')
plt.title('External Input to E Neurons')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize = (8, 4))
plt.plot(E_True, I_True, 'r', lw=2)
plt.xlabel('E')
plt.ylabel('I')
plt.title('Wilson–Cowan Simulation')
plt.legend()
plt.grid()
plt.show()