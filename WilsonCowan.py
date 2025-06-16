import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------- シミュレーション設定 ----------
dt = 0.001  # シミュレーション刻み幅 (ms)
T = 300.0   # 総シミュレーション時間 (ms)

Step = int(T / dt)  # ステップ数
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


def Sin_Wave(t, Amp, Period, phase):
    """
    任意の時間軸 t 上に Sin 波を生成する。

    Parameters
    ----------
    t : np.ndarray
        時間軸（ms など）。dtype は float。
    amplitude : float, optional
        波の振幅 A。デフォルト 1.0。
    period : float, optional
        周期 T（t と同じ単位）。デフォルト 100.0。
    phase : float, optional
        初期位相 φ [rad]。デフォルト 0.0。

    Returns
    -------
    np.ndarray
        同じ長さの Sin 波列  A·sin(2π·t/T + φ)
    """

    return Amp * np.sin(2 * np.pi * t / Period + phase)

# ------外部刺激 P(t) ----
def P(x):
    return 3 + np.random.rand(1) * x * 100


# ------外部刺激 Q(t) ----
def Q(x):
    return 4 + x


# ----- 外部刺激信号生成 -----
P_E = np.ones(Step)
Q_I = np.ones(Step) * 4

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
    P_E[i] = Sin_Wave(i * dt, 2 , 50, 0)

    k1_E, k1_I = Wilson_Cowan(E_True[i]                  , I_True[i]                  , P_E[i], Q_I[i])
    k2_E, k2_I = Wilson_Cowan(E_True[i] + 0.5 * dt * k1_E, I_True[i] + 0.5 * dt * k1_I, P_E[i], Q_I[i])
    k3_E, k3_I = Wilson_Cowan(E_True[i] + 0.5 * dt * k2_E, I_True[i] + 0.5 * dt * k2_I, P_E[i], Q_I[i])
    k4_E, k4_I = Wilson_Cowan(E_True[i] +       dt * k3_E, I_True[i] +       dt * k3_I, P_E[i], Q_I[i])

    E_True[i + 1] = E_True[i] + (dt / 6) * (k1_E + 2 * k2_E + 2 * k3_E + k4_E)
    I_True[i + 1] = I_True[i] + (dt / 6) * (k1_I + 2 * k2_I + 2 * k3_I + k4_I)

plt.figure(figsize = (8, 4))
plt.plot(time, E_True)
plt.plot(time, P_E)
plt.show()