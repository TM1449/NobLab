import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# シミュレーション時間とステップ数の計算
dt = 0.001  # シミュレーション刻み幅 (ms)
T = 300.0   # 総シミュレーション時間 (ms)

Step = int(T / dt)  # ステップ数
time = np.arange(Step) * dt # 時間配列

# ----- パラメータ -----
tau_E = 20.0  # 興奮性ニューロンの時間定数 (ms)
tau_I = 10.0  # 抑制性ニューロンの時間定数 (ms)

w_EE = 16.0  # E-E 接続の重み
w_EI = 26.0  # E-I 接続の重み

w_IE = 20.0  # I-E 接続の重み
w_II = 1.0   # I-I 接続の重み

r_E = 0.0  # E ニューロンの回復率
r_I = 0.0  # I ニューロンの回復率

I_i = 7.0  # 外部入力の強さ

# ----- 時変刺激 P_ext(t) -----
P_base = 2.0  # 基本刺激
pulseA = 3.0  # パルスの高さ（+3）
pulseWidth = 10.0  # パルス幅 (ms)
period = 20.0  # パルス周期 (ms)

def Pulse(t_ms):
    """外部入力 を計算する関数"""

    # 周期のどの部分かを計算
    phase = np.mod(t_ms, period)
    # パルスのオン/オフを判定
    pulseon = phase < pulseWidth

    # パルスがオンのときは pulseA を返し、オフのときは 0 を返す
    return P_base + pulseA * pulseon

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

# ----- 初期状態（高活動） -----
E = np.zeros(Step)
I = np.zeros(Step)

I_E = np.zeros(Step)  # 外部入力の配列
I_I = np.zeros(Step)  # I ニューロンの外部入力

E[0] = 0.5  # E ニューロンの初期活動
I[0] = 0.4  # I ニューロンの初期活動

# ----- オイラー積分 -----  
for t in tqdm(range(Step - 1)):
    t_ms = time[t]
    I_E[t] = Pulse(t_ms)  # 次のステップの外部入力
    I_I[t] = I_i  # I ニューロンの外部入力は一定

    dE = (-E[t] + (1 - r_E * E[t]) * S_E(w_EE * E[t] - w_EI * I[t] + I_E[t])) / tau_E
    dI = (-I[t] + (1 - r_I * I[t]) * S_I(w_IE * E[t] - w_II * I[t] + I_I[t])) / tau_I

    E[t + 1] = E[t] + dt * dE
    I[t + 1] = I[t] + dt * dI

    I_E[t+1] = Pulse(t_ms)  # 次のステップの外部入力
    I_I[t+1] = I_i  # I ニューロンの外部入力は一定

# ----- プロット -----
plt.figure(figsize=(8, 4))
plt.plot(time, E, 'r', lw=2, label='Excitatory (E)')
plt.plot(time, I, 'b', lw=2, label='Inhibitory (I)')
plt.xlabel('Time (ms)')
plt.ylabel('Firing Rate')
plt.title('Wilson–Cowan Simulation (pulsed input, high-activity start)')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(time, E, 'r', lw=2, label='Excitatory (E)')
plt.plot(time, I_E, 'g', lw=2, label='External Input (I_E)')
plt.xlabel('Time (ms)')
plt.ylabel('Firing Rate / Input')
plt.title('External Input to E Neurons')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(time, I, 'b', lw=2, label='Inhibitory (I)')
plt.plot(time, I_I, 'g', lw=2, label='External Input (I_I)')
plt.xlabel('Time (ms)')
plt.ylabel('Firing Rate / Input')
plt.title('External Input to I Neurons')
plt.legend()
plt.grid()
plt.show()