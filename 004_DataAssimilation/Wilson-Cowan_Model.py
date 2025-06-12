import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----- 時間設定 -----
dt   = 0.001            # [ms]
T    = 300            # 総シミュレーション時間 [ms]
time = np.arange(0, T + dt, dt)   # 0〜T まで dt 刻み
print(time)
time2 = np.arange(int(T / dt))  # 時間配列
print(time2)
N    = time.size

# ----- パラメータ -----
tau_E = 20.0;   tau_I = 10.0
w_EE  = 16.0;   w_EI  = 26.0;  w_IE = 20.0;  w_II = 1.0
r_E   = 0.0;    r_I   = 0.0
Qext  = 7.0

# ----- 時変刺激 P_ext(t) -----
P_base      = 2.0
pulseA      = 3.0        # パルスの高さ（+3）
pulseWidth  = 20.0       # パルス幅 20 ms
period      = 200.0      # 200 ms ごとにパルス

def P_ext(t_ms):
    """外部入力 P_ext(t) [ms]"""
    return P_base + pulseA * (np.mod(t_ms, period) < pulseWidth)

# ----- シグモイド関数 -----
a_E = 1.0; theta_E = 5.0
a_I = 1.0; theta_I = 20.0
S_E = lambda x: 1.0 / (1.0 + np.exp(-a_E * (x - theta_E)))
S_I = lambda x: 1.0 / (1.0 + np.exp(-a_I * (x - theta_I)))

# ----- 初期状態（高活動） -----
E = np.zeros(N)
I = np.zeros(N)
E[0] = 0.5
I[0] = 0.4

# ----- オイラー積分 -----
for k in tqdm(range(N - 1)):
    t_ms = time[k]
    Pin  = P_ext(t_ms)

    dE = (-E[k] + (1 - r_E * E[k]) * S_E(w_EE * E[k] - w_EI * I[k] + Pin)) / tau_E
    dI = (-I[k] + (1 - r_I * I[k]) * S_I(w_IE * E[k] - w_II * I[k] + Qext)) / tau_I

    E[k + 1] = E[k] + dt * dE
    I[k + 1] = I[k] + dt * dI

# ----- プロット -----
plt.figure(figsize=(8, 4))
plt.plot(time, E, 'r', lw=2, label='Excitatory (E)')
plt.plot(time, I, 'b', lw=2, label='Inhibitory (I)')
plt.xlabel('Time (ms)')
plt.ylabel('Firing Rate')
plt.title('Wilson–Cowan Simulation (pulsed input, high-activity start)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
