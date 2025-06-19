import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ---------- シミュレーション設定 ----------
dt      = 0.005             # シミュレーション刻み幅 [ms]
T_total = 300.0            # 総シミュレーション時間 [ms]
Step    = int(T_total / dt) # ステップ数
time    = np.arange(Step) * dt

# ---------- コラム数設定 ----------
Nc = 4  # 任意のコラム数に変更可能 (例: 2, 3, 4 ...)

# ---------- Wilson–Cowan パラメータ ----------
tau_E, tau_I = 20.0, 10.0   # 時定数 [ms]
w_EE = 16.0                  # ローカル E→E 結合強度
w_EI = 26.0                  # ローカル E→I 結合強度
w_IE = 20.0                  # ローカル I→E 結合強度
w_II = 1.0                   # ローカル I→I 結合強度
r_E, r_I = 0.0, 0.0          # 不応期係数

# ---------- 遠隔結合（Exc→Exc）の設定 ----------
C_inter = 0.7  # グローバル結合スケール因子
# 全結合（自己結合なし）の行列を生成
W_inter = np.ones((Nc, Nc)) - np.eye(Nc)

# ---------- シグモイド関数 ----------
a_E, theta_E = 1.0, 5.0
a_I, theta_I = 1.0, 20.0

def sigmoid(x, a, th):
    return 1.0 / (1.0 + np.exp(-a * (x - th)))

S_E = lambda x: sigmoid(x, a_E, theta_E)
S_I = lambda x: sigmoid(x, a_I, theta_I)

# ---------- 入力信号作成 ----------
def input_signal(idx, dt, amp, period, width=None, base=0.0):
    t = np.asarray(idx) * dt
    sig = np.where((t % period) < width, amp, 0.0) + base
    return sig

IDX = np.arange(Step)
P_E = input_signal(IDX, dt, amp=3.0, period=20.0, width=10.0, base=2.0)
Q_I = np.ones(Step) * 7.0

# ---------- 真の軌道生成 (4次RK) ----------
E_true = np.zeros((Nc, Step))
I_true = np.zeros((Nc, Step))
# 初期値 (小さく揺らす)
E_true[:, 0] = 0.5 + 0.1 * np.random.randn(Nc)
I_true[:, 0] = 0.4 + 0.1 * np.random.randn(Nc)

# 巡回微分方程式
def wc_deriv(E, I, P_t, Q_t, wEE, wEI, wIE, wII, C, Wint):
    inter = C * (Wint @ E)
    dE = (-E + (1 - r_E*E) * S_E(wEE*E - wEI*I + P_t + inter)) / tau_E
    dI = (-I + (1 - r_I*I) * S_I(wIE*E - wII*I + Q_t)) / tau_I
    return dE, dI

# RK4 統合
for k in tqdm(range(Step-1), desc='Truth RK4 Multi-column'):
    E, I = E_true[:, k], I_true[:, k]
    P_t, Q_t = P_E[k], Q_I[k]
    k1_E, k1_I = wc_deriv(E, I, P_t, Q_t, w_EE, w_EI, w_IE, w_II, C_inter, W_inter)
    k2_E, k2_I = wc_deriv(E+0.5*dt*k1_E, I+0.5*dt*k1_I, P_t, Q_t, w_EE, w_EI, w_IE, w_II, C_inter, W_inter)
    k3_E, k3_I = wc_deriv(E+0.5*dt*k2_E, I+0.5*dt*k2_I, P_t, Q_t, w_EE, w_EI, w_IE, w_II, C_inter, W_inter)
    k4_E, k4_I = wc_deriv(E+dt*k3_E, I+dt*k3_I, P_t, Q_t, w_EE, w_EI, w_IE, w_II, C_inter, W_inter)
    E_true[:, k+1] = E + (dt/6)*(k1_E + 2*k2_E + 2*k3_E + k4_E)
    I_true[:, k+1] = I + (dt/6)*(k1_I + 2*k2_I + 2*k3_I + k4_I)

# ---------- 時系列プロット (各コラム) ----------
plt.figure(figsize=(12, 8))
for i in range(Nc):
    plt.plot(time, E_true[i], label=f'E{i+1}')
plt.xlabel('Time [ms]')
plt.ylabel('E activity')
plt.title(f'Excitatory activities (Nc={Nc})')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
for i in range(Nc):
    plt.plot(time, I_true[i], label=f'I{i+1}')
plt.xlabel('Time [ms]')
plt.ylabel('I activity')
plt.title(f'Inhibitory activities (Nc={Nc})')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# ---------- Heatmap with seaborn ----------
sns.set(style='white')
plt.figure(figsize=(10, 6))
sns.heatmap(E_true, cmap='viridis', cbar_kws={'label': 'E activity'},
            yticklabels=[f'Col{i+1}' for i in range(Nc)], xticklabels=Step//10)
plt.xlabel('Time step')
plt.ylabel('Column')
plt.title(f'Excitatory Heatmap (Nc={Nc})')
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(I_true, cmap='plasma', cbar_kws={'label': 'I activity'},
            yticklabels=[f'Col{i+1}' for i in range(Nc)], xticklabels=Step//10)
plt.xlabel('Time step')
plt.ylabel('Column')
plt.title(f'Inhibitory Heatmap (Nc={Nc})')
plt.show()
