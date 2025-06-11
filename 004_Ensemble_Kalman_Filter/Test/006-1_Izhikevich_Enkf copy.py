import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------------------------------------
# 1. モデルとフィルタの設定: Izhikevich ニューロン + EnKF
# -----------------------------------------------------------

# 1.1 真の Izhikevich パラメータ (ここでは a を推定対象)
a_true = 0.02    # RS セルの a (推定したい未知パラメータ)
b = 0.2
c = -65.0
d = 8.0

# 1.2 真の初期状態 [v, u]
v0 = -65.0
u0 = b * v0

# 1.3 時間設定
dt_sim    = 1e-2        # ms
Time_max  = 200.0        # ms
N_steps   = int(np.ceil(Time_max / dt_sim))

# 1.4 ノイズ共分散
R  = 0.001                      # 観測ノイズ分散
Q  = np.diag([0.001, 0.001])   # 状態ノイズ
Q3 = 1e-8                       # パラメータノイズ (a は定数扱い)

# 1.5 観測行列 (v のみ観測)
C = np.array([[1.0, 0.0, 0.0]])  # 形状 (1×3)

# 1.6 刺激電流
def make_input(Time_max, dt, t_start, t_end, amp):
    n = int(np.ceil(Time_max / dt))
    I = np.zeros(n)
    I[int(t_start/dt):int(t_end/dt)] = amp
    return I

Signal_I = make_input(Time_max, dt_sim, 3.0, Time_max, 30.0)

# -----------------------------------------------------------
# 2. 真値データ生成 (状態 & 観測)
# -----------------------------------------------------------

V_true = np.zeros(N_steps)
U_true = np.zeros(N_steps)
A_true = np.zeros(N_steps)
Y_obs  = np.zeros(N_steps)

V_true[0], U_true[0], A_true[0] = v0, u0, c

# プロセスノイズ・観測ノイズ
wd = np.vstack((
    np.sqrt(Q[0,0]) * np.random.randn(N_steps),
    np.sqrt(Q[1,1]) * np.random.randn(N_steps),
))
vd = np.sqrt(R) * np.random.randn(N_steps)

for k in range(N_steps-1):
    # 真のモデルをオイラー積分
    dv = 0.04*V_true[k]**2 + 5*V_true[k] + 140 - U_true[k] + Signal_I[k]
    du = a_true * (b*V_true[k] - U_true[k])
    V_true[k+1] = V_true[k] + dv*dt_sim + np.sqrt(dt_sim)*wd[0,k]
    U_true[k+1] = U_true[k] + du*dt_sim + np.sqrt(dt_sim)*wd[1,k]
    A_true[k+1] = c
    # スパイクリセット
    if V_true[k+1] >= 30:
        V_true[k] = 30
        V_true[k+1] = c
        U_true[k+1] += d
    # 観測
    Y_obs[k] = C @ np.array([V_true[k], U_true[k], A_true[k]]) + vd[k]
# 最終ステップの観測
Y_obs[-1] = C @ np.array([V_true[-1], U_true[-1], A_true[-1]]) + vd[-1]

# -----------------------------------------------------------
# 3. EnKF 初期化
# -----------------------------------------------------------

n = 3   # 拡張状態次元 [v, u, a]
p = 1   # 観測次元
M = 100  # アンサンブルサイズ

# 3.1 初期アンサンブル (真値周りに摂動)
X_Ens = np.zeros((n, M))
for i in range(M):
    X_Ens[:, i] = [v0 + np.random.randn(),
                   u0 + np.random.randn(),
                   c + np.random.randn()*0.01]

# 3.2 解析後アンサンブル X_ef を初期化
X_ef = X_Ens.copy()

# 3.3 結果格納用
X_Hat = np.zeros((n, N_steps))     # 各時刻の ensemble mean
A_ts  = np.zeros((M, N_steps))     # 各メンバーの a 推定値

# 3.4 拡張ノイズ共分散
Qe = np.zeros((n,n))
Qe[0:2,0:2] = Q
Qe[2,2]     = Q3
Re = R

# -----------------------------------------------------------
# 4. EnKF メインループ (予報 → 分析)
# -----------------------------------------------------------
for k in tqdm(range(N_steps)):
    # ----------------------
    # (1) 予報ステップ
    # ----------------------
    # プロセスノイズ
    we = np.random.multivariate_normal(np.zeros(n), Qe, size=M).T  # shape (n,M)
    X_f = np.zeros_like(X_ef)  # 予報アンサンブル格納

    for i in range(M):
        v_i, u_i, a_i = X_ef[:, i]
        I_i = Signal_I[k]
        dv = 0.04*v_i**2 + 5*v_i + 140 - u_i + I_i
        du = a_i * (b*v_i - u_i)
        v_next = v_i + dv*dt_sim + np.sqrt(dt_sim)*we[0,i]
        u_next = u_i + du*dt_sim + np.sqrt(dt_sim)*we[1,i]
        a_next = a_i             + np.sqrt(dt_sim)*we[2,i]  # Q3=0 なら定数保持
        # スパイクリセット
        if v_next >= 30:
            v_i = 30
            v_next = c
            u_next += d
        X_f[:,i] = [v_next, u_next, a_next]

    # ----------------------
    # (2) 分析ステップ
    # ----------------------
    # 観測摂動
    ve = np.sqrt(Re) * np.random.randn(p, M)
    # 観測予測
    Y_pred = C @ X_f + ve  # shape (1,M)

    # 共分散行列計算
    x_mean = X_f.mean(axis=1, keepdims=True)
    y_mean = Y_pred.mean(axis=1, keepdims=True)
    Ex = X_f - x_mean
    Ey = Y_pred - y_mean

    Pxy = (Ex @ Ey.T) / (M-1)
    Pyy = (Ey @ Ey.T) / (M-1)

    # カルマンゲイン
    K = Pxy @ np.linalg.inv(Pyy)

    # アンサンブル更新
    for i in range(M):
        innov = Y_obs[k] - Y_pred[:,i]
        X_ef[:,i] = X_f[:,i] + K.flatten() * innov  # shape 適合

    # 結果格納
    X_Hat[:, k] = X_ef.mean(axis=1)
    A_ts[:, k]  = X_ef[2, :]

# -----------------------------------------------------------
# 5. 結果のプロット (英語表記のまま)
# -----------------------------------------------------------
time = np.arange(N_steps) * dt_sim

plt.figure(figsize=(8,4))
plt.plot(time, V_true, 'r-', label='True v')
plt.plot(time, X_Hat[0], 'b--', label='EnKF est.')
plt.xlabel('Time [ms]'); plt.ylabel('v [mV]')
plt.title('Izhikevich membrane potential estimation')
plt.legend(); plt.grid(True); plt.show()

plt.figure(figsize=(8,4))
plt.plot(time, U_true, 'r-', label='True u')
plt.plot(time, X_Hat[1], 'b--', label='EnKF est.')
plt.xlabel('Time [ms]'); plt.ylabel('u')
plt.title('Izhikevich recovery variable estimation')
plt.legend(); plt.grid(True); plt.show()

plt.figure(figsize=(8,4))
plt.plot(time, A_true, 'r-', label='True a')
plt.plot(time, X_Hat[2], 'b--', label='EnKF est.')
plt.xlabel('Time [ms]'); plt.ylabel('a')
plt.title('Izhikevich parameter a estimation')
plt.legend(); plt.grid(True); plt.show()
