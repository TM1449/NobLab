import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ===========================================================
# 1. モデルとフィルタの設定: Izhikevich ニューロン + EnKF
# ===========================================================

# -----------------------------
# 1.1 真の Izhikevich パラメータ (ここでは a を推定対象)
# -----------------------------
# dv/dt = 0.04 v^2 + 5 v + 140 - u + I
# du/dt = a ( b v - u )
# スパイク後 (v >= 30 mV): v <- c, u <- u + d

a_true = 0.02         # 未知パラメータ (RS セル)
b      = 0.2
c      = -65
d      = 8

# -----------------------------
# 1.2 真の初期状態ベクトル [v, u]
# -----------------------------
v0 = -65.0
u0 = b * v0

vu00 = np.array([v0, u0])  # 初期状態 [v, u]

# -----------------------------
# 1.3 時間離散化とシミュレーション設定
# -----------------------------
dt      = 1e-4        # タイムステップ Δt [ms]
sqrt_dt = np.sqrt(dt)  # プロセスノイズ付与時のスケーリング √Δt
T       = 300.0        # 総シミュレーション時間 [ms]
N       = int(T / dt)  # ステップ数

# 入力信号用の時間配列
Input_Start = 10.0  # 開始時間 [ms]
Input_End   = T     # 終了時間 [ms]
dt_Plot   = 0.5   # プロット刻み幅 [ms]
Plot_Time = np.arange(N)
Amplitude = 10.0  # 刺激電流の振幅 [μA/cm^2]

# -----------------------------
# 1.4 ノイズ共分散
# -----------------------------
#   観測ノイズ R: 膜電位 v に適用
#   プロセスノイズ Q: [v, u] へのノイズ共分散
#   ランダムウォークノイズ Q3: a へのノイズ
R  = 0.0                       # 観測ノイズ分散 (mV^2)
Q  = np.diag([0.0, 0.0])     # v, u のプロセスノイズ
Q3 = 0.0                        # a を定数とする (ドリフトさせたい場合は小さい値を設定)

# -----------------------------
# 1.5 観測モデル行列 C
# -----------------------------
# v のみ観測する:  y = v + ノイズ
C = np.array([1.0, 0.0, 0.0])  # 形状 (1×3)

# -----------------------------
# 1.6 外部電流刺激 I(t)
# -----------------------------
def stimulus(t):
    """10 ms 以降に定常電流を印加するステップ入力"""
    return 10.0 if t >= 10.0 else 0.0

def Input_Signal(Time_Max_D, dt_Sim_D, Input_Start_D, Input_End_D, Amplitude_D):
    """
    刺激電流の入力信号を生成する関数
    - Time_Max: シミュレーション総時間 (ms)
    - dt_Sim: シミュレーション刻み幅 (ms)
    - Input_Start, Input_End: プロット範囲 (ms)
    - Amplitude: 刺激電流の振幅 (μA/cm^2)
    """
    # シミュレーション時間とステップ数の計算
    n_steps = int(np.ceil(Time_Max_D / dt_Sim_D))
    print(f"\nTotal simulation steps: {n_steps} (Time_max = {Time_Max_D} ms, dt ={dt_Sim_D} ms)\n")
    # 時間配列の生成
    t_sim = np.arange(n_steps) * dt_Sim_D
    I = np.zeros(n_steps)
    
    # 刺激電流の設定
    I[int(Input_Start_D / dt_Sim_D):int(Input_End_D / dt_Sim_D)] = Amplitude
    return I

# ===========================================================
# 2. 真値データ生成 (状態 & 雑音付き観測)
# ===========================================================

# -----------------------------
# 2.1 真の状態ベクトル x と観測 y を格納する配列を初期化
# -----------------------------
X_True = np.zeros((3, N + 1))   # 行: v, u, a
Y_Obs  = np.zeros(N + 1)


# -----------------------------
# 2.2 真の初期状態を設定
# -----------------------------
# x[0:2,0] に真の初期状態 [x00] をセットし、
# x[2,0] に真の推定したいパラメータをセット

# 初期状態
X_True[0:2, 0] = vu00  # 初期状態 [v, u]
X_True[2, 0] = a_true  # a の初期値 (定数)


# -----------------------------
# 2.3 真のプロセスノイズ wd と 観測ノイズ vd を生成
# -----------------------------
# - wd: 2行 × (N+1)列 の配列。各行がそれぞれ v, u 用のノイズ系列
#     wd = sqrt(Q) * 標準正規乱数 * √Δt
# - vd: (N+1)要素のベクトル。観測ノイズは v の観測に加わる

# プロセスノイズ・観測ノイズの事前サンプリング
wd = np.vstack((
    np.sqrt(Q[0, 0]) * np.random.randn(N + 1),
    np.sqrt(Q[1, 1]) * np.random.randn(N + 1),
))
vd = np.sqrt(R) * np.random.randn(N + 1)

"""ここまでOK"""

# -----------------------------
# 2.4 真の Izhikevich 系を離散化シミュレーション
# -----------------------------

Signal_I = Input_Signal(
    Time_Max_D = T,
    dt_Sim_D = dt,
    Input_Start_D = Input_Start,  # 刺激開始時間 (ms)
    Input_End_D = Input_End,       # 刺激終了時間 (ms)
    Amplitude_D = Amplitude    # 刺激電流の振幅 (μA/cm^2)
)

print("True Izhikevich neuron simulation started...\n")

# シミュレーション開始
for i in tqdm(range(N)):
    # i時刻の真の状態を取り出す
    v_k, u_k, a_k = X_True[:, i]

    I_True = Signal_I[i]  # 真の刺激電流

    # Izhikevich の微分方程式
    dv_dt = 0.04 * pow(v_k, 2) + 5.0 * v_k + 140.0 - u_k + I_True
    du_dt = a_k * (b * v_k - u_k)

    # オイラー積分 + プロセスノイズ
    v_next = v_k + dt * dv_dt + sqrt_dt * wd[0, i]
    u_next = u_k + dt * du_dt + sqrt_dt * wd[1, i]
    a_next = a_k                     # 定数

    # スパイクリセット
    if v_next >= 30.0:
        v_k      = 30.0              # スパイクを記録
        v_next   = c
        u_next   = u_next + d

    # 現在の状態を保存
    X_True[:, i] = [v_k, u_k, a_k]
    # 次の状態を保存
    X_True[:, i + 1] = [v_next, u_next, a_next]

    # 観測
    Y_Obs[i] = C @ X_True[:, i] + vd[i]

t_ds, Y_Plot = Plot_Time[::int(dt_Plot / dt)], Y_Obs[::int(dt_Plot / dt)]  # プロット用の観測データ
print(f"length of t_ds: {len(t_ds)}, length of Y_Plot: {len(Y_Plot)}")

plt.figure(figsize=(8, 4))
plt.plot(t_ds, Y_Plot[:600], label='Observed v')
plt.title('Observed membrane potential (v) with noise')
plt.xlabel('Time [ms]')
plt.ylabel('v [mV]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



"""ここから下はまだ未実装"""
for k in range(N):
    v_k, u_k, a_k = X_True[:, k]

    # Izhikevich の微分方程式
    I_k   = stimulus(k * dt)
    dv_dt = 0.04 * v_k**2 + 5.0 * v_k + 140.0 - u_k + I_k
    du_dt = a_k * (b * v_k - u_k)

    # オイラー積分 + プロセスノイズ
    v_next = v_k + dt * dv_dt + sqrt_dt * wd[0, k]
    u_next = u_k + dt * du_dt + sqrt_dt * wd[1, k]
    a_next = a_k                     # 定数

    # スパイクリセット
    if v_next >= 30.0:
        v_k      = 30.0              # スパイクを記録
        v_next   = c
        u_next   = u_next + d

    # 次の状態を保存
    X_True[0, k] = v_k
    X_True[:, k + 1] = [v_next, u_next, a_next]

    # 観測
    Y_Obs[k] = C @ X_True[:, k] + vd[k]

Y_Obs[N] = C @ X_True[:, N] + vd[N]

plt.figure(figsize=(8, 4))
plt.plot()


# ===========================================================
# 3. EnKF 初期化
# ===========================================================

n = 3          # 拡張状態次元 [v, u, a]
p = 1          # 観測次元
M = 300        # アンサンブルサイズ

# 3.1 初期アンサンブル (真値周りにランダム摂動)
x_ens = np.zeros((n, M))
for i in range(M):
    x_ens[0, i] = v0 + np.random.normal(0.0, 2.0)
    x_ens[1, i] = u0 + np.random.normal(0.0, 2.0)
    x_ens[2, i] = a_true + np.random.normal(0.0, 0.01)  # 小さな散らばり

# 3.2 推定結果格納用
x_hat         = np.zeros((n, N + 1))
a_ensemble_ts = np.zeros((M, N + 1))

# 3.3 拡張ノイズ共分散
Qe           = np.zeros((n, n))
Qe[0:2, 0:2] = Q
Qe[2, 2]     = Q3
Re           = R

# 一時配列
x_post = np.zeros((n, M))
y_pred = np.zeros((p, M))
innov  = np.zeros((p, M))


# ===========================================================
# 4. EnKF ループ: 解析 (Update) + 予報 (Forecast)
# ===========================================================
for k in range(N + 1):
    # ----------------------------- 解析ステップ
    ve = np.sqrt(Re) * np.random.randn(p, M)  # 観測摂動
    for i in range(M):
        y_pred[:, i] = C @ x_ens[:, i] + ve[:, i]

    x_mean = np.mean(x_ens, axis=1, keepdims=True)
    y_mean = np.mean(y_pred, axis=1, keepdims=True)
    Ex     = x_ens - x_mean
    Ey     = y_pred - y_mean

    Pxy = (Ex @ Ey.T) / (M - 1)
    Pyy = (Ey @ Ey.T) / (M - 1)

    K = Pxy @ np.linalg.inv(Pyy)  # カルマンゲイン

    for i in range(M):
        innov[:, i]  = Y_Obs[k] - y_pred[:, i]   # イノベーション
        x_post[:, i] = x_ens[:, i] + (K @ innov[:, i])

    x_hat[:, k]         = np.mean(x_post, axis=1)
    a_ensemble_ts[:, k] = x_post[2, :]

    # ----------------------------- 予報ステップ
    we = np.random.multivariate_normal(np.zeros(n), Qe, size=M).T

    for i in range(M):
        v_i, u_i, a_i = x_post[:, i]

        # 区間 [k*dt, (k+1)*dt) における刺激
        I_i = stimulus(k * dt)

        dv_dt = 0.04 * v_i**2 + 5.0 * v_i + 140.0 - u_i + I_i
        du_dt = a_i * (b * v_i - u_i)

        v_next = v_i + dt * dv_dt + sqrt_dt * we[0, i]
        u_next = u_i + dt * du_dt + sqrt_dt * we[1, i]
        a_next = a_i + sqrt_dt * we[2, i]   # Q3=0 なら定数

        if v_next >= 30.0:
            v_i    = 30.0
            v_next = c
            u_next = u_next + d

        # スパイクプロット用に更新した v_i を保持
        x_ens[0, i] = v_i
        x_ens[:, i] = [v_next, u_next, a_next]

# ===========================================================
# 5. 診断 & プロット (図は英語表記のまま)
# ===========================================================

time = np.linspace(0, T, N + 1)

# (1) Membrane potential
plt.figure(figsize=(8, 4))
plt.plot(time, X_True[0, :], 'r-', label='True v')
plt.plot(time, x_hat[0, :], 'b--', label='EnKF est.')
plt.xlabel('Time [ms]')
plt.ylabel('v [mV]')
plt.title('Izhikevich membrane potential estimation')
plt.legend()
plt.grid(True)

# (2) Recovery variable u
plt.figure(figsize=(8, 4))
plt.plot(time, X_True[1, :], 'r-', label='True u')
plt.plot(time, x_hat[1, :], 'b--', label='EnKF est.')
plt.xlabel('Time [ms]')
plt.ylabel('u')
