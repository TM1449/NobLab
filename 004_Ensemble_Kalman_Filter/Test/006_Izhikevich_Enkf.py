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

a      = 0.02         # 未知パラメータ (RS セル)
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
dt_sim      = 1e-3        # タイムステップ Δt [ms]
sqrt_dt = np.sqrt(dt_sim)  # プロセスノイズ付与時のスケーリング √Δt
Time_max       = 50.0        # 総シミュレーション時間 [ms]
Step_sim       = int(np.ceil(Time_max / dt_sim))  # ステップ数

# -----------------------------
# 1.4 ノイズ共分散
# -----------------------------
#   観測ノイズ R: 膜電位 v に適用
#   プロセスノイズ Q: [v, u] へのノイズ共分散
#   ランダムウォークノイズ Q3: a へのノイズ
R  = 0.01                      # 観測ノイズ分散 (mV^2)
Q  = np.diag([0.001, 0.001])     # v, u のプロセスノイズ
Q3 = 0.0                        # a を定数とする (ドリフトさせたい場合は小さい値を設定)

# -----------------------------
# 1.5 観測モデル行列 C
# -----------------------------
# v のみ観測する:  y = v + ノイズ
C = np.array([1.0, 0.0, 0.0])  # 形状 (1×3)

# -----------------------------
# 1.6 外部電流刺激 I(t)
# -----------------------------
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
    I = np.zeros(n_steps)
    
    # 刺激電流の設定
    I[int(Input_Start_D / dt_Sim_D):int(Input_End_D / dt_Sim_D)] = Amplitude
    return I

# 入力信号用の時間配列
Input_Start = 3.0  # 開始時間 [ms]
Input_End   = Time_max     # 終了時間 [ms]

dt_Plot   = 1e-3   # プロット刻み幅 [ms]
Step_plot = int(np.ceil(Time_max / dt_sim))
Amplitude = 30.0  # 刺激電流の振幅 [μA/cm^2]

# ===========================================================
# 2. 真値データ生成 (状態 & 雑音付き観測)
# ===========================================================

# -----------------------------
# 2.1 真の状態ベクトル x と観測 y を格納する配列を初期化
# -----------------------------
V_True = np.zeros(Step_sim)  # 行: v, u, a
V_True[0] = v0  # 初期状態 v

U_True = np.zeros(Step_sim)  # 行: v, u, a
U_True[0] = u0  # 初期状態 u

A_True = np.zeros(Step_sim)  # 行: v, u, a
A_True[0] = a

X_True = np.zeros((3, Step_sim))   # 行: v, u, a
Y_Obs  = np.zeros(Step_sim)


# -----------------------------
# 2.2 真の初期状態を設定
# -----------------------------
# x[0:2,0] に真の初期状態 [x00] をセットし、
# x[2,0] に真の推定したいパラメータをセット

# 初期状態
X_True[0:2, 0] = vu00  # 初期状態 [v, u]
X_True[2, 0] = a  # a の初期値 (定数)


# -----------------------------
# 2.3 真のプロセスノイズ wd と 観測ノイズ vd を生成
# -----------------------------
# - wd: 2行 × (N+1)列 の配列。各行がそれぞれ v, u 用のノイズ系列
#     wd = sqrt(Q) * 標準正規乱数 * √Δt
# - vd: (N+1)要素のベクトル。観測ノイズは v の観測に加わる

# プロセスノイズ・観測ノイズの事前サンプリング
wd = np.vstack((
    np.sqrt(Q[0, 0]) * np.random.randn(Step_sim + 1),
    np.sqrt(Q[1, 1]) * np.random.randn(Step_sim + 1),
))
vd = np.sqrt(R) * np.random.randn(Step_sim + 1)

"""ここまでOK"""

# -----------------------------
# 2.4 真の Izhikevich 系を離散化シミュレーション
# -----------------------------

Signal_I = Input_Signal(
    Time_Max_D = Time_max,
    dt_Sim_D = dt_sim,  # シミュレーション刻み幅 (ms)
    Input_Start_D = Input_Start,  # 刺激開始時間 (ms)
    Input_End_D = Input_End,       # 刺激終了時間 (ms)
    Amplitude_D = Amplitude    # 刺激電流の振幅 (μA/cm^2)
)

print("True Izhikevich neuron simulation started...\n")

for j in tqdm(range(0, Step_sim - 1)):
    # j時刻の真の状態を取り出す
    dv = 0.04 * pow(V_True[j] ,2) + 5 * V_True[j] \
        + 140 - U_True[j] + Signal_I[j]
    du = a * (b * V_True[j] - U_True[j])

    # オイラー積分 + プロセスノイズ
    V_True[j + 1] = V_True[j] + dv * dt_sim + sqrt_dt * wd[0, j]
    U_True[j + 1] = U_True[j] + du * dt_sim + sqrt_dt * wd[1, j]

    A_True[j + 1] = a  # a は定数

    # スパイクリセット
    if V_True[j + 1] >= 30:
        V_True[j] = 30
        V_True[j + 1] = c
        U_True[j + 1] = U_True[j + 1] + d
    
    # 現在の状態を保存
    X_True[:, j] = [V_True[j], U_True[j], A_True[j]]
    # 次の状態を保存
    X_True[:, j + 1] = [V_True[j + 1], U_True[j + 1], A_True[j + 1]]
    # 観測
    Y_Obs[j] = C @ X_True[:, j] + vd[j]

# 最後の時刻の観測
Y_Obs[-1] = C @ np.array([V_True[-1], U_True[-1], A_True[-1]]) + vd[-1]

Plot_List = np.arange(Step_sim) * dt_sim  # プロット用の時間配列

t_ds = Plot_List[::int(dt_Plot / dt_sim)]  # プロット用の観測データ
V_Plot = V_True[::int(dt_Plot / dt_sim)]  # プロット用の膜電位
U_Plot = U_True[::int(dt_Plot / dt_sim)]  # プロット用の回復変数
Input_Plot = Signal_I[::int(dt_Plot / dt_sim)]  # プロット用の入力信号

Y_Obs_Plot = Y_Obs[::int(dt_Plot / dt_sim)]  # プロット用の観測データ

# ===========================================================
# 3. EnKF 初期化
# ===========================================================

n = 3          # 拡張状態次元 [v, u, a]
p = 1          # 観測次元
M = 30        # アンサンブルサイズ

# 3.1 初期アンサンブル (真値周りにランダム摂動)
X_Ens = np.zeros((n, M))

for i in range(M):
    X_Ens[0, i] = v0 + np.random.normal(0.0, 1.0) # 真の初期値 v0 に小さな摂動
    X_Ens[1, i] = u0 + np.random.normal(0.0, 1.0) # 真の初期値 u0 に小さな摂動
    X_Ens[2, i] = a + np.random.normal(0.0, 0.01)  # 小さな散らばり

# 3.2 推定結果格納用
# X_Hat: アンサンブル平均 [v, u, a]を時系列で保存
X_Hat         = np.zeros((n, Step_sim))
# A_ensemble_ts: 各メンバーの a 推定値を時系列で保存
A_ensemble_ts = np.zeros((M, Step_sim))

# 3.3 拡張ノイズ共分散
Qe           = np.zeros((n, n))
Qe[0:2, 0:2] = Q
Qe[2, 2]     = Q3
Re           = R

# Enkfで用いる一時配列
# X_ef : 解析後のアンサンブルを一時的に格納
X_ef    = X_Ens.copy()  # 初期アンサンブルをコピー
# Y_Pred : 観測予測値を一時的に格納
Y_Pred = np.zeros((p, M))
# Inov: イノベーション（観測誤差）を一時的に格納
Innov  = np.zeros((p, M))


# ===========================================================
# 4. EnKF ループ: 解析 (Update) + 予報 (Forecast)
# ===========================================================
for k in tqdm(range(Step_sim)):
    # -----------------------------
    # 4.1 分析ステップ (Analysis)：観測更新
    # -----------------------------

    ve = np.sqrt(Re) * np.random.randn(p, M)  # 観測摂動

    for i in range(M):
        Y_Pred[:, i] = C @ X_ef[:, i] + ve[:, i]

    # アンサンブル平均を計算
    x_mean = np.mean(X_ef, axis=1, keepdims=True)
    y_mean = np.mean(Y_Pred, axis=1, keepdims=True)
    Ex     = X_ef - x_mean
    Ey     = Y_Pred - y_mean

    # 共分散行列の計算
    Pxy = (Ex @ Ey.T) / (M - 1)
    Pyy = (Ey @ Ey.T) / (M - 1)

    # カルマンゲインの計算
    K = Pxy @ np.linalg.inv(Pyy)  # カルマンゲイン

    # 観測データを使って更新
    for i in range(M):
        # 観測誤差
        Innov[:, i]  = Y_Obs[k] - Y_Pred[:, i]   # イノベーション
        # アンサンブルメンバーの更新
        X_ef[:, i] = X_ef[:, i] + (K @ Innov[:, i])

    # アンサンブル平均を推定値として格納
    X_Hat[:, k]         = np.mean(X_ef, axis=1)
    A_ensemble_ts[:, k] = X_ef[2, :]

    # ----------------------------- 
    # 予報ステップ
    # -----------------------------
    we = np.random.multivariate_normal(np.zeros(n), Qe, size=M).T

    for i in range(M):
        v_i, u_i, a_i = X_ef[:, i]

        I_i = Signal_I[k]  # 刺激電流
        v_dt = 0.04 * pow(v_i, 2) + 5.0 * v_i + 140.0 - u_i + I_i
        u_dt = a_i * (b * v_i - u_i)

        v_next = v_i + dt_sim * v_dt + sqrt_dt * we[0, i]
        u_next = u_i + dt_sim * u_dt + sqrt_dt * we[1, i]
        a_next = a_i + sqrt_dt * we[2, i]   # Q3=0 なら定数

        if v_next >= 30.0:
            v_i    = 30.0
            v_next = c
            u_next = u_next + d

        # スパイクプロット用に更新した v_i を保持
        X_Ens[:, i] = [v_next, u_next, a_next]

# ===========================================================
# 5. 診断 & プロット (図は英語表記のまま)
# ===========================================================

time = np.arange(Step_sim) * dt_sim  # 時間配列 [ms]

# (1) Membrane potential
plt.figure(figsize=(8, 4))
plt.plot(time, V_True[:30001], 'r-', label='True v')
plt.plot(time, X_Hat[0, :30001], 'b--', label='EnKF est.')
plt.xlabel('Time [ms]')
plt.ylabel('v [mV]')
plt.title('Izhikevich membrane potential estimation')
plt.legend()
plt.grid(True)
plt.show()

# (2) Recovery variable u
plt.figure(figsize=(8, 4))
plt.plot(time, U_True[:30001], 'r-', label='True u')
plt.plot(time, X_Hat[1, :30001], 'b--', label='EnKF est.')
plt.xlabel('Time [ms]')
plt.ylabel('u')
plt.title('Izhikevich recovery variable estimation')
plt.legend()
plt.grid(True)
plt.show()

# (3) Parameter a
plt.figure(figsize=(8, 4))
plt.plot(time, A_True[:30001], 'r-', label='True a')
plt.plot(time, X_Hat[2,:30001], 'b--', label='EnKF est.')
plt.xlabel('Time [ms]')
plt.ylabel('a')
plt.title('Izhikevich parameter a estimation')
plt.legend()
plt.grid(True)
plt.show()

# (4) State estimation error
state_error = np.sqrt((V_True - X_Hat[0, :])**2 + \
        (U_True - X_Hat[1, :])**2)
plt.figure(figsize=(8, 4))
plt.plot(time, state_error, 'b-', linewidth=1.5)
plt.xlabel('Time [ms]')
plt.ylabel('State estimation error')
plt.title('State estimation error by EnKF')
plt.grid(True)
plt.show()