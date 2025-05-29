import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. モデル＆フィルタ設定
# -----------------------------
eps = 1.0                        # 真のパラメータ ε（未知とみなす）
x00 = np.array([0.2, 0.1])       # 真の初期状態ベクトル [x1, x2]

# 時間設定
dt = 0.1                         # タイムステップ Δt
d2 = np.sqrt(dt)                 # ノイズのスケーリング用 sqrt(Δt)

#print("d2の値:", d2)  # d2 の値を確認:0.31

T = 50.0                         # シミュレーション全体時間
N = int(T / dt)                  # ステップ数

# ノイズ共分散
R = 0.01                         # 観測ノイズ分散
Q = np.array([[0.026, 0.0],      # x1, x2 のシステムノイズ共分散行列
                [0.0,   0.01]])
Q3 = 1e-5                  # ε のランダムウォーク分散

# 観測行列（x2 のみ観測：これは絶対に変えない！）
C = np.array([0.0, 1.0, 0.0])

# -----------------------------
# 2. 真のデータ生成
# -----------------------------
x = np.zeros((3, N+1))           # 真の状態 [x1, x2, ε]
y = np.zeros(N+1)                # 観測データ

# 初期状態のセット
x[0:2, 0] = x00
x[2, 0] = eps

# プロセスノイズと観測ノイズを用意
vd = np.sqrt(R) * np.random.randn(N+1)                            # 観測ノイズ
wd = np.vstack((np.sqrt(Q[0,0]) * np.random.randn(N+1),         # システムノイズ
                np.sqrt(Q[1,1]) * np.random.randn(N+1)))        #wd ((2, N+1) の形状)

#print("wdの形状:", wd.shape)  # wd の形状確
#print("ノイズ:", wd[:,0] * d2)  # wd の値を確認


# 真の力学系をシミュレーションしつつ観測を取得
for k in range(N):

    # x1 の更新（更新式＋ノイズ）
    x[0, k+1] = x[0, k] + dt * x[1, k] \
        + d2 * wd[0, k]
    
    # x2 の更新（Van der Pol の項）
    vdp = eps * (1 - pow(x[0, k], 2)) * x[1, k] - x[0, k]

    # x2 の更新（更新式＋ノイズ）
    x[1, k+1] = x[1, k] + dt * vdp \
        + d2 * wd[1, k]
    
    # ε（推定パラメータ：x[2,:]） は定数としてそのまま保持
    x[2, k+1] = x[2, k]

    # 観測（x2 のみ）にノイズを加える
    y[k] = C @ x[:, k] + vd[k]

# 最終ステップの観測
y[N] = C @ x[:, N] + vd[N]


# -----------------------------
# 3. EnKF 初期化
# -----------------------------
n = 3      # 状態次元数（x1, x2, ε）
p = 1      # 観測次元数
M = 500    # アンサンブルメンバー数

# 初期アンサンブル（平均0、分散0.5）
#xep = np.sqrt(0.5) * np.random.randn(n, M)
xep = np.random.normal(0, 0.5, (n, M))  # 平均0、分散0.5の正規分布

# 推定結果格納用
xhat = np.zeros((n, N+1))

eps_ensemble = np.zeros((M, N+1))  # ε のアンサンブル値保存用


# フィルタ用の拡張システムノイズ共分散行列
Qe = np.zeros((n, n))
Qe[0:2, 0:2] = Q
Qe[2, 2] = Q3                              # ε 用に小さいノイズ

Re = R                                       # 観測ノイズ分散

#ここまでOK

# 一時的に使う配列群
xef = np.zeros((n, M))       # 事後アンサンブル
yep = np.zeros((p, M))       # 観測予測
nu  = np.zeros((p, M))       # イノベーション
Ex  = np.zeros((n, M))       # 状態偏差
Ey  = np.zeros((p, M))       # 観測偏差


# -----------------------------
# 4. 推定ループ
# -----------------------------
for k in range(N+1):
    # --- 分析ステップ（観測更新） ---
    # 観測予測を各メンバーで生成
    ve = np.sqrt(Re) * np.random.randn(p, M)
    for i in range(M):
        yep[:, i] = C @ xep[:, i] + ve[:, i] #初期Ensembleに観測ノイズを加える
    
    # アンサンブル平均を計算
    x_mean = np.mean(xep, axis=1, keepdims=True)
    y_mean = np.mean(yep, axis=1, keepdims=True)
    
    # 偏差を計算
    Ex = xep - x_mean
    Ey = yep - y_mean
    
    # 共分散行列を計算
    Pxy = (Ex @ Ey.T) / (M - 1)
    Pyy = (Ey @ Ey.T) / (M - 1)
    
    # カルマンゲインの計算
    Kt = Pxy @ np.linalg.inv(Pyy)
    
    # 各メンバーを実際の観測 y[k] で更新
    for i in range(M):
        nu[:, i]  = y[k] - yep[:, i]
        xef[:, i] = xep[:, i] + Kt @ nu[:, i]
    
    # アンサンブル平均をフィルタ推定値として保存
    xhat[:, k] = np.mean(xef, axis=1)
    # ε のアンサンブル値を保存
    eps_ensemble[:, k] = xef[2, :]
    
    # --- 予測ステップ（時間更新） ---
    we = np.random.multivariate_normal(np.zeros(n), Qe, size=M).T
    for i in range(M):
        # x1 の予測
        xep[0, i] = xef[0, i] + dt * xef[1, i] + d2 * we[0, i]
        # x2 の予測
        vdp_i = xef[2, i] * (1 - xef[0, i]**2) * xef[1, i] - xef[0, i]
        xep[1, i] = xef[1, i] + dt * vdp_i + d2 * we[1, i]
        # ε のランダムウォーク予測
        xep[2, i] = xef[2, i] + d2 * we[2, i]


# -----------------------------
# 5. 推定誤差の計算
# -----------------------------
# x1,x2 のユークリッド誤差
E_t = np.sqrt((x[0, :] - xhat[0, :])**2 + (x[1, :] - xhat[1, :])**2)

# プロット用の時間軸
time = np.linspace(0, T, N+1)


# -----------------------------
# 6. 結果プロット
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(time, x[0, :], 'r-', label='True x1', linewidth=1.5)
plt.plot(time, xhat[0, :], 'b--', label='EnKF estimate', linewidth=1.5)
plt.xlabel('Time t (sec)')
plt.ylabel('x1')
plt.title('Estimation of x1 by EnKF')
plt.axis([0, T, -4, 4])
plt.grid(True)
plt.legend()

plt.figure(figsize=(8, 5))
plt.plot(time, x[1, :], 'r-', label='True x2', linewidth=1.5)
plt.plot(time, xhat[1, :], 'b--', label='EnKF estimate', linewidth=1.5)
plt.xlabel('Time t (sec)')
plt.ylabel('x2')
plt.title('Estimation of x2 by EnKF')
plt.axis([0, T, -4, 4])
plt.grid(True)
plt.legend()

plt.figure(figsize=(8, 4))
plt.plot(time, E_t, 'b-', linewidth=1.5)
plt.xlabel('Time t (sec)')
plt.ylabel('Error E_t')
plt.title('State Estimation Error by EnKF')
plt.axis([0, T, 0, 2])
plt.grid(True)

plt.figure(figsize=(8, 5))
plt.plot(time, x[2, :], 'r-', label='True ε', linewidth=1.5)
plt.plot(time, xhat[2, :], 'b--', label='EnKF estimate', linewidth=1.5)
plt.xlabel('Time t (sec)')
plt.ylabel('ε')
plt.title('Parameter Estimation by EnKF')
plt.axis([0, T, -1, 2])
plt.grid(True)
plt.legend()

plt.figure(figsize=(8,5))
# 各メンバーのε推定値をプロット
for i in range(M):
    plt.plot(time, eps_ensemble[i,:], linewidth=0.5, alpha=0.5)
# アンサンブル平均を強調
plt.plot(time, xhat[2,:], 'r-', linewidth=2, label='EnKF mean ε')
plt.xlabel('Time t (sec)')
plt.ylabel('ε estimate')
plt.title('Ensemble trajectories of ε')
plt.legend()
plt.grid(True)

plt.show()
# -----------------------------