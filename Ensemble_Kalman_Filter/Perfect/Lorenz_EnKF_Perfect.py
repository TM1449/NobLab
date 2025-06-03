import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. モデル＆フィルタ設定
# -----------------------------
# Lorenz 系の真パラメータ（ρ を推定対象とする）
sigma_true = 10.0        # σ（既知）
rho_true   = 28.0        # ρ（未知とみなす）
beta_true  = 8.0 / 3.0   # β（既知）

# 真の初期状態 [x, y, z]
x00 = np.array([1.0, 1.0, 1.0])

# 時間設定
dt = 0.005
d2 = np.sqrt(dt)
T  = 50.0
N  = int(T / dt)

# ノイズ分散
R    = 0.01                         # 観測ノイズ分散
Qx   = 0.01                         # x のシステムノイズ分散
Qy   = 0.01                         # y のシステムノイズ分散
Qz   = 0.01                         # z のシステムノイズ分散
Q    = np.diag([Qx, Qy, Qz])        # システムノイズ共分散行列

Qrho = 1e-4                         # ρ のランダムウォーク分散

# 観測行列（x のみ観測）
C = np.array([0, 1, 0.0, 0.0])

# -----------------------------
# 2. 真のデータ生成
# -----------------------------
# 真値格納用配列 [x, y, z, rho]
x_truth = np.zeros((4, N+1))
y       = np.zeros(N+1)

# 初期状態セット
x_truth[0:3, 0] = x00
x_truth[3,   0] = rho_true

# ノイズ生成
vd = np.sqrt(R) * np.random.randn(N+1)           # 観測ノイズ
wd = np.vstack((
    np.sqrt(Q[0,0]) * np.random.randn(N+1),
    np.sqrt(Q[1,1]) * np.random.randn(N+1),
    np.sqrt(Q[2,2]) * np.random.randn(N+1)
))  # システムノイズ (3, N+1)

# 真の Lorenz 系シミュレーション + 観測取得
for k in range(N):
    x, y_, z, rho = x_truth[:, k]
    # Lorenz 方程式（Euler）
    dx = sigma_true * (y_ - x)
    dy = x * (rho - z) - y_
    dz = x * y_ - beta_true * z

    x_truth[0:3, k+1] = x_truth[0:3, k] + dt * np.array([dx, dy, dz]) + d2 * wd[:, k]
    x_truth[3,   k+1] = rho  # ρ は定数

    # 観測（x のみ）
    y[k] = C @ x_truth[:, k] + vd[k]

y[N] = C @ x_truth[:, N] + vd[N]

# -----------------------------
# 3. EnKF 初期化
# -----------------------------
n = 4    # 拡張状態次元 [x, y, z, rho]
p = 1    # 観測次元
M = 100  # アンサンブル数

# 初期アンサンブル（真値周りにばらつき）
xep = np.zeros((n, M))
for i in range(M):
    xep[0:3, i] = x00 + 0.5 * np.random.randn(3)
    xep[3,   i] = rho_true + 5.0 * np.random.randn()

# 結果格納配列
xhat         = np.zeros((n, N+1))   # 推定平均値
rho_ensemble = np.zeros((M, N+1))   # 各メンバーのρ
# 拡張システムノイズ共分散
Qe = np.zeros((n, n))
Qe[0:3, 0:3] = Q
Qe[3,   3]   = Qrho
Re = R

# 一時配列
xef   = np.zeros((n, M))
yep   = np.zeros((p, M))
nu    = np.zeros((p, M))
Ex    = np.zeros((n, M))
Ey    = np.zeros((p, M))

# -----------------------------
# 4. 推定ループ
# -----------------------------
for k in range(N+1):
    # --- 分析ステップ ---
    ve = np.sqrt(Re) * np.random.randn(p, M)
    for i in range(M):
        yep[:, i] = C @ xep[:, i] + ve[:, i]

    # アンサンブル平均・偏差
    x_mean = np.mean(xep, axis=1, keepdims=True)
    y_mean = np.mean(yep, axis=1, keepdims=True)
    Ex     = xep - x_mean
    Ey     = yep - y_mean

    # 共分散計算
    Pxy = (Ex @ Ey.T) / (M - 1)
    Pyy = (Ey @ Ey.T) / (M - 1)

    # カルマンゲイン
    K = Pxy @ np.linalg.inv(Pyy)

    # 更新
    for i in range(M):
        nu[:, i]  = y[k] - yep[:, i]
        xef[:, i] = xep[:, i] + K @ nu[:, i]

    # 平均を推定値として保存
    xhat[:, k]       = np.mean(xef, axis=1)
    rho_ensemble[:, k] = xef[3, :]

    # --- 予測ステップ ---
    we = np.random.multivariate_normal(np.zeros(n), Qe, size=M).T
    for i in range(M):
        xi, yi, zi, rhoi = xef[:, i]
        dx = sigma_true * (yi - xi)
        dy = xi * (rhoi - zi) - yi
        dz = xi * yi - beta_true * zi

        xep[0:3, i] = xef[0:3, i] + dt * np.array([dx, dy, dz]) + d2 * we[0:3, i]
        xep[3,   i] = xef[3,   i] + d2 * we[3, i]

# -----------------------------
# 5. 推定誤差の計算
# -----------------------------
E_t = np.sqrt(
    (x_truth[0, :] - xhat[0, :])**2 +
    (x_truth[1, :] - xhat[1, :])**2 +
    (x_truth[2, :] - xhat[2, :])**2
)

time = np.linspace(0, T, N+1)

# -----------------------------
# 6. 結果プロット
# -----------------------------
# 各状態の推定
for idx, label in enumerate(['x', 'y', 'z']):
    plt.figure(figsize=(8, 5))
    plt.plot(time, x_truth[idx, :], 'r-',  label=f'True {label}',    linewidth=1.5)
    plt.plot(time, xhat[idx, :],      'b--', label=f'EnKF estimate', linewidth=1.5)
    plt.xlabel('Time t (sec)')
    plt.ylabel(label)
    plt.title(f'Estimation of {label} by EnKF')
    plt.grid(True)
    plt.legend()

# 状態推定誤差
plt.figure(figsize=(8, 4))
plt.plot(time, E_t, 'b-', linewidth=1.5)
plt.xlabel('Time t (sec)')
plt.ylabel('Error E_t')
plt.title('State Estimation Error by EnKF')
plt.grid(True)

# ρ の推定
plt.figure(figsize=(8, 5))
plt.plot(time, rho_true * np.ones_like(time), 'r-',  label='True ρ')
plt.plot(time, xhat[3, :],                    'b--', label='EnKF ρ̂')
plt.xlabel('Time t (sec)')
plt.ylabel('ρ')
plt.title('Parameter Estimation of ρ')
plt.grid(True)
plt.legend()

# Ensemble トラジェクトリ
plt.figure(figsize=(8, 5))
for i in range(M):
    plt.plot(time, rho_ensemble[i, :], linewidth=0.5, alpha=0.5)
plt.plot(time, xhat[3, :], 'r-', linewidth=2, label='EnKF mean ρ')
plt.hlines(rho_true, 0, T, linestyles='dashed', label='True ρ')
plt.xlabel('Time t (sec)')
plt.ylabel('ρ estimate')
plt.title('Ensemble trajectories of ρ')
plt.grid(True)
plt.legend()

plt.show()
