import numpy as np
import matplotlib.pyplot as plt

# ===========================================================
# 1. モデル＆フィルタ設定：Lorenz 系のパラメータとノイズの定義
# ===========================================================

# -----------------------------
# 1.1 真の Lorenz 系パラメータ（ρ を推定対象とする）
# -----------------------------
# Lorenz 系の連続モデル（微小刻みの Euler 法で離散化）:
#   dx/dt = σ (y - x)
#   dy/dt = x (ρ - z) - y
#   dz/dt = x y - β z
#
# ここでは、σ（シグマ）と β（ベータ）は既知とし、ρ（ローリング）を未知パラメータとして同時推定する。
sigma_true = 10.0        # σ（既知の真値）
rho_true   = 28.0        # ρ（未知とみなし、推定対象）
beta_true  = 8.0 / 3.0   # β（既知の真値）

# -----------------------------
# 1.2 真の初期状態ベクトル [x, y, z]
# -----------------------------
# 参照用として「真の軌道」をシミュレーションするための初期値
x00 = np.array([1.0, 1.0, 1.0])  # 真の初期状態

# -----------------------------
# 1.3 時間離散化＆シミュレーション設定
# -----------------------------
dt = 0.005             # タイムステップ Δt
sqrt_dt = np.sqrt(dt)  # プロセスノイズ付与時に使う √Δt
T  = 50.0              # シミュレーション全時間 (秒)
N  = int(T / dt)       # ステップ数（整数化）

# -----------------------------
# 1.4 ノイズ分散の定義
# -----------------------------
# - 観測ノイズ R: 観測される x の値に対して加わるガウスノイズの分散
# - システムノイズ Qx,Qy,Qz: Lorenz 系を数値シミュレーションする際のモデル誤差として加える 
# - ランダムウォーク分散 Qrho: 推定対象 ρ が時間とともに変動すると仮定したときのノイズ分散
#
# ρ を定数としたい場合、Qrho は非常に小さく設定することが多い
R    = 0.01                          # 観測ノイズ分散 (x の観測時に使用)
Qx   = 0.01                          # x 状態のプロセスノイズ分散
Qy   = 0.01                          # y 状態のプロセスノイズ分散
Qz   = 0.01                          # z 状態のプロセスノイズ分散
Q    = np.diag([Qx, Qy, Qz])         # 3次元状態ベクトル用のプロセス共分散行列
Qrho = 1e-4                          # ρ のランダムウォーク分散（パラメータノイズ）

# -----------------------------
# 1.5 観測モデルの定義：観測行列 C
# -----------------------------
# 観測として「x のみを観測する」設定とし、観測値 y_k = x + 観測ノイズ
# 拡張状態ベクトルは [x, y, z, ρ] の4次元。C はそのうち x のみ選択する行列。
#   y_obs = C @ [x, y, z, ρ]^T + v
C = np.array([1.0, 0.0, 0.0, 0.0])  # 観測行列 (1×4)

# ===========================================================
# 2. 真のデータ生成：Lorenz 系の真の軌道と観測データをシミュレーション
# ===========================================================

# -----------------------------
# 2.1 真の状態ベクトル x_truth と観測 y を格納する配列を初期化
# -----------------------------
# x_truth: 4行 × (N+1)列 の配列。各列が時刻 k における [x, y, z, ρ]^T を表す。
# y: N+1個の観測データ（x のみにノイズを加えたもの）
x_truth = np.zeros((4, N + 1))
y        = np.zeros(N + 1)

# -----------------------------
# 2.2 真の初期状態を設定
# -----------------------------
# x_truth[0:3,0] に真の初期状態 [x00] をセットし、
# x_truth[3,0] に真の ρ の値をセット
x_truth[0:3, 0] = x00         # [x, y, z]
x_truth[3,   0] = rho_true    # ρ は定数として初期化

# -----------------------------
# 2.3 真のプロセスノイズ wd と 観測ノイズ vd を生成
# -----------------------------
# - wd: 3行 × (N+1)列 の配列。各行がそれぞれ x, y, z 用のノイズ系列
#     wd = sqrt(Q) * 標準正規乱数 * √dt
# - vd: (N+1)要素のベクトル。観測ノイズは x の観測に加わる
wd = np.vstack((
    np.sqrt(Q[0, 0]) * np.random.randn(N + 1),
    np.sqrt(Q[1, 1]) * np.random.randn(N + 1),
    np.sqrt(Q[2, 2]) * np.random.randn(N + 1),
))
vd = np.sqrt(R) * np.random.randn(N + 1)  # 観測ノイズ (x only)

# -----------------------------
# 2.4 真の Lorenz 系を Euler 法で離散化シミュレーション
# -----------------------------
for k in range(N):
    # 2.4.1 i時刻の真の状態を取り出す
    x, y_, z, rho = x_truth[:, k]

    # 2.4.2 Lorenz 方程式の右辺を計算
    dx = sigma_true * (y_ - x)
    dy = x * (rho - z) - y_
    dz = x * y_ - beta_true * z

    # 2.4.3 次のステップの状態を Euler 法で予測し、プロセスノイズを加える
    #   x_true[k+1] = x_true[k] + dt * [dx, dy, dz] + √dt * wd[:, k]
    x_truth[0:3, k + 1] = x_truth[0:3, k] + dt * np.array([dx, dy, dz]) + sqrt_dt * wd[:, k]
    # ρ は「真の定数」としてランダムウォークを加えず一定値を維持
    x_truth[3, k + 1] = rho

    # 2.4.4 観測データを生成：x に観測ノイズを加えたものを y[k] に格納
    y[k] = C @ x_truth[:, k] + vd[k]

# 最終ステップ (k=N) の観測も同様にノイズを加えて取得
y[N] = C @ x_truth[:, N] + vd[N]

# ===========================================================
# 3. EnKF の初期化：アンサンブル生成と初期設定
# ===========================================================

# -----------------------------
# 3.1 拡張状態ベクトル次元と観測次元、アンサンブル数の定義
# -----------------------------
n = 4    # 拡張状態次元 [x, y, z, ρ]
p = 1    # 観測次元 (x のみ)
M = 100  # アンサンブルメンバー数

# -----------------------------
# 3.2 各メンバーの初期アンサンブルを構成
# -----------------------------
# 真の初期状態 [x00, ρ_true] のまわりにランダムにばらつきを持たせる
#   - x, y, z は真の初期値 x00 に ±0.5 のノイズを加えたもの
#   - ρ は真の初期値 ρ_true に ±5.0 のノイズを加えたもの（大きめに不確実性を設定）
xep = np.zeros((n, M))
for i in range(M):
    # 状態 [x, y, z] 用
    xep[0:3, i] = x00 + 0.5 * np.random.randn(3)
    # パラメータ ρ 用
    xep[3,  i] = rho_true + 5.0 * np.random.randn()

# -----------------------------
# 3.3 推定結果を格納する配列の準備
# -----------------------------
# xhat: 各時刻でのアンサンブル平均 [x, y, z, ρ] を保存
xhat         = np.zeros((n, N + 1))
# rho_ensemble: 各メンバーの ρ の推定値を時系列で保存（プロット用に利用）
rho_ensemble = np.zeros((M, N + 1))

# -----------------------------
# 3.4 拡張システムノイズ共分散行列 Qe と 観測ノイズ分散 Re の定義
# -----------------------------
#   Qe は 4×4 の共分散行列で、上3×3 が状態 [x,y,z] のノイズ共分散 Q、
#   4行4列目に ρ 用の Qrho を入れる。残りは 0。
Qe = np.zeros((n, n))
Qe[0:3, 0:3] = Q     # [x, y, z] 用プロセスノイズ共分散
Qe[3,    3] = Qrho   # ρ 用ランダムウォークノイズ分散
Re = R               # 観測ノイズ分散（スカラー）

# -----------------------------
# 3.5 EnKF で使う一時変数を初期化
# -----------------------------
# xef: 解析後のアンサンブル一時格納 (4×M)
# yep: 各メンバーの観測予測 (1×M)
# nu:  イノベーション（観測誤差 y_obs[k] - y_pred）(1×M)
xef = np.zeros((n, M))
yep = np.zeros((p, M))
nu  = np.zeros((p, M))

# ===========================================================
# 4. EnKF 推定ループ：各時刻で (1) 分析 → (2) 予測 を実行
# ===========================================================
for k in range(N + 1):
    # -----------------------------
    # 4.1 分析ステップ (Analysis)：観測更新
    # -----------------------------
    # (1) 各メンバーの観測予測を計算し、擾乱観測ノイズを加える
    #     - 観測モデル: y_pred = C @ xep[:, i]
    #     - 擾乱観測ノイズ v_i ~ N(0, R)
    ve = np.sqrt(Re) * np.random.randn(p, M)  # 擾乱観測ノイズをメンバーごとに生成
    for i in range(M):
        # 各メンバー i に対し観測予測: yep[i] = C @ x_prior + v_i
        yep[:, i] = C @ xep[:, i] + ve[:, i]

    # (2) アンサンブル平均と偏差を計算
    x_mean = np.mean(xep, axis=1, keepdims=True)  # 状態平均 (4×1)
    y_mean = np.mean(yep, axis=1, keepdims=True)  # 観測平均 (1×1)
    Ex     = xep - x_mean                         # 各メンバーの状態偏差 (4×M)
    Ey     = yep - y_mean                         # 各メンバーの観測偏差 (1×M)

    # (3) サンプル共分散を計算
    #  Pxy: 状態-観測間のサンプル共分散 (4×1)
    #  Pyy: 観測-観測間のサンプル共分散 (1×1)
    Pxy = (Ex @ Ey.T) / (M - 1)  # 4×1 行列
    Pyy = (Ey @ Ey.T) / (M - 1)  # 1×1 行列

    # (4) カルマンゲインを計算 K = Pxy * inv(Pyy)
    K = Pxy @ np.linalg.inv(Pyy)  # (4×1) = (4×1) × (1×1)^{-1}

    # (5) 観測データ y[k] を使って各メンバーを更新
    for i in range(M):
        # イノベーション（観測誤差） ν_i = y_obs[k] - y_pred_i
        nu[:, i] = y[k] - yep[:, i]
        # 更新: x_posterior = x_prior + K @ ν
        xef[:, i] = xep[:, i] + K @ nu[:, i]

    # (6) 解析後アンサンブルの平均を推定値として保存
    xhat[:, k]         = np.mean(xef, axis=1)   # 推定平均 [x, y, z, ρ]
    rho_ensemble[:, k] = xef[3, :]              # 各メンバーの ρ 値を保存

    # -----------------------------
    # 4.2 予測ステップ (Forecast)：時間更新
    # -----------------------------
    # 解析後の各メンバー xef[:, i] を用いて次ステップの予測を行う
    # プロセスノイズ we_i ~ N(0, Qe) を各メンバーごとにサンプリング
    we = np.random.multivariate_normal(np.zeros(n), Qe, size=M).T  # (4×M) のノイズ行列

    for i in range(M):
        xi, yi, zi, rhoi = xef[:, i]

        # Lorenz 系の RHS を計算
        dx = sigma_true * (yi - xi)
        dy = xi * (rhoi - zi) - yi
        dz = xi * yi - beta_true * zi

        # 状態 x, y, z の予測ステップ: Euler 法 + プロセスノイズ
        xep[0:3, i] = xef[0:3, i] + dt * np.array([dx, dy, dz]) + sqrt_dt * we[0:3, i]

        # パラメータ ρ のランダムウォーク予測: ρ_new = ρ_old + ノイズ
        # もし Qrho＝0 にすれば ρ は定数として変化しないことになる
        xep[3, i] = xef[3, i] + sqrt_dt * we[3, i]

# ===========================================================
# 5. 推定誤差計算：状態推定誤差 E_t を算出
# ===========================================================
# 真の (x, y, z) と推定 (xhat, yhat, zhat) のユークリッド誤差を各時刻で計算
state_error = np.sqrt(
    (x_truth[0, :] - xhat[0, :])**2 +
    (x_truth[1, :] - xhat[1, :])**2 +
    (x_truth[2, :] - xhat[2, :])**2
)
time = np.linspace(0, T, N + 1)  # 時間軸（秒）

# ===========================================================
# 6. 結果プロット：状態・パラメータ推定結果と誤差を可視化
# ===========================================================

# (1) 状態 x, y, z の推定軌道 vs 真の軌道
for idx, label in enumerate(['x', 'y', 'z']):
    plt.figure(figsize=(8, 5))
    plt.plot(time, x_truth[idx, :], 'r-',  label=f'True {label}',    linewidth=1.5)
    plt.plot(time, xhat[idx, :],      'b--', label=f'EnKF estimate', linewidth=1.5)
    plt.xlabel('Time t (sec)')
    plt.ylabel(label)
    plt.title(f'Estimation of {label} by EnKF')
    plt.grid(True)
    plt.legend()

# (2) 状態推定誤差 E_t のプロット
plt.figure(figsize=(8, 4))
plt.plot(time, state_error, 'b-', linewidth=1.5)
plt.xlabel('Time t (sec)')
plt.ylabel('Error E_t')
plt.title('State Estimation Error by EnKF')
plt.grid(True)

# (3) パラメータ ρ の推定軌道 vs 真の定数値
plt.figure(figsize=(8, 5))
plt.plot(time, rho_true * np.ones_like(time), 'r-',  label='True ρ')
plt.plot(time, xhat[3, :],               'b--', label='EnKF ρ̂')
plt.xlabel('Time t (sec)')
plt.ylabel('ρ')
plt.title('Parameter Estimation of ρ')
plt.grid(True)
plt.legend()

# (4) 各アンサンブルメンバーの ρ 軌跡と平均値
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
