import numpy as np
import matplotlib.pyplot as plt

# ===========================================================
# ネットワーク版 EM-Chialvo マップ＋EnKF パラメータ k 推定 (N=10, M_ens=100)
# -----------------------------------------------------------
#  - ニューロン数 N = 10
#  - リングスター結合（R=2 の設定）
#  - 真のモデルパラメータ k_true を用いてデータ生成
#  - 観測は全ノードの x のみ（観測次元 p = N）
#  - アンサンブルサイズ M_ens = 100
#  - EnKF を用いてパラメータ k を推定
# ===========================================================

# -----------------------------
# 1. モデルおよびネットワーク設定
# -----------------------------
N = 10           # ニューロン数
R = 2            # リング結合の隣接幅

# Chialvo マップの固定パラメータ（真値として使用）
a_true     = 0.89
b_true     = 0.6
c_true     = 0.28
k0_true    = 0.04

alpha_true = 0.1
beta_true  = 0.2
k1_true    = 0.1
k2_true    = 0.2

# 真の推定対象パラメータ k_true
k_true     = 3.5    # 真の電磁結合強度

# パラメータノイズ分散（EnKF 内でのランダムウォーク分散）
Q_param = 1e-8

# -----------------------------
# 1.1 シミュレーション設定
# -----------------------------
Burn_in_steps = 15000   # バーンイン期間
T_steps       = 5000    # 推定に用いるステップ数
T_total       = Burn_in_steps + T_steps

# -----------------------------
# 1.2 ノイズ共分散設定
# -----------------------------
Qx   = 1e-8    # x のプロセスノイズ分散
Qy   = 1e-8    # y のプロセスノイズ分散
Qphi = 1e-8    # φ のプロセスノイズ分散
R_obs = 1e-8   # 観測ノイズ分散（x を観測）

sd_x   = np.sqrt(Qx)
sd_y   = np.sqrt(Qy)
sd_phi = np.sqrt(Qphi)
sd_r   = np.sqrt(R_obs)

# -----------------------------
# 1.3 観測次元 p, 拡張状態次元 n
# -----------------------------
p = N            # 観測は全ノードの x (N次元)
n = 3 * N + 1    # 拡張状態ベクトル [x_1..x_N, y_1..y_N, φ_1..φ_N, θ]

# -----------------------------
# 1.4 リングスター結合行列生成関数
# -----------------------------
def build_ring_star_matrix(N, R, Mu, Sigma):
    """
    リングスター結合行列 (N×N) を返す。
      - リング結合: 各ノードは左右 R 個の近隣ノードと Sigma 強度で結合
      - スター結合: 中心ノード (0 番) と他ノードが Mu 強度で相互結合
    """
    # ----- リング結合行列（一次元ラップアラウンド） -----
    Ring = np.zeros((N, N))
    for i in range(N):
        for d in range(-R, R+1):
            if d == 0 or i == 0:
                # 中心ノード i=0 はリング結合には含めない
                continue
            j = (i + d) % N
            Ring[i, j] = 1
    # 各行の対角成分
    for i in range(1, N):
        Ring[i, i] = -2 * R
    Ring_M = (Sigma / (2 * R)) * Ring

    # ----- スター結合行列 -----
    Star = np.zeros((N, N))
    # 中心ノード 0
    Star[0, 0] = -Mu * (N - 1)
    for j in range(1, N):
        Star[0, j] = Mu
        Star[j, 0] = -Mu
        Star[j, j] += Mu

    return Ring_M + Star

# ===========================================================
# 2. 真のデータ生成：リングスター結合版 Chialvo マップ
# ===========================================================
# (1) 真の結合行列（k_true は Chialvo 内部で使うのでネットワークには関係せず、mu と sigma を固定する）
mu_true    = 0.001
sigma_true = 0.0001
RingStar_true = build_ring_star_matrix(N, R, mu_true, sigma_true)

# (2) 真の軌道を格納する配列
x_true   = np.zeros((N, T_total + 1))
y_true   = np.zeros((N, T_total + 1))
phi_true = np.zeros((N, T_total + 1))

# 初期値を乱数で設定
np.random.seed(0)
x_true[:, 0]   = np.random.uniform(-1, 1, N)
y_true[:, 0]   = np.random.uniform(-1, 1, N)
phi_true[:, 0] = np.random.uniform(-1, 1, N)

# メムダクト関数
def M(phi):
    return alpha_true + 3.0 * beta_true * phi**2

# (3) 真の軌道をシミュレーション
for t in range(T_total):
    # リングスター結合項
    coupling = RingStar_true @ x_true[:, t]  # N 次元

    for i in range(N):
        xi   = x_true[i, t]
        yi   = y_true[i, t]
        phii = phi_true[i, t]

        # x の更新
        x_true[i, t+1] = (
            xi**2 * np.exp(yi - xi)
            + k0_true
            + k_true * xi * M(phii)
            + coupling[i]
        )
        # y の更新
        y_true[i, t+1] = a_true * yi - b_true * xi + c_true
        # φ の更新
        phi_true[i, t+1] = k1_true * xi - k2_true * phii

# (4) 観測データ生成：Burn‐in 後の全ノード x にノイズを加える
y_obs_data = np.zeros((N, T_steps))
for t in range(Burn_in_steps, Burn_in_steps + T_steps):
    idx = t - Burn_in_steps
    y_obs_data[:, idx] = x_true[:, t] + sd_r * np.random.randn(N)

print(f"[確認] y_obs_data の形状: {y_obs_data.shape}  # (N, T_steps) になっている")

# ===========================================================
# 3. EnKF 初期化：アンサンブル生成と初期設定
# ===========================================================
M_ens = 100  # アンサンブルメンバー数

# (1) 各メンバーの拡張状態ベクトル xep を初期化する配列
#    shape = (n, M_ens)，n = 3N + 1
xep = np.zeros((n, M_ens))

# 真値の初期状態 (Burn‐in の t=0 と同じ位置) を各メンバーにばらつきを持たせて初期化
for m in range(M_ens):
    # x, y, φ の初期値
    x0_m   = x_true[:, 0] + 0.01 * np.random.randn(N)
    y0_m   = y_true[:, 0] + 0.01 * np.random.randn(N)
    phi0_m = phi_true[:, 0] + 0.01 * np.random.randn(N)
    # パラメータ k の初期値
    k0_m   = k_true + 0.01 * np.random.randn()

    # 拡張状態ベクトルに詰め込む: [x, y, φ, θ]
    xep[0:   N,      m] = x0_m
    xep[N:   2*N,    m] = y0_m
    xep[2*N: 3*N,    m] = phi0_m
    xep[3*N,        m] = k0_m

# (2) 推定結果を保存する配列
xhat = np.zeros((n, T_steps + 1))  # 各時刻でのアンサンブル平均を格納
xhat[:, 0] = np.mean(xep, axis=1)   # 初期推定平均

# (3) 系のノイズ共分散 Qe と観測ノイズ分散 Re
Qe = np.zeros((n, n))
# x 成分ノイズ
for i in range(N):
    Qe[i, i] = Qx
# y 成分ノイズ
for i in range(N, 2*N):
    Qe[i, i] = Qy
# φ 成分ノイズ
for i in range(2*N, 3*N):
    Qe[i, i] = Qphi
# θ 成分ノイズ
Qe[3*N, 3*N] = Q_param

Re = R_obs * np.eye(p)  # 観測ノイズ分散行列 (p×p)

# (4) EnKF 内部で使うバッファ
yep = np.zeros((p, M_ens))    # 各メンバーの擾乱観測予測
nu  = np.zeros((p, M_ens))    # イノベーション(観測誤差)
xef = np.zeros((n, M_ens))    # 分析後の拡張状態

# ===========================================================
# 4. EnKF 推定ループ：各時刻で (1) 分析 → (2) 予測 を実行
# ===========================================================
for t in range(T_steps + 1):
    # --------------------------------
    # 4.1 分析ステップ (Analysis)
    # --------------------------------
    # (1) 各メンバーの観測予測 y_pred を計算し、擾乱観測ノイズを加える
    #     観測モデル： y_pred^(m) = xep[0:N, m]  （x のみ観測）
    ve = sd_r * np.random.randn(p, M_ens)  # 擾乱観測ノイズ
    for m in range(M_ens):
        yep[:, m] = xep[0:N, m] + ve[:, m]

    # (2) アンサンブル平均と偏差を計算
    x_mean = np.mean(xep, axis=1, keepdims=True)   # (n×1)
    y_mean = np.mean(yep, axis=1, keepdims=True)   # (p×1)
    Ex     = xep - x_mean                           # (n×M)
    Ey     = yep - y_mean                           # (p×M)

    # (3) サンプル共分散を計算
    Pxy = (Ex @ Ey.T) / (M_ens - 1)   # (n×p)
    Pyy = (Ey @ Ey.T) / (M_ens - 1)   # (p×p)

    # (4) カルマンゲインを計算: K = Pxy @ inv(Pyy)
    K_gain = Pxy @ np.linalg.inv(Pyy)  # (n×p)

    # (5) 観測値 y_obs_data[:, t] を用いて各メンバーを更新
    #     ただし t=0 のときは観測なしなのでスキップしてもよいが、ここでは t=0 のとき
    #     y_obs_data[:, 0] は最初の観測値として使う
    for m in range(M_ens):
        nu[:, m]  = y_obs_data[:, t] - yep[:, m]      # イノベーション (p×1)
        xef[:, m] = xep[:, m] + K_gain @ nu[:, m]     # 更新後の拡張状態 (n×1)

    # (6) 解析後アンサンブルの平均を推定結果として保存
    xhat[:, t] = np.mean(xef, axis=1)

    # --------------------------------
    # 4.2 予測ステップ (Forecast)
    # --------------------------------
    # 解析後の各メンバー xef[:, m] を使って次ステップの予測
    # プロセスノイズを追加
    we = np.random.multivariate_normal(np.zeros(n), Qe, size=M_ens).T  # (n×M_ens)

    for m in range(M_ens):
        # 各メンバーの現在の拡張状態
        xs   = xef[0:   N,    m]   # x_1 ... x_N
        ys   = xef[N:   2*N,  m]   # y_1 ... y_N
        phis = xef[2*N: 3*N,   m]   # φ_1 ... φ_N
        theta_m = xef[3*N,     m]   # パラメータ θ (ここでは k)

        # (a) そのメンバー固有の結合行列を構築（μ, σ は真値固定、k のみ推定対象）
        A_m = build_ring_star_matrix(N, R, mu_true, sigma_true)

        # (b) 結合入力
        coupling_m = A_m @ xs  # (N,)

        # (c) 各ノードの予測更新
        for i in range(N):
            xi   = xs[i]
            yi   = ys[i]
            phii = phis[i]
            M_phii = alpha_true + 3.0 * beta_true * phii**2

            # x の予測
            xep[i, m] = (
                xi**2 * np.exp(yi - xi)
                + k0_true
                + theta_m * xi * M_phii   # θ=m=推定中の k を利用
                + coupling_m[i]
                + we[i, m]
            )
            # y の予測
            xep[N + i, m] = (
                a_true * yi
                - b_true * xi
                + c_true
                + we[N + i, m]
            )
            # φ の予測
            xep[2*N + i, m] = (
                k1_true * xi
                - k2_true * phii
                + we[2*N + i, m]
            )

        # (d) θ のランダムウォーク予測
        xep[3*N, m] = theta_m + we[3*N, m]

# ===========================================================
# 5. 結果プロット：状態推定およびパラメータ推定軌跡の可視化
# ===========================================================
time = np.arange(T_steps + 1)

# (1) ノード 0 の x 推定 vs 真値
plt.figure(figsize=(8, 4))
plt.plot(time, x_true[0, Burn_in_steps:Burn_in_steps + T_steps + 1], 'r-', label='True x[0]')
plt.plot(time, xhat[0, :], 'b--', label='EnKF x̂[0]')
plt.xlabel('Time step (Burn‐in 後)')
plt.ylabel('x[0]')
plt.title('State Estimation of x[0]')
plt.legend()
plt.grid(True)

# (2) 推定パラメータ k の推移 vs 真値
plt.figure(figsize=(8, 4))
plt.plot(time, k_true * np.ones_like(time), 'r-', label='True k')
plt.plot(time, xhat[3*N, :], 'b--', label='EnKF estimate k')
plt.xlabel('Time step (Burn‐in 後)')
plt.ylabel('k')
plt.title('Parameter Estimation of k')
plt.legend()
plt.grid(True)

# (3) アンサンブルメンバーごとの k 時系列（Burn‐in 後）
plt.figure(figsize=(10, 6))
# ※分析後アンサンブル xef はループ終了後の最終値なので、
#    ここでは EnKF 実行中にメンバーごとの履歴を都度保存していないため、
#    代わりに xhat の平均値のみプロットしています。
#    メンバーごとの軌跡を表示したい場合は、xef_trajectory[m, t] のような配列を別途用意してください。
plt.plot(time, xhat[3*N, :], 'b-', linewidth=2, label='EnKF mean k')
plt.hlines(k_true, 0, T_steps, colors='k', linestyles='dashed', label='True k')
plt.xlabel('Time step (Burn‐in 後)')
plt.ylabel('k')
plt.title('Ensemble Mean Trajectory of k')
plt.legend()
plt.grid(True)

plt.show()
