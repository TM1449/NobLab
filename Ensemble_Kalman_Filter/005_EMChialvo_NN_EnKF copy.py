import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================================================
# ネットワーク版 EM-Chialvo マップ＋EnKF パラメータ推定
#  100個のニューロンをリングスター結合し、パラメータ k, mu, sigma のいずれかを EnKF で推定する
# ===========================================================

# -----------------------------
# 1. モデルおよびネットワーク設定
# -----------------------------
N = 10             # ニューロン数
R = 2              # リング結合における左右 R 個の近隣ノードを結ぶ

# 拡張状態次元 n = 3N + 1（各ノードの x,y,φ と 推定パラメータ 1次元）
n = 3 * N + 1
p = N

# Chialvo マップの固定パラメータ（真値として使う）
a_true     = 0.89
b_true     = 0.6
c_true     = 0.28
k0_true    = 0.04

alpha_true = 0.1
beta_true  = 0.2
k1_true    = 0.1
k2_true    = 0.2

# 真の推定対象パラメータ（用途に応じて k_true, mu_true, sigma_true のうち一つを推定対象にする）
k_true     = 3.5      # 電磁結合強度
sigma_true = 0.0001       # リング結合強度
mu_true    = 0.001      # スター結合強度（中心ノードと周辺ノード間）

# -----------------------------
# 1.1 推定対象パラメータの指定
#    'k', 'mu', 'sigma' のうちいずれか一つを選択
# -----------------------------
param_to_estimate = 'sigma'   # ← ここを 'k' または 'mu' または 'sigma' に変更

# パラメータノイズ分散（EnKF におけるパラメータのランダムウォーク分散）
Q_param = {
    'k':     1e-8,
    'mu':    1e-8,
    'sigma': 1e-8
}[param_to_estimate]

# -----------------------------
# 1.2 シミュレーション設定
# -----------------------------
Burn_in_steps = 15000     # バーンイン期間（初期値の安定化）
T_steps = 5000          # 時間ステップ数

# シミュレーション全体の時間ステップ数
T_total = Burn_in_steps + T_steps  # 総時間ステップ数

# -----------------------------
# 1.3 ノイズ共分散設定
#    - プロセスノイズ共分散 (状態ノイズ)
#    - 観測ノイズ分散
# -----------------------------
Qx   = 1e-8             # x のプロセスノイズ分散
Qy   = 1e-8             # y のプロセスノイズ分散
Qphi = 1e-8             # φ のプロセスノイズ分散

R_obs = 1e-8            # 観測ノイズ分散（x を観測）

# -----------------------------
# 1.4 観測設定：全ノードの x を観測
#    観測次元 p = N、観測行列 C は N×(3N+1) の行列で
#    各行 i: 状態ベクトルの x_i 成分を 1 で抽出
# -----------------------------


# 観測行列 C を作成
# 拡張状態ベクトルの並びを [ x_1, x_2, …, x_N, y_1, …, y_N, φ_1, …, φ_N, θ ]
# としたとき、各ノード i の x を観測するので、
# C[i, i] = 1、それ以外の列は 0
# -----------------------------
# 1.5 リングスター結合行列を作る関数
#    mu: スター結合強度
#    sigma: リング結合強度
# -----------------------------
def build_ring_star_matrix(N, R, Mu, Sigma):
    """
    リングスター結合行列 (N×N) を返す。
    - リング結合: 各ノードは左右 R 個の近隣ノードと sigma 強度で結合
    - スター結合: 中心ノード (0 番) と他ノードが mu 強度で相互結合
    """
    # リング結合：まず隣接行列 A_ring（一次元ラップアラウンド）を作る
    Ring = np.identity(N)  # N×N の単位行列
    Ring[0, 0] = 0  # 中心ノードは除外

    for i in range(1, N):
        # 対角成分周辺に 1 を配置
        Ring[i: i+R+1, i: i+R+1] = 1

        # 左下と右上のラップアラウンド（範囲外：左下や右上）を考慮
        Ring[N-R+i-1:, i] = 1
        Ring[i, N-R+i-1:] = 1
        
    # 対角成分は -2R に設定
    for j in range(1, N):
        Ring[j, j] = -2 * R
    
    Ring_M = (Sigma / (2 * R)) * Ring

    # スター結合：ノード 0 が中心
    Star_M = np.zeros((N, N))
    # 中心ノード 0 は他ノード0と結合する：行 0 の off-diagonal を +mu、対角には −mu*(N-1)
    Star_M[0, 0] = -Mu * (N - 1)
    Star_M[0, 1:] = Mu
    Star_M[1:, 0] = -Mu

    for z in range(1, N):
        Star_M[z, z] += Mu

    # 合成
    return (Ring_M + Star_M)

# ===========================================================
# 2. 真のデータ生成：リングスター接続版改良 Chialvo マップを用いたシミュレーション
# ===========================================================
# 真のパラメータをまとめておく
true_params = {
    'k':     k_true,
    'mu':    mu_true,
    'sigma': sigma_true
}

# (1) 真の結合行列を生成
#A_true = build_ring_star_matrix(N, R, mu_true, sigma_true)
RingStar = build_ring_star_matrix(N, R, mu_true, sigma_true)

# (2) 真の状態配列を初期化：拡張状態ベクトルは [ x_i, y_i, φ_i, θ_true ] の形で保持するが、
#     状態配列では k は定数なので別に管理しておけばよい。ここでは便宜上、真の状態は
#     separate arrays x_true, y_true, phi_true にしておく。
x_true   = np.zeros((N, T_total + 1))
y_true   = np.zeros((N, T_total + 1))
phi_true = np.zeros((N, T_total + 1))

# 初期値ランダムシード設定
np.random.seed(None)
x_true[:, 0] = np.random.uniform(-1, 1, N)
y_true[:, 0] = np.random.uniform(-1, 1, N)
phi_true[:, 0] = np.random.uniform(-1, 1, N)

# (3) 真のパラメータ θ_true
theta_true = true_params[param_to_estimate]

# (4) ノイズ標準偏差
sd_x   = np.sqrt(Qx)
sd_y   = np.sqrt(Qy)
sd_phi = np.sqrt(Qphi)
sd_r   = np.sqrt(R_obs)

def M(xxx):
    """メムダクト関数 M(φ) の定義"""
    return alpha_true + 3.0 * beta_true * xxx**2

# (5) 真の軌道をシミュレーション
for t in range(0, T_total):
    x_true[:, t + 1] = pow(x_true[:, t], 2) * np.exp(y_true[:, t] - x_true[:, t]) + k0_true \
        + k_true * x_true[:, t] * M(phi_true[:, t]) + RingStar @ x_true[:, t]
    y_true[:, t + 1] = a_true * y_true[:, t] - b_true * x_true[:, t] + c_true
    phi_true[:, t + 1] = k1_true * x_true[:, t] - k2_true * phi_true[:, t]

# (6) 観測データ生成：全ノードの x を観測し、ノイズを加える
y_obs_data = np.zeros((N, T_steps))

for t in range(Burn_in_steps, T_total):
    index = t - Burn_in_steps  # 観測データのインデックス
    y_obs_data[:, index] = x_true[:, index] + sd_r * np.random.randn(N)

print(f"y_obs_data shape: {y_obs_data.shape}, {t}")

fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(x_true[:, T_total-1], '.', label='True x')
plt.xlabel('Nodes')
plt.ylabel('State value')
plt.show()

x_true_T = x_true.T  # ヒートマップ用に転置
sns.heatmap(x_true_T[Burn_in_steps:], cmap='hsv')
plt.title('True State Trajectory Heatmap')
plt.xlabel('Time step')
plt.ylabel('Node index')
plt.gca().invert_yaxis()  # y軸の順序を反転
plt.show()



# ===========================================================
# 3. EnKF 初期化：アンサンブル生成と初期設定ここから
# ===========================================================
M_ens = 100  # アンサンブルメンバー数

# (1) 各メンバーの拡張状態ベクトル xep を初期化
#     拡張状態次元は n = 3N + 1。並び順: [ x_1,...,x_N, y_1,...,y_N, φ_1,...,φ_N, θ ]

xep = np.zeros((n, M_ens))  # 拡張状態ベクトル (n×M_ens)
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
    xep[3*N,         m] = k0_m


# (2) 推定結果を格納する配列
x_ave = np.zeros((n, T_steps + 1))      # 各時刻のアンサンブル平均
#print(f"x_ave shape: {x_ave.shape}")
x_ave[:, 0] = np.mean(xep, axis=1)      # 初期推定平均

#print(xhat[:, 0], xhat.shape)

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

Re = R_obs * np.eye(p)   # 観測ノイズ分散行列 (p×p)

# (4) EnKF 内部で使うバッファ
yep = np.zeros((p, M_ens))    # 各メンバーの擾乱観測予測
nu  = np.zeros((p, M_ens))    # イノベーション(観測誤差)
xef = np.zeros((n, M_ens))    # 分析後の拡張状態

# ===========================================================
# 4. EnKF 推定ループ：各時刻で (1) 分析 → (2) 予測 を実行
# ===========================================================
for t in range(T_steps + 1):
    # -----------------------------
    # 4.1 分析ステップ (Analysis)：観測更新
    # -----------------------------
    # (1) 各メンバーの観測予測 y_pred を計算し、擾乱観測ノイズを加える
    #     - 観測モデル: y_pred^(i) = C @ xep[:, i]
    #     - 擾乱観測ノイズ v_i ~ N(0, Re)
    ve = np.sqrt(R_obs) * np.random.randn(p, M_ens)
    for m in range(M_ens):
        yep[:, m] = xep[0:N, m] + ve[:, m]

    # (2) アンサンブル平均と偏差を計算
    x_mean = np.mean(xep, axis=1, keepdims=True)  # (n×1)
    y_mean = np.mean(yep, axis=1, keepdims=True)  # (p×1)
    Ex     = xep - x_mean                          # (n×M)
    Ey     = yep - y_mean                          # (p×M)

    # (3) サンプル共分散を計算
    Pxy = (Ex @ Ey.T) / (M_ens - 1)    # (n×p)
    Pyy = (Ey @ Ey.T) / (M_ens - 1)    # (p×p)

    # (4) カルマンゲインを計算: K = Pxy @ inv(Pyy)
    K_gain = Pxy @ np.linalg.inv(Pyy)  # (n×p)

    # (5) 観測値 y_obs_data[:, t] を使って各メンバーを更新
    for m in range(M_ens):
        nu[:, m]  = y_obs_data[:, t] - yep[:, m]         # イノベーション (p×1)
        xef[:, m] = xep[:, m] + K_gain @ nu[:, m]        # 更新後の拡張状態 (n×1)

    # (6) 解析後アンサンブルの平均を推定結果として保存
    x_ave[:, t] = np.mean(xef, axis=1)

    # -----------------------------
    # 4.2 予測ステップ (Forecast)：時間更新
    # -----------------------------
    # 解析後の各メンバー xef[:, m] を用いて次ステップの予測を行う
    #   - 状態ノードの動的更新 + 結合入力 + プロセスノイズ
    #   - パラメータはランダムウォーク: θ_{t+1} = θ_t + w_param
    we = np.random.multivariate_normal(np.zeros(n), Qe, size=M_ens).T  # (n×M)

    for m in range(M_ens):
        # 各メンバーの現在の拡張状態
        xs   = xef[0: N,       m]      # x_1 ... x_N
        ys   = xef[N: 2*N,     m]      # y_1 ... y_N
        phis = xef[2*N: 3*N,   m]      # φ_1 ... φ_N
        theta_m = xef[3*N,     m]      # パラメータ θ

        # (a) そのメンバー固有の結合行列を構築（mu または sigma を更新率として持つ場合）
        if param_to_estimate == 'mu':
            A_m = build_ring_star_matrix(N, R, theta_m, sigma_true)
        elif param_to_estimate == 'sigma':
            A_m = build_ring_star_matrix(N, R, mu_true, theta_m)
        else:  # param_to_estimate == 'k'
            A_m = build_ring_star_matrix(N, R, mu_true, sigma_true)

        # (c) 各ノードの予測更新
        for i in range(N):
            xi   = xs[i]
            yi   = ys[i]
            phii = phis[i]
            # メムダクト関数
            M_phii = alpha_true + 3.0 * beta_true * phii**2

            # x の先行予測
            xep[i, m] = (pow(xi, 2) * np.exp(yi - xi) + k0_true + \
                         (theta_m * xi * M_phii if param_to_estimate == 'k' else k_true * xi * M_phii)+ coupling_input_m[i]+ we[i, m])
            # y の先行予測
            xep[N + i, m] = (
                a_true * yi
                - b_true * xi
                + c_true
                + we[N + i, m]
            )
            # φ の先行予測
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

# (1) 代表ノード（例: ノード 0）の x, y, φ の推定 vs 真値 をプロット
node_idx = 0

plt.figure(figsize=(8, 4))
plt.plot(time, x_true[node_idx, :], 'r-', label='True x[0]')
plt.plot(time, xhat[node_idx, :],   'b--', label='EnKF x̂[0]')
plt.xlabel('Time step')
plt.ylabel(f'x[{node_idx}]')
plt.title('State Estimation of x[0]')
plt.legend()
plt.grid(True)

plt.figure(figsize=(8, 4))
plt.plot(time, y_true[node_idx, :], 'r-', label='True y[0]')
plt.plot(time, xhat[N + node_idx, :], 'b--', label='EnKF ŷ[0]')
plt.xlabel('Time step')
plt.ylabel(f'y[{node_idx}]')
plt.title('State Estimation of y[0]')
plt.legend()
plt.grid(True)

plt.figure(figsize=(8, 4))
plt.plot(time, phi_true[node_idx, :], 'r-', label='True φ[0]')
plt.plot(time, xhat[2*N + node_idx, :], 'b--', label='EnKF φ̂[0]')
plt.xlabel('Time step')
plt.ylabel(f'φ[{node_idx}]')
plt.title('State Estimation of φ[0]')
plt.legend()
plt.grid(True)

# (2) 推定パラメータ θ の推移 vs 真の値
plt.figure(figsize=(8, 4))
plt.plot(time, theta_true * np.ones_like(time), 'r-', label=f'True {param_to_estimate}')
plt.plot(time, xhat[3*N, :], 'b--', label=f'EnKF estimate {param_to_estimate}')
plt.xlabel('Time step')
plt.ylabel(param_to_estimate)
plt.title(f'Parameter Estimation of {param_to_estimate}')
plt.legend()
plt.grid(True)

# (3) アンサンブルメンバーごとの θ 時系列プロット
plt.figure(figsize=(10, 6))
for m in range(M_ens):
    plt.plot(time, xep[3*N, :], linewidth=0.5, alpha=0.2, label='_nolegend_')
plt.plot(time, xhat[3*N, :], 'r-', linewidth=2, label=f'EnKF mean {param_to_estimate}')
plt.hlines(theta_true, 0, T_steps, colors='k', linestyles='dashed', label=f'True {param_to_estimate}')
plt.xlabel('Time step')
plt.ylabel(param_to_estimate)
plt.title(f'Ensemble Member Trajectories of {param_to_estimate}')
plt.legend()
plt.grid(True)

plt.show()
