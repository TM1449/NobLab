import numpy as np
import matplotlib.pyplot as plt

# ===========================================================
# 1. モデルとフィルタ設定：電磁束影響下のChialvoマップとEnKFパラメータ推定
#    かつ観測変数を任意の組み合わせで選択できるようにする
#    さらにパラメータ k の各アンサンブルメンバーの時間変移を記録
# ===========================================================

# -----------------------------
# 1.1 真のマップパラメータ（未知パラメータ k を含む）
# -----------------------------
# 電磁束を含む改良Chialvoマップ：
#   x_{n+1} = x_n^2 * exp(y_n - x_n) + k0_true + k_true * x_n * M(φ_n) + w_x
#   y_{n+1} = a_true * y_n - b_true * x_n + c_true + w_y
#   φ_{n+1} = k1_true * x_n - k2_true * φ_n + w_φ
#
# ここで、M(φ_n) = α_true + 3 β_true φ_n^2 はメムダクト関数。
# k_true は推定対象の電磁結合強度パラメータ（真値として定数）。
# w_x, w_y, w_phi はそれぞれ状態のプロセスノイズ。
#
a_true     = 0.89      # y の復帰係数 a（既知）
b_true     = 0.18      # x 依存項の係数 b（既知）
c_true     = 0.28      # 定数 c（既知）
k0_true    = 0.06      # x 更新式内の定数項 k0（既知）
alpha_true = 0.1       # メムダクト関数の α パラメータ（既知）
beta_true  = 0.2       # メムダクト関数の β パラメータ（既知）
k1_true    = 0.1       # φ 更新式内の係数 k1（既知）
k2_true    = 0.2       # φ 更新式内の係数 k2（既知）
k_true     = -0.2      # 電磁結合強度 k（真の値、推定対象）

# -----------------------------
# 1.2 シミュレーション設定
# -----------------------------
T_steps = 5000         # 全体の時間ステップ数

# -----------------------------
# 1.3 ノイズ分散設定（各状態＋パラメータ k のランダムウォーク）
# -----------------------------
#  - プロセスノイズ共分散：モデル誤差を表す
#    Qx: x のプロセスノイズ分散
#    Qy: y のプロセスノイズ分散
#    Qphi: φ のプロセスノイズ分散
#
#  - パラメータノイズ共分散 Qk: パラメータ k のランダムウォーク分散
#    （推定精度 vs. 変動許容度のバランス調整用）
#
#  - 観測ノイズ分散 R: 観測される各変数に対するセンサ誤差分散
#
R      = 1.0e-6        # 観測ノイズ分散（各観測変数共通で同じ分散を仮定）
Qx     = 1.0e-4        # x のプロセスノイズ分散
Qy     = 1.0e-4        # y のプロセスノイズ分散
Qphi   = 1.0e-4        # φ のプロセスノイズ分散
Qk     = 1.0e-10       # k のランダムウォークノイズ分散（EnKF 用のパラメータノイズ）

# -----------------------------
# 1.4 観測変数設定：任意に切り替える
# -----------------------------
# 観測可能な変数の組み合わせをユーザが選択できるようにする。
#   可能な変数：'x', 'y', 'phi'
#   例：
#     obs_vars = ['x']         # x のみを観測
#     obs_vars = ['y']         # y のみを観測
#     obs_vars = ['x', 'y']    # x と y を観測
#     obs_vars = ['x', 'y', 'phi']  # x, y, φ をすべて観測
#
obs_vars = ['x', 'y', 'phi']  # ← ここを変更して観測する変数を設定

# 各変数が状態ベクトルのどのインデックスに対応するかを定義
var_indices = {'x': 0, 'y': 1, 'phi': 2}
# 状態ベクトル全体は [x, y, φ, k] の順序なので、
#   'x'→0, 'y'→1, 'phi'→2, 'k'→3（パラメータ k は観測に含めない）

# 観測次元 p と観測行列 C を obs_vars に基づいて自動生成
obs_idx = [var_indices[var] for var in obs_vars]        # 例：['x','y']→[0,1]
p       = len(obs_idx)                                  # 観測次元数
n       = 4                                             # 拡張状態次元 [x, y, φ, k]

# 観測行列 C (p×n) を作成：選択した変数に対応する列を 1、それ以外を 0 とする
C = np.zeros((p, n))
for i, idx in enumerate(obs_idx):
    C[i, idx] = 1.0

# ===========================================================
# 2. 真のデータ生成：改良Chialvoマップの真の軌道と観測データをシミュレーション
# ===========================================================

# -----------------------------
# 2.1 真の状態ベクトルを保持する配列を初期化
# -----------------------------
# true_states: 4行 × (T_steps+1)列
#   各列が時刻 t における [x_t, y_t, φ_t, k_true]^T
true_states = np.zeros((n, T_steps + 1))
# y_obs: p 行 × (T_steps+1)列 の観測データ行列
y_obs       = np.zeros((p, T_steps + 1))

# -----------------------------
# 2.2 真の初期状態を設定
# -----------------------------
x0    = 0.3     # 真の x(0)
y0    = 0.2     # 真の y(0)
phi0  = 0.1     # 真の φ(0)
true_states[:, 0] = [x0, y0, phi0, k_true]  # k_true は定数として初期化

# -----------------------------
# 2.3 ノイズ標準偏差を計算
# -----------------------------
sd_x   = np.sqrt(Qx)    # x のプロセスノイズ標準偏差
sd_y   = np.sqrt(Qy)    # y のプロセスノイズ標準偏差
sd_phi = np.sqrt(Qphi)  # φ のプロセスノイズ標準偏差
sd_k   = np.sqrt(Qk)    # k のパラメータノイズ標準偏差（データ生成時は使用せず）
sd_r   = np.sqrt(R)     # 観測ノイズ標準偏差

# -----------------------------
# 2.4 真の状態遷移ループ
# -----------------------------
for t in range(T_steps):
    x_t, y_t, phi_t, k_t = true_states[:, t]

    # --- メムダクト関数 M(φ_t) を計算 ---
    M_phi = alpha_true + 3.0 * beta_true * phi_t**2

    # --- 真のモデル更新（プロセスノイズ w_x, w_y, w_phi を付与） ---
    # 1) x_{t+1} の計算
    #    w_x ~ N(0, Qx)
    x_next = (
        x_t**2 * np.exp(y_t - x_t)
        + k0_true
        + k_true * x_t * M_phi
        + sd_x * np.random.randn()
    )

    # 2) y_{t+1} の計算
    #    w_y ~ N(0, Qy)
    y_next = (
        a_true * y_t
        - b_true * x_t
        + c_true
        + sd_y * np.random.randn()
    )

    # 3) φ_{t+1} の計算
    #    w_phi ~ N(0, Qphi)
    phi_next = (
        k1_true * x_t
        - k2_true * phi_t
        + sd_phi * np.random.randn()
    )

    # 4) k_true はデータ生成時は定数として変動無し
    k_next = k_t

    # 更新した真の状態を格納
    true_states[:, t + 1] = [x_next, y_next, phi_next, k_next]

    # --- 観測生成：選択した変数にノイズを加えて取得 ---
    #    y_obs[:, t] = C @ true_states[:, t] + 観測ノイズ
    y_obs[:, t] = C @ true_states[:, t] + sd_r * np.random.randn(p)

# 最終時刻の観測も同様にノイズを加えて取得
y_obs[:, T_steps] = C @ true_states[:, T_steps] + sd_r * np.random.randn(p)


# ===========================================================
# 3. EnKF（アンサンブルカルマンフィルタ）初期化
# ===========================================================

# -----------------------------
# 3.1 アンサンブルサイズの設定
# -----------------------------
M = 100  # アンサンブルメンバー数

# -----------------------------
# 3.2 各メンバーの初期アンサンブルを構成
# -----------------------------
# 真の初期状態 [x0, y0, φ0, k_true] のまわりにランダムにばらつきを持たせる
#   - x, y の初期アンサンブル：真値に ±0.1 のノイズ
#   - φ の初期アンサンブル：真値に ±0.05 のノイズ
#   - k の初期アンサンブル：真値に ±0.01 のノイズ（不確実性を小さめに設定）
xep = np.zeros((n, M))  # 予測アンサンブル格納
for i in range(M):
    xep[0, i] = x0 + 0.1 * np.random.randn()       # x(0) のアンサンブル
    xep[1, i] = y0 + 0.1 * np.random.randn()       # y(0) のアンサンブル
    xep[2, i] = phi0 + 0.05 * np.random.randn()    # φ(0) のアンサンブル
    xep[3, i] = k_true + 0.01 * np.random.randn()  # k のアンサンブル（推定対象）

# -----------------------------
# 3.3 推定結果を格納する配列を準備
# -----------------------------
# xhat: 各時刻でのアンサンブル平均 [x, y, φ, k] を保存 (4×(T_steps+1))
xhat = np.zeros((n, T_steps + 1))

# 追加：パラメータ k の各メンバー時系列を保存する配列を用意
#   k_ensemble[i, t] はメンバー i の時刻 t における k の値を表す
k_ensemble = np.zeros((M, T_steps + 1))

# -----------------------------
# 3.4 拡張システムノイズ共分散行列 Qe と 観測ノイズ分散 Re を定義
# -----------------------------
#   Qe は 4×4 の共分散行列で、対角成分に [Qx, Qy, Qphi, Qk] を配置
Qe = np.diag([Qx, Qy, Qphi, Qk])
Re = R  # 観測ノイズ分散（スカラー）

# -----------------------------
# 3.5 EnKF 内で使用する一時配列を初期化
# -----------------------------
# xef: 解析後のアンサンブル一時格納 (4×M)
# yep: 各メンバーの観測予測 (p×M)
# nu:  イノベーション（観測誤差 y_obs[:,t] - y_pred）(p×M)
xef = np.zeros((n, M))
yep = np.zeros((p, M))
nu  = np.zeros((p, M))


# ===========================================================
# 4. EnKF 推定ループ：各時刻で (1) 分析 → (2) 予測 を実行
# ===========================================================
for t in range(T_steps + 1):
    # -----------------------------
    # 4.1 分析ステップ (Analysis)：観測更新
    # -----------------------------
    # (1) 各メンバーの観測予測 y_pred を計算し、擾乱観測ノイズを加える
    #     - 観測モデル: y_pred = C @ xep[:, i]
    #     - 擾乱観測ノイズ v_i ~ N(0, R)（観測次元 p に対して）
    ve = sd_r * np.random.randn(p, M)   # p×M の擾乱観測ノイズをメンバーごとに生成
    for i in range(M):
        # 各メンバー i に対し観測予測
        #   yep[:, i] = C @ x_prior + v_i
        yep[:, i] = C @ xep[:, i] + ve[:, i]

    # (2) アンサンブル平均と偏差を計算
    x_mean = np.mean(xep, axis=1, keepdims=True)  # 状態平均ベクトル (n×1)
    y_mean = np.mean(yep, axis=1, keepdims=True)  # 観測平均ベクトル (p×1)
    Ex     = xep - x_mean                         # 各メンバーの状態偏差 (n×M)
    Ey     = yep - y_mean                         # 各メンバーの観測偏差 (p×M)

    # (3) サンプル共分散を計算
    #   Pxy: 状態-観測 間サンプル共分散 (n×p)
    #   Pyy: 観測-観測 間サンプル共分散 (p×p)
    Pxy = (Ex @ Ey.T) / (M - 1)  # (n×M) × (M×p) → (n×p)
    Pyy = (Ey @ Ey.T) / (M - 1)  # (p×M) × (M×p) → (p×p)

    # (4) カルマンゲインを計算：K = Pxy @ inv(Pyy)
    K = Pxy @ np.linalg.inv(Pyy)  # (n×p) × (p×p)^{-1} → (n×p)

    # (5) 観測データ y_obs[:, t] を使って各メンバーを更新
    for i in range(M):
        # イノベーション（観測誤差）： ν_i = y_obs[:,t] - y_pred_i
        nu[:, i] = y_obs[:, t] - yep[:, i]
        # 更新: x_posterior = x_prior + K @ ν
        xef[:, i] = xep[:, i] + K @ nu[:, i]

    # (6) 解析後アンサンブルの平均を推定値として保存
    xhat[:, t] = np.mean(xef, axis=1)  # 時刻 t の推定平均 [x, y, φ, k]

    # 追加：パラメータ k の各メンバー時系列を記録
    #   xef[3, i] が解析後の各メンバー i における k の値
    k_ensemble[:, t] = xef[3, :]

    # -----------------------------
    # 4.2 予測ステップ (Forecast)：時間更新
    # -----------------------------
    # 解析後の各メンバー xef[:, i] を用いて次ステップの予測を行う
    # プロセスノイズ we_i ~ N(0, Qe) を各メンバーごとにサンプリング
    we = np.random.multivariate_normal(np.zeros(n), Qe, size=M).T  # (n×M) のノイズ行列

    for i in range(M):
        x_i, y_i, phi_i, k_i = xef[:, i]

        # --- メムダクト関数 M(φ_i) を計算 ---
        M_phi_i = alpha_true + 3.0 * beta_true * phi_i**2

        # (1) x の予測ステップ：Chialvo マップ＋電磁誘導項＋プロセスノイズ
        xep[0, i] = (
            x_i**2 * np.exp(y_i - x_i)
            + k0_true
            + k_i * x_i * M_phi_i
            + we[0, i]
        )

        # (2) y の予測ステップ：従来の Chialvo マップの y 更新＋プロセスノイズ
        xep[1, i] = (
            a_true * y_i
            - b_true * x_i
            + c_true
            + we[1, i]
        )

        # (3) φ の予測ステップ：電磁束の更新式＋プロセスノイズ
        xep[2, i] = (
            k1_true * x_i
            - k2_true * phi_i
            + we[2, i]
        )

        # (4) k の予測ステップ：ランダムウォーク
        xep[3, i] = k_i + we[3, i]


# ===========================================================
# 5. 結果プロット：状態・パラメータ推定軌道および k のアンサンブルメンバー変移図を可視化
# ===========================================================
time = np.arange(T_steps + 1)

# (1) x の推定結果 vs 真の軌道
plt.figure(figsize=(8, 4))
plt.plot(time, true_states[0, :], 'r-',  label='True x')
plt.plot(time, xhat[0, :],           'b--', label='EnKF x̂')
plt.xlabel('Time step')
plt.ylabel('x')
plt.title('State Estimation of x')
plt.legend()
plt.grid(True)

# (2) y の推定結果 vs 真の軌道
plt.figure(figsize=(8, 4))
plt.plot(time, true_states[1, :], 'r-',  label='True y')
plt.plot(time, xhat[1, :],           'b--', label='EnKF ŷ')
plt.xlabel('Time step')
plt.ylabel('y')
plt.title('State Estimation of y')
plt.legend()
plt.grid(True)

# (3) φ の推定結果 vs 真の軌道
plt.figure(figsize=(8, 4))
plt.plot(time, true_states[2, :], 'r-',  label='True φ')
plt.plot(time, xhat[2, :],           'b--', label='EnKF φ̂')
plt.xlabel('Time step')
plt.ylabel('φ')
plt.title('State Estimation of φ')
plt.legend()
plt.grid(True)

# (4) k の推定結果 vs 真の定数値
plt.figure(figsize=(8, 4))
plt.plot(time, k_true * np.ones_like(time), 'r-', label='True k')
plt.plot(time, xhat[3, :],                  'b--', label='EnKF k̂')
plt.xlabel('Time step')
plt.ylabel('k')
plt.title('Parameter Estimation of k')
plt.legend()
plt.grid(True)

# (5) 各アンサンブルメンバーの k 時系列プロット
plt.figure(figsize=(10, 6))
for i in range(M):
    # 各メンバー i の k 時系列を薄い線で描画
    plt.plot(time, k_ensemble[i, :], linewidth=0.5, alpha=0.3)
# アンサンブル平均 k̂(t) を赤太線で重ねる
plt.plot(time, xhat[3, :], 'r-', linewidth=2, label='EnKF mean k̂')
# 真の定数値 k_true を破線で描画
plt.hlines(k_true, 0, T_steps, colors='k', linestyles='dashed', label='True k')
plt.xlabel('Time step')
plt.ylabel('k value')
plt.title('Ensemble Member Trajectories of k')
plt.legend()
plt.grid(True)

plt.show()
