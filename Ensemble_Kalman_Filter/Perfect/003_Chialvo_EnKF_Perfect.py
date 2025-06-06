import numpy as np
import matplotlib.pyplot as plt

# ===========================================================
# 1. モデルとフィルタ設定：ノイズやパラメータ、観測の関係を明示する
# ===========================================================

# -----------------------------
# 1.1 真のマップパラメータ（未知パラメータを含む）
# -----------------------------
# Chialvoマップの式（真のシステム）：
#   x_{t+1} = x_t^2 * exp(y_t - x_t) + k0_true + w_x
#   y_{t+1} = a_true * y_t - b_true * x_t + c_true + w_y
#
# ここで、
#   a_true, c_true, k0_true は既知の定数と仮定
#   b_true は推定対象の未知パラメータ（定数）として設定
#   w_x, w_y はそれぞれのプロセスノイズ
#
# 「パラメータノイズ（未知パラメータb_trueのランダムウォーク）」は
#   b_{t+1} = b_t + w_b  として付与するが、真のb_trueは定数とするので
#   w_b = 0 としてデータ生成する（ただしフィルタではノイズを少し載せる）。

a_true  = 0.89    # y の復帰係数 a（真の値、既知）
b_true  = 0.6     # x 依存項の係数 b（真の値、未知なので推定対象）
c_true  = 0.28    # 定数 c（真の値、既知）
k0_true = 0.04    # x 更新式内の定数項 k0（真の値、既知）

# -----------------------------
# 1.2 シミュレーション設定
# -----------------------------
T_steps = 2000    # 全体の時間ステップ数

# -----------------------------
# 1.3 ノイズ分散（共分散行列の対角要素として扱う）
# -----------------------------
#  - プロセスノイズ：モデル誤差や外乱として扱う
#    Qx: x に加わるノイズ分散
#    Qy: y に加わるノイズ分散
#    Qb: 推定対象パラメータ b のランダムウォークノイズ分散（EnKFで使用）
#
#  - 観測ノイズ：観測値 y_obs を取得する際のセンサ誤差として扱う
#    R: x を観測するときのノイズ分散（ガウスノイズと仮定）
#    推定したいパラメータが「時間によって変化しない」場合、ランダムウォークノイズは小さいほうが推定精度が高くなる。
#    反対に、「パラメータが時間的に変化する」場合は、ランダムウォークノイズを大きくしておく必要がある。

R   = 0.01       # 観測ノイズ分散（x を観測）
Qx  = 0.001     # x のプロセスノイズ分散
Qy  = 0.002      # y のプロセスノイズ分散
Qb  = 1e-8       # b のランダムウォークノイズ分散（EnKF 用のパラメータノイズ）

# -----------------------------
# 1.4 観測行列：観測モデル h(x,y,b) の線形部分
# -----------------------------
# ここでは「x のみを観測する」設定とするので、観測行列 C は次のように定義：
#   y_obs = C @ [x, y, b]^T + 観測ノイズ
#
# すなわち、C = [1, 0, 0] で x をそのまま観測。
C = np.array([1.0, 0.0, 0.0])  # 観測モデルの線形部分

# ===========================================================
# 2. 真のデータ生成：真の状態遷移と観測データをシミュレーション
# ===========================================================

# -----------------------------
# 2.1 真の状態ベクトルを保持する配列を初期化
# -----------------------------
# x_true[timestep] = [x_t, y_t, b_true]
# ただし b_true は定数なので、シミュレーション中は変化しない
x_true = np.zeros((3, T_steps + 1))  # インデックス 0 から T_steps まで
y_obs  = np.zeros(T_steps + 1)       # 観測データ（x のみ）

# -----------------------------
# 2.2 初期真値を設定
# -----------------------------
x0 = 0.02   # 真の x(0)
y0 = 0.01   # 真の y(0)
x_true[:, 0] = [x0, y0, b_true]

# -----------------------------
# 2.3 ノイズの標準偏差を計算
# -----------------------------
# ガウスノイズを np.random.randn() で発生させる際に
# 分散の平方根を掛けて調整する
sd_x = np.sqrt(Qx)  # x のプロセスノイズ標準偏差
sd_y = np.sqrt(Qy)  # y のプロセスノイズ標準偏差
sd_b = np.sqrt(Qb)  # b のランダムウォークノイズ標準偏差（ここでは使わない）
sd_r = np.sqrt(R)  # 観測ノイズ標準偏差

# -----------------------------
# 2.4 真の状態遷移ループ
# -----------------------------
for t in range(T_steps):
    x_t, y_t, b_t = x_true[:, t]
    # --- 真の Chialvo マップの更新（プロセスノイズ w_x, w_y を付与） ---
    # x の更新式：
    #   x_{t+1} = x_t^2 * exp(y_t - x_t) + k0_true + w_x
    #   w_x ~ N(0, Qx)
    x_next = x_t**2 * np.exp(y_t - x_t) + k0_true + sd_x * np.random.randn()

    # y の更新式：
    #   y_{t+1} = a_true * y_t - b_true * x_t + c_true + w_y
    #   w_y ~ N(0, Qy)
    y_next = a_true * y_t - b_true * x_t + c_true + sd_y * np.random.randn()

    # b_true は真値として定数を維持
    b_next = b_t

    # 更新した真の状態を格納
    x_true[:, t + 1] = [x_next, y_next, b_next]

    # --- 観測生成：x_t を観測し、観測ノイズを付与 ---
    # y_obs[t] = 真の x_t + v  （v ~ N(0, R)）
    y_obs[t] = x_t + sd_r * np.random.randn()

# 最終ステップの観測も同様にノイズを載せて取得
y_obs[T_steps] = x_true[0, T_steps] + sd_r * np.random.randn()


# ===========================================================
# 3. EnKF（アンサンブルカルマンフィルタ）初期化
# ===========================================================

# -----------------------------
# 3.1 状態次元・観測次元・アンサンブルサイズの設定
# -----------------------------
n = 3      # 拡張状態ベクトル次元数：[x, y, b] の 3 次元
p = 1      # 観測次元数（x のみ観測しているので 1 次元）
M = 100    # アンサンブル（メンバー）の数

# -----------------------------
# 3.2 各メンバーの初期アンサンブルを生成
# -----------------------------
# 真の初期値のまわりにランダムなばらつきを与えてアンサンブルを構成
#   xep[:, i] = [x0 + ノイズ, y0 + ノイズ, b_prior + ノイズ]
#
# b の事前平均は「まったく情報がない」想定で 0 に設定し、小さな分散を与える
xep = np.zeros((n, M))
xep[0, :] = x0 + 0.1 * np.random.randn(M)   # x の初期値アンサンブル
xep[1, :] = y0 + 0.1 * np.random.randn(M)   # y の初期値アンサンブル
xep[2, :] = 0.0 + 0.05 * np.random.randn(M) # b の初期値アンサンブル（事前平均0）

# -----------------------------
# 3.3 アンサンブル推定結果保管用配列
# -----------------------------
# 各時刻でのアンサンブル平均を保存して、最終的な推定軌跡とする
xhat = np.zeros((n, T_steps + 1))

# -----------------------------
# 3.4 ノイズ共分散行列（EnKFで使う形にまとめる）
# -----------------------------
# Qe: プロセスノイズ共分散行列 （状態空間すべて：[Qx, Qy, Qb] の対角行列）
# Re: 観測ノイズ分散（スカラー）
Qe = np.diag([Qx, Qy, Qb])
Re = R

# -----------------------------
# 3.5 一時変数：EnKF アルゴリズム内で使うバッファ
# -----------------------------
# xef: 予測ステップ後（解析前）のアンサンブルを一時保持
# yep: 各メンバーの観測予測 (y_pred) を保持
# nu: イノベーション（観測誤差 y_obs - y_pred）を保持
xef = np.zeros((n, M))
yep = np.zeros((p, M))
nu  = np.zeros((p, M))


# ===========================================================
# 4. EnKF 推定ループ：各時刻で予測→解析を繰り返す
# ===========================================================
for t in range(T_steps + 1):
    # -----------------------------
    # 4.1 分析ステップ（Analysis）：観測更新
    # -----------------------------
    #  (1) 各メンバーの観測予測 y_pred を計算し、擾乱観測 noise を加える
    #
    #  - C @ xep[:, i] がモデルの観測予測（x をそのまま返す）
    #  - ve[:, i] は観測ノイズ v ~ N(0, R) をメンバーごとにサンプリング（擾乱観測）
    #  - yep[:, i] = C @ xep[:, i] + ve[:, i] として、各メンバーの「観測予測にノイズを加えた値」を得る
    ve = sd_r * np.random.randn(p, M)
    for i in range(M):
        yep[:, i] = C @ xep[:, i] + ve[:, i]

    #  (2) アンサンブル平均と偏差を計算
    #    x_mean: 各メンバーの状態平均（3×1 ベクトル）
    #    y_mean: 各メンバーの観測平均（1×1 スカラー）
    x_mean = np.mean(xep, axis=1, keepdims=True)  # 形状 (3,1)
    y_mean = np.mean(yep, axis=1, keepdims=True)  # 形状 (1,1)

    #    Ex = xep - x_mean: 各メンバーの状態偏差
    #    Ey = yep - y_mean: 各メンバーの観測偏差
    Ex = xep - x_mean
    Ey = yep - y_mean

    #  (3) 共分散をサンプルから計算
    #    Pxy: 状態‐観測間の共分散行列 (3×1)
    #    Pyy: 観測‐観測間の共分散行列 (1×1)
    #    （+ R を既に擾乱観測で含めているので追加不要）
    Pxy = (Ex @ Ey.T) / (M - 1)  # 形状 (3,1)
    Pyy = (Ey @ Ey.T) / (M - 1)  # 形状 (1,1)

    #  (4) カルマンゲインを計算：K = Pxy * inv(Pyy)
    K = Pxy @ np.linalg.inv(Pyy)  # 形状 (3,1) * (1,1)^{-1} → (3,1)

    #  (5) 観測値 y_obs[t] を使って各メンバーを更新
    #    nu[:, i] = y_obs[t] - yep[:, i]: イノベーション（観測誤差）をメンバーごとに計算
    #    xef[:, i] = xep[:, i] + K @ nu[:, i]: 更新後（解析後）の状態を得る
    for i in range(M):
        nu[:, i]  = y_obs[t] - yep[:, i]       # 観測誤差
        xef[:, i] = xep[:, i] + K @ nu[:, i]    # 状態・パラメータの更新

    #  (6) 各メンバーを更新した後、そのアンサンブル平均を「推定値」として記録
    #    xhat[:, t] は時刻 t における [x, y, b] の推定平均
    xhat[:, t] = np.mean(xef, axis=1)

    # -----------------------------
    # 4.2 予測ステップ（Forecast）：時間更新
    # -----------------------------
    # 解析後の各メンバー xef[:, i] を使って、次ステップへ予測する
    for i in range(M):
        xi, yi, bi = xef[:, i]

        # --- プロセスノイズを各メンバーごとにサンプリング ---
        #   w_x ~ N(0, Qx), w_y ~ N(0, Qy), w_b ~ N(0, Qb)
        wx, wy, wb = np.random.randn(3) * np.sqrt([Qx, Qy, Qb])

        # --- 真のモデルと同じ式にノイズを載せて予測 ---
        # x の次の値を計算（各メンバー固有の bi を使うわけではなく、
        # 真の定数 k0_true を使って次の x を予測）
        xep[0, i] = xi**2 * np.exp(yi - xi) + k0_true + wx

        # y の次の値を計算：a_true は既知の真値を使うが、
        # b は各メンバーが持っている bi を用いて予測する（パラメータ推定に必要）
        xep[1, i] = a_true * yi - bi * xi + c_true + wy

        # b はランダムウォークで更新：bi_{t+1} = bi_t + w_b
        xep[2, i] = bi + wb


# ===========================================================
# 5. 結果プロット：真の軌道 vs 推定軌道 を可視化
# ===========================================================

time = np.arange(T_steps + 1)

# --- x の推定結果 ---
plt.figure(figsize=(8, 4))
plt.plot(time, x_true[0, :], 'r-',  label='True x')   # 真の x(t)
plt.plot(time, xhat[0, :], 'b--', label='EnKF x̂')   # EnKF 推定平均
plt.xlabel('Time step')
plt.ylabel('x')
plt.title('State estimation of x')
plt.legend()
plt.grid(True)

# --- y の推定結果 ---
plt.figure(figsize=(8, 4))
plt.plot(time, x_true[1, :], 'r-',  label='True y')   # 真の y(t)
plt.plot(time, xhat[1, :], 'b--', label='EnKF ŷ')   # EnKF 推定平均
plt.xlabel('Time step')
plt.ylabel('y')
plt.title('State estimation of y')
plt.legend()
plt.grid(True)

# --- b の推定結果（パラメータ推定） ---
plt.figure(figsize=(8, 4))
plt.plot(time, x_true[2, :], 'r-',  label='True b')   # 真の b_true（定数）
plt.plot(time, xhat[2, :], 'b--', label='EnKF b̂')   # EnKF 推定平均
plt.xlabel('Time step')
plt.ylabel('b')
plt.title('Parameter estimation of b')
plt.legend()
plt.grid(True)

plt.show()
