import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. モデル＆フィルタ設定
# -----------------------------
# 真のマップパラメータ
a_true  = 0.89    # y の復帰係数 a（既知）
b_true  = 0.6     # x 依存項の係数 b（推定対象）
c_true  = 0.28    # 定数項 c（既知）
k0_true = 0.04    # x 更新式内の定数項 k0（既知）

# シミュレーション設定
T_steps = 200     # 時間ステップ数

# ノイズ分散
R   = 0.05        # 観測ノイズ分散（x を観測）
Qx  = 0.001       # x のシステムノイズ分散
Qy  = 0.001       # y のシステムノイズ分散
Qb  = 1e-4        # b のランダムウォークノイズ分散

# 観測行列（x のみ観測）
C = np.array([1.0, 1.0, 0.0])   # y_obs = [1,0,0]·[x, y, b]ᵀ + noise

# -----------------------------
# 2. 真のデータ生成
# -----------------------------
x_true = np.zeros((3, T_steps+1))  # 真の状態 [x, y, b_true]
y_obs  = np.zeros(T_steps+1)       # 観測データ

# 初期真値セッティング
x0 = 0.02
y0 = 0.01
x_true[:,0] = [x0, y0, b_true]

# ノイズ標準偏差
sd_x = np.sqrt(Qx)
sd_y = np.sqrt(Qy)
sd_b = np.sqrt(Qb)
sd_r = np.sqrt(R)

for t in range(T_steps):
    x_t, y_t, b_t = x_true[:, t]
    # 真の Chialvo map 更新
    # x の更新：x_{t+1} = x_t^2 * exp(y_t - x_t) + k0_true + noise
    x_next = x_t**2 * np.exp(y_t - x_t) + k0_true + sd_x * np.random.randn()
    # y の更新：y_{t+1} = a_true*y_t - b_true*x_t + c_true + noise
    y_next = a_true * y_t - b_true * x_t + c_true + sd_y * np.random.randn()
    # b_true は定数
    b_next = b_t  

    x_true[:, t+1] = [x_next, y_next, b_next]
    # 観測（x のみ）にノイズを加える
    y_obs[t] = x_t + sd_r * np.random.randn()

# 最終ステップの観測
y_obs[T_steps] = x_true[0, T_steps] + sd_r * np.random.randn()


# -----------------------------
# 3. EnKF 初期化
# -----------------------------
n = 3      # 状態次元数 [x, y, b]
p = 1      # 観測次元数
M = 100    # アンサンブル数

# 初期アンサンブル：真の初期値周りにばらつきを持たせる
xep = np.zeros((n, M))
xep[0,:] = x0 + 0.1 * np.random.randn(M)
xep[1,:] = y0 + 0.1 * np.random.randn(M)
xep[2,:] = 0.0 + 0.01 * np.random.randn(M)   # b の事前平均を 0 に設定

# 推定結果格納用
xhat = np.zeros((n, T_steps+1))

# 共分散行列
Qe = np.diag([Qx, Qy, Qb])
Re = R

# 一時変数
xef = np.zeros((n, M))
yep = np.zeros((p, M))
nu  = np.zeros((p, M))


# -----------------------------
# 4. 推定ループ
# -----------------------------
for t in range(T_steps+1):
    # --- 分析ステップ（観測更新） ---
    # 各メンバーで観測予測
    ve = sd_r * np.random.randn(p, M)
    for i in range(M):
        yep[:, i] = C @ xep[:, i] + ve[:, i]
    # アンサンブル平均と偏差
    x_mean = np.mean(xep, axis=1, keepdims=True)
    y_mean = np.mean(yep, axis=1, keepdims=True)
    Ex = xep - x_mean
    Ey = yep - y_mean
    # 共分散算出
    Pxy = (Ex @ Ey.T) / (M - 1)
    Pyy = (Ey @ Ey.T) / (M - 1)
    # カルマンゲイン
    K = Pxy @ np.linalg.inv(Pyy)
    # 実観測で各メンバーを更新
    for i in range(M):
        nu[:,i]  = y_obs[t] - yep[:,i]
        xef[:,i] = xep[:,i] + K @ nu[:,i]
    # アンサンブル平均を推定値として保存
    xhat[:, t] = np.mean(xef, axis=1)

    # --- 予測ステップ（時間更新） ---
    for i in range(M):
        xi, yi, bi = xef[:, i]
        # プロセスノイズ
        wx, wy, wb = np.random.randn(3) * np.sqrt([Qx, Qy, Qb])
        # x 更新
        xep[0, i] = xi**2 * np.exp(yi - xi) + k0_true + wx
        # y 更新（b はアンサンブル固有の bi を使用）
        xep[1, i] = a_true * yi - bi * xi + c_true + wy
        # b はランダムウォーク
        xep[2, i] = bi + wb


# -----------------------------
# 5. 結果プロット
# -----------------------------
time = np.arange(T_steps+1)

plt.figure(figsize=(8,4))
plt.plot(time, x_true[0,:], 'r-', label='True x')
plt.plot(time, xhat[0,:], 'b--', label='EnKF x̂')
plt.xlabel('Time step')
plt.ylabel('x')
plt.title('State estimation of x')
plt.legend()
plt.grid(True)

plt.figure(figsize=(8,4))
plt.plot(time, x_true[1,:], 'r-', label='True y')
plt.plot(time, xhat[1,:], 'b--', label='EnKF ŷ')
plt.xlabel('Time step')
plt.ylabel('y')
plt.title('State estimation of y')
plt.legend()
plt.grid(True)

plt.figure(figsize=(8,4))
plt.plot(time, x_true[2,:], 'r-', label='True b')
plt.plot(time, xhat[2,:], 'b--', label='EnKF b̂')
plt.xlabel('Time step')
plt.ylabel('b')
plt.title('Parameter estimation of b')
plt.legend()
plt.grid(True)

plt.show()
