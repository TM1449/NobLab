import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. モデル＆フィルタ設定
# -----------------------------
# 真のマップパラメータ
# a_true: y の復帰係数 a（既知）
# b_true: x 依存項の係数 b（推定対象）
# c_true: 定数項 c（既知）
# k0_true: x 更新式内の定数項 k0（既知）
a_true  = 0.89
b_true  = 0.18
c_true  = 0.28
k0_true = 0.04

x00 = np.array([0.5, 0.5])  # 初期状態 [x, y]

# シミュレーション設定
N = 1000       # 時間ステップ数

# ノイズ分散（プロセスノイズ・観測ノイズ）
R  = 0.01           # 観測ノイズ分散（x のみ観測）

Qx = 0.01          # x のシステムノイズ分散
Qy = 0.01          # y のシステムノイズ分散
Q_Any = 1e-10           # b のランダムウォーク分散

# 観測行列 H（x のみ観測なので [1,0,0]）
H = np.array([[1.0, 0.0, 0.0]])

# -----------------------------
# 2. 真値生成と観測
# -----------------------------
# 真状態配列 x_true: [x, y, b] を格納
x_true = np.zeros((3, N+1))
y_obs  = np.zeros(N+1)

# 初期真値設定
x_true[0:2,0] = x00
x_true[2,0]   = b_true  # 初期パラメータ b

#ノイズ生成
vd = np.sqrt(R) * np.random.randn(N+1)  # 観測ノイズ
wd = np.vstack((
    np.sqrt(Qx) * np.random.randn(N+1),  # x のシステムノイズ
    np.sqrt(Qy) * np.random.randn(N+1),  # y のシステムノイズ
    np.sqrt(Q_Any) * np.random.randn(N+1)  # b のランダムウォークノイズ
))  # wd (3, N+1)

"""# ノイズ標準偏差
dx_sd = np.sqrt(Qx)
dy_sd = np.sqrt(Qy)
dw_sd = np.sqrt(Qb)
r_sd  = np.sqrt(R)"""

# 真状態シミュレーション
for k in range(N):
    xk, yk, bk = x_true[:,k]
    # Chialvo map の差分：時間更新ステップ (pdf eq. 時間更新)
    x_next = xk**2 * np.exp(yk - xk) + k0_true + wd[0,k]  # x 更新式
    y_next = a_true * yk - bk * xk + c_true + wd[1,k]  # y 更新式
    b_next = bk

    x_true[:,k+1] = [x_next, y_next, b_next]
    # 観測 (pdf eq. 計測モデル): y_obs = H x_true + noise
    y_obs[k] = H @ x_true[:,k] + vd[k]  # 観測値

# 最終ステップの観測
y_obs[N] = H @ x_true[:,N] + vd[N]

# -----------------------------
# 3. EnKF 初期化
# -----------------------------
n = 3       # 状態次元数 [x, y, b]
p = 1       # 観測次元数
M = 1000     # アンサンブル数

# アンサンブル状態 x_ensemble: shape(n, M)
x_ensemble = np.zeros((n,M))
for i in range(M):
    x_ensemble[0:2,i] = x_true[0:2,0] + 0.1 * np.random.randn(2)
    x_ensemble[2,i]   = 0.3 + 0.1 * np.random.randn()

# 結果保存用配列
x_est = np.zeros((n, N+1))
x_est[:,0] = np.mean(x_ensemble, axis=1)

# 一時配列
y_pred = np.zeros((p,M))
x_pred = np.zeros((n,M))

# -----------------------------
# 4. 推定ループ
# -----------------------------
for k in range(1, N+1):
    # --- 予測ステップ (forecast) ---
    for i in range(M):
        xk, yk, bk = x_ensemble[:,i]
        # 差分マップ適用
        x_pred[0,i] = xk**2 * np.exp(yk - xk) + k0_true + dx_sd * np.random.randn()
        x_pred[1,i] = a_true * yk - bk * xk + c_true + dy_sd * np.random.randn()
        x_pred[2,i] = bk + dw_sd * np.random.randn()
    
    # アンサンブル平均と偏差
    x_mean = np.mean(x_pred, axis=1, keepdims=True)
    X_dev  = x_pred - x_mean

    # --- 分析ステップ (analysis) ---
    # 観測予測 y_pred
    for i in range(M):
        y_pred[:,i] = H @ x_pred[:,i] + r_sd * np.random.randn()
    y_mean = np.mean(y_pred, axis=1, keepdims=True)
    Y_dev  = y_pred - y_mean

    # クロス共分散・観測共分散
    P_xy = (X_dev @ Y_dev.T) / (M-1)
    P_yy = (Y_dev @ Y_dev.T) / (M-1)
    # カルマンゲイン (pdf eq. カルマンゲイン)
    K    = P_xy @ np.linalg.inv(P_yy)

    # 更新
    for i in range(M):
        innov          = y_obs[k] - y_pred[:,i]
        x_ensemble[:,i] = x_pred[:,i] + K @ innov

    # 推定平均を保存
    x_est[:,k] = np.mean(x_ensemble, axis=1)

# -----------------------------
# 5. 結果プロット
# -----------------------------
plt.figure(figsize=(8,4))
plt.plot(time, x_true[0,:], 'r-', label='True x')
plt.plot(time, x_est[0,:], 'b--', label='EnKF x̂')
plt.xlabel('Step')
plt.ylabel('x')
plt.title('State estimation of x')
plt.legend()
plt.grid(True)

plt.figure(figsize=(8,4))
plt.plot(time, x_true[1,:], 'r-', label='True y')
plt.plot(time, x_est[1,:], 'b--', label='EnKF ŷ')
plt.xlabel('Step')
plt.ylabel('y')
plt.title('State estimation of y')
plt.legend()
plt.grid(True)

plt.figure(figsize=(8,4))
plt.plot(time, x_true[2,:], 'r-', label='True b')
plt.plot(time, x_est[2,:], 'b--', label='EnKF b̂')
plt.xlabel('Step')
plt.ylabel('b')
plt.title('Parameter estimation of b')
plt.legend()
plt.grid(True)

plt.show()
