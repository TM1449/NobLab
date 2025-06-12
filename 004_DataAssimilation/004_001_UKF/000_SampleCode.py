"""
ex6_5ukf.py
Section 6.5 ― Wiener モデルの推定（Unscented Kalman Filter）
元 MATLAB コード: ex6_5ukf.m
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 1. 乱数のシード（再現性確保）
# ----------------------------
np.random.seed(0)

# ----------------------------
# 2. モデルとシミュレーション設定
# ----------------------------
r = 0.25           # 観測雑音の分散
q = 0.36           # システム雑音の分散
N = 1000           # ステップ数
a = 0.9            # 状態遷移係数
b = 1.0            # 入力係数
n = 4              # 状態次元 (x, β1, β2, β3)

beta1, beta2, beta3 = 1.0, 0.05, -0.01    # 真の係数

# ----------------------------
# 3. 真値データ生成
# ----------------------------
t = np.arange(N + 1)
x = np.zeros((n, N + 1))
y = np.zeros(N + 1)
u = 2.0 * np.random.randn(N + 1)           # 外部入力 u_t
w = np.sqrt(q) * np.random.randn(N + 1)    # 系雑音
v = np.sqrt(r) * np.random.randn(N + 1)    # 観測雑音

# 初期値
x[:, 0] = [0.0, beta1, beta2, beta3]

# データ生成ループ
for i in range(N):
    # 状態遷移
    x[0, i + 1] = a * x[0, i] + b * u[i] + w[i]
    x[1:, i + 1] = [beta1, beta2, beta3]   # パラメータは定数

    z = x[0, i]                             # 簡略表記
    tau = (beta1 + (beta2 + beta3 * z) * z) * z
    y[i] = tau + v[i]                       # 観測式

# 最終時刻の観測（原コードのまま）
z = x[0, N]
tau = (beta1 + (beta2 + beta3 * z) * z) * z
y[N] = x[0, N] + v[N]

# ----------------------------
# 4. UKF パラメータ
# ----------------------------
lam = 2.0
nn = 2 * n + 1
nl = n + lam

# 重み (全時刻で不変)
ws = np.full(nn, 0.5 / nl)
ws[-1] = lam / nl         # 最後のシグマ点

# ----------------------------
# 5. UKF 初期化
# ----------------------------
P0 = np.eye(n)
P0[0, 0] = 4.0            # x の初期分散だけ大きめ
Pep = P0.copy()           # 事前誤差共分散
xep = np.zeros(n)         # 事前推定値

QQ = np.zeros((n, n))
QQ[0, 0] = q              # システム雑音共分散 (β はノイズなし)

# 推定結果格納用
xeff = np.zeros((n, N + 1))
Peff = np.zeros((n, n, N + 1))
Kff  = np.zeros((n, N + 1))

# ----------------------------
# 6. UKF ループ
# ----------------------------
for i in range(N + 1):
    # 6-1. シグマ点生成（事前分布）
    S = np.linalg.cholesky(Pep)            # Pep は正定値
    xsig = np.zeros((n, nn))
    xsig[:, :n]      = (xep[:, None] + np.sqrt(nl) *  S)   # +√(nl) S
    xsig[:, n:2*n]   = (xep[:, None] - np.sqrt(nl) *  S)   # -√(nl) S
    xsig[:, -1]      = xep                                   # 中心点

    # 6-2. シグマ点を観測空間へ (非線形 g)
    ysig = ( xsig[1] * xsig[0] +
             xsig[2] * xsig[0]**2 +
             xsig[3] * xsig[0]**3 )

    # 6-3. 観測平均・分散
    y_mean = np.dot(ws, ysig)
    Pnu = np.dot(ws, (ysig - y_mean)**2) + r          # イノベーション分散

    # 6-4. 交差共分散
    Pxnu = np.dot((xsig - xep[:, None]) * ws, (ysig - y_mean))

    # 6-5. カルマンゲイン
    Kt = Pxnu / Pnu

    # 6-6. 事後推定
    innov = y[i] - y_mean
    xef = xep + Kt * innov
    Pef = Pep - np.outer(Kt, Kt) * Pnu

    # 結果保存
    xeff[:, i] = xef
    Peff[:, :, i] = Pef
    Kff[:, i] = Kt

    # 6-7. シグマ点生成（事後分布）
    S = np.linalg.cholesky(Pef)
    xsig[:, :n]      = (xef[:, None] + np.sqrt(nl) * S)
    xsig[:, n:2*n]   = (xef[:, None] - np.sqrt(nl) * S)
    xsig[:, -1]      = xef

    # 6-8. シグマ点を状態遷移 f で予測
    xsig_pred = np.empty_like(xsig)
    xsig_pred[0] = a * xsig[0] + b * u[i]
    xsig_pred[1:] = xsig[1:]

    # 6-9. 予測平均・分散
    xep = np.dot(xsig_pred, ws)
    Pep = (xsig_pred - xep[:, None]) @ np.diag(ws) @ (xsig_pred - xep[:, None]).T + QQ

# ----------------------------
# 7. RMSE 計算
# ----------------------------
rmse = np.sqrt((x[0] - xeff[0])**2)
rmse_ukf = rmse.sum() / N
print(f"RMSE (UKF) = {rmse_ukf:.4f}")

# ----------------------------
# 8. 可視化
# ----------------------------
plt.figure(1)
plt.plot(t, x[0], "r",  label="True $x_t$")
plt.plot(t, y,   "gx-", label="Observation $y_t$")
plt.plot(t, xeff[0], "bo-", label="Filtered estimate $\hat{x}_{t|t}$")
plt.xlim(0, N); plt.ylim(-15, 15)
plt.xlabel("Time step $t$"); plt.grid(True); plt.legend()
plt.title("True state, observation and filtered estimate")

plt.figure(2)
plt.plot(t, x[1],     "r-", label=r"True $\beta_1=1$")
plt.plot(t, xeff[1], "b-", label="UKF estimate")
plt.xlim(0, N); plt.ylim(-2, 2)
plt.title(r"Parameter estimate: $\beta_1$"); plt.xlabel("Time step"); plt.grid(True); plt.legend()

plt.figure(3)
plt.plot(t, x[2],     "r-", label=r"True $\beta_2=0.05$")
plt.plot(t, xeff[2], "b-", label="UKF estimate")
plt.xlim(0, N); plt.ylim(-2, 2)
plt.title(r"Parameter estimate: $\beta_2$"); plt.xlabel("Time step"); plt.grid(True); plt.legend()

plt.figure(4)
plt.plot(t, x[3],     "r-", label=r"True $\beta_3=-0.01$")
plt.plot(t, xeff[3], "b-", label="UKF estimate")
plt.xlim(0, N); plt.ylim(-2, 2)
plt.title(r"Parameter estimate: $\beta_3$"); plt.xlabel("Time step"); plt.grid(True); plt.legend()

plt.figure(5)
plt.plot(t, rmse, "b-", linewidth=1.5, label="UKF RMSE")
plt.xlim(0, N); plt.ylim(0, 10)
plt.xlabel("Time step $t$"); plt.ylabel("Error"); plt.grid(True); plt.legend()
plt.title("State‐estimation error (RMSE)")

plt.show()
