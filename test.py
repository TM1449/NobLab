import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# パラメータ
beta = 2.0
gamma = 1.0
n = 9.65
tau = 2.0
dt = 0.01       # 時間刻み幅（固定）
num_steps = 5000  # ステップ数（固定）
t_max = dt * num_steps  # 最大時間を計算

# 時間配列
times = np.arange(0, t_max, dt)  # dt = 0.001, ステップ数 20000

# 遅延データを管理するリスト
delay_buffer = [(0, 1.2)]  # (t, x) のリスト（初期値 x(0) = 1.2）

# 遅延データを補間する関数
def delayed_x(t):
    for i in range(len(delay_buffer) - 1, -1, -1):
        if delay_buffer[i][0] <= t:
            return delay_buffer[i][1]
    return 1.2  # t < 0 の場合

# マッキー・グラス方程式（常微分方程式として表現）
def mackey_glass(t, x):
    xtau = delayed_x(t - tau)
    dxdt = beta * xtau / (1 + xtau ** n) - gamma * x
    return dxdt

# 数値計算
results = []
x = 1.2  # 初期値

for t in times:
    sol = solve_ivp(mackey_glass, [t, t + dt], [x], t_eval=[t + dt])
    x = sol.y[0][0]  # 最新の値を取得
    results.append(x)
    delay_buffer.append((t + dt, x))  # 遅延データを更新

# 可視化
plt.figure(figsize=(10, 5))
plt.plot(times, results, label="Mackey-Glass Equation")
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.title(f"Mackey-Glass Equation (dt={dt}, Steps={num_steps})")
plt.legend()
plt.show()
