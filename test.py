import numpy as np
import matplotlib.pyplot as plt
import ddeint

# Mackey-Glass 方程式のパラメータ
beta = 0.2
gamma = 0.1
n = 10
tau = 32

# 遅延微分方程式の定義（ddeint に合わせて history を第2引数にする）
def mackey_glass(history, t):
    x_tau = history(t - tau) if t - tau > 0 else 1.2  # 遅延の処理
    return (beta * x_tau) / (1 + x_tau**n) - gamma * history(t)

# 初期関数（遅延を考慮）
def history(t):
    return 1.2  # t <= 0 のときの値

# 時間範囲
times = np.linspace(0, 5000, 50000)

# 遅延微分方程式を解く
solution = ddeint.ddeint(mackey_glass, history, times)

# 結果をプロット
plt.figure(figsize=(10, 5))
plt.plot(solution, label='Mackey-Glass Equation')
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.title('Mackey-Glass Delay Differential Equation')
plt.legend()
plt.grid()
plt.show()
