import numpy as np
import matplotlib.pyplot as plt
import ddeint

# Mackey-Glass 方程式のパラメータ
beta = 0.2
gamma = 0.1
n = 10
tau = 32

dt = 0.1
steps = 10000

def MackeyGlass(X, t, tau):
    return (beta * X(t - tau)) / (1 + pow(X(t - tau), n)) - gamma * X(t)

def DLEEE(t):
    return 0.1

# 時間範囲
times = np.linspace(0, steps * dt, steps)

# 遅延微分方程式を解く
solution = ddeint.ddeint(MackeyGlass, DLEEE, times, fargs=(tau,))

print(len(solution))
# 結果をプロット
plt.figure(figsize=(10, 5))
plt.plot(solution, label='Mackey-Glass Equation')
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.title('Mackey-Glass Delay Differential Equation')
plt.legend()
plt.grid()
plt.show()

