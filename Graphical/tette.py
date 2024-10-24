import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定数の設定
k_0 = -0.44  # 例として適当な値を設定
k = 2.3
a = 0.5
b = 0.4
c = 0.89
k_1 = 0.1
k_2 = 0.2
alpha = 0.1
beta = 0.1

# φの範囲設定
phi_vals = np.linspace(-2, 2, 10000)

# パラメータの関数としてx, y, zを定義
# 例として、y = sin(φ), xは与えられた式に従って計算
y_vals = phi_vals
x_vals = np.power(y_vals, 2) * np.exp(((b - a + 1) * y_vals - c) / (a - 1)) + k_0 + 3 * k * beta * pow(k_1, 2) * y_vals ** 3 / (1 + k_2) ** 2 + y_vals * k * alpha

# 3次元プロット用のφに対応する値
z_vals = phi_vals

# 描画
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 軸のラベルを設定
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('φ')

# 曲線をプロット
ax.plot(x_vals, y_vals, z_vals, label='Parametric Curve', color='b')

# グリッドと凡例の表示
ax.grid(True)
ax.legend()

plt.show()
