import numpy as np
import matplotlib.pyplot as plt

#-----------------------------#
# 1. シミュレーションパラメータ #
#-----------------------------#
T = 1000     # シミュレーション総時間 (ms)
dt = 0.5     # タイムステップ (ms)
time = np.arange(0, T, dt)

#-----------------------------#
# 2. Izhikevich 単一ニューロンモデルパラメータ
#    ここではトニックスパイクを例に設定
#-----------------------------#
a = 0.02
b = 0.2
c = -65
d = 8

#-----------------------------#
# 3. 入力電流 I(t) の設定
#    ここでは 200 ms〜800 ms の間、定常的に 10 mA を与える
#-----------------------------#
I = np.zeros_like(time)
I[(time >= 200) & (time < 800)] = 10

#-----------------------------#
# 4. 状態変数の初期化
#   v: 膜電位, u: 回復変数
#-----------------------------#
v = np.full_like(time, -65.0)  # 初期膜電位 -65 mV
u = b * v                      # 初期回復変数 u = b * v

# スパイクタイミング検出用配列
spikes = np.zeros_like(time, dtype=bool)

#-----------------------------#
# 5. ループによる数値積分 (Euler 法)
#-----------------------------#
for i in range(1, len(time)):
    dv = (0.04 * v[i-1]**2 + 5 * v[i-1] + 140 - u[i-1] + I[i-1]) * dt
    du = (a * (b * v[i-1] - u[i-1])) * dt
    
    v[i] = v[i-1] + dv
    u[i] = u[i-1] + du
    
    # 発火判定: v >= 30 mV
    if v[i] >= 30:
        v[i-1] = 30       # スパイクピークを前の時刻にプロット
        v[i] = c          # 膜電位をリセット
        u[i] += d         # 回復変数に d を付加
        spikes[i] = True  # 発火フラグを記録

#-----------------------------#
# 6. プロット: 膜電位の時間変化
#-----------------------------#
plt.figure(figsize=(10, 4))
plt.plot(time, v, label=' v(t)')
plt.scatter(time[spikes], np.full(np.sum(spikes), 30), color='red', s=10, label='Spike')
plt.title('Izhikevich ')
plt.xlabel(' (ms)')
plt.ylabel(' (mV)')
plt.ylim([-80, 40])
plt.legend()
plt.tight_layout()
plt.show()
