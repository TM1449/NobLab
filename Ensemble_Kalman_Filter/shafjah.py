import numpy as np
import matplotlib.pyplot as plt

#-----------------------------#
# 1. ニューロン数の設定      #
#-----------------------------#
Ne = 800   # 興奮性ニューロン数
Ni = 200   # 抑制性ニューロン数

#-----------------------------#
# 2. ニューロン個体差パラメータ生成
#    (論文中の ri, re に対応)
#-----------------------------#
# 興奮性 (re) と 抑制性 (ri) の乱数をそれぞれ生成
re = np.random.rand(Ne)   # 0～1 の一様乱数ベクトル (長さ 800)
ri = np.random.rand(Ni)   # 0～1 の一様乱数ベクトル (長さ 200)

# a, b, c, d を論文の式通りに設定
a = np.concatenate([0.02 * np.ones(Ne), 0.02 + 0.08 * ri])      # (Ne+Ni)-次元
b = np.concatenate([0.2  * np.ones(Ne), 0.25 - 0.05 * ri])     # (Ne+Ni)-次元
c = np.concatenate([-65 + 15 * re**2, -65 * np.ones(Ni)])      # (Ne+Ni)-次元
d = np.concatenate([  8 -  6 * re**2,   2 * np.ones(Ni)])      # (Ne+Ni)-次元

#-----------------------------#
# 3. シナプス重み行列 S の生成
#   S = [ 0.5*rand(Ne+Ni, Ne) ,  -rand(Ne+Ni, Ni) ]
#   左半分：興奮性からの重み (0.5×一様乱数)
#   右半分：抑制性からの重み (−1×一様乱数)
#-----------------------------#
W_exc = 0.5 * np.random.rand(Ne+Ni, Ne)    # 興奮性シナプス行列
W_inh =      np.random.rand(Ne+Ni, Ni)    # 抑制性シナプス行列
S = np.concatenate([W_exc, -W_inh], axis=1)  # (Ne+Ni)×(Ne+Ni) の結合行列

#-----------------------------#
# 4. 状態変数 v, u の初期化
#   v: 膜電位（全ニューロン共通初期値 -65）
#   u: 回復変数 u = b * v
#-----------------------------#
v = -65.0 * np.ones(Ne + Ni)   # 膜電位ベクトル (mV)
u = b * v                      # 回復変数ベクトル

#-----------------------------#
# 5. スパイクタイミング記録用リストの初期化
#   (行列ではなく、Python のリストに逐次 append していく)
#   後で（timestep, neuron_index）の組みをまとめてプロットする
#-----------------------------#
firings = []  # [(t1, i1), (t1, i2), (t2, i3), ...] の形で保存していく

#-----------------------------#
# 6. シミュレーションループ
#   MATLAB の for t = 1:1000 に対応（1 ms ごとに 0.5 ms × 2 ステップ）
#   各ループ内で、
#     1) ノイズ入力 I を生成
#     2) v >= 30 のニューロンを “発火” とみなしてリセット
#     3) 発火ニューロンからのシナプス入力を I に加える
#     4) 差分方程式を 0.5 ms ずつ 2 回 (合計 1 ms) 更新
#   という流れ。t はミリ秒刻み (1～1000)。
#-----------------------------#
for t in range(1, 1001):  # 1 から 1000 までの整数
    # ===== 1) 外部入力 I の生成 =====
    # 興奮性各ニューロンには N(0, 5^2) のノイズ、抑制性には N(0, 2^2) のノイズ
    I_exc = 5.0 * np.random.randn(Ne)
    I_inh = 2.0 * np.random.randn(Ni)
    I = np.concatenate([I_exc, I_inh])  # 長さ Ne+Ni のベクトル

    # ===== 2) 発火ニューロンの検出とリセット =====
    fired = np.where(v >= 30)[0]  # v >= 30 を超えたインデックスの配列
    if fired.size > 0:
        # 発火した各ニューロンを (t, index) の組みで記録
        for idx in fired:
            firings.append((t, idx))
        # リセット条件 (v <- c, u <- u + d)
        v[fired] = c[fired]
        u[fired] = u[fired] + d[fired]

    # ===== 3) 発火ニューロンからのシナプス入力を I に加算 =====
    # S は (Ne+Ni)×(Ne+Ni) 行列、fired はインデックス配列なので、
    # それらの列を合算して I に足し込む。
    if fired.size > 0:
        I = I + np.sum(S[:, fired], axis=1)

    # ===== 4) 差分方程式の更新 (0.5 ms × 2 回) =====
    #   dv/dt = 0.04 v^2 + 5v + 140 - u + I
    #   du/dt = a ( b v - u )
    # MATLAB コードでは “v = v + 0.5*(…); v = v + 0.5*(…);” で 1 ms を二分割している
    dv = (0.04 * v * v + 5 * v + 140 - u + I)
    v = v + 0.5 * dv
    dv = (0.04 * v * v + 5 * v + 140 - u + I)
    v = v + 0.5 * dv

    # u の更新
    u = u + a * (b * v - u)

#-----------------------------#
# 7. 結果の可視化：ラスタープロット
#   firings に (t, idx) のペアが入っているので、x 軸に t, y 軸に idx を散布
#-----------------------------#
if len(firings) > 0:
    # firings を NumPy 配列に変換すると扱いやすい
    firings_arr = np.array(firings)  # shape = (発火回数, 2)
    times = firings_arr[:, 0]        # 発火ミリ秒時刻
    neuron_ids = firings_arr[:, 1]   # 発火ニューロンインデックス (0～999)

    plt.figure(figsize=(8, 4))
    plt.scatter(times, neuron_ids, s=1, color='black')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.title('Raster plot of 1000 Izhikevich neurons')
    plt.ylim(-1, Ne+Ni)  # y の範囲を 0～999 に合わせる
    plt.tight_layout()
    plt.show()
else:
    print("No spikes detected.")
