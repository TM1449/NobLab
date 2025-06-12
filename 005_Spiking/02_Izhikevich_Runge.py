import numpy as np
import matplotlib.pyplot as plt

def izhikevich_derivatives(v, u, I, a, b):
    """
    Izhikevich モデルの微分方程式の右辺を返す関数
    dv/dt = 0.04 v^2 + 5 v + 140 - u + I
    du/dt = a (b v - u)
    """
    dv = 0.04 * pow(v, 2) + 5 * v + 140 - u + I
    du = a * (b * v - u)
    return dv, du

def simulate_izhikevich_rk4(a, b, c, d, I, t_max, dt):
    """
    4次ルンゲクッタ法で Izhikevich ニューロンモデルをシミュレーションする関数

    Parameters
    ----------
    a, b, c, d : float
        Izhikevich モデルの内在パラメータ
    I : float or array_like
        刺激電流（定数または時間変化する配列）
    t_max : float
        シミュレーション時間 (ms)
    dt : float
        時間刻み幅 (ms)

    Returns
    -------
    t : np.ndarray
        時間配列
    v : np.ndarray
        膜電位 (mV) の時系列
    u : np.ndarray
        回復変数の時系列
    """
    n_steps = int(t_max / dt)
    t = np.linspace(0, t_max, n_steps)
    v = np.zeros(n_steps)
    u = np.zeros(n_steps)

    # 初期値
    v[0] = -65.0
    u[0] = b * v[0]

    # I がスカラーなら配列に拡張
    if np.isscalar(I):
        I = I * np.ones(n_steps)

    for i in range(n_steps - 1):
        # 現在の状態
        vi, ui, Ii = v[i], u[i], I[i]

        # RK4 の各ステップ
        k1_v, k1_u = izhikevich_derivatives(vi, ui, Ii, a, b)
        k2_v, k2_u = izhikevich_derivatives(
            vi + 0.5 * dt * k1_v,
            ui + 0.5 * dt * k1_u,
            I[i + 1] if i + 1 < n_steps else Ii,
            a, b
        )
        k3_v, k3_u = izhikevich_derivatives(
            vi + 0.5 * dt * k2_v,
            ui + 0.5 * dt * k2_u,
            I[i + 1] if i + 1 < n_steps else Ii,
            a, b
        )
        k4_v, k4_u = izhikevich_derivatives(
            vi + dt * k3_v,
            ui + dt * k3_u,
            I[i + 1] if i + 1 < n_steps else Ii,
            a, b
        )

        # 状態更新
        v_new = vi + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        u_new = ui + (dt / 6.0) * (k1_u + 2*k2_u + 2*k3_u + k4_u)

        # 発火処理
        if v_new >= 30:
            v[i] = 30       # スパイクを描画
            v_new = c       # リセット
            u_new += d      # 回復変数を増加

        v[i+1] = v_new
        u[i+1] = u_new

    return t, v, u

if __name__ == "__main__":
    # モデルパラメータ（例：Regular Spiking）
    a = 0.1
    b = 0.2
    c = -65
    d = 2

    # シミュレーション設定
    t_max = 300.0  # ms
    dt = 0.001       # ms
    t = np.arange(0, t_max, dt)

    # 刺激電流：20～80 ms の間に 10 μA/cm^2
    I = np.zeros_like(t) + 10
    #I[(t >= 20) & (t <= 80)] = 10

    # シミュレーション実行（RK4）
    t, v, u = simulate_izhikevich_rk4(a, b, c, d, I, t_max, dt)

    # 結果プロット
    plt.figure(figsize=(8, 4))
    plt.plot(t, v, label="v (mV)", color='blue')
    plt.plot(t, I, label="I (μA/cm²)", color='orange', linestyle='--')
    plt.title("Izhikevich Neuron Model Simulation")
    plt.xlabel("ms")
    plt.ylabel("mV")
    plt.ylim(-80, 40)
    plt.legend()
    plt.tight_layout()
    plt.show()
