import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def simulate_izhikevich(a, b, c, d, I, t_max, dt_sim):
    """
    Izhikevich ニューロンモデルを細かい刻み幅 dt_sim でシミュレーションする関数

    Parameters
    ----------
    a, b, c, d : float
        モデルパラメータ
    I : array_like
        刺激電流（長さ = t_max/dt_sim の配列）
    t_max : float
        シミュレーション総時間 (ms)
    dt_sim : float
        シミュレーション刻み幅 (ms)

    Returns
    -------
    t : np.ndarray
        時間配列（細かい刻み）
    v, u : np.ndarray
        膜電位と回復変数の時系列
    """
    n_steps = int(np.ceil(t_max / dt_sim))
    t = np.arange(n_steps) * dt_sim

    v = np.zeros(n_steps)
    u = np.zeros(n_steps)
    v[0] = -65.0
    u[0] = b * v[0]

    for i in tqdm(range(n_steps - 1)):
        dv = 0.04 * pow(v[i], 2) + 5 * v[i] + 140 - u[i] + I[i]
        du = a * (b * v[i] - u[i])

        # Euler 法で更新
        v[i+1] = v[i] + dv * dt_sim
        u[i+1] = u[i] + du * dt_sim

        if v[i+1] >= 30:
            v[i]   = 30      # spike を描画
            v[i+1] = c       # reset
            u[i+1] = u[i+1] + d

    return t, v, u

if __name__ == "__main__":
    # モデルパラメータ（Regular Spiking）
    a, b, c, d = 0.02, 0.2, -65, 8

    # シミュレーション設定
    t_max   = 310.0        # ms
    dt_sim  = 1e-5         # シミュレーション刻み幅 (ms)
    dt_plot = 0.5          # 描画用刻み幅 (ms)

    # シミュレーション時間とステップ数の計算
    n_steps = int(np.ceil(t_max / dt_sim))
    print(f"Total simulation steps: {n_steps} (Time_max = {t_max} ms, dt ={dt_sim} ms)\n")
    t_sim   = np.arange(n_steps) * dt_sim
    
    
    # 定常電流の例（コメントアウトして使用）
    Amplitude = 30.0  # 定常電流の振幅 (μA/cm^2)

    Start_Time = 10.0  # 定常電流の開始時間 (ms)
    End_Time = 310.0   # 定常電流の終了時間 (ms)
    I       = np.zeros(n_steps)  # 定常電流の例
    
    I[int(Start_Time / dt_sim):int(End_Time / dt_sim)] = Amplitude  # 定常電流を設定
    


    # サイン波の刺激電流を生成
    """
    A = 20.0               # 振幅 (μA/cm^2)
    f = 10.0              # 周波数 (Hz)
    n_steps = int(np.ceil(t_max / dt_sim))
    t_sim = np.arange(n_steps) * dt_sim
    # サイン波：時刻 t_sim（ms）を秒に直して振動
    I = A * np.sin(2 * np.pi * f * (t_sim / 1000.0)) + A
    """

    # シミュレーション実行
    t, v, u = simulate_izhikevich(a, b, c, d, I, t_max, dt_sim)

    #print(f"Simulation completed: {len(t)} time steps")
    #print(f"v length: {len(v)}, u length: {len(u)}")
    # ダウンサンプリングしてプロット用データを作成
    
    t_ds, v_ds, u_ds, I_ds = t[::100], v[::100], u[::100], I[::100]  # 100 ms ごとにダウンサンプリング

    # プロット
    plt.figure(figsize=(8, 4))
    plt.plot(t_ds, v_ds, label="mV")
    plt.plot(t_ds, I_ds, label="I (μA/cm²)", color='red', linestyle='--')
    plt.title("Izhikevich Neuron Model Simulation")
    plt.xlabel("ms")
    plt.ylabel("mV")
    plt.ylim(-80, 40)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    # 回復変数 u のプロット
    plt.figure(figsize=(8, 4))
    plt.plot(t_ds, u_ds, label="Recovery Variable u", color='orange')
    plt.title("Recovery Variable u in Izhikevich Neuron Model")
    plt.xlabel("ms")
    plt.ylabel("u")
    plt.ylim(6, 20)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()