import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class IzhikevichNeuron:
    def __init__(self, a, b, c, d, time, dt):
        """
        Izhikevich ニューロンモデルの初期化

        Parameters
        ----------
        a, b, c, d : float
            モデルパラメータ

        time : float
            シミュレーション時間 (ms)

        dt : float
            時間刻み幅 (ms)
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        self.time = time
        self.dt = dt

    def Calculate(self, inputs):
        """
        ニューロンの状態を1ステップ更新する

        Parameters
        ----------
        I : float
            刺激電流
        dt : float
            時間刻み幅 (ms)
        
        Returns
        -------
        v : float
            膜電位
        u : float
            回復変数
        """

        time_step = int(self.time / self.dt)

        self.v = np.zeros(time_step)
        self.u = np.zeros(time_step)

        self.v[0] = self.c
        self.u[0] = self.d

        for i in tqdm(range(time_step - 1)):

            # uの更新
            du = self.a * (self.b * self.v[i] - self.u[i])
            self.u[i + 1] = self.u[i] + du * self.dt

            # vの更新
            dv = 0.04 * self.v[i] ** 2 + 5 * self.v[i] + 140 - self.u[i] + inputs[i]
            self.v[i + 1] = self.v[i] + dv * self.dt

            # 発火処理
            if self.v[i + 1] >= 30:
                self.v[i + 1] = self.c
                self.u[i + 1] += self.d

        return self.v, self.u
    

if __name__ == "__main__":
    time = 600 # 実験時間 (ms)
    dt = 0.01 # 時間刻み幅 (ms)
    t = np.arange(0, time, dt)  # 時間配列

    # モデルパラメータ（Regular Spiking）
    a, b, c, d = 0.02, 0.2, -65, 8

    # 入力電流の設定
    input_data = 10 * np.sin(0.1 * np.arange(0, time, dt))  # 正弦波入力
    #input_data = np.where(input_data > 0, 20, 0) + 10 * np.random.rand(int(time/dt))
    #input_data_2 = np.cos(0.4 * np.arange(0, time, dt) + 0.5)
    #input_data_2 = np.where(input_data_2 > 0, 10, 0)
    #input_data += input_data_2

    # Izhikevich ニューロンモデルの生成
    neuron = IzhikevichNeuron(a, b, c, d, time, dt)
    # ニューロンの計算
    v, u = neuron.Calculate(input_data)

    # 結果のプロット
    plt.figure(figsize=(8, 4))
    plt.plot(t, v, label='Membrane Potential (v)')
    plt.plot(t, input_data, label='Input Current (I)', alpha=0.5)
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (v)')
    plt.legend()
    plt.show()