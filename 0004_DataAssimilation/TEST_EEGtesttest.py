import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# データの読み込み
df = pd.read_csv("./0004_DataAssimilation/Data/EEG_Signal_Pzch.csv", header=None)
df.columns = ['Time', 'EEG_Signal']
print(f"EEG Data\n\n{df}")

sampling_rate = 200  # サンプリングレート (Hz)
N = len(df)  # データの長さ

#時系列データのプロット
plt.figure(figsize=(8, 4))
plt.plot(df['Time'], df['EEG_Signal'])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('EEG Data')
plt.grid()
plt.show()

# --- データ解析 ---
# フーリエ変換
EEG_fft = np.fft.fft(df['EEG_Signal'])
# 振幅スペクトルの計算
EEG_Spectrum = np.abs(EEG_fft) / N
# パワースペクトルの計算
EEG_Power = pow(EEG_Spectrum, 2)

# 周波数軸の計算
EEG_freq = np.fft.fftfreq(N, d = 1.0 / sampling_rate)

# 片側スペクトルの範囲を抽出（正の周波数のみ）
half_N = N // 2
freqs = EEG_freq[:half_N]
power = EEG_Power[:half_N]

# 2倍補正（直流成分とナイキスト成分はそのまま）
power[1:-1] *= 2

# パワースペクトルをプロット
plt.figure(figsize=(8, 4))
plt.plot(freqs, power)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('EEG Power Spectrum')
plt.grid()
plt.xlim(0, sampling_rate / 2)
plt.show()