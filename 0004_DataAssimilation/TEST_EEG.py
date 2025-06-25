import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# データの読み込み
df = pd.read_csv("./0004_DataAssimilation/Data/EEG_Signal_Pzch.csv", header=None)
df.columns = ['Time', 'EEG_Signal']
print(df)

sampling_rate = 200  # サンプリングレート (Hz)
time_sec = 60  # 時間 (秒)
N = time_sec * sampling_rate  # 総サンプル数

plt.figure(figsize=(8, 4))
plt.plot(df['Time'], df['EEG_Signal'])
plt.xlabel('Time (s)')
plt.ylabel('EEG Signal')
plt.title('EEG Signal over Time')
plt.grid()
plt.show()

EEG_fft = np.fft.fft(df['EEG_Signal'])
EEG_freq = np.fft.fftfreq(N, d=1/sampling_rate)
plt.figure(figsize=(8, 4))
plt.plot(EEG_freq[:N//2], np.abs(EEG_fft)[:N//2])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT of EEG Signal')
plt.grid()
plt.show()