import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import datetime
import os

# ---------- 0. データの読み込み ----------
df = pd.read_csv("./004_DataAssimilation/Data/EEG_Signal_Pzch.csv", header=None)
df.columns = ['Time', 'EEG_Signal']
print(f"EEG Data\n\n{df}")

# サンプリングレート (Hz)
sampling_rate = 200
# データの長さ
N = len(df)

# ---------- 1. データ解析 ----------
# フーリエ変換
EEG_Original_fft = np.fft.fft(df['EEG_Signal'])
# 振幅スペクトルの計算
EEG_Original_Spectrum = np.abs(EEG_Original_fft) / N
# パワースペクトルの計算
EEG_Original_Power = pow(EEG_Original_Spectrum, 2)
# 周波数軸の計算
EEG_Original_freq = np.fft.fftfreq(N, d = 1.0 / sampling_rate)

# 片側スペクトルの範囲を抽出（正の周波数のみ）
half_N = N // 2
Original_freqs = EEG_Original_freq[:half_N]
Original_power = EEG_Original_Power[:half_N]
# 2倍補正（直流成分とナイキスト成分はそのまま）
Original_power[1:-1] *= 2

# ---------- 2. フィルタリング ----------
def Bandpass_Filter(data, lowcut, highcut, sampling_rate, order):
    """
    バンドパスフィルタを適用する関数
    :data: 入力信号
    :lowcut: 下限周波数
    :highcut: 上限周波数
    :sampling_rate: サンプリング周波数
    :order: フィルタの次数
    :return: フィルタ後の信号
    """
    nyq = sampling_rate / 2  # ナイキスト周波数
    low = lowcut / nyq # 下限周波数
    high = highcut / nyq # 上限周波数
    b, a = signal.butter(order, [low, high], btype='bandpass')
    y = signal.filtfilt(b, a, data)

    return y

low_cutoff = 8.0  # 下限周波数 (Hz)
high_cutoff = 13.0  # 上限周波数 (Hz)
order = 8  # フィルタの次数

# フィルタを適用
Filtered_Signal = Bandpass_Filter(df['EEG_Signal'], lowcut=low_cutoff, 
                        highcut=high_cutoff, sampling_rate=sampling_rate, order=order)

# フィルタ後の信号のフーリエ変換
EEG_Filtered_fft = np.fft.fft(Filtered_Signal)
# フィルタ後の振幅スペクトルの計算
EEG_Filtered_Spectrum = np.abs(EEG_Filtered_fft) / N
# フィルタ後のパワースペクトルの計算
EEG_Filtered_Power = pow(EEG_Filtered_Spectrum, 2)
# フィルタ後の周波数軸の計算
EEG_Filtered_freq = np.fft.fftfreq(N, d=1.0 / sampling_rate)

# フィルタ後の片側スペクトルの範囲を抽出（正の周波数のみ）
half_N = N // 2
Filtered_freqs = EEG_Filtered_freq[:half_N]
Filtered_power = EEG_Filtered_Power[:half_N]
# 2倍補正（直流成分とナイキスト成分はそのまま）
Filtered_power[1:-1] *= 2

# ==========================================
# 0. ディレクトリの作成
def TimeDate():
    Time_Delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(Time_Delta ,'JST')
    Now = datetime.datetime.now(JST)
    Date = Now.strftime('%Y_%m_%d_%H_%M_%S')
    return Date

NowTime = TimeDate()

def New_DirPath():
    Dir_Path = f"./004_DataAssimilation/Data/Result/{NowTime}"
    os.makedirs(Dir_Path, exist_ok=True)
    
New_DirPath()

# 1. 生データの時系列データをプロット
plt.figure(figsize=(8, 4))
plt.plot(df['Time'], df['EEG_Signal'], label='Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('EEG Data')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(f"./004_DataAssimilation/Data/Result/{NowTime}/EEG_Original_Time_Series.png")
plt.show()

# 2. 生データのパワースペクトルをプロット
plt.figure(figsize=(8, 4))
plt.plot(Original_freqs, Original_power, label='Original Power Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Power Spectrum')
plt.xticks(np.arange(0,101,10))
plt.grid()
plt.xlim(0, sampling_rate / 2)
plt.legend()
plt.tight_layout()
plt.savefig(f"./004_DataAssimilation/Data/Result/{NowTime}/EEG_Original_Power_Spectrum.png")
plt.show()

# 3. フィルタリング後の時系列データのプロット
plt.figure(figsize=(8, 4))
plt.plot(df['Time'], Filtered_Signal, label=f'Filtered Signal({low_cutoff} - {high_cutoff} (Hz))', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('EEG Data')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(f"./004_DataAssimilation/Data/Result/{NowTime}/EEG_Filtered_Time_Series.png")
plt.show()

# 3-1. フィルタリング後の時系列データのプロット（短い時間軸）
plt.figure(figsize=(8, 4))
plt.plot(df['Time'], Filtered_Signal, label=f'Filtered Signal({low_cutoff} - {high_cutoff} (Hz))', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('EEG Data (Short Time Series)')
plt.xlim(0.5, 3.5)  # 最初の10秒間を表示
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(f"./004_DataAssimilation/Data/Result/{NowTime}/EEG_Filtered_Time_Series_Short.png")
plt.show()



# 4. 元の信号とフィルタ後の信号を比較するプロット
plt.figure(figsize=(8, 4))
plt.plot(df['Time'], df['EEG_Signal'], label='Original Signal', alpha=0.5)
plt.plot(df['Time'], Filtered_Signal, label=f'Filtered Signal({low_cutoff} - {high_cutoff} (Hz))', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Comparison of Original and Filtered EEG Signal')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(f"./004_DataAssimilation/Data/Result/{NowTime}/EEG_Comparison_Time_Series.png")
plt.show()

# 5. フィルタ後のパワースペクトルをプロット
plt.figure(figsize=(8,4))
plt.plot(Filtered_freqs, Filtered_power, label=f'Filtered Power Spectrum({low_cutoff} - {high_cutoff} (Hz))', color='orange')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Filtered Power Spectrum')
plt.xticks(np.arange(0,101,10))
plt.grid()
plt.xlim(0, sampling_rate / 2)
plt.legend()
plt.tight_layout()
plt.savefig(f"./004_DataAssimilation/Data/Result/{NowTime}/EEG_Filtered_Power_Spectrum.png")
plt.show()