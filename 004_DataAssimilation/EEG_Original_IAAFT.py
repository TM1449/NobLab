import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import datetime
import os
from pathlib import Path

# ---------- 0. 初期設定 ----------

# 0. スクリプトのディレクトリとターゲットディレクトリの設定
Base_Directory = Path(__file__).resolve().parent
print(f"Script Directory: {Base_Directory}")

Data_Directory = Base_Directory/"Data"/"EEG_Signal_Pzch.csv"
print(f"Data Directory: {Data_Directory}")

Result_Directory = Base_Directory/"Data"/"Result"

# ---------- 0. データの読み込み ----------
df = pd.read_csv(Data_Directory, header=None)
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

low_cutoff = 30.0  # 下限周波数 (Hz)
high_cutoff = 55.0  # 上限周波数 (Hz)
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

def New_DirPath():
    Dir_Path = f"{Result_Directory}/{NowTime}"
    os.makedirs(Dir_Path, exist_ok=True)

# ==========================================
NowTime = TimeDate()

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
plt.savefig(f"{Result_Directory}/{NowTime}/EEG_Original_Time_Series.png")
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
plt.savefig(f"{Result_Directory}/{NowTime}/EEG_Original_Power_Spectrum.png")
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
plt.savefig(f"{Result_Directory}/{NowTime}/EEG_Filtered_Time_Series.png")
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
plt.savefig(f"{Result_Directory}/{NowTime}/EEG_Filtered_Time_Series_Short.png")
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
plt.savefig(f"{Result_Directory}/{NowTime}/EEG_Comparison_Time_Series.png")
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
plt.savefig(f"{Result_Directory}/{NowTime}/EEG_Filtered_Power_Spectrum.png")
plt.show()

# ==========================================
# 3. IAAFTサロゲートの生成
# ==========================================

def iaaft(original_signal, max_iter=1000, tol=1e-12):
    """
    IAAFT (Iterative Amplitude Adjusted Fourier Transform) サロゲートデータを生成する。

    :param original_signal: 元の時系列データ (1D numpy array)
    :param max_iter: 最大反復回数
    :param tol: 収束判定のための許容誤差
    :return: IAAFTサロゲートデータ (1D numpy array)
    """
    # 1. 元の信号の振幅スペクトルと、ソートされた値を取得
    original_amps = np.abs(np.fft.fft(original_signal))
    sorted_original = np.sort(original_signal)

    # 2. 元の信号をランダムにシャッフルして初期サロゲートを作成
    surrogate = np.random.permutation(original_signal)

    for i in range(max_iter):
        # 3. 現在のサロゲートの位相スペクトルを取得
        surrogate_fft = np.fft.fft(surrogate)
        surrogate_phases = np.angle(surrogate_fft)

        # 4. 元の信号の振幅と、現在のサロゲートの位相を組み合わせて新しいスペクトルを作成
        new_fft = original_amps * np.exp(1j * surrogate_phases)

        # 5. 逆フーリエ変換して、時間領域の信号に戻す
        time_domain_surrogate = np.fft.ifft(new_fft).real
        
        # 6. 元の信号の振幅分布に合わせるため、値を置き換える
        #    time_domain_surrogateの大小関係（順位）を保持したまま、
        #    sorted_originalの値で置き換える
        ranks = time_domain_surrogate.argsort().argsort()
        new_surrogate = sorted_original[ranks]
        
        # 収束チェック
        # 現在のサロゲートと前の反復のサロゲートの差が許容誤差以下であれば終了
        current_rms = np.sqrt(np.mean((new_surrogate - surrogate)**2))
        if current_rms < tol:
            break
            
        surrogate = new_surrogate

    return surrogate

# フィルタリング後の信号からIAAFTサロゲートを1つ生成
iaaft_surrogate = iaaft(Filtered_Signal)

# ---------- IAAFTサロゲートデータの解析 ----------
# フーリエ変換
iaaft_fft = np.fft.fft(iaaft_surrogate)
# パワースペクトルの計算
iaaft_power = (np.abs(iaaft_fft) / N)**2
# 周波数軸の計算
iaaft_freq = np.fft.fftfreq(N, d = 1.0 / sampling_rate)

# 片側スペクトルの範囲を抽出
iaaft_freqs = iaaft_freq[:half_N]
iaaft_power_half = iaaft_power[:half_N]
# 2倍補正
iaaft_power_half[1:-1] *= 2


# ==========================================
# 4. IAAFTサロゲートデータの描画
# ==========================================

# 6-1. IAAFTサロゲートの時系列データをプロット
plt.figure(figsize=(8, 4))
plt.plot(df['Time'], Filtered_Signal, label=f'Filtered Signal', color='orange', alpha=0.6)
plt.plot(df['Time'], iaaft_surrogate, label='IAAFT Surrogate Signal', color='green', alpha=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Filtered Signal and IAAFT Surrogate')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(f"{Result_Directory}/{NowTime}/IAAFT_Surrogate_Time_Series.png")
plt.show()

# 6-2. フィルタリング後の時系列データのプロット（短い時間軸）
plt.figure(figsize=(8, 4))
plt.plot(df['Time'], Filtered_Signal, label=f'Filtered Signal({low_cutoff} - {high_cutoff} (Hz))', color='orange', alpha=0.6)
plt.plot(df['Time'], iaaft_surrogate, label='IAAFT Surrogate Signal', color='green', alpha=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('EEG Data (Short Time Series)')
plt.xlim(0.5, 3.5)  # 最初の10秒間を表示
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(f"{Result_Directory}/{NowTime}/EEG_Filtered_Time_Series_Short_short.png")
plt.show()


# 7. フィルタリング後の信号とIAAFTサロゲートのパワースペクトルを比較
plt.figure(figsize=(8, 4))
plt.plot(Filtered_freqs, Filtered_power, label=f'Filtered Power Spectrum', color='orange')
plt.plot(iaaft_freqs, iaaft_power_half, label='IAAFT Surrogate Power Spectrum', color='green', linestyle='--', alpha=0.8)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Power Spectrum Comparison')
plt.xticks(np.arange(0,101,10))
plt.grid()
plt.xlim(0, sampling_rate / 2)
plt.legend()
plt.tight_layout()
plt.savefig(f"{Result_Directory}/{NowTime}/IAAFT_Power_Spectrum_Comparison.png")
plt.show()

"""
# 8. 振幅分布をヒストグラムで比較
plt.figure(figsize=(8, 4))
plt.hist(Filtered_Signal, bins=50, density=True, label='Filtered Signal', color='orange', alpha=0.7)
plt.hist(iaaft_surrogate, bins=50, density=True, label='IAAFT Surrogate', color='green', alpha=0.7)
plt.xlabel('Amplitude')
plt.ylabel('Probability Density')
plt.title('Amplitude Distribution Comparison')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"{Result_Directory}/{NowTime}/IAAFT_Amplitude_Distribution.png")
plt.show()
"""