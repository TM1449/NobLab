import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import datetime
import os
from pathlib import Path

# ---------- 0-1. 初期設定 ----------

# 0. スクリプトのディレクトリとターゲットディレクトリの設定
Base_Directory = Path(__file__).resolve().parent
#print(f"Script Directory: {Base_Directory}")

Data_Directory = Base_Directory/"Data"/"EEG_Signal_Pzch.csv"
#print(f"Data Directory: {Data_Directory}")

Result_Directory = Base_Directory/"Data"/"Result"
#print(f"Result Directory: {Result_Directory}")

# ---------- 0-2. データの読み込み ----------
df = pd.read_csv(Data_Directory, header=None)
df.columns = ['Time', 'EEG_Signal']

# サンプリングレート (Hz)
sampling_rate = 200
# データの長さ
N = len(df)

# データの確認
print(f"EEG Data\n\n{df}")

# ==========================================
# ---------- 1. 各関数の定義 ----------
# ==========================================

# 現時刻の取得する関数
def TimeDate():
    Time_Delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(Time_Delta ,'JST')
    Now = datetime.datetime.now(JST)
    Date = Now.strftime('%Y_%m_%d_%H_%M_%S')
    return Date

# 現在の時刻を取得
NowTime = TimeDate()

# ディレクトリの作成する関数
def New_DirPath():
    Dir_Path = f"{Result_Directory}/{NowTime}"
    os.makedirs(Dir_Path, exist_ok=True)

# ディレクトリの作成
New_DirPath()

# バンドパスフィルタの定義 
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

# パワースペクトルの計算関数
def calculate_power_spectrum(data, sampling_rate):
    n_data = len(data)
    fft_data = np.fft.fft(data)
    power = (np.abs(fft_data) / n_data)**2
    freq = np.fft.fftfreq(n_data, d=1.0 / sampling_rate)
    
    # 片側スペクトルに変換
    half_n = n_data // 2
    power_half = power[:half_n]
    power_half[1:-1] *= 2 # 直流成分とナイキスト成分以外を2倍
    freq_half = freq[:half_n]
    
    return freq_half, power_half

# IAAFTサロゲートの生成
def IAAFT_Surrogates(original_data, max_iter=1000, tol=1e-16):
    """
    IAAFT (Iterative Amplitude Adjusted Fourier Transform) サロゲートデータを生成する。

    :param original_data: 元の時系列データ (1D numpy array)
    :param max_iter: 最大反復回数
    :param tol: 収束判定のための許容誤差
    :return: IAAFTサロゲートデータ (1D numpy array)
    """

    n = len(original_data)

    # 1. 元の信号の振幅スペクトルと、ソートされた値を取得
    Original_Magnitudes = np.abs(np.fft.rfft(original_data))
    Sorted_Original = np.sort(original_data)

    # 2. 元の信号をランダムにシャッフルして初期サロゲートを作成
    Current_Surrogate = np.random.permutation(original_data)

    # 反復計算を開始
    for i in range(max_iter):
        # 3. 現在のサロゲートの位相スペクトルを取得
        Surrogate_FFT = np.fft.rfft(Current_Surrogate)
        Surrogate_Phases = np.angle(Surrogate_FFT)

        # 4. 元の信号の振幅と、現在のサロゲートの位相を組み合わせて新しいスペクトルを作成
        New_FFT = Original_Magnitudes * np.exp(1j * Surrogate_Phases)

        # 5. 逆フーリエ変換して、時間領域の信号に戻す
        Surrogate_with_Spectrum = np.fft.irfft(New_FFT, n)
        

        # 6. 元の信号の振幅分布に合わせるため、値を置き換える
        Ranks = np.argsort(np.argsort(Surrogate_with_Spectrum))
        New_Surrogate = Sorted_Original[Ranks]
        
        # 収束チェック
        # 現在のサロゲートと前の反復のサロゲートの差が許容誤差以下であれば終了
        if np.allclose(Current_Surrogate, New_Surrogate, atol=tol):
            print(f"Process Converged in {i+1} iterations.")
            break
            
        Current_Surrogate = New_Surrogate

    return Current_Surrogate

# ==========================================
# ---------- 2. データ解析 ----------
# ==========================================
# 元データのパワースペクトル
Original_freqs, Original_power \
    = calculate_power_spectrum(df['EEG_Signal'], sampling_rate)

# フィルタリングのパラメータ
low_cutoff = 30.0  # 下限周波数 (Hz)
high_cutoff = 55.0  # 上限周波数 (Hz)
order = 8  # フィルタの次数

# バンドパスフィルタの適用
Filtered_Signal = Bandpass_Filter(df['EEG_Signal'], lowcut=low_cutoff, 
                        highcut=high_cutoff, sampling_rate=sampling_rate, order=order)

# フィルタ後の信号のパワースペクトル
Filtered_freqs, Filtered_power \
    = calculate_power_spectrum(Filtered_Signal, sampling_rate)

# ==========================================
# ---------- 4. IAAFTサロゲートの生成 ----------
# ==========================================

# フィルタリング後の信号からIAAFTサロゲートを1つ生成
IAAFT_Surrogates_Data = IAAFT_Surrogates(Filtered_Signal)

# IAAFTサロゲートのパワースペクトル
IAAFT_FREQ, IAAFT_POWER \
    = calculate_power_spectrum(IAAFT_Surrogates_Data, sampling_rate)



# ==========================================
# 5.データの描画
# ==========================================
# 1-1. 生データの時系列データをプロット
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

# 1-2. 生データのパワースペクトルをプロット
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

# 2-1. フィルタリング後の時系列データのプロット
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

# 2-2. フィルタリング後の時系列データのプロット（短い時間軸）
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

# 2-3. フィルタ後のパワースペクトルをプロット
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

# 3. 元の信号とフィルタ後の信号を比較するプロット
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

# ==========================================
# ==========================================

# 4. IAAFTサロゲートの時系列データをプロット
plt.figure(figsize=(8, 4))
plt.plot(df['Time'], Filtered_Signal, label=f'Filtered Signal', color='orange', alpha=0.6)
plt.plot(df['Time'], IAAFT_Surrogates_Data, label='IAAFT Surrogate Signal', color='green', alpha=0.8)
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
plt.plot(df['Time'], IAAFT_Surrogates_Data, label='IAAFT Surrogate Signal', color='green', alpha=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('EEG Data (Short Time Series)')
plt.xlim(0.5, 3.5)  # 最初の3.5秒間を表示
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(f"{Result_Directory}/{NowTime}/EEG_Filtered_Time_Series_Short_short.png")
plt.show()


# 7. フィルタリング後の信号とIAAFTサロゲートのパワースペクトルを比較
plt.figure(figsize=(8, 4))
plt.plot(Filtered_freqs, Filtered_power, label=f'Filtered Power Spectrum', color='orange')
plt.plot(IAAFT_FREQ, IAAFT_POWER, label='IAAFT Surrogate Power Spectrum', color='green', linestyle='--', alpha=0.8)
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
