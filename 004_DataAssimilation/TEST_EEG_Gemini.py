import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# --- 1. バンドパスフィルタ関数の定義 ---
# この関数は、データ、カットオフ周波数、サンプリングレートを引数に取り、
# フィルタリングされたデータを返します。
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    指定された周波数帯域の信号を通過させるバンドパスフィルタを適用します。
    
    Args:
        data (array-like): フィルタを適用する時系列データ。
        lowcut (float): 低域カットオフ周波数 (Hz)。
        highcut (float): 高域カットオフ周波数 (Hz)。
        fs (int): データのサンプリングレート (Hz)。
        order (int, optional): フィルタの次数。デフォルトは5。

    Returns:
        array-like: フィルタリングされたデータ。
    """
    nyq = 0.5 * fs  # ナイキスト周波数
    low = lowcut / nyq
    high = highcut / nyq
    # バターワースフィルタの係数を計算
    b, a = butter(order, [low, high], btype='band')
    # filtfiltを使い、位相のずれなくフィルタを適用
    y = filtfilt(b, a, data)
    return y

# --- 2. データの読み込みと基本設定 ---
try:
    # データを読み込みます。ファイルパスはご自身の環境に合わせて変更してください。
    df = pd.read_csv("./0004_DataAssimilation/Data/EEG_Signal_Pzch.csv", header=None)
    df.columns = ['Time', 'EEG_Signal']
    print("--- 元データの先頭5行 ---")
    print(df.head())
except FileNotFoundError:
    print("エラー: EEG_Signal_Pzch.csv が見つかりません。")
    print("ファイルパスが正しいか確認してください。")
    exit()

sampling_rate = 200  # サンプリングレート (Hz)
# データ長は固定値ではなく、読み込んだデータの実際の長さを使用するのが堅牢です。
N = len(df['EEG_Signal'])

# --- 3. 元データの可視化 (時間領域) ---
plt.figure(figsize=(12, 6))
plt.plot(df['Time'], df['EEG_Signal'], label='Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('EEG Signal (μV)')
plt.title('Original EEG Signal over Time')
plt.grid(True)
plt.legend()
plt.savefig("original_eeg_time_domain.png")
print("\n元のEEG信号のグラフを original_eeg_time_domain.png として保存しました。")

# --- 4. 元データのFFT解析 ---
EEG_fft_original = np.fft.fft(df['EEG_Signal'])
freq = np.fft.fftfreq(N, d=1/sampling_rate)

plt.figure(figsize=(12, 6))
# 正の周波数成分のみをプロット (N//2まで)
plt.plot(freq[:N//2], np.abs(EEG_fft_original)[:N//2], label='Original FFT')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum of Original EEG Signal')
plt.grid(True)
plt.legend()
plt.savefig("original_eeg_frequency_domain.png")
print("元のEEG信号の周波数スペクトルを original_eeg_frequency_domain.png として保存しました。")


# --- 5. バンドパスフィルタの適用 ---
# ★★★ ここで抽出したい周波数範囲を指定します ★★★
low_cutoff = 30   # α波の下限 (例)
high_cutoff = 55.0 # α波の上限 (例)
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★

print(f"\nバンドパスフィルタを適用します (通過帯域: {low_cutoff} - {high_cutoff} Hz)...")
# 関数を呼び出してフィルタを適用
filtered_eeg = bandpass_filter(df['EEG_Signal'], lowcut=low_cutoff, highcut=high_cutoff, fs=sampling_rate, order=5)


# --- 6. フィルタ後のデータの可視化 (時間領域での比較) ---
plt.figure(figsize=(12, 6))
plt.plot(df['Time'], df['EEG_Signal'], label='Original Signal', color='silver', alpha=0.8)
plt.plot(df['Time'], filtered_eeg, label=f'Filtered Signal ({low_cutoff}-{high_cutoff} Hz)', color='red')
plt.xlabel('Time (s)')
plt.ylabel('EEG Signal (μV)')
plt.title('Original vs. Band-pass Filtered EEG Signal')
plt.grid(True)
plt.legend()
plt.savefig("filtered_eeg_time_comparison.png")
print("フィルタ前後の信号比較グラフを filtered_eeg_time_comparison.png として保存しました。")


# --- 7. フィルタ後のデータのFFT解析 (周波数領域での比較) ---
EEG_fft_filtered = np.fft.fft(filtered_eeg)

plt.figure(figsize=(12, 6))
# 元の信号のFFT
plt.plot(freq[:N//2], np.abs(EEG_fft_original)[:N//2], label='Original FFT', color='silver', alpha=0.8)
# フィルタ後の信号のFFT
plt.plot(freq[:N//2], np.abs(EEG_fft_filtered)[:N//2], label=f'Filtered FFT ({low_cutoff}-{high_cutoff} Hz)', color='red')

# フィルタの通過帯域を強調表示
plt.axvspan(low_cutoff, high_cutoff, color='orange', alpha=0.3, label='Passband')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum: Original vs. Filtered')
plt.grid(True)
plt.legend()
plt.savefig("filtered_eeg_frequency_comparison.png")
print("フィルタ前後の周波数スペクトル比較グラフを filtered_eeg_frequency_comparison.png として保存しました。")

# 全てのグラフを画面に表示
plt.show()

