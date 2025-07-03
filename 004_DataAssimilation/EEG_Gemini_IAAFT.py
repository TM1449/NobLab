import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import datetime
import os
from pathlib import Path

# ---------- IAAFT サロゲート生成関数 ----------
def iaaft(x, max_iter=1000, tol=1e-8, random_seed=None):
    x = np.asarray(x)
    n = len(x)
    A = np.abs(np.fft.fft(x))
    sorted_x = np.sort(x)
    if random_seed is not None:
        np.random.seed(random_seed)
    random_phases = np.exp(1j * 2 * np.pi * np.random.rand(n))
    y = np.real(np.fft.ifft(A * random_phases))
    prev = y.copy()
    for i in range(max_iter):
        ranks = y.argsort().argsort()
        y = sorted_x[ranks]
        fft_y = np.fft.fft(y)
        phases = np.exp(1j * np.angle(fft_y))
        y = np.real(np.fft.ifft(A * phases))
        if np.linalg.norm(y - prev) < tol:
            break
        prev = y.copy()
    return y

# ---------- ファジーエントロピー解析関数（1次元） ----------
def fuzzy_entropy(signal, m=2, r=0.2):
    N = len(signal)
    phi_m = 0.0
    phi_m1 = 0.0
    r *= np.std(signal)
    for i in range(N - m):
        xi = signal[i:i + m]
        for j in range(N - m):
            if i == j:
                continue
            xj = signal[j:j + m]
            d = np.max(np.abs(xi - xj))
            phi_m += np.exp(-(d**2) / r)
    for i in range(N - m - 1):
        xi = signal[i:i + m + 1]
        for j in range(N - m - 1):
            if i == j:
                continue
            xj = signal[j:j + m + 1]
            d = np.max(np.abs(xi - xj))
            phi_m1 += np.exp(-(d**2) / r)
    return -np.log(phi_m1 / phi_m) if phi_m1 > 0 and phi_m > 0 else np.nan

# ---------- マルチスケールファジーエントロピー解析関数 ----------
def multiscale_fuzzy_entropy(signal, scales=range(1, 21), m=2, r=0.2):
    entropies = []
    for scale in scales:
        if scale == 1:
            coarse = signal
        else:
            coarse = [np.mean(signal[i:i+scale]) for i in range(0, len(signal)-scale+1, scale)]
        ent = fuzzy_entropy(np.array(coarse), m=m, r=r)
        entropies.append(ent)
    return entropies

# ---------- 0. 初期設定 ----------
Base_Directory = Path(__file__).resolve().parent
Data_Directory = Base_Directory / "Data" / "EEG_Signal_Pzch.csv"
Result_Directory = Base_Directory / "Data" / "Result"
if not Data_Directory.exists():
    raise FileNotFoundError(f"データファイルが見つかりません: {Data_Directory}")
df = pd.read_csv(Data_Directory, header=None)
df.columns = ['Time', 'EEG_Signal']
sampling_rate = 200
N = len(df)

def compute_spectrum(signal_data, sampling_rate):
    fft_vals = np.fft.fft(signal_data)
    spectrum = np.abs(fft_vals) / N
    power = spectrum**2
    freqs = np.fft.fftfreq(N, d=1.0/sampling_rate)
    half = N // 2
    p = power[:half].copy()
    p[1:-1] *= 2
    return freqs[:half], p

Original_freqs, Original_power = compute_spectrum(df['EEG_Signal'], sampling_rate)

def Bandpass_Filter(data, lowcut, highcut, sr, order):
    nyq = sr / 2
    b, a = signal.butter(order, [lowcut/nyq, highcut/nyq], btype='bandpass')
    return signal.filtfilt(b, a, data)

low_cutoff, high_cutoff, order = 8.0, 13.0, 8
Filtered_Signal = Bandpass_Filter(df['EEG_Signal'], low_cutoff, high_cutoff, sampling_rate, order)
Filtered_freqs, Filtered_power = compute_spectrum(Filtered_Signal, sampling_rate)

print("Generating IAAFT surrogate...")
iaaft_surrogate = iaaft(Filtered_Signal, max_iter=1000, tol=1e-8, random_seed=42)
Surrogate_freqs, Surrogate_power = compute_spectrum(iaaft_surrogate, sampling_rate)

def make_result_dir(base, prefix="Result"):    
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
    path = base / "Data" / prefix / now
    path.mkdir(parents=True, exist_ok=True)
    return path

out_dir = make_result_dir(Base_Directory)
print(f"Results saved to: {out_dir}")

plt.figure(figsize=(10,5))
plt.plot(df['Time'], df['EEG_Signal'], label='Original', alpha=0.6)
plt.plot(df['Time'], Filtered_Signal, label=f'Filtered ({low_cutoff}-{high_cutoff} Hz)', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir/"Original_vs_Filtered_TimeSeries.png")
plt.close()

plt.figure(figsize=(10,5))
plt.plot(Original_freqs, Original_power, label='Original', alpha=0.6)
plt.plot(Filtered_freqs, Filtered_power, label='Filtered', color='orange')
plt.xlim(0, 40)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir/"Original_vs_Filtered_Spectrum.png")
plt.close()

plt.figure(figsize=(10,5))
plt.plot(df['Time'], Filtered_Signal, label='Filtered', color='orange')
plt.plot(df['Time'], iaaft_surrogate, label='IAAFT Surrogate', color='green', alpha=0.8)
plt.xlim(0.5, 3.5)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir/"Filtered_vs_Surrogate_TimeSeries.png")
plt.close()

plt.figure(figsize=(10,5))
plt.semilogy(Filtered_freqs, Filtered_power, label='Filtered', color='orange')
plt.semilogy(Surrogate_freqs, Surrogate_power, label='IAAFT Surrogate', color='green', linestyle='--', alpha=0.8)
plt.xlim(0,40)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (log scale)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir/"Filtered_vs_Surrogate_Spectrum.png")
plt.close()

plt.figure(figsize=(10,5))
plt.hist(Filtered_Signal, bins=50, density=True, alpha=0.7, label='Filtered')
plt.hist(iaaft_surrogate, bins=50, density=True, alpha=0.7, label='Surrogate')
plt.xlabel('Amplitude')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir/"Filtered_vs_Surrogate_Histogram.png")
plt.close()

# ---------- マルチスケールファジーエントロピー解析とプロット ----------
scales = range(1, 21)
entropy_filtered = multiscale_fuzzy_entropy(Filtered_Signal, scales=scales)
entropy_surrogate = multiscale_fuzzy_entropy(iaaft_surrogate, scales=scales)

plt.figure(figsize=(10,5))
plt.plot(scales, entropy_filtered, label='Filtered Signal', marker='o')
plt.plot(scales, entropy_surrogate, label='IAAFT Surrogate', marker='x')
plt.xlabel('Scale')
plt.ylabel('Fuzzy Entropy')
plt.title('Multiscale Fuzzy Entropy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir/"Multiscale_Fuzzy_Entropy.png")
plt.close()

print("Analysis complete.")
