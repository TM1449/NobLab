import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def generate_lorenz_data(length=5000, dt=0.01, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    ローレンツ方程式からカオス時系列データを生成します。

    Args:
        length (int): 生成する時系列の長さ。
        dt (float): 時間の刻み幅。
        sigma, rho, beta (float): ローレンツ方程式のパラメータ。

    Returns:
        numpy.ndarray: 生成された時系列データ（x成分）。
    """
    
    # ローレンツ方程式の定義
    def lorenz_system(t, state, sigma, rho, beta):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]

    # 初期値
    initial_state = [0.0, 1.0, 1.05]
    
    # 解くべき時間の範囲
    t_span = [0, length * dt]
    t_eval = np.arange(0, length * dt, dt)
    
    # 微分方程式を解く
    solution = solve_ivp(
        fun=lorenz_system,
        t_span=t_span,
        y0=initial_state,
        args=(sigma, rho, beta),
        dense_output=True,
        t_eval=t_eval
    )
    
    # x成分を時系列データとして返す
    return solution.y[0]


def generate_sin_wave(length=1000, num_cycles=10):
    """
    ノイズを含むサイン波の時系列データを生成します。

    Args:
        length (int): 生成する時系列の長さ。
        num_cycles (int): データ長に含めるサイクルの数。
        noise_level (float): サイン波に加える正規分布ノイズの標準偏差。

    Returns:
        numpy.ndarray: 生成された時系列データ。
    """
    # 時間軸を生成
    t = np.linspace(0, num_cycles * 2 * np.pi, length)
    
    # サイン波を生成
    sine_wave = np.sin(t)
    
    # サイン波にノイズを加える
    return sine_wave

def Random_Shuffle_Surrogate(data):
    """
    データをランダムにシャッフルしてサロゲートデータを生成します。

    Args:
        data (numpy.ndarray): 入力データ。

    Returns:
        numpy.ndarray: シャッフルされたサロゲートデータ。
    """
    surrogate = np.random.permutation(data)
    return surrogate

def Fourier_Transform_Surrogate(data):
    """
    フーリエ変換を用いてサロゲートデータを生成します。

    Args:
        data (numpy.ndarray): 入力データ。

    Returns:
        numpy.ndarray: フーリエ変換されたサロゲートデータ。
    """
    # 入力データの長さを取得
    n = len(data)
    # フーリエ変換
    y = np.fft.fft(data)
    
    # ランダムな位相を持つベクトルを生成
    if n % 2 == 0:
        # データ長が偶数の場合
        l = n // 2 - 1
        random_phases = np.exp(2j * np.pi * np.random.rand(l))
        # 位相ベクトルを構成 [直流, 正周波数, ナイキスト, 負周波数]
        v = np.concatenate([
            [1],                     # 直流成分 (DC)
            random_phases,           # 正の周波数
            [1],                     # ナイキスト周波数
            np.flip(np.conj(random_phases)) # 負の周波数 (エルミート対称性)
        ])
    else:
        # データ長が奇数の場合
        l = (n - 1) // 2
        random_phases = np.exp(2j * np.pi * np.random.rand(l))
        # 位相ベクトルを構成 [直流, 正周波数, 負周波数]
        v = np.concatenate([
            [1],                     # 直流成分 (DC)
            random_phases,           # 正の周波数
            np.flip(np.conj(random_phases)) # 負の周波数 (エルミート対称性)
        ])

    # 元のスペクトルにランダム位相を乗算し、逆フーリエ変換
    z = np.fft.ifft(y * v)

    # 数値誤差で残る微小な虚数部を捨てて実数部を返す
    return z.real

def generate_iaaft_surrogates(original_data, num_surrogates=1, max_iter=1000):
    """
    IAAFTアルゴリズムに基づき、サロゲートデータを生成します。

    元の時系列データが持つパワースペクトルと振幅分布を保持した
    サロゲートデータを生成します。

    Args:
        original_data (np.ndarray): 1次元の元時系列データ。
        num_surrogates (int): 生成するサロゲートデータの数。
        max_iter (int): 収束計算の最大反復回数。

    Returns:
        np.ndarray: 生成されたサロゲートデータ。
                    num_surrogatesが1より大きい場合、各行が1つのサロゲート。
    """
    # --- 入力検証 ---
    if not isinstance(original_data, np.ndarray) or original_data.ndim != 1:
        raise TypeError("入力 'original_data' は1次元のNumPy配列である必要があります。")
    if not isinstance(num_surrogates, int) or num_surrogates < 1:
        raise ValueError("'num_surrogates' は正の整数である必要があります。")

    n = len(original_data)

    # --- 元データが持つべき特性を計算 ---
    # 1. パワースペクトルの振幅
    original_magnitudes = np.abs(np.fft.rfft(original_data))
    # 2. 振幅分布（ソート済みデータ）
    sorted_original = np.sort(original_data)

    surrogates_list = []
    # --- 指定された数だけサロゲートを生成 ---
    for _ in range(num_surrogates):
        # 最初のサロゲートは元データのランダムシャッフルから開始
        current_surrogate = np.random.permutation(original_data)

        # --- 収束するまで反復計算 ---
        for i in range(max_iter):
            # 1. パワースペクトルを元データのものに合わせる
            surrogate_fft = np.fft.rfft(current_surrogate)
            phases = np.angle(surrogate_fft)
            new_fft = original_magnitudes * np.exp(1j * phases)
            surrogate_with_spectrum = np.fft.irfft(new_fft, n=n)

            # 2. 振幅分布を元データのものに合わせる
            #   スペクトル調整後のデータの順位を取得し、その順位に従って
            #   ソート済みの元データを並べ替える
            ranks = np.argsort(np.argsort(surrogate_with_spectrum))
            next_surrogate = sorted_original[ranks]

            # 3. 収束判定
            if np.allclose(current_surrogate, next_surrogate):
                break  # 前回の反復結果とほぼ同じなら収束したとみなし終了

            current_surrogate = next_surrogate
        
        surrogates_list.append(current_surrogate)

    # --- 結果をNumPy配列に変換して返す ---
    result = np.array(surrogates_list)
    
    # サロゲートが1つの場合は、1次元配列として返す
    if num_surrogates == 1:
        return result.flatten()
    
    return result

# サイン波の生成
Sin_Wave = generate_sin_wave(length=1000, num_cycles=10)
# RSサロゲートデータの生成
RS_Surrogate = Random_Shuffle_Surrogate(Sin_Wave)
# FTサロゲートデータの生成
FT_Surrogate = Fourier_Transform_Surrogate(Sin_Wave)
# IAAFTサロゲートデータの生成
IAAFT_Surrogate = generate_iaaft_surrogates(Sin_Wave, num_surrogates=1)

# ローレンツ方程式のデータ生成
Lorenz_Data = generate_lorenz_data(length=5000, dt=0.01)
# RSサロゲートデータの生成
RS_Lorenz_Surrogate = Random_Shuffle_Surrogate(Lorenz_Data)
# FTサロゲートデータの生成
FT_Lorenz_Surrogate = Fourier_Transform_Surrogate(Lorenz_Data)
# IAAFTサロゲートデータの生成
IAAFT_Lorenz_Surrogate = generate_iaaft_surrogates(Lorenz_Data, num_surrogates=1)

fig = plt.figure(figsize=(10, 4))
plt.plot(Sin_Wave, label='Sine Wave with Noise')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Sine Wave with Noise')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("Sine_Wave_with_Noise.png")
plt.show()

fig = plt.figure(figsize=(10, 4))
plt.plot(RS_Surrogate, label='RS Surrogate Data', color='orange')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('RS Surrogate Data')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("RS_Surrogate_Data.png")
plt.show()

fig = plt.figure(figsize=(10, 4))
plt.plot(FT_Surrogate, label='FT Surrogate Data', color='green')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('FT Surrogate Data')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("FT_Surrogate_Data.png")
plt.show()

fig = plt.figure(figsize=(10, 4))
plt.plot(IAAFT_Surrogate, label='IAAFT Surrogate Data', color='red')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('IAAFT Surrogate Data')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("IAAFT_Surrogate_Data.png")
plt.show()

fig = plt.figure(figsize=(10, 4))
plt.plot(Lorenz_Data, label='Lorenz Data')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Lorenz Data')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("Lorenz_Data.png")
plt.show()

fig = plt.figure(figsize=(10, 4))
plt.plot(RS_Lorenz_Surrogate, label='RS Surrogate Lorenz Data', color='orange')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('RS Surrogate Lorenz Data')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("RS_Surrogate_Lorenz_Data.png")
plt.show()

fig = plt.figure(figsize=(10, 4))
plt.plot(FT_Lorenz_Surrogate, label='FT Surrogate Lorenz Data', color='green')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('FT Surrogate Lorenz Data')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("FT_Surrogate_Lorenz_Data.png")
plt.show()

fig = plt.figure(figsize=(10, 4))
plt.plot(IAAFT_Lorenz_Surrogate, label='IAAFT Surrogate Lorenz Data', color='red')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('IAAFT Surrogate Lorenz Data')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("IAAFT_Surrogate_Lorenz_Data.png")
plt.show()