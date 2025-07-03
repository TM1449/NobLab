import numpy as np
import matplotlib.pyplot as plt
import EntropyHub as EH

# =================================================================
# 1. MFE解析をカプセル化する関数
# =================================================================
def perform_mfe_analysis(time_series, m, r_ratio, n, scales):
    """
    指定された時系列データに対してマルチスケールファジーエントロピー解析を実行する。
    """
    print(f"データ長: {len(time_series)} の解析を開始します...")
    
    # ファジーエントロピー用の設定オブジェクトを作成
    r_val = r_ratio * np.std(time_series, ddof=1)
    Mobj = EH.MSobject(EnType='FuzzEn', m=m, r=(r_val, n))
    
    # MSEn関数でMFEを計算
    mfe, ci = EH.MSEn(time_series, Mobj, Scales=scales, Plotx=False)
    
    print("解析完了。")
    return mfe, ci

# =================================================================
# 2. パラメータ設定
# =================================================================
# 信号パラメータ
fs = 200        # サンプリング周波数 (Hz)
duration = 60   # 秒数 (s)
N = fs * duration # ステップ数

# MFEパラメータ
m = 2           # 埋め込み次元
r_ratio = 0.2   # 許容範囲rを標準偏差の20%とする
n = 2           # ファジーべき指数
scales = 70     # 計算する最大スケール

# =================================================================
# 3. 時系列データとRSサロゲートデータの準備
# =================================================================
np.random.seed(42) # 結果の再現性のため乱数シードを固定

# --- データ1: 元の脳波データ (と仮定) ---
# ご自身の脳波データに置き換えてください
t = np.arange(N) / fs
original_data = np.sin(2 * np.pi * 35 * t)
""" + np.sin(2 * np.pi * 25 * t) + 0.5 * np.random.randn(N)"""

# --- データ2: RSサロゲートデータを作成 ---
# 元のデータを破壊しないように、必ずコピーを作成してからシャッフルします。
rs_surrogate_data = original_data.copy()
np.random.shuffle(rs_surrogate_data)

# =================================================================
# 4. 両方のデータにMFE解析を実行
# =================================================================
# 元のデータの解析
mfe_original, ci_original = perform_mfe_analysis(original_data, m, r_ratio, n, scales)

# RSサロゲートデータの解析
mfe_surrogate, ci_surrogate = perform_mfe_analysis(rs_surrogate_data, m, r_ratio, n, scales)

# =================================================================
# 5. 結果の表示とグラフ化による比較
# =================================================================
print("\n--- 解析結果 ---")
print(f"元のデータ       - 複雑性指数 (CI): {ci_original:.4f}")
print(f"RSサロゲートデータ - 複雑性指数 (CI): {ci_surrogate:.4f}")

plt.figure(figsize=(12, 7))
plt.plot(range(1, scales + 1), mfe_original, 'o-', color='blue', label=f'Original Data (CI={ci_original:.2f})')
plt.plot(range(1, scales + 1), mfe_surrogate, 's--', color='green', label=f'RS Surrogate Data (CI={ci_surrogate:.2f})')

plt.title('Multiscale Fuzzy Entropy: Original vs. RS Surrogate Data', fontsize=16)
plt.xlabel('Scale Factor', fontsize=12)
plt.ylabel('Fuzzy Entropy', fontsize=12)
plt.xticks(range(1, scales + 1))
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()