# -*- coding: utf-8 -*-
"""WC_network.py
========================================
Wilson–Cowan 大規模ネットワークモデル
(ベクトル化 + 4 次ルンゲクッタ + tqdm 進捗 + プロット)

 - Abeysuriya et al., *PLOS Comput. Biol.* 14 (2): e1006007 (2018) を基に実装
 - for 二重ループを排し NumPy 配列演算で高速化
 - 伝導遅延はリングバッファで管理
 - `tqdm` で進捗表示，終了後に平均 E/I をプロット
 - **バグ修正**: 遅延付き結合項 `coupling` の次元不一致を解消
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

###############################################################################
# ユーティリティ関数
###############################################################################

def sigmoid(x: np.ndarray, beta: float = 1.0, theta: float = 0.0) -> np.ndarray:
    """S(x) = 1 / (1 + exp(−β(x − θ)))"""
    return 1.0 / (1.0 + np.exp(-beta * (x - theta)))


def rk4_step(y: np.ndarray, dt: float, rhs):
    """4 次 Runge–Kutta 1 ステップ (ベクトル化)"""
    k1 = rhs(y)
    k2 = rhs(y + 0.5 * dt * k1)
    k3 = rhs(y + 0.5 * dt * k2)
    k4 = rhs(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

###############################################################################
# Wilson–Cowan ネットワーク RHS
###############################################################################

def make_rhs(
    tau_e: float,
    tau_i: float,
    c_ee: float,
    c_ei: float,
    c_ie_init: float,
    C_global: float,
    W: np.ndarray,
    delays_steps: np.ndarray,
    P_ext: float,
    beta: float = 1.5,
    theta: float = 0.0,
    rng=None,
):
    """右辺関数をクロージャとして生成。ノード = N。遅延結合 + 白色ノイズを含む。"""

    N = W.shape[0]

    # --- 遅延用リングバッファ (E の履歴) ---
    delay_max = int(delays_steps.max())
    E_buffer = np.zeros((delay_max + 1, N), dtype=np.float32)
    buf_idx = 0  # 現在の書き込み位置

    # 抑制結合係数 (ここでは固定値だが拡張可)
    c_ie = np.full(N, c_ie_init, dtype=np.float32)

    rng = np.random.default_rng(rng)

    def rhs(state: np.ndarray):
        nonlocal buf_idx, E_buffer

        # state = [E, I] を縦に連結した 1 次元長さ 2N
        E = state[:N]
        I = state[N:]

        # --------------------------------------------------
        # 遅延結合項  C Σ_j W_{jk} E_j(t − τ_{jk})
        # --------------------------------------------------
        #   delays_steps : (N, N)  [j, k]
        #   delayed_idx  : (N, N)  現在の buf_idx から遅延を引いたインデックス
        #   E_delayed_mat[j, k] = E_j(t − τ_{jk})
        # --------------------------------------------------
        delayed_idx = (buf_idx - delays_steps) % (delay_max + 1)  # (N, N)
        src_idx = np.arange(N)[:, None]                           # (N, 1)   j index
        # E_buffer[rows, cols] → (N, N)
        E_delayed_mat = E_buffer[delayed_idx, src_idx]

        # 重み行列と要素積を取り j で和 (axis=0) → ベクトル (N,)
        coupling = C_global * np.sum(W * E_delayed_mat, axis=0)

        # --------------------------------------------------
        # 白色ノイズ ξ_e, ξ_i
        # --------------------------------------------------
        xi_e = rng.normal(scale=0.01, size=N)
        xi_i = rng.normal(scale=0.01, size=N)

        # --------------------------------------------------
        # Wilson–Cowan 方程式
        # --------------------------------------------------
        dE = (-E + sigmoid(c_ee * E - c_ie * I + P_ext + coupling + xi_e,
                           beta=beta, theta=theta)) / tau_e
        dI = (-I + sigmoid(c_ei * E + xi_i,
                           beta=beta, theta=theta)) / tau_i

        # 現在時刻の E をバッファに保存し，インデックス更新
        E_buffer[buf_idx] = E
        buf_idx = (buf_idx + 1) % (delay_max + 1)

        return np.concatenate([dE, dI])

    return rhs

###############################################################################
# メインシミュレーション関数
###############################################################################

def run_simulation(sim_time=300.0, dt=0.01, N=68, save=None, rng_seed=42):
    """Wilson–Cowan ネットワークを RK4 で時間発展させる"""

    # ==== 1. パラメータ (論文 Table 1 相当) ====
    tau_e = 20.0  # [ms]
    tau_i = 10.0  # [ms]

    c_ee = 12.0
    c_ei = 13.0
    c_ie = 11.0

    C_global = 0.3  # 全結合スケール
    P_ext = 0.3     # 外部定常入力

    # ==== 2. 結合行列 W と伝導遅延行列 ====
    rng = np.random.default_rng(rng_seed)
    W = rng.uniform(0.0, 1.0, size=(N, N)).astype(np.float32)
    np.fill_diagonal(W, 0.0)  # 自己結合はゼロ

    distance = rng.uniform(10.0, 100.0, size=(N, N))  # [mm]
    velocity = 6.0                                    # [m/s]
    delays_ms = (distance / 1000.0) / velocity * 1e3  # [ms]
    delays_steps = np.rint(delays_ms / dt).astype(int)

    # ==== 3. 初期状態 ====
    E0 = rng.normal(loc=0.1, scale=0.01, size=N)
    I0 = rng.normal(loc=0.1, scale=0.01, size=N)
    y = np.concatenate([E0, I0]).astype(np.float32)

    # ==== 4.  時間軸 / 結果配列確保 ====
    n_steps = int(sim_time / dt) + 1
    t = np.arange(n_steps) * dt
    E_trace = np.empty((n_steps, N), dtype=np.float32)
    I_trace = np.empty((n_steps, N), dtype=np.float32)
    E_trace[0] = E0
    I_trace[0] = I0

    # ==== 5. RHS 生成 ====
    rhs = make_rhs(
        tau_e, tau_i, c_ee, c_ei, c_ie,
        C_global, W, delays_steps, P_ext,
        rng=rng,
    )

    # ==== 6. 時間積分ループ (tqdm 進捗表示) ====
    start = time.time()
    for step in tqdm(range(1, n_steps), desc="Simulating", unit="step"):
        y = rk4_step(y, dt, rhs)
        E_trace[step] = y[:N]
        I_trace[step] = y[N:]
    elapsed = time.time() - start
    print(f"\nElapsed time: {elapsed:.2f} s  (≈ {elapsed/n_steps*1e3:.2f} ms/step)")

    # ==== 7. 結果保存 ====
    if save is not None:
        save = save.lower()
        fname = Path(f"wc_{N}n_{sim_time:g}ms_{dt:g}dt.{save}")
        if save == "npz":
            np.savez_compressed(fname, E=E_trace, I=I_trace, t=t)
        elif save == "npy":
            np.save(fname.with_suffix(".npy"), {"E": E_trace, "I": I_trace, "t": t})
        else:
            raise ValueError("save must be 'npz' or 'npy'")
        print(f"Saved results to {fname}")

    # ==== 8. プロット (E/I 平均) ====
    plt.figure(figsize=(10, 4))
    plt.plot(t, E_trace.mean(axis=1), label="E (mean)")
    plt.plot(t, I_trace.mean(axis=1), label="I (mean)")
    plt.xlabel("Time [ms]")
    plt.ylabel("Activity")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return t, E_trace, I_trace

###############################################################################
# コマンドラインインターフェース
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vectorised Wilson–Cowan network simulator (RK4)")
    parser.add_argument("--sim-time", type=float, default=1000.0, help="シミュレーション時間 [ms]")
    parser.add_argument("--dt", type=float, default=0.001, help="タイムステップ [ms]")
    parser.add_argument("--nodes", type=int, default=68, help="ノード数 (脳領域数)")
    parser.add_argument("--save", choices=["npz", "npy"], help="結果保存フォーマット")
    args = parser.parse_args()

    run_simulation(sim_time=args.sim_time, dt=args.dt, N=args.nodes, save=args.save)