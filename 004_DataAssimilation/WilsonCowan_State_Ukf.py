"""
wilson_cowan_ukf_state.py
=========================

Wilson–Cowan model × Unscented Kalman Filter (UKF)
-------------------------------------------------
* **用途**: E/I 活動 (2 状態) のみをフィルタリング／予測
* **依存**: numpy ≥ 1.20, matplotlib ≥ 3.5

実行方法
--------
```bash
pip install numpy matplotlib  # 未インストールなら
python wilson_cowan_ukf_state.py          # フィルタ結果をプロット
python wilson_cowan_ukf_state.py --noplot # プロット抑止（数値のみ）
```
"""

from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Wilson–Cowan 補助関数
# -------------------------------

def logistic(x: np.ndarray, a: float, theta: float) -> np.ndarray:
    """Logistic (sigmoid) function"""
    return 1.0 / (1.0 + np.exp(-a * (x - theta)))


def S_E(x: np.ndarray, a_E: float = 1.0, theta_E: float = 5.0) -> np.ndarray:
    return logistic(x, a_E, theta_E)


def S_I(x: np.ndarray, a_I: float = 1.0, theta_I: float = 20.0) -> np.ndarray:
    return logistic(x, a_I, theta_I)


def wc_deriv(state: np.ndarray, params: np.ndarray, inputs: tuple[float, float]) -> np.ndarray:
    """Wilson–Cowan 微分方程式 (dE/dt, dI/dt)"""
    E, I = state
    w_EE, w_EI, w_IE, w_II, tau_E, tau_I, r_E, r_I = params
    P_E, P_I = inputs
    dE = (-E + (1.0 - r_E * E) * S_E(w_EE * E - w_EI * I + P_E)) / tau_E
    dI = (-I + (1.0 - r_I * I) * S_I(w_IE * E - w_II * I + P_I)) / tau_I
    return np.array([dE, dI])


def rk4_step(state: np.ndarray, dt: float, deriv, params: np.ndarray, inputs: tuple[float, float]) -> np.ndarray:
    """4th‑order Runge–Kutta 一歩進める"""
    k1 = deriv(state, params, inputs)
    k2 = deriv(state + 0.5 * dt * k1, params, inputs)
    k3 = deriv(state + 0.5 * dt * k2, params, inputs)
    k4 = deriv(state + dt * k3,       params, inputs)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# -------------------------------
# UKF 基本ルーチン
# -------------------------------

def merwe_sigma_points(x: np.ndarray, P: np.ndarray, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
    n = x.size
    lam = alpha**2 * (n + kappa) - n
    c = n + lam
    Wm = np.full(2*n + 1, 0.5 / c)
    Wc = Wm.copy()
    Wm[0] = lam / c
    Wc[0] = lam / c + (1 - alpha**2 + beta)
    U = np.linalg.cholesky(c * P)
    sigmas = np.zeros((2*n + 1, n))
    sigmas[0] = x
    for k in range(n):
        sigmas[k+1]     = x + U[:, k]
        sigmas[n+k+1]   = x - U[:, k]
    return sigmas, Wm, Wc


def unscented_transform(sigmas: np.ndarray, Wm: np.ndarray, Wc: np.ndarray, noise_cov: np.ndarray | None = None, fx=None):
    if fx is not None:
        sig_f = np.array([fx(s) for s in sigmas])
    else:
        sig_f = sigmas
    x_mean = np.dot(Wm, sig_f)
    P = np.zeros((x_mean.size, x_mean.size))
    for i in range(sig_f.shape[0]):
        y = sig_f[i] - x_mean
        P += Wc[i] * np.outer(y, y)
    if noise_cov is not None:
        P += noise_cov
    return x_mean, P, sig_f


class ManualUKF:
    """最小実装の Unscented Kalman Filter"""
    def __init__(self, dim_x: int, dim_z: int, fx, hx, dt: float, Q: np.ndarray, R: np.ndarray,
                 alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.fx = fx
        self.hx = hx
        self.dt = dt
        self.Q = Q
        self.R = R
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)

    def predict(self, *fx_args):
        sigmas, Wm, Wc = merwe_sigma_points(self.x, self.P, self.alpha, self.beta, self.kappa)
        self.x, self.P, self._sigmas_f = unscented_transform(sigmas, Wm, Wc, self.Q,
                                                             lambda s: self.fx(s, self.dt, *fx_args))
        self._Wm, self._Wc = Wm, Wc

    def update(self, z: np.ndarray):
        sigmas_h = np.array([self.hx(s) for s in self._sigmas_f])
        z_pred = np.dot(self._Wm, sigmas_h)
        P_zz = self.R.copy()
        for i in range(sigmas_h.shape[0]):
            err = sigmas_h[i] - z_pred
            P_zz += self._Wc[i] * np.outer(err, err)
        P_xz = np.zeros((self.dim_x, self.dim_z))
        for i in range(sigmas_h.shape[0]):
            dx = self._sigmas_f[i] - self.x
            dz = sigmas_h[i] - z_pred
            P_xz += self._Wc[i] * np.outer(dx, dz)
        K = P_xz @ np.linalg.inv(P_zz)
        self.x = self.x + K @ (z - z_pred)
        self.P = self.P - K @ P_zz @ K.T

# -------------------------------
# デモ: 状態フィルタリング
# -------------------------------

def ukf_state_forecast(show_plot: bool = True):
    """真軌道と推定軌道を生成し、オプションで描画"""
    # ---------- 真モデル設定 ----------
    dt = 0.01
    steps = 3000
    t_grid = np.linspace(0.0, dt*steps, steps + 1)
    params = np.array([16.0, 26.0, 20.0, 1.0, 0.020, 0.010, 0.0, 0.0])  # 8 パラ
    P_E_const, P_I_const = 2.0, 0.5
    external_input = lambda t: (P_E_const, P_I_const)

    true_state = np.zeros((steps + 1, 2))
    true_state[0] = [0.5, 0.4]
    for k in range(steps):
        P_E, P_I = external_input(t_grid[k])
        true_state[k+1] = rk4_step(true_state[k], dt, wc_deriv, params, (P_E, P_I))

    # ---------- 観測 (E のみ + ノイズ) ----------
    rng = np.random.default_rng(0)
    obs_std = 0.05
    z_obs = true_state[:, 0] + rng.normal(0.0, obs_std, size=steps + 1)

    # ---------- UKF セットアップ ----------
    Q = np.diag([1e-5, 1e-5])           # 状態雑音
    R = np.array([[obs_std**2]])        # 観測雑音
    fx = lambda x, dt_, inp: rk4_step(x, dt_, wc_deriv, params, inp)
    hx = lambda x: np.array([x[0]])     # E のみ観測

    ukf = ManualUKF(dim_x=2, dim_z=1, fx=fx, hx=hx, dt=dt, Q=Q, R=R)
    ukf.x = np.array([0.4, 0.3])        # 初期推定
    ukf.P = np.eye(2) * 0.1

    est_state = np.zeros_like(true_state)
    est_state[0] = ukf.x

    for k in range(steps):
        P_E, P_I = external_input(t_grid[k])
        ukf.predict((P_E, P_I))           # 予測
        ukf.update(np.array([z_obs[k+1]]))  # 更新
        est_state[k+1] = ukf.x

    # ---------- 可視化 ----------
    if show_plot:
        plt.figure("E activity")
        plt.plot(t_grid, true_state[:, 0], label="True E")
        plt.plot(t_grid, est_state[:, 0], "--", label="UKF E")
        plt.xlabel("Time [s]")
        plt.ylabel("E activity")
        plt.legend()

        plt.figure("I activity")
        plt.plot(t_grid, true_state[:, 1], label="True I")
        plt.plot(t_grid, est_state[:, 1], "--", label="UKF I")
        plt.xlabel("Time [s]")
        plt.ylabel("I activity")
        plt.legend()
        plt.show()

    return t_grid, true_state, est_state

# -------------------------------
# エントリーポイント
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Wilson–Cowan UKF (state only)")
    ap.add_argument("--noplot", action="store_true", help="プロットを表示しない")
    args = ap.parse_args()
    ukf_state_forecast(show_plot=not args.noplot)


if __name__ == "__main__":
    main()
