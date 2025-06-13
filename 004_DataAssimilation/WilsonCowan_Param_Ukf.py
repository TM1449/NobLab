
"""
wilson_cowan_Param_ukf_.py
========================

Wilson_Cowan_model_Unscented Kalman Filter (UKF)
-------------------------------------------------
* **目的**: E/I 活動 **+** 脱分極時定数 (tau_E) を同時推定（他パラ固定）
* **依存**: numpy ≥ 1.20, matplotlib ≥ 3.5

実行例
------
bash
pip install numpy matplotlib          # まだなら
python wilson_cowan_ukf_tauE.py       # 推定推移をプロット
python wilson_cowan_ukf_tauE.py --noplot  # 数値計算のみ
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


# Wilson–Cowan 微分方程式
# ------------------------------------------------------------

def wc_deriv(state: np.ndarray, params: np.ndarray, inputs: tuple[float, float]) -> np.ndarray:
    """d[E, I]/dt を返す"""
    E, I = state
    w_EE, w_EI, w_IE, w_II, tau_E, tau_I, r_E, r_I = params
    P_E, P_I = inputs
    dE = (-E + (1.0 - r_E * E) * S_E(w_EE * E - w_EI * I + P_E)) / tau_E
    dI = (-I + (1.0 - r_I * I) * S_I(w_IE * E - w_II * I + P_I)) / tau_I
    return np.array([dE, dI])


def rk4_step(state: np.ndarray, dt: float, deriv, params: np.ndarray, inputs: tuple[float, float]) -> np.ndarray:
    k1 = deriv(state, params, inputs)
    k2 = deriv(state + 0.5 * dt * k1, params, inputs)
    k3 = deriv(state + 0.5 * dt * k2, params, inputs)
    k4 = deriv(state + dt * k3,       params, inputs)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# -------------------------------
# UKF 低レベル関数
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

# -------------------------------
# UKF 本体
# -------------------------------

class ManualUKF:
    """最小実装の Unscented Kalman Filter (Merwe σ 点)"""
    def __init__(self, dim_x: int, dim_z: int, fx, hx, dt: float, Q: np.ndarray, R: np.ndarray, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        self.dim_x, self.dim_z = dim_x, dim_z
        self.fx, self.hx = fx, hx
        self.dt = dt
        self.Q, self.R = Q, R
        self.alpha, self.beta, self.kappa = alpha, beta, kappa
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)

    def predict(self, *fx_args):
        sigmas, Wm, Wc = merwe_sigma_points(self.x, self.P,
                                            self.alpha, self.beta, self.kappa)
        self.x, self.P, self._sigmas_f = unscented_transform(sigmas, Wm, Wc, self.Q,
                                                             lambda s: self.fx(s, self.dt, *fx_args))
        # τ_Eは正値制約を掛けておく（数値安定）
        if self.x[2] < 1e-4:
            self.x[2] = 1e-4
        self._Wm, self._Wc = Wm, Wc

    def update(self, z: np.ndarray):
        sigmas_h = np.array([self.hx(s) for s in self._sigmas_f])
        z_pred = np.dot(self._Wm, sigmas_h)
        P_zz = self.R.copy()
        for i in range(sigmas_h.shape[0]):
            dz = sigmas_h[i] - z_pred
            P_zz += self._Wc[i] * np.outer(dz, dz)
        P_xz = np.zeros((self.dim_x, self.dim_z))
        for i in range(sigmas_h.shape[0]):
            dx = self._sigmas_f[i] - self.x
            dz = sigmas_h[i] - z_pred
            P_xz += self._Wc[i] * np.outer(dx, dz)
        K = P_xz @ np.linalg.inv(P_zz)
        self.x = self.x + K @ (z - z_pred)
        self.P = self.P - K @ P_zz @ K.T

# -------------------------------
# デモ: 状態 + τ_E を同時推定
# -------------------------------

def ukf_tauE_estimation(show_plot: bool = True):
    """真の τ_E (定数) と UKF 推定 (E, I, τ_E) を比較"""
    dt = 0.01
    steps = 3000
    t_grid = np.linspace(0.0, dt*steps, steps + 1)

    # ---- Wilson–Cowan 固定パラメータ ----
    w_EE, w_EI, w_IE, w_II = 16.0, 26.0, 20.0, 1.0
    tau_I_true = 0.010
    r_E, r_I = 0.0, 0.0
    true_tau_E = 0.020   # ← 真の τ_E

    base_params = np.array([w_EE, w_EI, w_IE, w_II,
                            true_tau_E, tau_I_true,
                            r_E, r_I])

    P_E_const, P_I_const = 2.0, 0.5
    external_input = lambda t: (P_E_const, P_I_const)

    # ---- 真の状態軌道生成 ----
    true_state = np.zeros((steps + 1, 2))
    true_state[0] = [0.5, 0.4]
    for k in range(steps):
        P_E, P_I = external_input(t_grid[k])
        true_state[k+1] = rk4_step(true_state[k], dt, wc_deriv, base_params, (P_E, P_I))

    # ---- 観測 (E のみ) ----
    rng = np.random.default_rng(0)
    obs_std = 0.05
    z_obs = true_state[:, 0] + rng.normal(0.0, obs_std, size=steps + 1)

    # ---- UKF セットアップ ----
    # 拡張状態ベクトル: [E, I, τ_E]
    dim_x, dim_z = 3, 1
    process_var_state  = 1e-5
    process_var_param  = 1e-6   # τ_E ランダムウォーク分散（小さめ）
    Q = np.diag([process_var_state, process_var_state, process_var_param])
    R = np.array([[obs_std**2]])

    def fx(x, dt_, inp):
        E, I, tau_E = x
        # τ_E は x[2]
        params_dyn = np.array([w_EE, w_EI, w_IE, w_II,
                               tau_E, tau_I_true,
                               r_E, r_I])
        dE, dI = wc_deriv(np.array([E, I]), params_dyn, inp)
        # τ_E はランダムウォーク → 微分ゼロ
        return np.array([E, I]) + np.array([dE, dI]) * dt_, I*0 + tau_E  # placeholder

    def fx_full(x, dt_, inp):
        E, I, tau_E = x
        params_dyn = np.array([w_EE, w_EI, w_IE, w_II,
                               tau_E, tau_I_true,
                               r_E, r_I])
        new_state = rk4_step(np.array([E, I]), dt_, wc_deriv, params_dyn, inp)
        # τ_E: random walk (no change)
        return np.hstack([new_state, tau_E])

    hx = lambda x: np.array([x[0]])

    ukf = ManualUKF(dim_x, dim_z, fx_full, hx, dt, Q, R)
    ukf.x = np.array([0.4, 0.3, 0.030])   # 初期値: τ_E 仮に 0.03 s
    ukf.P = np.diag([0.01, 0.01, 0.005])

    est_state = np.zeros((steps + 1, 3))
    est_state[0] = ukf.x

    for k in range(steps):
        P_E, P_I = external_input(t_grid[k])
        ukf.predict((P_E, P_I))
        ukf.update(np.array([z_obs[k+1]]))
        est_state[k+1] = ukf.x

    if show_plot:
        plt.figure("E")
