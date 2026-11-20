"""Data-driven models, baselines and the GP emulator of the physics ODE.

  DataDriven     : GP (Matern 2.5 + White) on sensor/operating features -> VB (pure data-driven).
  Hybrid         : physics VB + GP residual on features (+ physics VB as input).
  ConstantMean   : training-fold mean of VB (baseline).
  TimePowerLaw   : VB = a * t^q, OLS on log VB, training folds only (baseline; no qn term).
  LogLinearBaseline: VB = a * qn^p * t^q (round-2 comparator, kept for reference only; the qn term is unstable
                   across folds and is no longer used as a comparator).
  ODEEmulator    : per-case GP emulating log(VB + EPS) over the 5-D physics box of its CV fold.
                   Predictions are returned in log space; the median VB curve is exp(ml) - EPS.
                   The mean of the lognormal is not used anywhere (round-2 crossing bias came from it).
"""
import time

import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import r2_score

from physics_model import simulate

EPS = 1e-3       # offset for the log-transform of the emulated VB
VB_CAP = 3.0     # mm, emulator training-output cap (the default, primary emulator). The VB_CAP=None ablation
                 # is built with the same code and reported in metrics.surrogate.cap_ablation.


def fit_gp(X, Y, seed=0, restarts=2):
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=2.5)
              + WhiteKernel(1e-3, (1e-9, 1.0)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=restarts, random_state=seed)
    gp.fit(X, Y)
    return gp


class DataDriven:
    def fit(self, X, y, seed=0):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-12
        self.gp = fit_gp((X - self.mu) / self.sd, y, seed=seed)
        return self

    def predict(self, X):
        return self.gp.predict((X - self.mu) / self.sd, return_std=True)


class Hybrid:
    """VB = VB_phys + GP residual(features, VB_phys)."""

    def fit(self, X, y, vb_phys, seed=0):
        Z = np.column_stack([X, vb_phys])
        self.mu = Z.mean(axis=0)
        self.sd = Z.std(axis=0) + 1e-12
        self.gp = fit_gp((Z - self.mu) / self.sd, y - vb_phys, seed=seed)
        return self

    def predict(self, X, vb_phys):
        Z = np.column_stack([X, vb_phys])
        r = self.gp.predict((Z - self.mu) / self.sd)
        return vb_phys + r


class ConstantMean:
    """Predicts the training-fold mean VB for every row."""

    def fit(self, vb):
        self.value = float(np.mean(vb))
        return self

    def predict(self, n):
        return np.full(int(n), self.value)


class TimePowerLaw:
    """Time-only power law VB = a * t^q, OLS on log VB over training rows with VB > 0 and t > 0."""

    def fit(self, t, vb):
        m = (vb > 0) & (t > 0)
        X = np.column_stack([np.ones(int(m.sum())), np.log(t[m])])
        self.beta, *_ = np.linalg.lstsq(X, np.log(vb[m]), rcond=None)
        self.n_fit = int(m.sum())
        return self

    def predict(self, t):
        t = np.maximum(np.asarray(t, float), 1e-9)
        return np.exp(self.beta[0]) * t ** self.beta[1]

    @property
    def params(self):
        return dict(a=float(np.exp(self.beta[0])), q=float(self.beta[1]))


class LogLinearBaseline:
    """Round-2 comparator VB = a * qn**p * t**q (OLS on log VB). Reference only: qn exponent unstable across folds."""

    def fit(self, qn, t, vb):
        m = (vb > 0) & (t > 0)
        X = np.column_stack([np.ones(int(m.sum())), np.log(qn[m]), np.log(t[m])])
        self.beta, *_ = np.linalg.lstsq(X, np.log(vb[m]), rcond=None)
        self.n_fit = int(m.sum())
        return self

    def predict(self, qn, t):
        t = np.maximum(np.asarray(t, float), 1e-9)
        return np.exp(self.beta[0] + self.beta[1] * np.log(qn) + self.beta[2] * np.log(t))

    @property
    def params(self):
        return dict(a=float(np.exp(self.beta[0])), p=float(self.beta[1]), q=float(self.beta[2]))


class ODEEmulator:
    """GP emulator of log(VB(t) + EPS) for one case as a function of theta (5-D physical parameters).

    lo, hi : the design box of the fold this emulator belongs to (training-fold derived).
    vb_cap : None (no cap) or a float; applied to the ODE outputs used to train the GP only.
    """

    def __init__(self, case, gamma, grid, lo, hi, vb_cap=VB_CAP):
        self.case = case
        self.gamma = float(gamma)
        self.grid = np.asarray(grid, float)
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        self.vb_cap = vb_cap

    def predict(self, theta):
        """Log-space mean ml and sd sl at every grid time, for physical theta of shape (N, 5)."""
        U = (np.atleast_2d(theta) - self.lo) / (self.hi - self.lo)
        return self.gp.predict(U, return_std=True)

    def median_vb(self, theta):
        """Median VB curve exp(ml) - EPS (clipped at 0) and the log-space sd."""
        ml, sl = self.predict(theta)
        return np.maximum(np.exp(ml) - EPS, 0.0), sl

    def build(self, n_train=400, n_test=100, seed=0):
        U = qmc.LatinHypercube(d=5, seed=seed).random(n_train + n_test)
        TH = self.lo + U * (self.hi - self.lo)
        t0 = time.perf_counter()
        Y = np.array([simulate(th, self.gamma, [self.case], [self.grid])[0] for th in TH])
        t_ode = (time.perf_counter() - t0) / len(TH)
        ok = np.all(np.isfinite(Y), axis=1)
        cap_fraction = float(np.mean(Y[ok] > VB_CAP)) if self.vb_cap is not None else 0.0
        U, Y = U[ok], Y[ok]
        if self.vb_cap is not None:
            Y = np.minimum(Y, self.vb_cap)
        ntr = int(min(n_train, len(U) - 1))
        self.gp = fit_gp(U[:ntr], np.log(Y[:ntr] + EPS), seed=seed)
        Ut, Yt = U[ntr:], Y[ntr:]
        ml, sl = self.gp.predict(Ut, return_std=True)
        mu = np.maximum(np.exp(ml) - EPS, 0.0)
        reps = 20
        t1 = time.perf_counter()
        for _ in range(reps):
            self.gp.predict(Ut)
        t_gp = (time.perf_counter() - t1) / (reps * len(Ut))
        self.test = (Ut, Yt, mu, np.exp(ml) * sl)     # sd column stored in VB units (delta method), for plots
        self.metrics = dict(
            n_train=int(ntr), n_test=int(len(Ut)), n_outputs=int(Y.shape[1]),
            r2=float(r2_score(Yt.ravel(), mu.ravel())),
            rmse=float(np.sqrt(np.mean((Yt - mu) ** 2))),
            coverage_2sigma=float(np.mean(np.abs(np.log(Yt + EPS) - ml) <= 2.0 * sl)),
            t_ode_per_eval_s=float(t_ode), t_gp_per_eval_s=float(t_gp),
            speedup_x=float(t_ode / t_gp),
            rmse_near_0p6=(float(np.sqrt(np.mean(((Yt - mu) ** 2)[Yt <= 0.6])))
                           if np.any(Yt <= 0.6) else float("nan")),
            cap_fraction=cap_fraction,
            cap_mm=(float(self.vb_cap) if self.vb_cap is not None else None),
            box_lo=self.lo.tolist(), box_hi=self.hi.tolist(),
        )
        return self


def build_emulator(args):
    """Module-level wrapper so that emulator builds can run in a process pool."""
    case, gamma, grid, lo, hi, cap, seed = args
    return ODEEmulator(case, gamma, grid, lo, hi, vb_cap=cap).build(n_train=400, n_test=100, seed=seed)
