"""GP surrogate of the LPTN (emulator over 6 uncertain parameters + operating scale d), its validation,
and the data-driven / hybrid (physics + residual) models."""
import time
import numpy as np
import pandas as pd
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as CK, Matern, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import physics_model as pm
from data_loader import DT

K_OUT = 120                                   # output time grid points per trajectory
GP_LO = np.r_[-0.5 * np.ones(6), 0.5]         # 6 log-multipliers + derating scale d
GP_HI = np.r_[0.5 * np.ones(6), 1.0]
GP_DIM = 7


def to_unit(X):
    return (np.asarray(X, float) - GP_LO) / (GP_HI - GP_LO)


def lhs_design(n, seed):
    s = qmc.LatinHypercube(d=GP_DIM, seed=seed).random(n)
    return GP_LO + s * (GP_HI - GP_LO)


def output_index(L):
    return np.linspace(0, L - 1, K_OUT).astype(int)


def reference_outputs(P_nom, prof, x0, Xd, idx_out):
    """Physics-model trajectories (node temps at K_OUT times) for design points Xd (N,7)."""
    P = pm.theta_to_params(P_nom, Xd[:, :6])
    L = len(prof["i2"])
    dsched = np.repeat(Xd[:, 6][:, None], L, axis=1)
    X = pm.simulate(P, prof, x0, dsched)
    return X[:, idx_out, :].reshape(len(Xd), -1)     # columns ordered (K, node)


class PCAGP:
    """GP emulator for multi-output trajectories: standardise outputs, PCA to m scores,
    one ARD Matern+White GP per score (sklearn's shared-kernel multi-output fit is too slow
    for 480 outputs). Predictive variance = sum_k sd_k^2 V_k^2 (independent score GPs), plus an
    optional per-output EMULATOR-VARIANCE INFLATION (PCA truncation + GP error, calibrated on a
    separate design). The inflation is an empirical variance scale of the emulator's own error
    against the LPTN ODE; it is not a model-discrepancy term."""

    def __init__(self, n_comp=12, seed=0):
        self.n_comp = n_comp
        self.seed = seed
        self.inflation_var_ = None
        self.inflation_scale_ = 1.0

    def fit(self, X, Y):
        self.mu_ = Y.mean(axis=0)
        self.sc_ = Y.std(axis=0)
        self.sc_[self.sc_ < 1e-12] = 1.0
        Z = (Y - self.mu_) / self.sc_
        _, s, Vt = np.linalg.svd(Z, full_matrices=False)
        self.V_ = Vt[:self.n_comp]                       # (m, out)
        self.var_ratio_ = float((s[:self.n_comp] ** 2).sum() / (s ** 2).sum())
        S = Z @ self.V_.T                                # (n, m)
        self.gps_ = []
        for k in range(self.n_comp):
            kern = CK(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(X.shape[1]),
                                                 length_scale_bounds=(1e-2, 1e2), nu=2.5) \
                + WhiteKernel(1e-5, (1e-10, 1e-1))
            g = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=1,
                                         random_state=self.seed + k)
            g.fit(X, S[:, k])
            self.gps_.append(g)
        self.kernels_ = [str(g.kernel_) for g in self.gps_]
        return self

    def predict(self, X, return_std=False):
        X = np.atleast_2d(X)
        mus, sds = [], []
        for g in self.gps_:
            m, s = g.predict(X, return_std=True)
            mus.append(m)
            sds.append(s)
        M = np.column_stack(mus)                          # (n, m) score means
        Sd = np.column_stack(sds)
        mean = (M @ self.V_) * self.sc_ + self.mu_
        if not return_std:
            return mean
        var = ((Sd ** 2) @ (self.V_ ** 2)) * (self.sc_ ** 2)
        if self.inflation_var_ is not None:
            var = var + self.inflation_var_[None, :]
        return mean, np.sqrt(np.maximum(var, 0.0))

    def calibrate_emulator_inflation(self, Xc, Yc):
        """Per-output inflation fitted on a SEPARATE design (unit-scaled Xc):
        infl_k = max(mean_i(res_ik^2) - mean_i(gp_var_ik), 0). Also stores the scale factor
        sqrt(mean(var_gp + infl)) / sqrt(mean(var_gp)) on that design."""
        self.inflation_var_ = None
        mu, sd = self.predict(Xc, return_std=True)            # GP-only sd (inflation not set yet)
        var_gp = np.mean(sd ** 2, axis=0)
        self.inflation_var_ = np.maximum(np.mean((Yc - mu) ** 2, axis=0) - var_gp, 0.0)
        self.inflation_scale_ = float(np.sqrt(np.mean(var_gp + self.inflation_var_)) / np.sqrt(np.mean(var_gp)))
        return self


def fit_gp(Xd, Y, seed=0):
    gp = PCAGP(n_comp=12, seed=seed)
    gp.fit(to_unit(Xd), Y)
    return gp


def validate(gp, P_nom, prof, x0, idx_out, n_test=100, seed=1):
    """Held-out LHS test set: R2 / RMSE / coverage of central intervals (50/80/95 and 2-sigma)
    on trajectory values + speed-up."""
    Xt = lhs_design(n_test, seed)
    Yt = reference_outputs(P_nom, prof, x0, Xt, idx_out)
    mu, sd = gp.predict(to_unit(Xt), return_std=True)
    r2 = r2_score(Yt.ravel(), mu.ravel())
    rmse = float(np.sqrt(np.mean((Yt - mu) ** 2)))
    cov2 = float(np.mean(np.abs(Yt - mu) <= 2 * sd))
    cov_lv = {lv: float(np.mean(np.abs(Yt - mu) <= norm.ppf(0.5 + lv / 2.0) * sd)) for lv in [0.5, 0.8, 0.95]}
    # wall-clock: direct ODE integration (solve_ivp) per sample vs GP prediction per sample
    p_ref = pm.theta_to_params(P_nom, Xt[:5, :6])
    t0 = time.perf_counter()
    for i in range(5):
        pm.simulate_reference(p_ref[i], prof, x0, d=Xt[i, 6])
    t_ode = (time.perf_counter() - t0) / 5
    Xbig = lhs_design(2000, seed + 7)
    t0 = time.perf_counter()
    gp.predict(to_unit(Xbig), return_std=True)
    t_gp = (time.perf_counter() - t0) / len(Xbig)
    return dict(r2=float(r2), rmse=rmse, coverage_2sigma=cov2,
                coverage_levels={str(k): v for k, v in cov_lv.items()},
                mean_sd=float(np.mean(sd)),
                speedup_x=float(t_ode / t_gp), t_ode_s=float(t_ode), t_gp_s=float(t_gp),
                Ytrue=Yt, Ymean=mu)


# ---------------- data-driven and hybrid models -----------------------------------------

def _ewm(x, tau_s):
    a = 1.0 - np.exp(-DT / tau_s)
    return pd.Series(x).ewm(alpha=a, adjust=False).mean().to_numpy()


def data_features(prof):
    """Causal features from measured inputs only (no temperatures)."""
    n = prof["n"]
    cols = [prof["i_d"], prof["i_q"], n, prof["coolant"], prof["ambient"],
            _ewm(prof["i2"], 60.0), _ewm(prof["i2"], 300.0), _ewm(prof["i2"], 1200.0),
            _ewm(np.abs(n), 300.0), _ewm(prof["coolant"], 1200.0), _ewm(prof["ambient"], 1200.0)]
    return np.column_stack(cols)


def physics_states(P_nom, prof):
    """Free-run physics trajectory (L,4) from measured initial state."""
    return pm.simulate(P_nom[None], prof, prof["meas"][0])[0]


def fit_rf(X, y, seed=0, min_leaf=20, max_depth=None):
    m = RandomForestRegressor(n_estimators=200, min_samples_leaf=min_leaf, max_depth=max_depth,
                              max_features=0.5, n_jobs=-1, random_state=seed)
    m.fit(X, y)
    return m
