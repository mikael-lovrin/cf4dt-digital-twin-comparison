"""GP emulator of the physics ODE (VB at fixed pseudo-time grid) and the hybrid residual model.

GP input u = (log k1, log k2, alpha, log hA, vc); output = VB(t_grid) (mm) on 0..2000 PTU (record length).

Round-3 changes:
* horizon back to the 2000-PTU record (no periodic forcing wrap inside the emulated window);
* prior-predictive support: designs whose simulated VB at the horizon exceeds VB_PRIOR_MAX are rejected
  before fitting. This uses only the prior and the ODE, not the data;
* per-column output standardisation (each grid column has its own mean and sd, with a floor);
* WhiteKernel nugget capped at 1e-6 in standardised units.
"""
import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from physics_model import simulate_batch

U_NAMES = ["log_k1", "log_k2", "alpha", "log_hA", "vc"]
VB_PRIOR_MAX = 1.0                 # mm; prior-predictive support: VB(horizon) <= 1 mm
NUGGET_BOUNDS = (1e-10, 1e-6)      # standardised units (round 2 upper bound was 1e-1)


def design_bounds(center):
    """Prior box around the least-squares centre (log-space for k1, k2, hA)."""
    c = np.asarray(center, float)
    lo = np.array([c[0] - 1.0, c[1] - 2.0, 0.0, c[3] - 1.0, 0.8])
    hi = np.array([c[0] + 1.0, c[1] + 2.0, 2.0, c[3] + 1.0, 1.2])
    return lo, hi


def u_to_theta(U):
    U = np.atleast_2d(U)
    return np.column_stack([np.exp(U[:, 0]), np.exp(U[:, 1]), U[:, 2], np.exp(U[:, 3]), U[:, 4]])


def lhs_design(lo, hi, n, seed):
    s = qmc.LatinHypercube(d=len(lo), seed=seed).random(n)
    return lo + s * (hi - lo)


def run_ode_batch(U, t_eval, F_series, eta, VB0):
    """Surrogate training/test targets: vectorised RK4 (dt=1 PTU) over all samples at once."""
    return simulate_batch(u_to_theta(U), t_eval, F_series, eta, VB0=VB0)


def prior_support_mask(Y):
    """Prior-predictive support (data-independent): finite simulation and VB at the horizon <= VB_PRIOR_MAX."""
    Y = np.asarray(Y, float)
    fin = np.all(np.isfinite(Y), axis=1)
    last = np.where(np.isfinite(Y[:, -1]), Y[:, -1], np.inf)
    return fin & (last <= VB_PRIOR_MAX)


class GPSurrogate:
    """Multi-output GP (shared Matern-2.5 ARD + White kernel), standardised inputs, per-column outputs."""

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, U, Y):
        self.xmu = U.mean(axis=0)
        self.xsd = U.std(axis=0) + 1e-12
        self.ymu = Y.mean(axis=0)                              # per grid column
        sd = Y.std(axis=0)
        self.ysd = np.maximum(sd, 1e-3 * sd.max()) + 1e-12     # floor: column at t=0 has ~zero spread
        Xs = (U - self.xmu) / self.xsd
        Ys = (Y - self.ymu) / self.ysd
        d = U.shape[1]
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=2.5)
                  + WhiteKernel(1e-7, NUGGET_BOUNDS))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, random_state=self.seed)
        self.gp.fit(Xs, Ys)
        self.noise_level = float(self.gp.kernel_.k2.noise_level)
        self.noise_at_bound = bool(self.noise_level >= 0.999 * NUGGET_BOUNDS[1])
        return self

    def predict(self, U):
        """Returns (mean, std) in VB units, shape (n, G)."""
        Xs = (np.atleast_2d(U) - self.xmu) / self.xsd
        m, s = self.gp.predict(Xs, return_std=True)
        return m * self.ysd + self.ymu, s * self.ysd
