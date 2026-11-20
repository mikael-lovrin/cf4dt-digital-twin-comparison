"""Data-driven regressors (round 3).

Round 2 used a GP emulator of the single-DOF physics. Round 3 removes it: the 2-DOF steady-state response is a
closed form that is already faster than any emulator (round-2 measurement: GP speedup 0.0059x), so MCMC and Sobol
call the closed form directly (physics_model.pred2_g). The 'surrogate' metrics in main.py are therefore a
verification of the closed form against an independent real 4x4 solve, not a GP fit.

Only the random-forest regressor used by the Pure-Data and Hybrid models is kept here.
"""
from sklearn.ensemble import RandomForestRegressor


def fit_rf(X, y, seed=0, n_estimators=200):
    m = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=3, max_features=0.5,
                              random_state=seed, n_jobs=-1)
    return m.fit(X, y)
