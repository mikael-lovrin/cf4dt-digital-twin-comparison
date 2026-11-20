"""Data loading, inspection, decimation and profile-level train/held-out split.

Dataset: Electric Motor Temperature (measures_v2.csv), PMSM test-bench sessions.
Assumptions (also written to results/summary.md):
  * rows of one profile_id are contiguous and chronological (checked: 69 runs for 69 ids);
  * nominal sampling rate 2 Hz (dataset documentation, no timestamp column) -> raw dt = 0.5 s;
    we decimate by 10 -> working step DT = 5 s.
"""
import numpy as np
import pandas as pd

RAW_DT = 0.5          # s, 2 Hz nominal (assumption, no timestamp column in the file)
DECIM = 10            # keep every 10th raw sample
DT = RAW_DT * DECIM   # 5 s working step

NODE_NAMES = ["winding", "tooth", "yoke", "magnet"]  # model node order W, T, Y, M
TEMP_COLS = ["stator_winding", "stator_tooth", "stator_yoke", "pm"]
INPUT_COLS = ["u_q", "u_d", "i_d", "i_q", "motor_speed", "torque", "coolant", "ambient"]

MIN_LEN_TEST = 17000    # raw samples (>= 2.36 h) so that a 75 % stage still has a 30 min horizon
MIN_LEN_TRAIN = 5000
N_TEST = 3
N_TRAIN = 8


def inspect(df):
    """Mechanical data inspection; returns a dict of facts (all computed here)."""
    rep = {}
    rep["n_rows"] = int(len(df))
    rep["n_cols"] = int(df.shape[1])
    rep["columns"] = list(df.columns)
    rep["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    rep["n_nan_total"] = int(df.isna().sum().sum())
    rep["n_duplicate_rows"] = int(df.duplicated().sum())
    rep["ranges"] = {c: [float(df[c].min()), float(df[c].max())] for c in df.columns}
    rep["means"] = {c: float(df[c].mean()) for c in df.columns}
    rep["stds"] = {c: float(df[c].std()) for c in df.columns}
    pid = df["profile_id"].to_numpy()
    same = pid[1:] == pid[:-1]
    rep["n_profiles"] = int(len(np.unique(pid)))
    rep["n_contiguous_runs"] = int((~same).sum() + 1)
    pm_v = df["pm"].to_numpy(float)
    rep["pm_lag1_autocorr"] = float(np.corrcoef(pm_v[:-1][same], pm_v[1:][same])[0, 1])
    rep["pm_abs_step_p99_per_raw_sample"] = float(np.quantile(np.abs(np.diff(pm_v))[same], 0.99))
    # high-frequency noise estimate: MAD of second differences / sqrt(6) (white-noise approximation)
    same2 = pid[2:] == pid[:-2]
    noise = {}
    for c in TEMP_COLS:
        x = df[c].to_numpy(float)
        d2 = (x[2:] - 2 * x[1:-1] + x[:-2])[same2]
        mad = np.median(np.abs(d2 - np.median(d2)))
        noise[c] = float(1.4826 * mad / np.sqrt(6.0))
    rep["hf_noise_std_est_degC"] = noise
    rep["units_check"] = {
        "temps_look_like_degC": bool(0 < df["ambient"].min() and df["pm"].max() < 200),
        "speed_is_physical_rpm_range": bool(df["motor_speed"].abs().max() > 1000),
        "is_standardised_check": bool(abs(df["i_q"].std() - 1.0) < 0.05 and abs(df["pm"].mean()) < 1),
    }
    return rep


def load_and_prepare(path):
    df = pd.read_csv(path)
    rep = inspect(df)
    profiles = {}
    for pid, g in df.groupby("profile_id", sort=True):
        gd = g.iloc[::DECIM]
        p = {
            "pid": int(pid),
            "raw_len": int(len(g)),
            "i_d": gd["i_d"].to_numpy(float),
            "i_q": gd["i_q"].to_numpy(float),
            "n": gd["motor_speed"].to_numpy(float),
            "coolant": gd["coolant"].to_numpy(float),
            "ambient": gd["ambient"].to_numpy(float),
            "torque": gd["torque"].to_numpy(float),
        }
        p["i2"] = p["i_d"] ** 2 + p["i_q"] ** 2
        p["meas"] = np.column_stack([gd["stator_winding"], gd["stator_tooth"],
                                     gd["stator_yoke"], gd["pm"]]).astype(float)
        p["t"] = np.arange(len(gd)) * DT
        profiles[int(pid)] = p
    del df
    rep["profile_raw_len_min_median_max"] = [
        int(min(p["raw_len"] for p in profiles.values())),
        float(np.median([p["raw_len"] for p in profiles.values()])),
        int(max(p["raw_len"] for p in profiles.values())),
    ]
    return rep, profiles


def split_profiles(profiles, seed):
    """Split by profile_id: N_TEST long profiles held out, N_TRAIN others for training."""
    rng = np.random.default_rng(seed)
    ids = sorted(profiles)
    long_ids = [p for p in ids if profiles[p]["raw_len"] >= MIN_LEN_TEST]
    test = sorted(int(x) for x in rng.choice(long_ids, N_TEST, replace=False))
    pool = [p for p in ids if p not in test and profiles[p]["raw_len"] >= MIN_LEN_TRAIN]
    train = sorted(int(x) for x in rng.choice(pool, N_TRAIN, replace=False))
    return train, test
