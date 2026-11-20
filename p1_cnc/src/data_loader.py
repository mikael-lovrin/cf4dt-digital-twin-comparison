"""Data loading, inspection and pseudo-time reconstruction for p1_cnc.

Assumptions (also written to results/summary.md):
* The CSV has no tool/run ID, no timestamp and no unit ID. All rows are treated as
  snapshots of ONE pseudo-trajectory.
* Pseudo-time is reconstructed by sorting on VB_mm (ascending). VB is therefore
  monotone in pseudo-time BY CONSTRUCTION. Unit: PTU (1 PTU = one sorted sample index).
* F_res is NOT a measured resultant force Fx,Fy,Fz. It is a documented combination of
  the feature columns CF_Feature_1..3: F_res = sqrt(CF_1^2 + CF_2^2 + CF_3^2).
* Blocked split: the pseudo-trajectory is cut into 20 contiguous segments of 100 PTU.
  Segments with index % 5 == 4 (4 of 20 = 20 %) are held out.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import kstest

CF_COLS = [f"CF_Feature_{i}" for i in range(1, 6)]
VIB_COLS = [f"Vib_Feature_{i}" for i in range(1, 6)]
AE_COLS = [f"AE_Feature_{i}" for i in range(1, 6)]
RAW_FEATURES = CF_COLS + VIB_COLS + AE_COLS
TARGET = "VB_mm"
LABEL = "Wear_Class"
SEG_LEN = 100
HOLDOUT_MOD = 5  # segment % 5 == 4 -> held out


def default_csv_path(project_dir):
    return os.path.normpath(os.path.join(project_dir, "..", "_data", "p1_cnc", "tool_wear_dataset.csv"))


def load_raw(path):
    return pd.read_csv(path)


def inspect_dataset(df):
    info = {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "n_missing": int(df.isna().sum().sum()),
        "id_or_time_like_columns": [c for c in df.columns
                                    if any(k in c.lower() for k in ("id", "time", "date", "run", "stamp"))],
        "label_counts": {str(k): int(v) for k, v in df[LABEL].value_counts().items()},
        "VB_min": float(df[TARGET].min()),
        "VB_max": float(df[TARGET].max()),
        "VB_mean": float(df[TARGET].mean()),
        "VB_std": float(df[TARGET].std()),
        "n_negative_VB": int((df[TARGET] < 0).sum()),
        "label_VB_ranges": {str(k): [float(g.min()), float(g.max())]
                            for k, g in df.groupby(LABEL)[TARGET]},
        "feature_ranges": {c: [float(df[c].min()), float(df[c].max())] for c in RAW_FEATURES},
        "duplicate_rows": int(df.duplicated().sum()),
    }
    return info


def feature_uniformity(df):
    """KS p-value of each raw feature against a uniform distribution on its observed range."""
    out = {}
    for c in RAW_FEATURES:
        x = df[c].to_numpy(float)
        out[c] = float(kstest(x, "uniform", args=(x.min(), x.max() - x.min())).pvalue)
    return out


def reconstruct_pseudo_trajectory(df):
    d = df.sort_values(TARGET, kind="mergesort").reset_index(drop=True)
    n = len(d)
    idx = np.arange(n)
    d["pseudo_t"] = idx.astype(float)
    d["segment"] = (idx // SEG_LEN).astype(int)
    d["holdout"] = (d["segment"] % HOLDOUT_MOD) == (HOLDOUT_MOD - 1)
    F = np.sqrt(d["CF_Feature_1"] ** 2 + d["CF_Feature_2"] ** 2 + d["CF_Feature_3"] ** 2).to_numpy(float)
    d["F_res"] = F
    X = d[RAW_FEATURES + ["F_res"]].to_numpy(float)
    return {
        "df": d,
        "n": n,
        "t": d["pseudo_t"].to_numpy(float),
        "vb": d[TARGET].to_numpy(float),
        "F": F,
        "X": X,
        "feature_names": RAW_FEATURES + ["F_res"],
        "holdout": d["holdout"].to_numpy(bool),
        "segment": d["segment"].to_numpy(int),
        "label": d[LABEL].to_numpy(),
    }
