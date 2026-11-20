"""Data loading, inspection, target-axis choice and condition-level CV folds for p4_rotor.

The raw CSV lives in 04_Project/_data/p4_rotor/features.csv and is READ-ONLY.
Facts established by inspect() (see results/summary.md):
  - 5980 rows, no NaN, 42 numeric features per row (3 axes x 14 features).
  - No time column and no serial number. The experimental condition
    (Misalignment, Unbalance, TypeA, TypeB) is the unit ID used for splits.
  - Each (condition, Position, RPM) cell holds 5 repeated segments; repeats
    always stay in the same fold.
  - "Root Mean Squared" and "Standard Deviation" are numerically identical
    (max abs diff < 0.01), so RMS is used as the target and std is not used.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]              # .../04_Project/p4_rotor
DATA_PATH = ROOT.parent / "_data" / "p4_rotor" / "features.csv"
AXES = ["X", "Y", "Z"]
GRAV = 9.80665                                          # m/s^2 per g
NORMAL_GROUP = "M0.0_U0.0"                              # healthy reference, always in training
META_COLS = ["RPM", "Position", "Misalignment", "Unbalance", "TypeA", "TypeB"]


def rms_col(axis):
    return f"{axis} Root Mean Squared"


def load_raw():
    df = pd.read_csv(DATA_PATH, index_col=0).reset_index(drop=True)   # read-only access
    df["Ueff_g"] = df["Unbalance"] * (df["TypeA"] + df["TypeB"])     # both disks -> 2x mass (in-phase assumption)
    df["group"] = [f"M{m:.1f}_U{u:.1f}" for m, u in zip(df["Misalignment"], df["Unbalance"])]
    df["cond"] = [f"M{m:.1f}_U{u:.1f}_A{int(a)}B{int(b)}"
                  for m, u, a, b in zip(df["Misalignment"], df["Unbalance"], df["TypeA"], df["TypeB"])]
    return df


def inspect(df):
    """Return a dict of data facts (printed and written into summary.md)."""
    cells = df.groupby(["cond", "Position", "RPM"]).size()
    per_cond = df.groupby("cond").size()
    partial = per_cond[per_cond != 165]
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "nan_total": int(df.isna().sum().sum()),
        "rpm_levels": sorted(df["RPM"].unique().tolist()),
        "n_conditions": int(df["cond"].nunique()),
        "n_groups": int(df["group"].nunique()),
        "n_cells": int(len(cells)),
        "repeats_per_cell": {int(k): int(v) for k, v in cells.value_counts().items()},
        "partial_conditions": {k: int(v) for k, v in partial.items()},
        "normal_rows": int(df["Normal"].sum()),
    }


def choose_target_axis(df):
    """Pick the axis whose RMS best tracks the unbalance drive, after removing the healthy baseline.

    Criterion (reported for all axes):
      corr_raw   : Pearson corr of log(RMS) with log(Ueff * RPM^2) on Misalignment=0, Unbalance>0 rows.
                   Shared speed trend with the healthy baseline inflates this.
      corr_excess: same, but RMS is first reduced in quadrature by the healthy baseline mean(RMS | Normal, RPM).
                   This measures the unbalance-induced part only and is the selection criterion.
    """
    normal = df[df["Normal"] == 1]
    sub = df[(df["Misalignment"] == 0) & (df["Unbalance"] > 0)]
    drive = np.log(sub["Ueff_g"] * sub["RPM"] ** 2)
    rows = []
    for ax in AXES:
        c = rms_col(ax)
        base = normal.groupby("RPM")[c].mean()
        excess2 = sub[c] ** 2 - sub["RPM"].map(base) ** 2
        n_clip = int((excess2 <= 0).sum())
        excess = np.sqrt(np.clip(excess2, 1e-10, None))
        rows.append({
            "axis": ax,
            "corr_raw": float(pearsonr(np.log(sub[c]), drive)[0]),
            "corr_excess": float(pearsonr(np.log(excess), drive)[0]),
            "n_clipped": n_clip,
        })
    table = pd.DataFrame(rows)
    chosen = str(table.sort_values("corr_excess", ascending=False)["axis"].iloc[0])
    return chosen, table


def baseline_floor(df, axis):
    """Healthy-machine background RMS per RPM (mean over Normal rows). Returns f(rpm) -> g."""
    base = df[df["Normal"] == 1].groupby("RPM")[rms_col(axis)].mean()
    xs, ys = base.index.values.astype(float), base.values

    def floor(rpm):
        return np.interp(np.asarray(rpm, float), xs, ys)

    return floor, base


def add_disc_columns(df):
    """Round 3: per-disc unbalance (UA_g, UB_g, no in-phase sum) and 0-based Position index."""
    df["UA_g"] = df["Unbalance"] * df["TypeA"]
    df["UB_g"] = df["Unbalance"] * df["TypeB"]
    df["pos"] = df["Position"].astype(int) - 1
    return df


def position_floor(df, axis):
    """Healthy background per Position: table (3, n_rpm) of mean axis RMS over Normal rows, and the RPM levels."""
    nrm = df[df["Normal"] == 1]
    tab = nrm.groupby(["Position", "RPM"])[rms_col(axis)].mean().unstack("RPM").sort_index()
    rpm_lv = np.asarray(tab.columns, float)
    return np.vstack([tab.loc[p].values for p in tab.index]).astype(float), rpm_lv


def floor_at(tab, rpm_lv, pos, rpm):
    """Per-row healthy floor from the per-position table (linear interpolation in RPM)."""
    pos = np.asarray(pos, int)
    rpm = np.asarray(rpm, float)
    out = np.empty(len(rpm))
    for p in np.unique(pos):
        m = pos == p
        out[m] = np.interp(rpm[m], rpm_lv, tab[p])
    return out


def feature_cols(df, target_axis):
    """Vibration features of the two NON-target axes (the target axis's own 14 features are excluded
    because they are the same signal as the target and would leak it)."""
    return [c for c in df.columns if c.split(" ")[0] in AXES and c.split(" ")[0] != target_axis]


def logo_folds(df):
    """Leave-one-(Unbalance, Misalignment)-group-out. The healthy Normal group stays in training."""
    for g in sorted(df["group"].unique()):
        if g == NORMAL_GROUP:
            continue
        te = np.where(df["group"].values == g)[0]
        tr = np.where(df["group"].values != g)[0]
        yield g, tr, te
