"""Data loading, cleaning and per-case tables for the NASA mill dataset (p2_nasa).

Documented facts (mill/Readme.pdf):
  - case = tool/experiment id (1-16); run = counter within case; time = cumulative
    experiment duration, restarts for each case (real time axis). Readme Fig. 6 labels
    the axis "machining time [mm]"; we use the CSV values as-is (unit ambiguity noted).
  - material: 1 = cast iron, 2 = steel. DOC and feed are constant within a case.
  - An operating setting = (DOC, feed, material). The 16 cases are 8 settings x 2 insert sets
    (replicates), so replicates of one setting must never be split between train and test.
  - VB is flank wear measured after some runs (NaN otherwise).
  - No force sensor: smcDC (spindle motor DC current) is used as the force proxy.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_CSV = Path(r"C:\Users\mikae\Documents\RWTH\02_WS2627\02_DigitalTwins\04_Project\_data\p2_nasa\mill.csv")
VB_CRIT = 0.45                 # mm, end-of-life threshold on flank wear
Q_REF = 0.75 * 0.5             # reference operating load DOC*feed (mm*mm/rev), normalises qn
FEATURES = ["time", "DOC", "feed", "material", "smcAC", "smcDC",
            "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
MIN_RUNS = 3                   # cases with fewer runs have no usable wear trajectory
TRANSIENT_SMCDC = 1.0          # smcDC below this is a run-in transient: the steady level is ~1.3-1.45 in all
                               # other runs (minimum 1.29), while case 1 runs 1-6 read 0.13-0.91 (and 0.68 in run 6)


def load_raw(path=DATA_CSV):
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "row_id"})
    return df


def clean(raw):
    """Drop degenerate sensor dumps and cases without a wear trajectory; flag the entry transient."""
    log = {}
    degenerate = raw["smcDC"] < 0.1           # row_94: all sensors at ~1e-8 (sensor dropout)
    log["degenerate_rows_dropped"] = raw.loc[degenerate, "row_id"].tolist()
    d = raw.loc[~degenerate].copy()
    d["transient"] = d["smcDC"] < TRANSIENT_SMCDC   # masked from the force proxy u(t) only (VB labels are kept)
    log["transient_rows_masked"] = d.loc[d["transient"], "row_id"].tolist()
    runs = d.groupby("case").size()
    short = runs[runs < MIN_RUNS].index.tolist()
    log["short_cases_dropped"] = [int(c) for c in short]
    d = d[~d["case"].isin(short)].sort_values(["case", "time"]).reset_index(drop=True)
    return d, log


def make_folds(cases, n_folds=5):
    """Case-grouped CV (secondary): round-robin over sorted case ids. Replicates can straddle folds."""
    cases = sorted(cases)
    folds = [[] for _ in range(n_folds)]
    for i, c in enumerate(cases):
        folds[i % n_folds].append(int(c))
    return folds


def make_setting_folds(tables, cases, n_folds=4):
    """Setting-grouped CV (primary): the 8 operating settings (DOC, feed, material) are assigned
    round-robin in sorted order, so both replicates of a setting always fall in the same fold."""
    key = {c: tables[c]["setting"] for c in cases}
    settings = sorted(set(key.values()))
    folds = [[] for _ in range(n_folds)]
    for i, s in enumerate(settings):
        folds[i % n_folds] += [int(c) for c in sorted(cases) if key[c] == s]
    return [sorted(f) for f in folds]


def case_table(d, c):
    s = d[d["case"] == c].sort_values("time")
    VB = s["VB"].to_numpy(float)
    DOC, feed = float(s["DOC"].iloc[0]), float(s["feed"].iloc[0])
    mat = int(s["material"].iloc[0] == 2)     # 1 = steel, 0 = cast iron
    return dict(
        case=int(c),
        t=s["time"].to_numpy(float),              # cumulative time, ascending
        u_raw=s["smcDC"].to_numpy(float),         # force proxy (raw)
        keep=~s["transient"].to_numpy(bool),      # False on the run-in transient rows
        VB=VB,                                    # NaN where not measured
        m=~np.isnan(VB),                          # measured-VB mask
        DOC=DOC,
        feed=feed,
        qn=float(DOC * feed / Q_REF),
        material=mat,
        setting=(DOC, feed, mat),
        tlast=float(s["time"].max()),
        n_runs=int(len(s)),
    )


def rows_xy(d, cases):
    """Feature matrix, measured VB and case ids for measured rows of the given cases (sorted order)."""
    s = d[d["case"].isin(cases) & d["VB"].notna()]
    return s[FEATURES].to_numpy(float), s["VB"].to_numpy(float), s["case"].to_numpy(int)


def scaled_cases(tables, cases, ref):
    """Case dicts with the force proxy normalised by a reference (training-set median).

    u     : full-length normalised proxy (documentation / observation fit only, includes the transient)
    t_u   : times of the non-transient rows; u_fit : normalised proxy on those rows.
            The ODE interpolates u(t) from (t_u, u_fit) only, so the run-in transient is never used.
    """
    out = {}
    for c in cases:
        T = tables[c]
        k = T["keep"]
        out[c] = dict(T, u=T["u_raw"] / ref, t_u=T["t"][k], u_fit=T["u_raw"][k] / ref)
    return out
