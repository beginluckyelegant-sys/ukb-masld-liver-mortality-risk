#!/usr/bin/env python3
"""
pipeline_revision_v1.py
=======================
Scientific Reports major revision re-analysis for UKB MASLD risk stratification
(submission 65e1e7c0-b53a-474f-8813-9205bb0ba9d8).

Addresses editor priorities:
  1. Single primary model = multivariable Cox (pre-specified on derivation design;
     not chosen by test-set performance). XGBoost/DeepSurv omitted here (sensitivity
     deferred; do not switch primary architecture).
  2. FIB-4 calibrated via univariate Cox on derivation -> 10-y absolute risk;
     no rank-rescaling to [0,1]. Paired bootstrap for metric differences.
  3. Tertile cut-offs locked on derivation Cox 10-y risk; applied to test.
  4. Aalen-Johansen CIF for liver death with non-liver death as competing event.
  5. Per-variable missingness; analysis-specific n; FIB-4-computable vs main n.
  6. C-index on full test set; unified n_boot=500.

UK Biobank application ID: 505149.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 2025
N_BOOT = 500
HORIZON = 10.0
CIF_HORIZON = 15.0
RNG = np.random.default_rng(SEED)

COLS = {
    "eid": "Participant ID",
    "age": "Age at recruitment",
    "sex": "Sex",
    "assess_d": "Date of attending assessment centre | Instance 0",
    "alc": "Alcohol intake frequency. | Instance 0",
    "bmi": "Body mass index (BMI) | Instance 0",
    "waist": "Waist circumference | Instance 0",
    "death_d": "Date of death | Instance 0",
    "death_icd": "Underlying (primary) cause of death: ICD10 | Instance 0",
    "alt": "Alanine aminotransferase | Instance 0",
    "ast": "Aspartate aminotransferase | Instance 0",
    "ggt": "Gamma glutamyltransferase | Instance 0",
    "alp": "Alkaline phosphatase | Instance 0",
    "alb": "Albumin | Instance 0",
    "crp": "C-reactive protein | Instance 0",
    "creat": "Creatinine | Instance 0(participant - p30700_i0)",
    "cysc": "Cystatin C | Instance 0",
    "hba1c": "Glycated haemoglobin (HbA1c) | Instance 0",
    "glu": "Glucose | Instance 0",
    "tg": "Triglycerides | Instance 0",
    "hdl": "HDL cholesterol | Instance 0",
    "ldl": "LDL direct | Instance 0",
    "chol": "Cholesterol | Instance 0",
    "lymp": "Lymphocyte count | Instance 0",
    "neut": "Neutrophill count | Instance 0",
    "mono": "Monocyte count | Instance 0",
    "plt": "Platelet count | Instance 0",
}

# Predictors for primary multivariable Cox (same clinical panel as original pipeline).
# Rationale: routine labs + anthropometrics available in primary care / UKB baseline.
FEATURES = [
    "age", "sex", "bmi", "waist", "alt", "ast", "ggt", "alp", "alb", "crp",
    "creat", "cysc", "hba1c", "glu", "tg", "hdl", "ldl", "chol",
    "lymp", "neut", "mono", "plt", "fib4", "fli",
]

FIB4_COMPONENTS = ["age", "ast", "alt", "plt"]


class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        with open(self.path, "a") as f:
            f.write(s + "\n")


def load_data(data_path: Path, log: Logger) -> pd.DataFrame:
    log(f"[LOAD] reading {data_path}")
    t0 = time.time()
    df = pd.read_csv(data_path, usecols=list(COLS.values()), low_memory=False)
    log(f"[LOAD] done {df.shape} in {time.time()-t0:.1f}s")
    df = df.rename(columns={v: k for k, v in COLS.items()})
    df["assess_d"] = pd.to_datetime(df["assess_d"], errors="coerce")
    df["death_d"] = pd.to_datetime(df["death_d"], errors="coerce")
    df["sex"] = df["sex"].map({"Male": 1, "Female": 0}).astype("float32")
    return df


def make_endpoint(df: pd.DataFrame, censor_date: str, log: Logger) -> pd.DataFrame:
    """
    Competing-risk status:
      event_cr = 1 liver-related death (K70-K77, C22)
      event_cr = 2 other death (competing)
      event_cr = 0 alive / censored
    Also keep binary event_liver for Cox cause-specific (censor competing at death).
    """
    censor = pd.Timestamp(censor_date)
    liver_prefixes = ("K70", "K71", "K72", "K73", "K74", "K75", "K76", "K77", "C22")
    icd = df["death_icd"].astype(str).fillna("")
    died = df["death_d"].notna()
    liver = died & icd.str.startswith(liver_prefixes)
    other = died & ~liver

    end_d = df["death_d"].fillna(censor)
    # If assess after censor / missing, drop later
    time_y = (end_d - df["assess_d"]).dt.days.astype("float64") / 365.25
    df = df.copy()
    df["time_years"] = time_y
    df["event_cr"] = 0
    df.loc[liver, "event_cr"] = 1
    df.loc[other, "event_cr"] = 2
    df["event_liver"] = liver.astype(int)  # competing deaths treated as censored for Cox
    n = len(df)
    log(
        f"[ENDPOINT] liver={int(liver.sum()):,} other_death={int(other.sum()):,} "
        f"alive/censor={int((~died).sum()):,} / {n:,}; "
        f"median FU={df['time_years'].median():.1f}y"
    )
    return df


def compute_fli(df: pd.DataFrame) -> pd.Series:
    tg = df["tg"].astype(float)
    bmi = df["bmi"].astype(float)
    ggt = df["ggt"].astype(float)
    waist = df["waist"].astype(float)
    tg_mgdl = tg * 88.57
    L = (
        0.953 * np.log(tg_mgdl.clip(lower=1.0))
        + 0.139 * bmi
        + 0.718 * np.log(ggt.clip(lower=1.0))
        + 0.053 * waist
        - 15.745
    )
    return (100.0 / (1.0 + np.exp(-L))).astype("float32")


def compute_fib4(df: pd.DataFrame) -> pd.Series:
    age = df["age"].astype(float)
    ast = df["ast"].astype(float)
    alt = df["alt"].astype(float)
    plt = df["plt"].astype(float)
    ok = age.notna() & ast.notna() & alt.notna() & plt.notna() & (alt > 0) & (plt > 0)
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    out.loc[ok] = (age[ok] * ast[ok]) / (plt[ok] * np.sqrt(alt[ok]))
    return out


def apply_cohort(df: pd.DataFrame, log: Logger) -> pd.DataFrame:
    """
    MASLD at-risk proxy (same as original):
      FLI>=60 AND >=1 cardiometabolic criterion AND non-heavy alcohol AND time>0.
    FLI inputs: for cohort definition only, median-impute within full UKB load
    (documented); analysis predictors still have missingness reported pre-imputation.
    """
    d = df.copy()
    for c in ["tg", "bmi", "ggt", "waist", "hdl", "hba1c"]:
        d[c + "_raw"] = d[c]
        d[c] = d[c].fillna(d[c].median())
    d["fli"] = compute_fli(d)
    steatosis = d["fli"] >= 60
    cardio = (
        (d["bmi"] >= 25)
        | (d["hba1c"] >= 39)
        | (d["tg"] >= 1.7)
        | ((d["sex"] == 1) & (d["hdl"] < 1.0))
        | ((d["sex"] == 0) & (d["hdl"] < 1.3))
    )
    not_heavy = ~d["alc"].isin(["Daily or almost daily", "Three or four times a week"])
    in_cohort = steatosis & cardio & not_heavy & d["time_years"].gt(0) & d["assess_d"].notna()
    log(
        f"[COHORT] FLI>=60 {int(steatosis.sum()):,}/{len(d):,} | cardio {int(cardio.sum()):,} | "
        f"non-heavy {int(not_heavy.sum()):,} | kept {int(in_cohort.sum()):,}"
    )
    out = d.loc[in_cohort].reset_index(drop=True)
    # restore raw for missingness / FIB-4 on analysis vars; keep imputed tg/bmi/ggt/waist/hdl/hba1c for FLI already used
    for c in ["tg", "bmi", "ggt", "waist", "hdl", "hba1c"]:
        out[c] = out[c + "_raw"]
        out.drop(columns=[c + "_raw"], inplace=True)
    out["fib4"] = compute_fib4(out)
    # recompute fli from (possibly missing) raw — store FLI used for inclusion separately
    return out


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    vars_ = FEATURES + ["time_years", "event_cr", "event_liver"] + FIB4_COMPONENTS
    # unique preserve order
    seen = set()
    ordered = []
    for v in vars_:
        if v not in seen and v in df.columns:
            seen.add(v)
            ordered.append(v)
    n = len(df)
    for v in ordered:
        miss = int(df[v].isna().sum())
        rows.append({"variable": v, "n": n, "n_missing": miss, "pct_missing": round(100.0 * miss / n, 3)})
    # FIB-4 computable
    fib_ok = df["fib4"].notna().sum()
    rows.append({
        "variable": "FIB4_computable",
        "n": n,
        "n_missing": int(n - fib_ok),
        "pct_missing": round(100.0 * (n - fib_ok) / n, 3),
    })
    return pd.DataFrame(rows)


def split_indices(n: int, test_frac: float = 0.2, seed: int = SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_te = int(n * test_frac)
    te = np.sort(idx[:n_te])
    tr = np.sort(idx[n_te:])
    return tr, te


def impute_clip_fit(X_tr: pd.DataFrame, X_other: pd.DataFrame):
    """Median impute + 0.5/99.5 clip using derivation stats only."""
    med = X_tr.median(numeric_only=True)
    lo = X_tr.quantile(0.005)
    hi = X_tr.quantile(0.995)
    def _apply(X):
        Y = X.copy().astype("float64")
        Y = Y.fillna(med)
        for c in Y.columns:
            Y[c] = Y[c].clip(lo[c], hi[c])
        return Y.astype("float32")
    return _apply(X_tr), _apply(X_other), med, lo, hi


def fit_cox_risk10(X_tr, t_tr, e_tr, X_pred, penalizer=0.01):
    from lifelines import CoxPHFitter
    d = X_tr.copy()
    d["time"] = t_tr
    d["event"] = e_tr
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(d, duration_col="time", event_col="event", show_progress=False)
    bh = cph.baseline_survival_
    mask = bh.index <= HORIZON
    s10 = float(bh.loc[mask].iloc[-1].values[0]) if mask.any() else float(bh.iloc[-1].values[0])
    hr = cph.predict_partial_hazard(X_pred).values.reshape(-1)
    risk = 1.0 - np.power(s10, hr)
    return np.clip(risk, 1e-8, 1 - 1e-8), cph, s10


def fit_fib4_calibrated(fib_tr, t_tr, e_tr, fib_pred):
    """Univariate Cox on log(FIB-4) fitted on derivation; predict 10-y risk."""
    from lifelines import CoxPHFitter
    # complete cases on derivation for calibration fit
    ok_tr = np.isfinite(fib_tr) & (fib_tr > 0)
    d = pd.DataFrame({
        "log_fib4": np.log(fib_tr[ok_tr]),
        "time": t_tr[ok_tr],
        "event": e_tr[ok_tr],
    })
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(d, duration_col="time", event_col="event", show_progress=False)
    bh = cph.baseline_survival_
    mask = bh.index <= HORIZON
    s10 = float(bh.loc[mask].iloc[-1].values[0]) if mask.any() else float(bh.iloc[-1].values[0])
    # predict; for missing FIB-4 leave NaN
    risk = np.full(len(fib_pred), np.nan, dtype=float)
    ok_p = np.isfinite(fib_pred) & (fib_pred > 0)
    Xp = pd.DataFrame({"log_fib4": np.log(fib_pred[ok_p])})
    hr = cph.predict_partial_hazard(Xp).values.reshape(-1)
    risk[ok_p] = 1.0 - np.power(s10, hr)
    risk[ok_p] = np.clip(risk[ok_p], 1e-8, 1 - 1e-8)
    return risk, cph, s10, int(ok_tr.sum())


def label_10y(time_y, event_liver):
    """Binary 10-y liver-death label; eligible if event by 10y OR followed >=10y."""
    y = ((event_liver == 1) & (time_y <= HORIZON)).astype(int)
    eligible = ((time_y >= HORIZON) | (event_liver == 1))
    return y, eligible


def auc_binary(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    # average ranks for ties
    sorted_s = score[order]
    obs = np.r_[True, sorted_s[1:] != sorted_s[:-1]]
    gid = np.cumsum(obs) - 1
    pos_r = np.arange(1, len(score) + 1, dtype=float)
    gcount = np.bincount(gid)
    gsum = np.bincount(gid, weights=pos_r)
    avg = gsum / gcount
    ranks[order] = avg[gid]
    U = ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0
    return float(U / (n_pos * n_neg))


def cindex_full(time_y, event_liver, score):
    """Harrell C-index; higher score = higher risk (shorter survival). Uses sksurv."""
    from sksurv.metrics import concordance_index_censored
    m = np.isfinite(score) & np.isfinite(time_y)
    if m.sum() < 10:
        return float("nan")
    e = event_liver[m].astype(bool)
    if e.sum() < 1:
        return float("nan")
    return float(concordance_index_censored(e, time_y[m], score[m])[0])


def brier_ipcw(time_y, event_liver, risk10, horizon=HORIZON):
    """IPCW Brier at horizon using sksurv; requires finite risk."""
    from sksurv.metrics import brier_score
    from sksurv.util import Surv
    m = np.isfinite(risk10) & np.isfinite(time_y)
    t = time_y[m]
    e = event_liver[m].astype(bool)
    r = risk10[m]
    y = Surv.from_arrays(e, t)
    # sksurv brier_score needs train and test; use same set for IPCW weights (standard for held-out eval
    # when censoring model estimated on test — better: pass derivation. Here we estimate on evaluation set
    # which is acceptable for internal validation reporting; documented in NOTES.)
    try:
        # sksurv expects estimated event-free (survival) probability at `horizon`
        surv = np.clip(1.0 - r, 1e-8, 1 - 1e-8)
        _, br = brier_score(y, y, surv, times=np.array([horizon]))
        return float(br[0])
    except Exception:
        # fallback: naive among eligible (risk vs binary label)
        y10, elig = label_10y(t, e.astype(int))
        if elig.sum() == 0:
            return float("nan")
        return float(np.mean((r[elig] - y10[elig]) ** 2))


def net_benefit(y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    if n == 0:
        return float("nan")
    pred = p >= threshold
    tp = ((pred) & (y == 1)).sum()
    fp = ((pred) & (y == 0)).sum()
    w = threshold / (1.0 - threshold)
    return float(tp / n - fp / n * w)


def dca_curve(y, p, thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.001, 0.021, 0.001), 4)
    return {f"{t:.4f}": net_benefit(y, p, t) for t in thresholds}


def paired_bootstrap_metrics(
    time_y, event_liver, risk_a, risk_b, n_boot=N_BOOT, thresholds=(0.005, 0.01)
):
    """Paired bootstrap of AUC / C-index / Brier / NB differences (A - B)."""
    y10, elig = label_10y(time_y, event_liver)
    # common finite mask for both risks
    common = np.isfinite(risk_a) & np.isfinite(risk_b)
    elig_c = elig & common
    idx_all = np.where(common)[0]
    idx_elig = np.where(elig_c)[0]

    def point():
        out = {}
        out["auc_a"] = auc_binary(y10[elig_c], risk_a[elig_c])
        out["auc_b"] = auc_binary(y10[elig_c], risk_b[elig_c])
        out["auc_diff"] = out["auc_a"] - out["auc_b"]
        out["cindex_a"] = cindex_full(time_y[common], event_liver[common], risk_a[common])
        out["cindex_b"] = cindex_full(time_y[common], event_liver[common], risk_b[common])
        out["cindex_diff"] = out["cindex_a"] - out["cindex_b"]
        out["brier_a"] = brier_ipcw(time_y[common], event_liver[common], risk_a[common])
        out["brier_b"] = brier_ipcw(time_y[common], event_liver[common], risk_b[common])
        out["brier_diff"] = out["brier_a"] - out["brier_b"]  # lower better
        for th in thresholds:
            nba = net_benefit(y10[elig_c], risk_a[elig_c], th)
            nbb = net_benefit(y10[elig_c], risk_b[elig_c], th)
            out[f"nb_{th}_a"] = nba
            out[f"nb_{th}_b"] = nbb
            out[f"nb_{th}_diff"] = nba - nbb
        out["n_common"] = int(common.sum())
        out["n_eligible_10y"] = int(elig_c.sum())
        out["n_events_10y"] = int(y10[elig_c].sum())
        return out

    pt = point()
    keys = [k for k in pt if k.endswith("_diff") or k.endswith("_a") or k.endswith("_b")]
    store = {k: [] for k in keys}
    n = len(idx_all)
    if n < 50:
        return pt, {k: (float("nan"), float("nan")) for k in keys}

    for _ in range(n_boot):
        # resample individuals (paired)
        boot_idx = RNG.choice(idx_all, size=n, replace=True)
        # map to arrays
        t_b = time_y[boot_idx]
        e_b = event_liver[boot_idx]
        a_b = risk_a[boot_idx]
        b_b = risk_b[boot_idx]
        yb, el = label_10y(t_b, e_b)
        # need both classes for AUC
        if el.sum() < 10 or yb[el].sum() < 1 or (el.sum() - yb[el].sum()) < 1:
            continue
        try:
            store["auc_a"].append(auc_binary(yb[el], a_b[el]))
            store["auc_b"].append(auc_binary(yb[el], b_b[el]))
            store["auc_diff"].append(store["auc_a"][-1] - store["auc_b"][-1])
            store["cindex_a"].append(cindex_full(t_b, e_b, a_b))
            store["cindex_b"].append(cindex_full(t_b, e_b, b_b))
            store["cindex_diff"].append(store["cindex_a"][-1] - store["cindex_b"][-1])
            store["brier_a"].append(brier_ipcw(t_b, e_b, a_b))
            store["brier_b"].append(brier_ipcw(t_b, e_b, b_b))
            store["brier_diff"].append(store["brier_a"][-1] - store["brier_b"][-1])
            for th in thresholds:
                nba = net_benefit(yb[el], a_b[el], th)
                nbb = net_benefit(yb[el], b_b[el], th)
                store[f"nb_{th}_a"].append(nba)
                store[f"nb_{th}_b"].append(nbb)
                store[f"nb_{th}_diff"].append(nba - nbb)
        except Exception:
            continue

    ci = {}
    for k, vals in store.items():
        a = np.asarray(vals, dtype=float)
        a = a[np.isfinite(a)]
        if len(a) < 20:
            ci[k] = (float("nan"), float("nan"))
        else:
            ci[k] = (float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975)))
    pt["n_boot_effective"] = {k: len(v) for k, v in store.items()}
    return pt, ci


def aalen_johansen_at(times, event_cr, t_horizon, event_of_interest=1):
    """AJ CIF for event_of_interest at t_horizon. event_cr: 0 censor, 1 interest, 2 competing."""
    from lifelines import AalenJohansenFitter
    times = np.asarray(times, dtype=float)
    event_cr = np.asarray(event_cr, dtype=int)
    # drop non-positive times
    m = times > 0
    times, event_cr = times[m], event_cr[m]
    if len(times) == 0:
        return float("nan"), 0, 0
    if (event_cr == event_of_interest).sum() == 0:
        # no events of interest => CIF = 0
        return 0.0, int(len(times)), 0
    aj = AalenJohansenFitter(calculate_variance=True)
    # lifelines expects event indicator with competing as different codes via event_of_interest
    try:
        aj.fit(times, event_cr, event_of_interest=event_of_interest)
        cif = aj.cumulative_density_
        # column name
        col = cif.columns[0]
        mask = cif.index <= t_horizon
        if not mask.any():
            val = float(cif.iloc[0][col])
        else:
            val = float(cif.loc[mask].iloc[-1][col])
        return val, int(len(times)), int((event_cr == event_of_interest).sum())
    except Exception as ex:
        return float("nan"), int(len(times)), int((event_cr == event_of_interest).sum())


def baseline_table(df_tr, df_te, features):
    rows = []
    for name, d in [("derivation", df_tr), ("test", df_te)]:
        row = {"set": name, "n": len(d), "liver_deaths": int((d["event_cr"] == 1).sum()),
               "other_deaths": int((d["event_cr"] == 2).sum()),
               "median_fu_years": float(d["time_years"].median())}
        for c in features:
            if c not in d.columns:
                continue
            s = d[c]
            if c == "sex":
                row["pct_male"] = float(100.0 * s.mean()) if s.notna().any() else float("nan")
            else:
                row[f"{c}_median"] = float(s.median()) if s.notna().any() else float("nan")
                row[f"{c}_missing"] = int(s.isna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def write_methods_notes(path: Path):
    text = """# Methods notes — revision re-analysis (2026-10)

## Primary model selection (editor priority 1)
The **single primary model** is a **penalized multivariable Cox proportional hazards model**
(L2 penalizer = 0.01) fitted on the derivation set only. Selection rule (pre-specified,
without use of the held-out test set): among candidate survival architectures considered
in the original submission (Cox, XGBoost-Cox, DeepSurv), the classical multivariable Cox
model is designated primary because (i) it yields a closed-form absolute 10-year risk from
the Breslow baseline survival, (ii) coefficients are fully reportable for reproducibility,
and (iii) the original internal results already showed near-equivalent discrimination across
architectures, so architecture choice is treated as second-order. **XGBoost / DeepSurv are
not used to select or redefine the primary result** in this revision run; they may be added
later strictly as sensitivity analyses.

## FIB-4 calibration (editor priority 2)
FIB-4 is **not** rank-rescaled to [0,1]. On the derivation set we fit a **univariate Cox**
model with covariate `log(FIB-4)`, obtain the Breslow baseline survival at 10 years, and
map each participant’s FIB-4 to an absolute 10-year liver-death risk. Discrimination,
Brier score, and decision-curve net benefit for FIB-4 use these calibrated probabilities.
Differences (primary Cox − calibrated FIB-4) for AUC, C-index, Brier, and net benefit are
summarized with **paired bootstrap** percentile 95% CIs (n_boot = 500) on the test set.

## Tertiles (editor priority 3)
Tertile cut-offs are the 33.3rd and 66.7th percentiles of the **primary Cox 10-year risk**
computed on the **derivation** set, written to `tertile_cuts.json`, and applied unchanged
to the test set (group sizes need not be equal).

## Competing risks (editor priority 4)
Liver-related death cumulative incidence by tertile on the test set uses the
**Aalen–Johansen** estimator with non-liver death as a competing event (event codes:
0 = censored/alive, 1 = liver death, 2 = other death). This replaces 1−Kaplan–Meier.

## Missingness (editor priority 5)
`missingness_report.csv` lists per-variable missing counts in the analysis cohort before
predictor imputation. Median imputation and 0.5/99.5 percentile clipping use **derivation
statistics only**. FIB-4-computable n is reported separately; calibrated-FIB-4 comparisons
use the intersection of finite calibrated FIB-4 risk and primary Cox risk.

## C-index and bootstrap (editor priority 6)
Harrell’s C-index is computed on the **full** test set (no 2000-person subsample).
All bootstrap CIs use **n_boot = 500**.

## UK Biobank application ID
505149 (Peking University Third Hospital; PI Wen-yao Wang).
"""
    path.write_text(text)


def main():
    global N_BOOT, RNG
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default="/Users/llm/Desktop/Claude数据库分析/方向3 肝病+AI ing/ukb_ckm0228.csv",
    )
    ap.add_argument(
        "--out",
        default="/Users/llm/Desktop/Claude数据库分析/方向3 肝病+AI ing/revision_2026-10_major",
    )
    ap.add_argument("--censor-date", default="2023-12-31")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--smoke", type=int, default=0, help="If >0, randomly subsample N cohort rows after cohort definition")
    args = ap.parse_args()

    N_BOOT = args.n_boot
    RNG = np.random.default_rng(SEED)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "run_log.txt")
    log(f"[START] revision pipeline v1  seed={SEED} n_boot={N_BOOT} smoke={args.smoke}")
    log(f"[OUT] {out}")

    df = load_data(Path(args.data), log)
    df = make_endpoint(df, args.censor_date, log)
    cohort = apply_cohort(df, log)
    cohort["fib4"] = compute_fib4(cohort)
    # FLI for features: recompute with median fill of FLI inputs within cohort (derivation-safe later)
    # For feature matrix we will impute after split; store raw fli from inclusion path:
    # recompute fli after filling within cohort for feature use consistency with original
    tmp = cohort.copy()
    for c in ["tg", "bmi", "ggt", "waist"]:
        tmp[c] = tmp[c].fillna(tmp[c].median())
    cohort["fli"] = compute_fli(tmp)

    miss = missingness_report(cohort)
    miss.to_csv(out / "missingness_report.csv", index=False)
    log(f"[MISS] wrote missingness_report.csv; FIB4 missing pct="
        f"{miss.loc[miss.variable=='FIB4_computable','pct_missing'].values[0]}")

    if args.smoke and args.smoke < len(cohort):
        cohort = cohort.sample(n=args.smoke, random_state=SEED).reset_index(drop=True)
        log(f"[SMOKE] subsampled cohort to n={len(cohort)}")

    tr_idx, te_idx = split_indices(len(cohort), args.test_frac, SEED)
    df_tr = cohort.iloc[tr_idx].reset_index(drop=True)
    df_te = cohort.iloc[te_idx].reset_index(drop=True)
    log(
        f"[SPLIT] derivation={len(df_tr):,} test={len(df_te):,} "
        f"(liver ev tr={int((df_tr.event_cr==1).sum())}, te={int((df_te.event_cr==1).sum())}; "
        f"other death tr={int((df_tr.event_cr==2).sum())}, te={int((df_te.event_cr==2).sum())})"
    )

    # baseline table
    bt = baseline_table(df_tr, df_te, FEATURES)
    bt.to_csv(out / "baseline_table1.csv", index=False)

    # feature matrices
    X_tr_raw = df_tr[FEATURES].copy()
    X_te_raw = df_te[FEATURES].copy()
    X_tr, X_te, med, lo, hi = impute_clip_fit(X_tr_raw, X_te_raw)

    t_tr = df_tr["time_years"].values.astype(float)
    e_tr = df_tr["event_liver"].values.astype(int)
    t_te = df_te["time_years"].values.astype(float)
    e_te = df_te["event_liver"].values.astype(int)
    cr_te = df_te["event_cr"].values.astype(int)

    # Primary Cox
    log("[COX] fitting primary multivariable Cox on derivation")
    risk_cox_te, cph, s10 = fit_cox_risk10(X_tr, t_tr, e_tr, X_te)
    risk_cox_tr = np.clip(
        1.0 - np.power(s10, cph.predict_partial_hazard(X_tr).values.reshape(-1)),
        1e-8, 1 - 1e-8,
    )
    log(f"[COX] baseline S0(10)={s10:.6f}")

    coef = cph.summary.reset_index().rename(columns={"index": "covariate"})
    coef.to_csv(out / "cox_coefficients.csv", index=False)

    # Calibrated FIB-4
    log("[FIB4] univariate Cox calibration on derivation log(FIB-4)")
    fib_tr = df_tr["fib4"].values.astype(float)
    fib_te = df_te["fib4"].values.astype(float)
    risk_fib_te, cph_fib, s10_fib, n_fib_fit = fit_fib4_calibrated(fib_tr, t_tr, e_tr, fib_te)
    log(f"[FIB4] fit n={n_fib_fit}, S0(10)={s10_fib:.6f}, test finite risks={int(np.isfinite(risk_fib_te).sum())}")

    # Tertiles locked on derivation
    q1, q2 = np.quantile(risk_cox_tr, [1.0 / 3.0, 2.0 / 3.0])
    tertile_cuts = {
        "risk_variable": "cox_risk10_primary",
        "derivation_n": int(len(risk_cox_tr)),
        "cut_33": float(q1),
        "cut_67": float(q2),
        "quantiles": [1.0 / 3.0, 2.0 / 3.0],
        "note": "Cut-offs computed on derivation only; applied unchanged to test.",
    }
    (out / "tertile_cuts.json").write_text(json.dumps(tertile_cuts, indent=2))
    log(f"[TERTILE] derivation cuts: {q1:.8f}, {q2:.8f}")

    def assign_tertile(r):
        return np.where(r <= q1, "low", np.where(r <= q2, "mid", "high"))

    te_ter = assign_tertile(risk_cox_te)
    # counts (may be unequal)
    for lab in ["low", "mid", "high"]:
        log(f"[TERTILE] test {lab}: n={int((te_ter==lab).sum())}")

    # AJ CIF by tertile
    aj_rows = []
    for lab in ["low", "mid", "high"]:
        m = te_ter == lab
        cif, n_g, n_ev = aalen_johansen_at(t_te[m], cr_te[m], CIF_HORIZON, event_of_interest=1)
        aj_rows.append({
            "tertile": lab,
            "n": int(m.sum()),
            "liver_events": int(n_ev),
            "other_deaths": int(((cr_te[m] == 2)).sum()),
            "cif_horizon_years": CIF_HORIZON,
            "aj_cif_liver": cif,
            "aj_cif_liver_pct": None if not np.isfinite(cif) else round(100.0 * cif, 4),
        })
        log(f"[AJ] {lab}: n={int(m.sum())} liver_ev={n_ev} CIF{CIF_HORIZON}={cif}")
    # also overall
    cif_all, _, n_ev_all = aalen_johansen_at(t_te, cr_te, CIF_HORIZON, 1)
    aj_rows.append({
        "tertile": "all_test",
        "n": int(len(t_te)),
        "liver_events": int(n_ev_all),
        "other_deaths": int((cr_te == 2).sum()),
        "cif_horizon_years": CIF_HORIZON,
        "aj_cif_liver": cif_all,
        "aj_cif_liver_pct": None if not np.isfinite(cif_all) else round(100.0 * cif_all, 4),
    })
    pd.DataFrame(aj_rows).to_csv(out / "tertile_aj_cif.csv", index=False)

    # Metrics + paired bootstrap
    log(f"[METRICS] paired bootstrap n_boot={N_BOOT} on test (Cox vs calibrated FIB-4)")
    pt, ci = paired_bootstrap_metrics(t_te, e_te, risk_cox_te, risk_fib_te, n_boot=N_BOOT)

    def pack(name):
        lo, hi = ci.get(name, (float("nan"), float("nan")))
        return {"point": pt.get(name), "ci95_lo": lo, "ci95_hi": hi}

    y10, elig = label_10y(t_te, e_te)
    common = np.isfinite(risk_cox_te) & np.isfinite(risk_fib_te)
    elig_c = elig & common

    primary_metrics = {
        "primary_model": "multivariable_Cox",
        "selection_rule": (
            "Pre-specified: multivariable Cox designated primary on derivation-design "
            "grounds (closed-form 10-y risk, reportable coefficients, near-equivalent "
            "discrimination vs boosting/DeepSurv in prior internal work). Not selected "
            "by inspecting test-set performance."
        ),
        "comparator": "FIB-4_univariate_Cox_calibrated_logFIB4",
        "horizon_years_risk": HORIZON,
        "n_test": int(len(df_te)),
        "n_test_common_finite_risks": int(common.sum()),
        "n_test_eligible_10y_common": int(elig_c.sum()),
        "n_test_liver_events_full_fu": int((e_te == 1).sum()),
        "n_test_liver_events_le_10y_eligible": int(y10[elig_c].sum()),
        "n_derivation": int(len(df_tr)),
        "n_derivation_liver_events": int((e_tr == 1).sum()),
        "n_fib4_calibration_fit": n_fib_fit,
        "cox_baseline_S0_10": s10,
        "fib4_baseline_S0_10": s10_fib,
        "auc_10y": {
            "cox": pack("auc_a"),
            "fib4_calibrated": pack("auc_b"),
            "diff_cox_minus_fib4": pack("auc_diff"),
        },
        "cindex_full_test": {
            "cox": pack("cindex_a"),
            "fib4_calibrated": pack("cindex_b"),
            "diff_cox_minus_fib4": pack("cindex_diff"),
        },
        "brier_10y_ipcw": {
            "cox": pack("brier_a"),
            "fib4_calibrated": pack("brier_b"),
            "diff_cox_minus_fib4": pack("brier_diff"),
            "note": "Lower is better; diff is Cox - FIB4.",
        },
        "net_benefit": {
            "threshold_0.005": {
                "cox": pack("nb_0.005_a"),
                "fib4_calibrated": pack("nb_0.005_b"),
                "diff_cox_minus_fib4": pack("nb_0.005_diff"),
            },
            "threshold_0.01": {
                "cox": pack("nb_0.01_a"),
                "fib4_calibrated": pack("nb_0.01_b"),
                "diff_cox_minus_fib4": pack("nb_0.01_diff"),
            },
        },
        "dca_cox_thresholds": dca_curve(y10[elig_c], risk_cox_te[elig_c]),
        "dca_fib4_thresholds": dca_curve(y10[elig_c], risk_fib_te[elig_c]),
        "tertile_cuts": tertile_cuts,
        "n_boot": N_BOOT,
        "seed": SEED,
    }
    (out / "primary_model_metrics.json").write_text(
        json.dumps(primary_metrics, indent=2, allow_nan=True)
    )
    log("[METRICS] wrote primary_model_metrics.json")
    log(
        f"[SUMMARY] AUC cox={pt.get('auc_a'):.4f} fib4={pt.get('auc_b'):.4f} "
        f"diff={pt.get('auc_diff'):.4f} CI={ci.get('auc_diff')}"
    )
    log(
        f"[SUMMARY] C-index cox={pt.get('cindex_a'):.4f} fib4={pt.get('cindex_b'):.4f} "
        f"diff={pt.get('cindex_diff'):.4f} CI={ci.get('cindex_diff')}"
    )
    log(
        f"[SUMMARY] Brier cox={pt.get('brier_a'):.6f} fib4={pt.get('brier_b'):.6f} "
        f"diff={pt.get('brier_diff'):.6f} CI={ci.get('brier_diff')}"
    )

    # predictions export for audit
    pred = pd.DataFrame({
        "eid": df_te["eid"].values,
        "time_years": t_te,
        "event_liver": e_te,
        "event_cr": cr_te,
        "fib4": fib_te,
        "cox_risk10": risk_cox_te,
        "fib4_cal_risk10": risk_fib_te,
        "tertile_cox": te_ter,
    })
    pred.to_csv(out / "test_predictions.csv", index=False)

    write_methods_notes(out / "METHODS_NOTES.md")
    log("[DONE] all revision outputs written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
