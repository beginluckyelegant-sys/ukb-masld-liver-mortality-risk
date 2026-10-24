# Methods notes — revision re-analysis (2026-10)

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
**505149** (Peking University Third Hospital; PI Wen-yao Wang).

## C-index implementation note
Harrell’s C-index is computed with `sksurv.metrics.concordance_index_censored`, where a **higher score denotes higher risk** (shorter survival). (lifelines `concordance_index` uses the opposite score polarity in the installed version and must not be used with raw risk probabilities without negation.)

## Sensitivity architectures
XGBoost and DeepSurv were **not** re-fit in this revision run. They remain candidates for secondary/sensitivity analyses only and must not redefine the primary model or tertiles.
