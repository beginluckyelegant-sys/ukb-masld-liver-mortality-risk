# UK Biobank MASLD liver-mortality clinical-risk model

Reproducible analysis code for a **multivariable Cox proportional-hazards clinical-risk model** predicting **liver-related mortality** among participants with **MASLD** (metabolic dysfunction–associated steatotic liver disease) in the **UK Biobank** (**application 505149**). This package supports the major revision of Scientific Reports manuscript **65e1e7c0-b53a-474f-8813-9205bb0ba9d8**.

**DOI:** [10.5281/zenodo.23612629](https://doi.org/10.5281/zenodo.23612629) (concept DOI [10.5281/zenodo.23612628](https://doi.org/10.5281/zenodo.23612628))

## What is included / not included

| Included | Not included |
| --- | --- |
| Primary revision pipeline (`src/pipeline_revision_v1.py`) | UK Biobank **participant-level** extracts (`ukb*.csv`, eid-level files) |
| Methods notes (`notes/METHODS_NOTES.md`) | Raw death registry dumps or linked hospital records |
| Aggregate revision outputs under `results/` | Trained XGBoost / DeepSurv weights (sensitivity not re-fit here) |
| Dependency pin list and data-access guidance | Manuscript Word drafts |

See [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md) for how approved researchers obtain UKB data under application **505149**. **No individual-level data are shipped in this repository.**

## Environment

- **Python 3.11** (recommended)
- Core packages: `numpy`, `pandas`, `scikit-learn`, `scikit-survival`, `lifelines`, `matplotlib`

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Primary run does **not** require `torch` or `xgboost`.

## How to run

Provide a local UK Biobank extract CSV whose column headers match the names in `COLS` inside `src/pipeline_revision_v1.py`.

```bash
python src/pipeline_revision_v1.py \
  --data /path/to/your_ukb_extract.csv \
  --out  /path/to/revision_outputs \
  --censor-date 2023-12-31 \
  --test-frac 0.2 \
  --n-boot 500
```

Optional smoke test (subsample cohort after definition):

```bash
python src/pipeline_revision_v1.py --data /path/to/extract.csv --out /tmp/smoke --smoke 5000 --n-boot 20
```

CLI flags: `--data`, `--out`, `--censor-date`, `--test-frac`, `--n-boot`, `--smoke`.

Outputs written to `--out` include `primary_model_metrics.json`, `tertile_cuts.json`, `tertile_aj_cif.csv`, `baseline_table1.csv`, `cox_coefficients.csv`, `missingness_report.csv`, and `run_log.txt`. Copies of the locked revision aggregates are already present under [`results/`](results/).

## Locked primary results (revision package)

Pre-specified **primary model**: L2-penalized multivariable Cox on the derivation set (not selected by test-set shopping). Comparator: **FIB-4** mapped to absolute 10-year risk via a univariate Cox on `log(FIB-4)` fitted on derivation only (no rank-rescaling to [0, 1]).

| Metric (test set) | Multivariable Cox | Calibrated FIB-4 | Difference (Cox − FIB-4) |
| --- | ---: | ---: | ---: |
| AUC at 10 years | **0.873** | 0.774 | **+0.099** (95% CI 0.054–0.154) |
| Harrell C-index (full test) | **0.848** | 0.771 | +0.077 (95% CI 0.043–0.109) |

- Derivation / test *n*: 88,046 / 22,011 (seed 2025; 80/20 split).
- Tertile cuts of Cox 10-year risk (derivation only): **0.001140** / **0.001928**.
- Aalen–Johansen 15-year CIF for liver death by tertile on test (competing: non-liver death): **0.094%** / **0.203%** / **1.261%**.
- Bootstrap: `n_boot = 500` paired percentile CIs.

Exact machine-precision values are in [`results/primary_model_metrics.json`](results/primary_model_metrics.json) and [`results/tertile_aj_cif.csv`](results/tertile_aj_cif.csv).

## Scope note (sensitivity models)

**XGBoost-Cox and DeepSurv were not re-fit in this revision package.** They remain candidates for secondary/sensitivity analyses only and must not redefine the primary model or tertile cut-offs.

## Repository layout

```
ukb-masld-liver-mortality-risk/
  src/pipeline_revision_v1.py
  notes/METHODS_NOTES.md
  results/          # aggregate metrics only
  docs/DATA_ACCESS.md
  requirements.txt
  LICENSE           # MIT (code)
  CITATION.cff
```

## Citation

Please cite the Scientific Reports article (when published) and this repository. See [`CITATION.cff`](CITATION.cff).

```
Wang WY, et al. Multivariable Cox clinical-risk model for liver-related mortality
in MASLD. Scientific Reports (revision). UK Biobank application 505149.
Code: https://github.com/beginluckyelegant-sys/ukb-masld-liver-mortality-risk
DOI: https://doi.org/10.5281/zenodo.23612629
```

## License

Code is released under the [MIT License](LICENSE). UK Biobank individual-level data remain subject to UK Biobank terms and the Material Transfer Agreement for application **505149** and are **not** redistributed here.

## Contact

Corresponding author / application PI details: see manuscript (Peking University Third Hospital; UKB application **505149**). Repository issues may be opened after the public GitHub remote is created.
