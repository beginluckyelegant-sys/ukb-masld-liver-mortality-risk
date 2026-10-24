# Data access

This repository ships **analysis code and aggregate summary outputs only**.
It does **not** contain UK Biobank participant-level data.

## Approved application

- **UK Biobank Application ID:** 505149
- **Institution:** Peking University Third Hospital
- **Principal Investigator:** Wen-yao Wang

Individual-level data used for the manuscript were obtained under a UK Biobank
Material Transfer Agreement for application **505149**. Redistribution of those
data is prohibited.

## How to obtain data for re-running the pipeline

1. Apply for UK Biobank access via [https://www.ukbiobank.ac.uk/](https://www.ukbiobank.ac.uk/)
   (or request collaboration / data sharing under an approved application that
   covers the same research aims).
2. After approval, download the fields corresponding to the column names listed
   in `COLS` inside `src/pipeline_revision_v1.py` (demographics, anthropometrics,
   biochemistry, blood counts, death registry).
3. Build a single wide CSV extract whose headers match those `COLS` values
   (or adapt the mapping in the script).
4. Point the pipeline at your local extract:

```bash
python src/pipeline_revision_v1.py \
  --data /path/to/your_ukb_extract.csv \
  --out  /path/to/output_dir
```

## What this repo does include

Aggregate, non-identifying revision outputs under `results/` (model metrics,
tertile cuts, Aalen–Johansen CIFs, Table 1 summaries, Cox coefficients,
missingness rates, run log). These cannot be used to re-identify participants.

## Contact

For data-access questions related to application 505149, contact the approved
application PI / corresponding author listed in the manuscript (placeholder
until public contact details are confirmed at publication).
