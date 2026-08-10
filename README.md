# Aluminium-alloy data and figure-generation code

This repository contains the 452 aluminium-alloy records for which yield strength (YS), ultimate tensile strength (UTS), and elongation (El) are all available, together with the small reproducible script used to generate the variable-distribution figure.

## Contents

- `data/al_alloy_complete_target_records.csv` — 452 complete-target records and the 20 variables used in the analysis.
- `data/data_dictionary.csv` — variable descriptions and units.
- `code/generate_variable_distributions.py` — reproducible 4×5 histogram-figure script.
- `figures/variable_distributions.png` — 600-dpi generated figure.

The released CSV intentionally contains the 20 measurement and processing variables only; local split labels and model-analysis artifacts are not included.

## Reproduce the figure

From the repository root:

```bash
python -m pip install -r requirements.txt
python code/generate_variable_distributions.py
```

## Data availability statement

The complete-target aluminium-alloy dataset and the code used to generate the variable-distribution figure are openly available in this repository:

> The dataset and supporting code are available at https://github.com/mskrcnis/al-alloy-data-code.

Please replace the repository URL above if the repository name or GitHub account is changed before manuscript submission.

The dataset contains 452 records with complete YS, UTS, and elongation targets. Composition variables are reported in wt.%, heat-treatment temperatures in °C, aging time in hours, strength values in MPa, and elongation in percent.

## Provenance

The CSV release was created from the project’s aluminium-alloy workbook by retaining rows with non-missing YS, UTS, and El values. No model predictions, calibration outcomes, test labels, or uncertainty intervals were used to construct the released records.
