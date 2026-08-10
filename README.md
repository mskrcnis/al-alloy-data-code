# Aluminium-alloy data and reproducibility package

This repository releases the processed 452-record aluminium-alloy dataset and
the code and exported results needed to reproduce the manuscript analyses. It
contains the verified 270/91/91 proper-training, calibration, and final-test
partition, the completed fixed-k search inputs, the fresh fixed-k=12
continuation, Reviewer 1 Comment 1--3 outputs, and the corrected Comment 4--7
outputs.

The package is an analysis release: it does not rerun the completed
36,000-configuration fixed-k experiment by default. Its exported search
records are included as inputs, and the fresh final-k=12 continuation artifacts
are included as results.

## Required Comment 3 modules

All three modules are present together in `code/workflow/`, so the continuation
script's imports resolve without relying on files outside the repository:

- `comment3_k12_continuation_core.py`
- `comment3_final_core.py`
- `comment3_final_joint_search_core.py`

## Contents

- `data/processed_retained_dataset_452.csv` — 452 complete-target records with
  stable `Original_Index` and `Data_Split` labels.
- `data/split_assignments_270_91_91.csv` and
  `data/split_summary_270_91_91.csv` — the verified 270/91/91 assignments.
- `data/outer_fold_indices_explicit.csv` — all 25 repeated outer folds with
  explicit memberships.
- `data/inner_fold_indices_explicit.csv` — all 1,875 fixed-k inner-fold
  memberships (25 outer evaluations × 15 feature counts × 5 folds), including
  explicit IDs, seeds, and hashes. Its hashes were checked against the completed
  search table.
- `search_inputs/fixed_k_completed_search/` — exported completed-search tables,
  stability records, one-SE records, provenance, and seed information. The
  180,000-row inner-result table is gzip-compressed and can be read by pandas
  with `compression="infer"`.
- `search_inputs/alpha_sensitivity/` — alpha-selection extraction, reconciliation
  reports, unit tests, and the associated figure.
- `code/workflow/` — Comment 1--3 and Comment 4--7 workflow scripts and cores.
- `code/shap_treeexplainer_stability.py` — the TreeExplainer-based SHAP script;
  corresponding SHAP tables and figures are in the output package.
- `code/optional_comparisons/` — the Comment 10 and GPR uncertainty-comparison
  scripts and their core modules.
- `outputs/k12_continuation/` — locked k=12 model, predictions, metrics,
  conformal results, SHAP results, Comment 2 candidate-screening outputs, and
  Comment 3 stability figures/tables.
- `outputs/comment4_7/` — corrected Reviewer 1 Comment 4--7 outputs, including
  grouped-fold assignments and SHAP integrity evidence.
- `seeds/` and `environment/` — random-seed manifests, environment records,
  execution provenance, and the validation environment.
- `figures/variable_distributions.png` — the high-resolution 4×5 distribution
  figure for the 20 released variables.
- `figures/high_resolution/` — high-resolution PNG copies of all three
  actual-vs-predicted plots, all three conformal-interval plots, all three
  TreeExplainer SHAP summaries, and the variable-distribution figure. Each is
  at least 150 KB; the model figures are rendered at 600 DPI.

## Execution order

For a full analysis rerun, use this order from the repository root:

1. Install the environment with `python -m pip install -r requirements.txt`.
2. Inspect `data/processed_retained_dataset_452.csv` and the split manifests.
3. Rebuild and verify explicit fold memberships:

   ```bash
   python code/build_explicit_fold_indices.py
   ```

4. Treat `search_inputs/fixed_k_completed_search/` as the exported input from
   the completed 36,000-configuration/180,000-inner-fold search. Do not rerun
   that experiment unless an independent computational rerun is intended.
5. Run the fresh k=12 continuation with
   `code/workflow/al-alloy-comment3-final-k12-continuation.py`. Keep the three
   Comment 3 modules in the same directory. The continuation uses the 270
   proper-training records for selection and leaves calibration and final-test
   records untouched during selection.
6. Regenerate Comment 1 with
   `code/workflow/al-alloy-comment1-conformal-corrected.py`, then Comment 2 with
   `code/workflow/al-alloy-comment2-screening-analysis.py`, using the locked
   final ensemble and the exported candidate-screening outputs as references.
7. Use `code/workflow/al-alloy-comments4-7-corrections.py` for the corrected
   Comment 4--7 workflow when its package-relative source inputs are available.
8. Use the scripts in `code/optional_comparisons/` only for the additional
   uncertainty comparisons.
9. Regenerate the standalone distribution figure:

   ```bash
   python code/generate_variable_distributions.py
   ```

The repository contains exported results so that the reported tables and
figures can be inspected without spending the original 30-hour search runtime.

## Validation and provenance

The released validation records confirm the unchanged, disjoint 270/91/91
partition; 96 fresh k=12 aggregate configurations and 480 inner-fold fits;
exactly 12 final features; no calibration or final-test use during selection;
same-ensemble use by Comment 1 and Comment 2; metric reproduction from exported
predictions; complete zero-inclusive stability matrices; and nonempty required
figures. The explicit inner-fold builder independently verifies all reconstructed
hashes against the completed fixed-k search input.

The package preserves the selected 12-feature continuation as the final feature
count. Any k=8 files are retained only as the documented one-standard-error
alternative and are not the final model.

## Data availability statement

The complete-target aluminium-alloy dataset and supporting analysis code are
openly available at:

> https://github.com/mskrcnis/al-alloy-data-code

The dataset contains 452 records with complete YS, UTS, and elongation targets.
Composition variables are reported in wt.%, heat-treatment temperatures in °C,
aging time in hours, strength values in MPa, and elongation in percent.

The released processed dataset was created from the project workbook by
retaining rows with non-missing YS, UTS, and El values. Model predictions,
calibration outcomes, and uncertainty intervals were not used to construct the
released records.
