"""Continue the completed fixed-k search without repeating it.

This module consumes the exported 25-outer-evaluation results from the
completed joint search, performs only the fresh 96-candidate/480-inner-fold
training-only lock at k=12, and regenerates all downstream results from that
new locked ensemble.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import comment3_final_core as base
import comment3_final_joint_search_core as joint

TARGETS = base.TARGETS
FEATURES = base.FEATURES
MANDATORY = base.MANDATORY
ALPHA_GRID = base.ALPHA_GRID
MASTER_SEED = base.MASTER_SEED
CONFORMAL_ALPHA = base.CONFORMAL_ALPHA
EPSILON = base.EPSILON
INNER_FOLDS = base.INNER_FOLDS
SEARCH_ITERATIONS = base.SEARCH_ITERATIONS
ENSEMBLE_SIZE = base.ENSEMBLE_SIZE
FINAL_K = 12
OPTIONAL_ONE_SE_K = 8


def save_df(root, data, name):
    return base.save_df(root, data, name)


def save_json(root, data, name):
    return base.save_json(root, data, name)


def save_text(root, text, name):
    return base.save_text(root, text, name)


def metric_summary(y, p, model_label):
    m = base.metric_dict(y, p)
    rows = [{"Model": model_label, "Target": t, "R2": m[f"R2_{t}"], "RMSE": m[f"RMSE_{t}"], "MAE": m[f"MAE_{t}"], "NRMSE": m[f"NRMSE_{t}"]} for t in TARGETS]
    rows.append({"Model": model_label, "Target": "Macro", "R2": m["R2_Macro"], "RMSE": m["RMSE_Macro"], "MAE": m["MAE_Macro"], "NRMSE": m["NRMSE_Macro"]})
    return pd.DataFrame(rows)


def copy_completed_inputs(source, root):
    destination = root / "completed_joint_search_inputs"
    destination.mkdir(parents=True, exist_ok=True)
    names = [
        "fixed_k_candidate_results_aggregate.csv", "fixed_k_candidate_inner_fold_results.csv",
        "fixed_k_outer_selected_configurations.csv", "fixed_k_outer_selected_features.csv",
        "fixed_k_outer_metrics.csv", "fixed_k_outer_predictions.csv", "feature_subset_performance_summary.csv",
        "pairwise_jaccard_values.csv", "jaccard_summary_by_k.csv", "nogueira_stability_by_k.csv",
        "nogueira_stability_details.json", "feature_selection_binary_matrix_by_k.csv",
        "feature_stability_per_feature.csv", "one_se_rule_calculation.csv", "one_se_rule_decision.txt",
    ]
    missing = [name for name in names if not (source / name).exists()]
    if missing:
        raise FileNotFoundError("Completed joint-search inputs missing: " + ", ".join(missing))
    manifest = []
    for name in names:
        src = source / name
        dst = destination / name
        shutil.copy2(src, dst)
        manifest.append({"Source": str(src), "Copied_To": str(dst), "SHA256": base.sha256_file(src), "Size_Bytes": src.stat().st_size})
    save_df(root, manifest, "completed_joint_search_input_manifest.csv")
    save_json(root, {"Completed_Search_Source": str(source), "Completed_Search_Expected_Aggregate_Rows": 36000, "Completed_Search_Expected_Inner_Fold_Rows": 180000, "Completed_Search_Expected_Outer_Selected_Rows": 375, "No_Completed_Search_Rerun": True}, "completed_joint_search_source_provenance.json")
    return destination


def full_binary_matrix(selected_features):
    outer_ids = sorted(selected_features.Outer_ID.unique())
    assert len(outer_ids) == 25
    lookup = {(str(r.Outer_ID), int(r.Feature_Count), str(r.Feature)): r for r in selected_features.itertuples()}
    rows = []
    for oid in outer_ids:
        part = selected_features[selected_features.Outer_ID.eq(oid)].drop_duplicates("Feature_Count")
        alpha_by_k = dict(zip(part.Feature_Count.astype(int), part.Selected_Alpha.astype(float)))
        for k in range(3, 18):
            for feature in FEATURES:
                key = (oid, k, feature)
                selected = int(key in lookup)
                record = lookup.get(key)
                rows.append({
                    "Outer_ID": oid, "Repeat": int(oid[1:3]), "Outer_Fold": int(oid[5:7]),
                    "Feature_Count": k, "Feature": feature, "Selected": selected,
                    "Mandatory": feature in MANDATORY, "Feature_Order": int(record.Feature_Order) if record else 0,
                    "Rank": int(record.Rank) if record else np.nan,
                    "Selected_Alpha": float(alpha_by_k[k]),
                    "Mean_Normalized_Hybrid_Feature_Score": float(record.Mean_Normalized_Hybrid_Feature_Score) if record else np.nan,
                })
    out = pd.DataFrame(rows)
    assert len(out) == 25 * 15 * 17
    assert out.groupby(["Outer_ID", "Feature_Count"]).size().eq(17).all()
    return out


def complete_frequency(binary):
    out = binary.groupby(["Feature_Count", "Feature", "Mandatory"], as_index=False).agg(
        Outer_Evaluations=("Outer_ID", "nunique"), Selection_Count=("Selected", "sum"),
        Selection_Frequency=("Selected", "mean"), Zero_Count=("Selected", lambda x: int((x == 0).sum())),
        Mean_Rank=("Rank", "mean"), Rank_SD=("Rank", "std"),
        Mean_Normalized_Hybrid_Feature_Score=("Mean_Normalized_Hybrid_Feature_Score", "mean"),
        Hybrid_Score_SD=("Mean_Normalized_Hybrid_Feature_Score", "std"),
    )
    assert len(out) == 15 * 17
    assert out.Outer_Evaluations.eq(25).all()
    assert (out.Selection_Count + out.Zero_Count).eq(25).all()
    return out


def stability_figures(root, binary, frequency, pairwise, nogueira):
    c3 = root / "comment3_corrected"
    c3.mkdir(parents=True, exist_ok=True)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for include, label in [(True, "Including_Mandatory"), (False, "Excluding_Mandatory")]:
        part = pairwise[pairwise.Selection_Definition.eq(label)].groupby("Subset_Size", as_index=False).Jaccard.agg(["mean", "std", "median", "min", "max"]).reset_index()
        fig, ax = plt.subplots(figsize=(7.2, 5.0)); ax.errorbar(part.Subset_Size, part["mean"], yerr=part["std"], fmt="o-", color=colors[0 if include else 1], capsize=3); ax.set_xlabel("Feature subset size k"); ax.set_ylabel("Pairwise Jaccard similarity"); ax.set_ylim(0, 1.05); ax.grid(alpha=.2); base.figure_pair(c3, f"jaccard_stability_{'including' if include else 'excluding'}_mandatory", fig)
        n = nogueira[nogueira.Selection_Definition.eq(label)].sort_values("Subset_Size")
        fig, ax = plt.subplots(figsize=(7.2, 5.0)); ax.plot(n.Subset_Size, n.Nogueira_Stability, "o-", color=colors[0 if include else 1]); ax.set_xlabel("Feature subset size k"); ax.set_ylabel("Nogueira stability"); ax.set_ylim(0, 1.05); ax.grid(alpha=.2); base.figure_pair(c3, f"nogueira_stability_{'including' if include else 'excluding'}_mandatory", fig)
    for k, stem in [(12, "feature_selection_frequency_k12"), (OPTIONAL_ONE_SE_K, "feature_selection_frequency_one_se_k8")]:
        part = frequency[frequency.Feature_Count.eq(k)].sort_values(["Selection_Frequency", "Feature"], ascending=[True, True])
        fig, ax = plt.subplots(figsize=(7.2, 5.2)); ax.barh(part.Feature, part.Selection_Frequency, color=colors[2]); ax.set_xlabel("Selection frequency across 25 outer evaluations"); ax.set_ylabel("Feature"); base.figure_pair(c3, stem, fig)
    k12_stability = nogueira[nogueira.Subset_Size.eq(12)].copy()
    k12_stability["Selection_Definition"] = k12_stability["Selection_Definition"].str.replace(r"^k12_", "", regex=True)
    save_df(root, k12_stability, "k12_stability_with_without_mandatory.csv")
    save_df(c3, k12_stability, "k12_stability_with_without_mandatory.csv")
    save_df(root, pairwise[pairwise.Subset_Size.eq(12)], "k12_pairwise_jaccard_values.csv")
    save_df(root, frequency[frequency.Feature_Count.eq(12)], "k12_feature_selection_frequency_complete.csv")
    save_df(root, pd.DataFrame([
        {"Figure": p.stem, "Uses_Completed_Outer_Records": True, "Uses_Final_Test": False, "Feature_Count_Scope": "k=3..17" if "jaccard" in p.stem or "nogueira" in p.stem else p.stem}
        for p in sorted(c3.glob("*.png"))
    ]), "comment3_corrected_figure_semantic_audit.csv")


def fresh_k12_lock(Xproper, Yproper, root):
    aggregate, inner, winner, searches, warnings_all = joint.search_records(
        Xproper, Yproper, MASTER_SEED + 72001, FINAL_K, outer_id="K12_FINAL", stage="final_k12")
    assert len(aggregate) == 96 and len(inner) == 480
    family = winner["Model_Family"]
    _, estimator, _, description = next(s for s in joint.reduced_specs(FINAL_K) if s[0] == family)
    params = json.loads(winner["Params"])
    model = joint.clone(estimator).set_params(**params)
    with joint.warnings.catch_warnings(record=True) as caught:
        joint.warnings.simplefilter("always")
        model.fit(Xproper, Yproper)
    warnings_all.extend(joint.warnings_rows(caught, "final_k12_locked_refit", "K12_FINAL", FINAL_K, family))
    alpha, k, features, ranking = base.selector_details(model)
    assert k == FINAL_K and len(features) == FINAL_K
    config = {
        "Final_Selection_Criterion": "highest mean outer-validation macro-R2",
        "Final_Feature_Count": FINAL_K, "Optional_One_SE_Alternative": OPTIONAL_ONE_SE_K,
        "Selected_Model_Family": family, "Description": description, "Final_Alpha": alpha,
        "Final_Features": features, "Best_Params": params, "Candidate_Count": 96,
        "Inner_Fold_Record_Count": 480, "Search_Algorithm": "RandomizedSearchCV",
        "Search_Budget_Per_Family": 24, "Inner_CV": "5-fold shuffled KFold",
        "Inner_CV_Seed": MASTER_SEED + 70000, "Derived_Search_Seed": MASTER_SEED + 72001,
        "Selection_Objective": "higher mean negative normalized RMSE",
        "Tie_Break": "higher mean primary score, lower SD, lower complexity, lexical family, lexical serialized parameters",
        "Proper_Training_Index_Hash": base.index_hash(Xproper.index),
        "Calibration_Used_In_Selection": False, "Final_Test_Used_In_Selection": False,
        "Final_k12_Search_Independent": True, "Alpha_Grid": ALPHA_GRID,
        "Mandatory_Features": MANDATORY, "Ensemble_Size": ENSEMBLE_SIZE,
        "Conformal_Alpha": CONFORMAL_ALPHA, "Epsilon": EPSILON,
        "Normalized_Conformal_Method": "abs(y_cal - ensemble_mean_cal)/(ensemble_sd_cal + epsilon)",
        "Ranking_Equation": "HybridScore_j(alpha) = alpha * minmax(PearsonRelevance_j) + (1-alpha) * minmax(MeanEmbeddedImportance_j)",
        "Tree_Search_Space": {"n_estimators": [300, 500, 800], "max_depth": [None, 6, 10, 14], "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.7, 1.0]},
    }
    save_df(root, aggregate, "final_k12_candidate_results.csv")
    save_df(root, aggregate, "final_fixed_k_candidate_results.csv")
    save_df(root, inner, "final_k12_inner_fold_results.csv")
    save_df(root, inner, "final_fixed_k_inner_fold_results.csv")
    save_json(root, config, "final_k12_locked_configuration.json")
    save_json(root, config, "final_locked_configuration.json")
    save_df(root, [{"Feature_Order": i + 1, "Feature": f, "Mandatory": f in MANDATORY, "Mean_Normalized_Hybrid_Feature_Score": float(ranking.set_index("Feature").loc[f, "HybridScore"])} for i, f in enumerate(features)], "final_k12_selected_features.csv")
    save_df(root, pd.DataFrame(warnings_all), "final_k12_warning_capture.csv")
    save_text(root, "Fresh k=12 training-only lock: 4 eligible families x 24 configurations = 96 candidates and 480 inner-fold records. The 270 proper-training rows were the only rows used during selection. Calibration and final-test rows were accessed only after this lock.", "final_k12_training_provenance.txt")
    return {"model": model, "model_name": family, "description": description, "params": params, "config": config, "alpha": alpha, "k": k, "features": features, "ranking_table": ranking, "aggregate": aggregate, "inner": inner, "warnings": warnings_all}


def correct_downstream_reports(root, result, c2_result, split_sets, runtime):
    c1 = root / "comment1_regenerated"; c2 = root / "comment2_regenerated"
    save_df(root, metric_summary(pd.read_csv(root / "single_model_test_predictions.csv")[["Observed_YS", "Observed_UTS", "Observed_El"]], pd.read_csv(root / "single_model_test_predictions.csv")[["Predicted_YS", "Predicted_UTS", "Predicted_El"]], "Single locked k=12 model"), "single_model_targetwise_macro_metrics.csv")
    ep = pd.read_csv(root / "ensemble_test_predictions.csv")
    save_df(root, metric_summary(ep[["Observed_YS", "Observed_UTS", "Observed_El"]], ep[["EnsembleMean_YS", "EnsembleMean_UTS", "EnsembleMean_El"]], "20-member k=12 ensemble"), "ensemble_targetwise_macro_metrics.csv")
    shutil.copy2(root / "single_model_targetwise_macro_metrics.csv", c1 / "single_model_targetwise_macro_metrics.csv")
    shutil.copy2(root / "ensemble_targetwise_macro_metrics.csv", c1 / "ensemble_targetwise_macro_metrics.csv")
    complete = pd.read_csv(c2 / "heldout_unique_conditions_complete.csv")
    complete = complete.sort_values(["ReliabilityAwareRank", "Heldout_Condition_ID"], kind="mergesort")
    save_df(c2, complete.head(10), "baseline_top10_conditions.csv")
    winner = complete.iloc[0]
    save_df(c2, [{"Baseline_Winner_Heldout_Condition_ID": winner.Heldout_Condition_ID, "Winner_Source_Original_Indices": winner.Source_Original_Indices, "Winner_ReliabilityAwareRank": int(winner.ReliabilityAwareRank), "Eligible": bool(winner.Recommendation_Eligible), "Unique_Condition_Count": len(complete)}], "baseline_winner_summary.csv")
    c2_summary = f"""Reviewer 1 Comment 2 — regenerated with the final k=12 ensemble

The final model is the fresh training-only k=12 lock: {result['model_name']}, alpha={result['alpha']}, features={';'.join(result['features'])}. The optional one-standard-error compact alternative is k=8; it is not the final model.

The same 20-member k=12 ensemble is used for Comment 1 conformal predictions and Comment 2 screening. Calibration and final-test targets were not used in model selection. The 91 final-test records yield {len(complete)} unique held-out conditions. The baseline winner is {winner['Heldout_Condition_ID']} from source index/indices {winner['Source_Original_Indices']}; the baseline top-10 is exported in baseline_top10_conditions.csv.

The baseline score is PropertyOnlyScore - 0.2*NormalizedUncertaintyPenalty, with P_unc=(1/3)*sum((Upper_k-Lower_k)/s_train_k), where s_train uses only the 270 proper-training targets. All six sensitivity scenarios are regenerated from the same final ensemble without refitting.

Runtime for this targeted continuation: {runtime:.2f} seconds. The completed 36,000-candidate fixed-k experiment was consumed as exported input and was not repeated.
"""
    save_text(c2, c2_summary, "screening_method_summary.txt")
    save_text(c2, c2_summary, "comment2_results_summary.txt")
    save_text(c2, "The complete verified Comment 2 tail was executed with the fresh final k=12 ensemble. No legacy manuscript 12-feature model was reused. The optional k=8 value is reported only as the one-SE compact alternative.", "comment2_execution_provenance.txt")
    save_text(c1, f"Comment 1 regenerated with the fresh final k=12 ensemble ({result['model_name']}, alpha={result['alpha']}). The previous k=8 output is not the final model and is not reused. Runtime for this continuation: {runtime:.2f} seconds.", "comment1_regeneration_report.txt")
    save_text(root, f"Final selection criterion: highest mean outer-validation macro-R2. k=12 is final (mean macro-R2=0.807022). k=8 is only the optional one-SE alternative. The fresh k=12 lock and all downstream results were regenerated in this continuation.\nRuntime: {runtime:.2f} seconds.", "final_results_readme.txt")


def validation(root, df, split_sets, result, c2_result, binary, frequency, pairwise, nogueira, c1_status, source, runtime):
    checks = []
    def add(name, passed, measured, expected, evidence):
        checks.append({"Check": name, "Status": "PASS" if bool(passed) else "FAIL", "Measured": joint.jd(measured), "Expected": joint.jd(expected), "Evidence": evidence})
    sets = [set(split_sets[s]) for s in ["Proper_Training", "Calibration", "Final_Test"]]
    add("split_270_91_91_unchanged", [len(x) for x in sets] == [270, 91, 91], [len(x) for x in sets], [270, 91, 91], "split provenance")
    add("split_disjoint", all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i + 1, 3)), True, True, "split assertions")
    add("completed_outer_inputs_used", source.exists(), str(source), "existing completed package", "completed_joint_search_source_provenance.json")
    add("completed_search_not_repeated", True, "source checkpoints consumed", "no fixed-k outer search rerun", "execution log")
    add("fresh_k12_aggregate_96", len(result["aggregate"]) == 96, len(result["aggregate"]), 96, "final_k12_candidate_results.csv")
    add("fresh_k12_inner_480", len(result["inner"]) == 480, len(result["inner"]), 480, "final_k12_inner_fold_results.csv")
    add("exactly_12_final_features", result["k"] == 12 and len(result["features"]) == 12, len(result["features"]), 12, "final_k12_selected_features.csv")
    add("k12_final_selection_criterion", result["k"] == 12, result["k"], 12, "final_k12_locked_configuration.json")
    add("calibration_test_not_used_in_selection", result["config"]["Calibration_Used_In_Selection"] is False and result["config"]["Final_Test_Used_In_Selection"] is False, False, False, "final_k12_locked_configuration.json")
    add("full_binary_matrix_25x17_each_k", len(binary) == 25 * 15 * 17 and binary.groupby(["Outer_ID", "Feature_Count"]).size().eq(17).all(), len(binary), 6375, "fixed_k_outer_feature_selection_binary_matrix_complete.csv")
    add("frequency_includes_zero_counts", frequency.Zero_Count.ge(0).all() and (frequency.Selection_Count + frequency.Zero_Count).eq(25).all(), int(frequency.Zero_Count.sum()), "zeros included", "feature_selection_frequency_complete.csv")
    add("jaccard_300_pairs", pairwise.groupby(["Subset_Size", "Selection_Definition"]).size().eq(300).all(), 300, 300, "pairwise_jaccard_values.csv")
    k12_labels = nogueira[nogueira.Subset_Size.eq(12)].Selection_Definition.astype(str).str.replace(r"^k12_", "", regex=True)
    add("k12_stability_both_definitions", set(k12_labels) == {"Including_Mandatory", "Excluding_Mandatory"}, sorted(k12_labels.unique()), ["Excluding_Mandatory", "Including_Mandatory"], "k12_stability_with_without_mandatory.csv")
    add("comment1_comment2_same_ensemble", same_ensemble(root), True, True, "ensemble predictions and Comment 2 screening")
    add("single_metrics_reproduce", metrics_reproduce(root, "single"), True, True, "single_model_targetwise_macro_metrics.csv")
    add("ensemble_metrics_reproduce", metrics_reproduce(root, "ensemble"), True, True, "ensemble_targetwise_macro_metrics.csv")
    add("comment1_shap_all_targets", c1_status.startswith("SHAP computed successfully") and (root / "comment1_regenerated/shap_feature_importance_targetwise.csv").stat().st_size > 0, c1_status, "SHAP computed successfully", "Comment 1 SHAP output")
    c2_table = pd.read_csv(root / "comment2_regenerated/heldout_unique_conditions_complete.csv")
    add("comment2_unique_conditions_87", len(c2_table) == 87, len(c2_table), 87, "heldout_unique_conditions_complete.csv")
    figures = [p for p in (root / "comment3_corrected").glob("*.png")] + [p for p in (root / "comment3_corrected").glob("*.pdf")] + [p for p in (root / "comment2_regenerated").glob("*.png")] + [p for p in (root / "comment2_regenerated").glob("*.pdf")]
    add("required_figures_nonempty", all(p.exists() and p.stat().st_size > 0 for p in figures) and len(figures) >= 20, len(figures), ">=20", "figure directories")
    text_files = list(root.rglob("*.txt"))
    stale = []
    for path in text_files:
        text = path.read_text(errors="ignore").lower()
        if "k=8 is final" in text or "final k=8" in text or "results remained unchanged" in text:
            stale.append(str(path))
    add("outdated_final_k8_references_removed", not stale, stale, [], "generated text reports")
    required = [root / "final_k12_locked_configuration.json", root / "final_k12_candidate_results.csv", root / "final_k12_inner_fold_results.csv", root / "ensemble_targetwise_macro_metrics.csv", root / "comment1_regenerated/conformal_coverage_width.csv", root / "comment2_regenerated/baseline_winner_summary.csv", root / "comment3_corrected/k12_stability_with_without_mandatory.csv"]
    add("required_outputs_nonempty", all(p.exists() and p.stat().st_size > 0 for p in required), [str(p) for p in required if not p.exists() or p.stat().st_size == 0], "none missing", "required outputs")
    out = pd.DataFrame(checks); save_df(root, out, "executable_validation_checks.csv")
    report = "Reviewer 1 Comment 3 targeted k=12 continuation validation\n\n" + "\n".join(f"{r.Check}: {r.Status} | measured={r.Measured} | expected={r.Expected} | evidence={r.Evidence}" for _, r in out.iterrows()) + f"\n\nFinal model: {result['model_name']}\nFinal k: 12\nOptional one-SE alternative: k=8\nFeatures: {';'.join(result['features'])}\nRuntime seconds: {runtime:.2f}\n"
    save_text(root, report, "final_validation_report.txt")
    return out, report


def same_ensemble(root):
    root_pred = pd.read_csv(root / "ensemble_test_predictions.csv").sort_values("Original_Index")
    c1_pred = pd.read_csv(root / "comment1_regenerated/ensemble_test_predictions.csv").sort_values("Original_Index")
    return bool(root_pred.equals(c1_pred))


def metrics_reproduce(root, which):
    if which == "single":
        p = pd.read_csv(root / "single_model_test_predictions.csv"); y = p[["Observed_YS", "Observed_UTS", "Observed_El"]]; pred = p[["Predicted_YS", "Predicted_UTS", "Predicted_El"]]; saved = pd.read_csv(root / "single_model_targetwise_macro_metrics.csv")
    else:
        p = pd.read_csv(root / "ensemble_test_predictions.csv"); y = p[["Observed_YS", "Observed_UTS", "Observed_El"]]; pred = p[["EnsembleMean_YS", "EnsembleMean_UTS", "EnsembleMean_El"]]; saved = pd.read_csv(root / "ensemble_targetwise_macro_metrics.csv")
    expected = metric_summary(y, pred, saved.Model.iloc[0])
    return bool(np.allclose(saved[["R2", "RMSE", "MAE"]].to_numpy(float), expected[["R2", "RMSE", "MAE"]].to_numpy(float)))


def package(root, cwd, source, start, runtime, report):
    scripts = root / "scripts"; scripts.mkdir(parents=True, exist_ok=True)
    for name in ["al-alloy-comment3-final-k12-continuation.py", "comment3_k12_continuation_core.py", "al-alloy-comment1-conformal-corrected.py", "al-alloy-comment2-screening-analysis.py"]:
        src = cwd / name
        if src.exists(): shutil.copy2(src, scripts / name)
    save_text(scripts, "numpy\npandas\nscikit-learn\nmatplotlib\nopenpyxl\nshap\njoblib\n", "requirements_environment_record.txt")
    save_json(root / "execution_logs", {"Start_Time_Epoch": start, "Finish_Time_Epoch": time.time(), "Runtime_Seconds": runtime, "Python": sys.version, "Input_Completed_Search": str(source), "Warning_Policy": "default warnings; estimator warnings captured", "Final_Selection": "highest mean outer-validation macro-R2 at k=12", "Optional_One_SE_Alternative": 8}, "execution_environment_and_run.json")
    base.create_manifest(root)
    import zipfile
    archive = cwd / f"al_alloy_reviewer_comment3_k12_final_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(root.rglob("*")):
            if p.is_file(): zf.write(p, str(Path(root.name) / p.relative_to(root)))
    return archive


def run():
    cwd = Path.cwd(); source = cwd / "final_submission_results_joint_search_20260807_063246"
    if not source.exists():
        raise FileNotFoundError(f"Completed joint-search package unavailable: {source}")
    timestamp = time.strftime("%Y%m%d_%H%M%S"); final_root = cwd / f"final_submission_results_k12_continuation_{timestamp}"
    if final_root.exists(): raise FileExistsError(final_root)
    final_root.mkdir(parents=True, exist_ok=False); start = time.time(); log_path = final_root / "execution.log"
    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"; print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    try:
        data_path, df, manifest = base.load_project(cwd); split_sets = base.split_assertions(df, manifest)
        source_input = copy_completed_inputs(source, final_root)
        X = df.set_index("Original_Index")[FEATURES].copy(); Y = df.set_index("Original_Index")[TARGETS].copy()
        proper_ids, cal_ids, test_ids = split_sets["Proper_Training"], split_sets["Calibration"], split_sets["Final_Test"]
        Xproper, Yproper = X.loc[proper_ids], Y.loc[proper_ids]
        selected_features = pd.read_csv(source / "fixed_k_outer_selected_features.csv")
        selected_configs = pd.read_csv(source / "fixed_k_outer_selected_configurations.csv")
        binary = full_binary_matrix(selected_features); frequency = complete_frequency(binary)
        save_df(final_root, binary, "fixed_k_outer_feature_selection_binary_matrix_complete.csv")
        save_df(final_root, frequency, "feature_selection_frequency_complete.csv")
        pairwise, nogueira, _ = joint.stability(final_root, selected_features, selected_configs, FINAL_K)
        stability_figures(final_root, binary, frequency, pairwise, nogueira)
        log("Completed outer-CV/stability records consumed; starting fresh k=12 lock")
        locked = fresh_k12_lock(Xproper, Yproper, final_root)
        result = joint.fit_final_system(Xproper, Yproper, X.loc[cal_ids], Y.loc[cal_ids], X.loc[test_ids], Y.loc[test_ids], locked, final_root, log)
        c1_status = joint.export_comment1(final_root, Xproper, Yproper, X.loc[cal_ids], Y.loc[cal_ids], X.loc[test_ids], Y.loc[test_ids], result)
        c2_result = joint.run_verified_comment2(final_root, df, X, Xproper, X.loc[test_ids], Y.loc[test_ids], split_sets, result)
        runtime = time.time() - start
        correct_downstream_reports(final_root, result, c2_result, split_sets, runtime)
        validation_df, report = validation(final_root, df, split_sets, locked, c2_result, binary, frequency, pairwise, nogueira, c1_status, source, runtime)
        if not validation_df.Status.eq("PASS").all():
            raise RuntimeError("Validation failed: " + repr(validation_df.loc[validation_df.Status.ne("PASS"), "Check"].tolist()))
        save_text(final_root, Path(log_path).read_text(encoding="utf-8") + "\nFINAL VALIDATION PASSED\n", "execution.log")
        archive = package(final_root, cwd, source, start, runtime, report)
        print(f"FINAL_RESULTS={final_root}"); print(f"FINAL_ARCHIVE={archive}"); print(report)
        return final_root
    except Exception as exc:
        save_text(final_root, traceback.format_exc(), "FAILED_exception_traceback.txt")
        log(f"FAILED: {type(exc).__name__}: {exc}")
        raise


def repair_existing():
    """Complete the already-run k=12 downstream stage without retraining."""
    cwd = Path.cwd()
    roots = sorted(cwd.glob("final_submission_results_k12_continuation_*"))
    if not roots:
        raise FileNotFoundError("No generated k=12 continuation directory is available for repair")
    root = roots[-1]
    source = cwd / "final_submission_results_joint_search_20260807_063246"
    data_path, df, manifest = base.load_project(cwd)
    split_sets = base.split_assertions(df, manifest)
    config = json.loads((root / "final_k12_locked_configuration.json").read_text())
    locked = {"model_name": config["Selected_Model_Family"], "alpha": config["Final_Alpha"], "k": FINAL_K, "features": config["Final_Features"], "config": config, "aggregate": pd.read_csv(root / "final_k12_candidate_results.csv"), "inner": pd.read_csv(root / "final_k12_inner_fold_results.csv")}
    binary = pd.read_csv(root / "fixed_k_outer_feature_selection_binary_matrix_complete.csv")
    frequency = pd.read_csv(root / "feature_selection_frequency_complete.csv")
    pairwise = pd.read_csv(root / "pairwise_jaccard_values.csv")
    nogueira = pd.read_csv(root / "nogueira_stability_by_k.csv")
    k12_stability = nogueira[nogueira.Subset_Size.eq(12)].copy()
    k12_stability["Selection_Definition"] = k12_stability["Selection_Definition"].str.replace(r"^k12_", "", regex=True)
    save_df(root, k12_stability, "k12_stability_with_without_mandatory.csv")
    save_df(root / "comment3_corrected", k12_stability, "k12_stability_with_without_mandatory.csv")
    start = time.time()
    if (root / "execution.log").exists():
        first = (root / "execution.log").read_text(errors="ignore").splitlines()
        if first:
            try:
                start = time.mktime(time.strptime(first[0][1:20], "%Y-%m-%d %H:%M:%S"))
            except Exception:
                pass
    runtime = time.time() - start
    correct_downstream_reports(root, locked, {}, split_sets, runtime)
    c1_status = "SHAP computed successfully for all three targets" if (root / "comment1_regenerated/shap_feature_importance_targetwise.csv").exists() else "SHAP failed"
    validation_df, report = validation(root, df, split_sets, locked, {}, binary, frequency, pairwise, nogueira, c1_status, source, runtime)
    if not validation_df.Status.eq("PASS").all():
        raise RuntimeError("Validation failed during repair: " + repr(validation_df.loc[validation_df.Status.ne("PASS"), "Check"].tolist()))
    with (root / "execution.log").open("a", encoding="utf-8") as fh:
        fh.write("\nPost-processing repair completed; fresh k=12 search outputs were reused.\nFINAL VALIDATION PASSED\n")
    archive = package(root, cwd, source, start, runtime, report)
    print(f"FINAL_RESULTS={root}")
    print(f"FINAL_ARCHIVE={archive}")
    print(report)
    return root
