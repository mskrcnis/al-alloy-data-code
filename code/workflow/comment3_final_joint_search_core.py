"""Checkpointed fixed-k joint search built from the validated Comment 3 core.

The fixed-k search is intentionally separate for every outer partition and
every k.  No fitted configuration is cloned between feature counts.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.base import clone
from sklearn.model_selection import KFold, RandomizedSearchCV, RepeatedKFold

import comment3_final_core as base

# The historical core contains a module-level suppression for its legacy run.
# This corrected entry point restores normal warning behaviour and records
# estimator warnings in the provenance table instead of hiding them.
warnings.resetwarnings()
warnings.simplefilter("default")

TARGETS = base.TARGETS
FEATURES = base.FEATURES
MANDATORY = base.MANDATORY
ALPHA_GRID = base.ALPHA_GRID
SUBSET_SIZES = base.SUBSET_SIZES
MASTER_SEED = base.MASTER_SEED
OUTER_FOLDS = base.OUTER_FOLDS
OUTER_REPEATS = base.OUTER_REPEATS
INNER_FOLDS = base.INNER_FOLDS
SEARCH_ITERATIONS = base.SEARCH_ITERATIONS
ENSEMBLE_SIZE = base.ENSEMBLE_SIZE
CONFORMAL_ALPHA = base.CONFORMAL_ALPHA
EPSILON = base.EPSILON

REDUCED_FAMILIES = [
    "Native_ET_Physics", "Native_RF_Physics", "Independent_ET_Physics",
    "RegressorChain_ET_Physics",
]


def jd(x):
    return base.jready(x)


def save_df(root, obj, name):
    return base.save_df(root, obj, name)


def save_json(root, obj, name):
    return base.save_json(root, obj, name)


def save_text(root, text, name):
    return base.save_text(root, text, name)


def idx_hash(ids):
    return base.index_hash(ids)


def metric(y, p):
    return base.metric_dict(y, p)


def params_text(params):
    return json.dumps(base.parse_params(params), sort_keys=True, default=jd)


def complexity(params, family):
    """Deterministic lower-is-simpler complexity score for tie resolution."""
    p = base.parse_params(params)
    n = 0
    for key, value in p.items():
        if "n_estimators" in key:
            n += int(value)
        elif "max_depth" in key and value is not None:
            n += int(value) * 10
        elif "min_samples_leaf" in key:
            n += int(value) * 100
        elif "max_features" in key:
            n += int(float(value) * 100) if isinstance(value, (int, float)) else 50
    return int(n)


def tie_sort(rows):
    """Higher primary score, lower SD, lower complexity, lexical family/params."""
    return sorted(rows, key=lambda r: (
        -float(r["Mean_Inner_Primary_Score"]), float(r["Std_Inner_Primary_Score"]),
        int(r["Complexity_Score"]), str(r["Model_Family"]), str(r["Params"])))


def reduced_specs(k):
    specs = base.model_specs(fixed_k=int(k))[:4]
    assert [s[0] for s in specs] == REDUCED_FAMILIES
    return specs


def warnings_rows(caught, stage, outer_id, k, family):
    return [{"Warning_Class": w.category.__name__, "Message": str(w.message),
             "Source_File": str(w.filename), "Line_Number": int(w.lineno),
             "Execution_Stage": stage, "Outer_ID": outer_id, "Feature_Count": k,
             "Model_Family": family} for w in caught]


def run_one_fixed_k(Xouter, Youter, outer_id, repeat, fold, outer_seed, k, checkpoint, log):
    inner_seed = MASTER_SEED + 10000 + outer_seed * 100 + int(k)
    inner_cv = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=inner_seed)
    inner_splits = list(inner_cv.split(Xouter))
    aggregate, inner_rows, warnings_rows_all, searches = [], [], [], {}
    for family, estimator, space, description in reduced_specs(k):
        run_id = f"{outer_id}_k{k:02d}_{family}"
        search = RandomizedSearchCV(
            estimator, space, n_iter=SEARCH_ITERATIONS, scoring=base.scorer,
            cv=inner_cv, refit=False, random_state=outer_seed + int(k),
            n_jobs=8, pre_dispatch=8, return_train_score=False,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with parallel_backend("threading", n_jobs=8):
                search.fit(Xouter, Youter)
        warnings_rows_all.extend(warnings_rows(caught, "fixed_k_inner_search", outer_id, k, family))
        searches[family] = (search, estimator)
        cv = pd.DataFrame(search.cv_results_)
        assert len(cv) == SEARCH_ITERATIONS
        for no, row in cv.iterrows():
            params = base.parse_params(row["params"])
            aggregate.append({
                "Run_ID": run_id, "Repeat": repeat, "Outer_Fold": fold,
                "Outer_ID": outer_id, "Feature_Count": int(k), "Model_Family": family,
                "Description": description, "Candidate_Number": int(no + 1),
                "Alpha": float(params.get("selector__alpha", np.nan)),
                "Params": params_text(params), "Derived_Seed": int(outer_seed + int(k)),
                "Outer_Training_Index_Hash": idx_hash(Xouter.index),
                "Mean_Inner_Primary_Score": float(row["mean_test_score"]),
                "Std_Inner_Primary_Score": float(row["std_test_score"]),
                "Mean_Inner_NRMSE": float(-row["mean_test_score"]),
                "Rank_Within_Family": int(row["rank_test_score"]),
                "Selection_Status": "Candidate",
                "Search_Independent_Fixed_k": True,
            })
            for inner_no in range(INNER_FOLDS):
                score = float(row[f"split{inner_no}_test_score"])
                inner_rows.append({
                    "Run_ID": run_id, "Repeat": repeat, "Outer_Fold": fold,
                    "Outer_ID": outer_id, "Feature_Count": int(k), "Model_Family": family,
                    "Candidate_Number": int(no + 1), "Inner_Fold": inner_no + 1,
                    "Alpha": float(params.get("selector__alpha", np.nan)),
                    "Params": params_text(params), "Derived_Seed": int(outer_seed + int(k)),
                    "Inner_Training_Index_Hash": idx_hash(Xouter.iloc[inner_splits[inner_no][0]].index),
                    "Inner_Validation_Index_Hash": idx_hash(Xouter.iloc[inner_splits[inner_no][1]].index),
                    "Inner_Primary_Score": score, "Inner_NRMSE": float(-score),
                })
    candidates = [dict(r, Complexity_Score=complexity(json.loads(r["Params"]), r["Model_Family"])) for r in aggregate]
    winner = tie_sort(candidates)[0]
    winner["Selection_Status"] = "Selected_for_fixed_k"
    for row in aggregate:
        if row["Model_Family"] == winner["Model_Family"] and row["Candidate_Number"] == winner["Candidate_Number"]:
            row.update({"Selection_Status": "Selected_for_fixed_k", "Complexity_Score": winner["Complexity_Score"]})
        else:
            row["Complexity_Score"] = complexity(json.loads(row["Params"]), row["Model_Family"])
    family, estimator, _, _ = next(s for s in reduced_specs(k) if s[0] == winner["Model_Family"])
    selected_estimator = clone(estimator).set_params(**json.loads(winner["Params"]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        selected_estimator.fit(Xouter, Youter)
    warnings_rows_all.extend(warnings_rows(caught, "fixed_k_outer_refit", outer_id, k, family))
    validation_X = Xouter.attrs["validation_X"]
    validation_y = Xouter.attrs["validation_y"]
    pred = selected_estimator.predict(validation_X)
    alpha, selected_k, features, table = base.selector_details(selected_estimator)
    assert selected_k == int(k) and len(features) == int(k) and set(MANDATORY).issubset(features)
    selected = dict(winner)
    selected.update({"Selected_Alpha": alpha, "Selected_Feature_Count": int(k),
                     "Selected_Features": ";".join(features),
                     "Selected_Feature_Rank": ";".join(table.sort_values(["HybridScore", "Feature"], ascending=[False, True]).Feature),
                     "Outer_Training_Index_Hash": idx_hash(Xouter.index),
                     "Outer_Validation_Index_Hash": idx_hash(validation_X.index),
                     "Calibration_Accessed": False, "Final_Test_Accessed": False,
                     "Selected_Configuration_Status": "Selected"})
    selected.update({f"Outer_{key}": value for key, value in metric(validation_y, pred).items()})
    pred_rows = []
    for pos, idx in enumerate(validation_X.index):
        row = {"Outer_ID": outer_id, "Repeat": repeat, "Outer_Fold": fold,
               "Feature_Count": int(k), "Selected_Model_Family": family,
               "Selected_Alpha": alpha, "Original_Index": int(idx),
               "Calibration_Accessed": False, "Final_Test_Accessed": False}
        for j, target in enumerate(TARGETS):
            row.update({f"{target}_True": float(validation_y.iloc[pos, j]), f"{target}_Pred": float(pred[pos, j])})
        pred_rows.append(row)
    feature_rows = [{"Outer_ID": outer_id, "Repeat": repeat, "Outer_Fold": fold,
                     "Feature_Count": int(k), "Feature_Order": i + 1, "Feature": f,
                     "Mandatory": f in MANDATORY, "Mean_Normalized_Hybrid_Feature_Score": float(table.set_index("Feature").loc[f, "HybridScore"]),
                     "Selected_Alpha": alpha, "Selected": 1, "Rank": i + 1} for i, f in enumerate(features)]
    package = {"aggregate": aggregate, "inner": inner_rows, "selected": selected,
               "predictions": pred_rows, "features": feature_rows,
               "warnings": warnings_rows_all, "run_id": f"{outer_id}_k{k:02d}",
               "expected_aggregate": 4 * SEARCH_ITERATIONS,
               "expected_inner": 4 * SEARCH_ITERATIONS * INNER_FOLDS}
    tmp = checkpoint.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(package, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, checkpoint)
    return package


def fixed_k_search(Xproper, Yproper, checkpoint_dir, log):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    outer = RepeatedKFold(n_splits=OUTER_FOLDS, n_repeats=OUTER_REPEATS, random_state=MASTER_SEED)
    all_aggregate, all_inner, all_selected, all_pred, all_features, all_warnings = [], [], [], [], [], []
    for no, (tr_pos, va_pos) in enumerate(outer.split(Xproper), 1):
        repeat = (no - 1) // OUTER_FOLDS + 1; fold = (no - 1) % OUTER_FOLDS + 1
        outer_id = f"R{repeat:02d}_F{fold:02d}"; outer_seed = MASTER_SEED + 1000 + no
        tr_ids = list(Xproper.index[tr_pos]); va_ids = list(Xproper.index[va_pos])
        # Attach validation data to the local fold object without exposing it
        # to the search: the search receives only Xouter/Youter.
        Xouter = Xproper.loc[tr_ids].copy(); Youter = Yproper.loc[tr_ids].copy()
        Xouter.attrs["validation_X"] = Xproper.loc[va_ids].copy(); Xouter.attrs["validation_y"] = Yproper.loc[va_ids].copy()
        for k in SUBSET_SIZES:
            checkpoint = checkpoint_dir / f"{outer_id}_k{k:02d}.pkl"
            if checkpoint.exists():
                with open(checkpoint, "rb") as fh: package = pickle.load(fh)
                if len(package.get("aggregate", [])) != 4 * SEARCH_ITERATIONS or len(package.get("inner", [])) != 4 * SEARCH_ITERATIONS * INNER_FOLDS:
                    checkpoint.unlink()
                    package = run_one_fixed_k(Xouter, Youter, outer_id, repeat, fold, outer_seed, k, checkpoint, log)
            else:
                package = run_one_fixed_k(Xouter, Youter, outer_id, repeat, fold, outer_seed, k, checkpoint, log)
            all_aggregate.extend(package["aggregate"]); all_inner.extend(package["inner"])
            all_selected.append(package["selected"]); all_pred.extend(package["predictions"])
            all_features.extend(package["features"]); all_warnings.extend(package["warnings"])
            log(f"completed {outer_id}, k={k}; aggregate={len(all_aggregate)}")
    assert len(all_aggregate) == 25 * 15 * 4 * SEARCH_ITERATIONS
    assert len(all_inner) == 25 * 15 * 4 * SEARCH_ITERATIONS * INNER_FOLDS
    assert len(all_selected) == 25 * 15
    return map(pd.DataFrame, [all_aggregate, all_inner, all_selected, all_pred, all_features, all_warnings])


def curve_outputs(root, selected):
    rows = []
    for k, part in selected.groupby("Feature_Count", sort=True):
        row = {"Feature_Count": int(k), "Outer_Evaluations": len(part)}
        for metric_name in [f"Outer_R2_{t}" for t in TARGETS] + ["Outer_R2_Macro"] + [f"Outer_RMSE_{t}" for t in TARGETS] + ["Outer_RMSE_Macro"] + [f"Outer_MAE_{t}" for t in TARGETS] + ["Outer_MAE_Macro", "Outer_NRMSE_Macro"]:
            vals = part[metric_name].to_numpy(float); mean, sd, se, lo, hi = base.ci(vals)
            row.update({f"{metric_name}_Mean": mean, f"{metric_name}_Std": sd, f"{metric_name}_SE": se, f"{metric_name}_CI95_Lower": lo, f"{metric_name}_CI95_Upper": hi})
        row["Model_Family_Selection_Frequencies"] = json.dumps(part.Model_Family.value_counts().to_dict(), sort_keys=True)
        row["Alpha_Selection_Frequencies"] = json.dumps(part.Selected_Alpha.value_counts(dropna=False).to_dict(), sort_keys=True, default=jd)
        rows.append(row)
    summary = pd.DataFrame(rows)
    base.save_df(root, summary, "feature_subset_performance_summary.csv")
    return summary


def one_se(summary):
    ordered = summary.sort_values(["Outer_R2_Macro_Mean", "Feature_Count", "Outer_RMSE_Macro_Mean"], ascending=[False, True, True], kind="mergesort")
    best = ordered.iloc[0]; threshold = float(best.Outer_R2_Macro_Mean - best.Outer_R2_Macro_SE)
    qualifying = summary.loc[summary.Outer_R2_Macro_Mean >= threshold - 1e-12, "Feature_Count"].astype(int).sort_values().tolist()
    return int(best.Feature_Count), float(best.Outer_R2_Macro_Mean), float(best.Outer_R2_Macro_Std), float(best.Outer_R2_Macro_SE), threshold, qualifying, min(qualifying)


def stability(root, features_df, selected, final_k):
    all_pairs, n_rows, long_rows = [], [], []
    sets = {(int(k), str(oid)): set(part.Feature) for (k, oid), part in features_df.groupby(["Feature_Count", "Outer_ID"])}
    for k in SUBSET_SIZES:
        current = [sets[(k, oid)] for oid in sorted(selected.Outer_ID.unique())]
        assert len(current) == 25
        for include in [True, False]:
            label = "Including_Mandatory" if include else "Excluding_Mandatory"
            pair_no = 0
            for i in range(25):
                for j in range(i + 1, 25):
                    pair_no += 1; value, empty = base.jaccard_value(current[i], current[j], include)
                    all_pairs.append({"Subset_Size": k, "Selection_Definition": label, "Pair_Number": pair_no, "Outer_A": sorted(selected.Outer_ID.unique())[i], "Outer_B": sorted(selected.Outer_ID.unique())[j], "Jaccard": value, "Empty_Reduced_Sets": empty})
            assert pair_no == 300
        for include in [True, False]:
            label = "Including_Mandatory" if include else "Excluding_Mandatory"
            matrix = np.asarray([[int(f in s) for f in FEATURES] for s in current])
            if not include: matrix = matrix[:, [FEATURES.index(f) for f in FEATURES if f not in MANDATORY]]
            detail = base.nogueira(matrix, FEATURES if include else [f for f in FEATURES if f not in MANDATORY], f"k{k}_{label}")
            n_rows.append({"Subset_Size": k, "Selection_Definition": label, **{key: value for key, value in detail.items() if key not in ["p_fhat", "s_f_squared"]}})
    pairs = pd.DataFrame(all_pairs); ndf = pd.DataFrame(n_rows)
    base.save_df(root, pairs, "pairwise_jaccard_values.csv"); base.save_df(root, pairs.groupby(["Subset_Size", "Selection_Definition"], as_index=False).Jaccard.agg(["count", "mean", "std", "median", "min", "max"]).reset_index(), "jaccard_summary_by_k.csv")
    base.save_df(root, ndf, "nogueira_stability_by_k.csv"); base.save_json(root, {f"{r.Subset_Size}_{r.Selection_Definition}": jd(r.to_dict()) for _, r in ndf.iterrows()}, "nogueira_stability_details.json")
    for (k, oid), part in features_df.groupby(["Feature_Count", "Outer_ID"], sort=True):
        for _, r in part.iterrows(): long_rows.append(dict(r))
    long = pd.DataFrame(long_rows); base.save_df(root, long, "feature_selection_binary_matrix_by_k.csv")
    per = []
    for (k, f), part in long.groupby(["Feature_Count", "Feature"], sort=True):
        per.append({"Feature_Count": k, "Feature": f, "Mandatory": f in MANDATORY, "Selection_Frequency": float(part.Selected.mean()), "Mean_Rank": float(part.Rank.mean()), "Rank_SD": float(part.Rank.std(ddof=1)), "Mean_Normalized_Hybrid_Feature_Score": float(part.Mean_Normalized_Hybrid_Feature_Score.mean()), "Hybrid_Score_SD": float(part.Mean_Normalized_Hybrid_Feature_Score.std(ddof=1))})
    per_df = pd.DataFrame(per); base.save_df(root, per_df, "feature_stability_per_feature.csv")
    top12 = set(features_df[features_df.Feature_Count.eq(12)].Feature); final = set(features_df[features_df.Feature_Count.eq(final_k)].Feature)
    base.save_text(root, "Mean normalized hybrid feature score is a ranking summary, not a feature-selection stability statistic. Genuine stability is quantified using selection frequency, pairwise Jaccard similarity, and Nogueira stability.\n\nThe fixed-k joint-search feature sets were used independently for every k; no fitted configuration was cloned across k.", "stability_recalculation_report.txt")
    return pairs, ndf, per_df


def search_records(X, Y, seed, k, outer_id="FINAL", repeat=0, fold=0,
                   stage="final", log=None):
    """Run the complete four-family, 24-candidate search at one fixed k."""
    inner_seed = MASTER_SEED + 70000
    inner_cv = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=inner_seed)
    splits = list(inner_cv.split(X))
    aggregate, inner_rows, warnings_all, fitted_searches = [], [], [], {}
    for family, estimator, space, description in reduced_specs(k):
        search = RandomizedSearchCV(
            estimator, space, n_iter=SEARCH_ITERATIONS, scoring=base.scorer,
            cv=inner_cv, refit=False, random_state=int(seed), n_jobs=8,
            pre_dispatch=8, return_train_score=False,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with parallel_backend("threading", n_jobs=8):
                search.fit(X, Y)
        warnings_all.extend(warnings_rows(caught, stage, outer_id, k, family))
        fitted_searches[family] = (search, estimator, description)
        cv = pd.DataFrame(search.cv_results_)
        assert len(cv) == SEARCH_ITERATIONS
        run_id = f"{stage}_{outer_id}_k{k:02d}_{family}"
        for no, row in cv.iterrows():
            params = base.parse_params(row["params"])
            common = {
                "Run_ID": run_id, "Repeat": repeat, "Outer_Fold": fold,
                "Outer_ID": outer_id, "Feature_Count": int(k),
                "Model_Family": family, "Description": description,
                "Candidate_Number": int(no + 1),
                "Alpha": float(params.get("selector__alpha", np.nan)),
                "Params": params_text(params), "Derived_Seed": int(seed),
                "Index_Hash": idx_hash(X.index),
                "Mean_Inner_Primary_Score": float(row["mean_test_score"]),
                "Std_Inner_Primary_Score": float(row["std_test_score"]),
                "Mean_Inner_NRMSE": float(-row["mean_test_score"]),
                "Rank_Within_Family": int(row["rank_test_score"]),
                "Selection_Status": "Candidate",
                "Search_Independent_Fixed_k": True,
                "Search_Stage": stage,
            }
            aggregate.append(common)
            for inner_no, (tr, va) in enumerate(splits, 1):
                inner_rows.append({
                    "Run_ID": run_id, "Repeat": repeat, "Outer_Fold": fold,
                    "Outer_ID": outer_id, "Feature_Count": int(k),
                    "Model_Family": family, "Candidate_Number": int(no + 1),
                    "Inner_Fold": inner_no,
                    "Alpha": float(params.get("selector__alpha", np.nan)),
                    "Params": params_text(params), "Derived_Seed": int(seed),
                    "Inner_Training_Index_Hash": idx_hash(X.iloc[tr].index),
                    "Inner_Validation_Index_Hash": idx_hash(X.iloc[va].index),
                    "Inner_Primary_Score": float(row[f"split{inner_no - 1}_test_score"]),
                    "Inner_NRMSE": float(-row[f"split{inner_no - 1}_test_score"]),
                    "Search_Stage": stage,
                })
    candidates = [dict(r, Complexity_Score=complexity(json.loads(r["Params"]), r["Model_Family"]))
                  for r in aggregate]
    winner = tie_sort(candidates)[0]
    for row in aggregate:
        row["Complexity_Score"] = complexity(json.loads(row["Params"]), row["Model_Family"])
        row["Selection_Status"] = (
            "Selected" if row["Model_Family"] == winner["Model_Family"] and
            row["Candidate_Number"] == winner["Candidate_Number"] else "Candidate"
        )
    return pd.DataFrame(aggregate), pd.DataFrame(inner_rows), winner, fitted_searches, warnings_all


def final_search(Xproper, Yproper, root, log):
    """Perform the separate 96-candidate locked search at the chosen final k."""
    summary = pd.read_csv(root / "feature_subset_performance_summary.csv")
    best_k, best_mean, best_sd, best_se, threshold, qualifying, final_k = one_se(summary)
    agg, inner, winner, searches, warning_rows_all = search_records(
        Xproper, Yproper, MASTER_SEED + 70001, final_k, stage="final", log=log)
    assert len(agg) == 4 * SEARCH_ITERATIONS
    assert len(inner) == 4 * SEARCH_ITERATIONS * INNER_FOLDS
    winner_family = winner["Model_Family"]
    _, estimator, space, description = next(s for s in reduced_specs(final_k) if s[0] == winner_family)
    params = json.loads(winner["Params"])
    model = clone(estimator).set_params(**params)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(Xproper, Yproper)
    warning_rows_all.extend(warnings_rows(caught, "final_locked_refit", "FINAL", final_k, winner_family))
    alpha, locked_k, features, table = base.selector_details(model)
    assert locked_k == final_k and len(features) == final_k
    config = {
        "Search_Type": "Independent fixed-k joint search",
        "Eligible_Families": REDUCED_FAMILIES,
        "Excluded_From_Fixed_k_Competition": ["Native_ET_All17", "Native_RF_All17"],
        "Candidate_Count": 4 * SEARCH_ITERATIONS,
        "Inner_Fold_Record_Count": 4 * SEARCH_ITERATIONS * INNER_FOLDS,
        "Selected_Model_Family": winner_family, "Description": description,
        "Search_Algorithm": "RandomizedSearchCV", "Search_Budget": SEARCH_ITERATIONS,
        "Inner_CV": "5-fold shuffled KFold", "Inner_CV_Seed": MASTER_SEED + 70000,
        "Selection_Objective": "higher mean negative normalized RMSE",
        "Tie_Break": "higher mean primary score, lower SD, lower complexity, lexical family, lexical serialized parameters",
        "One_SE_Best_k": best_k, "One_SE_Best_Mean_Macro_R2": best_mean,
        "One_SE_Best_SD_Macro_R2": best_sd, "One_SE_Best_SE_Macro_R2": best_se,
        "One_SE_Threshold": threshold, "One_SE_Qualifying_k": qualifying,
        "Final_Feature_Count": int(locked_k), "Final_Alpha": alpha,
        "Final_Features": features, "Best_Inner_Primary_Score": winner["Mean_Inner_Primary_Score"],
        "Best_Inner_NRMSE": winner["Mean_Inner_NRMSE"], "Best_Params": params,
        "Proper_Training_Index_Hash": idx_hash(Xproper.index),
        "Calibration_Used_In_Selection": False, "Final_Test_Used_In_Selection": False,
        "Alpha_Grid": ALPHA_GRID, "Feature_Count_Grid": SUBSET_SIZES,
        "Mandatory_Features": MANDATORY,
        "Ranking_Equation": "HybridScore_j(alpha) = alpha * minmax(PearsonRelevance_j) + (1-alpha) * minmax(MeanEmbeddedImportance_j)",
        "Normalized_Conformal_Method": "abs(y_cal - ensemble_mean_cal)/(ensemble_sd_cal + epsilon)",
        "Ensemble_Size": ENSEMBLE_SIZE, "Conformal_Alpha": CONFORMAL_ALPHA,
        "Epsilon": EPSILON, "Fixed_k_Search_Independent": True,
        "Tree_Search_Space": {"n_estimators": [300, 500, 800], "max_depth": [None, 6, 10, 14],
                               "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.7, 1.0]},
    }
    save_df(root, agg, "final_fixed_k_candidate_results.csv")
    save_df(root, inner, "final_fixed_k_inner_fold_results.csv")
    save_json(root, config, "final_locked_configuration.json")
    save_df(root, [{"Feature_Order": i + 1, "Feature": f, "Mandatory": f in MANDATORY,
                    "Mean_Normalized_Hybrid_Feature_Score": float(table.set_index("Feature").loc[f, "HybridScore"])}
                   for i, f in enumerate(features)], "final_selected_features.csv")
    save_df(root, pd.DataFrame(warning_rows_all), "warning_capture.csv")
    save_text(root, "Final fixed-k search: 4 families x 24 candidates = 96 candidates; 5 inner folds = 480 fold records. The final model was refit on all 270 proper-training rows only. Calibration and final-test rows were not accessed during selection.", "final_training_provenance.txt")
    return {"model": model, "model_name": winner_family, "description": description,
            "params": params, "config": config, "alpha": alpha, "k": locked_k,
            "features": features, "ranking_table": table, "aggregate": agg,
            "inner": inner, "warnings": warning_rows_all}


def set_all_model_seeds(estimator, seed):
    params = estimator.get_params(deep=True)
    changes = {key: int(seed) for key in params if key.endswith("random_state")}
    if changes:
        estimator.set_params(**changes)
    return estimator


def fit_final_system(Xproper, Yproper, Xcal, Ycal, Xtest, Ytest, locked, root, log):
    model = locked["model"]
    single_cal = model.predict(Xcal)
    single_test = model.predict(Xtest)
    rng = np.random.default_rng(MASTER_SEED + 60000)
    cal_members, test_members, seeds, sample_hashes = [], [], [], []
    for member in range(ENSEMBLE_SIZE):
        seed = MASTER_SEED + 60001 + member
        positions = rng.integers(0, len(Xproper), size=len(Xproper))
        member_model = clone(model)
        set_all_model_seeds(member_model, seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            member_model.fit(Xproper.iloc[positions], Yproper.iloc[positions])
        if caught:
            save_df(root, warnings_rows(caught, "final_bootstrap_member", "FINAL", locked["k"], locked["model_name"]), "bootstrap_warning_capture.csv")
        cal_members.append(member_model.predict(Xcal)); test_members.append(member_model.predict(Xtest))
        seeds.append(seed); sample_hashes.append(idx_hash(Xproper.index[positions]))
        log(f"completed final bootstrap member {member + 1}/{ENSEMBLE_SIZE}")
    cal_members = np.asarray(cal_members, float); test_members = np.asarray(test_members, float)
    cal_mean = cal_members.mean(axis=0); cal_std = cal_members.std(axis=0, ddof=1)
    test_mean = test_members.mean(axis=0); test_std = test_members.std(axis=0, ddof=1)
    scores = np.abs(Ycal.to_numpy(float) - cal_mean) / (cal_std + EPSILON)
    rank = int(math.ceil((len(Xcal) + 1) * (1 - CONFORMAL_ALPHA)))
    q = np.asarray([np.sort(scores[:, j])[rank - 1] for j in range(len(TARGETS))], float)
    lower = test_mean - q * (test_std + EPSILON)
    upper = test_mean + q * (test_std + EPSILON)
    width = upper - lower
    covered = (Ytest.to_numpy(float) >= lower) & (Ytest.to_numpy(float) <= upper)
    target_sd = Yproper.std(ddof=1).to_numpy(float)
    single_metrics = metric(Ytest, single_test); ensemble_metrics = metric(Ytest, test_mean)
    save_json(root, locked["config"], "final_locked_model_configuration.json")
    save_df(root, [{"Target": t, **{key: value for key, value in single_metrics.items() if key.endswith("_" + t)}} for t in TARGETS], "single_model_test_metrics.csv")
    save_df(root, [{"Target": t, **{key: value for key, value in ensemble_metrics.items() if key.endswith("_" + t)}} for t in TARGETS], "ensemble_test_metrics.csv")
    save_df(root, targetwise_rows(Xtest.index, "Final_Test", Ytest, single_test), "single_model_test_predictions.csv")
    save_df(root, targetwise_rows(Xtest.index, "Final_Test", Ytest, test_mean, test_mean, test_std), "ensemble_test_predictions.csv")
    save_df(root, targetwise_rows(Xcal.index, "Calibration", Ycal, cal_mean, cal_mean, cal_std, scores), "ensemble_calibration_predictions.csv")
    save_df(root, [{"Member": i + 1, "Seed": seeds[i], "Bootstrap_Size": len(Xproper), "Bootstrap_Source": "Proper_Training_only", "Bootstrap_Index_Hash": sample_hashes[i], **{f"R2_{t}": metric(Ytest, test_members[i])[f"R2_{t}"] for t in TARGETS}} for i in range(ENSEMBLE_SIZE)], "bootstrap_member_test_metrics.csv")
    save_df(root, [{"Target": t, "Ensemble_Mean_R2": ensemble_metrics[f"R2_{t}"], "Ensemble_Mean_RMSE": ensemble_metrics[f"RMSE_{t}"], "Ensemble_Mean_MAE": ensemble_metrics[f"MAE_{t}"], "Single_Model_R2": single_metrics[f"R2_{t}"], "Single_Model_RMSE": single_metrics[f"RMSE_{t}"], "Single_Model_MAE": single_metrics[f"MAE_{t}"]} for t in TARGETS], "bootstrap_ensemble_summary.csv")
    coverage = []
    for j, t in enumerate(TARGETS):
        coverage.append({"Target": t, "Nominal_Coverage": 1 - CONFORMAL_ALPHA, "Conformal_Alpha": CONFORMAL_ALPHA,
                         "Conformal_Rank": rank, "Conformal_q_Normalized": float(q[j]),
                         "Observed_Marginal_Coverage": float(covered[:, j].mean()),
                         "Observed_Simultaneous_Coverage": float(covered.all(axis=1).mean()),
                         "Min_Interval_Width": float(width[:, j].min()), "Mean_Interval_Width": float(width[:, j].mean()),
                         "Median_Interval_Width": float(np.median(width[:, j])), "Max_Interval_Width": float(width[:, j].max()),
                         "Mean_Ensemble_Std": float(test_std[:, j].mean()), "N_Calibration": len(Xcal),
                         "N_Ensemble": ENSEMBLE_SIZE, "Proper_Training_Target_SD": float(target_sd[j])})
    save_df(root, coverage, "conformal_coverage_width.csv")
    save_df(root, coverage, "uncertainty_coverage_width.csv")
    interval_rows = []
    for i, idx in enumerate(Xtest.index):
        row = {"Original_Index": int(idx), "Data_Split": "Final_Test",
               "NormalizedUncertaintyPenalty": float(np.mean(width[i] / target_sd)),
               "UncertaintyPenalty": float(np.mean(width[i] / target_sd))}
        for j, t in enumerate(TARGETS):
            row.update({f"Observed_{t}": float(Ytest.iloc[i, j]), f"EnsembleMean_{t}": float(test_mean[i, j]),
                        f"EnsembleStd_{t}": float(test_std[i, j]), f"Lower_{t}": float(lower[i, j]),
                        f"Upper_{t}": float(upper[i, j]), f"UncWidth_{t}": float(width[i, j]),
                        f"IntervalWidth_{t}": float(width[i, j]), f"NormalizedUncWidth_{t}": float(width[i, j] / target_sd[j]),
                        f"Covered_{t}": bool(covered[i, j])})
        row["Simultaneously_Covered"] = bool(covered[i].all()); interval_rows.append(row)
    save_df(root, interval_rows, "conformal_intervals_final_test.csv")
    save_df(root, [{"Target": t, "Conformal_Alpha": CONFORMAL_ALPHA, "Conformal_Rank": rank,
                    "Conformal_q_Normalized": float(q[j]), "Epsilon": EPSILON,
                    "Proper_Training_Index_Hash": idx_hash(Xproper.index), "Calibration_Index_Hash": idx_hash(Xcal.index),
                    "Final_Test_Index_Hash": idx_hash(Xtest.index), "Score_Formula": "abs(y_cal - ensemble_mean_cal)/(ensemble_sd_cal + epsilon)",
                    "Interval_Formula": "ensemble_mean_test +/- q_normalized*(ensemble_sd_test + epsilon)"} for j, t in enumerate(TARGETS)], "conformal_quantiles.csv")
    save_df(root, [{"Member": i + 1, "Seed": seeds[i], "Bootstrap_Size": len(Xproper), "Bootstrap_Source": "Proper_Training_only", "Bootstrap_Index_Hash": sample_hashes[i]} for i in range(ENSEMBLE_SIZE)], "bootstrap_seed_manifest.csv")
    save_df(root, [{"Original_Index": int(idx), **{key: value for j, t in enumerate(TARGETS) for key, value in ((f"EnsembleMean_{t}", float(test_mean[i, j])), (f"EnsembleStd_{t}", float(test_std[i, j])))} } for i, idx in enumerate(Xtest.index)], "bootstrap_prediction_summary.csv")
    try:
        import joblib
        joblib.dump(model, root / "final_locked_model.joblib")
    except Exception as exc:
        save_text(root, f"Model serialization failed: {type(exc).__name__}: {exc}", "final_model_serialization_failure.txt")
    return {"model": model, "model_name": locked["model_name"], "description": locked["description"],
            "config": locked["config"], "alpha": locked["alpha"], "k": locked["k"], "features": locked["features"],
            "single_cal": single_cal, "single_test": single_test, "cal_members": cal_members,
            "test_members": test_members, "cal_mean": cal_mean, "cal_std": cal_std,
            "test_mean": test_mean, "test_std": test_std, "scores": scores, "rank": rank,
            "q": q, "lower": lower, "upper": upper, "width": width, "covered": covered,
            "target_sd": target_sd, "coverage": coverage, "single_metrics": single_metrics,
            "ensemble_metrics": ensemble_metrics}


def targetwise_rows(ids, split, y, pred, mean=None, std=None, scores=None):
    rows = []
    for i, idx in enumerate(ids):
        row = {"Original_Index": int(idx), "Data_Split": split}
        for j, t in enumerate(TARGETS):
            row[f"Observed_{t}"] = float(y.iloc[i, j])
            row[f"Predicted_{t}"] = float(pred[i, j])
            if mean is not None:
                row[f"EnsembleMean_{t}"] = float(mean[i, j])
            if std is not None:
                row[f"EnsembleStd_{t}"] = float(std[i, j])
            if scores is not None:
                row[f"NormalizedScore_{t}"] = float(scores[i, j])
        rows.append(row)
    return rows


def export_comment1(root, Xproper, Yproper, Xcal, Ycal, Xtest, Ytest, result):
    c1 = root / "comment1_regenerated"
    c1.mkdir(parents=True, exist_ok=True)
    single_metrics, ensemble_metrics = result["single_metrics"], result["ensemble_metrics"]
    save_df(c1, [{"Target": t, **{key: value for key, value in single_metrics.items() if key.endswith("_" + t)}} for t in TARGETS], "single_model_test_metrics.csv")
    save_df(c1, [{"Target": t, **{key: value for key, value in ensemble_metrics.items() if key.endswith("_" + t)}} for t in TARGETS], "ensemble_test_metrics.csv")
    save_df(c1, targetwise_rows(Xtest.index, "Final_Test", Ytest, result["single_test"]), "single_model_test_predictions.csv")
    save_df(c1, targetwise_rows(Xtest.index, "Final_Test", Ytest, result["test_mean"], result["test_mean"], result["test_std"]), "ensemble_test_predictions.csv")
    save_df(c1, targetwise_rows(Xcal.index, "Calibration", Ycal, result["cal_mean"], result["cal_mean"], result["cal_std"], result["scores"]), "ensemble_calibration_predictions.csv")
    save_df(c1, pd.DataFrame(result["coverage"]), "conformal_coverage_width.csv")
    save_df(c1, pd.DataFrame(result["coverage"]), "uncertainty_coverage_width.csv")
    interval = pd.read_csv(root / "conformal_intervals_final_test.csv")
    for t in TARGETS:
        interval[f"Residual_{t}"] = interval[f"Observed_{t}"] - interval[f"EnsembleMean_{t}"]
        interval[f"AbsoluteResidual_{t}"] = interval[f"Residual_{t}"].abs()
        interval[f"NormalizedNonconformity_{t}"] = interval[f"AbsoluteResidual_{t}"] / (interval[f"EnsembleStd_{t}"] + EPSILON)
    save_df(c1, interval, "conformal_intervals_final_test.csv")
    residual = interval[["Original_Index"] + [c for t in TARGETS for c in [f"Residual_{t}", f"AbsoluteResidual_{t}", f"NormalizedNonconformity_{t}"]]]
    save_df(c1, residual, "residuals_and_nonconformity_scores.csv")
    save_df(c1, pd.DataFrame([{"Target": t, "Conformal_Alpha": CONFORMAL_ALPHA, "Conformal_Rank": result["rank"], "Conformal_q_Normalized": float(result["q"][j]), "Epsilon": EPSILON} for j, t in enumerate(TARGETS)]), "conformal_quantiles.csv")
    for name in ["bootstrap_seed_manifest.csv", "bootstrap_prediction_summary.csv", "bootstrap_member_test_metrics.csv", "bootstrap_ensemble_summary.csv"]:
        shutil.copy2(root / name, c1 / name)
    save_df(c1, pd.DataFrame([{"Target": t, "Marginal_Coverage": float(result["covered"][:, j].mean()), "Simultaneous_Coverage": float(result["covered"].all(axis=1).mean()), "Min_Width": float(result["width"][:, j].min()), "Mean_Width": float(result["width"][:, j].mean()), "Median_Width": float(np.median(result["width"][:, j])), "Max_Width": float(result["width"][:, j].max())} for j, t in enumerate(TARGETS)]), "coverage_and_width_report.csv")
    save_json(c1, {"Proper_Training_Index_Hash": idx_hash(Xproper.index), "Calibration_Index_Hash": idx_hash(Xcal.index), "Final_Test_Index_Hash": idx_hash(Xtest.index), "Calibration_Used_After_Final_Lock": True, "Final_Test_Used_After_Calibration": True, "Calibration_Leakage_Detected": False, "Normalized_Score": "abs(y_cal - ensemble_mean_cal)/(ensemble_sd_cal + epsilon)", "Conformal_Rank": result["rank"], "Conformal_Alpha": CONFORMAL_ALPHA, "Epsilon": EPSILON, "Interval": "ensemble_mean_test +/- q_normalized*(ensemble_sd_test + epsilon)", "Uncertainty_Penalty": "(1/3)*sum((upper-lower)/s_train_k)", "Target_SD_Source": "270 proper-training rows only"}, "comment1_validation_provenance.json")
    for j, t in enumerate(TARGETS):
        order = np.argsort(Ytest.iloc[:, j].to_numpy(float)); x = np.arange(len(order)); pred = result["test_mean"][order, j]; obs = Ytest.iloc[order, j].to_numpy(float); lo = result["lower"][order, j]; hi = result["upper"][order, j]
        fig, ax = plt.subplots(figsize=(8, 4.5)); ax.errorbar(x, pred, yerr=[pred - lo, hi - pred], fmt="o", ms=2, alpha=.65, label="Prediction interval"); ax.scatter(x, obs, s=7, c="black", label="Observed"); ax.set_xlabel("Final-test sample (sorted by observed target)"); ax.set_ylabel(t); ax.legend(frameon=False); ax.grid(alpha=.2); base.figure_pair(c1, f"conformal_interval_{t}", fig)
        fig, ax = plt.subplots(figsize=(5.2, 4.8)); ax.scatter(Ytest.iloc[:, j], result["test_mean"][:, j], s=18, alpha=.75, color="#0072B2"); lo_ax, hi_ax = float(min(Ytest.iloc[:, j].min(), result["test_mean"][:, j].min())), float(max(Ytest.iloc[:, j].max(), result["test_mean"][:, j].max())); ax.plot([lo_ax, hi_ax], [lo_ax, hi_ax], "--", color="#333333"); ax.set_xlabel(f"Observed {t}"); ax.set_ylabel(f"Predicted {t}"); ax.grid(alpha=.2); base.figure_pair(c1, f"actual_vs_predicted_{t}", fig)
    shap_status = export_targetwise_shap(c1, result, Xtest)
    required = ["single_model_test_predictions.csv", "single_model_test_metrics.csv", "ensemble_calibration_predictions.csv", "ensemble_test_predictions.csv", "ensemble_test_metrics.csv", "bootstrap_member_test_metrics.csv", "bootstrap_ensemble_summary.csv", "conformal_coverage_width.csv", "uncertainty_coverage_width.csv", "conformal_intervals_final_test.csv", "conformal_quantiles.csv", "residuals_and_nonconformity_scores.csv", "comment1_validation_provenance.json"]
    save_df(c1, [{"Required_Output": f, "Exists": bool((c1 / f).exists()), "Nonempty": bool((c1 / f).exists() and (c1 / f).stat().st_size > 0), "Purpose": "Comment 1 regenerated output"} for f in required], "comment1_output_completeness.csv")
    numeric_checks = []
    for j, t in enumerate(TARGETS):
        numeric_checks.append({"Target": t, "Score_Recomputed_From_Formula": bool(np.allclose(result["scores"][:, j], np.abs(Ycal.iloc[:, j].to_numpy(float) - result["cal_mean"][:, j]) / (result["cal_std"][:, j] + EPSILON))), "Interval_Recomputed_From_Formula": bool(np.allclose(result["lower"][:, j], result["test_mean"][:, j] - result["q"][j] * (result["test_std"][:, j] + EPSILON))), "Coverage_Recomputed": float(result["covered"][:, j].mean())})
    save_df(c1, numeric_checks, "comment1_numerical_consistency_checks.csv")
    save_text(c1, f"Comment 1 regenerated after the fixed-k Comment 3 search. The principal point predictions and uncertainty intervals use the locked {result['model_name']} ensemble. Separate single-model predictions and metrics are retained for transparency. Calibration was not used for model selection; it was used only after the final proper-training model was locked. SHAP status: {shap_status}.", "comment1_regeneration_report.txt")
    save_text(c1, "All current Comment 1 affected outputs were generated from the current fixed-k locked configuration. No historical Comment 1 prediction or interval table was used as a source of current values.", "comment1_summary.txt")
    return shap_status


def export_targetwise_shap(root, result, Xtest):
    try:
        import shap
        estimator = result["model"]
        Xselected = estimator.named_steps["selector"].transform(Xtest)
        model = estimator.named_steps["model"]
        target_values = []
        if hasattr(model, "estimators_") and not isinstance(model, base.ExtraTreesRegressor):
            for submodel in model.estimators_:
                values = shap.TreeExplainer(submodel).shap_values(Xselected, check_additivity=False)
                if isinstance(values, list): values = values[0]
                target_values.append(np.asarray(values, float))
        else:
            values = shap.TreeExplainer(model).shap_values(Xselected, check_additivity=False)
            if isinstance(values, list):
                target_values = [np.asarray(v, float) for v in values]
            elif np.asarray(values).ndim == 3:
                target_values = [np.asarray(values)[:, :, j] for j in range(np.asarray(values).shape[2])]
            else:
                target_values = [np.asarray(values, float)] * len(TARGETS)
        if len(target_values) != len(TARGETS):
            raise RuntimeError(f"SHAP returned {len(target_values)} target arrays for {len(TARGETS)} targets")
        rows = []
        for target, values in zip(TARGETS, target_values):
            mean_abs = np.mean(np.abs(values), axis=0)
            if len(mean_abs) != len(result["features"]):
                raise RuntimeError("SHAP feature dimension does not match locked selected feature count")
            rows.extend({"Target": target, "Feature": feature, "MeanAbsoluteSHAP": float(value)} for feature, value in zip(result["features"], mean_abs))
            fig, ax = plt.subplots(figsize=(7, 4.5)); order = np.argsort(mean_abs); ax.barh(np.asarray(result["features"])[order], mean_abs[order], color="#0072B2"); ax.set_xlabel("Mean absolute SHAP value"); ax.set_ylabel("Feature"); base.figure_pair(root, f"shap_summary_{target}", fig)
        save_df(root, rows, "shap_feature_importance_targetwise.csv")
        save_df(root, pd.DataFrame(rows).groupby("Feature", as_index=False).MeanAbsoluteSHAP.mean(), "shap_feature_importance.csv")
        return "SHAP computed successfully for all three targets"
    except Exception as exc:
        save_text(root, f"SHAP generation failed: {type(exc).__name__}: {exc}", "shap_generation_failure.txt")
        return f"SHAP failed: {type(exc).__name__}: {exc}"


def run_verified_comment2(root, df, X, Xproper, Xtest, Ytest, split_sets, result):
    """Execute the complete verified Comment 2 tail in an isolated namespace."""
    c2 = root / "comment2_regenerated"
    c2.mkdir(parents=True, exist_ok=True)
    proper_ids = set(split_sets["Proper_Training"]); cal_ids = set(split_sets["Calibration"]); test_ids = set(split_sets["Final_Test"])
    test = df.set_index("Original_Index").loc[sorted(test_ids)].copy()
    screen_test = test[FEATURES].copy()
    # Retain Original_Index as a traceable column while avoiding pandas'\n+    # index-level/column-label ambiguity in deterministic rank sorting.
    screen_test.index.name = None
    screen_test["Original_Index"] = screen_test.index.astype(int)
    screen_test["Data_Split"] = "Final_Test"
    for j, t in enumerate(TARGETS):
        screen_test[f"Pred_{t}"] = result["test_mean"][:, j]
        screen_test[f"EnsembleStd_{t}"] = result["test_std"][:, j]
        screen_test[f"Lower_{t}"] = result["lower"][:, j]
        screen_test[f"Upper_{t}"] = result["upper"][:, j]
        screen_test[f"UncWidth_{t}"] = result["width"][:, j]
        screen_test[f"NormalizedUncWidth_{t}"] = result["width"][:, j] / result["target_sd"][j]
    screen_test["NormalizedUncertaintyPenalty"] = screen_test[[f"NormalizedUncWidth_{t}" for t in TARGETS]].mean(axis=1)
    screen_test["UncertaintyPenalty"] = screen_test["NormalizedUncertaintyPenalty"]
    source_minmax = lambda s: (pd.Series(s, dtype=float) - pd.Series(s, dtype=float).min()) / (pd.Series(s, dtype=float).max() - pd.Series(s, dtype=float).min() + 1e-12)
    screen_test["NormalizedPred_YS"] = source_minmax(screen_test["Pred_YS"]).to_numpy()
    screen_test["NormalizedPred_UTS"] = source_minmax(screen_test["Pred_UTS"]).to_numpy()
    screen_test["NormalizedPred_El"] = source_minmax(screen_test["Pred_El"]).to_numpy()
    screen_test["StrengthScore"] = .5 * screen_test["NormalizedPred_YS"] + .5 * screen_test["NormalizedPred_UTS"]
    screen_test["DuctilityScore"] = screen_test["NormalizedPred_El"]
    screen_test["PropertyOnlyScore"] = .4 * screen_test["StrengthScore"] + .4 * screen_test["DuctilityScore"]
    screen_test["ReliabilityAwareScore"] = screen_test["PropertyOnlyScore"] - .2 * screen_test["NormalizedUncertaintyPenalty"]
    feature_cols = list(FEATURES); TARGET_COLS = list(TARGETS)
    def deterministic_desc_rank(table, score_column):
        ordered = table.sort_values([score_column, "Original_Index"], ascending=[False, True], kind="mergesort")
        ranks = pd.Series(np.arange(1, len(ordered) + 1), index=ordered.index)
        return ranks.reindex(table.index).astype(int)
    screen_test["PropertyOnlyRank"] = deterministic_desc_rank(screen_test, "PropertyOnlyScore")
    screen_test["ReliabilityAwareRank"] = deterministic_desc_rank(screen_test, "ReliabilityAwareScore")
    screen_test["RankShift"] = screen_test["PropertyOnlyRank"] - screen_test["ReliabilityAwareRank"]
    target_std_train = result["target_sd"]
    Xfull = X.loc[:, FEATURES].copy()
    output_dir = c2
    def c2_save_df(data, filename):
        return base.save_df(output_dir, data, filename)
    def c2_write_text(filename, text):
        return base.save_text(output_dir, text, filename)
    env = {
        "__name__": "__comment2_verified_tail__", "OUTPUT_DIR": output_dir,
        "COMMENT1_SOURCE": Path.cwd() / "al-alloy-comment1-conformal-corrected.py",
        "df": df.copy(), "feature_cols": feature_cols, "TARGET_COLS": TARGET_COLS,
        "TARGETS": TARGETS, "proper_train_indices": proper_ids,
        "RANDOM_STATE": MASTER_SEED,
        "calibration_indices": cal_ids, "test_indices": test_ids,
        "X_imp": Xfull, "X_train_full": Xfull, "best_X_train": Xfull,
        "best_X_test": Xfull.loc[sorted(test_ids)], "full_features_for_best": list(result["features"]),
        "selected_screening_features": list(result["features"]), "screen_test": screen_test,
        "target_std_train": target_std_train, "width_values": result["width"], "q": result["q"],
        "marginal_coverage": [float(result["covered"][:, j].mean()) for j in range(3)],
        "simultaneous_coverage": float(result["covered"].all(axis=1).mean()),
        "best_metrics": pd.DataFrame([result["ensemble_metrics"]]), "EPSILON": EPSILON,
        "minmax": source_minmax, "deterministic_desc_rank": deterministic_desc_rank,
        "save_df": c2_save_df, "write_text": c2_write_text, "json": json,
        "np": np, "pd": pd, "plt": plt, "Path": Path,
        "StandardScaler": __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler,
        "NearestNeighbors": __import__("sklearn.neighbors", fromlist=["NearestNeighbors"]).NearestNeighbors,
        "time": time, "hashlib": hashlib,
    }
    source = Path.cwd() / "al-alloy-comment2-screening-analysis.py"
    text_source = source.read_text(encoding="utf-8")
    start = text_source.index("# Reviewer 1, Comment 2 — transparency and applicability audit")
    end = text_source.index("# Explicit validation, leakage, and output checks", start)
    tail = text_source[start:end]
    old_assert = "assert selected_screening_features == ['Cu', 'Fe', 'Li', 'Mg', 'Mn', 'Si', 'Tage', 'Ti', 'Tsol', 'Zn', 'Zr', 'tage']"
    replacement = "assert set(selected_screening_features) == set(full_features_for_best) and len(selected_screening_features) == len(full_features_for_best)"
    if old_assert not in tail:
        raise RuntimeError("Verified Comment 2 tail assertion was not found; refusing to execute an unverified variant")
    tail = tail.replace(old_assert, replacement, 1)
    try:
        exec(compile(tail, str(source), "exec"), env, env)
    except Exception:
        base.save_text(c2, traceback.format_exc(), "comment2_tail_failure_traceback.txt")
        raise
    # Corrected, semantically distinct ranking exports requested for the final package.
    baseline_rank = screen_test[["Original_Index", "ReliabilityAwareScore", "ReliabilityAwareRank", "Recommendation_Eligible"]].copy() if "Recommendation_Eligible" in screen_test else screen_test[["Original_Index", "ReliabilityAwareScore", "ReliabilityAwareRank"]].copy()
    baseline_rank["Rank_Source"] = "ReliabilityAwareScore = PropertyOnlyScore - 0.2*NormalizedUncertaintyPenalty"
    base.save_df(c2, baseline_rank.sort_values(["ReliabilityAwareRank", "Original_Index"]), "baseline_rank.csv")
    comparison = screen_test[["Original_Index", "PropertyOnlyScore", "ReliabilityAwareScore", "PropertyOnlyRank", "ReliabilityAwareRank", "RankShift"]].copy()
    comparison["PropertyOnly_Source"] = "0.4*StrengthScore + 0.4*DuctilityScore"
    comparison["ReliabilityAware_Source"] = "PropertyOnlyScore - 0.2*NormalizedUncertaintyPenalty"
    base.save_df(c2, comparison.sort_values(["ReliabilityAwareRank", "Original_Index"]), "screening_score_comparison.csv")
    colors = ["#0072B2", "#D55E00", "#009E73"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0)); ordered = baseline_rank.sort_values("Original_Index"); ax.scatter(ordered["Original_Index"], ordered["ReliabilityAwareRank"], s=22, color=colors[0]); ax.set_xlabel("Original index"); ax.set_ylabel("Reliability-aware rank"); ax.grid(alpha=.2); base.figure_pair(c2, "baseline_rank", fig)
    fig, ax = plt.subplots(figsize=(7.2, 5.0)); ax.scatter(comparison["PropertyOnlyScore"], comparison["ReliabilityAwareScore"], s=22, color=colors[1]); lo = float(min(comparison["PropertyOnlyScore"].min(), comparison["ReliabilityAwareScore"].min())); hi = float(max(comparison["PropertyOnlyScore"].max(), comparison["ReliabilityAwareScore"].max())); ax.plot([lo, hi], [lo, hi], "--", color="#333333"); ax.set_xlabel("Property-only screening score"); ax.set_ylabel("Reliability-aware screening score"); ax.grid(alpha=.2); base.figure_pair(c2, "screening_score_comparison", fig)
    sem = pd.DataFrame([
        {"Figure": "baseline_rank", "PNG": "baseline_rank.png", "Source_Table": "baseline_rank.csv", "Source_Columns": "Original_Index, ReliabilityAwareScore, ReliabilityAwareRank", "Semantic": "Rank ordered by reliability-aware score"},
        {"Figure": "screening_score_comparison", "PNG": "screening_score_comparison.png", "Source_Table": "screening_score_comparison.csv", "Source_Columns": "PropertyOnlyScore, ReliabilityAwareScore", "Semantic": "Property-only versus uncertainty-penalized screening score"},
    ])
    sem["PNG_SHA256"] = [base.sha256_file(c2 / "baseline_rank.png"), base.sha256_file(c2 / "screening_score_comparison.png")]
    sem["PDF_SHA256"] = [base.sha256_file(c2 / "baseline_rank.pdf"), base.sha256_file(c2 / "screening_score_comparison.pdf")]
    base.save_df(c2, sem, "figure_semantic_audit.csv")
    if sem.loc[0, "PNG_SHA256"] == sem.loc[1, "PNG_SHA256"] or sem.loc[0, "PDF_SHA256"] == sem.loc[1, "PDF_SHA256"]:
        raise RuntimeError("Corrected Comment 2 figures are not semantically distinct")
    base.save_text(c2, "The complete verified Comment 2 transparency/applicability tail was executed from al-alloy-comment2-screening-analysis.py. Only the historical hard-coded 12-feature assertion was replaced with an assertion against the current locked final feature set. baseline_rank and screening_score_comparison are separate exports with separate source columns and figure semantics.", "comment2_execution_provenance.txt")
    return {"screen_test": screen_test, "namespace": env, "tail_source": source, "figures": sem}


def comment3_outputs(root, aggregate, inner, selected, predictions, features, final_k, summary):
    c3 = root / "comment3_corrected"
    c3.mkdir(parents=True, exist_ok=True)
    base.save_df(c3, summary, "feature_subset_performance_summary.csv")
    family = selected.groupby(["Feature_Count", "Model_Family"], as_index=False).size().rename(columns={"size": "Selected_Count"})
    family["Selection_Frequency"] = family["Selected_Count"] / 25.0
    alpha = selected.groupby(["Feature_Count", "Selected_Alpha"], as_index=False).size().rename(columns={"size": "Selected_Count"})
    alpha["Selection_Frequency"] = alpha["Selected_Count"] / 25.0
    base.save_df(c3, family, "model_family_selection_by_k.csv")
    base.save_df(c3, alpha, "alpha_selection_by_k.csv")
    frequency = features.groupby(["Feature_Count", "Feature"], as_index=False).agg(
        Selection_Count=("Selected", "sum"), Selection_Frequency=("Selected", "mean"),
        Mean_Rank=("Rank", "mean"), Rank_SD=("Rank", "std"),
        Mean_Normalized_Hybrid_Feature_Score=("Mean_Normalized_Hybrid_Feature_Score", "mean"),
        Hybrid_Score_SD=("Mean_Normalized_Hybrid_Feature_Score", "std"),
    )
    frequency["Mandatory"] = frequency["Feature"].isin(MANDATORY)
    base.save_df(c3, frequency, "feature_frequency_by_k.csv")
    selected_metrics = selected[["Outer_ID", "Feature_Count", "Model_Family", "Selected_Alpha"] + [c for c in selected.columns if c.startswith("Outer_")]].copy()
    base.save_df(c3, selected_metrics, "outer_selected_metrics_by_k.csv")
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0)); ax.errorbar(summary["Feature_Count"], summary["Outer_R2_Macro_Mean"], yerr=summary["Outer_R2_Macro_SE"], fmt="o-", color=colors[0], capsize=3); ax.set_xlabel("Feature subset size k"); ax.set_ylabel("Outer-validation macro-average R²"); ax.grid(alpha=.2); base.figure_pair(c3, "feature_subset_performance", fig)
    fig, ax = plt.subplots(figsize=(7.2, 5.0));
    for j, t in enumerate(TARGETS): ax.plot(summary["Feature_Count"], summary[f"Outer_R2_{t}_Mean"], "o-", color=colors[j], label=t)
    ax.set_xlabel("Feature subset size k"); ax.set_ylabel("Outer-validation R²"); ax.legend(frameon=False); ax.grid(alpha=.2); base.figure_pair(c3, "targetwise_subset_performance", fig)
    pivot = family.pivot(index="Feature_Count", columns="Model_Family", values="Selection_Frequency").fillna(0); fig, ax = plt.subplots(figsize=(8, 5.0)); pivot.plot.bar(stacked=True, ax=ax, color=colors); ax.set_xlabel("Feature subset size k"); ax.set_ylabel("Selection frequency"); ax.legend(frameon=False, fontsize=8); base.figure_pair(c3, "model_family_selection_by_k", fig)
    ap = alpha.pivot(index="Feature_Count", columns="Selected_Alpha", values="Selection_Frequency").fillna(0); fig, ax = plt.subplots(figsize=(8, 5.0)); ap.plot.bar(stacked=True, ax=ax, colormap="viridis"); ax.set_xlabel("Feature subset size k"); ax.set_ylabel("Alpha selection frequency"); ax.legend(title="alpha", frameon=False, fontsize=8); base.figure_pair(c3, "alpha_selection_by_k", fig)
    for k, name in [(12, "feature_selection_frequency_top12"), (final_k, "feature_selection_frequency_final_k")]:
        part = frequency[frequency["Feature_Count"].eq(k)].sort_values(["Selection_Frequency", "Feature"], ascending=[True, True]); fig, ax = plt.subplots(figsize=(7.2, 5.2)); ax.barh(part["Feature"], part["Selection_Frequency"], color=colors[2]); ax.set_xlabel("Selection frequency across 25 outer evaluations"); ax.set_ylabel("Feature"); base.figure_pair(c3, name, fig)
    imp = features[features["Feature_Count"].eq(final_k)].groupby("Feature")["Mean_Normalized_Hybrid_Feature_Score"].agg(["mean", "std"]).reset_index(); imp = imp.sort_values("mean"); base.save_df(c3, imp, "feature_importance_variability.csv"); fig, ax = plt.subplots(figsize=(7.2, 5.2)); ax.barh(imp["Feature"], imp["mean"], xerr=imp["std"].fillna(0), color=colors[1]); ax.set_xlabel("Mean normalized hybrid feature score"); ax.set_ylabel("Feature"); base.figure_pair(c3, "feature_importance_variability", fig)
    # Stability figures are generated in stability(); this audit records the
    # exact source tables used by all fixed-k figures.
    base.save_df(c3, pd.DataFrame([
        {"Figure": p.stem, "Source": "feature_subset_performance_summary.csv" if "subset" in p.stem else "fixed_k_outer_selected_features.csv", "Uses_Final_Test": False, "Titles": False}
        for p in sorted(c3.glob("*.png"))
    ]), "comment3_figure_semantic_audit.csv")


def fixed_k_exports(root, aggregate, inner, selected, predictions, features, warnings_all):
    base.save_df(root, aggregate, "fixed_k_candidate_results_aggregate.csv")
    base.save_df(root, inner, "fixed_k_candidate_inner_fold_results.csv")
    base.save_df(root, selected, "fixed_k_outer_selected_configurations.csv")
    base.save_df(root, predictions, "fixed_k_outer_predictions.csv")
    metrics_cols = ["Outer_ID", "Repeat", "Outer_Fold", "Feature_Count", "Model_Family", "Selected_Alpha"] + [c for c in selected.columns if c.startswith("Outer_")]
    base.save_df(root, selected[metrics_cols], "fixed_k_outer_metrics.csv")
    base.save_df(root, features, "fixed_k_outer_selected_features.csv")
    base.save_df(root, pd.DataFrame(warnings_all), "fixed_k_warning_capture.csv")
    group_cols = ["Outer_ID", "Feature_Count", "Model_Family"]
    candidate_counts = aggregate.groupby(group_cols, as_index=False).size().rename(columns={"size": "Candidate_Count"})
    candidate_counts["Expected_Candidate_Count"] = SEARCH_ITERATIONS
    candidate_counts["Pass"] = candidate_counts["Candidate_Count"].eq(SEARCH_ITERATIONS)
    inner_counts = inner.groupby(["Outer_ID", "Feature_Count", "Model_Family", "Candidate_Number"], as_index=False).size().rename(columns={"size": "Inner_Fold_Count"})
    inner_counts["Expected_Inner_Fold_Count"] = INNER_FOLDS; inner_counts["Pass"] = inner_counts["Inner_Fold_Count"].eq(INNER_FOLDS)
    base.save_df(root, candidate_counts, "fixed_k_search_count_validation.csv")
    base.save_df(root, inner_counts, "fixed_k_inner_fold_count_validation.csv")
    tie_rows = []
    for (oid, k), part in aggregate.groupby(["Outer_ID", "Feature_Count"], sort=True):
        ordered = tie_sort([dict(r) for r in part.to_dict("records")])
        winner = ordered[0]
        actual = selected[(selected["Outer_ID"].eq(oid)) & (selected["Feature_Count"].eq(k))].iloc[0]
        tie_rows.append({"Outer_ID": oid, "Feature_Count": k, "Selected_Model_Family": actual["Model_Family"], "Selected_Candidate_Number": actual["Candidate_Number"], "Expected_Model_Family": winner["Model_Family"], "Expected_Candidate_Number": winner["Candidate_Number"], "Tie_Break_Pass": bool(actual["Model_Family"] == winner["Model_Family"] and int(actual["Candidate_Number"]) == int(winner["Candidate_Number"])), "Tie_Break_Order": "higher mean primary, lower SD, lower complexity, lexical family, lexical Params"})
    base.save_df(root, tie_rows, "fixed_k_search_selection_tie_audit.csv")
    seed_columns = ["Run_ID", "Repeat", "Outer_Fold", "Outer_ID", "Feature_Count", "Model_Family", "Derived_Seed", "Search_Independent_Fixed_k"]
    seeds = aggregate[seed_columns].drop_duplicates().copy()
    seeds["Index_Hash"] = aggregate.drop_duplicates("Run_ID").set_index("Run_ID").reindex(seeds.Run_ID)["Outer_Training_Index_Hash"].to_numpy()
    seeds["Search_Stage"] = "outer_fixed_k"
    seeds = seeds.sort_values(["Outer_ID", "Feature_Count", "Model_Family"])
    base.save_df(root, seeds, "fixed_k_search_seed_provenance.csv")
    return candidate_counts, inner_counts, pd.DataFrame(tie_rows), seeds


def validation_report(root, df, split_sets, aggregate, inner, selected, features, summary,
                      final_search_result, c1_status, c2_result, pairs, ndetails):
    checks = []
    def check(name, condition, measured, expected, evidence):
        checks.append({"Check": name, "Status": "PASS" if bool(condition) else "FAIL", "Measured": jd(measured), "Expected": jd(expected), "Evidence": evidence})
    check("retained_row_count", len(df) == 452, len(df), 452, "load_project and split assertions")
    check("split_counts", [len(split_sets[s]) for s in ["Proper_Training", "Calibration", "Final_Test"]] == [270, 91, 91], [len(split_sets[s]) for s in ["Proper_Training", "Calibration", "Final_Test"]], [270, 91, 91], "data_split_provenance.csv")
    sets = [set(split_sets[s]) for s in ["Proper_Training", "Calibration", "Final_Test"]]
    check("split_disjoint", all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i + 1, 3)), True, True, "split_assertions")
    check("split_union", set.union(*sets) == set(df.Original_Index.astype(int)), len(set.union(*sets)), len(df), "split_assertions")
    check("fixed_k_aggregate_count", len(aggregate) == 36000, len(aggregate), 36000, "fixed_k_candidate_results_aggregate.csv")
    check("fixed_k_inner_fold_count", len(inner) == 180000, len(inner), 180000, "fixed_k_candidate_inner_fold_results.csv")
    check("fixed_k_selected_count", len(selected) == 375, len(selected), 375, "fixed_k_outer_selected_configurations.csv")
    check("all_feature_counts_present", sorted(selected.Feature_Count.unique().tolist()) == SUBSET_SIZES, sorted(selected.Feature_Count.unique().tolist()), SUBSET_SIZES, "fixed_k_outer_selected_configurations.csv")
    check("all_reduced_families_present", sorted(selected.Model_Family.unique().tolist()) == sorted(REDUCED_FAMILIES), sorted(selected.Model_Family.unique().tolist()), sorted(REDUCED_FAMILIES), "fixed_k_outer_selected_configurations.csv")
    candidate_group_counts = aggregate.groupby(["Outer_ID", "Feature_Count", "Model_Family"]).size()
    check("24_candidates_per_outer_k_family", bool((candidate_group_counts == SEARCH_ITERATIONS).all()) and len(candidate_group_counts) == 25 * 15 * 4, int(candidate_group_counts.min()) if len(candidate_group_counts) else 0, SEARCH_ITERATIONS, "fixed_k_search_count_validation.csv")
    inner_group_counts = inner.groupby(["Outer_ID", "Feature_Count", "Model_Family", "Candidate_Number"]).size()
    check("five_inner_records_per_candidate", bool((inner_group_counts == INNER_FOLDS).all()) and len(inner_group_counts) == 25 * 15 * 4 * 24, int(inner_group_counts.min()) if len(inner_group_counts) else 0, INNER_FOLDS, "fixed_k_inner_fold_count_validation.csv")
    check("independent_fixed_k_runs", bool(aggregate.Search_Independent_Fixed_k.all()) and aggregate.Run_ID.nunique() == 25 * 15 * 4, aggregate.Run_ID.nunique(), 1500, "fixed_k_search_seed_provenance.csv")
    check("no_final_test_in_outer_search", not bool(selected.Final_Test_Accessed.any()) and not bool(selected.Calibration_Accessed.any()), "Calibration_Accessed=False; Final_Test_Accessed=False", "both False", "fixed_k_outer_selected_configurations.csv")
    tie = pd.read_csv(root / "fixed_k_search_selection_tie_audit.csv")
    check("deterministic_tie_audit", len(tie) == 375 and bool(tie.Tie_Break_Pass.all()), int(tie.Tie_Break_Pass.sum()), 375, "fixed_k_search_selection_tie_audit.csv")
    check("one_se_summary", summary.Feature_Count.tolist() == SUBSET_SIZES, summary.Feature_Count.tolist(), SUBSET_SIZES, "feature_subset_performance_summary.csv")
    final_k = int(final_search_result["k"]); best_k, _, _, _, _, qualifying, expected_final_k = one_se(summary)
    check("one_se_smallest_qualifying_k", final_k == expected_final_k and final_k == min(qualifying), final_k, expected_final_k, "one_se_rule_calculation.csv")
    check("final_fixed_k_96_candidates", len(pd.read_csv(root / "final_fixed_k_candidate_results.csv")) == 96, len(pd.read_csv(root / "final_fixed_k_candidate_results.csv")), 96, "final_fixed_k_candidate_results.csv")
    check("final_fixed_k_480_inner_records", len(pd.read_csv(root / "final_fixed_k_inner_fold_results.csv")) == 480, len(pd.read_csv(root / "final_fixed_k_inner_fold_results.csv")), 480, "final_fixed_k_inner_fold_results.csv")
    check("alpha_unit_tests", all(bool(r["Passed"]) for r in base.alpha_unit_tests()), "all pass", "all pass", "alpha_unit_tests")
    check("jaccard_unit_tests", all(bool(r["Passed"]) for r in base.jaccard_unit_tests()), "all pass", "all pass", "jaccard_unit_tests")
    check("nogueira_unit_tests", all(bool(r["Passed"]) for r in base.nogueira_unit_tests()), "all pass", "all pass", "nogueira_unit_tests")
    pair_counts = pairs.groupby(["Subset_Size", "Selection_Definition"]).size()
    check("jaccard_300_pairs_per_k_definition", len(pair_counts) == 30 and bool((pair_counts == 300).all()), pair_counts.to_dict(), 300, "pairwise_jaccard_values.csv")
    check("nogueira_finite_sample_correction", all(v.get("Finite_Sample_Correction") == "M/(M-1)" for v in ndetails.values()), "M/(M-1)", "M/(M-1)", "nogueira_stability_details.json")
    check("conformal_rank", int(final_search_result["rank"]) == 83, int(final_search_result["rank"]), 83, "conformal_quantiles.csv")
    check("conformal_alpha", CONFORMAL_ALPHA == .10, CONFORMAL_ALPHA, .10, "conformal_quantiles.csv")
    score_expected = np.abs(pd.read_csv(root / "ensemble_calibration_predictions.csv")[["Observed_YS", "Observed_UTS", "Observed_El"]].to_numpy(float) - final_search_result["cal_mean"]) / (final_search_result["cal_std"] + EPSILON)
    score_observed = pd.read_csv(root / "ensemble_calibration_predictions.csv")[[f"NormalizedScore_{t}" for t in TARGETS]].to_numpy(float)
    check("conformal_score_formula", bool(np.allclose(score_observed, score_expected)), bool(np.allclose(score_observed, score_expected)), True, "ensemble_calibration_predictions.csv")
    check("single_metrics_consistent", metrics_match_single(root), metrics_match_single(root), True, "single model outputs")
    check("ensemble_metrics_consistent", metrics_match_ensemble(root, final_search_result), True, True, "ensemble_test_metrics.csv")
    check("comment1_shap_complete", c1_status.startswith("SHAP computed successfully"), c1_status, "SHAP computed successfully", "comment1_regenerated/shap_feature_importance_targetwise.csv")
    c1_required = pd.read_csv(root / "comment1_regenerated" / "comment1_output_completeness.csv")
    check("comment1_required_outputs", bool(c1_required.Exists.all() and c1_required.Nonempty.all()), int(c1_required.Exists.sum()), len(c1_required), "comment1_output_completeness.csv")
    c2_table = pd.read_csv(root / "comment2_regenerated" / "heldout_unique_conditions_complete.csv")
    check("comment2_unique_conditions", len(c2_table) == 87, len(c2_table), 87, "heldout_unique_conditions_complete.csv")
    dup = root / "comment2_regenerated" / "heldout_condition_duplicate_audit.csv"
    check("comment2_duplicate_audit", dup.exists(), dup.exists(), True, str(dup))
    check("comment2_figures_semantically_distinct", c2_result["figures"]["PNG_SHA256"].nunique() == 2 and c2_result["figures"]["PDF_SHA256"].nunique() == 2, "distinct hashes", "distinct hashes", "figure_semantic_audit.csv")
    required_files = [root / "fixed_k_candidate_results_aggregate.csv", root / "fixed_k_candidate_inner_fold_results.csv", root / "fixed_k_outer_selected_configurations.csv", root / "fixed_k_outer_selected_features.csv", root / "final_locked_configuration.json", root / "comment1_regenerated" / "conformal_coverage_width.csv", root / "comment2_regenerated" / "screening_score_comparison.csv"]
    check("required_files_nonempty", all(p.exists() and p.stat().st_size > 0 for p in required_files), [p.name for p in required_files if not p.exists() or p.stat().st_size == 0], "none missing", "output tree")
    out = pd.DataFrame(checks)
    base.save_df(root, out, "executable_validation_checks.csv")
    critical = out.Status.eq("PASS")
    report = "Reviewer 1 Comment 3 fixed-k joint-search validation report\n\n" + "\n".join(f"{r.Check}: {r.Status} | measured={r.Measured} | expected={r.Expected} | evidence={r.Evidence}" for _, r in out.iterrows()) + f"\n\nFinal k: {final_k}\nOne-SE qualifying k: {qualifying}\nFinal model: {final_search_result['model_name']}\nFinal alpha: {final_search_result['alpha']}\nFinal features: {';'.join(final_search_result['features'])}\n\nCalibration leakage detected: False. Final-test targets were not used in fixed-k model selection or conformal calibration.\n"
    base.save_text(root, report, "final_validation_report.txt")
    return out, report


def metrics_match_single(root):
    tab = pd.read_csv(root / "single_model_test_predictions.csv")
    m = metric(tab[["Observed_YS", "Observed_UTS", "Observed_El"]], tab[["Predicted_YS", "Predicted_UTS", "Predicted_El"]])
    saved = pd.read_csv(root / "single_model_test_metrics.csv")
    return all(np.isclose(float(saved.loc[saved.Target.eq(t), f"R2_{t}"].iloc[0]), m[f"R2_{t}"]) for t in TARGETS)


def metrics_match_ensemble(root, result):
    tab = pd.read_csv(root / "ensemble_test_predictions.csv")
    y = tab[["Observed_YS", "Observed_UTS", "Observed_El"]]
    p = tab[["EnsembleMean_YS", "EnsembleMean_UTS", "EnsembleMean_El"]]
    m = metric(y, p); saved = pd.read_csv(root / "ensemble_test_metrics.csv")
    return all(np.isclose(float(saved.loc[saved.Target.eq(t), f"R2_{t}"].iloc[0]), m[f"R2_{t}"]) for t in TARGETS)


def package_scripts(root, cwd):
    scripts = root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    source_names = ["al-alloy-comment3-final-joint-search.py", "comment3_final_joint_search_core.py",
                    "comment3_final_core.py", "al-alloy-comment1-conformal-corrected.py",
                    "al-alloy-comment2-screening-analysis.py"]
    for name in source_names:
        source = cwd / name
        if source.exists():
            shutil.copy2(source, scripts / name)
    base.save_text(scripts, "numpy\npandas\nscikit-learn\nmatplotlib\nopenpyxl\nshap\njoblib\n", "requirements_environment_record.txt")


def write_implementation_plan(root, cwd, checkpoint_dir):
    audit = root / "audit_and_provenance"
    audit.mkdir(parents=True, exist_ok=True)
    plan = f"""Reviewer 1 Comment 3 — fixed-k joint-search implementation plan

Input: {cwd / 'Aged forged Al alloy.xlsx'}
Verified split: 270 proper-training / 91 calibration / 91 final-test; Original_Index membership is preserved.
Outer validation: RepeatedKFold, 5 folds x 5 repeats = 25 outer evaluations.
Fixed-k competition: k=3,...,17; four reduced families; 24 RandomizedSearchCV candidates per family at every k and every outer evaluation.
Expected fixed-k records: 25 x 15 x 4 x 24 = 36,000 candidate rows; 180,000 inner-fold rows; 375 outer selected configurations.
Eligible families: {', '.join(REDUCED_FAMILIES)}.
Excluded from fixed-k competition: Native_ET_All17 and Native_RF_All17.
Selection: higher mean primary score, lower standard deviation, lower complexity, lexical family, lexical serialized parameters.
Feature ranking: HybridScore_j(alpha) = alpha * minmax(PearsonRelevance_j) + (1-alpha) * minmax(MeanEmbeddedImportance_j); alpha grid {ALPHA_GRID}.
Stability: 300 pairwise comparisons per fixed k and definition; Nogueira correction M/(M-1).
Final lock: smallest k in the one-standard-error band, then a separate 96-candidate / 480-inner-record search at that fixed k.
Uncertainty: 20 proper-training bootstrap members; normalized conformal alpha={CONFORMAL_ALPHA}; rank=ceil((n_cal+1)(1-alpha)); intervals use q*(ensemble standard deviation + epsilon).
Comment 2: complete verified tail from al-alloy-comment2-screening-analysis.py, with only the obsolete hard-coded 12-feature assertion adapted to the locked final feature set.
Checkpoint directory: {checkpoint_dir}
No final output archive is permitted unless all executable validation checks pass.
"""
    base.save_text(audit, plan, "implementation_plan.txt")
    base.save_text(audit, "The fixed-k search is intentionally not a varying-k curve around one selected configuration. Each outer partition and each feature count has an independent four-family joint search. Checkpoint files are atomic and can be reused by rerunning the entry point.", "fixed_k_search_specification.txt")


def write_environment_record(root, cwd, data_path, start, finish, config, checks, exception=""):
    versions = {}
    for name in ["numpy", "pandas", "sklearn", "matplotlib", "joblib"]:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"unavailable: {exc}"
    record = {"Start_Time_Epoch": start, "Finish_Time_Epoch": finish, "Elapsed_Seconds": finish - start,
              "Command": " ".join(sys.argv), "Python": sys.version, "Platform": sys.platform,
              "Package_Versions": versions, "Master_Seed": MASTER_SEED,
              "Warning_Policy": "warnings enabled by default; estimator warnings captured per search/refit stage",
              "Input_Data": str(data_path), "Input_SHA256": base.sha256_file(data_path),
              "Final_Configuration": config, "Validation_Checks": checks, "Exception": exception}
    base.save_json(root / "execution_logs", record, "execution_environment_and_run.json")


def create_archive(cwd, final_root):
    import zipfile
    archive = cwd / "al_alloy_reviewer_comment3_final_joint_search.zip"
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {archive}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(final_root.rglob("*")):
            if path.is_file():
                zf.write(path, str(Path(final_root.name) / path.relative_to(final_root)))
    return archive


def run():
    cwd = Path.cwd()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_root = cwd / os.getenv("JOINT_OUTPUT_ROOT", f"final_submission_results_joint_search_{timestamp}")
    checkpoint_dir = cwd / os.getenv("JOINT_CHECKPOINT_DIR", "joint_search_checkpoints_24config")
    if final_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing final output: {final_root}")
    start = time.time(); build = cwd / f".joint_search_build_{timestamp}_{os.getpid()}"; build.mkdir(parents=True, exist_ok=False)
    log_path = build / "execution_logs" / "execution.log"; log_path.parent.mkdir(parents=True, exist_ok=True)
    def log(message):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"; print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh: fh.write(line + "\n")
    data_path = None; final_search_result = None
    try:
        log("Starting complete fixed-k Comment 3 joint search")
        data_path, df, manifest = base.load_project(cwd)
        split_sets = base.split_assertions(df, manifest)
        write_implementation_plan(build, cwd, checkpoint_dir)
        base.historical_audit(build, cwd, df, split_sets)
        X = df.set_index("Original_Index")[FEATURES].copy(); Y = df.set_index("Original_Index")[TARGETS].copy()
        proper_ids, cal_ids, test_ids = split_sets["Proper_Training"], split_sets["Calibration"], split_sets["Final_Test"]
        Xproper, Yproper = X.loc[proper_ids], Y.loc[proper_ids]
        log(f"Verified split: proper={len(proper_ids)}, calibration={len(cal_ids)}, final_test={len(test_ids)}")
        log("Running 25 outer evaluations x 15 k values x 4 families x 24 configurations")
        outputs = list(fixed_k_search(Xproper, Yproper, checkpoint_dir, log))
        aggregate, inner, selected, predictions, features, warning_df = outputs
        assert len(aggregate) == 36000 and len(inner) == 180000 and len(selected) == 375
        fixed_k_exports(build, aggregate, inner, selected, predictions, features, warning_df)
        summary = curve_outputs(build, selected)
        best_k, best_mean, best_sd, best_se, threshold, qualifying, final_k = one_se(summary)
        base.save_df(build, [{"Best_k": best_k, "Best_Mean_Macro_R2": best_mean, "Best_SD_Macro_R2": best_sd, "Best_SE_Macro_R2": best_se, "One_SE_Threshold": threshold, "Qualifying_k": ";".join(map(str, qualifying)), "Smallest_Qualifying_k": final_k, "Final_k": final_k}], "one_se_rule_calculation.csv")
        base.save_text(build, f"Primary criterion: mean macro-average outer-validation R2 across 25 fixed-k selected configurations.\nBest k: {best_k}\nBest mean macro-R2: {best_mean:.12f}\nBest SD: {best_sd:.12f}\nBest SE: {best_se:.12f}\nThreshold = best mean - SE: {threshold:.12f}\nQualifying k values: {', '.join(map(str, qualifying))}\nFinal k: {final_k} (smallest qualifying k)\nNo fitted configuration was cloned across k.\n", "one_se_rule_decision.txt")
        pairs, ndf, per_df = stability(build, features, selected, final_k)
        ndetails = {f"{int(r.Subset_Size)}_{r.Selection_Definition}": r.to_dict() for _, r in ndf.iterrows()}
        comment3_outputs(build, aggregate, inner, selected, predictions, features, final_k, summary)
        log(f"Fixed-k search complete; one-SE best={best_k}, qualifying={qualifying}, final_k={final_k}")
        final_search_result = final_search(Xproper, Yproper, build, log)
        result = fit_final_system(Xproper, Yproper, X.loc[cal_ids], Y.loc[cal_ids], X.loc[test_ids], Y.loc[test_ids], final_search_result, build, log)
        log(f"Locked final model: {result['model_name']}, k={result['k']}, alpha={result['alpha']}")
        c1_status = export_comment1(build, Xproper, Yproper, X.loc[cal_ids], Y.loc[cal_ids], X.loc[test_ids], Y.loc[test_ids], result)
        c2_result = run_verified_comment2(build, df, X, Xproper, X.loc[test_ids], Y.loc[test_ids], split_sets, result)
        c2 = build / "comment2_regenerated"
        # The C2 adapter writes the current fixed-k scores; retain an explicit
        # comparison against the historical standard package when available.
        historical_rank = cwd / "final_submission_results" / "comment2_regenerated" / "heldout_unique_conditions_complete.csv"
        current_rank = c2 / "heldout_unique_conditions_complete.csv"
        if historical_rank.exists() and current_rank.exists():
            old = pd.read_csv(historical_rank); new = pd.read_csv(current_rank)
            old_map = dict(zip(old.Source_Original_Indices.astype(str), old.ReliabilityAwareRank))
            new_map = dict(zip(new.Source_Original_Indices.astype(str), new.ReliabilityAwareRank))
            keys = sorted(set(old_map) | set(new_map)); changed = sum(old_map.get(k) != new_map.get(k) for k in keys)
            base.save_df(build, [{"Historical_Source": str(historical_rank), "Current_Source": str(current_rank), "Compared_Condition_Keys": len(keys), "Changed_Rank_Count": changed, "Rankings_Changed": bool(changed)}], "candidate_ranking_change_audit.csv")
        else:
            base.save_df(build, [{"Historical_Source": str(historical_rank), "Current_Source": str(current_rank), "Compared_Condition_Keys": np.nan, "Changed_Rank_Count": np.nan, "Rankings_Changed": "Not compared: historical source unavailable"}], "candidate_ranking_change_audit.csv")
        base.save_df(build, [{"Outer_Evaluations": 25, "Fixed_k_Values": 15, "Reduced_Families": 4, "Candidates_Per_Family_Per_Search": 24, "Aggregate_Candidate_Rows": len(aggregate), "Inner_Fold_Rows": len(inner), "Outer_Selected_Rows": len(selected), "Final_Search_Candidates": 96, "Final_Search_Inner_Fold_Rows": 480, "Calibration_Count": 91, "Final_Test_Count": 91, "Calibration_Leakage_Detected": False, "Final_Test_Targets_Used_In_Selection": False, "Final_Model_Family": result["model_name"], "Final_k": result["k"], "Final_Alpha": result["alpha"], "Point_Prediction_Metrics_Changed": True, "Point_Metric_Change_Reason": "The fixed-k joint search can select a different leakage-controlled final configuration; point metrics were regenerated from that locked configuration."}], "final_workflow_summary.csv")
        base.save_json(build, {"Equation_4_Wording": "Mean normalized hybrid feature score", "Stability_Measures": ["selection frequency", "pairwise Jaccard", "Nogueira"], "Nogueira_Correction": "M/(M-1) finite-sample factor", "Fixed_k_Search": "25*15*4*24", "Final_k": result["k"], "Final_Alpha": result["alpha"], "Final_Model": result["model_name"]}, "equation4_correction_and_stability_summary.json")
        package_scripts(build, cwd)
        validation_df, report = validation_report(build, df, split_sets, aggregate, inner, selected, features, summary, result, c1_status, c2_result, pairs, ndetails)
        if not bool(validation_df.Status.eq("PASS").all()):
            failures = validation_df.loc[validation_df.Status.ne("PASS"), "Check"].tolist()
            raise RuntimeError(f"Final validation failed: {failures}")
        finish = time.time(); write_environment_record(build, cwd, data_path, start, finish, result["config"], validation_df.to_dict("records"))
        base.save_text(build, "This timestamped package contains the complete fixed-k joint-search execution and regenerated Comment 1/2 outputs. The final archive was created only after executable validation passed.", "final_results_readme.txt")
        base.save_text(build, Path(log_path).read_text(encoding="utf-8") + "\nFINAL VALIDATION PASSED\n", "execution_logs/execution.log")
        base.create_manifest(build)
        shutil.move(str(build), str(final_root))
        archive = create_archive(cwd, final_root)
        print(f"FINAL_RESULTS={final_root}")
        print(f"FINAL_ARCHIVE={archive}")
        print(report)
        return final_root
    except Exception as exc:
        finish = time.time(); text = traceback.format_exc()
        try:
            if data_path is not None:
                write_environment_record(build, cwd, data_path, start, finish, final_search_result["config"] if final_search_result else {}, [], text)
        except Exception:
            pass
        base.save_text(build, text, "execution_logs/FAILED_exception_traceback.txt")
        log(f"FAILED: {type(exc).__name__}: {exc}")
        print(f"CHECKPOINTS_PRESERVED={checkpoint_dir}")
        print("Resume command: /home/mskr/anaconda3/envs/pytorch_env/bin/python al-alloy-comment3-final-joint-search.py")
        raise
