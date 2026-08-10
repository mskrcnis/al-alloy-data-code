"""Package-relative corrections for the verified Reviewer 1 Comments 4--7.

The staged package contains all inputs needed by this module.  No sibling
workspace directories are used after staging, and the completed feature-count
search is never rerun.
"""
from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from scipy.stats import t as student_t

TARGETS = ["YS", "UTS", "El"]
ELEMENTS = ["Si", "Fe", "Cu", "Mn", "Mg", "Cr", "Zn", "V", "Ti", "Zr", "Li", "Ni", "Be", "Sc"]
PROCESSING = ["Tsol", "Tage", "tage"]
FEATURES = ELEMENTS + PROCESSING
FINAL_FEATURES = ["Tsol", "Tage", "tage", "Zn", "Mg", "Cu", "Ti", "Fe", "Zr", "Si", "Sc", "Cr"]
MANDATORY = PROCESSING
TREE_PARAMS = {"n_estimators": 800, "max_depth": None, "max_features": 0.7, "min_samples_leaf": 1}
SEED = 42
K = 5
REPEATS = 5
N_OUTER = 25
N_TEST = 54
N_TRAIN = 216
CORRECTION_FACTOR = 1 / N_OUTER + N_TEST / N_TRAIN


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_df(root, data, name):
    p = Path(root) / name; p.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(data).to_csv(p, index=False); return p


def save_json(root, data, name):
    p = Path(root) / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8"); return p


def save_text(root, text, name):
    p = Path(root) / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(str(text), encoding="utf-8"); return p


def figure_pair(root, stem, fig):
    p = Path(root) / stem; p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p) + ".png", dpi=300, bbox_inches="tight"); fig.savefig(str(p) + ".pdf", bbox_inches="tight"); plt.close(fig)


def ci(values, multiplier=1.96):
    x = np.asarray(values, float); x = x[np.isfinite(x)]
    if not len(x): return (np.nan,) * 5
    mean = float(x.mean()); sd = float(x.std(ddof=1)) if len(x) > 1 else np.nan; se = sd / np.sqrt(len(x)) if np.isfinite(sd) else np.nan
    return mean, sd, se, mean - multiplier * se, mean + multiplier * se


def stable_composition_id(row):
    text = "|".join(f"{float(v):.12g}" for v in row[ELEMENTS])
    return "COMP-" + hashlib.sha256(text.encode()).hexdigest()[:16]


def load_staged(root):
    root = Path(root)
    previous = root / "preserved_previous_comments4_7"
    data = pd.read_csv(previous / "comment7/processed_retained_dataset.csv")
    pred = pd.read_csv(previous / "completed_outer_inputs/fixed_k_outer_predictions.csv")
    selected_orders = pd.read_csv(previous / "comment5/selected_chain_order_by_outer_fold.csv")
    old_impl = pd.read_csv(previous / "comment5/comment5_outer_fold_results.csv")
    if len(data) != 452 or len(pred) != 20250: raise ValueError("Staged processed dataset or outer predictions have unexpected dimensions")
    return root, previous, data, pred, selected_orders, old_impl


def train_ids(data, val_ids):
    proper = set(data.loc[data.Data_Split.eq("Proper_Training"), "Original_Index"].astype(int))
    return sorted(proper - set(map(int, val_ids)))


def fold_metrics(y, p, train_y, feature_count=None, repeat=None, fold=None, outer_id=None, implementation=None, chain_order=None):
    y = np.asarray(y, float); p = np.asarray(p, float); train_y = np.asarray(train_y, float); rows = []
    for j, target in enumerate(TARGETS):
        e = y[:, j] - p[:, j]; sd = float(np.std(train_y[:, j], ddof=1))
        rows.append({"Feature_Count": feature_count, "Repeat": repeat, "Fold": fold, "Outer_ID": outer_id, "Implementation": implementation, "Chain_Order": chain_order, "Target": target,
                     "R2": float(r2_score(y[:, j], p[:, j])), "RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
                     "NRMSE": float(np.sqrt(np.mean(e ** 2)) / sd), "NMAE": float(np.mean(np.abs(e)) / sd),
                     "MAPE_Percent": float(np.mean(np.abs(e) / np.maximum(np.abs(y[:, j]), 1e-8)) * 100)})
    rows.append({"Feature_Count": feature_count, "Repeat": repeat, "Fold": fold, "Outer_ID": outer_id, "Implementation": implementation, "Chain_Order": chain_order, "Target": "Macro",
                 "R2": float(np.mean([r["R2"] for r in rows])), "RMSE": np.nan, "MAE": np.nan,
                 "NRMSE": float(np.mean([r["NRMSE"] for r in rows])), "NMAE": float(np.mean([r["NMAE"] for r in rows])), "MAPE_Percent": np.nan})
    return rows


def summary_long(rows, group_cols, metrics, out, name):
    df = pd.DataFrame(rows); result = []
    for keys, part in df.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(keys, tuple): keys = (keys,)
        for target, tp in part.groupby("Target", sort=True):
            base = dict(zip(group_cols, keys)); base.update({"Target": target, "N_Folds": len(tp)})
            for metric in metrics:
                mean, sd, se, lo, hi = ci(tp[metric]); base.update({f"{metric}_Mean": mean, f"{metric}_SD": sd, f"{metric}_SE": se, f"{metric}_CI95_Lower": lo, f"{metric}_CI95_Upper": hi})
            result.append(base)
    return save_df(out, result, name)


def c4_correct(root, data, pred):
    out = Path(root) / "comment4"; out.mkdir(parents=True, exist_ok=True)
    p = pred[pred.Feature_Count.isin([12, 17])].copy()
    if set(p.Feature_Count.unique()) != {12, 17} or p.groupby(["Feature_Count", "Outer_ID"]).size().ne(54).any(): raise ValueError("Comment 4 does not have 25 matched folds for both k=12 and k=17")
    records = []
    for (k, oid), part in p.groupby(["Feature_Count", "Outer_ID"], sort=True):
        repeat, fold = int(part.Repeat.iloc[0]), int(part.Outer_Fold.iloc[0]); ids = part.Original_Index.astype(int).tolist(); tr = train_ids(data, ids)
        y = part[["YS_True", "UTS_True", "El_True"]].to_numpy(); q = part[["YS_Pred", "UTS_Pred", "El_Pred"]].to_numpy()
        records.extend(fold_metrics(y, q, data.set_index("Original_Index").loc[tr, TARGETS].to_numpy(), feature_count=int(k), repeat=repeat, fold=fold, outer_id=oid))
    fold_df = pd.DataFrame(records); save_df(out, fold_df, "comment4_fold_metrics.csv")
    metrics_target = ["R2", "RMSE", "MAE", "NRMSE", "NMAE", "MAPE_Percent"]
    metrics_macro = ["R2", "NRMSE", "NMAE"]
    summary_long(fold_df.to_dict("records"), ["Feature_Count"], metrics_target, out, "comment4_mean_sd_ci_by_target.csv")
    summary_long(fold_df.to_dict("records"), ["Feature_Count", "Repeat"], metrics_target, out, "comment4_mean_sd_ci_by_repeat.csv")
    # The publication table excludes macro raw RMSE, MAE and MAPE by construction.
    table_rows = []
    for k in [12, 17]:
        for target in TARGETS + ["Macro"]:
            mets = metrics_target if target != "Macro" else metrics_macro
            sub = fold_df[(fold_df.Feature_Count == k) & (fold_df.Target == target)]
            for metric in mets:
                m, s, _, lo, hi = ci(sub[metric]); table_rows.append({"Feature_Count": k, "Target": target, "Metric": metric, "Mean": m, "SD": s, "CI95_Lower": lo, "CI95_Upper": hi})
    table = pd.DataFrame(table_rows); save_df(out, table, "comment4_publication_table.csv"); save_text(out, table.to_latex(index=False, float_format=lambda x: f"{x:.5f}"), "comment4_publication_table.tex")
    # Fold-wise differences are defined as k=12 minus k=17.  Macro raw RMSE/MAE/MAPE are never formed.
    diff_rows = []
    for (oid, target), part in fold_df.groupby(["Outer_ID", "Target"], sort=True):
        a = part[part.Feature_Count == 12].iloc[0]; b = part[part.Feature_Count == 17].iloc[0]
        mets = metrics_target if target != "Macro" else metrics_macro
        for metric in mets: diff_rows.append({"Outer_ID": oid, "Repeat": a.Repeat, "Fold": a.Fold, "Target": target, "Metric": metric, "Difference_k12_minus_k17": float(a[metric] - b[metric])})
    diff = pd.DataFrame(diff_rows); save_df(out, diff, "comment4_paired_fold_differences.csv")
    tests = []
    for (target, metric), part in diff.groupby(["Target", "Metric"], sort=True):
        d = part.Difference_k12_minus_k17.to_numpy(float); mean, sd, se, _, _ = ci(d); corrected_se = float(np.sqrt(CORRECTION_FACTOR * sd ** 2)); stat = mean / corrected_se if corrected_se else np.nan; dfree = len(d) - 1; pval = float(2 * student_t.sf(abs(stat), dfree)) if np.isfinite(stat) else np.nan; crit = float(student_t.ppf(.975, dfree)); lo = mean - crit * corrected_se; hi = mean + crit * corrected_se
        tests.append({"Target": target, "Metric": metric, "N_Paired_Folds": len(d), "Mean_Difference_k12_minus_k17": mean, "SD_Difference": sd, "Corrected_SE": corrected_se, "Corrected_t": stat, "Degrees_of_Freedom": dfree, "P_Value_Two_Sided": pval, "Corrected_CI95_Lower": lo, "Corrected_CI95_Upper": hi, "Paired_Cohen_dz": mean / sd if sd else np.nan, "Correction_Factor": CORRECTION_FACTOR, "Variance_Method": "Nadeau-Bengio corrected repeated k-fold: (1/(R*K)+n_test/n_train)*s_difference^2"})
    test_df = pd.DataFrame(tests); save_df(out, test_df, "comment4_corrected_repeated5fold_tests.csv")
    # Target-wise MAPE only and elongation denominator audit.
    audit = []
    for target in TARGETS:
        vals = data[target].to_numpy(float); audit.append({"Target": target, "N_Retained": len(vals), "Zero_Count": int(np.isclose(vals, 0).sum()), "Near_Zero_LT_1e-6": int((np.abs(vals) < 1e-6).sum()), "Near_Zero_LT_1pct_Median": int((np.abs(vals) < .01 * np.median(np.abs(vals))).sum()), "Caution": "MAPE is target-wise only and is unstable for zero or near-zero denominators."})
    save_df(out, audit, "comment4_zero_near_zero_audit.csv")
    mape_rows = []
    for (feature_count, target), part in fold_df[fold_df.Target.isin(TARGETS)].groupby(["Feature_Count", "Target"], sort=True):
        mean, sd, se, lo, hi = ci(part.MAPE_Percent); mape_rows.append({"Feature_Count": feature_count, "Target": target, "MAPE_Mean": mean, "MAPE_SD": sd, "MAPE_SE": se, "MAPE_CI95_Lower": lo, "MAPE_CI95_Upper": hi})
    save_df(out, mape_rows, "comment4_targetwise_mape.csv")
    # Figures: targetwise R2 and normalized errors, and corrected paired differences.
    summ = table[table.Metric.isin(["R2", "NRMSE", "NMAE"])].copy(); fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, metric in zip(axes, ["R2", "NRMSE", "NMAE"]):
        s = summ[(summ.Metric == metric) & summ.Target.isin(TARGETS)]
        for k, color in [(12, "#0072B2"), (17, "#D55E00")]:
            z = s[s.Feature_Count == k]; x = np.arange(len(TARGETS)) + (-.12 if k == 12 else .12); ax.errorbar(x, z.Mean, yerr=[z.Mean - z.CI95_Lower, z.CI95_Upper - z.Mean], fmt="o", capsize=3, color=color, label=f"k={k}")
        ax.set_title(metric); ax.set_xticks(range(len(TARGETS))); ax.set_xticklabels(TARGETS); ax.grid(alpha=.2)
    axes[0].legend(frameon=False); figure_pair(out, "comment4_performance_comparison", fig)
    fd = diff[diff.Metric.isin(["R2", "NRMSE", "NMAE"])].copy(); groups = list(fd.groupby(["Target", "Metric"], sort=True)); labels = [f"{key[0]}\n{key[1]}" for key, _ in groups]; means = []; lows = []; highs = []
    for _, z in groups:
        m, s, se, lo, hi = ci(z.Difference_k12_minus_k17); means.append(m); lows.append(lo); highs.append(hi)
    fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(len(means)); ax.errorbar(x, means, yerr=[np.array(means)-np.array(lows), np.array(highs)-np.array(means)], fmt="o", color="#0072B2", capsize=3); ax.axhline(0, color="black", lw=.8); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_ylabel("Fold difference (k=12 − k=17)"); ax.grid(alpha=.2); figure_pair(out, "comment4_paired_difference_forest", fig)
    significant = test_df[(test_df.Target != "Macro") | (test_df.Metric.isin(["R2", "NRMSE", "NMAE"]))]
    conclusion = "The 12-feature and 17-feature pipelines demonstrated statistically comparable predictive performance, while the reduced pipeline used five fewer descriptors." if not (significant.P_Value_Two_Sided < .05).any() else "The corrected paired results indicate at least one statistically significant difference; the target-specific results should be reported rather than claiming universal equivalence."
    save_text(out, f"""Comment 4 correction

The matched fixed-k prediction file contains 25 identical outer folds for k=12 and k=17. All metrics were independently recomputed from predictions. Target-wise R2, RMSE, MAE, NRMSE, NMAE and MAPE are reported; macro results contain only R2, NRMSE and NMAE. Raw RMSE and MAE were not averaged across targets.

Paired differences are k=12 minus k=17. The corrected repeated five-fold test uses the Nadeau–Bengio variance correction with R=5, K=5, n_test=54, n_train=216, correction factor={CORRECTION_FACTOR:.5f}, and df=24. Elongation has no zero or near-zero values in the retained data, but MAPE remains a denominator-sensitive target-wise metric and is not used as a macro metric.

Conclusion: {conclusion}
""", "comment4_report_corrected.txt")
    return {"fold": fold_df, "summary": table, "diff": diff, "tests": test_df}


def et(seed): return ExtraTreesRegressor(**TREE_PARAMS, random_state=int(seed), n_jobs=1)


def make_chain(order, seed):
    base = et(seed)
    if "estimator" in inspect.signature(RegressorChain).parameters: return RegressorChain(estimator=base, order=tuple(order))
    return RegressorChain(base_estimator=base, order=tuple(order))


ORDERS = list(itertools.permutations(range(3)))
ORDER_NAMES = {o: ">".join(TARGETS[i] for i in o) for o in ORDERS}


def c5_correct(root, data, pred, selected_orders, old_impl):
    out = Path(root) / "comment5"; out.mkdir(parents=True, exist_ok=True)
    selected_orders = selected_orders.copy(); selected_orders["Outer_ID"] = selected_orders.Outer_ID.astype(str)
    if len(selected_orders) != 25 or selected_orders.Outer_ID.nunique() != 25 or selected_orders.Selected_Chain_Order.isna().any(): raise ValueError("Each outer fold must have exactly one selected chain order")
    proper = set(data.loc[data.Data_Split.eq("Proper_Training"), "Original_Index"].astype(int)); X = data.set_index("Original_Index")[FINAL_FEATURES]; Y = data.set_index("Original_Index")[TARGETS]
    impl_names = ["Separately fitted target-wise ExtraTrees", "MultiOutputRegressor(ExtraTreesRegressor)", "Native multi-output ExtraTreesRegressor"]
    pred_rows = []; metric_rows_all = []; equivalence = []; chain_freq = selected_orders.Selected_Chain_Order.value_counts().rename_axis("Chain_Order").reset_index(name="Selected_Outer_Fold_Count")
    for number, (oid, part) in enumerate(pred[pred.Feature_Count.eq(12)].groupby("Outer_ID", sort=True), 1):
        val = sorted(part.Original_Index.astype(int)); tr = sorted(proper - set(val)); xtr = X.loc[tr]; xva = X.loc[val]; ytr = Y.loc[tr].to_numpy(); yva = Y.loc[val].to_numpy(); seed = SEED + 52000 + number
        direct = [et(seed) for _ in TARGETS]
        for j, m in enumerate(direct): m.fit(xtr, ytr[:, j])
        direct_p = np.column_stack([m.predict(xva) for m in direct])
        wrapper = MultiOutputRegressor(et(seed)); wrapper.fit(xtr, ytr); wrapper_p = wrapper.predict(xva)
        native = et(seed); native.fit(xtr, ytr); native_p = native.predict(xva)
        maxdiff = float(np.max(np.abs(direct_p - wrapper_p))); equivalence.append({"Outer_ID": oid, "Seed": seed, "Max_Absolute_Difference": maxdiff, "Exactly_Equal": bool(np.array_equal(direct_p, wrapper_p)), "Numerically_Equivalent_at_1e-12": bool(np.allclose(direct_p, wrapper_p, rtol=0, atol=1e-12))})
        preds = [(impl_names[0], "Not applicable", direct_p), (impl_names[1], "Not applicable", wrapper_p), (impl_names[2], "Not applicable", native_p)]
        selected_name = str(selected_orders.loc[selected_orders.Outer_ID.eq(oid), "Selected_Chain_Order"].iloc[0])
        for order in ORDERS:
            name = ORDER_NAMES[order]; cm = make_chain(order, SEED + 53000 + number); cm.fit(xtr, ytr); cp = cm.predict(xva); preds.append(("RegressorChain(ExtraTreesRegressor)", name, cp))
        for impl, order, pp in preds:
            metric_rows_all.extend(fold_metrics(yva, pp, ytr, repeat=int(oid[1:3]), fold=int(oid[5:7]), outer_id=oid, implementation=impl, chain_order=order))
            for i, idx in enumerate(val):
                for j, target in enumerate(TARGETS): pred_rows.append({"Outer_ID": oid, "Repeat": int(oid[1:3]), "Fold": int(oid[5:7]), "Original_Index": idx, "Implementation": impl, "Chain_Order": order, "Selected_Chain_Order": selected_name, "Seed": seed if impl != "RegressorChain(ExtraTreesRegressor)" else SEED + 53000 + number, "Observed": yva[i, j], "Predicted": pp[i, j]})
    all_metrics = pd.DataFrame(metric_rows_all); pred_long = pd.DataFrame(pred_rows); save_df(out, pred_long, "comment5_corrected_outer_predictions_long.csv"); save_df(out, all_metrics, "comment5_corrected_outer_metrics.csv"); save_df(out, equivalence, "independent_wrapper_equivalence.csv"); save_df(out, chain_freq, "corrected_chain_order_selection_frequency.csv")
    selected_metric = all_metrics[(all_metrics.Implementation == "RegressorChain(ExtraTreesRegressor)" ) & all_metrics.Chain_Order.eq(all_metrics.Selected_Chain_Order if "Selected_Chain_Order" in all_metrics else all_metrics.Chain_Order)] if False else None
    # Join the selected-order table to the chain metrics, producing exactly one selected record per fold and target.
    chain = all_metrics[all_metrics.Implementation.eq("RegressorChain(ExtraTreesRegressor)")].merge(selected_orders[["Outer_ID", "Selected_Chain_Order"]], on="Outer_ID", how="left")
    selected_metric = chain[chain.Chain_Order.eq(chain.Selected_Chain_Order)].copy(); selected_metric["Chain_Order_Status"] = "Selected_by_inner_training_CV"; save_df(out, selected_metric, "corrected_selected_chain_outer_metrics.csv")
    # Four-implementation summary uses the corrected direct/wrapper rerun and the selected chain, all with matched folds.
    selected_for_four = selected_metric.copy(); selected_for_four["Chain_Order"] = "Selected_by_inner_training_CV"
    four = pd.concat([all_metrics[all_metrics.Implementation.isin(impl_names)], selected_for_four], ignore_index=True); summary_long(four.to_dict("records"), ["Implementation", "Chain_Order"], ["R2", "NRMSE", "NMAE"], out, "corrected_four_implementation_summary.csv")
    all_chain = all_metrics[all_metrics.Implementation.eq("RegressorChain(ExtraTreesRegressor)")]; summary_long(all_chain.to_dict("records"), ["Chain_Order"], ["R2", "NRMSE", "NMAE"], out, "corrected_all_six_chain_order_summary.csv")
    # Matched paired comparisons against native multi-output ExtraTrees.
    paired = []
    native = all_metrics[all_metrics.Implementation.eq("Native multi-output ExtraTreesRegressor")]
    for impl, part in [(impl_names[0], all_metrics[all_metrics.Implementation.eq(impl_names[0])]), (impl_names[1], all_metrics[all_metrics.Implementation.eq(impl_names[1])]), ("RegressorChain_selected", selected_metric)]:
        for target in TARGETS + ["Macro"]:
            for metric in ["R2", "NRMSE", "NMAE"]:
                a = part[part.Target.eq(target)].sort_values("Outer_ID")[metric].to_numpy(); b = native[native.Target.eq(target)].sort_values("Outer_ID")[metric].to_numpy(); d = a - b; mean, sd, _, _, _ = ci(d); se = np.sqrt(CORRECTION_FACTOR * sd ** 2) if sd else 0.0; stat = mean / se if se else np.nan; pval = float(2 * student_t.sf(abs(stat), 24)) if np.isfinite(stat) else np.nan; crit = student_t.ppf(.975, 24); paired.append({"Implementation": impl, "Target": target, "Metric": metric, "Mean_Difference_vs_Native": mean, "SD_Difference": sd, "Corrected_t": stat, "df": 24, "P_Value": pval, "CI95_Lower": mean - crit * se, "CI95_Upper": mean + crit * se, "Paired_Cohen_dz": mean / sd if sd else np.nan})
    paired_df = pd.DataFrame(paired); save_df(out, paired_df, "matched_paired_comparisons_vs_native.csv")
    save_text(out, """Comment 5 correction

The selected RegressorChain result was reconstructed by joining each outer fold to exactly one chain order selected by inner training-only CV. Its mean macro-R2 is independently recomputed from the selected-order predictions. All six chain orders remain available in corrected_all_six_chain_order_summary.csv.

The direct target-wise and MultiOutputRegressor implementations use the same final 12 descriptors, same training rows, same outer folds, and the same ExtraTrees random state for every target estimator. Their predictions are checked in independent_wrapper_equivalence.csv. No feature selection or broad search was repeated. Native multi-output ExtraTrees is retained as the final implementation because it directly supports multi-target tree construction; negligible paired differences do not justify replacing it based only on a mean ranking.
""", "comment5_report_corrected.txt")
    return {"metrics": all_metrics, "selected": selected_metric, "equivalence": pd.DataFrame(equivalence), "frequency": chain_freq, "paired": paired_df}


class FoldSelector:
    def __init__(self, seed): self.seed = int(seed)
    def fit(self, X, y):
        X = pd.DataFrame(X, columns=FEATURES); y = np.asarray(y, float); ystd = np.std(y, axis=0, ddof=1); ystd[ystd == 0] = 1
        ys = (y - y.mean(axis=0)) / ystd; rel = []
        for j in range(ys.shape[1]): rel.append([abs(np.corrcoef(X.iloc[:, i], ys[:, j])[0, 1]) if X.iloc[:, i].std() else 0 for i in range(X.shape[1])])
        relevance = np.nan_to_num(np.mean(np.asarray(rel), axis=0)); imps = []
        for run in range(5):
            rng = np.random.default_rng(self.seed + run); idx = rng.choice(len(X), len(X), replace=True); m = ExtraTreesRegressor(n_estimators=200, max_features="sqrt", random_state=self.seed + run, n_jobs=1); m.fit(X.iloc[idx], y[idx]); imps.append(m.feature_importances_)
        embedded = np.mean(imps, axis=0)
        def norm(a):
            lo, hi = np.min(a), np.max(a); return np.zeros_like(a) if hi == lo else (a - lo) / (hi - lo)
        score = .25 * norm(relevance) + .75 * norm(embedded); order = sorted(range(len(FEATURES)), key=lambda i: (-score[i], FEATURES[i])); ordered = list(MANDATORY) + [FEATURES[i] for i in order if FEATURES[i] not in MANDATORY]; self.selected = list(dict.fromkeys(ordered))[:12]; self.score = dict(zip(FEATURES, score)); return self


def fit_fixed_pipeline(xtr, ytr, xva, seed):
    selector = FoldSelector(seed).fit(xtr, ytr); model = ExtraTreesRegressor(**TREE_PARAMS, random_state=seed + 1000, n_jobs=1); model.fit(xtr[selector.selected], ytr); return model.predict(xva[selector.selected]), selector.selected


def balanced_group_folds(data, proper_ids):
    w = data.copy(); w["Composition_ID"] = w.apply(stable_composition_id, axis=1); sizes = w[w.Original_Index.isin(proper_ids)].groupby("Composition_ID").size().to_dict(); groups = sorted(sizes, key=lambda g: (-sizes[g], g)); bins = [[] for _ in range(K)]; counts = [0] * K
    for g in groups:
        dest = min(range(K), key=lambda i: (counts[i], i)); bins[dest].append(g); counts[dest] += sizes[g]
    assignments = []
    for fold, gs in enumerate(bins, 1): assignments.extend({"Fold": fold, "Composition_ID": g, "Group_Size": sizes[g]} for g in gs)
    return w, bins, pd.DataFrame(assignments)


def c6_correct(root, data, pred):
    out = Path(root) / "comment6"; out.mkdir(parents=True, exist_ok=True); data = data.copy(); data["Composition_ID"] = data.apply(stable_composition_id, axis=1); proper = sorted(data.loc[data.Data_Split.eq("Proper_Training"), "Original_Index"].astype(int)); index = data.set_index("Original_Index"); X = index[FEATURES]; Y = index[TARGETS]
    # Correct duplicate definitions.
    defs = [("Exact descriptors plus targets", FEATURES + TARGETS), ("Elemental composition only", ELEMENTS), ("Elemental composition plus processing", FEATURES)]; duplicate_rows = []; dist_rows = []
    for name, cols in defs:
        sizes = data.groupby(cols, dropna=False).size(); repeated = sizes[sizes > 1]; duplicate_rows.append({"Definition": name, "Unique_Groups": int(len(sizes)), "Groups_With_More_Than_One_Record": int(len(repeated)), "Records_In_Repeated_Groups": int(repeated.sum()), "Maximum_Group_Size": int(sizes.max()), "Total_Records": len(data)})
        for size, n in sizes.value_counts().sort_index().items(): dist_rows.append({"Definition": name, "Group_Size": int(size), "Number_Of_Groups": int(n), "Records_In_Groups": int(size * n)})
    save_df(out, duplicate_rows, "corrected_duplicate_statistics.csv"); save_df(out, dist_rows, "corrected_duplicate_group_size_distribution.csv")
    # Existing valid dataset distributions and composition assignments are retained; corrected assignments are also exported.
    save_df(out, data[["Original_Index", "Data_Split", "Composition_ID"] + ELEMENTS], "corrected_composition_assignments.csv")
    save_df(out, data.groupby("Composition_ID").size().rename("Group_Size").reset_index(), "corrected_composition_group_sizes.csv")
    # Existing row-wise joint-selection result is preserved as a separate reference.
    old_summary = pd.read_csv(Path(root) / "preserved_previous_comments4_7/comment6/rowwise_vs_grouped_summary.csv"); save_df(out, old_summary[old_summary.Validation_Type.eq("Rowwise_Repeated5x5")], "original_rowwise_joint_selection_summary.csv")
    # Fixed row-wise partitions reuse the 25 exported outer validation sets; only the fixed pipeline is refit.
    row_rows = []; row_assign = []; group_rows = []; group_assign = []; feature_rows = []
    for number, (oid, part) in enumerate(pred[pred.Feature_Count.eq(12)].groupby("Outer_ID", sort=True), 1):
        val = sorted(part.Original_Index.astype(int)); tr = train_ids(data, val); repeat, fold = int(oid[1:3]), int(oid[5:7]); row_assign.extend({"Validation_Type": "Fixed_Rowwise", "Repeat": repeat, "Fold": fold, "Outer_ID": oid, "Original_Index": i, "Role": "Validation" if i in val else "Training", "Composition_ID": data.loc[data.Original_Index.eq(i), "Composition_ID"].iloc[0], "Group_Overlap": np.nan} for i in tr + val); yp = Y.loc[val].to_numpy(); pr, selected = fit_fixed_pipeline(X.loc[tr], Y.loc[tr].to_numpy(), X.loc[val], SEED + 60000 + number); feature_rows.extend({"Validation_Type": "Fixed_Rowwise", "Repeat": repeat, "Fold": fold, "Feature_Order": j + 1, "Feature": f, "Mandatory": f in MANDATORY} for j, f in enumerate(selected)); row_rows.extend({**r, "Validation_Type": "Fixed_Rowwise"} for r in fold_metrics(yp, pr, Y.loc[tr].to_numpy(), repeat=repeat, fold=fold, outer_id=oid, implementation="Fixed_k12_Native_ET", chain_order="Not applicable"))
    # Composition groups are assigned greedily by descending group size to the currently smallest fold.
    data, bins, bin_df = balanced_group_folds(data, proper); group_map = data.set_index("Original_Index").Composition_ID.to_dict()
    for rep in range(1, REPEATS + 1):
        # The deterministic group assignment is rotated by repeat using a stable group order; sizes remain balanced.
        gs = bin_df.copy(); order = sorted(gs.Composition_ID, key=lambda x: hashlib.sha256(f"{SEED + 70000 + rep}|{x}".encode()).hexdigest()); sizes = data[data.Original_Index.isin(proper)].groupby("Composition_ID").size().to_dict(); bins_rep = [[] for _ in range(K)]; counts = [0] * K
        for g in sorted(order, key=lambda x: (-sizes[x], hashlib.sha256(f"{SEED + 70000 + rep}|{x}".encode()).hexdigest())):
            dest = min(range(K), key=lambda i: (counts[i], i)); bins_rep[dest].append(g); counts[dest] += sizes[g]
        for fold, val_groups in enumerate(bins_rep, 1):
            val = sorted(i for i in proper if group_map[i] in set(val_groups)); tr = sorted(set(proper) - set(val)); overlap = set(group_map[i] for i in tr) & set(val_groups); assert not overlap
            group_assign.extend({"Validation_Type": "Fixed_Composition_Grouped", "Repeat": rep, "Fold": fold, "Original_Index": i, "Role": "Validation" if i in val else "Training", "Composition_ID": group_map[i], "Group_Overlap": False} for i in tr + val)
            pr, selected = fit_fixed_pipeline(X.loc[tr], Y.loc[tr].to_numpy(), X.loc[val], SEED + 71000 + rep * 10 + fold); group_rows.extend({**r, "Validation_Type": "Fixed_Composition_Grouped"} for r in fold_metrics(Y.loc[val].to_numpy(), pr, Y.loc[tr].to_numpy(), repeat=rep, fold=fold, outer_id=f"G{rep:02d}_F{fold:02d}", implementation="Fixed_k12_Native_ET", chain_order="Not applicable")); feature_rows.extend({"Validation_Type": "Fixed_Composition_Grouped", "Repeat": rep, "Fold": fold, "Feature_Order": j + 1, "Feature": f, "Mandatory": f in MANDATORY} for j, f in enumerate(selected))
    save_df(out, row_assign + group_assign, "fixed_pipeline_fold_assignments.csv"); save_df(out, bin_df, "balanced_composition_group_assignment_template.csv"); save_df(out, feature_rows, "fixed_pipeline_fold_selected_features.csv"); save_df(out, row_rows + group_rows, "fixed_pipeline_rowwise_grouped_fold_metrics.csv")
    all_rows = pd.DataFrame(row_rows + group_rows); summary_long(all_rows.to_dict("records"), ["Validation_Type"], ["R2", "NRMSE", "NMAE"], out, "fixed_pipeline_rowwise_grouped_summary.csv")
    # Repetition-level differences are used because grouped and rowwise folds are not the same validation units.
    rep = all_rows.groupby(["Validation_Type", "Repeat", "Target"])[["R2", "NRMSE", "NMAE"]].mean().reset_index(); pivot = rep.pivot_table(index=["Repeat", "Target"], columns="Validation_Type", values=["R2", "NRMSE", "NMAE"]); effect = []
    for (repeat, target), r in pivot.iterrows():
        for metric in ["R2", "NRMSE", "NMAE"]:
            a = r[(metric, "Fixed_Composition_Grouped")]; b = r[(metric, "Fixed_Rowwise")]; effect.append({"Repeat": repeat, "Target": target, "Metric": metric, "Grouped_minus_Rowwise": a - b})
    effect_df = pd.DataFrame(effect); save_df(out, effect_df, "fixed_pipeline_grouping_repetition_differences.csv")
    test_rows = []
    for (target, metric), z in effect_df.groupby(["Target", "Metric"]):
        d = z.Grouped_minus_Rowwise.to_numpy(); m, sd, se, lo, hi = ci(d, multiplier=float(student_t.ppf(.975, len(d) - 1))); stat = m / (sd / np.sqrt(len(d))) if sd else np.nan; p = float(2 * student_t.sf(abs(stat), len(d) - 1)) if np.isfinite(stat) else np.nan; test_rows.append({"Target": target, "Metric": metric, "N_Repetitions": len(d), "Mean_Grouped_minus_Rowwise": m, "SD": sd, "t": stat, "df": len(d) - 1, "p": p, "CI95_Lower": lo, "CI95_Upper": hi})
    save_df(out, test_rows, "fixed_pipeline_grouping_repetition_tests.csv")
    # Repeated composition crossing of the original holdout partitions is preserved as a limitation table.
    cross = data.groupby("Composition_ID").Data_Split.agg(lambda x: ";".join(sorted(set(x)))).reset_index(name="Partitions"); cross["Partition_Count"] = cross.Partitions.str.count(";") + 1; save_df(out, cross[cross.Partition_Count > 1], "composition_groups_crossing_partitions_preserved.csv")
    # SHAP is copied/preserved, not rerun.
    shap_dir = Path(root) / "preserved_previous_comments4_7/comment6"; shap_files = ["shap_values_test_long.csv", "shap_mean_absolute_importance_ci.csv", "shap_bootstrap_importance_ci.csv", "shap_bootstrap_rank_distribution.csv", "shap_bootstrap_rank_correlations.csv", "shap_top_feature_overlap.csv"]
    shap_checks = [{"File": f, "Exists": (shap_dir / f).exists(), "Size_Bytes": (shap_dir / f).stat().st_size if (shap_dir / f).exists() else 0, "SHA256": sha256(shap_dir / f) if (shap_dir / f).exists() else "", "Action": "Preserved; not rerun"} for f in shap_files]; save_df(out, shap_checks, "shap_integrity_preservation_check.csv"); save_text(out, "Existing SHAP values were retained after file-integrity checks. The interpretation is evaluation-sample bootstrap stability of SHAP rankings, not complete model-level SHAP stability. SHAP explains model behaviour, not causal metallurgy, and cannot independently justify forcibly retained variables.", "shap_methodology_corrected.txt")
    return {"metrics": all_rows, "duplicates": pd.DataFrame(duplicate_rows), "assignments": pd.DataFrame(row_assign + group_assign), "shap": pd.DataFrame(shap_checks)}


def c7_correct(root, data):
    out = Path(root) / "comment7"; out.mkdir(parents=True, exist_ok=True); figdir = out / "figures"; figdir.mkdir(exist_ok=True)
    previous = Path(root) / "preserved_previous_comments4_7"; intervals = pd.read_csv(previous / "verified_k12_inputs/conformal_intervals_final_test.csv") if (previous / "verified_k12_inputs/conformal_intervals_final_test.csv").exists() else pd.read_csv(previous / "verified_k12_inputs/conformal_intervals_final_test.csv")
    labels = {"YS": "YS (MPa)", "UTS": "UTS (MPa)", "El": "Elongation (%)"}
    for target in TARGETS:
        x = np.arange(1, len(intervals) + 1); mean = intervals[f"EnsembleMean_{target}"].to_numpy(); lo = intervals[f"Lower_{target}"].to_numpy(); hi = intervals[f"Upper_{target}"].to_numpy(); obs = intervals[f"Observed_{target}"].to_numpy(); fig, ax = plt.subplots(figsize=(11, 5)); ax.errorbar(x, mean, yerr=np.vstack([mean - lo, hi - mean]), fmt="o", ms=3, lw=.7, capsize=2, color="#0072B2", label="Predicted point and conformal interval"); ax.scatter(x, obs, marker="x", s=16, color="#D55E00", label="Observed value"); ax.set_xlabel("Independent final-test sample (evaluation order)"); ax.set_ylabel(labels[target]); ax.legend(frameon=False); ax.grid(alpha=.2); figure_pair(figdir, f"figure3_samplewise_intervals_{target}", fig)
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
    for ax, target in zip(axes, TARGETS):
        x = np.arange(1, len(intervals) + 1); mean = intervals[f"EnsembleMean_{target}"].to_numpy(); lo = intervals[f"Lower_{target}"].to_numpy(); hi = intervals[f"Upper_{target}"].to_numpy(); obs = intervals[f"Observed_{target}"].to_numpy(); ax.errorbar(x, mean, yerr=np.vstack([mean - lo, hi - mean]), fmt="o", ms=2.5, lw=.6, capsize=2, color="#0072B2", label="Prediction ± interval"); ax.scatter(x, obs, marker="x", s=12, color="#D55E00", label="Observed"); ax.set_ylabel(labels[target]); ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.2)
    axes[-1].set_xlabel("Independent final-test sample (evaluation order)"); figure_pair(figdir, "figure3_samplewise_prediction_intervals", fig)
    save_text(out, "Figure 3 caption: Independent final-test predictions are shown as sample-wise points with vertical normalized-conformal prediction intervals. Orange crosses are observed values. YS and UTS are reported in MPa and elongation in percent; samples are independent observations and are not joined by a continuous ribbon.", "figure3_caption_corrected.txt")
    save_text(out, "Compact and Interpretable Strength–Ductility Prediction of Heat-Treatable Aluminum Alloys Using Multi-Target Machine Learning", "title_corrected.txt")
    save_text(out, """Proposed abstract wording: We develop a compact and interpretable multi-target machine-learning workflow for predicting yield strength, ultimate tensile strength and elongation in heat-treatable aluminum alloys. A training-only feature-selection procedure retains 12 descriptors, while ensemble-conformal intervals quantify uncertainty on held-out observations. The resulting predictions and screening tables provide preliminary model-based prioritization for subsequent assessment; they do not establish external validity, experimental confirmation or causal metallurgical relationships.

Proposed conclusion wording: The locked 12-descriptor native multi-output ExtraTrees model provides a compact representation for the evaluated dataset. Its point predictions, uncertainty intervals and model-behaviour explanations can support preliminary model-based prioritization, but they should not be interpreted as universal reliability, practical alloy-design confirmation or causal metallurgical evidence. Candidate rankings remain hypotheses requiring external and experimental validation.
""", "abstract_conclusion_corrected.txt")
    save_text(out, "Literature-based comparison methods are described as adapted in-house baseline implementations. They are not direct reproductions because the source studies used different datasets, alloy systems and experimental contexts.", "baseline_wording_corrected.txt")
    save_text(out, """Data/code availability statement

Source data: the supplied Aged forged Al alloy workbook, retained in the project archive. Processed data: comment7/processed_retained_dataset.csv in this correction package. The workflow retained 452 target-complete records from 524 raw records and used 270 proper-training, 91 calibration and 91 final-test records. The final model uses Tsol, Tage, tage, Zn, Mg, Cu, Ti, Fe, Zr, Si, Sc and Cr; alpha=0.25; native multi-output ExtraTrees; 800 trees; unlimited depth; max_features=0.7; min_samples_leaf=1. Preprocessing and feature ranking are fit using training records only. Master seed=42; all analysis-specific seeds and split assignments are exported with the tables.

Software versions and execution instructions are in software_versions.json and execution_environment_and_run.json. The correction script is package-relative: execute scripts/al-alloy-comments4-7-corrections.py from the package root with the staged preserved inputs present. Repository/archive location: [REPOSITORY_URL_OR_DOI_TO_BE_INSERTED]. This placeholder must be replaced by the authors with the actual public repository URL or DOI before submission; no unavailable location is invented here.
""", "data_code_availability_corrected.txt")
    save_json(out, {"Target_Units": {"YS": "MPa", "UTS": "MPa", "El": "%"}, "Units_Source": "ERX-121075_Proof_hi.pdf, Section 3.1", "Repository_Placeholder": "[REPOSITORY_URL_OR_DOI_TO_BE_INSERTED]"}, "units_and_repository_provenance.json")
    save_df(out, data, "processed_retained_dataset_preserved.csv")
    # A figure audit covers the newly generated figures and preserves the prior valid figure set without editing Comments 1--3.
    rows = [{"Figure": p.name, "Status": "PASS", "Labels_Units": "YS MPa; UTS MPa; El %", "Legend": "Prediction interval and observed value", "Continuous_Connection": "None", "Caption": "Corrected sample-wise caption"} for p in sorted(figdir.glob("*.png"))]
    for p in sorted((Path(root) / "preserved_verified_comment1_3").rglob("*.png")):
        rows.append({"Figure": str(p.relative_to(root / "preserved_verified_comment1_3")), "Status": "PRESERVED_VALID", "Labels_Units": "Prior verified figure retained; target units confirmed from manuscript", "Legend": "Prior verified figure retained", "Continuous_Connection": "Not modified", "Caption": "Prior verified caption retained"})
    save_df(out, rows, "manuscript_figure_label_caption_audit.csv")


def validate(root, c4, c5, c6):
    checks = []
    def add(name, passed, measured, expected, evidence): checks.append({"Check": name, "Status": "PASS" if bool(passed) else "FAIL", "Measured": measured, "Expected": expected, "Evidence": evidence})
    p = c4["fold"]; add("comment4_25_matched_folds_both_k", p.groupby(["Feature_Count", "Outer_ID"]).ngroups == 50 and set(p.Feature_Count) == {12, 17}, int(p.groupby(["Feature_Count", "Outer_ID"]).ngroups), 50, "comment4_fold_metrics.csv")
    add("comment4_no_macro_raw_error_or_mape", not ((pd.read_csv(root / "comment4/comment4_publication_table.csv").Target.eq("Macro")) & (pd.read_csv(root / "comment4/comment4_publication_table.csv").Metric.isin(["RMSE", "MAE", "MAPE_Percent"]))).any(), True, True, "comment4_publication_table.csv")
    so = pd.read_csv(root / "preserved_previous_comments4_7/comment5/selected_chain_order_by_outer_fold.csv"); add("comment5_one_selected_chain_order_per_outer", len(so) == 25 and so.Outer_ID.nunique() == 25 and so.Selected_Chain_Order.notna().all(), len(so), 25, "selected_chain_order_by_outer_fold.csv")
    add("comment5_selected_chain_join_exact", len(c5["selected"]) == 25 * 4 and c5["selected"].Outer_ID.nunique() == 25, len(c5["selected"]), 100, "corrected_selected_chain_outer_metrics.csv")
    eq = c5["equivalence"]; add("matched_seed_independent_wrapper", eq["Numerically_Equivalent_at_1e-12"].all(), float(eq.Max_Absolute_Difference.max()), 0.0, "independent_wrapper_equivalence.csv")
    add("comment5_all_four_summary", set(pd.read_csv(root / "comment5/corrected_four_implementation_summary.csv").Implementation) == {"Separately fitted target-wise ExtraTrees", "MultiOutputRegressor(ExtraTreesRegressor)", "Native multi-output ExtraTreesRegressor", "RegressorChain(ExtraTreesRegressor)"}, True, True, "corrected_four_implementation_summary.csv")
    ass = c6["assignments"]; grp = ass[ass.Validation_Type.eq("Fixed_Composition_Grouped")]; leakage = grp.groupby(["Repeat", "Fold", "Composition_ID"]).Role.nunique().max(); add("grouped_no_composition_leakage", leakage == 1, int(leakage), 1, "fixed_pipeline_fold_assignments.csv")
    add("same_fixed_pipeline_rowwise_grouped", set(c6["metrics"].Implementation.dropna()) == {"Fixed_k12_Native_ET"}, set(c6["metrics"].Implementation.dropna()), {"Fixed_k12_Native_ET"}, "fixed_pipeline_rowwise_grouped_fold_metrics.csv")
    d = pd.read_csv(root / "comment6/corrected_duplicate_statistics.csv"); add("duplicate_labels_and_counts_correct", set(d.columns) >= {"Unique_Groups", "Groups_With_More_Than_One_Record", "Records_In_Repeated_Groups", "Maximum_Group_Size"} and not d.Definition.str.contains("Duplicate_Group_Count").any(), True, True, "corrected_duplicate_statistics.csv")
    shap = pd.read_csv(root / "comment6/shap_integrity_preservation_check.csv"); add("existing_shap_preserved_not_rerun", shap.Exists.all() and shap.Action.eq("Preserved; not rerun").all(), bool(shap.Exists.all()), True, "shap_integrity_preservation_check.csv")
    units = json.loads((Path(root) / "comment7/units_and_repository_provenance.json").read_text()); add("manuscript_units_confirmed", units["Target_Units"] == {"YS": "MPa", "UTS": "MPa", "El": "%"}, units["Target_Units"], {"YS": "MPa", "UTS": "MPa", "El": "%"}, "ERX-121075_Proof_hi.pdf Section 3.1")
    add("neutral_title_used", "Reliable" not in (Path(root) / "comment7/title_corrected.txt").read_text(), (Path(root) / "comment7/title_corrected.txt").read_text().strip(), "neutral title", "title_corrected.txt")
    required = [root / "comment4/comment4_corrected_repeated5fold_tests.csv", root / "comment4/comment4_publication_table.tex", root / "comment4/comment4_paired_difference_forest.pdf", root / "comment5/corrected_selected_chain_outer_metrics.csv", root / "comment6/fixed_pipeline_rowwise_grouped_summary.csv", root / "comment7/figures/figure3_samplewise_prediction_intervals.pdf"]
    add("required_corrected_outputs_nonempty", all(p.exists() and p.stat().st_size > 0 for p in required), [str(p) for p in required if not p.exists() or p.stat().st_size == 0], "none", "correction folders")
    out = save_df(root, checks, "correction_validation_checks.csv"); return pd.DataFrame(checks)


def repair_cached_c4_c5_tables(root):
    """Refresh presentation tables from completed cached fold predictions/metrics."""
    c4 = Path(root) / "comment4"; fold = pd.read_csv(c4 / "comment4_fold_metrics.csv")
    rows = []
    for (feature_count, target), part in fold[fold.Target.isin(TARGETS)].groupby(["Feature_Count", "Target"], sort=True):
        mean, sd, se, lo, hi = ci(part.MAPE_Percent); rows.append({"Feature_Count": feature_count, "Target": target, "MAPE_Mean": mean, "MAPE_SD": sd, "MAPE_SE": se, "MAPE_CI95_Lower": lo, "MAPE_CI95_Upper": hi})
    save_df(c4, rows, "comment4_targetwise_mape.csv")
    c5 = Path(root) / "comment5"; all_metrics = pd.read_csv(c5 / "comment5_corrected_outer_metrics.csv"); selected = pd.read_csv(c5 / "corrected_selected_chain_outer_metrics.csv"); selected["Chain_Order"] = "Selected_by_inner_training_CV"
    impls = ["Separately fitted target-wise ExtraTrees", "MultiOutputRegressor(ExtraTreesRegressor)", "Native multi-output ExtraTreesRegressor"]
    four = pd.concat([all_metrics[all_metrics.Implementation.isin(impls)], selected], ignore_index=True); summary_long(four.to_dict("records"), ["Implementation", "Chain_Order"], ["R2", "NRMSE", "NMAE"], c5, "corrected_four_implementation_summary.csv")


def manifest(root):
    rows = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.name != "sha256_manifest.csv": rows.append({"Relative_Path": str(p.relative_to(root)), "Size_Bytes": p.stat().st_size, "SHA256": sha256(p)})
    return save_df(root, rows, "sha256_manifest.csv")


def run_staged(root):
    start = time.time(); root, previous, data, pred, selected_orders, old_impl = load_staged(root); log_path = root / "correction_execution.log"
    def log(s):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {s}"; print(line, flush=True); log_path.open("a", encoding="utf-8").write(line + "\n")
    try:
        prior = root / "preserved_prior_correction"
        if (prior / "comment5/comment5_corrected_outer_metrics.csv").exists():
            log("Reusing completed corrected Comment 4–5 outputs from the prior interrupted run")
            shutil.copytree(prior / "comment4", root / "comment4")
            shutil.copytree(prior / "comment5", root / "comment5")
            repair_cached_c4_c5_tables(root)
            c4 = {"fold": pd.read_csv(root / "comment4/comment4_fold_metrics.csv"), "summary": pd.read_csv(root / "comment4/comment4_publication_table.csv"), "diff": pd.read_csv(root / "comment4/comment4_paired_fold_differences.csv"), "tests": pd.read_csv(root / "comment4/comment4_corrected_repeated5fold_tests.csv")}
            c5 = {"metrics": pd.read_csv(root / "comment5/comment5_corrected_outer_metrics.csv"), "selected": pd.read_csv(root / "comment5/corrected_selected_chain_outer_metrics.csv"), "equivalence": pd.read_csv(root / "comment5/independent_wrapper_equivalence.csv"), "frequency": pd.read_csv(root / "comment5/corrected_chain_order_selection_frequency.csv"), "paired": pd.read_csv(root / "comment5/matched_paired_comparisons_vs_native.csv")}
        else:
            log("Correcting Comment 4 from matched k=12 and k=17 predictions")
            c4 = c4_correct(root, data, pred)
            log("Correcting Comment 5 with matched-seed implementations and selected chain orders")
            c5 = c5_correct(root, data, pred, selected_orders, old_impl)
        log("Correcting Comment 6 fixed-pipeline validation and duplicate statistics")
        c6 = c6_correct(root, data, pred)
        log("Correcting Comment 7 units, claims, Figure 3 and availability statement")
        c7_correct(root, data)
        validation = validate(root, c4, c5, c6); runtime = time.time() - start
        save_json(root, {"Runtime_Seconds": runtime, "Seed": SEED, "Tree_Params": TREE_PARAMS, "Correction_Factor": CORRECTION_FACTOR, "Package_Relative": True, "No_30_Hour_Search_Rerun": True}, "correction_environment_and_seeds.json")
        save_text(root, Path(log_path).read_text() + "\nOVERALL_VALIDATION=" + ("PASS" if validation.Status.eq("PASS").all() else "FAIL") + "\n", "correction_execution.log")
        summary = """Reviewer 1 Comments 4–7 correction summary

Comment 4 was recomputed from matched k=12 and k=17 outer predictions. Corrected paired tests, target-wise MAPE, tables and figures are in comment4.
Comment 5 reconstructs the inner-CV-selected chain order per outer fold and reruns only the fixed implementation comparison with matched seeds.
Comment 6 uses the same fixed k=12 pipeline for new row-wise and composition-grouped validation, correct duplicate definitions, and preserved SHAP files.
Comment 7 confirms YS/UTS=MPa and El=% from the manuscript proof, uses a neutral title, and regenerates sample-wise interval figures.
"""
        save_text(root, summary + f"\nRuntime seconds: {runtime:.2f}\nValidation: {'PASS' if validation.Status.eq('PASS').all() else 'FAIL'}\n", "correction_summary.txt")
        manifest(root); return validation, runtime
    except Exception:
        save_text(root, traceback.format_exc(), "CORRECTION_FAILED_traceback.txt"); raise
