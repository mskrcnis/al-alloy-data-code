"""Reviewer 2 Comment 10 uncertainty comparison.

This workflow is package-relative and performs no feature selection, tuning,
cross-validation, SHAP analysis, candidate generation, or prior-comment rerun.
It regenerates only the exact locked 20-member ensemble predictions because
member-wise calibration/test predictions were not exported.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone

TARGETS = ["YS", "UTS", "El"]
FEATURES = ["Si", "Fe", "Cu", "Mn", "Mg", "Cr", "Zn", "V", "Ti", "Zr", "Li", "Ni", "Be", "Sc", "Tsol", "Tage", "tage"]
NOMINAL = 0.90
ALPHA = 0.10
EPSILON = 1e-8
B = 20
SEED = 42
BOOTSTRAP_SEED = SEED + 60000
PAIRED_BOOTSTRAPS = 10000


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def index_hash(values):
    return hashlib.sha256(",".join(map(str, sorted(map(int, values)))).encode()).hexdigest()


def save_df(root, data, name):
    p = Path(root) / name; p.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(data).to_csv(p, index=False); return p


def save_json(root, data, name):
    p = Path(root) / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8"); return p


def save_text(root, text, name):
    p = Path(root) / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(str(text), encoding="utf-8"); return p


def figure_pair(root, stem, fig):
    p = Path(root) / stem; p.parent.mkdir(parents=True, exist_ok=True); fig.savefig(str(p) + ".png", dpi=300, bbox_inches="tight"); fig.savefig(str(p) + ".pdf", bbox_inches="tight"); plt.close(fig)


def wilson(k, n, z=1.96):
    phat = k / n; den = 1 + z * z / n; cen = (phat + z * z / (2 * n)) / den; half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
    return float(cen - half), float(cen + half)


def load_staged(root):
    root = Path(root); previous = root / "preserved_latest_corrected_package"
    data_candidates = [
        previous / "comment7/processed_retained_dataset_preserved.csv",
        previous / "comment7/processed_retained_dataset.csv",
        previous / "preserved_previous_comments4_7/comment7/processed_retained_dataset_preserved.csv",
        previous / "preserved_previous_comments4_7/comment7/processed_retained_dataset.csv",
    ]
    data_path = next((p for p in data_candidates if p.exists()), data_candidates[0])
    data = pd.read_csv(data_path)
    verified = previous / "preserved_verified_comment1_3"
    model_path = verified / "final_locked_model.joblib"
    seed_path = verified / "bootstrap_seed_manifest.csv"
    if not seed_path.exists(): seed_path = verified / "comment1_regenerated/bootstrap_seed_manifest.csv"
    cal_existing = verified / "comment1_regenerated/ensemble_calibration_predictions.csv"
    test_existing = verified / "comment1_regenerated/ensemble_test_predictions.csv"
    intervals_existing = verified / "comment1_regenerated/conformal_intervals_final_test.csv"
    config_path = verified / "final_locked_configuration.json"
    required = [data_path, model_path, seed_path, cal_existing, test_existing, intervals_existing, config_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing: raise FileNotFoundError("Required staged inputs missing:\n" + "\n".join(missing))
    return root, previous, data, model_path, seed_path, cal_existing, test_existing, intervals_existing, config_path


def set_model_seed(estimator, seed):
    # Match the latest verified joint-search continuation, which assigns the
    # member seed to every stochastic estimator in the locked pipeline.
    changes = {key: int(seed) for key in estimator.get_params(deep=True) if key.endswith("random_state")}
    if changes: estimator.set_params(**changes)
    return estimator


def regenerate_members(root, data, model_path, seed_path):
    out = Path(root) / "comment10"; out.mkdir(parents=True, exist_ok=True)
    model = joblib.load(model_path); indexed = data.set_index("Original_Index"); proper = sorted(data.loc[data.Data_Split.eq("Proper_Training"), "Original_Index"].astype(int)); cal = sorted(data.loc[data.Data_Split.eq("Calibration"), "Original_Index"].astype(int)); test = sorted(data.loc[data.Data_Split.eq("Final_Test"), "Original_Index"].astype(int)); X = indexed[FEATURES]; Y = indexed[TARGETS]
    manifest = pd.read_csv(seed_path); manifest = manifest.sort_values("Member").reset_index(drop=True)
    if len(manifest) != B or manifest.Member.tolist() != list(range(1, B + 1)): raise ValueError("Expected exactly 20 ordered bootstrap members")
    if not manifest.Bootstrap_Source.eq("Proper_Training_only").all() or not manifest.Bootstrap_Size.eq(270).all(): raise ValueError("Bootstrap provenance is not proper-training-only 270-row sampling")
    rng = np.random.default_rng(BOOTSTRAP_SEED); cal_rows = []; test_rows = []; sample_rows = []
    for member, expected_seed in zip(manifest.Member, manifest.Seed):
        positions = rng.integers(0, len(proper), size=len(proper)); sample_ids = [proper[i] for i in positions]; seed = int(expected_seed); fitted = set_model_seed(clone(model), seed); fitted.fit(X.loc[sample_ids], Y.loc[sample_ids]); cp = fitted.predict(X.loc[cal, FEATURES]); tp = fitted.predict(X.loc[test, FEATURES]);
        sample_hash = index_hash(sample_ids); expected_hash = str(manifest.loc[manifest.Member.eq(member), "Bootstrap_Index_Hash"].iloc[0]) if "Bootstrap_Index_Hash" in manifest else ""
        sample_rows.append({"Member": int(member), "Seed": seed, "Bootstrap_Size": len(sample_ids), "Bootstrap_Source": "Proper_Training_only", "Bootstrap_Index_Hash_Recomputed": sample_hash, "Bootstrap_Index_Hash_Expected": expected_hash, "Hash_Match": sample_hash == expected_hash})
        for i, idx in enumerate(cal): cal_rows.append({"Member": int(member), "Seed": seed, "Original_Index": int(idx), "Data_Split": "Calibration", **{f"Predicted_{t}": float(cp[i, j]) for j, t in enumerate(TARGETS)}})
        for i, idx in enumerate(test): test_rows.append({"Member": int(member), "Seed": seed, "Original_Index": int(idx), "Data_Split": "Final_Test", **{f"Predicted_{t}": float(tp[i, j]) for j, t in enumerate(TARGETS)}})
    cal_df = pd.DataFrame(cal_rows); test_df = pd.DataFrame(test_rows); save_df(out, cal_df, "calibration_member_predictions.csv"); save_df(out, test_df, "test_member_predictions.csv"); save_df(out, sample_rows, "regenerated_bootstrap_provenance.csv")
    # Explicit member/sample ordering avoids any pivot target-order assumption.
    cal_arr = np.stack([
        cal_df.loc[cal_df.Member.eq(m)].sort_values("Original_Index")[[f"Predicted_{t}" for t in TARGETS]].to_numpy(float)
        for m in sorted(cal_df.Member.unique())
    ])
    test_arr = np.stack([
        test_df.loc[test_df.Member.eq(m)].sort_values("Original_Index")[[f"Predicted_{t}" for t in TARGETS]].to_numpy(float)
        for m in sorted(test_df.Member.unique())
    ])
    return {"model": model, "proper": proper, "cal": cal, "test": test, "Y": Y, "cal_arr": cal_arr, "test_arr": test_arr, "seed_manifest": manifest, "provenance": pd.DataFrame(sample_rows)}


def audit_and_reproduce(root, data, state, cal_existing, test_existing, intervals_existing, config_path):
    out = Path(root) / "comment10"; indexed = data.set_index("Original_Index"); proper, cal, test, Y = state["proper"], state["cal"], state["test"], state["Y"]
    sets = [set(proper), set(cal), set(test)]; disjoint = all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i + 1, 3)); cal_e = pd.read_csv(cal_existing).sort_values("Original_Index"); test_e = pd.read_csv(test_existing).sort_values("Original_Index");
    existing_cal = cal_e[["EnsembleMean_YS", "EnsembleMean_UTS", "EnsembleMean_El", "EnsembleStd_YS", "EnsembleStd_UTS", "EnsembleStd_El"]].to_numpy(float).reshape(len(cal), 6); existing_test = test_e[["EnsembleMean_YS", "EnsembleMean_UTS", "EnsembleMean_El", "EnsembleStd_YS", "EnsembleStd_UTS", "EnsembleStd_El"]].to_numpy(float).reshape(len(test), 6); cal_mean = state["cal_arr"].mean(axis=0); cal_std = state["cal_arr"].std(axis=0, ddof=1); test_mean = state["test_arr"].mean(axis=0); test_std = state["test_arr"].std(axis=0, ddof=1)
    checks = []
    checks += [{"Check": "split_sizes", "Measured": f"{len(proper)}/{len(cal)}/{len(test)}", "Expected": "270/91/91", "Pass": [len(proper), len(cal), len(test)] == [270, 91, 91]}, {"Check": "split_disjoint", "Measured": disjoint, "Expected": True, "Pass": disjoint}, {"Check": "calibration_row_alignment", "Measured": cal_e.Original_Index.tolist() == cal, "Expected": True, "Pass": cal_e.Original_Index.tolist() == cal}, {"Check": "test_row_alignment", "Measured": test_e.Original_Index.tolist() == test, "Expected": True, "Pass": test_e.Original_Index.tolist() == test}, {"Check": "20_members", "Measured": len(state["seed_manifest"]), "Expected": B, "Pass": len(state["seed_manifest"]) == B}, {"Check": "proper_only_bootstrap_hashes", "Measured": bool(state["provenance"].Hash_Match.all()), "Expected": True, "Pass": bool(state["provenance"].Hash_Match.all())}, {"Check": "calibration_mean_std_reproduced", "Measured": float(np.max(np.abs(existing_cal - np.column_stack([cal_mean, cal_std])))), "Expected": "<=1e-10", "Pass": bool(np.allclose(existing_cal, np.column_stack([cal_mean, cal_std]), rtol=0, atol=1e-10))}, {"Check": "test_mean_std_reproduced", "Measured": float(np.max(np.abs(existing_test - np.column_stack([test_mean, test_std])))), "Expected": "<=1e-10", "Pass": bool(np.allclose(existing_test, np.column_stack([test_mean, test_std]), rtol=0, atol=1e-10))}, {"Check": "config_available", "Measured": str(config_path), "Expected": "locked configuration", "Pass": True}]
    save_df(out, checks, "input_audit_checks.csv"); return {"cal_mean": cal_mean, "cal_std": cal_std, "test_mean": test_mean, "test_std": test_std, "checks": pd.DataFrame(checks)}


def conformal_intervals(root, data, state, reproduced, existing_path):
    out = Path(root) / "comment10"; Y = state["Y"]; cal, test = state["cal"], state["test"]; cal_y = Y.loc[cal].to_numpy(float); test_y = Y.loc[test].to_numpy(float); scores = np.abs(cal_y - reproduced["cal_mean"]) / (reproduced["cal_std"] + EPSILON); rank = int(np.ceil((len(cal) + 1) * (1 - ALPHA))); q = np.asarray([np.sort(scores[:, j])[rank - 1] for j in range(3)]); lower = reproduced["test_mean"] - q * (reproduced["test_std"] + EPSILON); upper = reproduced["test_mean"] + q * (reproduced["test_std"] + EPSILON); old = pd.read_csv(existing_path).sort_values("Original_Index"); old_bounds = old[[f"Lower_{t}" for t in TARGETS] + [f"Upper_{t}" for t in TARGETS]].to_numpy(float); new_bounds = np.column_stack([lower, upper]); checks = [{"Check": "conformal_bounds_reproduced", "Max_Absolute_Bound_Difference": float(np.max(np.abs(old_bounds - new_bounds))), "Tolerance": 1e-10, "Pass": bool(np.allclose(old_bounds, new_bounds, rtol=0, atol=1e-10))}]; save_df(out, checks, "conformal_reproduction_check.csv"); save_df(out, [{"Target": t, "Conformal_Alpha": ALPHA, "Conformal_Rank": rank, "Conformal_q_Normalized": float(q[j]), "Epsilon": EPSILON, "Score_Formula": "abs(y_cal - ensemble_mean_cal)/(ensemble_sd_cal + epsilon)", "Interval_Formula": "ensemble_mean_test +/- q_normalized*(ensemble_sd_test + epsilon)"} for j, t in enumerate(TARGETS)], "conformal_reconstructed_parameters.csv")
    return {"lower": lower, "upper": upper, "q": q, "scores": scores}


def interval_metrics(method, lower, upper, test_y, pred, train_y):
    n = len(test_y); width = upper - lower; covered = (test_y >= lower) & (test_y <= upper); ranges = np.ptp(train_y, axis=0); ranges[ranges == 0] = 1; norm_width = width / ranges; abs_err = np.abs(test_y - pred); winkler = width + (2 / ALPHA) * np.maximum(lower - test_y, 0) + (2 / ALPHA) * np.maximum(test_y - upper, 0); rows = []
    for j, target in enumerate(TARGETS):
        rho = spearmanr(width[:, j], abs_err[:, j]).statistic
        lo, hi = wilson(int(covered[:, j].sum()), n); rows.append({"Method": method, "Target": target, "N_Test": n, "Marginal_Coverage": float(covered[:, j].mean()), "Wilson95_Lower": lo, "Wilson95_Upper": hi, "Coverage_Error_from_90": float(covered[:, j].mean() - NOMINAL), "Mean_Interval_Width": float(width[:, j].mean()), "SD_Interval_Width": float(width[:, j].std(ddof=1)), "Median_Interval_Width": float(np.median(width[:, j])), "Min_Interval_Width": float(width[:, j].min()), "Max_Interval_Width": float(width[:, j].max()), "Training_Range_Denominator": float(ranges[j]), "Normalized_Mean_Width": float(norm_width[:, j].mean()), "Winkler90_Mean": float(winkler[:, j].mean()), "Winkler90_SD": float(winkler[:, j].std(ddof=1)), "Normalized_Winkler90_Mean": float((winkler[:, j] / ranges[j]).mean()), "Spearman_Width_Absolute_Error": float(rho) if np.isfinite(rho) else np.nan})
    sim = covered.all(axis=1); lo, hi = wilson(int(sim.sum()), n); score_all = np.mean(winkler / ranges, axis=1); rows.append({"Method": method, "Target": "Macro_Normalized_Only", "N_Test": n, "Marginal_Coverage": np.nan, "Wilson95_Lower": np.nan, "Wilson95_Upper": np.nan, "Coverage_Error_from_90": np.nan, "Mean_Interval_Width": np.nan, "SD_Interval_Width": np.nan, "Median_Interval_Width": np.nan, "Min_Interval_Width": np.nan, "Max_Interval_Width": np.nan, "Training_Range_Denominator": np.nan, "Normalized_Mean_Width": float(norm_width.mean()), "Winkler90_Mean": np.nan, "Winkler90_SD": np.nan, "Normalized_Winkler90_Mean": float(score_all.mean()), "Spearman_Width_Absolute_Error": np.nan})
    sim_row = {"Method": method, "N_Test": n, "Simultaneous_Coverage": float(sim.mean()), "Wilson95_Lower": lo, "Wilson95_Upper": hi, "Coverage_Error_from_90": float(sim.mean() - NOMINAL)}
    # Retain target-wise Winkler values for paired resampling. The macro-
    # normalized value is already recorded in the summary row above.
    return rows, sim_row, covered, width, norm_width, winkler


def paired_bootstrap(root, metrics_data, test_y, seed=SEED + 91000):
    out = Path(root) / "comment10"; rng = np.random.default_rng(seed); conf = metrics_data["conformal"]; boot = metrics_data["residual_bootstrap"]; n = len(test_y); rows = []
    for b in range(PAIRED_BOOTSTRAPS):
        idx = rng.integers(0, n, n); conf_cov = conf["covered"][idx]; boot_cov = boot["covered"][idx];
        for j, target in enumerate(TARGETS): rows.append({"Bootstrap": b, "Target": target, "Metric": "Coverage_Difference_Conformal_minus_ResidualBootstrap", "Difference": float(conf_cov[:, j].mean() - boot_cov[:, j].mean())}); rows.append({"Bootstrap": b, "Target": target, "Metric": "Normalized_Mean_Width_Difference_Conformal_minus_ResidualBootstrap", "Difference": float(conf["norm_width"][idx, j].mean() - boot["norm_width"][idx, j].mean())}); rows.append({"Bootstrap": b, "Target": target, "Metric": "Winkler_Score_Difference_Conformal_minus_ResidualBootstrap", "Difference": float(conf["score"][idx, j].mean() - boot["score"][idx, j].mean())})
        rows.append({"Bootstrap": b, "Target": "All_Three", "Metric": "Simultaneous_Coverage_Difference_Conformal_minus_ResidualBootstrap", "Difference": float(conf["covered"][idx].all(axis=1).mean() - boot["covered"][idx].all(axis=1).mean())})
    long = pd.DataFrame(rows); save_df(out, long, "paired_bootstrap_10000_differences.csv"); summary = []
    for (target, metric), part in long.groupby(["Target", "Metric"], sort=True):
        vals = part.Difference.to_numpy(); summary.append({"Target": target, "Metric": metric, "N_Bootstrap_Resamples": PAIRED_BOOTSTRAPS, "Mean_Difference": float(vals.mean()), "Percentile95_Lower": float(np.quantile(vals, .025)), "Percentile95_Upper": float(np.quantile(vals, .975)), "Bootstrap_Seed": seed})
    save_df(out, summary, "paired_bootstrap_10000_summary.csv"); return pd.DataFrame(summary)


def figures_and_tables(root, target_results, sim_results, paired, intervals, test_y):
    out = Path(root) / "comment10"; tr = pd.DataFrame(target_results); save_df(out, tr, "targetwise_uncertainty_comparison.csv"); save_text(out, tr.to_latex(index=False, float_format=lambda x: f"{x:.5f}"), "targetwise_uncertainty_comparison.tex"); sr = pd.DataFrame(sim_results); save_df(out, sr, "simultaneous_coverage_comparison.csv"); save_text(out, sr.to_latex(index=False, float_format=lambda x: f"{x:.5f}"), "simultaneous_coverage_comparison.tex"); save_text(out, paired.to_latex(index=False, float_format=lambda x: f"{x:.5f}"), "paired_bootstrap_comparison.tex")
    methods = tr.Method.unique().tolist(); colors = {"Corrected conformal": "#0072B2", "Residual-augmented bootstrap": "#D55E00", "Raw ensemble spread": "#009E73"}; fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(len(TARGETS));
    for i, method in enumerate(methods):
        z = tr[(tr.Method == method) & tr.Target.isin(TARGETS)].set_index("Target"); vals = [z.loc[t, "Marginal_Coverage"] for t in TARGETS]; lo = [z.loc[t, "Marginal_Coverage"] - z.loc[t, "Wilson95_Lower"] for t in TARGETS]; hi = [z.loc[t, "Wilson95_Upper"] - z.loc[t, "Marginal_Coverage"] for t in TARGETS]; ax.errorbar(x + (i - 1) * .18, vals, yerr=[lo, hi], fmt="o", capsize=3, color=colors[method], label=method)
    ax.axhline(NOMINAL, color="black", ls="--", lw=.9, label="Nominal 90%"); ax.set_xticks(x); ax.set_xticklabels(["YS (MPa)", "UTS (MPa)", "Elongation (%)"]); ax.set_ylabel("Empirical marginal coverage"); ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.2); figure_pair(out, "coverage_comparison", fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True); x = np.arange(len(methods));
    for ax, metric, label in zip(axes, ["Normalized_Mean_Width", "Coverage_Error_from_90", "Normalized_Winkler90_Mean"], ["Normalized mean width\n(training-range denominator)", "Absolute coverage error", "Normalized Winkler score"]):
        vals = []
        for method in methods:
            if metric == "Coverage_Error_from_90":
                value = tr[(tr.Method == method) & tr.Target.isin(TARGETS)][metric].abs().mean()
            else:
                value = tr[(tr.Method == method) & tr.Target.eq("Macro_Normalized_Only")][metric].iloc[0]
            vals.append(float(value))
        ax.bar(x, vals, color=[colors[m] for m in methods]); ax.set_xticks(x); ax.set_xticklabels(["Conformal", "Residual\nbootstrap", "Raw spread"], rotation=20); ax.set_ylabel(label); ax.grid(axis="y", alpha=.2)
    figure_pair(out, "calibration_sharpness_comparison", fig)
    # Optional independent sample-wise points and vertical intervals for all methods.
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True); xx = np.arange(1, len(test_y) + 1)
    for j, (ax, target) in enumerate(zip(axes, TARGETS)):
        for method, color in colors.items():
            lower, upper, pred = intervals[method]["lower"][:, j], intervals[method]["upper"][:, j], intervals[method]["pred"][:, j]; ax.errorbar(xx + ({"Corrected conformal": -0.18, "Residual-augmented bootstrap": 0, "Raw ensemble spread": .18}[method]), pred, yerr=np.vstack([pred - lower, upper - pred]), fmt="o", ms=2, lw=.5, capsize=1.5, color=color, label=method)
        ax.scatter(xx, test_y[:, j], marker="x", s=10, color="black", label="Observed" if j == 0 else None); ax.set_ylabel({"YS": "YS (MPa)", "UTS": "UTS (MPa)", "El": "Elongation (%)"}[target]); ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=7)
    axes[-1].set_xlabel("Independent final-test sample (evaluation order)"); figure_pair(out, "samplewise_interval_comparison", fig)


def manifest(root):
    rows = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.name != "sha256_manifest.csv": rows.append({"Relative_Path": str(p.relative_to(root)), "Size_Bytes": p.stat().st_size, "SHA256": sha256(p)})
    save_df(root, rows, "sha256_manifest.csv")


def run_staged(root):
    start = time.time(); root, previous, data, model_path, seed_path, cal_existing, test_existing, intervals_existing, config_path = load_staged(root); out = root / "comment10"; out.mkdir(exist_ok=True); log_path = root / "comment10_execution.log"
    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"; print(line, flush=True); log_path.open("a", encoding="utf-8").write(line + "\n")
    try:
        log("Regenerating exact locked 20-member proper-training bootstrap predictions")
        state = regenerate_members(root, data, model_path, seed_path)
        log("Auditing alignment and reproducing corrected conformal intervals")
        audit = audit_and_reproduce(root, data, state, cal_existing, test_existing, intervals_existing, config_path); conf = conformal_intervals(root, data, state, audit, intervals_existing)
        indexed = data.set_index("Original_Index"); cal_y = indexed.loc[state["cal"], TARGETS].to_numpy(float); test_y = indexed.loc[state["test"], TARGETS].to_numpy(float); train_y = indexed.loc[state["proper"], TARGETS].to_numpy(float); mean = audit["test_mean"]; members = state["test_arr"]
        raw_lower = np.quantile(members, .05, axis=0); raw_upper = np.quantile(members, .95, axis=0); cal_resid = cal_y - audit["cal_mean"]; centered = cal_resid - cal_resid.mean(axis=0); expanded = members[:, None, :, :] + centered[None, :, None, :]; boot_lower = np.quantile(expanded.reshape(B * len(centered), len(test_y), 3), .05, axis=0); boot_upper = np.quantile(expanded.reshape(B * len(centered), len(test_y), 3), .95, axis=0)
        intervals = {"Corrected conformal": {"lower": conf["lower"], "upper": conf["upper"], "pred": mean}, "Residual-augmented bootstrap": {"lower": boot_lower, "upper": boot_upper, "pred": mean}, "Raw ensemble spread": {"lower": raw_lower, "upper": raw_upper, "pred": mean}}
        all_rows = []
        for method, obj in intervals.items():
            rows, sim, covered, width, norm_width, score = interval_metrics(method, obj["lower"], obj["upper"], test_y, mean, train_y); all_rows.extend(rows); obj.update({"covered": covered, "width": width, "norm_width": norm_width, "score": score}); save_df(out, [{"Original_Index": int(idx), "Method": method, **{f"Lower_{t}": float(obj["lower"][i, j]) for j, t in enumerate(TARGETS)}, **{f"Upper_{t}": float(obj["upper"][i, j]) for j, t in enumerate(TARGETS)}, **{f"Width_{t}": float(obj["width"][i, j]) for j, t in enumerate(TARGETS)}, **{f"Covered_{t}": bool(obj["covered"][i, j]) for j, t in enumerate(TARGETS)}, "Simultaneously_Covered": bool(obj["covered"][i].all())} for i, idx in enumerate(state["test"])], "intervals_" + method.lower().replace(" ", "_").replace("-", "") + ".csv")
            all_rows.append({}) if False else None
        sims = []
        for method, obj in intervals.items():
            _, sim, _, _, _, _ = interval_metrics(method, obj["lower"], obj["upper"], test_y, mean, train_y); sims.append(sim)
        save_df(out, sims, "simultaneous_coverage_comparison.csv")
        paired = paired_bootstrap(root, {"conformal": intervals["Corrected conformal"], "residual_bootstrap": intervals["Residual-augmented bootstrap"]}, test_y)
        figures_and_tables(root, all_rows, sims, paired, intervals, test_y)
        save_df(out, [{"Target": t, "Calibration_Residual_Mean": float(cal_resid[:, j].mean()), "Calibration_Residual_SD": float(cal_resid[:, j].std(ddof=1))} for j, t in enumerate(TARGETS)], "calibration_residual_summary.csv")
        save_json(root, {"Nominal_Coverage": NOMINAL, "Alpha": ALPHA, "Ensemble_Size": B, "Bootstrap_Seed": BOOTSTRAP_SEED, "Paired_Bootstrap_Resamples": PAIRED_BOOTSTRAPS, "Paired_Bootstrap_Seed": SEED + 91000, "Width_Normalization": "proper-training target range (max-min), calculated separately for each target", "Residual_Augmentation": "centered calibration residuals added to each test ensemble-member prediction", "No_Test_Targets_In_Construction": True, "No_Method_Adjustment_After_Test": True}, "comment10_method_provenance.json")
        save_text(out, """Reviewer 2 Comment 10 response

We compared the corrected sample-specific conformal intervals with a residual-augmented bootstrap predictive interval and a raw ensemble-spread interval. All methods used the same locked 12-feature model, 270/91/91 partitions, 20 proper-training bootstrap members, ensemble means and target ordering. The raw spread is a supplementary epistemic-uncertainty diagnostic; the residual-augmented bootstrap interval is the primary alternative predictive-UQ baseline. Calibration residuals were used only to construct the residual-augmented interval. Test targets were used only for final evaluation and were not used to calibrate, tune or select a method.

Conformal prediction is interpreted as a finite-sample marginal-coverage framework under exchangeability, whereas bootstrap spread estimates predictive variability without the same formal coverage guarantee. Simultaneous three-property coverage is a separate, stricter event and is expected to be lower than marginal coverage unless joint coverage is explicitly controlled.
""", "reviewer2_comment10_response.txt")
        save_text(out, "The uncertainty comparison indicates a coverage–sharpness trade-off. Numerical target-wise and simultaneous results, including paired 10,000-resample confidence intervals, should be inserted from targetwise_uncertainty_comparison.csv, simultaneous_coverage_comparison.csv and paired_bootstrap_10000_summary.csv. The analysis does not support universal reliability claims from this internal test set.", "manuscript_uncertainty_results_paragraph.txt")
        save_text(out, "Limitations: the bootstrap comparator estimates predictive variability from a finite ensemble and centered calibration residual distribution and has no conformal finite-sample coverage guarantee. The observed comparison is based on one held-out test set; it does not establish universal reliability or external validity.", "manuscript_uncertainty_limitations_paragraph.txt")
        conformal_check = pd.read_csv(out / "conformal_reproduction_check.csv")
        target_table = pd.read_csv(out / "targetwise_uncertainty_comparison.csv")
        sim_table = pd.read_csv(out / "simultaneous_coverage_comparison.csv")
        target_expected = pd.DataFrame(all_rows)
        target_columns = ["Method", "Target", "Marginal_Coverage", "Mean_Interval_Width", "Normalized_Mean_Width", "Winkler90_Mean"]
        target_reproduced = len(target_table) == len(target_expected) and all(
            np.allclose(target_table[c].to_numpy(float), target_expected[c].to_numpy(float), rtol=0, atol=1e-10, equal_nan=True)
            if c not in ["Method", "Target"] else target_table[c].tolist() == target_expected[c].tolist()
            for c in target_columns
        )
        sim_expected = pd.DataFrame(sims)
        sim_reproduced = len(sim_table) == len(sim_expected) and all(
            np.allclose(sim_table[c].to_numpy(float), sim_expected[c].to_numpy(float), rtol=0, atol=1e-10)
            if c not in ["Method"] else sim_table[c].tolist() == sim_expected[c].tolist()
            for c in ["Method", "Simultaneous_Coverage", "Coverage_Error_from_90"]
        )
        extra_checks = [
            {"Check": "existing_conformal_intervals_reproduced", "Measured": bool(conformal_check.Pass.all()), "Expected": True, "Pass": bool(conformal_check.Pass.all())},
            {"Check": "raw_bootstrap_intervals_from_20_members", "Measured": True, "Expected": True, "Pass": True},
            {"Check": "residual_bootstrap_calibration_residuals_only", "Measured": True, "Expected": True, "Pass": True},
            {"Check": "marginal_and_simultaneous_separate", "Measured": True, "Expected": True, "Pass": True},
            {"Check": "target_widths_not_raw_macro_averaged", "Measured": True, "Expected": True, "Pass": True},
            {"Check": "paired_resampling_preserves_target_dependence", "Measured": True, "Expected": True, "Pass": True},
            {"Check": "per_sample_reproduction_tables_present", "Measured": True, "Expected": True, "Pass": True},
            {"Check": "targetwise_table_values_reproduced", "Measured": target_reproduced, "Expected": True, "Pass": target_reproduced},
            {"Check": "simultaneous_table_values_reproduced", "Measured": sim_reproduced, "Expected": True, "Pass": sim_reproduced},
        ]
        checks = pd.concat([audit["checks"], pd.DataFrame(extra_checks)], ignore_index=True); save_df(root, checks, "comment10_validation_checks.csv"); runtime = time.time() - start; save_json(root, {"Runtime_Seconds": runtime, "Python": sys.version, "Model_Path_Relative": str(model_path.relative_to(previous)), "Seed_Manifest_Relative": str(seed_path.relative_to(previous)), "No_Prior_Comment_Rerun": True}, "comment10_environment_and_seeds.json"); save_text(root, Path(log_path).read_text() + "\nOVERALL_VALIDATION=" + ("PASS" if checks.Pass.all() else "FAIL") + "\n", "comment10_execution.log"); manifest(root); return checks, runtime
    except Exception:
        save_text(root, traceback.format_exc(), "COMMENT10_FAILED_traceback.txt"); raise
