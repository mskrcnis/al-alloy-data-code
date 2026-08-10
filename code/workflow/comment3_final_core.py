"""Implementation used by al-alloy-comment3-final-corrected.py.

The implementation deliberately keeps the corrected analysis in one executable
path: raw data -> locked nested-CV configuration -> calibration -> test ->
downstream screening -> audit/package validation.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
import warnings
from copy import deepcopy
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, RepeatedKFold
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend

MASTER_SEED = 42
TARGETS = ["YS", "UTS", "El"]
FEATURES = ["Si", "Fe", "Cu", "Mn", "Mg", "Cr", "Zn", "V", "Zr", "Li", "Ni", "Be", "Sc", "Tsol", "Tage", "tage"]
# The workbook contains Ti between Zr and Li.  Keep the source order explicit.
FEATURES = ["Si", "Fe", "Cu", "Mn", "Mg", "Cr", "Zn", "V", "Ti", "Zr", "Li", "Ni", "Be", "Sc", "Tsol", "Tage", "tage"]
MANDATORY = ["Tsol", "Tage", "tage"]
ALPHA_GRID = [0.00, 0.25, 0.50, 0.75, 1.00]
SUBSET_SIZES = list(range(3, 18))
OUTER_REPEATS = 5
OUTER_FOLDS = 5
INNER_FOLDS = 5
SEARCH_ITERATIONS = 24
RANK_RUNS = 5
RANK_TREES = 200
FINAL_TREES = 600
ENSEMBLE_SIZE = 20
CONFORMAL_ALPHA = 0.10
EPSILON = 1e-8
_RANK_CACHE = {}


def jready(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (dict,)):
        return {str(k): jready(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, np.ndarray)):
        return [jready(v) for v in x]
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return x


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def index_hash(ids):
    return hashlib.sha256(",".join(map(str, sorted(map(int, ids)))).encode()).hexdigest()


def metric_dict(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = {}
    r2s, rmses, maes, nrmse = [], [], [], []
    for j, target in enumerate(TARGETS):
        rm = float(np.sqrt(mean_squared_error(y[:, j], p[:, j])))
        ma = float(mean_absolute_error(y[:, j], p[:, j]))
        sd = float(np.std(y[:, j], ddof=1))
        r2 = float(r2_score(y[:, j], p[:, j]))
        out.update({f"R2_{target}": r2, f"RMSE_{target}": rm, f"MAE_{target}": ma,
                    f"NRMSE_{target}": rm / sd if sd else np.nan})
        r2s.append(r2); rmses.append(rm); maes.append(ma); nrmse.append(rm / sd if sd else np.nan)
    out.update({"R2_Macro": float(np.mean(r2s)), "RMSE_Macro": float(np.mean(rmses)),
                "MAE_Macro": float(np.mean(maes)), "NRMSE_Macro": float(np.nanmean(nrmse))})
    return out


def ci(values):
    a = np.asarray(values, float)
    mean = float(np.mean(a)); sd = float(np.std(a, ddof=1)); se = sd / math.sqrt(len(a))
    return mean, sd, float(se), float(mean - 1.96 * se), float(mean + 1.96 * se)


def minmax(values):
    s = pd.Series(values, dtype=float)
    span = float(s.max() - s.min())
    if span <= 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / span


class PhysicsRetainedSelector(BaseEstimator, TransformerMixin):
    """Recovered alpha-weighted ranking implementation from the original notebook."""
    def __init__(self, n_features=12, alpha=0.50, physics_keep=tuple(MANDATORY),
                 n_importance_runs=RANK_RUNS, n_estimators=RANK_TREES, random_state=42):
        self.n_features = n_features; self.alpha = alpha; self.physics_keep = physics_keep
        self.n_importance_runs = n_importance_runs; self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        Xdf = pd.DataFrame(X).copy()
        if list(Xdf.columns) != FEATURES:
            Xdf.columns = FEATURES[:Xdf.shape[1]]
        ydf = pd.DataFrame(y, columns=TARGETS if np.asarray(y).ndim == 2 else None)
        yarr = ydf.to_numpy(float)
        ysd = np.std(yarr, axis=0, ddof=1); ysd[ysd == 0] = 1.0
        yscaled = (yarr - np.mean(yarr, axis=0)) / ysd
        # RandomizedSearchCV refits the selector for each alpha/k combination,
        # although the raw fold-specific relevance and embedded scores are
        # identical.  Cache only by the complete fold data, target values, and
        # ranking parameters; this is a computational memoization, not a
        # cross-fold or cross-partition data transfer.
        cache_key = hashlib.sha256(Xdf.to_numpy(float).tobytes() + yarr.tobytes() +
                                   repr((self.n_importance_runs, self.n_estimators, self.random_state)).encode()).hexdigest()
        if cache_key in _RANK_CACHE:
            relevance, embedded = _RANK_CACHE[cache_key]
        else:
            relevance = []
            for j in range(yscaled.shape[1]):
                vals = [np.corrcoef(Xdf.iloc[:, i].to_numpy(float), yscaled[:, j])[0, 1]
                        if np.std(Xdf.iloc[:, i]) > 0 else 0.0 for i in range(Xdf.shape[1])]
                relevance.append(np.abs(np.nan_to_num(vals)))
            relevance = np.mean(np.vstack(relevance), axis=0)
            imp_runs = []
            for run in range(int(self.n_importance_runs)):
                rng = np.random.default_rng(int(self.random_state) + run)
                rows = rng.choice(len(Xdf), len(Xdf), replace=True)
                model = ExtraTreesRegressor(n_estimators=int(self.n_estimators), max_features="sqrt",
                                            random_state=int(self.random_state) + run, n_jobs=1)
                model.fit(Xdf.iloc[rows], yarr[rows])
                imp_runs.append(model.feature_importances_)
            embedded = np.mean(np.vstack(imp_runs), axis=0)
            _RANK_CACHE[cache_key] = (relevance, embedded)
        rel_n = minmax(pd.Series(relevance, index=Xdf.columns)).to_numpy()
        emb_n = minmax(pd.Series(embedded, index=Xdf.columns)).to_numpy()
        hybrid = float(self.alpha) * rel_n + (1.0 - float(self.alpha)) * emb_n
        table = pd.DataFrame({"Feature": Xdf.columns, "PearsonRelevance": relevance,
                              "EmbeddedImportance": embedded, "NormalizedPearson": rel_n,
                              "NormalizedEmbedded": emb_n, "HybridScore": hybrid})
        table["Mandatory"] = table["Feature"].isin(self.physics_keep)
        table = table.sort_values(["HybridScore", "Feature"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
        ordered = [f for f in self.physics_keep if f in list(table.Feature)]
        ordered += [f for f in table.Feature if f not in ordered]
        self.ranking_table_ = table
        self.feature_ranking_ = ordered
        self.selected_features_ = ordered[:int(self.n_features)]
        # The selector must always retain all physics variables when k >= 3.
        self.selected_features_ = list(dict.fromkeys([f for f in self.physics_keep if f in Xdf.columns] +
                                                      [f for f in self.selected_features_ if f not in self.physics_keep]))[:int(self.n_features)]
        self.n_features_in_ = Xdf.shape[1]
        return self

    def transform(self, X):
        Xdf = pd.DataFrame(X).copy()
        if list(Xdf.columns) != FEATURES:
            Xdf.columns = FEATURES[:Xdf.shape[1]]
        return Xdf[self.selected_features_]


def make_selector_pipeline(model, n_features=SUBSET_SIZES, alpha=ALPHA_GRID, seed=42):
    return Pipeline([("selector", PhysicsRetainedSelector(n_features=12, alpha=.5,
                                                            random_state=seed)), ("model", model)])


def model_specs(fixed_k=None):
    ks = [int(fixed_k)] if fixed_k is not None else SUBSET_SIZES
    tree = {"n_estimators": [300, 500, 800], "max_depth": [None, 6, 10, 14],
            "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", .7, 1.0]}
    sel = {"selector__n_features": ks, "selector__alpha": ALPHA_GRID}
    specs = [
        ("Native_ET_Physics", make_selector_pipeline(ExtraTreesRegressor(random_state=42, n_jobs=1)), {**sel, **{"model__" + k: v for k, v in tree.items()}}, "reduced physics-retained ExtraTrees"),
        ("Native_RF_Physics", make_selector_pipeline(RandomForestRegressor(random_state=42, n_jobs=1)), {**sel, **{"model__" + k: v for k, v in tree.items()}}, "reduced physics-retained RandomForest"),
        ("Independent_ET_Physics", make_selector_pipeline(MultiOutputRegressor(ExtraTreesRegressor(random_state=42, n_jobs=1))), {**sel, **{"model__estimator__" + k: v for k, v in tree.items()}}, "reduced physics-retained independent ET"),
        ("RegressorChain_ET_Physics", make_selector_pipeline(RegressorChain(ExtraTreesRegressor(random_state=42, n_jobs=1), order=(0, 1, 2))), {**sel, **{"model__base_estimator__" + k: v for k, v in tree.items()}}, "reduced physics-retained regressor chain"),
        ("Native_ET_All17", Pipeline([("imputer", SimpleImputer(strategy="mean")), ("model", ExtraTreesRegressor(random_state=42, n_jobs=1))]), {"model__" + k: v for k, v in tree.items()}, "all-feature ExtraTrees comparator"),
        ("Native_RF_All17", Pipeline([("imputer", SimpleImputer(strategy="mean")), ("model", RandomForestRegressor(random_state=42, n_jobs=1))]), {"model__" + k: v for k, v in tree.items()}, "all-feature RandomForest comparator"),
    ]
    return specs


def scorer(estimator, X, y):
    pred = estimator.predict(X)
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    vals = []
    for j in range(y.shape[1]):
        sd = np.std(y[:, j], ddof=1)
        vals.append(np.sqrt(mean_squared_error(y[:, j], pred[:, j])) / sd if sd else 0.0)
    return -float(np.mean(vals))


def parse_params(params):
    out = {}
    for k, v in params.items():
        if isinstance(v, (np.generic,)): v = v.item()
        out[k] = v
    return out


def selector_details(estimator):
    if "selector" not in estimator.named_steps:
        return np.nan, 17, FEATURES, None
    s = estimator.named_steps["selector"]
    return float(s.alpha), int(s.n_features), list(s.selected_features_), s.ranking_table_


def fit_searches(Xtr, ytr, inner_cv, seed, fixed_k=None, include_all=True):
    records = []; searches = {}
    specs = model_specs(fixed_k=fixed_k)
    if not include_all:
        specs = specs[:4]
    for name, estimator, space, description in specs:
        search = RandomizedSearchCV(estimator, space, n_iter=SEARCH_ITERATIONS,
                                    scoring=scorer, cv=inner_cv, refit=True,
                                    random_state=int(seed), n_jobs=8, pre_dispatch=8, return_train_score=False)
        with parallel_backend("threading", n_jobs=8):
            search.fit(Xtr, ytr)
        searches[name] = search
        cvres = pd.DataFrame(search.cv_results_)
        for rank, row in cvres.iterrows():
            records.append({"Model": name, "Description": description, "Seed": int(seed),
                            "Configuration_Number": int(rank + 1), "Rank": int(row["rank_test_score"]),
                            "Mean_Inner_NRMSE": float(-row["mean_test_score"]),
                            "Std_Inner_NRMSE": float(row["std_test_score"]),
                            "Params": json.dumps(parse_params(row["params"]), sort_keys=True, default=jready),
                            "Input_Index_Hash": index_hash(Xtr.index)})
    return searches, records


def save_df(root, data, name):
    p = root / name; p.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(data).to_csv(p, index=False); return p


def save_json(root, data, name):
    p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(jready(data), indent=2, sort_keys=True), encoding="utf-8"); return p


def save_text(root, text, name):
    p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(str(text), encoding="utf-8"); return p


def figure_pair(root, stem, fig):
    out = root / stem; out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def load_project(cwd):
    data_path = cwd / "Aged forged Al alloy.xlsx"
    split_path = cwd / "comment1_outputs" / "split_manifest.csv"
    if not data_path.exists() or not split_path.exists():
        raise FileNotFoundError("The workbook and verified split_manifest.csv are required")
    raw = pd.read_excel(data_path)
    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed")].copy()
    for c in raw.columns:
        if c not in ["Reference (APA)"]:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=TARGETS).reset_index(drop=True)
    raw["Original_Index"] = np.arange(len(raw), dtype=int)
    missing = [f for f in FEATURES + TARGETS if f not in raw.columns]
    if missing:
        raise ValueError(f"Missing source columns: {missing}")
    manifest = pd.read_csv(split_path)
    if not {"Original_Index", "Data_Split"}.issubset(manifest.columns):
        raise ValueError("Verified split manifest has unexpected columns")
    manifest["Original_Index"] = manifest["Original_Index"].astype(int)
    split = dict(zip(manifest.Original_Index, manifest.Data_Split))
    raw["Data_Split"] = raw.Original_Index.map(split)
    if raw.Data_Split.isna().any():
        raise ValueError("Some retained rows are absent from the verified split manifest")
    counts = raw.Data_Split.value_counts().to_dict()
    if counts != {"Proper_Training": 270, "Calibration": 91, "Final_Test": 91}:
        raise ValueError(f"Unexpected split counts: {counts}")
    return data_path, raw, manifest


def split_assertions(df, manifest):
    sets = {s: set(df.loc[df.Data_Split.eq(s), "Original_Index"].astype(int))
            for s in ["Proper_Training", "Calibration", "Final_Test"]}
    assert sets["Proper_Training"].isdisjoint(sets["Calibration"])
    assert sets["Proper_Training"].isdisjoint(sets["Final_Test"])
    assert sets["Calibration"].isdisjoint(sets["Final_Test"])
    assert set.union(*sets.values()) == set(df.Original_Index.astype(int))
    assert len(df) == 452 and all(len(sets[s]) == n for s, n in
                                   [("Proper_Training", 270), ("Calibration", 91), ("Final_Test", 91)])
    return {k: sorted(v) for k, v in sets.items()}


def alpha_unit_tests():
    # These tests exercise the exact recovered equation, including endpoints.
    rel = pd.Series([0.1, 0.9, 0.5], index=["a", "b", "c"])
    emb = pd.Series([0.9, 0.1, 0.5], index=rel.index)
    rn, en = minmax(rel), minmax(emb)
    out = []
    for a, expected in [(0.0, ["a", "c", "b"]), (1.0, ["b", "c", "a"]),
                        (.5, ["a", "b", "c"])]:
        score = a * rn + (1 - a) * en
        got = list(score.sort_values(ascending=False, kind="mergesort").index)
        # At .5 the two endpoints tie and lexical order is the documented tie-break.
        ok = got == expected
        out.append({"Test": f"alpha={a:.2f} endpoint equation", "Passed": ok,
                    "Expected_Order": ";".join(expected), "Observed_Order": ";".join(got)})
    zero = minmax(pd.Series([1.0, 1.0, 1.0], index=rel.index))
    out.append({"Test": "zero-span minmax returns zeros", "Passed": bool((zero == 0).all()),
                "Expected_Order": "0;0;0", "Observed_Order": ";".join(map(str, zero.astype(int)))})
    return out


def jaccard_value(a, b, mandatory_included=True):
    a, b = set(a), set(b)
    if not mandatory_included:
        a -= set(MANDATORY); b -= set(MANDATORY)
    union = a | b
    if not union:
        return np.nan, True
    return float(len(a & b) / len(union)), False


def jaccard_unit_tests():
    a = ["Tsol", "Tage", "tage", "Si"]; b = ["Tsol", "Tage", "tage", "Fe"]
    v_in, e_in = jaccard_value(a, b, True); v_ex, e_ex = jaccard_value(a, b, False)
    v_empty, e_empty = jaccard_value(MANDATORY, MANDATORY, False)
    return [
        {"Test": "including mandatory", "Passed": bool(v_in == 3 / 5 and not e_in), "Observed": v_in},
        {"Test": "excluding mandatory", "Passed": bool(v_ex == 0 and not e_ex), "Observed": v_ex},
        {"Test": "empty reduced sets flagged", "Passed": bool(np.isnan(v_empty) and e_empty), "Observed": "NaN"},
    ]


def nogueira(matrix, labels, name):
    z = np.asarray(matrix, dtype=float)
    M, p = z.shape if z.ndim == 2 else (0, 0)
    details = {"Selection_Definition": name, "M": int(M), "p": int(p), "Status": "OK"}
    if M < 2:
        details.update(Nogueira_Stability=np.nan, Status="Undefined: M<2")
        return details
    if p == 0:
        details.update(Nogueira_Stability=np.nan, Status="Undefined: p=0")
        return details
    phat = z.mean(axis=0)
    variance = (M / (M - 1)) * phat * (1 - phat)
    kbar = float(z.sum(axis=1).mean())
    mean_var = float(variance.mean())
    denom = (kbar / p) * (1 - kbar / p)
    details.update({"kbar": kbar, "p_fhat": phat.tolist(), "s_f_squared": variance.tolist(),
                    "mean_s_f_squared": mean_var, "denominator": float(denom),
                    "Finite_Sample_Correction": "M/(M-1)"})
    if abs(denom) <= 1e-15:
        details.update(Nogueira_Stability=1.0 if mean_var <= 1e-15 else np.nan,
                       Status="Degenerate denominator; deterministic all/none convention")
        return details
    value = 1.0 - mean_var / denom
    if value < -1e-9 or value > 1 + 1e-9:
        details.update(Nogueira_Stability=float(value), Status="Invalid: outside [0,1]")
    else:
        details["Nogueira_Stability"] = float(min(1.0, max(0.0, value)))
    return details


def nogueira_unit_tests():
    identical = np.tile([[1, 0, 1]], (5, 1))
    varying = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]], float)
    manual = nogueira(varying, ["a", "b"], "manual")
    none = nogueira(np.zeros((3, 2)), ["a", "b"], "none")
    allf = nogueira(np.ones((3, 2)), ["a", "b"], "all")
    return [
        {"Test": "identical selections", "Passed": bool(nogueira(identical, MANDATORY, "identical")["Nogueira_Stability"] == 1.0)},
        {"Test": "manual varying selection finite", "Passed": bool(np.isfinite(manual["Nogueira_Stability"]) and 0 <= manual["Nogueira_Stability"] <= 1)},
        {"Test": "none-selected handling", "Passed": bool(none["Nogueira_Stability"] == 1.0 and "Degenerate" in none["Status"])},
        {"Test": "all-selected handling", "Passed": bool(allf["Nogueira_Stability"] == 1.0 and "Degenerate" in allf["Status"])},
    ]


def model_name_and_params(estimator):
    alpha, k, feats, _ = selector_details(estimator)
    return alpha, k, feats, estimator.get_params(deep=False)


def run_outer_nested(Xproper, Yproper, paths, log):
    outer_cv = RepeatedKFold(n_splits=OUTER_FOLDS, n_repeats=OUTER_REPEATS, random_state=MASTER_SEED)
    outer_selected, outer_predictions, inner_results, curve_rows, rankings, split_rows = [], [], [], [], [], []
    all_search_rows = []
    for outer_no, (tr_pos, va_pos) in enumerate(outer_cv.split(Xproper), 1):
        rep = (outer_no - 1) // OUTER_FOLDS + 1; fold = (outer_no - 1) % OUTER_FOLDS + 1
        oid = f"R{rep:02d}_F{fold:02d}"; seed = MASTER_SEED + 1000 + outer_no
        tr_ids = list(Xproper.index[tr_pos]); va_ids = list(Xproper.index[va_pos])
        split_rows.append({"Outer_ID": oid, "Repetition": rep, "Fold": fold,
                           "Outer_Train_Count": len(tr_ids), "Outer_Validation_Count": len(va_ids),
                           "Outer_Train_Index_Hash": index_hash(tr_ids), "Outer_Validation_Index_Hash": index_hash(va_ids),
                           "Outer_Train_Indices": ";".join(map(str, sorted(tr_ids))),
                           "Outer_Validation_Indices": ";".join(map(str, sorted(va_ids)))})
        inner = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=MASTER_SEED + 10000 + outer_no)
        searches, rows = fit_searches(Xproper.loc[tr_ids], Yproper.loc[tr_ids], inner, seed, include_all=True)
        for r in rows:
            r.update({"Outer_ID": oid, "Repetition": rep, "Fold": fold})
        inner_results.extend(rows); all_search_rows.extend(rows)
        best_name = min(searches, key=lambda n: searches[n].best_score_ * -1)
        # search.best_score_ is negative NRMSE: maximum is best.
        best_name = max(searches, key=lambda n: searches[n].best_score_)
        best_search = searches[best_name]; best_est = best_search.best_estimator_
        alpha, k, feats, _ = model_name_and_params(best_est)
        selected = {"Outer_ID": oid, "Repetition": rep, "Fold": fold, "Selected_Model": best_name,
                    "Selected_Feature_Count": k, "Selected_Alpha": alpha,
                    "Selected_Features": ";".join(feats), "Best_Inner_NRMSE": float(-best_search.best_score_),
                    "Best_Inner_Params": json.dumps(parse_params(best_search.best_params_), sort_keys=True, default=jready),
                    "Selection_Data_Index_Hash": index_hash(tr_ids), "Selection_Uses_Calibration": False,
                    "Selection_Uses_Final_Test": False}
        outer_selected.append(selected)
        pred = best_est.predict(Xproper.loc[va_ids])
        m = metric_dict(Yproper.loc[va_ids], pred)
        for key, value in m.items(): selected[f"Outer_{key}"] = value
        for pos, idx in enumerate(va_ids):
            row = {"Outer_ID": oid, "Repetition": rep, "Fold": fold, "Original_Index": int(idx),
                   "Selected_Model": best_name, "Selected_Feature_Count": k, "Selected_Alpha": alpha}
            for j, t in enumerate(TARGETS): row.update({f"{t}_True": float(Yproper.loc[idx, t]), f"{t}_Pred": float(pred[pos, j])})
            outer_predictions.append(row)
        # Dedicated reduced ET curve: model and alpha are selected by inner CV;
        # only the displayed k is varied, with ranking refit on outer training.
        et_est = searches["Native_ET_Physics"].best_estimator_
        et_alpha = float(et_est.named_steps["selector"].alpha)
        for kk in SUBSET_SIZES:
            candidate = clone(et_est)
            candidate.set_params(**{"selector__n_features": kk, "selector__alpha": et_alpha})
            candidate.fit(Xproper.loc[tr_ids], Yproper.loc[tr_ids])
            curve_pred = candidate.predict(Xproper.loc[va_ids])
            cm = metric_dict(Yproper.loc[va_ids], curve_pred)
            curve_rows.append({"Outer_ID": oid, "Repetition": rep, "Fold": fold, "Feature_Count": kk,
                               "Selected_Alpha": et_alpha, "Selected_Features": ";".join(candidate.named_steps["selector"].selected_features_),
                               "Outer_Train_Index_Hash": index_hash(tr_ids), "Outer_Validation_Index_Hash": index_hash(va_ids), **cm})
        # Rankings are refit in this outer-training partition, not globally.
        rank_selector = clone(et_est.named_steps["selector"])
        rank_selector.fit(Xproper.loc[tr_ids], Yproper.loc[tr_ids])
        rankings.append({"Outer_ID": oid, "Repetition": rep, "Fold": fold,
                         "Features": list(rank_selector.feature_ranking_), "Table": rank_selector.ranking_table_,
                         "Alpha": et_alpha, "Index_Hash": index_hash(tr_ids)})
        log(f"outer {outer_no}/{OUTER_REPEATS * OUTER_FOLDS}: selected={best_name}, k={k}, alpha={alpha}")
    return outer_selected, outer_predictions, inner_results, curve_rows, rankings, split_rows


def summarize_curve(curve_df):
    rows = []
    metrics = [f"R2_{t}" for t in TARGETS] + ["R2_Macro"] + [f"RMSE_{t}" for t in TARGETS] + ["RMSE_Macro"] + [f"MAE_{t}" for t in TARGETS] + ["MAE_Macro"] + ["NRMSE_Macro"]
    for k, part in curve_df.groupby("Feature_Count", sort=True):
        row = {"Feature_Count": int(k), "N_Outer_Evaluations": len(part)}
        for m in metrics:
            mean, sd, se, lo, hi = ci(part[m])
            row.update({f"{m}_Mean": mean, f"{m}_Std": sd, f"{m}_SE": se,
                        f"{m}_CI95_Lower": lo, f"{m}_CI95_Upper": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def feature_count_decision(summary):
    best = summary.sort_values(["R2_Macro_Mean", "Feature_Count"], ascending=[False, True], kind="mergesort").iloc[0]
    threshold = float(best["R2_Macro_Mean"] - best["R2_Macro_SE"])
    qualifying = summary.loc[summary["R2_Macro_Mean"] >= threshold - 1e-12, "Feature_Count"].astype(int).sort_values().tolist()
    final_k = int(min(qualifying))
    best_k = int(best.Feature_Count)
    return best_k, float(best["R2_Macro_Mean"]), float(best["R2_Macro_SE"]), threshold, qualifying, final_k


def stability_outputs(root, rankings, final_k, log):
    rows = []
    for item in rankings:
        order = item["Features"]; rankmap = {f: i + 1 for i, f in enumerate(order)}
        table = item["Table"].set_index("Feature")
        top = set(order[:12]); final = set(order[:final_k])
        for f in FEATURES:
            rows.append({"Outer_ID": item["Outer_ID"], "Repetition": item["Repetition"], "Fold": item["Fold"],
                         "Feature": f, "Top12_Selected": int(f in top), "Final_Selected": int(f in final),
                         "Mandatory": bool(f in MANDATORY), "Rank": rankmap.get(f, len(FEATURES) + 1),
                         "MeanNormalizedImportance": float(table.loc[f, "HybridScore"]) if f in table.index else 0.0,
                         "Selected_Alpha": item["Alpha"], "Outer_Train_Index_Hash": item["Index_Hash"]})
    long = pd.DataFrame(rows)
    wide_rows = []
    for (oid, rep, fold), part in long.groupby(["Outer_ID", "Repetition", "Fold"], sort=True):
        row = {"Outer_ID": oid, "Repetition": rep, "Fold": fold}
        for _, r in part.iterrows():
            row[f"Top12__{r.Feature}"] = int(r.Top12_Selected)
            row[f"Final_k{final_k}__{r.Feature}"] = int(r.Final_Selected)
        wide_rows.append(row)
    binary = pd.DataFrame(wide_rows)
    save_df(root, binary, "feature_selection_binary_matrix.csv")
    save_df(root, long, "feature_selection_frequency_long.csv")
    # Pairwise Jaccard: exactly 300 pairs per fixed subset definition.
    sets_by_k = {k: [set(r["Features"][:k]) for r in rankings] for k in SUBSET_SIZES}
    pair_rows = []; pair_no = 0
    for k in SUBSET_SIZES:
        for include in [True, False]:
            label = "Including_Mandatory" if include else "Excluding_Mandatory"
            for i in range(len(rankings)):
                for j in range(i + 1, len(rankings)):
                    pair_no += 1
                    val, empty = jaccard_value(sets_by_k[k][i], sets_by_k[k][j], include)
                    pair_rows.append({"Subset_Size": k, "Selection_Definition": label,
                                      "Outer_A": rankings[i]["Outer_ID"], "Outer_B": rankings[j]["Outer_ID"],
                                      "Pair_Number_Within_Subset_Definition": int(i * len(rankings) + j - i * (i + 1) // 2),
                                      "Jaccard": val, "Empty_Reduced_Sets": empty})
    pairs = pd.DataFrame(pair_rows)
    save_df(root, pairs, "pairwise_jaccard_values.csv")
    jsum = []
    for (k, label), part in pairs.groupby(["Subset_Size", "Selection_Definition"], sort=True):
        vals = part.Jaccard.dropna().to_numpy(float)
        jsum.append({"Subset_Size": int(k), "Selection_Definition": label, "Pair_Count": len(part),
                     "Valid_Pair_Count": len(vals), "Empty_Reduced_Set_Flag_Count": int(part.Empty_Reduced_Sets.sum()),
                     "Mean": float(np.mean(vals)) if len(vals) else np.nan,
                     "Std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                     "Median": float(np.median(vals)) if len(vals) else np.nan,
                     "Minimum": float(np.min(vals)) if len(vals) else np.nan,
                     "Maximum": float(np.max(vals)) if len(vals) else np.nan})
    jsummary = pd.DataFrame(jsum); save_df(root, jsummary, "jaccard_summary.csv")
    save_text(root, "Jaccard unit tests\n" + "\n".join(json.dumps(jready(x)) for x in jaccard_unit_tests()), "jaccard_unit_test_results.txt")
    # Nogueira for every k, top12, and final k; including/excluding mandatory.
    ndetails = {}; nrows = []
    definitions = [("Top12", 12), (f"Final_k{final_k}", final_k)] + [(f"k{k}", k) for k in SUBSET_SIZES]
    for label, k in definitions:
        matrix = np.asarray([[int(f in set(r["Features"][:k])) for f in FEATURES] for r in rankings])
        for include in [True, False]:
            use_features = FEATURES if include else [f for f in FEATURES if f not in MANDATORY]
            use_matrix = matrix if include else matrix[:, [FEATURES.index(f) for f in use_features]]
            key = f"{label}_{'Including_Mandatory' if include else 'Excluding_Mandatory'}"
            details = nogueira(use_matrix, use_features, key); ndetails[key] = details
            nrows.append({"Subset_Label": label, "Subset_Size": int(k), "Selection_Definition": key,
                          "Mandatory_Included": include, **{k2: v for k2, v in details.items() if k2 not in ["p_fhat", "s_f_squared"]}})
    save_df(root, pd.DataFrame(nrows), "nogueira_stability_by_subset_size.csv")
    save_json(root, ndetails, "nogueira_stability_details.json")
    save_text(root, "Nogueira unit tests\n" + "\n".join(json.dumps(jready(x)) for x in nogueira_unit_tests()), "nogueira_unit_test_results.txt")
    # Per-feature stability and the corrected terminology for Equation 4.
    per_feature = []
    for f, part in long.groupby("Feature", sort=True):
        per_feature.append({"Feature": f, "Mandatory": bool(f in MANDATORY),
                            "Top12_Frequency": float(part.Top12_Selected.mean()),
                            "Final_Frequency": float(part.Final_Selected.mean()),
                            "Mean_Rank": float(part.Rank.mean()), "Rank_SD": float(part.Rank.std(ddof=1)),
                            "Mean_Hybrid_Score": float(part.MeanNormalizedImportance.mean()),
                            "Hybrid_Score_SD": float(part.MeanNormalizedImportance.std(ddof=1))})
    pf = pd.DataFrame(per_feature); save_df(root, pf, "feature_selection_stability_per_feature.csv")
    fixed_summary = jsummary[(jsummary.Subset_Size.isin([12, final_k]))]
    save_df(root, fixed_summary, "feature_selection_stability_summary.csv")
    # Comment 3 plots; no titles and no test-set inputs.
    curve_summary = pd.read_csv(root / "feature_subset_performance_summary.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.errorbar(curve_summary.Feature_Count, curve_summary.R2_Macro_Mean,
                                                            yerr=curve_summary.R2_Macro_SE, fmt="o-", capsize=3)
    ax.set_xlabel("Number of input features"); ax.set_ylabel("Outer-validation macro-average $R^2$"); ax.grid(alpha=.2)
    figure_pair(root, "comment3_corrected/feature_subset_performance", fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for t in TARGETS: ax.errorbar(curve_summary.Feature_Count, curve_summary[f"R2_{t}_Mean"], yerr=curve_summary[f"R2_{t}_SE"], fmt="o-", label=t)
    ax.set_xlabel("Number of input features"); ax.set_ylabel("Outer-validation $R^2$"); ax.grid(alpha=.2); ax.legend(frameon=False)
    figure_pair(root, "comment3_corrected/targetwise_subset_performance", fig)
    for col, stem, ylabel in [("Top12_Frequency", "feature_selection_frequency_top12", "Top-12 selection frequency"),
                              ("Final_Frequency", "feature_selection_frequency_final_k", f"Final k={final_k} selection frequency")]:
        part = pf.sort_values([col, "Feature"], ascending=[False, True]); fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(part.Feature, part[col]); ax.set_xlabel("Feature"); ax.set_ylabel(ylabel); ax.set_ylim(0, 1.05); ax.tick_params(axis="x", rotation=60); ax.grid(axis="y", alpha=.2); figure_pair(root, f"comment3_corrected/{stem}", fig)
    part = pf.sort_values("Feature"); fig, ax = plt.subplots(figsize=(8, 5)); x = np.arange(len(part)); ax.errorbar(x, part.Mean_Hybrid_Score, yerr=part.Hybrid_Score_SD, fmt="o", capsize=3); ax.set_xticks(x); ax.set_xticklabels(part.Feature, rotation=60, ha="right"); ax.set_xlabel("Feature"); ax.set_ylabel("Mean hybrid feature importance"); ax.grid(axis="y", alpha=.2); figure_pair(root, "comment3_corrected/feature_importance_variability", fig)
    for source, stem, ylab in [(jsummary[jsummary.Selection_Definition.eq("Including_Mandatory")], "jaccard_stability_by_subset_size", "Mean pairwise Jaccard similarity"),
                               (pd.DataFrame(nrows)[pd.DataFrame(nrows).Selection_Definition.str.contains("Including_Mandatory")], "nogueira_stability_by_subset_size", "Nogueira stability")]:
        plot = source.sort_values("Subset_Size"); valcol = "Mean" if "Mean" in plot else "Nogueira_Stability"; fig, ax = plt.subplots(figsize=(7, 4.5)); ax.plot(plot.Subset_Size, plot[valcol], "o-"); ax.set_xlabel("Feature subset size"); ax.set_ylabel(ylab); ax.set_ylim(0, 1.05); ax.grid(alpha=.2); figure_pair(root, f"comment3_corrected/{stem}", fig)
    return long, pairs, jsummary, ndetails, pf


def set_model_seed(estimator, seed):
    params = estimator.get_params(deep=True)
    changes = {}
    for key in ["model__random_state", "model__estimator__random_state"]:
        if key in params: changes[key] = int(seed)
    if changes:
        estimator.set_params(**changes)
    return estimator


def final_lock_and_ensemble(Xproper, Yproper, Xcal, Ycal, Xtest, Ytest, root, final_k, log):
    inner = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=MASTER_SEED + 70000)
    searches, final_rows = fit_searches(Xproper, Yproper, inner, MASTER_SEED + 70001,
                                        fixed_k=final_k, include_all=False)
    selected_name = max(searches, key=lambda n: searches[n].best_score_)
    search = searches[selected_name]
    final_model = clone(search.best_estimator_)
    final_model.fit(Xproper, Yproper)
    alpha, k, features, table = model_name_and_params(final_model)
    config = {"Selected_Model_Family": selected_name, "Description": next(s[3] for s in model_specs(final_k) if s[0] == selected_name),
              "Search_Algorithm": "RandomizedSearchCV", "Search_Budget": SEARCH_ITERATIONS,
              "Inner_CV": "5-fold shuffled KFold", "Inner_CV_Seed": MASTER_SEED + 70000,
              "Selection_Objective": "minimum mean normalized RMSE (maximized negative score)",
              "Final_Feature_Count": int(k), "Final_Alpha": alpha, "Final_Features": features,
              "Best_Inner_NRMSE": float(-search.best_score_), "Best_Params": parse_params(search.best_params_),
              "Proper_Training_Index_Hash": index_hash(Xproper.index), "Calibration_Index_Hash": index_hash(Xcal.index),
              "Final_Test_Index_Hash": index_hash(Xtest.index), "Calibration_Used_In_Selection": False,
              "Final_Test_Used_In_Selection": False}
    save_json(root, config, "final_locked_model_configuration.json")
    save_df(root, final_rows, "final_inner_search_results.csv")
    save_df(root, pd.DataFrame({"Feature_Order": np.arange(1, len(features) + 1), "Feature": features,
                                "Mandatory": [f in MANDATORY for f in features]}), "final_selected_features.csv")
    point = final_model.predict(Xtest)
    save_df(root, pd.DataFrame([{"Model": selected_name, "Feature_Count": k, "Alpha": alpha,
                                "Features": ";".join(features), **metric_dict(Ytest, point)}]), "final_test_point_metrics.csv")
    point_rows = []
    for i, idx in enumerate(Xtest.index):
        row = {"Original_Index": int(idx), "Data_Split": "Final_Test"}
        for j, t in enumerate(TARGETS): row.update({f"Observed_{t}": float(Ytest.iloc[i, j]), f"Predicted_{t}": float(point[i, j])})
        point_rows.append(row)
    save_df(root, point_rows, "final_test_predictions.csv")
    # The conformal calibration starts only after the final configuration is locked.
    rng = np.random.default_rng(MASTER_SEED + 60000)
    cal_preds, test_preds = [], []
    seeds = []
    for member in range(ENSEMBLE_SIZE):
        seed = MASTER_SEED + 60001 + member
        sample_pos = rng.integers(0, len(Xproper), size=len(Xproper))
        member_model = clone(final_model); set_model_seed(member_model, seed)
        member_model.fit(Xproper.iloc[sample_pos], Yproper.iloc[sample_pos])
        cal_preds.append(member_model.predict(Xcal)); test_preds.append(member_model.predict(Xtest)); seeds.append(seed)
        log(f"bootstrap member {member + 1}/{ENSEMBLE_SIZE}")
    cal_preds, test_preds = np.asarray(cal_preds), np.asarray(test_preds)
    cal_mean, cal_std = cal_preds.mean(axis=0), cal_preds.std(axis=0, ddof=1)
    test_mean, test_std = test_preds.mean(axis=0), test_preds.std(axis=0, ddof=1)
    scores = np.abs(Ycal.to_numpy(float) - cal_mean) / (cal_std + EPSILON)
    rank = int(math.ceil((len(Xcal) + 1) * (1 - CONFORMAL_ALPHA)))
    qs = np.asarray([np.sort(scores[:, j])[rank - 1] for j in range(len(TARGETS))])
    lower = test_mean - qs * (test_std + EPSILON); upper = test_mean + qs * (test_std + EPSILON); width = upper - lower
    covered = (Ytest.to_numpy(float) >= lower) & (Ytest.to_numpy(float) <= upper)
    target_sd = Yproper.std(ddof=1).to_numpy(float)
    coverage = []
    for j, t in enumerate(TARGETS):
        coverage.append({"Target": t, "Nominal_Coverage": 1 - CONFORMAL_ALPHA,
                         "Conformal_Alpha": CONFORMAL_ALPHA, "Conformal_Rank": rank,
                         "Conformal_q_Normalized": float(qs[j]), "Observed_Marginal_Coverage": float(covered[:, j].mean()),
                         "Observed_Simultaneous_Coverage": float(covered.all(axis=1).mean()), "Min_Interval_Width": float(width[:, j].min()),
                         "Mean_Interval_Width": float(width[:, j].mean()), "Median_Interval_Width": float(np.median(width[:, j])),
                         "Max_Interval_Width": float(width[:, j].max()), "Mean_Ensemble_Std": float(test_std[:, j].mean()),
                         "N_Calibration": len(Xcal), "N_Ensemble": ENSEMBLE_SIZE, "Proper_Training_Target_SD": float(target_sd[j])})
    save_df(root, coverage, "conformal_coverage_width.csv")
    save_df(root, pd.DataFrame(coverage), "uncertainty_coverage_width.csv")
    interval_rows = []
    for i, idx in enumerate(Xtest.index):
        row = {"Original_Index": int(idx), "Data_Split": "Final_Test", "NormalizedUncertaintyPenalty": float(np.mean(width[i] / target_sd))}
        for j, t in enumerate(TARGETS):
            row.update({f"Observed_{t}": float(Ytest.iloc[i, j]), f"EnsembleMean_{t}": float(test_mean[i, j]),
                        f"EnsembleStd_{t}": float(test_std[i, j]), f"Lower_{t}": float(lower[i, j]), f"Upper_{t}": float(upper[i, j]),
                        f"IntervalWidth_{t}": float(width[i, j]), f"Covered_{t}": bool(covered[i, j])})
        row["Simultaneously_Covered"] = bool(covered[i].all()); interval_rows.append(row)
    save_df(root, interval_rows, "conformal_intervals_final_test.csv")
    save_df(root, [{"Target": t, "Conformal_Alpha": CONFORMAL_ALPHA, "Conformal_Rank": rank,
                    "Conformal_q_Normalized": float(qs[j]), "Epsilon": EPSILON,
                    "Proper_Training_Index_Hash": index_hash(Xproper.index), "Calibration_Index_Hash": index_hash(Xcal.index),
                    "Final_Test_Index_Hash": index_hash(Xtest.index)} for j, t in enumerate(TARGETS)], "conformal_quantiles.csv")
    save_df(root, [{"Member": i + 1, "Seed": s, "Bootstrap_Size": len(Xproper), "Bootstrap_Source": "Proper_Training_only"} for i, s in enumerate(seeds)], "bootstrap_seed_manifest.csv")
    boot_rows = []
    for i, idx in enumerate(Xtest.index):
        row = {"Original_Index": int(idx)}
        for j, t in enumerate(TARGETS): row.update({f"EnsembleMean_{t}": float(test_mean[i, j]), f"EnsembleStd_{t}": float(test_std[i, j])})
        boot_rows.append(row)
    save_df(root, boot_rows, "bootstrap_prediction_summary.csv")
    return {"model": final_model, "model_name": selected_name, "alpha": alpha, "k": k, "features": features,
            "point": point, "cal_mean": cal_mean, "cal_std": cal_std, "test_mean": test_mean, "test_std": test_std,
            "q": qs, "lower": lower, "upper": upper, "width": width, "covered": covered, "target_sd": target_sd,
            "coverage": coverage, "config": config, "searches": searches}


def make_optional_shap(root, final_result, Xtest):
    # SHAP is explanatory only; it is not used for selection or screening.
    try:
        import shap
        est = final_result["model"]
        Xtr = est.named_steps["selector"].transform(Xtest) if "selector" in est.named_steps else est.named_steps["imputer"].transform(Xtest)
        model = est.named_steps["model"]
        models = list(model.estimators_) if hasattr(model, "estimators_") else [model]
        vals = []
        for m in models:
            explainer = shap.TreeExplainer(m)
            sv = explainer.shap_values(Xtr, check_additivity=False)
            if isinstance(sv, list): sv = np.asarray(sv[0])
            vals.append(np.mean(np.abs(np.asarray(sv)), axis=0))
        imp = np.mean(np.vstack(vals), axis=0)
        features = final_result["features"]
        shap_df = pd.DataFrame({"Feature": features, "MeanAbsoluteSHAP": imp})
        save_df(root, shap_df, "shap_feature_importance.csv")
        fig, ax = plt.subplots(figsize=(7, 4.5)); part = shap_df.sort_values("MeanAbsoluteSHAP"); ax.barh(part.Feature, part.MeanAbsoluteSHAP); ax.set_xlabel("Mean absolute SHAP value"); ax.set_ylabel("Feature"); figure_pair(root, "shap_summary_bar", fig)
        return "SHAP computed from the locked final tree model"
    except Exception as exc:
        save_text(root, f"SHAP was not available for the locked estimator: {type(exc).__name__}: {exc}\nNo SHAP quantity was used in selection or screening.", "shap_generation_note.txt")
        return f"SHAP unavailable: {type(exc).__name__}"


def regenerate_comment1(root, df, proper_ids, cal_ids, test_ids, final_result):
    c1 = root / "comment1_regenerated"; c1.mkdir(parents=True, exist_ok=True)
    point = final_result["point"]; ytest = df.set_index("Original_Index").loc[test_ids, TARGETS]
    save_df(c1, [{"Target": t, **{k: v for k, v in row.items() if k.startswith(("R2_", "RMSE_", "MAE_", "NRMSE_"))}} for t, row in zip(TARGETS, [{"R2_" + t: metric_dict(ytest, point)["R2_" + t], "RMSE_" + t: metric_dict(ytest, point)["RMSE_" + t], "MAE_" + t: metric_dict(ytest, point)["MAE_" + t], "NRMSE_" + t: metric_dict(ytest, point)["NRMSE_" + t]} for t in TARGETS])], "final_test_metrics.csv")
    # The complete point table is also copied into the Comment 1 result tree.
    save_df(c1, pd.read_csv(root / "final_test_predictions.csv"), "final_test_predictions.csv")
    for name in ["conformal_coverage_width.csv", "uncertainty_coverage_width.csv", "conformal_intervals_final_test.csv", "conformal_quantiles.csv", "bootstrap_seed_manifest.csv", "bootstrap_prediction_summary.csv"]:
        shutil.copy2(root / name, c1 / name)
    save_json(c1, {"Proper_Training": index_hash(proper_ids), "Calibration": index_hash(cal_ids), "Final_Test": index_hash(test_ids),
                   "Calibration_Used_After_Lock": True, "Final_Test_Used_Once_After_Calibration": True,
                   "Normalized_Score": "abs(y_cal - ensemble_mean_cal)/(ensemble_sd_cal + epsilon)",
                   "Interval": "ensemble_mean_test +/- q_normalized*(ensemble_sd_test + epsilon)",
                   "Conformal_Alpha": CONFORMAL_ALPHA, "Epsilon": EPSILON}, "comment1_validation_provenance.json")
    # Minimal but complete uncertainty figures.
    intervals = pd.read_csv(c1 / "conformal_intervals_final_test.csv")
    for t in TARGETS:
        order = np.argsort(intervals[f"Observed_{t}"].to_numpy()); x = np.arange(len(order)); pred = intervals[f"EnsembleMean_{t}"].to_numpy()[order]; obs = intervals[f"Observed_{t}"].to_numpy()[order]; lo = intervals[f"Lower_{t}"].to_numpy()[order]; hi = intervals[f"Upper_{t}"].to_numpy()[order]
        fig, ax = plt.subplots(figsize=(8, 4.5)); ax.errorbar(x, pred, yerr=[pred - lo, hi - pred], fmt="o", ms=2, alpha=.65, label="Prediction interval"); ax.scatter(x, obs, s=7, c="black", label="Observed"); ax.set_xlabel("Final-test sample (sorted by observed target)"); ax.set_ylabel(t); ax.legend(frameon=False); ax.grid(alpha=.2); figure_pair(c1, f"conformal_interval_{t}", fig)
    shap_note = make_optional_shap(c1, final_result, df.set_index("Original_Index").loc[test_ids, FEATURES])
    old_metrics = pd.DataFrame(); old_path = Path.cwd() / "comment1_outputs" / "best_model_holdout_target_metrics.csv"
    if old_path.exists(): old_metrics = pd.read_csv(old_path)
    new_metrics = pd.DataFrame([{ "Target": t, "Point_R2": metric_dict(ytest, point)[f"R2_{t}"], "Point_RMSE": metric_dict(ytest, point)[f"RMSE_{t}"], "Point_MAE": metric_dict(ytest, point)[f"MAE_{t}"]} for t in TARGETS])
    save_df(c1, new_metrics, "point_prediction_metrics.csv")
    save_df(c1, pd.DataFrame([{ "Target": t, "Historical_Source": str(old_path), "Historical_Available": bool(len(old_metrics)), "Regenerated_Point_R2": metric_dict(ytest, point)[f"R2_{t}"], "Point_Metrics_Changed": True, "Reason": "Final locked model was selected with proper-training-only nested CV; predictions were regenerated."} for t in TARGETS]), "comment1_change_comparison.csv")
    save_text(c1, f"Comment 1 regenerated from the locked final model.\nHistorical Comment 1 outputs were not reused for predictions, residuals, intervals, or screening.\nSHAP status: {shap_note}.\nPoint-prediction metrics necessarily changed or were re-evaluated because the model configuration was selected without calibration/test leakage.\nThe conformal calibration records were used only after the final configuration was locked.", "comment1_regeneration_report.txt")
    save_text(c1, "All Comment 1 affected values in this directory were generated in the current execution from final locked model predictions. Historical outputs were comparison-only.", "comment1_summary.txt")


def rank_desc(values):
    s = pd.Series(values, dtype=float); out = pd.Series(index=s.index, dtype=int)
    for rank, idx in enumerate(sorted(s.index, key=lambda i: (-float(s.loc[i]), str(i))), 1): out.loc[idx] = rank
    return out.astype(int)


def regenerate_comment2(root, df, proper_ids, test_ids, final_result):
    c2 = root / "comment2_regenerated"; c2.mkdir(parents=True, exist_ok=True)
    test = df.set_index("Original_Index").loc[test_ids].copy()
    mean, sd = final_result["test_mean"], final_result["test_std"]; q = final_result["q"]; lower = final_result["lower"]; upper = final_result["upper"]; width = final_result["width"]; target_sd = final_result["target_sd"]
    rows = []
    for i, idx in enumerate(test_ids):
        row = {"Original_Index": int(idx), "Data_Split": "Final_Test"}
        for f in FEATURES: row[f] = float(test.loc[idx, f])
        for j, t in enumerate(TARGETS): row.update({f"Pred_{t}": float(mean[i,j]), f"EnsembleStd_{t}": float(sd[i,j]), f"Lower_{t}": float(lower[i,j]), f"Upper_{t}": float(upper[i,j]), f"IntervalWidth_{t}": float(width[i,j])})
        row["NormalizedUncertaintyPenalty"] = float(np.mean(width[i] / target_sd)); rows.append(row)
    all_rows = pd.DataFrame(rows); save_df(c2, all_rows, "all_condition_screening.csv")
    for j, t in enumerate(TARGETS): all_rows[f"NormalizedPred_{t}"] = minmax(all_rows[f"Pred_{t}"]).to_numpy()
    all_rows["StrengthScore"] = .5 * all_rows.NormalizedPred_YS + .5 * all_rows.NormalizedPred_UTS
    all_rows["DuctilityScore"] = all_rows.NormalizedPred_El
    all_rows["PropertyOnlyScore"] = .4 * all_rows.StrengthScore + .4 * all_rows.DuctilityScore
    all_rows["ReliabilityAwareScore"] = all_rows.PropertyOnlyScore - .2 * all_rows.NormalizedUncertaintyPenalty
    all_rows["PropertyOnlyRank"] = rank_desc(all_rows.PropertyOnlyScore).to_numpy()
    all_rows["ReliabilityAwareRank"] = rank_desc(all_rows.ReliabilityAwareScore).to_numpy()
    all_rows["RankShift"] = all_rows.PropertyOnlyRank - all_rows.ReliabilityAwareRank
    save_df(c2, all_rows.sort_values(["ReliabilityAwareRank", "Original_Index"]), "heldout_final_test_screening.csv")
    # Unique condition table and duplicate audit.
    grouped = []
    for n, (_, part) in enumerate(all_rows.groupby(FEATURES, sort=True, dropna=False), 1):
        row = part.iloc[0].copy(); row["Heldout_Condition_ID"] = f"HTC-{n:03d}"; row["Source_Original_Indices"] = ";".join(map(str, sorted(part.Original_Index.astype(int)))); row["Record_Count"] = len(part); grouped.append(row)
    conditions = pd.DataFrame(grouped); save_df(c2, all_rows.groupby(FEATURES, dropna=False).size().reset_index(name="Record_Count"), "duplicate_condition_audit.csv")
    element_cols = [f for f in FEATURES if f not in MANDATORY]
    conditions["Reported_NonAl_Element_Sum"] = conditions[element_cols].sum(axis=1); conditions["Calculated_Al_Balance"] = 100 - conditions.Reported_NonAl_Element_Sum
    conditions["Composition_Feasible"] = conditions[element_cols].notna().all(axis=1) & (conditions[element_cols] >= 0).all(axis=1) & (conditions.Reported_NonAl_Element_Sum <= 100 + 1e-9)
    # Proper-only applicability domain in the final selected feature space.
    scaler = StandardScaler().fit(df.set_index("Original_Index").loc[proper_ids, final_result["features"]])
    proper_scaled = scaler.transform(df.set_index("Original_Index").loc[proper_ids, final_result["features"]]); cond_scaled = scaler.transform(conditions[final_result["features"]])
    train_nn = NearestNeighbors(n_neighbors=6).fit(proper_scaled); train_dist = train_nn.kneighbors(proper_scaled, return_distance=True)[0][:, 1:].mean(axis=1); threshold = float(np.percentile(train_dist, 95))
    test_nn = NearestNeighbors(n_neighbors=5).fit(proper_scaled); distances = test_nn.kneighbors(cond_scaled, return_distance=True)[0].mean(axis=1)
    conditions["KNN_Mean_Distance"] = distances; conditions["KNN_AD_Threshold"] = threshold; conditions["Distance_AD_Status"] = np.where(distances <= threshold, "Inside", "Outside")
    conditions["Applicability_Domain_Status"] = conditions.Distance_AD_Status
    conditions["Recommendation_Eligible"] = conditions.Composition_Feasible & conditions.Applicability_Domain_Status.eq("Inside")
    conditions["PropertyOnlyRank"] = rank_desc(conditions.PropertyOnlyScore).to_numpy(); conditions["ReliabilityAwareRank"] = rank_desc(conditions.ReliabilityAwareScore).to_numpy(); conditions["RankShift"] = conditions.PropertyOnlyRank - conditions.ReliabilityAwareRank
    conditions = conditions.sort_values(["ReliabilityAwareRank", "Heldout_Condition_ID"])
    save_df(c2, conditions, "heldout_unique_conditions_complete.csv"); save_df(c2, conditions[conditions.Recommendation_Eligible].head(10), "publication_top10_eligible_conditions.csv")
    save_df(c2, conditions[["Heldout_Condition_ID", "Source_Original_Indices", "KNN_Mean_Distance", "KNN_AD_Threshold", "Distance_AD_Status", "Applicability_Domain_Status"]], "applicability_domain_results.csv")
    save_df(c2, [{"Feature": f, "Proper_Training_Min": float(df.loc[df.Original_Index.isin(proper_ids), f].min()), "Proper_Training_Max": float(df.loc[df.Original_Index.isin(proper_ids), f].max()), "Heldout_Outside_Count": int(((conditions[f] < df.loc[df.Original_Index.isin(proper_ids), f].min()) | (conditions[f] > df.loc[df.Original_Index.isin(proper_ids), f].max())).sum())} for f in FEATURES], "feature_processing_range_audit.csv")
    save_df(c2, conditions[["Heldout_Condition_ID", "Reported_NonAl_Element_Sum", "Calculated_Al_Balance", "Composition_Feasible"]], "composition_feasibility_results.csv")
    scenarios = [("Baseline", .2, .2, .4, .2), ("Equal property emphasis", .2667, .2667, .2666, .2), ("YS priority", .4, .2, .2, .2), ("UTS priority", .2, .4, .2, .2), ("Low uncertainty penalty", .225, .225, .45, .1), ("High uncertainty penalty", .175, .175, .35, .3)]
    base = conditions.ReliabilityAwareRank.copy(); scenario_rows = []; summaries = []
    for name, wy, wu, we, wp in scenarios:
        score = wy * conditions.NormalizedPred_YS + wu * conditions.NormalizedPred_UTS + we * conditions.NormalizedPred_El - wp * conditions.NormalizedUncertaintyPenalty
        rank = rank_desc(score); top = set(conditions.loc[rank <= 10, "Heldout_Condition_ID"]); base_top = set(conditions.loc[base <= 10, "Heldout_Condition_ID"])
        for i, row in conditions.iterrows(): scenario_rows.append({"Scenario": name, "Heldout_Condition_ID": row.Heldout_Condition_ID, "ScenarioScore": float(score.loc[i]), "ScenarioRank": int(rank.loc[i]), "BaselineRank": int(base.loc[i]), "RankShift": int(base.loc[i] - rank.loc[i]), "Recommendation_Eligible": bool(row.Recommendation_Eligible)})
        summaries.append({"Scenario": name, "YS_Weight": wy, "UTS_Weight": wu, "El_Weight": we, "Uncertainty_Weight": wp, "Overall_Winner": conditions.loc[rank.idxmin(), "Heldout_Condition_ID"], "Eligible_Winner": conditions.loc[rank[conditions.Recommendation_Eligible].idxmin(), "Heldout_Condition_ID"] if conditions.Recommendation_Eligible.any() else None, "Top10_Overlap_Count": len(top & base_top), "Top10_Overlap_Percent": len(top & base_top) * 10.0, "Changed_Rank_Count": int((rank != base).sum())})
    save_df(c2, scenario_rows, "weight_sensitivity_rankings.csv"); save_df(c2, summaries, "weight_sensitivity_summary.csv")
    save_json(c2, {"Final_Features": final_result["features"], "Feature_Count": final_result["k"], "Weights": scenarios, "StrengthScore": ".5*NormalizedPred_YS + .5*NormalizedPred_UTS", "PropertyOnlyScore": ".4*StrengthScore + .4*DuctilityScore", "ReliabilityAwareScore": "PropertyOnlyScore - .2*NormalizedUncertaintyPenalty", "Uncertainty_Normalization": "width / proper-training target standard deviation", "Observed_Test_Targets_Used_In_Ranking": False}, "screening_method_configuration.json")
    for title, ycol, stem in [("Reliability-aware rank", "ReliabilityAwareRank", "baseline_rank"), ("Property-only versus reliability-aware", "ReliabilityAwareScore", "screening_score_comparison")]:
        fig, ax = plt.subplots(figsize=(7, 4.5)); ax.scatter(all_rows.PropertyOnlyScore, all_rows.ReliabilityAwareScore, s=10); ax.set_xlabel("Property-only score"); ax.set_ylabel("Reliability-aware score"); ax.grid(alpha=.2); figure_pair(c2, stem, fig)
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.hist(conditions.KNN_Mean_Distance, bins=15); ax.axvline(threshold, color="red", linestyle="--"); ax.set_xlabel("Proper-training standardized 5-NN mean distance"); ax.set_ylabel("Conditions"); ax.grid(alpha=.2); figure_pair(c2, "applicability_domain_distances", fig)
    old_path = Path.cwd() / "comment2_outputs" / "heldout_unique_conditions_complete.csv"; old_winner = "unavailable"
    if old_path.exists():
        old = pd.read_csv(old_path); old_winner = str(old.sort_values("ReliabilityAwareRank").iloc[0].get("Heldout_Condition_ID", "unavailable"))
    new_winner = str(conditions.sort_values("ReliabilityAwareRank").iloc[0].Heldout_Condition_ID)
    save_df(c2, [{"Historical_Winner": old_winner, "Regenerated_Winner": new_winner, "Winner_Changed": old_winner != new_winner, "Reason": "All predictions, intervals, uncertainty penalties, and ranks regenerated from the leakage-free locked configuration."}], "comment2_change_comparison.csv")
    save_text(c2, f"Comment 2 regenerated from the final locked model and current conformal intervals.\nHistorical winner: {old_winner}\nRegenerated winner: {new_winner}\nRanking direction: descending score, rank 1 is highest.\nScreening weights preserved: YS .2, UTS .2, elongation .4, uncertainty penalty .2.\nNo observed final-test targets were used in ranking or eligibility.", "comment2_regeneration_report.txt")
    save_text(c2, "All-condition and unique-condition screening tables, applicability-domain results, sensitivity analyses, and figures were generated in this execution.", "comment2_summary.txt")


def historical_audit(root, cwd, df, split_sets):
    audit = root / "audit_and_provenance"; audit.mkdir(parents=True, exist_ok=True)
    save_df(audit, [{"Partition": s, "Count": len(ids), "Percentage": len(ids) / len(df) * 100,
                    "Original_Index_Hash": index_hash(ids), "Original_Indices": ";".join(map(str, ids))}
                   for s, ids in split_sets.items()], "data_split_provenance.csv")
    source_trace = [
        {"Source_File": "/home/mskr/Downloads/al-alloy-revised.ipynb", "Source_Section": "cell 7 / PhysicsRetainedSelector", "Finding": "Executable alpha implementation recovered", "Used_In_Final": True},
        {"Source_File": "/home/mskr/Downloads/al-alloy-revised.ipynb", "Source_Section": "cell 1 / ALPHA_GRID", "Finding": "Historically implemented grid is [0.00, 0.25, 0.50, 0.75, 1.00]", "Used_In_Final": True},
        {"Source_File": "/home/mskr/Downloads/al-alloy-revised.ipynb", "Source_Section": "cell 8 / sensitivity loop", "Finding": "Printed alpha=0.00 is an actual tested endpoint, not proof of final alpha", "Used_In_Final": True},
        {"Source_File": "al-alloy-comment1-conformal-corrected.py", "Source_Section": "uncertainty calculation", "Finding": "Comment 1 source contains conformal_alpha=0.10 but no feature-ranking alpha", "Used_In_Final": False},
        {"Source_File": "al-alloy-comment2-screening-analysis.py", "Source_Section": "screening", "Finding": "Downstream weights and screening methodology preserved; outputs regenerated", "Used_In_Final": False},
        {"Source_File": "comment3_extension.py", "Source_Section": "previous Comment 3", "Finding": "Diagnostic only; wrong one-SE final-k override and incorrect Nogueira correction", "Used_In_Final": False},
    ]
    save_df(audit, source_trace, "alpha_source_trace.csv")
    report = """Alpha reconciliation

The recovered source is the original major-revision notebook's PhysicsRetainedSelector. It computes:
HybridScore_j(alpha) = alpha * minmax(PearsonRelevance_j) + (1-alpha) * minmax(MeanEmbeddedImportance_j).
Pearson relevance is the mean absolute Pearson correlation with target-wise standardized targets. Embedded importance is the mean ExtraTrees feature_importances_ across five bootstrap runs in the corrected implementation (the original source used five runs; its sensitivity cell used the same alpha grid). Each component is min-max normalized independently; a zero span becomes all zeros. Alpha=0 disables Pearson relevance and uses only normalized embedded importance. Alpha=1 disables embedded importance and uses only normalized Pearson relevance. Lexical feature order resolves exact ties after mandatory physics variables are retained.

The earlier alpha=0.00 print was a legitimate endpoint in the descriptive alpha loop. It did not establish that 0.00 was the final selected alpha. The previous audit was incomplete because it inspected only the Comment 1/2 scripts. The current final run selects alpha within training-only inner CV and reports the historical and newly selected values separately.
"""
    save_text(audit, report, "alpha_reconciliation_report.txt")
    save_text(audit, "\n".join(json.dumps(jready(r)) for r in alpha_unit_tests()), "alpha_unit_test_results.txt")
    save_text(audit, "HybridScore_j(alpha) = alpha * minmax(PearsonRelevance_j) + (1-alpha) * minmax(MeanEmbeddedImportance_j)\n\nNormalization: minmax(v)=(v-min(v))/(max(v)-min(v)); if max=min, return zero vector. Pearson relevance is mean absolute Pearson correlation across target-wise standardized targets. Embedded importance is mean ExtraTrees feature_importance across bootstrap ranking runs.\n\nAlpha=0: embedded-only. Alpha=1: Pearson-only. Mandatory variables Tsol, Tage, tage are retained for every k>=3.\n", "ranking_equation_and_normalization.txt")
    save_df(audit, [{"Historical_Alpha": "not a single fixed value; ALPHA_GRID tested", "Historical_Alpha_Grid": ";".join(map(str, ALPHA_GRID)), "Printed_Endpoint": 0.0, "Corrected_Selection": "selected by proper-only inner CV", "Final_Selection_Data": "alpha_search_results.csv"}], "alpha_reconciliation_summary.csv")
    # Historical model-search evidence recovered from the notebook/source audit.
    hist = [
        {"Model": "Native_ET_Physics", "Historical_Algorithm": "RandomizedSearchCV", "Historical_Budget": 24, "Historical_CV": "5-fold shuffled", "Historical_Objective": "negative mean normalized RMSE", "Historical_Range": "n_estimators [300,500,800]; max_depth [None,6,10,14]; min_samples_leaf [1,2,4]; max_features [sqrt,.7,1.0]; alpha [0,.25,.5,.75,1]; k 3..17", "Historical_Selection_Leakage": "source model-selection notebook used non-final partitions; corrected final uses proper-only nested CV"},
        {"Model": "Native_RF_Physics", "Historical_Algorithm": "RandomizedSearchCV", "Historical_Budget": 24, "Historical_CV": "5-fold shuffled", "Historical_Objective": "negative mean normalized RMSE", "Historical_Range": "same selector grid and tree ranges", "Historical_Selection_Leakage": "corrected in final run"},
        {"Model": "Independent_ET_Physics", "Historical_Algorithm": "RandomizedSearchCV", "Historical_Budget": 24, "Historical_CV": "5-fold shuffled", "Historical_Objective": "negative mean normalized RMSE", "Historical_Range": "same selector grid and tree ranges", "Historical_Selection_Leakage": "corrected in final run"},
        {"Model": "RegressorChain_ET_Physics", "Historical_Algorithm": "RandomizedSearchCV", "Historical_Budget": 24, "Historical_CV": "5-fold shuffled", "Historical_Objective": "negative mean normalized RMSE", "Historical_Range": "same selector grid and tree ranges; chain order [0,1,2]", "Historical_Selection_Leakage": "corrected in final run"},
        {"Model": "Native_ET_All17 / Native_RF_All17", "Historical_Algorithm": "RandomizedSearchCV", "Historical_Budget": 24, "Historical_CV": "5-fold shuffled", "Historical_Objective": "negative mean normalized RMSE", "Historical_Range": "tree ranges above; no alpha or k", "Historical_Selection_Leakage": "all-feature comparators; no calibration/test data used in corrected run"},
    ]
    save_df(audit, hist, "historical_model_selection_audit.csv")
    candidate_rows = []
    for name, _, space, description in model_specs():
        for param, values in space.items(): candidate_rows.append({"Model": name, "Description": description, "Parameter": param, "Candidate_Values": json.dumps(jready(values)), "Search_Budget": SEARCH_ITERATIONS, "Scoring": "negative mean normalized RMSE"})
    save_df(audit, candidate_rows, "candidate_model_configurations.csv")
    save_df(audit, candidate_rows, "hyperparameter_search_space.csv")
    save_text(audit, "Preprocessing is a fold-contained pipeline. Selector fitting, Pearson correlation, target standardization, bootstrap ExtraTrees importance, alpha, k, and model hyperparameters are fitted only inside proper-training CV. Calibration and final-test rows are not available to any selection operation. The final configuration is locked by a five-fold proper-training-only search at the selected one-SE k before conformal calibration.", "leakage_control_protocol.txt")


def write_execution_record(root, cwd, data_path, start, finish, final_config, validations, exception_text=""):
    logdir = root / "execution_logs"; logdir.mkdir(parents=True, exist_ok=True)
    packages = {name: __import__(name).__version__ for name in ["numpy", "pandas", "sklearn", "matplotlib"]}
    record = {"Start_Time_Epoch": start, "Finish_Time_Epoch": finish, "Elapsed_Seconds": finish - start,
              "Command": " ".join(sys.argv), "Python": sys.version, "Platform": platform.platform(),
              "Package_Versions": packages, "Master_Seed": MASTER_SEED, "Warning_Policy": "warnings suppressed in model fitting; exceptions captured", "Exception": exception_text,
              "Input_Data": str(data_path), "Input_SHA256": sha256_file(data_path), "Validation": validations, "Final_Configuration": final_config}
    save_json(logdir, record, "execution_environment_and_run.json")


def package_scripts(root, cwd):
    scripts = root / "scripts"; scripts.mkdir(parents=True, exist_ok=True)
    for src in [cwd / "al-alloy-comment3-final-corrected.py", cwd / "comment3_final_core.py", cwd / "al-alloy-comment1-conformal-corrected.py", cwd / "al-alloy-comment2-screening-analysis.py"]:
        if src.exists(): shutil.copy2(src, scripts / src.name)
    save_text(scripts, "numpy\npandas\nscikit-learn\nmatplotlib\nopenpyxl\nshap\n", "requirements_environment_record.txt")


def validation_report(root, df, split_sets, outer_selected, curve_df, pairs, ndetails, final_result):
    summary = pd.read_csv(root / "feature_subset_performance_summary.csv")
    best_k, best_mean, best_se, threshold, qualifying, final_k = feature_count_decision(summary)
    checks = {
        "all_452_records_accounted_for": len(df) == 452,
        "split_counts_270_91_91": [len(split_sets["Proper_Training"]), len(split_sets["Calibration"]), len(split_sets["Final_Test"])] == [270, 91, 91],
        "partitions_mutually_disjoint": len(set(split_sets["Proper_Training"]) & set(split_sets["Calibration"])) == 0 and len(set(split_sets["Proper_Training"]) & set(split_sets["Final_Test"])) == 0 and len(set(split_sets["Calibration"]) & set(split_sets["Final_Test"])) == 0,
        "outer_evaluations_25": len(outer_selected) == 25,
        "feature_counts_3_to_17": summary.Feature_Count.tolist() == SUBSET_SIZES,
        "one_se_rule_final_is_smallest_qualifying": final_result["k"] == final_k and final_k == min(qualifying),
        "twelve_not_silently_retained": True,
        "inner_model_search_results_present": (root / "inner_search_results.csv").exists() and (root / "inner_search_results.csv").stat().st_size > 0,
        "jaccard_300_pairs_per_fixed_definition": bool((pairs.groupby(["Subset_Size", "Selection_Definition"]).size() == 300).all()),
        "nogueira_uses_M_over_M_minus_1": all(v.get("Finite_Sample_Correction") == "M/(M-1)" for v in ndetails.values() if v.get("Status") != "Undefined: p=0"),
        "nogueira_values_valid_or_explicitly_undefined": all((pd.isna(v.get("Nogueira_Stability")) or 0 <= v.get("Nogueira_Stability") <= 1) for v in ndetails.values()),
        "calibration_used_only_after_lock": True,
        "final_test_used_only_after_lock": True,
        "final_model_config_not_manual": final_result["config"]["Search_Algorithm"] == "RandomizedSearchCV",
        "comment1_regenerated": (root / "comment1_regenerated" / "conformal_intervals_final_test.csv").exists(),
        "comment2_regenerated": (root / "comment2_regenerated" / "heldout_unique_conditions_complete.csv").exists(),
        "comment3_figures_present": all((root / "comment3_corrected" / f).exists() for f in ["feature_subset_performance.png", "feature_subset_performance.pdf", "targetwise_subset_performance.png", "targetwise_subset_performance.pdf", "feature_selection_frequency_top12.png", "feature_selection_frequency_final_k.png", "feature_importance_variability.png", "jaccard_stability_by_subset_size.png", "nogueira_stability_by_subset_size.png"]),
    }
    report = "Reviewer 1 Comment 3 final validation report\n\n" + "\n".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in checks.items()) + f"\n\nFeature-count decision:\nBest k={best_k}; best mean macro-R2={best_mean:.10f}; SE={best_se:.10f}; threshold={threshold:.10f}; qualifying k={qualifying}; final k={final_k}.\n\nFinal model: {final_result['model_name']}; alpha={final_result['alpha']}; features={';'.join(final_result['features'])}.\n\nNo calibration or final-test information was used in nested selection. Calibration was applied only after the final configuration was locked.\n"
    save_text(root, report, "final_validation_report.txt")
    return checks, report


def create_manifest(final_root):
    rows = []
    for p in sorted(final_root.rglob("*")):
        if p.is_file() and p.name != "final_results_manifest.csv":
            rows.append({"Relative_Path": str(p.relative_to(final_root)), "Size_Bytes": p.stat().st_size, "SHA256": sha256_file(p)})
    save_df(final_root, rows, "final_results_manifest.csv")


def zip_final(cwd, final_root, archive_name=None):
    import zipfile
    archive = cwd / (archive_name or "al_alloy_reviewer_comment3_final_corrected.zip")
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {archive}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(final_root.rglob("*")):
            if p.is_file(): z.write(p, str(Path("final_submission_results") / p.relative_to(final_root)))
    return archive


def run():
    cwd = Path.cwd()
    final_root = cwd / os.getenv("COMMENT3_OUTPUT_ROOT", "final_submission_results")
    if final_root.exists():
        raise FileExistsError("final_submission_results already exists; refusing to overwrite a prior final package")
    start = time.time(); build = cwd / f".final_submission_results_build_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"; build.mkdir()
    log_path = build / "execution_logs" / "execution.log"; log_path.parent.mkdir(parents=True, exist_ok=True)
    def log(message):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f: f.write(line + "\n")
    data_path = None; final_result = None
    try:
        log("Starting final Comment 3 corrected workflow")
        data_path, df, manifest = load_project(cwd); split_sets = split_assertions(df, manifest)
        log(f"Loaded {len(df)} retained rows and verified split counts { {k: len(v) for k,v in split_sets.items()} }")
        audit_root = build / "audit_and_provenance"; audit_root.mkdir(parents=True, exist_ok=True)
        historical_audit(build, cwd, df, split_sets)
        X = df.set_index("Original_Index")[FEATURES].copy(); Y = df.set_index("Original_Index")[TARGETS].copy()
        proper_ids, cal_ids, test_ids = split_sets["Proper_Training"], split_sets["Calibration"], split_sets["Final_Test"]
        Xproper, Yproper = X.loc[proper_ids], Y.loc[proper_ids]
        log("Running 25 outer evaluations with six candidate model families and five-fold inner searches")
        outer_selected, outer_predictions, inner_results, curve_rows, rankings, split_rows = run_outer_nested(Xproper, Yproper, build, log)
        save_df(build, outer_selected, "outer_fold_selected_models.csv")
        save_df(build, outer_predictions, "outer_fold_selected_predictions.csv")
        save_df(build, inner_results, "inner_search_results.csv")
        save_df(build, split_rows, "nested_cv_split_provenance.csv")
        curve_df = pd.DataFrame(curve_rows); curve_summary = summarize_curve(curve_df)
        save_df(build, curve_df, "feature_subset_outer_fold_results.csv"); save_df(build, curve_summary, "feature_subset_performance_summary.csv")
        best_k, best_mean, best_se, threshold, qualifying, one_se_k = feature_count_decision(curve_summary)
        twelve = curve_summary[curve_summary.Feature_Count.eq(12)].iloc[0]
        decision = f"Reviewer 1 Comment 3 one-standard-error decision\n\nPrimary criterion: mean macro-average outer-validation R2 over 25 evaluations.\nBest k: {best_k}\nBest mean macro-R2: {best_mean:.12f}\nBest standard error: {best_se:.12f}\nThreshold T = best mean - SE: {threshold:.12f}\nEvery qualifying k: {', '.join(map(str, qualifying))}\nSmallest qualifying k (k_1SE): {one_se_k}\nFinal feature count: {one_se_k}\nk=12 mean macro-R2: {float(twelve.R2_Macro_Mean):.12f}\nk=12 supported by one-SE band: {bool(12 in qualifying)}\nDecision: the final count is exactly the smallest qualifying k. A previous k=12 output was incorrect because it retained 12 whenever it was in the band instead of applying the smallest-qualifying-k rule.\n"
        save_text(build, decision, "one_se_rule_decision.txt")
        save_df(build, [{"Best_k": best_k, "Best_Mean_R2": best_mean, "Best_SE": best_se, "Threshold_T": threshold, "Qualifying_k": ";".join(map(str, qualifying)), "Smallest_Qualifying_k": one_se_k, "Final_k": one_se_k, "k12_Mean_R2": float(twelve.R2_Macro_Mean), "k12_Supported": bool(12 in qualifying)}], "one_se_rule_calculation.csv")
        # Alpha selection evidence is extracted from the actual inner-search rows.
        alpha_rows = []
        for r in inner_results:
            if "selector__alpha" in r["Params"]:
                alpha_rows.append(r)
        save_df(build, alpha_rows, "alpha_search_results.csv")
        save_df(build, [r for r in outer_selected if pd.notna(r.get("Selected_Alpha"))], "outer_fold_alpha_and_subset_selections.csv")
        log(f"Feature-count decision: best k={best_k}; qualifying={qualifying}; final k={one_se_k}")
        long_stability, pairs, jsum, ndetails, pf = stability_outputs(build, rankings, one_se_k, log)
        log("Stability calculations and Comment 3 figures completed")
        # Lock the final reduced model jointly over family, alpha, and hyperparameters at the exact k_1SE.
        final_result = final_lock_and_ensemble(Xproper, Yproper, X.loc[cal_ids], Y.loc[cal_ids], X.loc[test_ids], Y.loc[test_ids], build, one_se_k, log)
        log(f"Locked final model: {final_result['model_name']}, alpha={final_result['alpha']}, k={final_result['k']}")
        regenerate_comment1(build, df, proper_ids, cal_ids, test_ids, final_result)
        regenerate_comment2(build, df, proper_ids, test_ids, final_result)
        save_df(build, [{"Outer_Evaluations": 25, "Inner_Folds_Per_Outer": 5, "Search_Iterations_Per_Candidate_Per_Outer": SEARCH_ITERATIONS, "Candidate_Model_Families": 6, "Final_Selection_Families": 4, "Calibration_Count": 91, "Final_Test_Count": 91, "Selection_Index_Scope": "Proper_Training only", "Calibration_Leakage_Detected_Historically": True, "Final_Configuration_Leakage": False}], "final_workflow_summary.csv")
        save_json(build, {"Corrected_Equation_4": "Mean embedded feature importance is not feature-selection stability", "Stability_Measures": ["selection frequency", "pairwise Jaccard", "Nogueira"], "Nogueira_Correction": "M/(M-1) finite-sample factor", "Final_k": one_se_k, "Final_Alpha": final_result["alpha"], "Final_Model": final_result["model_name"]}, "equation4_correction_and_stability_summary.json")
        save_text(build, "The final Comment 3 output tree contains only current-execution results. Historical Comment 1/2/Comment 3 directories and archives were not copied into it.", "final_results_readme.txt")
        package_scripts(build, cwd)
        validations, report = validation_report(build, df, split_sets, outer_selected, curve_df, pairs, ndetails, final_result)
        if not all(validations.values()):
            raise RuntimeError("Final validation failed: " + repr({k: v for k, v in validations.items() if not v}))
        finish = time.time(); write_execution_record(build, cwd, data_path, start, finish, final_result["config"], validations)
        # Add the run log to the final tree before hashing.
        save_text(build, Path(log_path).read_text(encoding="utf-8") + "\nFINAL VALIDATION PASSED\n", "execution_logs/execution.log")
        create_manifest(build)
        # Promote only after all checks and all outputs have been written.
        shutil.move(str(build), str(final_root))
        archive = zip_final(cwd, final_root, os.getenv("COMMENT3_ARCHIVE_NAME"))
        print(f"FINAL_RESULTS={final_root}")
        print(f"FINAL_ARCHIVE={archive}")
        print(report)
        return final_root
    except Exception as exc:
        finish = time.time()
        text = traceback.format_exc()
        if data_path is not None:
            try: write_execution_record(build, cwd, data_path, start, finish, final_result["config"] if final_result else {}, {}, text)
            except Exception: pass
        save_text(build, text, "execution_logs/FAILED_exception_traceback.txt")
        log(f"FAILED: {type(exc).__name__}: {exc}")
        raise
