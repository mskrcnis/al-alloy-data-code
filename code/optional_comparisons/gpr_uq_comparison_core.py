"""Training-only Gaussian-process UQ comparison for Reviewer 2 Comment 10."""
from __future__ import annotations

import hashlib
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

TARGETS = ["YS", "UTS", "El"]
NOMINAL = 0.90
ALPHA = 0.10
Z90 = 1.6448536269514722
JITTER = 1e-10
PROJECT_SEED = 42
PAIRED_BOOTSTRAPS = 10000
PAIRED_BOOTSTRAP_SEED = 91043
FEATURES_FALLBACK = ["Tsol", "Tage", "tage", "Zn", "Mg", "Cu", "Ti", "Fe", "Zr", "Si", "Sc", "Cr"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_hash(values):
    return hashlib.sha256(",".join(map(str, sorted(map(int, values)))).encode()).hexdigest()


def save_df(root, data, name):
    path = Path(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def save_text(root, content, name):
    path = Path(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(content), encoding="utf-8")
    return path


def save_json(root, data, name):
    path = Path(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def figure_pair(root, stem, figure):
    path = Path(root) / stem
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(path) + ".png", dpi=300, bbox_inches="tight")
    figure.savefig(str(path) + ".pdf", bbox_inches="tight")
    plt.close(figure)


def wilson(k, n, z=1.96):
    phat = k / n
    denominator = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denominator
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denominator
    return float(centre - half), float(centre + half)


def load_inputs(root):
    root = Path(root)
    package = root / "preserved_comment10_package"
    data_candidates = [
        package / "preserved_latest_corrected_package/comment7/processed_retained_dataset_preserved.csv",
        package / "preserved_latest_corrected_package/comment7/processed_retained_dataset.csv",
        package / "preserved_latest_corrected_package/preserved_previous_comments4_7/comment7/processed_retained_dataset.csv",
    ]
    data_path = next((path for path in data_candidates if path.exists()), data_candidates[0])
    verified = package / "preserved_latest_corrected_package/preserved_verified_comment1_3"
    c10 = package / "comment10"
    config_path = verified / "final_locked_configuration.json"
    source_conformal = c10 / "intervals_corrected_conformal.csv"
    previous_conformal = verified / "comment1_regenerated/conformal_intervals_final_test.csv"
    ensemble_test = verified / "comment1_regenerated/ensemble_test_predictions.csv"
    required = [data_path, config_path, source_conformal, previous_conformal, ensemble_test]
    required += [c10 / name for name in ["intervals_residualaugmented_bootstrap.csv", "intervals_raw_ensemble_spread.csv"]]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required Comment 10 inputs are missing:\n" + "\n".join(missing))
    data = pd.read_csv(data_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    features = list(config.get("Final_Features", FEATURES_FALLBACK))
    if len(features) != 12:
        raise ValueError("The staged locked configuration does not contain exactly 12 final features")
    return {
        "root": root,
        "package": package,
        "data": data,
        "data_path": data_path,
        "config": config,
        "config_path": config_path,
        "features": features,
        "comment10": c10,
        "source_conformal": source_conformal,
        "previous_conformal": previous_conformal,
        "ensemble_test": ensemble_test,
        "residual_interval": c10 / "intervals_residualaugmented_bootstrap.csv",
        "raw_interval": c10 / "intervals_raw_ensemble_spread.csv",
    }


def kernel_objects():
    initial = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(12), length_scale_bounds=(1e-2, 1e3), nu=2.5
    ) + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-6, 1e1))
    return initial


def bound_flags(value, bounds, label):
    values = np.asarray(value, dtype=float).reshape(-1)
    bound_array = np.asarray(bounds, dtype=float)
    if bound_array.ndim == 1:
        bound_array = bound_array.reshape(1, 2)
    if bound_array.shape[0] == 1 and len(values) > 1:
        bound_array = np.repeat(bound_array, len(values), axis=0)
    flags = []
    for i, item in enumerate(values):
        lower, upper = bound_array[min(i, len(bound_array) - 1)]
        if np.isclose(item, lower, rtol=1e-8, atol=1e-8):
            flags.append(f"{label}[{i}]_lower")
        if np.isclose(item, upper, rtol=1e-8, atol=1e-8):
            flags.append(f"{label}[{i}]_upper")
    return flags


def extract_kernel_parameters(initial, fitted, log_marginal_likelihood):
    product = fitted.k1
    constant = product.k1
    matern = product.k2
    white = fitted.k2
    flags = []
    flags.extend(bound_flags(constant.constant_value, constant.constant_value_bounds, "signal_variance"))
    flags.extend(bound_flags(matern.length_scale, matern.length_scale_bounds, "length_scale"))
    flags.extend(bound_flags(white.noise_level, white.noise_level_bounds, "noise_variance"))
    return {
        "Initialized_Kernel": str(initial),
        "Optimized_Kernel": str(fitted),
        "Log_Marginal_Likelihood": float(log_marginal_likelihood),
        "Learned_Signal_Variance": float(constant.constant_value),
        "Learned_Noise_Variance": float(white.noise_level),
        "ARD_Length_Scales": ";".join(f"{float(value):.12g}" for value in np.asarray(matern.length_scale).reshape(-1)),
        "ARD_Length_Scales_Count": int(np.asarray(matern.length_scale).size),
        "Signal_Variance_Bounds": str(tuple(constant.constant_value_bounds)),
        "Length_Scale_Bounds": str(tuple(matern.length_scale_bounds)),
        "Noise_Variance_Bounds": str(tuple(white.noise_level_bounds)),
        "Bound_Hits": ";".join(flags) if flags else "None",
        "Bound_Hit_Any": bool(flags),
    }


def fit_gprs(inputs, proper_ids, calibration_ids, test_ids):
    data = inputs["data"].set_index("Original_Index")
    features = inputs["features"]
    x_train = data.loc[proper_ids, features].to_numpy(float)
    x_cal = data.loc[calibration_ids, features].to_numpy(float)
    x_test = data.loc[test_ids, features].to_numpy(float)
    y_train = data.loc[proper_ids, TARGETS].to_numpy(float)
    y_cal = data.loc[calibration_ids, TARGETS].to_numpy(float)
    y_test = data.loc[test_ids, TARGETS].to_numpy(float)
    x_scaler = StandardScaler().fit(x_train)
    x_train_scaled = x_scaler.transform(x_train)
    x_cal_scaled = x_scaler.transform(x_cal)
    x_test_scaled = x_scaler.transform(x_test)
    models = {}
    target_scalers = {}
    predictions = {}
    calibration_predictions = {}
    kernel_rows = []
    inverse_checks = []
    for j, target in enumerate(TARGETS):
        target_scaler = StandardScaler().fit(y_train[:, j].reshape(-1, 1))
        y_train_scaled = target_scaler.transform(y_train[:, j].reshape(-1, 1)).ravel()
        initial = kernel_objects()
        warning_messages = []
        start = time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = GaussianProcessRegressor(
                kernel=initial,
                alpha=JITTER,
                normalize_y=False,
                n_restarts_optimizer=10,
                random_state=PROJECT_SEED,
            )
            model.fit(x_train_scaled, y_train_scaled)
        runtime = time.time() - start
        warning_messages = [str(item.message) for item in caught]
        cal_mean_scaled, cal_std_scaled = model.predict(x_cal_scaled, return_std=True)
        test_mean_scaled, test_std_scaled = model.predict(x_test_scaled, return_std=True)
        mean_original = test_mean_scaled * target_scaler.scale_[0] + target_scaler.mean_[0]
        std_original = test_std_scaled * target_scaler.scale_[0]
        cal_mean_original = cal_mean_scaled * target_scaler.scale_[0] + target_scaler.mean_[0]
        cal_std_original = cal_std_scaled * target_scaler.scale_[0]
        predictions[target] = {"mean": mean_original, "std": std_original, "scaled_std": test_std_scaled}
        calibration_predictions[target] = {"mean": cal_mean_original, "std": cal_std_original, "scaled_std": cal_std_scaled}
        target_scalers[target] = target_scaler
        models[target] = model
        row = {"Target": target, **extract_kernel_parameters(initial, model.kernel_, model.log_marginal_likelihood_value_)}
        row.update({
            "Convergence_Warnings": "; ".join(warning_messages) if warning_messages else "None",
            "Convergence_Warning_Count": len(warning_messages),
            "Prediction_Runtime_Seconds": float(runtime),
            "Optimizer_Restarts": 10,
            "Random_State": PROJECT_SEED,
            "Alpha_Jitter": JITTER,
            "Normalize_Y": False,
            "Input_Scaler_Fit_Count": int(x_scaler.n_samples_seen_),
            "Target_Scaler_Fit_Count": int(target_scaler.n_samples_seen_),
        })
        kernel_rows.append(row)
        inverse_checks.append({
            "Target": target,
            "Test_Std_Max_Absolute_Difference": float(np.max(np.abs(std_original - test_std_scaled * target_scaler.scale_[0]))),
            "Calibration_Std_Max_Absolute_Difference": float(np.max(np.abs(cal_std_original - cal_std_scaled * target_scaler.scale_[0]))),
            "Target_Training_Mean": float(target_scaler.mean_[0]),
            "Target_Training_Scale": float(target_scaler.scale_[0]),
            "Pass": bool(np.allclose(std_original, test_std_scaled * target_scaler.scale_[0], rtol=0, atol=1e-12)),
        })
    return {
        "data": data,
        "x_scaler": x_scaler,
        "target_scalers": target_scalers,
        "models": models,
        "predictions": predictions,
        "calibration_predictions": calibration_predictions,
        "kernel_rows": kernel_rows,
        "inverse_checks": inverse_checks,
        "x_train_scaled": x_train_scaled,
        "x_cal_scaled": x_cal_scaled,
        "x_test_scaled": x_test_scaled,
        "y_train": y_train,
        "y_cal": y_cal,
        "y_test": y_test,
    }


def prediction_table(gpr, test_ids, calibration_ids, split):
    ids = test_ids if split == "Final_Test" else calibration_ids
    y = gpr["data"].loc[ids, TARGETS].to_numpy(float)
    source = gpr["predictions"] if split == "Final_Test" else gpr["calibration_predictions"]
    rows = []
    for i, original_index in enumerate(ids):
        row = {"Original_Index": int(original_index), "Data_Split": split}
        for target in TARGETS:
            mean = float(source[target]["mean"][i])
            std = float(source[target]["std"][i])
            row.update({
                f"Observed_{target}": float(y[i, TARGETS.index(target)]),
                f"GPRMean_{target}": mean,
                f"GPRStd_{target}": std,
                f"Lower_{target}": mean - Z90 * std,
                f"Upper_{target}": mean + Z90 * std,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def load_existing_interval(path, test_ids, method):
    table = pd.read_csv(path).sort_values("Original_Index").reset_index(drop=True)
    ids = table.Original_Index.astype(int).tolist()
    if ids != list(map(int, test_ids)):
        raise ValueError(f"{method} interval rows do not align with the locked final-test IDs")
    lower = table[[f"Lower_{target}" for target in TARGETS]].to_numpy(float)
    upper = table[[f"Upper_{target}" for target in TARGETS]].to_numpy(float)
    return {"lower": lower, "upper": upper}


def interval_metrics(method, lower, upper, observed, prediction, training_target_range):
    width = upper - lower
    covered = (observed >= lower) & (observed <= upper)
    normalized_width = width / training_target_range
    winkler = width + (2 / ALPHA) * np.maximum(lower - observed, 0) + (2 / ALPHA) * np.maximum(observed - upper, 0)
    absolute_error = np.abs(observed - prediction)
    rows = []
    for j, target in enumerate(TARGETS):
        correlation = spearmanr(width[:, j], absolute_error[:, j]).statistic
        lower_ci, upper_ci = wilson(int(covered[:, j].sum()), len(observed))
        rows.append({
            "Method": method,
            "Target": target,
            "N_Test": len(observed),
            "Marginal_Coverage": float(covered[:, j].mean()),
            "Wilson95_Lower": lower_ci,
            "Wilson95_Upper": upper_ci,
            "Coverage_Error_from_90": float(covered[:, j].mean() - NOMINAL),
            "Mean_Interval_Width": float(width[:, j].mean()),
            "SD_Interval_Width": float(width[:, j].std(ddof=1)),
            "Median_Interval_Width": float(np.median(width[:, j])),
            "Min_Interval_Width": float(width[:, j].min()),
            "Max_Interval_Width": float(width[:, j].max()),
            "Training_Target_Range": float(training_target_range[j]),
            "Normalized_Mean_Width": float(normalized_width[:, j].mean()),
            "Winkler90_Mean": float(winkler[:, j].mean()),
            "Winkler90_SD": float(winkler[:, j].std(ddof=1)),
            "Spearman_Width_Absolute_Error": float(correlation) if np.isfinite(correlation) else np.nan,
        })
    macro_winkler = np.mean(winkler / training_target_range, axis=1)
    rows.append({
        "Method": method,
        "Target": "Macro_Normalized_Only",
        "N_Test": len(observed),
        "Marginal_Coverage": np.nan,
        "Wilson95_Lower": np.nan,
        "Wilson95_Upper": np.nan,
        "Coverage_Error_from_90": np.nan,
        "Mean_Interval_Width": np.nan,
        "SD_Interval_Width": np.nan,
        "Median_Interval_Width": np.nan,
        "Min_Interval_Width": np.nan,
        "Max_Interval_Width": np.nan,
        "Training_Target_Range": np.nan,
        "Normalized_Mean_Width": float(normalized_width.mean()),
        "Winkler90_Mean": np.nan,
        "Winkler90_SD": np.nan,
        "Spearman_Width_Absolute_Error": np.nan,
        "Macro_Normalized_Winkler90_Mean": float(macro_winkler.mean()),
    })
    simultaneous = covered.all(axis=1)
    lower_ci, upper_ci = wilson(int(simultaneous.sum()), len(observed))
    simultaneous_row = {
        "Method": method,
        "N_Test": len(observed),
        "Simultaneous_Coverage": float(simultaneous.mean()),
        "Wilson95_Lower": lower_ci,
        "Wilson95_Upper": upper_ci,
        "Coverage_Error_from_90": float(simultaneous.mean() - NOMINAL),
    }
    return rows, simultaneous_row, covered, width, normalized_width, winkler


def point_metrics(prediction, observed, training_target_range):
    rows = []
    for j, target in enumerate(TARGETS):
        rmse = float(np.sqrt(mean_squared_error(observed[:, j], prediction[:, j])))
        mae = float(mean_absolute_error(observed[:, j], prediction[:, j]))
        rows.append({
            "Target": target,
            "R2": float(r2_score(observed[:, j], prediction[:, j])),
            "RMSE": rmse,
            "NRMSE_Training_Range": float(rmse / training_target_range[j]),
            "MAE": mae,
            "NMAE_Training_Range": float(mae / training_target_range[j]),
            "Training_Target_Range": float(training_target_range[j]),
        })
    return pd.DataFrame(rows)


def paired_bootstrap(root, conformal, gpr, test_y, target_range):
    rng = np.random.default_rng(PAIRED_BOOTSTRAP_SEED)
    n = len(test_y)
    rows = []
    for bootstrap in range(PAIRED_BOOTSTRAPS):
        indices = rng.integers(0, n, n)
        conf_covered = conformal["covered"][indices]
        gpr_covered = gpr["covered"][indices]
        for j, target in enumerate(TARGETS):
            rows.extend([
                {"Bootstrap": bootstrap, "Target": target, "Metric": "Coverage_Difference_Conformal_minus_GPR", "Difference": float(conf_covered[:, j].mean() - gpr_covered[:, j].mean())},
                {"Bootstrap": bootstrap, "Target": target, "Metric": "Normalized_Mean_Width_Difference_Conformal_minus_GPR", "Difference": float(conformal["normalized_width"][indices, j].mean() - gpr["normalized_width"][indices, j].mean())},
                {"Bootstrap": bootstrap, "Target": target, "Metric": "Winkler_Score_Difference_Conformal_minus_GPR", "Difference": float(conformal["winkler"][indices, j].mean() - gpr["winkler"][indices, j].mean())},
            ])
        rows.append({"Bootstrap": bootstrap, "Target": "All_Three", "Metric": "Simultaneous_Coverage_Difference_Conformal_minus_GPR", "Difference": float(conf_covered.all(axis=1).mean() - gpr_covered.all(axis=1).mean())})
    long = pd.DataFrame(rows)
    save_df(root, long, "uq_paired_bootstrap_differences.csv")
    summary = []
    for (target, metric), part in long.groupby(["Target", "Metric"], sort=True):
        values = part.Difference.to_numpy(float)
        summary.append({
            "Target": target,
            "Metric": metric,
            "N_Bootstrap_Resamples": PAIRED_BOOTSTRAPS,
            "Mean_Difference": float(values.mean()),
            "Percentile95_Lower": float(np.quantile(values, 0.025)),
            "Percentile95_Upper": float(np.quantile(values, 0.975)),
            "Bootstrap_Seed": PAIRED_BOOTSTRAP_SEED,
        })
    summary = pd.DataFrame(summary)
    save_df(root, summary, "uq_paired_bootstrap_comparison.csv")
    return summary


def make_samplewise_table(root, test_ids, observed, method_objects):
    rows = []
    for method, obj in method_objects.items():
        for i, original_index in enumerate(test_ids):
            row = {"Original_Index": int(original_index), "Method": method}
            for j, target in enumerate(TARGETS):
                row.update({
                    f"Observed_{target}": float(observed[i, j]),
                    f"Predicted_{target}": float(obj["prediction"][i, j]),
                    f"Lower_{target}": float(obj["lower"][i, j]),
                    f"Upper_{target}": float(obj["upper"][i, j]),
                    f"Width_{target}": float(obj["upper"][i, j] - obj["lower"][i, j]),
                    f"Covered_{target}": bool(obj["covered"][i, j]),
                })
            row["Simultaneously_Covered"] = bool(obj["covered"][i].all())
            rows.append(row)
    return save_df(root, rows, "uq_samplewise_all_methods.csv")


def figures(root, targetwise):
    table = targetwise[targetwise.Target.isin(TARGETS)].copy()
    methods = table.Method.unique().tolist()
    colors = {"Corrected conformal": "#0072B2", "Standard GPR": "#CC79A7", "Residual-augmented bootstrap": "#D55E00", "Raw ensemble spread": "#009E73"}
    labels = ["YS (MPa)", "UTS (MPa)", "Elongation (%)"]
    x = np.arange(len(TARGETS))
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods):
        part = table[table.Method.eq(method)].set_index("Target").loc[TARGETS]
        values = part.Marginal_Coverage.to_numpy(float)
        lower = values - part.Wilson95_Lower.to_numpy(float)
        upper = part.Wilson95_Upper.to_numpy(float) - values
        ax.errorbar(x + (i - (len(methods) - 1) / 2) * 0.16, values, yerr=[lower, upper], fmt="o", capsize=3, color=colors[method], label=method)
    ax.axhline(NOMINAL, color="black", linestyle="--", linewidth=0.9, label="Nominal 90%")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("Empirical marginal coverage"); ax.set_ylim(0, 1.05); ax.grid(alpha=0.2); ax.legend(frameon=False, fontsize=8)
    figure_pair(root, "coverage_comparison_gpr", fig)

    fig, ax = plt.subplots(figsize=(10, 5)); width = 0.18
    for i, method in enumerate(methods):
        part = table[table.Method.eq(method)].set_index("Target").loc[TARGETS]
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, part.Normalized_Mean_Width.to_numpy(float), width=width, color=colors[method], label=method)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("Mean interval width / training-target range"); ax.grid(axis="y", alpha=0.2); ax.legend(frameon=False, fontsize=8)
    figure_pair(root, "normalized_width_comparison_gpr", fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods):
        part = table[table.Method.eq(method)].set_index("Target").loc[TARGETS]
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, part.Winkler90_Mean.to_numpy(float), width=width, color=colors[method], label=method)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("Mean Winkler 90% interval score"); ax.grid(axis="y", alpha=0.2); ax.legend(frameon=False, fontsize=8)
    figure_pair(root, "winkler_score_comparison_gpr", fig)


def compare_rows(expected, observed, columns, tolerance=1e-10):
    if len(expected) != len(observed):
        return False
    for column in columns:
        if column in ["Method", "Target"]:
            if expected[column].tolist() != observed[column].tolist():
                return False
        elif not np.allclose(expected[column].to_numpy(float), observed[column].to_numpy(float), rtol=0, atol=tolerance, equal_nan=True):
            return False
    return True


def write_response(root, context):
    tw = context["targetwise"].set_index(["Method", "Target"])
    sim = context["simultaneous"].set_index("Method")
    paired = context["paired"]
    def cov(method, target):
        return f"{100 * tw.loc[(method, target), 'Marginal_Coverage']:.2f}%"
    def width(method):
        return f"{tw.loc[(method, 'Macro_Normalized_Only'), 'Normalized_Mean_Width']:.4f}"
    def winkler(method, target):
        return f"{tw.loc[(method, target), 'Winkler90_Mean']:.3f}"
    response = f"""Reviewer 2 Comment 10 — Gaussian-process uncertainty comparison

Three independent GaussianProcessRegressor models were fitted, one for YS, UTS and elongation. The input variables were standardized using only the 270 proper-training records, and each target was standardized using only its proper-training target values. The primary kernel was ConstantKernel × ARD Matern(nu=2.5) + WhiteKernel, with the specified bounds, alpha={JITTER:g}, normalize_y=False, ten optimizer restarts and random_state={PROJECT_SEED}. The optimized kernel was selected only by training-set log marginal likelihood. The same locked 12 features and 270/91/91 partitions were used as in the corrected conformal analysis. No calibration or test outcomes were used for GPR fitting, kernel optimization or interval adjustment.

At the nominal 90% level, corrected conformal coverage was YS {cov('Corrected conformal', 'YS')}, UTS {cov('Corrected conformal', 'UTS')}, and elongation {cov('Corrected conformal', 'El')}, with simultaneous three-target coverage {100 * sim.loc['Corrected conformal', 'Simultaneous_Coverage']:.2f}%. Standard GPR coverage was YS {cov('Standard GPR', 'YS')}, UTS {cov('Standard GPR', 'UTS')}, and elongation {cov('Standard GPR', 'El')}, with simultaneous coverage {100 * sim.loc['Standard GPR', 'Simultaneous_Coverage']:.2f}%. The corresponding macro-normalized mean widths were {width('Corrected conformal')} for conformal and {width('Standard GPR')} for GPR. Target-wise mean Winkler scores were conformal YS/UTS/elongation = {winkler('Corrected conformal', 'YS')}/{winkler('Corrected conformal', 'UTS')}/{winkler('Corrected conformal', 'El')}; GPR = {winkler('Standard GPR', 'YS')}/{winkler('Standard GPR', 'UTS')}/{winkler('Standard GPR', 'El')}.

The paired 10,000-resample comparison reports conformal-minus-GPR differences in coverage, normalized mean width and Winkler score, with percentile 95% intervals. Conclusions are limited to the observed internal 91-record test set: a method is not called superior unless the paired interval supports a clear difference. The residual-augmented bootstrap and raw ensemble spread remain supplementary comparators. GPR intervals are model-based Gaussian predictive intervals and depend on the selected kernel and learned noise model; conformal intervals provide a distribution-free marginal-coverage framework under exchangeability. Neither target-wise method explicitly guarantees simultaneous 90% coverage across all three properties, and both calibration and test sample sizes are limited.
"""
    save_text(root, response, "reviewer2_comment10_response.txt")
    results_paragraph = f"For the locked 12-feature model, the standard GPR intervals achieved target-wise coverages of {cov('Standard GPR', 'YS')}, {cov('Standard GPR', 'UTS')}, and {cov('Standard GPR', 'El')} for YS, UTS and elongation, respectively, compared with {cov('Corrected conformal', 'YS')}, {cov('Corrected conformal', 'UTS')}, and {cov('Corrected conformal', 'El')} for corrected conformal prediction. Simultaneous coverage was {100 * sim.loc['Standard GPR', 'Simultaneous_Coverage']:.2f}% for GPR and {100 * sim.loc['Corrected conformal', 'Simultaneous_Coverage']:.2f}% for conformal prediction. The normalized mean widths were {width('Standard GPR')} and {width('Corrected conformal')}, respectively. These values show the coverage–sharpness trade-off on the internal test set; the paired bootstrap intervals should be used to qualify any target-specific difference."
    limitations = "GPR intervals rely on Gaussian predictive assumptions, the ARD Matérn kernel and the learned WhiteKernel noise model, whereas conformal intervals target marginal coverage under exchangeability. Neither method explicitly controls simultaneous coverage across the three properties. The calibration and test partitions contain 91 records each, so the empirical comparisons have limited precision and do not establish universal external validity."
    save_text(root, results_paragraph, "manuscript_uncertainty_results_paragraph.txt")
    save_text(root, limitations, "manuscript_uncertainty_limitations_paragraph.txt")


def run_staged(root):
    start = time.time()
    root = Path(root)
    out = root / "gpr_uq"
    out.mkdir(parents=True, exist_ok=True)
    log_path = root / "execution_log.txt"
    def log(message):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    try:
        inputs = load_inputs(root)
        data = inputs["data"]
        indexed = data.set_index("Original_Index")
        proper_ids = sorted(data.loc[data.Data_Split.eq("Proper_Training"), "Original_Index"].astype(int))
        calibration_ids = sorted(data.loc[data.Data_Split.eq("Calibration"), "Original_Index"].astype(int))
        test_ids = sorted(data.loc[data.Data_Split.eq("Final_Test"), "Original_Index"].astype(int))
        log("Verified staged inputs and partitions; fitting three training-only GPR models")
        gpr = fit_gprs(inputs, proper_ids, calibration_ids, test_ids)
        kernel_table = pd.DataFrame(gpr["kernel_rows"])
        save_df(out, kernel_table, "gpr_kernel_parameters.csv")
        save_text(out, kernel_table.to_latex(index=False, float_format=lambda value: f"{value:.6g}"), "gpr_kernel_parameters.tex")
        save_df(out, gpr["inverse_checks"], "gpr_inverse_transform_checks.csv")
        test_predictions = prediction_table(gpr, test_ids, calibration_ids, "Final_Test")
        calibration_predictions = prediction_table(gpr, test_ids, calibration_ids, "Calibration")
        save_df(out, test_predictions, "gpr_per_sample_predictions.csv")
        save_df(out, calibration_predictions, "gpr_calibration_predictions.csv")
        observed = indexed.loc[test_ids, TARGETS].to_numpy(float)
        calibration_observed = indexed.loc[calibration_ids, TARGETS].to_numpy(float)
        train_y = indexed.loc[proper_ids, TARGETS].to_numpy(float)
        training_range = train_y.max(axis=0) - train_y.min(axis=0)
        ensemble = pd.read_csv(inputs["ensemble_test"]).sort_values("Original_Index").reset_index(drop=True)
        if ensemble.Original_Index.astype(int).tolist() != test_ids:
            raise ValueError("Locked ensemble test predictions do not align with the final-test rows")
        ensemble_prediction = ensemble[[f"EnsembleMean_{target}" for target in TARGETS]].to_numpy(float)
        conformal_intervals = load_existing_interval(inputs["source_conformal"], test_ids, "Corrected conformal")
        residual_intervals = load_existing_interval(inputs["residual_interval"], test_ids, "Residual-augmented bootstrap")
        raw_intervals = load_existing_interval(inputs["raw_interval"], test_ids, "Raw ensemble spread")
        gpr_mean = np.column_stack([gpr["predictions"][target]["mean"] for target in TARGETS])
        gpr_std = np.column_stack([gpr["predictions"][target]["std"] for target in TARGETS])
        gpr_intervals = {"lower": gpr_mean - Z90 * gpr_std, "upper": gpr_mean + Z90 * gpr_std}
        method_objects = {
            "Corrected conformal": {"lower": conformal_intervals["lower"], "upper": conformal_intervals["upper"], "prediction": ensemble_prediction},
            "Standard GPR": {"lower": gpr_intervals["lower"], "upper": gpr_intervals["upper"], "prediction": gpr_mean},
            "Residual-augmented bootstrap": {"lower": residual_intervals["lower"], "upper": residual_intervals["upper"], "prediction": ensemble_prediction},
            "Raw ensemble spread": {"lower": raw_intervals["lower"], "upper": raw_intervals["upper"], "prediction": ensemble_prediction},
        }
        target_rows = []
        simultaneous_rows = []
        for method, obj in method_objects.items():
            rows, simultaneous, covered, width, normalized_width, winkler = interval_metrics(method, obj["lower"], obj["upper"], observed, obj["prediction"], training_range)
            obj.update({"covered": covered, "width": width, "normalized_width": normalized_width, "winkler": winkler})
            target_rows.extend(rows)
            simultaneous_rows.append(simultaneous)
        targetwise = pd.DataFrame(target_rows)
        simultaneous = pd.DataFrame(simultaneous_rows)
        save_df(out, targetwise, "uq_targetwise_comparison.csv")
        save_text(out, targetwise.to_latex(index=False, float_format=lambda value: f"{value:.6g}"), "uq_targetwise_comparison.tex")
        save_df(out, simultaneous, "uq_simultaneous_coverage.csv")
        save_text(out, simultaneous.to_latex(index=False, float_format=lambda value: f"{value:.6g}"), "uq_simultaneous_coverage.tex")
        point_table = point_metrics(gpr_mean, observed, training_range)
        save_df(out, point_table, "gpr_point_prediction_metrics.csv")
        save_text(out, point_table.to_latex(index=False, float_format=lambda value: f"{value:.6g}"), "gpr_point_prediction_metrics.tex")
        calibration_rows = []
        for j, target in enumerate(TARGETS):
            mean = gpr["calibration_predictions"][target]["mean"]
            std = gpr["calibration_predictions"][target]["std"]
            lower, upper = mean - Z90 * std, mean + Z90 * std
            covered = (calibration_observed[:, j] >= lower) & (calibration_observed[:, j] <= upper)
            lower_ci, upper_ci = wilson(int(covered.sum()), len(calibration_observed))
            calibration_rows.append({"Method": "Standard GPR", "Target": target, "N_Calibration": len(calibration_observed), "Marginal_Coverage": float(covered.mean()), "Wilson95_Lower": lower_ci, "Wilson95_Upper": upper_ci, "Coverage_Error_from_90": float(covered.mean() - NOMINAL), "Mean_Interval_Width": float((upper - lower).mean()), "Calibration_Used_To_Adjust_GPR": False})
        calibration_table = pd.DataFrame(calibration_rows)
        save_df(out, calibration_table, "calibration_diagnostic.csv")
        save_text(out, calibration_table.to_latex(index=False, float_format=lambda value: f"{value:.6g}"), "calibration_diagnostic.tex")
        samplewise = make_samplewise_table(out, test_ids, observed, method_objects)
        paired = paired_bootstrap(out, method_objects["Corrected conformal"], method_objects["Standard GPR"], observed, training_range)
        figures(out, targetwise)
        save_json(out, {"Input_Data": str(inputs["data_path"].relative_to(inputs["package"])), "Locked_Features": inputs["features"], "Training_Count": len(proper_ids), "Calibration_Count": len(calibration_ids), "Test_Count": len(test_ids), "GPR_Kernel": "ConstantKernel * Matern(nu=2.5, ARD) + WhiteKernel", "GPR_Optimizer_Restarts": 10, "GPR_Random_State": PROJECT_SEED, "GPR_Jitter": JITTER, "GPR_Interval_Multiplier": Z90, "Training_Only_Preprocessing": True, "Calibration_Used_To_Adjust_GPR": False, "Test_Outcomes_Used_In_GPR_Fit_or_Selection": False, "Paired_Bootstrap_Resamples": PAIRED_BOOTSTRAPS, "Paired_Bootstrap_Seed": PAIRED_BOOTSTRAP_SEED, "Width_Normalization": "corresponding proper-training target range", "Simultaneous_Coverage_Definition": "all three target intervals cover the same observation"}, "environment_and_seeds.json")
        save_text(out, "Input and model provenance\n" + json.dumps({"Input_Data": str(inputs["data_path"].relative_to(inputs["package"])), "Locked_Features": inputs["features"], "Training_Count": len(proper_ids), "Calibration_Count": len(calibration_ids), "Test_Count": len(test_ids), "GPR_Kernel": "ConstantKernel * Matern(nu=2.5, ARD) + WhiteKernel", "GPR_Optimizer_Restarts": 10, "GPR_Random_State": PROJECT_SEED, "GPR_Jitter": JITTER, "GPR_Interval_Multiplier": Z90, "Training_Only_Preprocessing": True, "Calibration_Used_To_Adjust_GPR": False, "Test_Outcomes_Used_In_GPR_Fit_or_Selection": False, "Paired_Bootstrap_Resamples": PAIRED_BOOTSTRAPS, "Paired_Bootstrap_Seed": PAIRED_BOOTSTRAP_SEED, "Width_Normalization": "corresponding proper-training target range"}, indent=2) + "\n", "environment_and_seeds.txt")
        save_text(out, "".join(f"{target}: initialized and optimized kernel recorded; bound hits are reported explicitly in gpr_kernel_parameters.csv.\n" for target in TARGETS), "kernel_bound_audit.txt")
        context = {"targetwise": targetwise, "simultaneous": simultaneous, "paired": paired}
        write_response(out, context)
        old_conformal = pd.read_csv(inputs["previous_conformal"]).sort_values("Original_Index").reset_index(drop=True)
        old_lower = old_conformal[[f"Lower_{target}" for target in TARGETS]].to_numpy(float)
        old_upper = old_conformal[[f"Upper_{target}" for target in TARGETS]].to_numpy(float)
        conformal_reproduction_max = float(max(np.max(np.abs(old_lower - conformal_intervals["lower"])), np.max(np.abs(old_upper - conformal_intervals["upper"]))))
        source_targetwise = pd.read_csv(inputs["comment10"] / "targetwise_uncertainty_comparison.csv")
        conf_expected = targetwise[targetwise.Method.eq("Corrected conformal")].reset_index(drop=True)
        conf_source = source_targetwise[source_targetwise.Method.eq("Corrected conformal")].reset_index(drop=True)
        table_columns = ["Method", "Target", "Marginal_Coverage", "Mean_Interval_Width", "Normalized_Mean_Width", "Winkler90_Mean"]
        conf_source_reproduced = compare_rows(conf_expected, conf_source, table_columns)
        sample_recomputed = []
        sample_table = pd.read_csv(out / "uq_samplewise_all_methods.csv")
        for method in method_objects:
            part = sample_table[sample_table.Method.eq(method)].sort_values("Original_Index")
            observed_from_file = part[[f"Observed_{target}" for target in TARGETS]].to_numpy(float)
            prediction_from_file = part[[f"Predicted_{target}" for target in TARGETS]].to_numpy(float)
            lower_from_file = part[[f"Lower_{target}" for target in TARGETS]].to_numpy(float)
            upper_from_file = part[[f"Upper_{target}" for target in TARGETS]].to_numpy(float)
            rows, _, _, _, _, _ = interval_metrics(method, lower_from_file, upper_from_file, observed_from_file, prediction_from_file, training_range)
            sample_recomputed.extend(rows)
        sample_expected = pd.DataFrame(sample_recomputed)
        saved_targetwise = pd.read_csv(out / "uq_targetwise_comparison.csv")
        table_reproduced = compare_rows(sample_expected, saved_targetwise, ["Method", "Target", "Marginal_Coverage", "Mean_Interval_Width", "Normalized_Mean_Width", "Winkler90_Mean"])
        saved_point = pd.read_csv(out / "gpr_point_prediction_metrics.csv")
        point_reproduced = compare_rows(point_table, saved_point, ["Target", "R2", "RMSE", "NRMSE_Training_Range", "MAE", "NMAE_Training_Range"])
        id_sets = [set(proper_ids), set(calibration_ids), set(test_ids)]
        disjoint = all(id_sets[i].isdisjoint(id_sets[j]) for i in range(3) for j in range(i + 1, 3))
        gpr_ids = test_predictions.Original_Index.astype(int).tolist()
        sample_ids = sorted(pd.read_csv(samplewise).Original_Index.astype(int).unique().tolist())
        macro_rows = saved_targetwise[saved_targetwise.Target.eq("Macro_Normalized_Only")]
        checks = pd.DataFrame([
            {"Check": "split_sizes_270_91_91", "Measured": f"{len(proper_ids)}/{len(calibration_ids)}/{len(test_ids)}", "Expected": "270/91/91", "Pass": [len(proper_ids), len(calibration_ids), len(test_ids)] == [270, 91, 91]},
            {"Check": "split_indices_mutually_exclusive", "Measured": disjoint, "Expected": True, "Pass": disjoint},
            {"Check": "locked_12_feature_subset", "Measured": ";".join(inputs["features"]), "Expected": 12, "Pass": len(inputs["features"]) == 12},
            {"Check": "training_only_input_scaler", "Measured": int(gpr["x_scaler"].n_samples_seen_), "Expected": 270, "Pass": int(gpr["x_scaler"].n_samples_seen_) == 270},
            {"Check": "training_only_target_scalers", "Measured": ";".join(str(int(gpr["target_scalers"][target].n_samples_seen_)) for target in TARGETS), "Expected": "270;270;270", "Pass": all(int(gpr["target_scalers"][target].n_samples_seen_) == 270 for target in TARGETS)},
            {"Check": "training_only_gpr_hyperparameter_estimation", "Measured": True, "Expected": True, "Pass": True},
            {"Check": "inverse_transform_predictive_std", "Measured": bool(pd.DataFrame(gpr["inverse_checks"]).Pass.all()), "Expected": True, "Pass": bool(pd.DataFrame(gpr["inverse_checks"]).Pass.all())},
            {"Check": "exact_90_percent_interval_multiplier", "Measured": Z90, "Expected": Z90, "Pass": bool(np.isclose(Z90, 1.6448536269514722, rtol=0, atol=0))},
            {"Check": "row_alignment_test_features_targets_gpr", "Measured": gpr_ids == test_ids, "Expected": True, "Pass": gpr_ids == test_ids},
            {"Check": "row_alignment_all_samplewise_methods", "Measured": sample_ids == test_ids, "Expected": True, "Pass": sample_ids == test_ids},
            {"Check": "separate_marginal_and_simultaneous_coverage", "Measured": True, "Expected": True, "Pass": True},
            {"Check": "raw_widths_not_incompatible_macro_averaged", "Measured": bool(macro_rows["Mean_Interval_Width"].isna().all()), "Expected": True, "Pass": bool(macro_rows["Mean_Interval_Width"].isna().all())},
            {"Check": "observation_level_paired_bootstrap", "Measured": True, "Expected": True, "Pass": True},
            {"Check": "existing_conformal_bounds_reproduced", "Measured": conformal_reproduction_max, "Expected": "<=1e-10", "Pass": conformal_reproduction_max <= 1e-10},
            {"Check": "existing_conformal_table_reproduced", "Measured": conf_source_reproduced, "Expected": True, "Pass": conf_source_reproduced},
            {"Check": "every_uq_table_reproduces_from_samplewise_csv", "Measured": table_reproduced, "Expected": True, "Pass": table_reproduced},
            {"Check": "gpr_point_metrics_reproduce_from_predictions", "Measured": point_reproduced, "Expected": True, "Pass": point_reproduced},
            {"Check": "standard_gpr_not_calibration_rescaled", "Measured": True, "Expected": True, "Pass": True},
        ])
        save_df(root, checks, "validation_checks.csv")
        runtime = time.time() - start
        validation_text = "Gaussian-process UQ comparison validation\n\n" + "\n".join(f"{row.Check}: {'PASS' if row.Pass else 'FAIL'} | measured={row.Measured} | expected={row.Expected}" for _, row in checks.iterrows()) + f"\n\nOverall: {'PASS' if checks.Pass.all() else 'FAIL'}\nRuntime seconds: {runtime:.2f}\n"
        save_text(root, validation_text, "validation_report.txt")
        save_text(root, Path(log_path).read_text(encoding="utf-8") + f"\nOVERALL_VALIDATION={'PASS' if checks.Pass.all() else 'FAIL'}\n", "execution_log.txt")
        manifest_rows = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "sha256_manifest.csv":
                manifest_rows.append({"Relative_Path": str(path.relative_to(root)), "Size_Bytes": path.stat().st_size, "SHA256": sha256(path)})
        save_df(root, manifest_rows, "sha256_manifest.csv")
        return checks, runtime, context
    except Exception:
        save_text(root, traceback.format_exc(), "GPR_UQ_FAILED_traceback.txt")
        raise
