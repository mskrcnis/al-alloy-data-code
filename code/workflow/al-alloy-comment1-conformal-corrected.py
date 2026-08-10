# Auto-generated from al-alloy-comment1-conformal-corrected.ipynb
# Cells 14-15 contain the Reviewer 1 Comment 1 conformal correction.

# %% [markdown] cell 0
# # UA-XMTL Aluminum Alloy Notebook — Choi Baselines + Yang Top-12 Feature-Selection Pipeline + Proposed Physics-Retained Models
#
# This Kaggle notebook implements a fair same-dataset comparison for aluminum-alloy strength–ductility prediction.
#
# **Included:**
# - **Choi-style baselines:** Ridge, RF, XGB, and MLP using all available features with mean imputation, matching Choi's no-formal-feature-selection baseline logic.
# - **Yang-style Top-12 pipeline with feature selection:** target-wise Top-12 Pearson-correlation feature selection + target-wise CV-tuned RBF-SVR for YS, UTS, and El.
# - **Proposed method:** hybrid feature selection with a physics-retention rule, always retaining `Tsol`, `Tage`, and `tage` when present.
# - Same-dataset holdout and repeated-CV comparison.
# - SHAP explainability, ensemble-conformal uncertainty, reliability-aware candidate screening with sample-specific uncertainty, and output ZIP export.
#
# **Important:** Pichlmann-style SVR is intentionally excluded because its original method depends on CALPHAD phase features and elemental physical descriptors that are not available in this Excel dataset.

# %% cell 1

# ============================================================
# 1. Environment and imports
# ============================================================
import os, sys, subprocess, warnings, json, math, copy, random
from pathlib import Path
from IPython.display import display
warnings.filterwarnings("ignore")

# Install optional packages if missing. Kaggle usually has xgboost/shap; catboost/optuna may need install.
def ensure_pkg(import_name, pip_name=None):
    pip_name = pip_name or import_name
    try:
        __import__(import_name)
    except Exception:
        print(f"Installing {pip_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])

for import_name, pip_name in [("optuna", "optuna"), ("xgboost", "xgboost"), ("shap", "shap"), ("catboost", "catboost")]:
    ensure_pkg(import_name, pip_name)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import train_test_split, RepeatedKFold, KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import mutual_info_regression, RFECV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.neural_network import MLPRegressor

import optuna
from optuna.samplers import TPESampler

from xgboost import XGBRegressor
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

import shap

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

OUTPUT_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd() / 'comment1_outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GPU availability check. For this small tabular dataset CPU is often as fast/stable, but XGB can use GPU if available.
GPU_AVAILABLE = Path('/proc/driver/nvidia/version').exists() or bool(os.environ.get('CUDA_VISIBLE_DEVICES'))
print('GPU available:', GPU_AVAILABLE)
print('Output directory:', OUTPUT_DIR)

# %% cell 2

# ============================================================
# 2. Utility functions
# ============================================================
def save_df(df, filename):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print('Saved:', path)
    return path

def save_fig(filename, dpi=600):
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print('Saved:', path)
    plt.show()
    return path

def rmse_safe(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def regression_metrics(y_true, y_pred, target_names=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if target_names is None:
        target_names = [f'Target_{i}' for i in range(y_true.shape[1])]
    rows=[]
    for i,t in enumerate(target_names):
        yt=y_true[:,i]; yp=y_pred[:,i]
        rows.append({
            'Target': t,
            'R2': r2_score(yt, yp),
            'MAE': mean_absolute_error(yt, yp),
            'RMSE': rmse_safe(yt, yp),
            'MAPE_%': float(np.mean(np.abs((yt-yp)/np.maximum(np.abs(yt), 1e-8))) * 100)
        })
    df=pd.DataFrame(rows)
    avg={'Target':'Average','R2':df['R2'].mean(),'MAE':df['MAE'].mean(),'RMSE':df['RMSE'].mean(),'MAPE_%':df['MAPE_%'].mean()}
    return pd.concat([df,pd.DataFrame([avg])], ignore_index=True)

def normalized_rmse_score(y_true, y_pred):
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred)
    vals=[]
    for j in range(y_true.shape[1]):
        rmse=rmse_safe(y_true[:,j], y_pred[:,j])
        scale=np.nanmax(y_true[:,j])-np.nanmin(y_true[:,j])
        vals.append(rmse if scale == 0 or np.isnan(scale) else rmse/scale)
    return float(np.mean(vals))

def display_and_save_metrics(name, y_true, y_pred, target_names, filename):
    met = regression_metrics(y_true, y_pred, target_names)
    met.insert(0, 'Model', name)
    display(met)
    save_df(met, filename)
    return met

def get_xgb_params(seed=RANDOM_STATE, gpu=False):
    # XGBoost 2.x prefers device='cuda'; older versions use tree_method='gpu_hist'. Keep conservative.
    params = dict(
        random_state=seed,
        n_estimators=500,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective='reg:squarederror',
        n_jobs=-1,
        verbosity=0
    )
    if gpu:
        # Try modern setting; if fails later, rerun with GPU_AVAILABLE=False.
        params.update(dict(tree_method='hist', device='cuda'))
    else:
        params.update(dict(tree_method='hist'))
    return params

# %% cell 3

# ============================================================
# 3. Dataset loading
# ============================================================
def find_excel_file():
    candidates = []
    for root in ['/kaggle/input', '/mnt/data', '.']:
        p = Path(root)
        if p.exists():
            candidates.extend(list(p.rglob('*.xlsx')))
            candidates.extend(list(p.rglob('*.xls')))
    if not candidates:
        raise FileNotFoundError('No Excel file found. Upload Aged forged Al alloy.xlsx as a Kaggle dataset.')
    # Prefer file with aged/forged/al/alloy in name
    candidates = sorted(candidates, key=lambda x: (('aged' not in x.name.lower()) + ('alloy' not in x.name.lower()), len(str(x))))
    return candidates[0]

DATA_PATH = find_excel_file()
print('Using dataset:', DATA_PATH)
raw = pd.read_excel(DATA_PATH)
print('Raw shape:', raw.shape)
display(raw.head())
print(raw.columns.tolist())

# %% cell 4

# ============================================================
# 4. Data preparation
# ============================================================
TARGET_COLS = ['YS', 'UTS', 'El']
PHYSICS_KEEP = ['Tsol', 'Tage', 'tage']
REFERENCE_COLS = ['Reference (APA)']

# Keep numeric columns only for modeling; drop unnamed columns and references.
df = raw.copy()
df = df.loc[:, ~df.columns.astype(str).str.startswith('Unnamed')]

for col in TARGET_COLS:
    if col not in df.columns:
        raise ValueError(f'Missing target column: {col}')

# Convert all non-reference columns to numeric when possible.
for col in df.columns:
    if col not in REFERENCE_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows with missing targets.  The retained-dataset row number is the
# permanent identifier used for all subsequent splits and exported tables.
df = df.dropna(subset=TARGET_COLS).reset_index(drop=True)
df['Original_Index'] = df.index.astype(int)

feature_cols = [c for c in df.columns if c not in TARGET_COLS + REFERENCE_COLS + ['Original_Index'] and pd.api.types.is_numeric_dtype(df[c])]
X_full = df[feature_cols].copy()
Y = df[TARGET_COLS].copy()

print('Cleaned shape:', df.shape)
print('Feature count:', len(feature_cols))
print('Features:', feature_cols)
print('Targets:', TARGET_COLS)
print('Physics-retention features present:', [c for c in PHYSICS_KEEP if c in X_full.columns])

desc = pd.DataFrame({
    'Column': feature_cols + TARGET_COLS,
    'Missing_%': [df[c].isna().mean()*100 for c in feature_cols + TARGET_COLS],
    'Min': [df[c].min() for c in feature_cols + TARGET_COLS],
    'Max': [df[c].max() for c in feature_cols + TARGET_COLS],
    'Mean': [df[c].mean() for c in feature_cols + TARGET_COLS],
    'Std': [df[c].std() for c in feature_cols + TARGET_COLS],
})
display(desc)
save_df(desc, 'dataset_column_summary.csv')

# %% cell 5

# ============================================================
# 5. Missing-value handling comparison
# ============================================================
imputers = {
    'mean': SimpleImputer(strategy='mean'),
    'median': SimpleImputer(strategy='median'),
    'knn': KNNImputer(n_neighbors=5, weights='distance'),
    'iterative': IterativeImputer(random_state=RANDOM_STATE, max_iter=30, sample_posterior=False)
}

base_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
missing_results=[]

for imp_name, imp in imputers.items():
    scores=[]
    for tr_idx, te_idx in rkf.split(X_full):
        X_tr = pd.DataFrame(imp.fit_transform(X_full.iloc[tr_idx]), columns=X_full.columns)
        X_te = pd.DataFrame(imp.transform(X_full.iloc[te_idx]), columns=X_full.columns)
        y_tr = Y.iloc[tr_idx].values
        y_te = Y.iloc[te_idx].values
        m = clone(base_model)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        scores.append(normalized_rmse_score(y_te, pred))
    missing_results.append({'Imputer': imp_name, 'Mean_NRMSE': np.mean(scores), 'Std_NRMSE': np.std(scores)})

missing_df = pd.DataFrame(missing_results).sort_values('Mean_NRMSE')
display(missing_df)
save_df(missing_df, 'missing_value_strategy_comparison.csv')

BEST_IMPUTER_NAME = missing_df.iloc[0]['Imputer']
BEST_IMPUTER = imputers[BEST_IMPUTER_NAME]
print('Selected imputer:', BEST_IMPUTER_NAME)

X_imp = pd.DataFrame(BEST_IMPUTER.fit_transform(X_full), columns=X_full.columns)

# %% cell 6

# ============================================================
# 6. Hybrid target-aware feature selection with physics retention
# ============================================================
def average_mi_scores(X, Y, random_state=RANDOM_STATE):
    X = pd.DataFrame(X).copy()
    Y = pd.DataFrame(Y).copy()
    mi_all = []
    for col in Y.columns:
        mi = mutual_info_regression(X, Y[col].values, random_state=random_state)
        mi = np.asarray(mi, dtype=float)
        if np.nanmax(mi) > 0:
            mi = mi / np.nanmax(mi)
        mi_all.append(mi)
    avg = np.mean(np.vstack(mi_all), axis=0)
    return pd.Series(avg, index=X.columns).sort_values(ascending=False)

def correlation_filter(X, relevance, threshold=0.95, physics_keep=None):
    physics_keep = set(physics_keep or [])
    X = pd.DataFrame(X).copy()
    corr = X.corr().abs().fillna(0)
    cols = list(X.columns)
    drop = set()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            a,b = cols[i], cols[j]
            if a in drop or b in drop:
                continue
            if corr.loc[a,b] > threshold:
                # Always protect physics-retained variables when possible.
                if a in physics_keep and b not in physics_keep:
                    drop.add(b)
                elif b in physics_keep and a not in physics_keep:
                    drop.add(a)
                else:
                    # Drop lower relevance; if equal, drop b.
                    if relevance.get(a,0) >= relevance.get(b,0):
                        drop.add(b)
                    else:
                        drop.add(a)
    kept = [c for c in cols if c not in drop]
    return kept, sorted(drop)

def embedded_stability_scores(X, Y, n_runs=10, top_k=10, random_state=RANDOM_STATE):
    X = pd.DataFrame(X).copy(); Y = pd.DataFrame(Y).copy()
    counts = pd.Series(0, index=X.columns, dtype=float)
    importances = pd.Series(0.0, index=X.columns, dtype=float)
    for s in range(n_runs):
        model = ExtraTreesRegressor(n_estimators=300, random_state=random_state+s, n_jobs=-1, max_features='sqrt')
        model.fit(X, Y.values)
        imp = pd.Series(model.feature_importances_, index=X.columns)
        importances += imp
        top = imp.sort_values(ascending=False).head(min(top_k, X.shape[1])).index
        counts.loc[top] += 1
    stability = counts / n_runs
    mean_importance = importances / n_runs
    return pd.DataFrame({'Feature': X.columns, 'Stability': stability.values, 'MeanImportance': mean_importance.values}).sort_values(['Stability','MeanImportance'], ascending=False)

def hybrid_select_features(X_train, Y_train, physics_keep=None, corr_threshold=0.95, mi_top_k=12, stability_top_k=12, stability_runs=10, rfecv=True):
    physics_keep = [f for f in (physics_keep or []) if f in X_train.columns]
    X = pd.DataFrame(X_train).copy(); Ydf = pd.DataFrame(Y_train, columns=TARGET_COLS)

    # Remove near-zero variance, but keep physics features if present.
    variances = X.var(numeric_only=True)
    keep_var = [c for c in X.columns if (variances.get(c,0) > 1e-12) or (c in physics_keep)]
    Xv = X[keep_var].copy()

    # MI relevance before correlation filter.
    mi_scores = average_mi_scores(Xv, Ydf)
    kept_corr, dropped_corr = correlation_filter(Xv, mi_scores, threshold=corr_threshold, physics_keep=physics_keep)
    Xc = Xv[kept_corr].copy()
    mi_scores_c = average_mi_scores(Xc, Ydf)
    mi_features = list(mi_scores_c.head(min(mi_top_k, len(mi_scores_c))).index)

    # Embedded stability.
    stab_df = embedded_stability_scores(Xc, Ydf, n_runs=stability_runs, top_k=min(stability_top_k, Xc.shape[1]))
    stable_features = list(stab_df.head(min(stability_top_k, len(stab_df)))['Feature'])

    pure_candidate = sorted(set(mi_features).union(stable_features))
    physics_retained_candidate = sorted(set(pure_candidate).union(physics_keep))

    # Optional RFECV refinement over candidate features. Physics features are re-added after RFECV.
    final_pure = pure_candidate.copy()
    final_physics = physics_retained_candidate.copy()
    rfecv_selected = []
    if rfecv and len(physics_retained_candidate) >= 5:
        try:
            estimator = ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
            selector = RFECV(estimator=estimator, step=1, cv=KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE), scoring='neg_mean_squared_error', min_features_to_select=min(5, len(physics_retained_candidate)))
            selector.fit(Xc[physics_retained_candidate], Ydf.values)
            rfecv_selected = list(pd.Index(physics_retained_candidate)[selector.support_])
            final_physics = sorted(set(rfecv_selected).union(physics_keep))
            # Pure version removes forced physics if RFECV did not select them.
            final_pure = sorted([f for f in rfecv_selected if f not in physics_keep or f in pure_candidate])
            if len(final_pure) < 3:
                final_pure = pure_candidate
        except Exception as e:
            print('RFECV failed, using candidate features. Reason:', repr(e))

    details = {
        'variance_kept': keep_var,
        'correlation_kept': kept_corr,
        'correlation_dropped': dropped_corr,
        'mi_scores': mi_scores_c.reset_index().rename(columns={'index':'Feature',0:'AvgMI'}),
        'stability': stab_df,
        'pure_candidate': pure_candidate,
        'physics_retained_candidate': physics_retained_candidate,
        'rfecv_selected': rfecv_selected,
        'final_pure': final_pure,
        'final_physics': final_physics,
        'physics_forced': physics_keep
    }
    return final_pure, final_physics, details

# Holdout split first: feature selection must be fit only on training data.
X_train_full, X_test_full, y_train, y_test = train_test_split(X_imp, Y, test_size=0.20, random_state=RANDOM_STATE)

# Permanent split manifest.  The identifier is retained-dataset row identity,
# not a modeling feature, and is never regenerated after this point.
outer_train_indices = set(X_train_full.index.astype(int))
test_indices = set(X_test_full.index.astype(int))
proper_train_indices, calibration_indices = train_test_split(
    X_train_full.index.to_numpy(), test_size=0.25, random_state=RANDOM_STATE
)
proper_train_indices = set(map(int, proper_train_indices))
calibration_indices = set(map(int, calibration_indices))

assert proper_train_indices.isdisjoint(calibration_indices)
assert proper_train_indices.isdisjoint(test_indices)
assert calibration_indices.isdisjoint(test_indices)
assert proper_train_indices | calibration_indices | test_indices == set(df.index.astype(int))
assert len(proper_train_indices) == 270
assert len(calibration_indices) == 91
assert len(test_indices) == 91

def split_name(row_index):
    row_index = int(row_index)
    if row_index in proper_train_indices:
        return 'Proper_Training'
    if row_index in calibration_indices:
        return 'Calibration'
    if row_index in test_indices:
        return 'Final_Test'
    raise KeyError(f'Unassigned retained-dataset index: {row_index}')

df['Data_Split'] = [split_name(i) for i in df.index]
split_manifest = df[['Original_Index', 'Data_Split']].copy().sort_values('Original_Index')
split_summary = (
    split_manifest.groupby('Data_Split', sort=False)
    .size().rename('Count').reset_index()
)
split_summary['Expected_Count'] = split_summary['Data_Split'].map({
    'Proper_Training': 270, 'Calibration': 91, 'Final_Test': 91
})
assert set(split_summary['Data_Split']) == {'Proper_Training', 'Calibration', 'Final_Test'}
assert (split_summary['Count'] == split_summary['Expected_Count']).all()
save_df(split_manifest, 'split_manifest.csv')
save_df(split_summary, 'split_summary.csv')
print('Split counts:', dict(zip(split_summary['Data_Split'], split_summary['Count'])))

pure_features, physics_features, fs_details = hybrid_select_features(
    X_train_full, y_train,
    physics_keep=PHYSICS_KEEP,
    corr_threshold=0.95,
    mi_top_k=12,
    stability_top_k=12,
    stability_runs=10,
    rfecv=True
)

print('Pure hybrid features:', pure_features)
print('Physics-retained features:', physics_features)
print('Forced physics features:', fs_details['physics_forced'])

save_df(pd.DataFrame({'Pure_Hybrid_Features': pd.Series(pure_features)}), 'selected_features_pure_hybrid.csv')
save_df(pd.DataFrame({'Physics_Retained_Features': pd.Series(physics_features)}), 'selected_features_physics_retained.csv')
save_df(fs_details['mi_scores'], 'feature_selection_mi_scores.csv')
save_df(fs_details['stability'], 'feature_selection_stability_scores.csv')

X_train_pure = X_train_full[pure_features]
X_test_pure  = X_test_full[pure_features]
X_train_phys = X_train_full[physics_features]
X_test_phys  = X_test_full[physics_features]

# %% cell 7

# ============================================================
# 7. Article-style baseline definitions: Choi + Yang Top-12 feature-selection pipeline
# ============================================================
def make_choi_models(gpu=False):
    """
    Choi-style baseline:
    Ridge, RF, XGB, and MLP using all available composition + heat-treatment features.
    This closely matches the modelling family used by Choi et al., while being evaluated on our dataset.
    """
    xgb_params = get_xgb_params(gpu=gpu)
    return {
        'Choi_Ridge_all_mean': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0, random_state=RANDOM_STATE))
        ]),
        'Choi_RF_all_mean': RandomForestRegressor(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=1
        ),
        'Choi_XGB_all_mean': MultiOutputRegressor(XGBRegressor(**xgb_params)),
        'Choi_MLP_all_mean': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(
                hidden_layer_sizes=(30,),
                activation='relu',
                max_iter=3000,
                random_state=RANDOM_STATE,
                early_stopping=True
            ))
        ])
    }


def yang_select_top_k_features(X, y, target_name, top_k=12):
    """
    Yang-style target-wise Pearson feature selection.

    This adapted benchmark selects exactly top_k features for each target
    using absolute Pearson correlation. It follows Yang et al.'s target-wise
    correlation/predictor-importance feature-selection philosophy, but uses
    an equal feature budget for all targets for fair comparison with our
    12-feature physics-retained model.
    """
    Xdf = pd.DataFrame(X).copy()
    y = np.asarray(y)

    rows = []
    for col in Xdf.columns:
        x = Xdf[col].values
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            corr = 0.0
        else:
            try:
                corr = np.corrcoef(x, y)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            except Exception:
                corr = 0.0

        rows.append({
            "Target": target_name,
            "Feature": col,
            "Pearson_Corr": float(corr),
            "Abs_Pearson_Corr": float(abs(corr))
        })

    corr_df = pd.DataFrame(rows).sort_values("Abs_Pearson_Corr", ascending=False)
    selected = corr_df.head(min(top_k, Xdf.shape[1]))["Feature"].tolist()

    return selected, corr_df


class YangTop12TargetWiseSVR(BaseEstimator, RegressorMixin):
    """
    Yang-style Top-12 target-wise SVR baseline.

    Original Yang et al. logic included:
    1) Target-wise correlation / predictor-importance based input selection.
    2) Separate SVR model for each target: YS, UTS, and El.
    3) Hyperparameter optimization and CV.

    This implementation adapts that logic to the present aged-forged Al alloy dataset:
    - Exactly top_k_features features are selected separately for each target
      using absolute Pearson correlation.
    - SVR hyperparameters are tuned separately for each target using CV.
    - One independent RBF-SVR is trained for each target.

    We do NOT use Yang's fixed hyperparameters because those were optimized
    on a different die-casting dataset.
    """
    def __init__(
        self,
        target_names=None,
        top_k_features=12,
        n_iter=35,
        cv=5,
        random_state=42,
        n_jobs=-1,
        kernel='rbf'
    ):
        self.target_names = target_names
        self.top_k_features = top_k_features
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.kernel = kernel

    def fit(self, X, y):
        Xdf = pd.DataFrame(X).copy()
        y = np.asarray(y)

        target_names = self.target_names
        if target_names is None:
            target_names = ['YS', 'UTS', 'El']
        self.target_names_ = list(target_names)

        # Search space chosen around common SVR settings and Yang-reported ranges,
        # but optimized on the present dataset rather than fixed from another dataset.
        param_dist = {
            'svr__C': np.logspace(-1, 3, 25),
            'svr__gamma': list(np.logspace(-3, 1, 25)) + ['scale', 'auto'],
            'svr__epsilon': [0.0004, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        }

        self.models_ = []
        self.features_ = {}
        self.feature_correlations_ = {}
        self.best_params_ = {}
        self.best_cv_scores_ = {}

        for i, target in enumerate(self.target_names_):
            feats, corr_df = yang_select_top_k_features(
                Xdf,
                y[:, i],
                target_name=target,
                top_k=self.top_k_features
            )

            self.features_[target] = feats
            self.feature_correlations_[target] = corr_df

            base = Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(kernel=self.kernel))
            ])

            search = RandomizedSearchCV(
                estimator=base,
                param_distributions=param_dist,
                n_iter=self.n_iter,
                scoring='neg_root_mean_squared_error',
                cv=self.cv,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                refit=True
            )

            print(f"Yang-style Top-{self.top_k_features} FS + SVR tuning for {target}: {feats}")
            search.fit(Xdf[feats], y[:, i])

            self.models_.append(search.best_estimator_)
            self.best_params_[target] = search.best_params_
            self.best_cv_scores_[target] = search.best_score_

        return self

    def predict(self, X):
        Xdf = pd.DataFrame(X).copy()
        preds = []
        for target, model in zip(self.target_names_, self.models_):
            preds.append(model.predict(Xdf[self.features_[target]]))
        return np.vstack(preds).T

    def get_summary(self):
        rows = []
        if not hasattr(self, 'target_names_'):
            return pd.DataFrame()
        for target in self.target_names_:
            rows.append({
                'Target': target,
                'Selected_Features': ', '.join(self.features_.get(target, [])),
                'No_Features': len(self.features_.get(target, [])),
                'Best_Params': json.dumps(self.best_params_.get(target, {})),
                'Best_CV_Neg_RMSE': self.best_cv_scores_.get(target, np.nan)
            })
        return pd.DataFrame(rows)

    def get_correlation_table(self):
        """Return target-wise Pearson correlation ranking used for Yang Top-12 feature selection."""
        rows = []
        if not hasattr(self, 'feature_correlations_'):
            return pd.DataFrame()
        for target, corr_df in self.feature_correlations_.items():
            selected = set(self.features_.get(target, []))
            tmp = corr_df.copy()
            tmp['Selected_By_Yang_FS'] = tmp['Feature'].isin(selected)
            rows.append(tmp)
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)


def make_yang_model():
    return {
        'Yang_Top12_TargetWise_SVR': YangTop12TargetWiseSVR(
            target_names=['YS', 'UTS', 'El'],
            top_k_features=12,
            n_iter=35,
            cv=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            kernel='rbf'
        )
    }

# %% cell 8

# ============================================================
# 8. Proposed Optuna-tuned models
# ============================================================
def cv_objective_score(model, X, y, cv_splits=3):
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    scores=[]
    for tr, va in kf.split(X):
        m = clone(model)
        m.fit(X.iloc[tr] if isinstance(X, pd.DataFrame) else X[tr], y.iloc[tr].values if isinstance(y, pd.DataFrame) else y[tr])
        pred = m.predict(X.iloc[va] if isinstance(X, pd.DataFrame) else X[va])
        yy = y.iloc[va].values if isinstance(y, pd.DataFrame) else y[va]
        scores.append(normalized_rmse_score(yy, pred))
    return float(np.mean(scores))

def tune_xgb_mo(X, y, n_trials=40, gpu=False):
    def objective(trial):
        params = get_xgb_params(seed=RANDOM_STATE, gpu=gpu)
        params.update({
            'n_estimators': trial.suggest_int('n_estimators', 150, 900),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.55, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.55, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 20.0, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0, log=True)
        })
        model = MultiOutputRegressor(XGBRegressor(**params))
        return cv_objective_score(model, X, y, cv_splits=3)
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = get_xgb_params(seed=RANDOM_STATE, gpu=gpu); best.update(study.best_params)
    return MultiOutputRegressor(XGBRegressor(**best)), study

def tune_xgb_chain(X, y, n_trials=40, gpu=False):
    def objective(trial):
        params = get_xgb_params(seed=RANDOM_STATE, gpu=gpu)
        params.update({
            'n_estimators': trial.suggest_int('n_estimators', 150, 900),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.55, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.55, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 20.0, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0, log=True)
        })
        model = RegressorChain(XGBRegressor(**params), order=[0,1,2], random_state=RANDOM_STATE)
        return cv_objective_score(model, X, y, cv_splits=3)
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_STATE+1))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = get_xgb_params(seed=RANDOM_STATE, gpu=gpu); best.update(study.best_params)
    return RegressorChain(XGBRegressor(**best), order=[0,1,2], random_state=RANDOM_STATE), study

def tune_catboost_mo(X, y, n_trials=30):
    if not CATBOOST_AVAILABLE:
        return None, None
    def objective(trial):
        params = dict(
            iterations=trial.suggest_int('iterations', 200, 900),
            depth=trial.suggest_int('depth', 2, 8),
            learning_rate=trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1e-3, 20.0, log=True),
            loss_function='RMSE',
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False
        )
        model = MultiOutputRegressor(CatBoostRegressor(**params))
        return cv_objective_score(model, X, y, cv_splits=3)
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_STATE+2))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    params = dict(study.best_params)
    params.update(dict(loss_function='RMSE', random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False))
    return MultiOutputRegressor(CatBoostRegressor(**params)), study

# Tune on physics-retained features and all features to test whether feature selection helps or hurts.
N_TRIALS_XGB = 40
N_TRIALS_CAT = 25

best_models = {}
studies = {}

print('Tuning proposed models on physics-retained features:', physics_features)
model, study = tune_xgb_mo(X_train_phys, y_train, n_trials=N_TRIALS_XGB, gpu=False)  # CPU stable for small data
best_models['Proposed_XGB_MO_physFS'] = model; studies['Proposed_XGB_MO_physFS'] = study
model, study = tune_xgb_chain(X_train_phys, y_train, n_trials=N_TRIALS_XGB, gpu=False)
best_models['Proposed_XGB_Chain_physFS'] = model; studies['Proposed_XGB_Chain_physFS'] = study
model, study = tune_catboost_mo(X_train_phys, y_train, n_trials=N_TRIALS_CAT)
if model is not None:
    best_models['Proposed_CatBoost_MO_physFS'] = model; studies['Proposed_CatBoost_MO_physFS'] = study

print('Tuning proposed XGB on all features as predictive backbone...')
model, study = tune_xgb_mo(X_train_full, y_train, n_trials=N_TRIALS_XGB, gpu=False)
best_models['Proposed_XGB_MO_allFeatures'] = model; studies['Proposed_XGB_MO_allFeatures'] = study

# Also keep an untuned ExtraTrees proposed baseline on physics features.
best_models['Proposed_ExtraTrees_physFS'] = ExtraTreesRegressor(n_estimators=600, random_state=RANDOM_STATE, n_jobs=-1)

# Save Optuna summaries.
study_rows=[]
for name, st in studies.items():
    study_rows.append({'Model': name, 'Best_NRMSE': st.best_value, 'Best_Params': json.dumps(st.best_params)})
study_df=pd.DataFrame(study_rows)
display(study_df)
save_df(study_df, 'optuna_best_params_summary.csv')

# %% cell 9

# ============================================================
# 9. Holdout comparison - Choi/Yang baselines + proposed models
# ============================================================
article_models = {}
article_models.update(make_choi_models(gpu=False))
article_models.update(make_yang_model())
all_models = {**article_models, **best_models}

def get_feature_set_for_model(name):
    if name.endswith('_physFS') or 'physFS' in name:
        return X_train_phys, X_test_phys, 'Physics-retained hybrid FS'
    elif name.endswith('_pureFS') or 'pureFS' in name:
        return X_train_pure, X_test_pure, 'Pure hybrid FS'
    else:
        return X_train_full, X_test_full, 'All features / article logic'

holdout_rows=[]; fitted_models={}
for name, model in all_models.items():
    print('\nFitting:', name)
    Xtr, Xte, fset = get_feature_set_for_model(name)
    try:
        m = clone(model)
    except Exception:
        m = copy.deepcopy(model)
    try:
        m.fit(Xtr, y_train.values)
    except Exception as e:
        # If XGB GPU settings fail, retry CPU for XGB models.
        print('Initial fit failed:', repr(e))
        raise
    pred = m.predict(Xte)
    fitted_models[name] = m
    met = regression_metrics(y_test.values, pred, target_names=TARGET_COLS)
    avg = met[met['Target']=='Average'].iloc[0].to_dict()
    avg['Model'] = name
    avg['Feature_Set'] = fset
    # Correct feature-count reporting
    if name == "Yang_Top12_TargetWise_SVR" and hasattr(m, "features_"):
        yang_counts = {t: len(f) for t, f in m.features_.items()}
        avg["No_Features"] = "; ".join([f"{t}:{n}" for t, n in yang_counts.items()])
        avg["Feature_Set"] = "Yang target-wise Top-12 Pearson FS"
    else:
        avg["No_Features"] = Xtr.shape[1]
        avg["Feature_Set"] = fset
    holdout_rows.append(avg)
    display(met)

holdout_df = pd.DataFrame(holdout_rows).sort_values('R2', ascending=False)
display(holdout_df[['Model','Feature_Set','No_Features','R2','MAE','RMSE','MAPE_%']])
save_df(holdout_df, 'holdout_model_comparison_average.csv')

best_holdout_name = holdout_df.iloc[0]['Model']
best_holdout_model = fitted_models[best_holdout_name]
best_X_train, best_X_test, best_feature_label = get_feature_set_for_model(best_holdout_name)
best_pred = best_holdout_model.predict(best_X_test)
best_metrics = regression_metrics(y_test.values, best_pred, target_names=TARGET_COLS)
print('Best holdout model:', best_holdout_name, '|', best_feature_label)
display(best_metrics)
save_df(best_metrics, 'best_model_holdout_target_metrics.csv')

# %% cell 10

# ============================================================
# 10. Same-environment repeated CV comparison
# ============================================================
def evaluate_repeated_cv_model(name, model, X, y, repeats=5, splits=5):
    rkf = RepeatedKFold(n_splits=splits, n_repeats=repeats, random_state=RANDOM_STATE)
    rows=[]
    for fold, (tr, te) in enumerate(rkf.split(X), 1):
        try:
            m = clone(model)
        except Exception:
            m = copy.deepcopy(model)
        Xtr = X.iloc[tr] if isinstance(X, pd.DataFrame) else X[tr]
        Xte = X.iloc[te] if isinstance(X, pd.DataFrame) else X[te]
        ytr = y.iloc[tr].values if isinstance(y, pd.DataFrame) else y[tr]
        yte = y.iloc[te].values if isinstance(y, pd.DataFrame) else y[te]
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        met = regression_metrics(yte, pred, TARGET_COLS)
        for _, r in met.iterrows():
            d = r.to_dict(); d['Model']=name; d['Fold']=fold; rows.append(d)
    return pd.DataFrame(rows)

# Fixed feature-set CV. This compares trained model logic under same folds.
# For final manuscript, mention that holdout feature selection was fit only on train, while this table uses fixed selected features for computational practicality.
cv_all=[]
for name, model in all_models.items():
    Xtr, Xte, fset = get_feature_set_for_model(name)
    if 'physFS' in name:
        Xcv = X_imp[physics_features]
    elif 'pureFS' in name:
        Xcv = X_imp[pure_features]
    else:
        Xcv = X_imp
    print('Repeated CV:', name, '| features:', Xcv.shape[1])
    cvdf = evaluate_repeated_cv_model(name, model, Xcv, Y, repeats=3, splits=5)
    if name == "Yang_Top12_TargetWise_SVR":
        cvdf["Feature_Set"] = "Yang target-wise Top-12 Pearson FS"
        cvdf["No_Features"] = "YS:12; UTS:12; El:12"
    else:
        cvdf["Feature_Set"] = fset
        cvdf["No_Features"] = Xcv.shape[1]
    cv_all.append(cvdf)

cv_results = pd.concat(cv_all, ignore_index=True)
save_df(cv_results, 'repeated_cv_all_fold_metrics.csv')
cv_summary = cv_results[cv_results['Target']=='Average'].groupby(['Model','Feature_Set','No_Features']).agg(
    R2_mean=('R2','mean'), R2_std=('R2','std'), MAE_mean=('MAE','mean'), RMSE_mean=('RMSE','mean'), MAPE_mean=('MAPE_%','mean')
).reset_index().sort_values('R2_mean', ascending=False)
display(cv_summary)
save_df(cv_summary, 'repeated_cv_model_comparison_average.csv')

# %% cell 11

# ============================================================
# 11. Yang-style feature-selection diagnostics
# ============================================================
# Export target-wise selected features, tuned SVR parameters, and full correlation rankings.

yang_model_name = 'Yang_Top12_TargetWise_SVR'
if yang_model_name in fitted_models:
    yang_fitted = fitted_models[yang_model_name]

    if hasattr(yang_fitted, 'get_summary'):
        yang_summary = yang_fitted.get_summary()
        display(yang_summary)
        save_df(yang_summary, 'yang_targetwise_selected_features_and_svr_params.csv')

    if hasattr(yang_fitted, 'get_correlation_table'):
        yang_corr_table = yang_fitted.get_correlation_table()
        display(yang_corr_table.head(30))
        save_df(yang_corr_table, 'yang_targetwise_feature_selection_correlations.csv')
else:
    print('Yang adaptive model was not found in fitted_models. Run the holdout comparison cell first.')

# %% cell 12

# ============================================================
# 12. Actual vs predicted plots for best holdout model
# ============================================================
plot_df=[]
for i,t in enumerate(TARGET_COLS):
    yt = y_test.values[:,i]
    yp = best_pred[:,i]
    plt.figure(figsize=(5,5))
    plt.scatter(yt, yp, alpha=0.75)
    mn = min(np.min(yt), np.min(yp)); mx = max(np.max(yt), np.max(yp))
    plt.plot([mn,mx],[mn,mx], linestyle='--')
    plt.xlabel(f'Actual {t}')
    plt.ylabel(f'Predicted {t}')
    plt.title(f'{best_holdout_name}: {t}\nR²={r2_score(yt,yp):.3f}, RMSE={rmse_safe(yt,yp):.3f}')
    save_fig(f'actual_vs_predicted_{t}.png')
    for row_pos, (a, p) in enumerate(zip(yt, yp)):
        retained_index = int(best_X_test.index[row_pos])
        plot_df.append({
            'Original_Index': int(df.loc[retained_index, 'Original_Index']),
            'Data_Split': df.loc[retained_index, 'Data_Split'],
            'Target': t,
            'Actual': a,
            'Predicted': p
        })
save_df(pd.DataFrame(plot_df), 'actual_vs_predicted_values_best_model.csv')

# %% cell 13

# ============================================================
# 13. SHAP explainability
# ============================================================
# Use best XGB-like model if available; otherwise use a target-wise XGB surrogate on the best feature set.
# This avoids explaining SVR/MLP with model-agnostic SHAP, which is slower.

shap_X_train = best_X_train.copy()
shap_X_test = best_X_test.copy()

# Train target-wise XGB surrogate using best feature set for stable SHAP analysis.
shap_models = {}
shap_importance_rows=[]
for i,t in enumerate(TARGET_COLS):
    model = XGBRegressor(**get_xgb_params(seed=RANDOM_STATE+i, gpu=False))
    model.fit(shap_X_train, y_train.iloc[:,i].values)
    shap_models[t] = model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_X_test)
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame({'Target':t, 'Feature':shap_X_test.columns, 'MeanAbsSHAP':mean_abs}).sort_values('MeanAbsSHAP', ascending=False)
    shap_importance_rows.append(imp_df)
    display(imp_df.head(15))
    plt.figure(figsize=(7,5))
    shap.summary_plot(shap_values, shap_X_test, show=False, max_display=min(15, shap_X_test.shape[1]))
    plt.title(f'SHAP summary for {t}')
    save_fig(f'shap_summary_{t}.png')

shap_imp = pd.concat(shap_importance_rows, ignore_index=True)
save_df(shap_imp, 'shap_feature_importance_targetwise.csv')

# %% cell 14
# ============================================================
# 14. Corrected ensemble-conformal uncertainty for Reviewer 1, Comment 1
# ============================================================
# Only the original training portion is divided into proper-training
# and calibration subsets. The original test set remains untouched.
# The selected point-prediction model and its hyperparameters are retained.

base_u = clone(best_holdout_model)

Xtr_u, Xcal_u, ytr_u, ycal_u = train_test_split(
    best_X_train, y_train, test_size=0.25, random_state=RANDOM_STATE
)

# Confirm that the uncertainty split is exactly the permanent split manifest.
assert set(Xtr_u.index.astype(int)) == proper_train_indices
assert set(Xcal_u.index.astype(int)) == calibration_indices
assert set(best_X_test.index.astype(int)) == test_indices
assert proper_train_indices.isdisjoint(calibration_indices)
assert proper_train_indices.isdisjoint(test_indices)
assert calibration_indices.isdisjoint(test_indices)

N_ENSEMBLE = 20
EPSILON = 1e-8
conformal_alpha = 0.10

ensemble_preds_cal = []
ensemble_preds_test = []

for s in range(N_ENSEMBLE):
    m = clone(base_u)
    try:
        if hasattr(m, 'random_state'):
            m.set_params(random_state=RANDOM_STATE + s)
        elif isinstance(m, MultiOutputRegressor) and hasattr(m.estimator, 'random_state'):
            m.estimator.set_params(random_state=RANDOM_STATE + s)
    except Exception:
        pass

    rng = np.random.default_rng(RANDOM_STATE + s)
    boot_idx = rng.choice(len(Xtr_u), size=len(Xtr_u), replace=True)
    m.fit(Xtr_u.iloc[boot_idx], ytr_u.iloc[boot_idx].values)
    ensemble_preds_cal.append(m.predict(Xcal_u))
    ensemble_preds_test.append(m.predict(best_X_test))

cal_preds = np.stack(ensemble_preds_cal, axis=0)
test_preds = np.stack(ensemble_preds_test, axis=0)
cal_mean = cal_preds.mean(axis=0)
test_mean = test_preds.mean(axis=0)
cal_std = cal_preds.std(axis=0, ddof=0)
test_std = test_preds.std(axis=0, ddof=0)

# Locally adaptive normalized conformal scores.
cal_scores = np.abs(ycal_u.values - cal_mean) / (cal_std + EPSILON)

# Finite-sample conformal order statistic.
n_cal = len(ycal_u)
conformal_rank = int(np.ceil((n_cal + 1) * (1.0 - conformal_alpha)))
conformal_rank = min(max(conformal_rank, 1), n_cal)
q = np.sort(cal_scores, axis=0)[conformal_rank - 1, :]

# Corrected sample-specific intervals.
test_radius = q[None, :] * (test_std + EPSILON)
lower = test_mean - test_radius
upper = test_mean + test_radius
width_values = upper - lower

covered = (y_test.values >= lower) & (y_test.values <= upper)
marginal_coverage = covered.mean(axis=0)
simultaneous_coverage = covered.all(axis=1).mean()

unc_df = pd.DataFrame({
    'Target': TARGET_COLS,
    'Nominal_Coverage': 1.0 - conformal_alpha,
    'Observed_Marginal_Coverage': marginal_coverage,
    'Observed_Simultaneous_Coverage': simultaneous_coverage,
    'Conformal_q_Normalized': q,
    'Min_Interval_Width': width_values.min(axis=0),
    'Mean_Interval_Width': width_values.mean(axis=0),
    'Median_Interval_Width': np.median(width_values, axis=0),
    'Max_Interval_Width': width_values.max(axis=0),
    'Mean_Ensemble_Std': test_std.mean(axis=0),
    'N_Calibration': n_cal,
    'Conformal_Rank': conformal_rank,
    'N_Ensemble': N_ENSEMBLE,
    'Random_State': RANDOM_STATE
})
display(unc_df)
save_df(unc_df, 'uncertainty_coverage_width.csv')

coverage_audit = pd.DataFrame([{
    'Original_Training_N': len(best_X_train),
    'Proper_Training_N': len(Xtr_u),
    'Calibration_N': len(Xcal_u),
    'Final_Test_N': len(best_X_test),
    'Calibration_Uses_Final_Test_Targets': False,
    'Ensemble_Fit_Uses_Final_Test_Targets': False,
    'Conformal_Alpha': conformal_alpha,
    'Conformal_Rank': conformal_rank,
    'Simultaneous_Coverage': simultaneous_coverage
}])
display(coverage_audit)
save_df(coverage_audit, 'comment1_uncertainty_audit.csv')
print('Calibration leakage detected:', False)
print('Proper training / calibration / final test sizes:', len(Xtr_u), len(Xcal_u), len(best_X_test))
print('Conformal rank:', conformal_rank, '| alpha:', conformal_alpha)
print('Observed simultaneous coverage:', round(float(simultaneous_coverage), 6))

rows = []
for idx in range(len(best_X_test)):
    retained_index = int(best_X_test.index[idx])
    for j, t in enumerate(TARGET_COLS):
        rows.append({
            'SampleIndex': retained_index,
            'Original_Index': int(df.loc[retained_index, 'Original_Index']),
            'Data_Split': df.loc[retained_index, 'Data_Split'],
            'Target': t,
            'Actual': y_test.values[idx, j],
            'PredMean': test_mean[idx, j],
            'Lower': lower[idx, j],
            'Upper': upper[idx, j],
            'IntervalWidth': width_values[idx, j],
            'EnsembleStd': test_std[idx, j],
            'Conformal_q_Normalized': q[j]
        })
interval_df = pd.DataFrame(rows)
save_df(interval_df, 'test_prediction_intervals_best_model.csv')

for j, t in enumerate(TARGET_COLS):
    order = np.argsort(y_test.values[:, j])
    plt.figure(figsize=(9, 4))
    x = np.arange(len(order))
    plt.plot(x, y_test.values[order, j], label='Actual')
    plt.plot(x, test_mean[order, j], label='Predicted')
    plt.fill_between(x, lower[order, j], upper[order, j], alpha=0.25, label='90% interval')
    plt.title(f'Corrected conformal interval: {t}')
    plt.xlabel('Test samples sorted by actual value')
    plt.ylabel(t)
    plt.legend()
    save_fig(f'uncertainty_interval_{t}.png')

# %% cell 15
# ============================================================
# 15. Corrected reliability-aware candidate screening
# ============================================================
# Candidate ensembles use only the proper-training subset from Cell 14.
# Candidate target values, including final-test targets, are not used
# for fitting, conformal calibration, scoring, or ranking.
# The all-row result is retained as a diagnostic; the publication result is
# restricted to the 91 untouched final-test conditions.

if 'q' not in globals():
    raise RuntimeError("Run Cell 14 first. It computes corrected conformal quantiles.")

if 'physFS' in best_holdout_name:
    full_features_for_best = physics_features
elif 'pureFS' in best_holdout_name:
    full_features_for_best = pure_features
else:
    full_features_for_best = list(X_imp.columns)

X_screen = X_imp[full_features_for_best].copy()

def set_random_state_safely(model, seed):
    try:
        if hasattr(model, 'random_state'):
            model.set_params(random_state=seed)
    except Exception:
        pass
    try:
        if isinstance(model, MultiOutputRegressor) and hasattr(model.estimator, 'random_state'):
            model.estimator.set_params(random_state=seed)
    except Exception:
        pass
    try:
        if isinstance(model, RegressorChain):
            if hasattr(model, 'random_state'):
                model.set_params(random_state=seed)
            if hasattr(model, 'base_estimator') and hasattr(model.base_estimator, 'random_state'):
                model.base_estimator.set_params(random_state=seed)
            if hasattr(model, 'estimator') and hasattr(model.estimator, 'random_state'):
                model.estimator.set_params(random_state=seed)
    except Exception:
        pass
    return model

N_CAND_ENSEMBLE = N_ENSEMBLE
candidate_preds_ensemble = []

print(f"Building candidate uncertainty ensemble with {N_CAND_ENSEMBLE} bootstrap models...")
print("Best holdout model:", best_holdout_name)
print("Feature set used:", full_features_for_best)
print("Candidate ensemble training rows:", len(Xtr_u))

for s in range(N_CAND_ENSEMBLE):
    m = clone(base_u)
    m = set_random_state_safely(m, RANDOM_STATE + 1000 + s)
    rng = np.random.default_rng(RANDOM_STATE + 1000 + s)
    boot_idx = rng.choice(len(Xtr_u), size=len(Xtr_u), replace=True)
    m.fit(Xtr_u.iloc[boot_idx], ytr_u.iloc[boot_idx].values)
    candidate_preds_ensemble.append(m.predict(X_screen))

candidate_preds_ensemble = np.stack(candidate_preds_ensemble, axis=0)
candidate_pred_mean = candidate_preds_ensemble.mean(axis=0)
candidate_pred_std = candidate_preds_ensemble.std(axis=0, ddof=0)

# Keep only predictors and traceability fields in the scoring table.  Observed
# targets are joined later with an explicit retrospective label and are never
# passed to a score or rank calculation.
screen = df[['Original_Index', 'Data_Split'] + feature_cols].copy()
target_std_train = ytr_u.std(axis=0, ddof=1).values

for j, t in enumerate(TARGET_COLS):
    candidate_radius = q[j] * (candidate_pred_std[:, j] + EPSILON)
    screen[f'Pred_{t}'] = candidate_pred_mean[:, j]
    screen[f'EnsembleStd_{t}'] = candidate_pred_std[:, j]
    screen[f'Lower_{t}'] = candidate_pred_mean[:, j] - candidate_radius
    screen[f'Upper_{t}'] = candidate_pred_mean[:, j] + candidate_radius
    screen[f'UncWidth_{t}'] = 2.0 * candidate_radius
    screen[f'NormalizedUncWidth_{t}'] = screen[f'UncWidth_{t}'] / max(target_std_train[j], EPSILON)

def minmax(s):
    s = pd.Series(s, dtype=float)
    return (s - s.min()) / (s.max() - s.min() + 1e-12)

def add_score_components(table):
    """Apply the existing score components without observed target values."""
    table = table.copy()
    table['StrengthScore'] = 0.5 * minmax(table['Pred_YS']) + 0.5 * minmax(table['Pred_UTS'])
    table['DuctilityScore'] = minmax(table['Pred_El'])
    table['NormalizedUncertaintyPenalty'] = table[
        [f'NormalizedUncWidth_{t}' for t in TARGET_COLS]
    ].mean(axis=1)
    # Backward-compatible alias used by the earlier diagnostic output.
    table['UncertaintyPenalty'] = table['NormalizedUncertaintyPenalty']
    table['PropertyOnlyScore'] = 0.4 * table['StrengthScore'] + 0.4 * table['DuctilityScore']
    table['ReliabilityAwareScore'] = table['PropertyOnlyScore'] - 0.2 * table['NormalizedUncertaintyPenalty']
    return table

def deterministic_desc_rank(table, score_column):
    """Descending score rank; ascending Original_Index resolves ties."""
    ordered = table.sort_values(
        [score_column, 'Original_Index'], ascending=[False, True], kind='mergesort'
    )
    ranks = pd.Series(np.arange(1, len(ordered) + 1), index=ordered.index)
    return ranks.reindex(table.index).astype(int)

# Diagnostic ranking across all 452 retained rows.  It is not used as the
# publication screening result.
screen_all = add_score_components(screen)
screen_all['PropertyOnlyRank'] = deterministic_desc_rank(screen_all, 'PropertyOnlyScore')
screen_all['ReliabilityAwareRank'] = deterministic_desc_rank(screen_all, 'ReliabilityAwareScore')
screen_all['RankShift'] = screen_all['PropertyOnlyRank'] - screen_all['ReliabilityAwareRank']

dedupe_cols = [c for c in feature_cols if c in screen_all.columns]
screen_all_dedup = (
    screen_all.sort_values(
        ['ReliabilityAwareScore', 'Original_Index'], ascending=[False, True], kind='mergesort'
    ).drop_duplicates(subset=dedupe_cols, keep='first')
)
diagnostic_candidate_cols = (
    ['Original_Index', 'Data_Split'] + dedupe_cols +
    [f'Pred_{t}' for t in TARGET_COLS] +
    [f'Lower_{t}' for t in TARGET_COLS] +
    [f'Upper_{t}' for t in TARGET_COLS] +
    [f'EnsembleStd_{t}' for t in TARGET_COLS] +
    [f'UncWidth_{t}' for t in TARGET_COLS] +
    [f'NormalizedUncWidth_{t}' for t in TARGET_COLS] +
    ['StrengthScore', 'DuctilityScore', 'PropertyOnlyScore', 'PropertyOnlyRank',
     'NormalizedUncertaintyPenalty', 'UncertaintyPenalty', 'ReliabilityAwareScore',
     'ReliabilityAwareRank', 'RankShift']
)
top_candidates = screen_all_dedup[diagnostic_candidate_cols].head(30)
display(top_candidates)
save_df(top_candidates, 'top_reliability_aware_candidates_deduplicated.csv')
save_df(screen_all_dedup[diagnostic_candidate_cols], 'all_reliability_aware_candidates_deduplicated.csv')

# Diagnose the previously reported indices before restricting the publication
# result to held-out rows.
diagnostic_indices = [344, 267]
winner_diagnostics = screen_all[screen_all['Original_Index'].isin(diagnostic_indices)].copy()
winner_diagnostics['Ensemble_Fit_Contains_Record'] = winner_diagnostics['Data_Split'].eq('Proper_Training')
winner_diagnostics['Near_Zero_Interval_Width_Any_Target'] = (
    winner_diagnostics[[f'UncWidth_{t}' for t in TARGET_COLS]].max(axis=1) <= 1e-6
)
winner_diagnostics['Interpretation'] = np.where(
    winner_diagnostics['Data_Split'].eq('Proper_Training'),
    'Proper-training observation; near-zero width is not independent held-out uncertainty.',
    'Diagnostic observation; not used as the publication winner.'
)
winner_diagnostics = winner_diagnostics.sort_values('Original_Index')
save_df(winner_diagnostics, 'candidate_winner_diagnostics.csv')

# Publication-ready result: retrospective reliability-aware screening of
# previously unseen held-out alloy conditions.  Normalization is computed
# within these 91 rows; the existing 0.4/0.4/-0.2 weights are unchanged.
screen_test = add_score_components(screen[screen['Data_Split'].eq('Final_Test')].copy())
screen_test['PropertyOnlyRank'] = deterministic_desc_rank(screen_test, 'PropertyOnlyScore')
screen_test['ReliabilityAwareRank'] = deterministic_desc_rank(screen_test, 'ReliabilityAwareScore')
screen_test['RankShift'] = screen_test['PropertyOnlyRank'] - screen_test['ReliabilityAwareRank']
assert len(screen_test) == 91
assert screen_test['Original_Index'].is_unique
assert screen_test['Data_Split'].eq('Final_Test').all()

heldout_ordered = screen_test.sort_values(
    ['ReliabilityAwareRank', 'Original_Index'], ascending=[True, True], kind='mergesort'
)
heldout_cols = (
    ['Original_Index', 'Data_Split'] + dedupe_cols +
    [f'Pred_{t}' for t in TARGET_COLS] +
    [f'Lower_{t}' for t in TARGET_COLS] +
    [f'Upper_{t}' for t in TARGET_COLS] +
    [f'UncWidth_{t}' for t in TARGET_COLS] +
    [f'NormalizedUncWidth_{t}' for t in TARGET_COLS] +
    ['NormalizedUncertaintyPenalty', 'StrengthScore', 'DuctilityScore',
     'PropertyOnlyScore', 'PropertyOnlyRank', 'ReliabilityAwareScore',
     'ReliabilityAwareRank', 'RankShift']
)
heldout_table = heldout_ordered[heldout_cols].copy()
for t in TARGET_COLS:
    heldout_table[f'Observed_{t}_Retrospective'] = df.loc[heldout_table.index, t].values
save_df(heldout_table, 'heldout_final_test_screening.csv')
save_df(heldout_table.head(30), 'top_heldout_reliability_aware_candidates.csv')
display(heldout_table.head(30))

# Direct before/after ranking comparison on the 91 held-out conditions.
property_winner = screen_test.loc[screen_test['PropertyOnlyRank'].idxmin()]
reliability_winner = screen_test.loc[screen_test['ReliabilityAwareRank'].idxmin()]
rank_changed = screen_test['RankShift'].ne(0)
ranking_comparison = pd.DataFrame([{
    'Screening_Description': 'Retrospective reliability-aware screening of previously unseen held-out alloy conditions',
    'N_Heldout': len(screen_test),
    'PropertyOnlyWinner_Original_Index': int(property_winner['Original_Index']),
    'ReliabilityAwareWinner_Original_Index': int(reliability_winner['Original_Index']),
    'Winner_Changed': bool(property_winner['Original_Index'] != reliability_winner['Original_Index']),
    'Changed_Rank_Count': int(rank_changed.sum()),
    'Changed_Rank_Percent': float(rank_changed.mean() * 100.0),
    'Mean_Absolute_Rank_Shift': float(screen_test['RankShift'].abs().mean()),
    'Maximum_Upward_Rank_Shift': int(screen_test['RankShift'].max()),
    'Maximum_Downward_Rank_Shift': int(screen_test['RankShift'].min()),
    'Spearman_Correlation_Between_Ranks': float(screen_test['PropertyOnlyRank'].corr(screen_test['ReliabilityAwareRank'], method='spearman')),
    'Score_Normalization_Population': 'Final_Test',
    'Ranking_Tie_Break': 'Descending score, then ascending Original_Index',
    'Scoring_Equation': 'S_property = 0.4*S_strength + 0.4*S_ductility; S_final = S_property - 0.2*P_unc'
}])
display(ranking_comparison)
save_df(ranking_comparison, 'heldout_candidate_ranking_comparison.csv')
print('Scoring equation: S_property = 0.4*StrengthScore + 0.4*DuctilityScore; '
      'S_final = S_property - 0.2*NormalizedUncertaintyPenalty')

# Validate held-out uncertainty and ranking integrity.  EPSILON is the only
# stabilizer; no zero or near-zero ensemble standard deviation is clipped.
test_positions = [X_screen.index.get_loc(i) for i in screen_test.index]
heldout_std = candidate_pred_std[test_positions, :]
near_zero_threshold = 1e-12
near_zero_mask = heldout_std <= near_zero_threshold
near_zero_rows = []
for pos, retained_index in enumerate(screen_test.index):
    targets = [TARGET_COLS[j] for j in range(len(TARGET_COLS)) if near_zero_mask[pos, j]]
    if targets:
        near_zero_rows.append({
            'Original_Index': int(df.loc[retained_index, 'Original_Index']),
            'Data_Split': df.loc[retained_index, 'Data_Split'],
            'Near_Zero_EnsembleStd_Targets': ', '.join(targets),
            'YS_EnsembleStd': heldout_std[pos, 0],
            'UTS_EnsembleStd': heldout_std[pos, 1],
            'El_EnsembleStd': heldout_std[pos, 2]
        })
near_zero_df = pd.DataFrame(near_zero_rows, columns=[
    'Original_Index', 'Data_Split', 'Near_Zero_EnsembleStd_Targets',
    'YS_EnsembleStd', 'UTS_EnsembleStd', 'El_EnsembleStd'
])
save_df(near_zero_df, 'heldout_near_zero_ensemble_std.csv')

assert set(screen_test['Original_Index'].astype(int)) == set(df.loc[list(test_indices), 'Original_Index'].astype(int))
assert np.all(screen_test[[f'UncWidth_{t}' for t in TARGET_COLS]].values >= -1e-12)
assert np.isfinite(screen_test['NormalizedUncertaintyPenalty']).all()
assert screen_test['NormalizedUncertaintyPenalty'].nunique() > 1
assert np.isfinite(screen_test['PropertyOnlyScore']).all()
assert np.isfinite(screen_test['ReliabilityAwareScore']).all()
assert set(screen_test['PropertyOnlyRank']) == set(range(1, 92))
assert set(screen_test['ReliabilityAwareRank']) == set(range(1, 92))
assert int(reliability_winner['Original_Index']) in test_indices

heldout_audit = pd.DataFrame([{
    'Screening_Description': 'Retrospective reliability-aware screening of previously unseen held-out alloy conditions',
    'Proper_Training_N': len(proper_train_indices),
    'Calibration_N': len(calibration_indices),
    'Final_Test_N': len(test_indices),
    'Final_Test_Records_Used_In_Ensemble_Fit': False,
    'Final_Test_Records_Used_In_Conformal_Calibration': False,
    'Original_Indices_Unique': bool(screen_test['Original_Index'].is_unique),
    'All_Data_Split_Values_Final_Test': bool(screen_test['Data_Split'].eq('Final_Test').all()),
    'Interval_Widths_Nonnegative': bool(np.all(screen_test[[f'UncWidth_{t}' for t in TARGET_COLS]].values >= -1e-12)),
    'Normalized_Penalties_Finite': bool(np.isfinite(screen_test['NormalizedUncertaintyPenalty']).all()),
    'Normalized_Penalties_Not_Constant': bool(screen_test['NormalizedUncertaintyPenalty'].nunique() > 1),
    'Ranking_Scores_Finite': bool(np.isfinite(screen_test[['PropertyOnlyScore', 'ReliabilityAwareScore']]).all().all()),
    'Property_Ranks_Cover_1_to_91': bool(set(screen_test['PropertyOnlyRank']) == set(range(1, 92))),
    'Reliability_Ranks_Cover_1_to_91': bool(set(screen_test['ReliabilityAwareRank']) == set(range(1, 92))),
    'Selected_Winner_Is_Final_Test': bool(int(reliability_winner['Original_Index']) in test_indices),
    'Observed_Test_Targets_Used_In_Score_or_Rank': False,
    'Near_Zero_EnsembleStd_Record_Count': len(near_zero_df),
    'Near_Zero_EnsembleStd_Threshold': near_zero_threshold,
    'Numerical_Stabilizer_Epsilon': EPSILON
}])
save_df(heldout_audit, 'heldout_screening_audit.csv')

summary_rows = []
for j, t in enumerate(TARGET_COLS):
    summary_rows.append({
        'Target': t,
        'Mean_Prediction': float(screen_test[f'Pred_{t}'].mean()),
        'Mean_EnsembleStd': float(screen_test[f'EnsembleStd_{t}'].mean()),
        'Mean_UncWidth': float(screen_test[f'UncWidth_{t}'].mean()),
        'Min_UncWidth': float(screen_test[f'UncWidth_{t}'].min()),
        'Median_UncWidth': float(screen_test[f'UncWidth_{t}'].median()),
        'Max_UncWidth': float(screen_test[f'UncWidth_{t}'].max()),
        'Training_Target_Std': float(target_std_train[j])
    })
candidate_unc_summary = pd.DataFrame(summary_rows)
display(candidate_unc_summary)
save_df(candidate_unc_summary, 'heldout_screening_uncertainty_summary.csv')

print('Held-out property-only winner:', int(property_winner['Original_Index']))
print('Held-out reliability-aware winner:', int(reliability_winner['Original_Index']))
print('Held-out rank changes:', int(rank_changed.sum()), '/', len(screen_test))
print('Top diagnostic all-row candidate:', int(top_candidates.iloc[0]['Original_Index']))
print('Candidate screening completed with held-out-only publication ranking and corrected sample-specific uncertainty widths.')

# %% cell 16

# ============================================================
# 16. Final comparison table for manuscript drafting
# ============================================================
# Combine article method context and our same-dataset results.
reported_context = pd.DataFrame([
    {
        'Study_or_Method':'Choi et al. style',
        'Original_Method':'Ridge, RF, XGB, MLP; SHAP; inverse design',
        'Original_Targets':'YS, UTS, El',
        'Reimplemented_As':'Choi_* models using all available features on our dataset'
    },
    {
        'Study_or_Method':'Yang et al. feature-selection style',
        'Original_Method':'Correlation/predictor-importance selected target-wise SVR + CV + SOM/GA application mapping',
        'Original_Targets':'YS, UTS, El',
        'Reimplemented_As':'Target-wise Top-12 Pearson feature selection + CV-tuned target-wise RBF-SVR on our dataset'
    },
    {
        'Study_or_Method':'Proposed',
        'Original_Method':'Physics-retained hybrid FS + Optuna-tuned multi-target learning + SHAP + uncertainty',
        'Original_Targets':'YS, UTS, El',
        'Reimplemented_As':'Proposed_* models'
    }
])
display(reported_context)
save_df(reported_context, 'article_reimplementation_context.csv')

final_table = holdout_df[['Model','Feature_Set','No_Features','R2','MAE','RMSE','MAPE_%']].copy()
final_table = final_table.merge(cv_summary[['Model','R2_mean','R2_std','RMSE_mean']], on='Model', how='left')
display(final_table)
save_df(final_table, 'final_holdout_and_cv_comparison_table.csv')

# %% cell 17
# ============================================================
# 17. Safe ZIP creation without recursive self-inclusion
# ============================================================
import os
import zipfile
import shutil
from pathlib import Path

TEMP_DIR = OUTPUT_DIR.parent / f"{OUTPUT_DIR.name}_tmp"
TEMP_ZIP_PATH = TEMP_DIR / "al_alloy_outputs.zip"
FINAL_ZIP_PATH = OUTPUT_DIR / "al_alloy_outputs.zip"

# Create temp folder if missing
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Remove old zip files
for p in [TEMP_ZIP_PATH, FINAL_ZIP_PATH]:
    if p.exists():
        p.unlink()

EXCLUDE_SUFFIXES = {
    ".zip", ".tmp", ".log"
}

EXCLUDE_DIR_NAMES = {
    "__pycache__", ".ipynb_checkpoints"
}

def should_exclude(path: Path):
    if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
        return True

    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True

    if path.name in {
        "al_alloy_outputs.zip",
        "al_alloy_outputs_light.zip"
    }:
        return True

    return False

print("Creating zip safely...")

with zipfile.ZipFile(
    TEMP_ZIP_PATH,
    "w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=1,
    allowZip64=True
) as zipf:
    for file_path in OUTPUT_DIR.rglob("*"):
        if file_path.is_file() and not should_exclude(file_path):
            arcname = file_path.relative_to(OUTPUT_DIR)
            zipf.write(file_path, arcname)

shutil.copy2(TEMP_ZIP_PATH, FINAL_ZIP_PATH)

print("ZIP created successfully:")
print(FINAL_ZIP_PATH)
print("Size:", round(FINAL_ZIP_PATH.stat().st_size / (1024**2), 2), "MB")

# %% cell 18
