# econ_model.py
# Core modeling + plotting helpers for the crisis dashboard
# ---------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, make_scorer
)
from sklearn.calibration import calibration_curve
from statsmodels.tsa.api import VAR
import matplotlib.ticker as mtick

# =========================
# Feature engineering
# =========================
def make_features(df: pd.DataFrame, win: int = 3) -> pd.DataFrame:
    """
    Build the feature set used by the models. Assumes columns:
      ['Year','GDP','CPI','HH Consumption','Capital Formation', 'Crisis']
    Returns a new DataFrame (sorted by Year) with engineered features.
    """
    d = df.sort_values('Year').copy()

    # Deltas
    for col in ['GDP', 'CPI', 'HH Consumption', 'Capital Formation']:
        d[f'{col}_delta'] = d[col].diff()

    # Moving averages, std, z-scores (for GDP & CPI)
    for col in ['GDP', 'CPI']:
        d[f'{col}_ma{win}'] = d[col].rolling(win).mean()
        d[f'{col}_std{win}'] = d[col].rolling(win).std()
        d[f'{col}_z{win}'] = (d[col] - d[f'{col}_ma{win}']) / (d[f'{col}_std{win}'])

    # Simple flags
    d['GDP_under0'] = (d['GDP'] < 0).astype(int)
    d['Joint_stress'] = ((d['GDP'] < d['GDP_ma3']) & (d['CPI_z3'] > 0)).astype(int)

    # Lags (t-1)
    for col in ['GDP', 'CPI', 'GDP_delta', 'CPI_delta', 'GDP_z3', 'CPI_z3']:
        if col in d.columns:
            d[f'{col}_lag1'] = d[col].shift(1)

    return d.dropna().reset_index(drop=True)


BASE_FEATURES = [
    'GDP', 'CPI', 'HH Consumption', 'Capital Formation',
    'GDP_delta', 'CPI_delta',
    'GDP_ma3', 'CPI_ma3', 'GDP_std3', 'CPI_std3', 'GDP_z3', 'CPI_z3',
    'GDP_under0', 'Joint_stress',
    'GDP_lag1', 'CPI_lag1', 'GDP_delta_lag1', 'CPI_delta_lag1', 'GDP_z3_lag1', 'CPI_z3_lag1'
]

# =========================
# Threshold & metrics
# =========================
def pick_threshold_fbeta(y_true, prob, beta: float = 2.0) -> float:
    p, r, th = precision_recall_curve(y_true, prob)
    f = (1 + beta**2) * (p * r) / (beta**2 * p + r + 1e-12)
    return th[np.argmax(f)] if len(th) else 0.5

def evaluate_probs(y_true, prob, threshold: float) -> Dict:
    y_hat = (prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    report = classification_report(y_true, y_hat, digits=4, output_dict=True, zero_division=0)
    ap = average_precision_score(y_true, prob) if np.sum(y_true) > 0 else 0.0
    return {
        'threshold': float(np.round(threshold, 3)),
        'PR_AUC': float(np.round(ap, 4)),
        'precision_1': float(np.round(report['1']['precision'], 4)),
        'recall_1': float(np.round(report['1']['recall'], 4)),
        'f1_1': float(np.round(report['1']['f1-score'], 4)),
        'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]), 'fn': int(cm[1, 0]), 'tp': int(cm[1, 1]),
    }

# =========================
# Robust scorer helpers
# =========================
def _proba_pos(scores: np.ndarray) -> np.ndarray:
    """Return positive-class vector for either predict_proba (2D) or decision_function (1D)."""
    s = np.asarray(scores)
    if s.ndim == 2:
        if s.shape[1] == 2:
            return s[:, 1]
        return s.max(axis=1)  # multi-class fallback
    return s

def _safe_ap(y_true, scores):
    p = _proba_pos(scores)
    return average_precision_score(y_true, p) if np.sum(y_true) > 0 else 0.0

ap_scorer = make_scorer(_safe_ap, needs_threshold=True)

# =========================
# Custom time-aware splitter
# =========================
class WalkForwardWithPositives:
    """
    Expanding-window walk-forward CV that enforces:
      - at least one positive in TRAIN
      - at least one positive in VALIDATION
    """
    def __init__(self, y: pd.Series, n_splits: int = 4, min_train_years: int = 8,
                 ensure_pos_in_valid: bool = True):
        self.y = np.asarray(y)
        self.n_splits = max(1, n_splits)
        self.min_train_years = max(1, min_train_years)
        self.ensure_pos_in_valid = ensure_pos_in_valid

    def split(self):
        n = len(self.y)
        val_len = max(1, n // (self.n_splits + 1))

        first_pos_idx = np.where(self.y == 1)[0]
        start_after = (first_pos_idx.min() + 1) if len(first_pos_idx) else self.min_train_years
        start = max(self.min_train_years, start_after)

        for val_start in range(start, n - val_len + 1, val_len):
            tr_idx = np.arange(0, val_start)
            va_idx = np.arange(val_start, val_start + val_len)
            if self.y[tr_idx].sum() == 0:
                continue
            if self.ensure_pos_in_valid and self.y[va_idx].sum() == 0:
                continue
            yield tr_idx, va_idx

# =========================
# Model registry
# =========================
@dataclass
class ModelSpec:
    name: str
    pipeline: Pipeline
    param_grid: Dict

def get_models() -> List[ModelSpec]:
    models = []

    # 1) L1-Logistic Regression
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            class_weight='balanced', penalty='l1',
            solver='liblinear', max_iter=2000, random_state=0))
    ])
    models.append(ModelSpec('LogReg_L1', lr, {'clf__C': [0.1, 0.3, 1.0, 3.0]}))

    # 2) Random Forest
    rf = Pipeline([
        ('clf', RandomForestClassifier(
            class_weight='balanced', n_estimators=500, random_state=0))
    ])
    models.append(ModelSpec('RandomForest', rf, {
        'clf__max_depth': [3, 5, 8, None],
        'clf__min_samples_leaf': [1, 2, 4]
    }))

    # 3) Gradient Boosting
    gb = Pipeline([('clf', GradientBoostingClassifier(random_state=0))])
    models.append(ModelSpec('GradBoost', gb, {
        'clf__n_estimators': [150, 300],
        'clf__learning_rate': [0.03, 0.07, 0.1],
        'clf__max_depth': [2, 3]
    }))
    return models

# =========================
# Plot helpers (return Axes)
# =========================
def plot_confusion_matrix_custom(y_true, y_pred, title, ax=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", cmap="Blues",
        xticklabels=["Normal","Crisis"], yticklabels=["Normal","Crisis"], ax=ax
    )
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    return ax

def plot_timeline_probs(year, y_true, y_prob, threshold, title, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10,4))
    ax.plot(year, y_prob, marker='o')
    ax.axhline(threshold, linestyle='--')
    for y, c in zip(year, y_true):
        if c == 1:
            ax.axvspan(y-0.5, y+0.5, alpha=0.15, color='tab:red')
    ax.set_title(title); ax.set_xlabel('Year'); ax.set_ylabel('Predicted crisis probability')
    ax.set_ylim(0,1)
    return ax

def plot_pr_curve(y_true, y_prob, title, ax=None):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob) if np.sum(y_true)>0 else 0.0
    if ax is None:
        _, ax = plt.subplots(figsize=(5,4))
    ax.plot(r, p)
    ax.set_title(f'{title} (AP={ap:.3f})'); ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    return ax

def plot_reliability(y_true, y_prob, n_bins=6, title='Calibration', ax=None):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='quantile')
    if ax is None:
        _, ax = plt.subplots(figsize=(5,4))
    ax.plot(mean_pred, frac_pos, marker='o')
    ax.plot([0,1],[0,1],'--')
    ax.set_title(title); ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Empirical frequency')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    return ax

def plot_logreg_coefficients(best_pipeline, feature_names, top_k=12, title='Logistic coefficients', ax=None):
    clf = best_pipeline.named_steps.get('clf', None)
    if clf is None or not hasattr(clf, 'coef_'):
        return None
    coefs = pd.Series(clf.coef_.ravel(), index=feature_names)\
             .sort_values(key=lambda s: s.abs(), ascending=False).head(top_k)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, max(3, 0.3*len(coefs))))
    coefs.sort_values().plot(kind='barh', ax=ax)
    ax.set_title(title); ax.set_xlabel('Weight')
    return ax

def plot_feature_importance(best_pipeline, feature_names, top_k=12, title='Feature importance', ax=None):
    clf = best_pipeline.named_steps.get('clf', None)
    if clf is None or not hasattr(clf, 'feature_importances_'):
        return None
    imp = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False).head(top_k)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, max(3, 0.3*len(imp))))
    imp.sort_values().plot(kind='barh', ax=ax)
    ax.set_title(title); ax.set_xlabel('Importance')
    return ax

# ======= New: easy-to-read story helpers =======
def events_from_predictions(year, y_true, y_pred) -> Dict[str, List[int]]:
    year = np.asarray(year).astype(int)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    crisis_years = [int(y) for y in year[y_true == 1]]
    flagged_years = [int(y) for y in year[y_pred == 1]]
    tp_years = sorted(set(crisis_years).intersection(flagged_years))
    missed_years = sorted([y for y in crisis_years if y not in tp_years])
    false_alarm_years = sorted([y for y in flagged_years if y not in crisis_years])
    return dict(crisis_years=crisis_years, flagged_years=flagged_years,
                tp_years=tp_years, missed_years=missed_years,
                false_alarm_years=false_alarm_years)

def estimate_lead_times(year, y_true, y_prob, threshold) -> List[int]:
    year = np.asarray(year).astype(int)
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    idx_crisis = np.where(y_true == 1)[0]
    leads = []
    for i in idx_crisis:
        j = None
        for t in range(i-1, -1, -1):
            if y_prob[t] >= threshold:
                j = t; break
        if j is not None:
            leads.append(int(year[i] - year[j]))
    return leads

def plot_risk_strip(year, y_prob, threshold, ax=None):
    """Green/amber/red bars by year to show risk level at a glance."""
    year = np.asarray(year).astype(int)
    y_prob = np.asarray(y_prob)
    colors = np.where(y_prob >= max(0.7, threshold), '#d62728',      # red high risk
             np.where(y_prob >= min(0.4, threshold), '#ff7f0e', '#2ca02c'))  # amber / green
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 1.2))
    ax.bar(year, np.ones_like(year), color=colors, width=0.9)
    ax.set_yticks([]); ax.set_ylim(0, 1.1)
    ax.set_xlabel('Year'); ax.set_title('Risk strip (green = low, amber = watch, red = high)')
    return ax

# =========================
# Scenario A — Train on A, test on B
# =========================
def train_on_A_test_on_B(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    beta: float = 2.0,
    n_splits: int = 4,
    min_train_years: int = 8,
    top_k=None,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Returns:
      - tbl: DataFrame with metrics per model
      - artifacts: dict[model_name] -> {best, threshold, feature_cols, year, y_true, y_prob, y_pred}
    """
    A, B = make_features(train_df), make_features(test_df)
    cols = [c for c in feature_cols if c in A.columns and c in B.columns]
    X_tr, y_tr = A[cols], A['Crisis']
    X_te, y_te = B[cols], B['Crisis']

    wf = WalkForwardWithPositives(y_tr, n_splits=n_splits,
                                  min_train_years=min_train_years,
                                  ensure_pos_in_valid=True)
    cv = list(wf.split())
    if len(cv) == 0:
        raise ValueError("WalkForwardWithPositives produced 0 folds.")

    rows, artifacts = [], {}

    for spec in get_models():
        gcv = GridSearchCV(spec.pipeline, spec.param_grid, cv=cv,
                           scoring=ap_scorer, error_score=0.0)
        gcv.fit(X_tr, y_tr)
        best = gcv.best_estimator_

        prob_tr = _proba_pos(best.predict_proba(X_tr) if hasattr(best, "predict_proba")
                             else best.decision_function(X_tr))
        t_star = pick_threshold_fbeta(y_tr, prob_tr, beta=beta)

        prob_te = _proba_pos(best.predict_proba(X_te) if hasattr(best, "predict_proba")
                             else best.decision_function(X_te))
        metrics = evaluate_probs(y_te, prob_te, t_star)
        metrics.update({'model': spec.name, 'best_params': gcv.best_params_})
        rows.append(metrics)

        artifacts[spec.name] = dict(
            best=best, threshold=t_star, feature_cols=cols,
            year=B['Year'].values, y_true=y_te.values,
            y_prob=prob_te, y_pred=(prob_te >= t_star).astype(int)
        )
    tbl = pd.DataFrame(rows).sort_values(['PR_AUC', 'recall_1'], ascending=False).reset_index(drop=True)
    if top_k is not None:
        tbl = tbl.head(top_k)
    return tbl, artifacts

# =========================
# Scenario B — LOCO (group-aware CV)
# =========================
def _prepare_country(df: pd.DataFrame, name: str) -> pd.DataFrame:
    d = make_features(df).copy()
    d['Country'] = name
    return d

def loco(
    country_dfs: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    beta: float = 2.0,
    top_k_per_country: int = 1
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Dict]]]:
    """
    Returns:
      - tbl: metrics for each held-out country and model
      - artifacts: dict[country][model] -> {best, threshold, feature_cols, year, y_true, y_prob, y_pred}
    """
    prepared = {k: _prepare_country(v, k) for k, v in country_dfs.items()}
    rows = []
    artifacts = {k: {} for k in prepared.keys()}

    for test_name in prepared.keys():
        te = prepared[test_name]
        tr = pd.concat([prepared[k] for k in prepared.keys() if k != test_name], ignore_index=True)

        cols = [c for c in feature_cols if c in tr.columns and c in te.columns]
        X_tr, y_tr, g_tr = tr[cols], tr['Crisis'], tr['Country']
        X_te, y_te = te[cols], te['Crisis']

        for spec in get_models():
            gkf = GroupKFold(n_splits=min(3, g_tr.nunique()))
            gcv = GridSearchCV(spec.pipeline, spec.param_grid,
                               cv=gkf.split(X_tr, y_tr, groups=g_tr),
                               scoring=ap_scorer, error_score=0.0)
            gcv.fit(X_tr, y_tr)
            best = gcv.best_estimator_

            prob_tr = _proba_pos(best.predict_proba(X_tr) if hasattr(best, "predict_proba")
                                 else best.decision_function(X_tr))
            t_star = pick_threshold_fbeta(y_tr, prob_tr, beta=beta)

            prob_te = _proba_pos(best.predict_proba(X_te) if hasattr(best, "predict_proba")
                                 else best.decision_function(X_te))
            metrics = evaluate_probs(y_te, prob_te, t_star)
            metrics.update({'test_country': test_name, 'model': spec.name, 'best_params': gcv.best_params_})
            rows.append(metrics)

            artifacts[test_name][spec.name] = dict(
                best=best, threshold=t_star, feature_cols=cols,
                year=te['Year'].values, y_true=y_te.values,
                y_prob=prob_te, y_pred=(prob_te >= t_star).astype(int)
            )

    tbl = (pd.DataFrame(rows)
             .sort_values(['test_country', 'PR_AUC', 'recall_1'], ascending=[True, False, False])
             .reset_index(drop=True))
    return tbl, artifacts

# ---------- Cross-country transfer matrix & heatmap ----------

def transfer_matrix(
    country_dfs: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    beta: float = 2.0,
    n_splits: int = 4,
    min_train_years: int = 8
) -> pd.DataFrame:
    """
    Build an A→B matrix of PR-AUC using the *best* model per pair.
    Diagonal is NaN (we only care about transfer).
    """
    countries = list(country_dfs.keys())
    M = pd.DataFrame(index=countries, columns=countries, dtype=float)

    for a in countries:
        for b in countries:
            if a == b:
                M.loc[a, b] = np.nan
                continue
            tbl, _ = train_on_A_test_on_B(
                country_dfs[a], country_dfs[b], feature_cols,
                beta=beta, n_splits=n_splits, min_train_years=min_train_years
            )
            M.loc[a, b] = float(tbl.iloc[0]['PR_AUC'])
    return M


def plot_transfer_heatmap(M: pd.DataFrame, labels: Dict[str, str], ax=None):
    """
    Pretty heatmap: rows = Train on (A), cols = Test on (B), values = PR-AUC.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    # Re-label axes with nice country names
    M_disp = M.copy()
    M_disp.index = [labels.get(i, i) for i in M.index]
    M_disp.columns = [labels.get(j, j) for j in M.columns]

    sns.heatmap(
        M_disp, vmin=0, vmax=1, cmap="YlGnBu", annot=True, fmt=".2f", ax=ax,
        cbar_kws={'label': 'PR-AUC'}
    )
    ax.set_xlabel("Test on (B)")
    ax.set_ylabel("Train on (A)")
    ax.set_title("Cross-country transfer performance")
    return ax

# Base macro variables used by the VAR
BASE_VARS = ['GDP','CPI','HH Consumption','Capital Formation']

def fit_var(df_base: pd.DataFrame, maxlags: int = 3):
    """
    df_base must be indexed by Year and contain BASE_VARS columns.
    Returns fitted VAR model and lag order.
    """
    model = VAR(df_base)
    try:
        sel = model.select_order(maxlags=maxlags)
        candidates = [sel.aic, sel.bic, sel.fpe, sel.hqic]
        p = min([int(c) for c in candidates if np.isfinite(c)] + [1])
    except Exception:
        p = 1
    res = model.fit(maxlags=p)
    return res, res.k_ar

def _nearest_psd(Sigma):
    """Project covariance to PSD if needed."""
    try:
        np.linalg.cholesky(Sigma)
        return Sigma
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(Sigma)
        vals_clipped = np.clip(vals, 1e-10, None)
        return (vecs @ np.diag(vals_clipped) @ vecs.T)

def simulate_var_paths(res, steps: int, sims: int, seed: int = 0):
    """
    Monte Carlo simulate a VAR(p). Output: (sims, steps, k)
    """
    rng = np.random.default_rng(seed)
    p = res.k_ar
    k = res.neqs
    A = res.coefs            # (p, k, k)
    c = res.intercept        # (k,)
    Sigma = _nearest_psd(res.sigma_u + 1e-8*np.eye(k))

    y_init = res.endog[-p:, :]   # (p, k)
    paths = np.zeros((sims, steps, k))
    for s in range(sims):
        y_hist = y_init.copy()
        for t in range(steps):
            y_mean = c.copy()
            for lag in range(1, p+1):
                y_mean += A[lag-1] @ y_hist[-lag]
            eps = rng.multivariate_normal(mean=np.zeros(k), cov=Sigma)
            y_t = y_mean + eps
            paths[s, t, :] = y_t
            y_hist = np.vstack([y_hist, y_t])[-p:, :]
    return paths

def apply_stress(paths, var_names, scenario='baseline'):
    """
    paths: (sims, steps, k); scenario: 'baseline'|'mild'|'severe'
    Shocks are additive in the variable's native units.
    """
    out = paths.copy()
    if scenario == 'baseline':
        return out
    idx = {v:i for i,v in enumerate(var_names)}

    if scenario == 'mild':
        out[:,0,idx['GDP']] -= 2.0
        if out.shape[1] > 1: out[:,1,idx['GDP']] -= 1.0
        out[:,0,idx['CPI']] += 2.0
        if out.shape[1] > 1: out[:,1,idx['CPI']] += 1.0

    elif scenario == 'severe':
        out[:,0,idx['GDP']] -= 5.0
        if out.shape[1] > 1: out[:,1,idx['GDP']] -= 5.0
        if out.shape[1] > 2: out[:,2,idx['GDP']] -= 2.0
        out[:,0,idx['CPI']] += 5.0
        if out.shape[1] > 1: out[:,1,idx['CPI']] += 5.0
        if out.shape[1] > 2: out[:,2,idx['CPI']] += 3.0

    return out

def probs_from_simulations(country_code: str,
                           hist_df: pd.DataFrame,
                           best_entry: dict,
                           steps: int = 5,
                           sims: int = 500,
                           scenarios=('baseline','mild','severe'),
                           seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Use a VAR on BASE_VARS to simulate macro paths and score model probabilities.
    best_entry must contain: {'art': { 'best': fitted_pipeline, 'feature_cols': [...] }}
    Returns: {scenario: DataFrame[Year, mean, p05, p50, p95]}
    """
    d = hist_df.sort_values('Year').copy()
    base = d[['Year'] + BASE_VARS].dropna().set_index('Year')
    res, _ = fit_var(base[BASE_VARS])

    paths = simulate_var_paths(res, steps=steps, sims=sims, seed=seed)
    var_names = BASE_VARS[:]

    def score_scenario(sim_paths):
        years_last = int(base.index.max())
        future_years = np.arange(years_last+1, years_last+1+steps)
        P = np.zeros((sims, steps))

        model = best_entry['art']['best']
        feat_cols = best_entry['art']['feature_cols']
        hist_block = d[['Year'] + BASE_VARS].copy()

        for i in range(sims):
            fut_block = pd.DataFrame(sim_paths[i], columns=var_names)
            fut_block.insert(0, 'Year', future_years)
            tmp = pd.concat([hist_block, fut_block], ignore_index=True)
            feats = make_features(tmp)
            X_fut = feats.iloc[-steps:][feat_cols].copy()

            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X_fut)[:,1]
            else:
                raw = model.decision_function(X_fut)
                p = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
            P[i, :] = p

        return pd.DataFrame({
            'Year': future_years,
            'mean': P.mean(axis=0),
            'p05' : np.quantile(P, 0.05, axis=0),
            'p50' : np.quantile(P, 0.50, axis=0),
            'p95' : np.quantile(P, 0.95, axis=0),
        })

    results = {}
    for sc in scenarios:
        sim_sc = apply_stress(paths, var_names, scenario=sc)
        results[sc] = score_scenario(sim_sc)
    return results

def plot_fan_chart(country_code: str, results: Dict[str, pd.DataFrame], scenario: str, ax=None):
    """
    Plot a single scenario's fan chart (p05–p95 band, mean, median), % y-axis, with peak label.
    """
    df = results[scenario]
    y = df['Year'].values
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))

    # 5–95% band
    ax.fill_between(y, df['p05'], df['p95'], alpha=0.18, label='5–95% range')

    # Lines
    ax.plot(y, df['mean'], linewidth=2.0, label='Mean')
    ax.plot(y, df['p50'], linestyle='--', linewidth=1.5, label='Median')

    # Percent axis and tidy grid
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', alpha=0.25)

    # Title/labels
    ax.set_title(f'{country_code}: {scenario.capitalize()} scenario — crisis probability')
    ax.set_ylabel('Probability'); ax.set_xlabel('Year')
    ax.legend(loc='upper left')

    # Peak label on the mean
    i_peak = int(np.argmax(df['mean'].values))
    ax.scatter([y[i_peak]], [df['mean'].iloc[i_peak]], zorder=3)
    ax.annotate(
        f"peak: {int(y[i_peak])}\n{df['mean'].iloc[i_peak]*100:.0f}%",
        xy=(y[i_peak], df['mean'].iloc[i_peak]),
        xytext=(10, 10), textcoords='offset points',
        fontsize=9, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.6', alpha=0.9)
    )
    return ax

def forecast_table(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Tidy table across scenarios."""
    tbl = []
    for sc, df in results.items():
        t = df.copy()
        t['Scenario'] = sc
        tbl.append(t[['Year','Scenario','mean','p05','p50','p95']])
    return pd.concat(tbl, ignore_index=True)

def summarize_forecast(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    For each scenario: peak year & level, average over horizon, and count of years above 40%.
    """
    rows = []
    for sc, df in results.items():
        mean = df['mean'].values
        years = df['Year'].astype(int).values
        i_peak = int(np.argmax(mean))
        rows.append({
            'Scenario': sc,
            'Peak year': int(years[i_peak]),
            'Peak prob (%)': round(mean[i_peak]*100, 1),
            'Avg over horizon (%)': round(mean.mean()*100, 1),
            'Years ≥ 40%': int((mean >= 0.40).sum())
        })
    return pd.DataFrame(rows).sort_values('Scenario').reset_index(drop=True)

def prettify_forecast_table(tidy: pd.DataFrame) -> pd.DataFrame.style:
    """
    Clean “Numbers behind the charts” table: percent formatting + a compact range column.
    """
    t = tidy.copy()
    # Build a “range” text column for readability
    t['Range (p05–p95)'] = (t['p05']*100).round(0).astype(int).astype(str) + '–' + \
                           (t['p95']*100).round(0).astype(int).astype(str) + '%'
    t['Mean (%)'] = (t['mean']*100).round(1)
    t['Median (%)'] = (t['p50']*100).round(1)

    # Keep a friendly column order
    t = t[['Year', 'Scenario', 'Mean (%)', 'Median (%)', 'Range (p05–p95)']]

    # Style: bold the per-scenario peak mean
    def _bold_peaks(df):
        out = pd.DataFrame('', index=df.index, columns=df.columns)
        for sc, grp in df.groupby('Scenario'):
            if 'Mean (%)' in grp:
                idx = grp['Mean (%)'].idxmax()
                out.loc[idx, 'Mean (%)'] = 'font-weight: 700;'
        return out

    return (t.style
            .apply(_bold_peaks, axis=None)
            .format({'Mean (%)':'{:.1f}', 'Median (%)':'{:.1f}'})
           )
