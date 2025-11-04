# Variance explained & partitioning in Python (ready-to-run)
# - Works for linear regression with intercept
# - Computes:
#   * In-sample R^2 and r^2 (they match for OLS with intercept)
#   * Cross-validated R^2 (optional, for out-of-sample)
#   * Unique variance per feature (semi-partial R^2): R^2_full - R^2_without_i
#   * Shapley/LMG contributions (variance partitioning across all subsets) — sums to total R^2
#
# Demo at the bottom on synthetic, correlated features.
#
# Notes:
# - For many features (p > ~12), exact Shapley gets expensive (2^p subsets).
#   You can switch to a Monte Carlo approximation by setting `approx_permutations`>0.

from __future__ import annotations
import itertools
from math import comb
import numpy as np
import pandas as pd
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score
from typing import Dict, Tuple, Optional

def _ensure_array(X, y):
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X = X.to_numpy()
    else:
        feature_names = [f"x{i}" for i in range(X.shape[1])]
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = np.asarray(y).reshape(-1)
    else:
        y = np.asarray(y).reshape(-1)
    return X, y, feature_names

def fit_ols_r2(X: np.ndarray, y: np.ndarray) -> Tuple[LinearRegression, float, np.ndarray]:
    """Fit OLS with intercept and return model, in-sample R^2, predictions."""
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X, y)
    yhat = lr.predict(X)
    r2 = r2_score(y, yhat)
    return lr, r2, yhat

def cv_r2(X: np.ndarray, y: np.ndarray, n_splits: int = 5, random_state: int = 0) -> float:
    """Return mean cross-validated R^2 (out-of-sample)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    lr = LinearRegression(fit_intercept=True)
    scores = cross_val_score(lr, X, y, scoring="r2", cv=kf)
    return float(np.mean(scores))

def subset_r2_cache(X: np.ndarray, y: np.ndarray) -> Dict[int, float]:
    """
    Compute R^2 for every subset of features using bitmasks.
    Returns a dict: mask -> R^2
    """
    n, p = X.shape
    cache: Dict[int, float] = {}
    # Empty set has R^2 = 0 (intercept-only)
    cache[0] = 0.0
    # Precompute to speed up re-use of columns
    cols = list(range(p))
    for k in range(1, p + 1):
        for subset in itertools.combinations(cols, k):
            mask = sum(1 << j for j in subset)
            Xi = X[:, subset]
            _, r2, _ = fit_ols_r2(Xi, y)
            cache[mask] = r2
    return cache

def unique_r2(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Unique (semi-partial) R^2 for each feature: R^2_full - R^2_without_i.
    Sums to <= total R^2 (remainder is shared variance).
    """
    _, r2_full, _ = fit_ols_r2(X, y)
    p = X.shape[1]
    uniques = np.zeros(p)
    for i in range(p):
        mask_without_i = [j for j in range(p) if j != i]
        Xi = X[:, mask_without_i] if len(mask_without_i) > 0 else np.zeros((X.shape[0], 0))
        _, r2_red, _ = fit_ols_r2(Xi, y) if Xi.shape[1] > 0 else (None, 0.0, None)
        uniques[i] = r2_full - r2_red
    return uniques

def shapley_lmg(
    X: np.ndarray,
    y: np.ndarray,
    approx_permutations: int = 0,
    random_state: Optional[int] = 0
) -> np.ndarray:
    """
    Exact LMG/Shapley for linear R^2 (Averaged incremental R^2 over all orderings).
    For p <= ~12, exact computation via subset cache is feasible.
    For larger p, set approx_permutations > 0 to use random permutations.
    """
    n, p = X.shape
    if approx_permutations and approx_permutations > 0:
        rng = np.random.RandomState(random_state)
        totals = np.zeros(p)
        for _ in range(approx_permutations):
            order = rng.permutation(p)
            included = []
            r2_prev = 0.0
            for j in order:
                Xi = X[:, included + [j]] if included else X[:, [j]]
                _, r2_curr, _ = fit_ols_r2(Xi, y)
                totals[j] += (r2_curr - r2_prev)
                included = included + [j]
                r2_prev = r2_curr
        return totals / approx_permutations

    # Exact
    cache = subset_r2_cache(X, y)
    contributions = np.zeros(p)
    full_mask = (1 << p) - 1
    for j in range(p):
        contrib = 0.0
        for S_mask in range(full_mask + 1):
            if S_mask & (1 << j):  # S already contains j -> skip (we want S not containing j)
                continue
            size_S = bin(S_mask).count("1")
            if size_S == p - 1:
                # Adding j would make full set; weight still works
                pass
            # R^2(S ∪ {j}) - R^2(S)
            r2_S = cache[S_mask]
            r2_Sj = cache[S_mask | (1 << j)]
            marginal = r2_Sj - r2_S
            weight = (scipy.special.factorial(size_S) * scipy.special.factorial(p - size_S - 1)) / scipy.special.factorial(p)
            contrib += weight * marginal
        contributions[j] = contrib
    return contributions

def variance_partition(
    X,
    y,
    approx_permutations: int = 0,
    random_state: int = 0
) -> Dict[str, object]:
    """
    Main helper: returns total R^2, r^2(Y, Yhat), CV R^2, unique (semi-partial) R^2,
    and Shapley/LMG partition (sums to total R^2).
    """
    X, y, feature_names = _ensure_array(X, y)
    model, r2_in, yhat = fit_ols_r2(X, y)
    # For OLS with intercept, r^2(Y, Yhat) == R^2_in
    r = np.corrcoef(y, yhat)[0, 1]
    r2_corr = r**2
    cv = cv_r2(X, y, n_splits=min(5, len(y)))
    uniq = unique_r2(X, y)
    lmg = shapley_lmg(X, y, approx_permutations=approx_permutations, random_state=random_state)
    shared = r2_in - uniq.sum()
    df = pd.DataFrame({
        "feature": feature_names,
        "unique_R2": uniq,
        "LMG_Shapley": lmg
    }).sort_values("LMG_Shapley", ascending=False).reset_index(drop=True)

    summary = {
        "R2_in_sample": r2_in,
        "r2_correlation": r2_corr,
        "R2_cv_mean": cv,
        "Unique_sum": float(uniq.sum()),
        "Shared_R2": float(shared),
        "features": df
    }
    return summary

# -------------------- DEMO --------------------
# Synthetic example: predict a voxel's response from 5 correlated image features
rng = np.random.RandomState(42)
n = 300
p = 5
# Correlated features
A = rng.normal(size=(p, p))
Sigma = A @ A.T
X_raw = rng.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)
# True weights
beta = np.array([0.8, 0.5, 0.0, 0.0, 0.2])
y_true = X_raw @ beta + rng.normal(scale=1.0, size=n)
X_df = pd.DataFrame(X_raw, columns=[f"feat_{i}" for i in range(p)])

results = variance_partition(X_df, y_true, approx_permutations=0, random_state=0)
print(results['features'])

metrics = pd.DataFrame({
    "metric": ["R2_in_sample", "r2_correlation (Y,Yhat)", "R2_cv_mean", "Sum(unique_R2)", "Shared_R2"],
    "value": [results["R2_in_sample"], results["r2_correlation"], results["R2_cv_mean"], results["Unique_sum"], results["Shared_R2"]]
})
print(metrics)

print("Done. You can now inspect 'Variance partition per feature' and 'Key metrics' tables above.")
