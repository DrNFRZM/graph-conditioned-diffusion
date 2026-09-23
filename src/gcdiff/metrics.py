"""Evaluation metrics. Each function's docstring says what it does and does NOT show.

Real data are the held-out TEST users; synthetic data never see test attributes.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

NUM_COLS = ["age"]
CAT_COLS = ["gender", "occupation", "zip_region"]
ALL_COLS = NUM_COLS + CAT_COLS


# ------------------------------------------------------------------ marginal fidelity
def marginal_fidelity(real: pd.DataFrame, syn: pd.DataFrame) -> dict[str, float]:
    """Kolmogorov-Smirnov statistic (numeric) and total-variation distance (categorical), both in [0, 1].
    Shows: one-column-at-a-time distribution match. Does NOT show: any dependence between columns."""
    out = {}
    for c in NUM_COLS:
        out[f"ks_{c}"] = float(ks_2samp(real[c], syn[c]).statistic)
    for c in CAT_COLS:
        p = real[c].value_counts(normalize=True)
        q = syn[c].value_counts(normalize=True)
        cats = p.index.union(q.index)
        out[f"tv_{c}"] = float(0.5 * np.abs(p.reindex(cats, fill_value=0) - q.reindex(cats, fill_value=0)).sum())
    out["marginal_score"] = float(np.mean([out[k] for k in out]))
    return out


# ------------------------------------------------------------------ relationship preservation
def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Bias-corrected Cramer's V (Bergsma 2013); the correction matters because n differs between real and synthetic."""
    tab = pd.crosstab(a, b).to_numpy(float)
    n = tab.sum()
    exp = tab.sum(1, keepdims=True) @ tab.sum(0, keepdims=True) / n
    chi2 = ((tab - exp) ** 2 / np.where(exp > 0, exp, 1)).sum()
    r, k = tab.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_c, k_c = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    d = min(k_c - 1, r_c - 1)
    return float(np.sqrt(phi2 / d)) if d > 0 else 0.0


def _corr_ratio(cat: pd.Series, num: pd.Series) -> float:
    """Correlation ratio eta in [0, 1]: share of numeric variance explained by the category."""
    x = num.to_numpy(float)
    tot = ((x - x.mean()) ** 2).sum()
    if tot == 0:
        return 0.0
    between = sum(len(g) * (g.mean() - x.mean()) ** 2 for g in (num[cat == v].to_numpy(float) for v in cat.unique()))
    return float(np.sqrt(between / tot))


def association_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise association in [0, 1]: |Pearson| (num-num), eta (cat-num), Cramer's V (cat-cat)."""
    m = pd.DataFrame(np.eye(len(ALL_COLS)), index=ALL_COLS, columns=ALL_COLS)
    for a, b in itertools.combinations(ALL_COLS, 2):
        if a in NUM_COLS and b in NUM_COLS:
            v = abs(np.corrcoef(df[a], df[b])[0, 1])
        elif a in NUM_COLS or b in NUM_COLS:
            num, cat = (a, b) if a in NUM_COLS else (b, a)
            v = _corr_ratio(df[cat], df[num])
        else:
            v = _cramers_v(df[a], df[b])
        m.loc[a, b] = m.loc[b, a] = v
    return m


def association_error(real: pd.DataFrame, syn: pd.DataFrame) -> float:
    """Mean absolute difference over the off-diagonal association entries. Lower is better.
    Shows: pairwise dependence between columns is preserved. Does NOT show: higher-order structure,
    or dependence on the graph (this compares attribute columns to each other)."""
    d = (association_matrix(real) - association_matrix(syn)).to_numpy()
    return float(np.abs(d[np.triu_indices(len(ALL_COLS), 1)]).mean())


# ------------------------------------------------------------------ downstream utility (attribute table)
def _table_features(df: pd.DataFrame, ref_cols: dict[str, list[str]], age_mu: float, age_sd: float) -> np.ndarray:
    parts = [((df["age"].to_numpy(float) - age_mu) / age_sd)[:, None]]
    for c in ("occupation", "zip_region"):
        parts.append(np.stack([(df[c] == v).to_numpy(float) for v in ref_cols[c]], 1))
    return np.concatenate(parts, 1)


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else 0.5


def tstr_table_auc(train_df: pd.DataFrame, real_test: pd.DataFrame, vocab: dict[str, list[str]],
                   age_mean: float, age_std: float) -> float:
    """Train-on-X, test-on-real: logistic regression predicting gender from (age, occupation, zip_region).
    Age scaling is supplied from the real TRAIN partition, never the test set. Shows: whether
    attribute-to-attribute structure in X transfers to real users."""
    y = (train_df["gender"] == "M").to_numpy()
    if len(np.unique(y)) < 2:
        return 0.5
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(
        _table_features(train_df, vocab, age_mean, age_std), y)
    xt = _table_features(real_test, vocab, age_mean, age_std)
    return _safe_auc((real_test["gender"] == "M").to_numpy(), clf.predict_proba(xt)[:, 1])


# ------------------------------------------------------------------ node-level (conditional) metrics
def conditional_consistency(samples: pd.DataFrame, S: int, real_test: pd.DataFrame, train_age_mean: float) -> dict[str, float]:
    """`samples` holds S draws per test user (s-major order: row s*n + i is draw s for user i).
    Shows: whether draws conditioned on user i's graph carry information about user i's REAL attributes
    (the generator never saw them). An unconditional generator must score at chance
    (AUC 0.5, R2 ~ 0, majority-class accuracy). Does NOT show: marginal fidelity, or that the
    dependence is 'causal' - only that the graph signal is predictive."""
    n = len(real_test)
    p_male = (samples["gender"].to_numpy() == "M").reshape(S, n).mean(0)
    age_hat = samples["age"].to_numpy(float).reshape(S, n).mean(0)
    age = real_test["age"].to_numpy(float)
    occ_mode = pd.DataFrame(samples["occupation"].to_numpy().reshape(S, n)).mode(axis=0).iloc[0].to_numpy()
    return {
        "cond_gender_auc": _safe_auc((real_test["gender"] == "M").to_numpy(), p_male),
        "cond_age_r2": float(1 - ((age - age_hat) ** 2).sum() / ((age - train_age_mean) ** 2).sum()),
        "cond_occ_acc": float((occ_mode == real_test["occupation"].to_numpy()).mean()),
    }


def node_probe_metrics(flat_train, y_train: pd.DataFrame, flat_test, real_test: pd.DataFrame,
                       train_age_mean: float) -> dict[str, float]:
    """Predict test users' attributes from flat behaviour features with models fit on `y_train` labels.
    With REAL train labels this is a non-generative discriminative reference. With SYNTHETIC train labels
    it is 'train on synthetic, test on real' for a node-level task: it shows whether the synthetic labels
    contain graph-dependent signal that transfers to real users."""
    yg = (y_train["gender"] == "M").to_numpy()
    if len(np.unique(yg)) < 2:
        auc = 0.5
    else:
        clf = LogisticRegression(C=0.1, max_iter=3000).fit(flat_train, yg)
        auc = _safe_auc((real_test["gender"] == "M").to_numpy(), clf.predict_proba(flat_test)[:, 1])
    reg = Ridge(alpha=100.0).fit(flat_train, y_train["age"].to_numpy(float))
    age = real_test["age"].to_numpy(float)
    r2 = 1 - ((age - reg.predict(flat_test)) ** 2).sum() / ((age - train_age_mean) ** 2).sum()
    return {"gender_auc": auc, "age_r2": float(r2)}


def rating_knn(ratings: pd.DataFrame, users: np.ndarray, n_items: int, k: int) -> np.ndarray:
    """kNN among `users` from their mean-centred rating vectors (cosine). Used only for evaluation."""
    pos = {u: j for j, u in enumerate(users)}
    R = np.zeros((len(users), n_items), np.float32)
    sel = ratings[ratings["user"].isin(pos)]
    R[sel["user"].map(pos).to_numpy(), sel["item"].to_numpy()] = sel["rating"].to_numpy()
    rated = R > 0
    R = np.where(rated, R - (R.sum(1) / np.maximum(rated.sum(1), 1))[:, None], 0)
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-8)
    sim = Rn @ Rn.T
    np.fill_diagonal(sim, -np.inf)
    return np.argpartition(-sim, k, axis=1)[:, :k]


def homophily(attrs: pd.DataFrame, nbrs: np.ndarray) -> dict[str, float]:
    """Attribute homophily over the evaluation kNN graph among test users: same-gender edge rate and
    age assortativity (Pearson over edge endpoints). Compare synthetic (draw 0 assigned to each user) with real.
    Shows: whether synthetic attributes are arranged over the graph like real ones. Note that it can only
    exceed chance for a generator that conditions on graph information."""
    src = np.repeat(np.arange(len(attrs)), nbrs.shape[1])
    dst = nbrs.reshape(-1)
    g = attrs["gender"].to_numpy()
    a = attrs["age"].to_numpy(float)
    return {"same_gender_rate": float((g[src] == g[dst]).mean()),
            "age_assort": float(np.corrcoef(a[src], a[dst])[0, 1])}


# ------------------------------------------------------------------ memorisation sanity check
def _gower(a: pd.DataFrame, b: pd.DataFrame, age_range: float) -> np.ndarray:
    """Mean per-column distance: |dage|/range for numeric, 0/1 mismatch for categorical. Shape [len(a), len(b)]."""
    d = np.abs(a["age"].to_numpy(float)[:, None] - b["age"].to_numpy(float)[None, :]) / age_range
    for c in CAT_COLS:
        d = d + (a[c].to_numpy()[:, None] != b[c].to_numpy()[None, :])
    return d / len(ALL_COLS)


def memorisation_check(syn: pd.DataFrame, real_train: pd.DataFrame, real_test: pd.DataFrame) -> dict[str, float]:
    """Distance-to-closest-record (DCR) sanity check against the FULL train set.
    dcr_ratio = mean DCR(synthetic -> train) / mean DCR(held-out real users -> train).
    ~1 means synthetic rows are as far from the training rows as genuinely new real users are;
    << 1 means they sit closer to training rows than new real users do (copying). exact_match is the share of
    synthetic rows identical to some train row; compare with `exact_match_real_test`.
    Shows: absence of gross copying. Does NOT show privacy: no membership-inference attack, no
    differential-privacy guarantee, and low-cardinality columns make coincidental matches likely."""
    age_range = float(real_train["age"].max() - real_train["age"].min())
    d_syn = _gower(syn, real_train, age_range).min(1)
    d_te = _gower(real_test, real_train, age_range).min(1)
    return {"dcr_ratio": float(d_syn.mean() / d_te.mean()), "exact_match": float((d_syn == 0).mean()),
            "exact_match_real_test": float((d_te == 0).mean())}
