import numpy as np
import pandas as pd

from gcdiff import baselines, metrics


def _df(n=400, seed=0, dependent=True):
    rng = np.random.default_rng(seed)
    occ = rng.choice(["a", "b", "c"], n)
    if dependent:
        age = np.where(occ == "a", 20, np.where(occ == "b", 35, 50)) + rng.normal(0, 3, n)
    else:
        age = rng.normal(35, 10, n)
    return pd.DataFrame({"age": np.rint(age), "gender": rng.choice(["F", "M"], n), "occupation": occ,
                         "zip_region": rng.choice(list("0123"), n)})


def test_marginal_identical_is_zero_and_shifted_is_positive():
    d = _df()
    assert metrics.marginal_fidelity(d, d)["marginal_score"] == 0
    e = d.copy()
    e["age"] += 100
    assert metrics.marginal_fidelity(d, e)["ks_age"] > 0.5


def test_association_detects_broken_dependence():
    d = _df()
    assert metrics.association_error(d, d) == 0
    ind = baselines.marginal_bootstrap(d, len(d), np.random.default_rng(1))
    assert metrics.association_error(d, ind) > 0.1                    # age-occupation link destroyed
    assert metrics.association_matrix(d).loc["age", "occupation"] > 0.9


def test_cramers_v_bias_correction_near_zero_for_independent_columns():
    d = _df(300, dependent=False)
    assert metrics._cramers_v(d["occupation"], d["zip_region"]) < 0.1


def test_memorisation_check_has_power():
    tr, te = _df(300, 0), _df(100, 1)
    copy = baselines.row_bootstrap(tr, 100, np.random.default_rng(0))
    m = metrics.memorisation_check(copy, tr, te)
    assert m["dcr_ratio"] == 0 and m["exact_match"] == 1
    fresh = metrics.memorisation_check(_df(100, 2), tr, te)
    assert 0.6 < fresh["dcr_ratio"] < 1.4                            # new draws from the same population ~ 1


def test_conditional_consistency_chance_and_signal():
    real = _df(120, 0)
    S, n = 6, len(real)
    indep = pd.concat([_df(n, 10 + s) for s in range(S)], ignore_index=True)
    r = metrics.conditional_consistency(indep, S, real, real["age"].mean())
    assert 0.35 < r["cond_gender_auc"] < 0.65 and r["cond_age_r2"] < 0.1
    oracle = pd.concat([real] * S, ignore_index=True)                  # copies of the truth
    r = metrics.conditional_consistency(oracle, S, real, real["age"].mean())
    assert r["cond_gender_auc"] == 1 and r["cond_occ_acc"] == 1 and r["cond_age_r2"] > 0.99


def test_conditional_consistency_layout_is_s_major():
    real = _df(10, 0)
    S = 3
    samples = pd.concat([real.assign(age=real["age"] + s) for s in range(S)], ignore_index=True)
    r = metrics.conditional_consistency(samples, S, real, real["age"].mean())
    # draw-mean age of user i is real age + 1 (mean of 0, 1, 2) only if row s*n + i is draw s of user i
    age_hat = samples["age"].to_numpy().reshape(S, 10).mean(0)
    assert np.allclose(age_hat, real["age"].to_numpy() + 1) and r["cond_gender_auc"] == 1


def test_node_probe_recovers_planted_signal_and_rejects_noise():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 5))
    y = pd.DataFrame({"gender": np.where(X[:, 0] + 0.3 * rng.normal(size=600) > 0, "M", "F"), "age": 30 + 5 * X[:, 1]})
    test = y.iloc[400:].reset_index(drop=True)
    m = metrics.node_probe_metrics(X[:400], y.iloc[:400], X[400:], test, 30.0)
    assert m["gender_auc"] > 0.9 and m["age_r2"] > 0.8
    noise = y.iloc[:400].sample(frac=1, random_state=0).reset_index(drop=True)
    m = metrics.node_probe_metrics(X[:400], noise, X[400:], test, 30.0)
    assert m["gender_auc"] < 0.7 and m["age_r2"] < 0.1


def test_tstr_table_uses_explicit_training_age_scale():
    train, test = _df(300, 0), _df(120, 1)
    vocab = {"occupation": ["a", "b", "c"], "zip_region": list("0123")}
    mu, sd = float(train.age.mean()), float(train.age.std())
    auc = metrics.tstr_table_auc(train, test, vocab, mu, sd)
    assert 0 <= auc <= 1


def test_homophily_and_rating_knn(tables):
    users = np.arange(30)
    nbrs = metrics.rating_knn(tables.ratings, users, tables.n_items, 5)
    assert nbrs.shape == (30, 5) and (nbrs != np.arange(30)[:, None]).all()
    h = metrics.homophily(tables.users.iloc[:30].reset_index(drop=True), nbrs)
    assert 0 <= h["same_gender_rate"] <= 1
