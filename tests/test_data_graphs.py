import numpy as np
import pytest

from conftest import make_fake_tables
from gcdiff.data import TableCodec, make_split
from gcdiff.graphs import (EDGE_TYPES, behaviour_features, build_hetero_graph, knn_user_graph,
                           shuffle_item_endpoints, standardise_users)


def test_codec_roundtrip_and_train_only_statistics(tables):
    users = tables.users
    train = users.iloc[:40]
    c = TableCodec.fit(train, tables.vocab)
    xn, xc = c.encode(users)
    assert xn.shape == (80, 1) and xc.shape == (80, 3)
    assert abs(xn[:40].mean()) < 1e-5 and abs(xn[:40].std() - 1) < 1e-3     # standardised on train
    assert abs(xn[40:].mean()) > 1e-4                                       # ... not on everything
    dec = c.decode(*c.encode(train))
    assert (dec["age"].to_numpy() == train["age"].to_numpy()).all()
    assert (dec["occupation"].to_numpy() == train["occupation"].to_numpy()).all()
    assert c.cat_sizes == [2, 4, 11]


def test_decode_clips_age_to_train_range(tables):
    c = TableCodec.fit(tables.users, tables.vocab)
    d = c.decode(np.array([[50.0], [-50.0]], np.float32), np.zeros((2, 3), np.int64))
    assert d["age"].tolist() == [tables.users["age"].max(), tables.users["age"].min()]


def test_split_is_disjoint_complete_stratified_and_seeded(tables):
    y = (tables.users["gender"] == "M").to_numpy()
    a, b = make_split(y, (0.6, 0.15, 0.25), 3), make_split(y, (0.6, 0.15, 0.25), 3)
    allidx = np.concatenate([a.train, a.val, a.test])
    assert sorted(allidx) == list(range(80)) and len(set(allidx)) == 80
    assert (a.train == b.train).all() and (a.test == b.test).all()
    assert not np.array_equal(a.test, make_split(y, (0.6, 0.15, 0.25), 4).test)
    assert abs(y[a.test].mean() - y.mean()) < 0.1
    with pytest.raises(ValueError):
        make_split(y, (0.6, 0.15, 0.2), 3)


def test_graph_relations_partition_ratings(tables):
    raw = behaviour_features(tables.ratings, tables.items, tables.genres, tables.n_users)
    g = build_hetero_graph(tables.ratings, raw.user_struct, raw.item_x)
    n_like = int((tables.ratings.rating >= 4).sum())
    assert g[EDGE_TYPES[0]].edge_index.shape[1] == n_like
    assert g[EDGE_TYPES[2]].edge_index.shape[1] == len(tables.ratings) - n_like
    assert (g[EDGE_TYPES[1]].edge_index == g[EDGE_TYPES[0]].edge_index.flip(0)).all()
    assert raw.flat.shape == (80, 2 * 5 + 2 + 1) and g["item"].x.shape == (60, 6)


def test_no_target_leakage_into_graph_or_features():
    """Two worlds with IDENTICAL ratings/items but different user attributes -> identical conditioning inputs."""
    a, b = make_fake_tables(), make_fake_tables()
    b.users = b.users.sample(frac=1, random_state=1).reset_index(drop=True)
    b.users["gender"] = np.where(b.users["gender"] == "M", "F", "M")
    fa = behaviour_features(a.ratings, a.items, a.genres, a.n_users)
    fb = behaviour_features(b.ratings, b.items, b.genres, b.n_users)
    assert np.array_equal(fa.flat, fb.flat) and np.array_equal(fa.user_struct, fb.user_struct)
    idx = np.arange(40)
    assert np.array_equal(standardise_users(fa, idx).flat, standardise_users(fb, idx).flat)


def test_standardise_uses_train_users_only(tables):
    raw = behaviour_features(tables.ratings, tables.items, tables.genres, tables.n_users)
    z = standardise_users(raw, np.arange(30))
    assert np.allclose(z.flat[:30].mean(0), 0, atol=1e-4)


def test_shuffle_preserves_user_degree_and_rating_multiset(tables):
    sh = shuffle_item_endpoints(tables.ratings, 0)
    assert (sh.groupby("user").size() == tables.ratings.groupby("user").size()).all()
    assert sh.groupby("user").rating.value_counts().sort_index().equals(
        tables.ratings.groupby("user").rating.value_counts().sort_index())
    assert sh.groupby("item").size().sort_index().equals(
        tables.ratings.groupby("item").size().sort_index())
    assert not (sh["item"].to_numpy() == tables.ratings["item"].to_numpy()).all()
    assert (shuffle_item_endpoints(tables.ratings, 0)["item"].to_numpy() == sh["item"].to_numpy()).all()


def test_knn_graph_is_symmetric_without_self_loops():
    x = np.random.default_rng(0).normal(size=(40, 6)).astype(np.float32)
    g = knn_user_graph(x, 5)
    ei = g.edge_index
    assert (ei[0] != ei[1]).all()
    fwd = set(map(tuple, ei.t().tolist()))
    assert all((j, i) in fwd for i, j in fwd)
