import numpy as np
import pandas as pd
import pytest

from gcdiff.data import MovieLensTables

OCC = ["student", "engineer", "artist", "other"]


def make_fake_tables(n_users=80, n_items=60, seed=0) -> MovieLensTables:
    """Tiny MovieLens-shaped fixture (no download)."""
    rng = np.random.default_rng(seed)
    genres = [f"g{i}" for i in range(5)]
    users = pd.DataFrame({
        "age": rng.integers(15, 60, n_users).astype(float),
        "gender": rng.choice(["F", "M"], n_users),
        "occupation": rng.choice(OCC, n_users),
        "zip_region": rng.choice([str(d) for d in range(10)] + ["other"], n_users),
    })
    items = pd.DataFrame(rng.integers(0, 2, (n_items, len(genres))).astype(float), columns=genres)
    items["release_year"] = rng.integers(1980, 1998, n_items).astype(float)
    rows = []
    for u in range(n_users):
        k = rng.integers(20, 30)
        for i in rng.choice(n_items, k, replace=False):
            rows.append((u, i, int(rng.integers(1, 6))))
    ratings = pd.DataFrame(rows, columns=["user", "item", "rating"])
    return MovieLensTables(users, items, ratings, genres, OCC)


@pytest.fixture
def tables():
    return make_fake_tables()
