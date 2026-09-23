"""MovieLens 100K: download, tables, mixed-type codec, node splits.

Target table (one row per user, "heterogeneous" = numeric + categorical):
    age (numeric), gender (2), occupation (21), zip_region (11)
The rating graph is built separately (graphs.py) and never sees these columns.
"""
from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Source: GroupLens Research, https://grouplens.org/datasets/movielens/100k/
ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML100K_SHA256 = "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"
ML100K_REQUIRED = ("u.data", "u.user", "u.item", "u.genre", "u.occupation")

NUM_COLS = ["age"]
CAT_COLS = ["gender", "occupation", "zip_region"]
GENDERS = ["F", "M"]
ZIP_REGIONS = [str(d) for d in range(10)] + ["other"]  # first digit of US zip; letters (Canadian codes) -> other


@dataclass
class MovieLensTables:
    users: pd.DataFrame    # index = user index 0..n-1; columns NUM_COLS + CAT_COLS
    items: pd.DataFrame    # index = item index; genre columns + release_year
    ratings: pd.DataFrame  # columns user, item, rating (0-based indices)
    genres: list[str]
    occupations: list[str]

    @property
    def n_users(self) -> int:
        return len(self.users)

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def vocab(self) -> dict[str, list[str]]:
        # schema-level category lists (no frequencies), so unseen-in-train categories still encode
        return {"gender": GENDERS, "occupation": self.occupations, "zip_region": ZIP_REGIONS}


def download_ml100k(root: str | Path = "data") -> Path:
    """Download and verify ml-100k.zip (~5 MB, no credentials). Returns the extracted directory."""
    root = Path(root)
    target = root / "ml-100k"
    if all((target / name).is_file() for name in ML100K_REQUIRED):
        return target
    root.mkdir(parents=True, exist_ok=True)
    zpath = root / "ml-100k.zip"
    urllib.request.urlretrieve(ML100K_URL, zpath)
    digest = hashlib.sha256(zpath.read_bytes()).hexdigest()
    if digest != ML100K_SHA256:
        zpath.unlink()
        raise RuntimeError(f"SHA-256 mismatch for {ML100K_URL}: got {digest}. "
                           "GroupLens may have re-issued the archive; inspect it before updating the hash.")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(root)
    missing = [name for name in ML100K_REQUIRED if not (target / name).is_file()]
    if missing:
        raise RuntimeError(f"verified MovieLens archive is missing required files: {missing}")
    return target


def load_movielens(root: str | Path = "data") -> MovieLensTables:
    d = download_ml100k(root)
    users = pd.read_csv(d / "u.user", sep="|", names=["user_id", "age", "gender", "occupation", "zip"])
    zip_first = users["zip"].astype(str).str[0]
    users["zip_region"] = np.where(zip_first.str.isdigit(), zip_first, "other")
    users = users.sort_values("user_id").reset_index(drop=True)
    user_map = {uid: i for i, uid in enumerate(users["user_id"])}
    users = users[NUM_COLS + CAT_COLS]

    genres = pd.read_csv(d / "u.genre", sep="|", names=["genre", "id"]).dropna()["genre"].tolist()
    occupations = [o.strip() for o in (d / "u.occupation").read_text().split("\n") if o.strip()]
    icols = ["item_id", "title", "release_date", "video_release", "url"] + genres
    items = pd.read_csv(d / "u.item", sep="|", names=icols, encoding="latin-1")
    items = items.sort_values("item_id").reset_index(drop=True)
    item_map = {iid: i for i, iid in enumerate(items["item_id"])}
    year = pd.to_datetime(items["release_date"], format="%d-%b-%Y", errors="coerce").dt.year
    items["release_year"] = year.fillna(year.median()).astype(float)  # one item has no date
    items = items[genres + ["release_year"]]

    r = pd.read_csv(d / "u.data", sep="\t", names=["user", "item", "rating", "ts"])
    ratings = pd.DataFrame({"user": r["user"].map(user_map), "item": r["item"].map(item_map),
                            "rating": r["rating"].astype(int)})
    return MovieLensTables(users, items, ratings, genres, occupations)


@dataclass
class TableCodec:
    """Numeric columns: z-score with TRAIN statistics. Categorical: integer codes over a fixed vocabulary."""
    num_cols: list[str]
    cat_cols: list[str]
    vocab: dict[str, list[str]]
    mean: np.ndarray
    std: np.ndarray
    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def fit(cls, train_df: pd.DataFrame, vocab: dict[str, list[str]],
            num_cols: list[str] = NUM_COLS, cat_cols: list[str] = CAT_COLS) -> "TableCodec":
        x = train_df[num_cols].to_numpy(float)
        return cls(list(num_cols), list(cat_cols), vocab, x.mean(0), x.std(0) + 1e-8, x.min(0), x.max(0))

    @property
    def cat_sizes(self) -> list[int]:
        return [len(self.vocab[c]) for c in self.cat_cols]

    def encode(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x_num = ((df[self.num_cols].to_numpy(float) - self.mean) / self.std).astype(np.float32)
        codes = [pd.Categorical(df[c], categories=self.vocab[c]).codes for c in self.cat_cols]
        x_cat = np.stack(codes, 1).astype(np.int64)
        if (x_cat < 0).any():
            raise ValueError("category outside the fixed vocabulary")
        return x_num, x_cat

    def decode(self, x_num: np.ndarray, x_cat: np.ndarray) -> pd.DataFrame:
        raw = x_num * self.std + self.mean
        out = {c: np.clip(np.rint(raw[:, j]), self.lo[j], self.hi[j]) for j, c in enumerate(self.num_cols)}
        for j, c in enumerate(self.cat_cols):
            out[c] = np.asarray(self.vocab[c], dtype=object)[x_cat[:, j]]
        return pd.DataFrame(out)


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def make_split(strata: np.ndarray, fractions: tuple[float, float, float], seed: int) -> Split:
    """Node-level split of users, stratified (here: by gender). Fractions = (train, val, test)."""
    f_tr, f_va, f_te = fractions
    if min(fractions) <= 0 or abs(f_tr + f_va + f_te - 1) >= 1e-8:
        raise ValueError("split fractions must be positive and sum to one")
    idx = np.arange(len(strata))
    rest, test = train_test_split(idx, test_size=f_te, stratify=strata, random_state=seed)
    train, val = train_test_split(rest, test_size=f_va / (f_tr + f_va), stratify=strata[rest], random_state=seed)
    return Split(np.sort(train), np.sort(val), np.sort(test))
