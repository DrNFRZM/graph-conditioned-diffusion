"""Graph construction and non-learned behaviour features.

LEAKAGE RULE: nothing here takes the user attribute table (age/gender/occupation/zip).
Inputs are only the rating triples and the ITEM table, so the conditioning signal
cannot contain the target by construction (tests/test_data_graphs.py checks this).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_undirected

EDGE_TYPES = [
    ("user", "likes", "item"), ("item", "liked_by", "user"),
    ("user", "dislikes", "item"), ("item", "disliked_by", "user"),
]


@dataclass
class Features:
    flat: np.ndarray       # [n_users, 41] behaviour summary used by the non-graph baseline and the kNN graph
    user_struct: np.ndarray  # [n_users, 2] log-degree per relation (node input of the bipartite GNN)
    item_x: np.ndarray     # [n_items, n_genres + 1] genres + standardised release year


def behaviour_features(ratings: pd.DataFrame, items: pd.DataFrame, genres: list[str], n_users: int,
                       like_threshold: int = 4) -> Features:
    """Graph-derived but non-learned features. 'Like' = rating >= like_threshold."""
    n_items = len(items)
    G = items[genres].to_numpy(np.float32)
    year = items["release_year"].to_numpy(np.float32)
    year = (year - year.mean()) / (year.std() + 1e-8)  # item statistics only, no user attributes involved
    like = np.zeros((n_users, n_items), np.float32)
    dis = np.zeros((n_users, n_items), np.float32)
    u, i, r = ratings["user"].to_numpy(), ratings["item"].to_numpy(), ratings["rating"].to_numpy()
    like[u[r >= like_threshold], i[r >= like_threshold]] = 1
    dis[u[r < like_threshold], i[r < like_threshold]] = 1
    n_like, n_dis = like.sum(1), dis.sum(1)
    prof_like = like @ G / np.maximum(n_like, 1)[:, None]
    prof_dis = dis @ G / np.maximum(n_dis, 1)[:, None]
    mean_year = (like + dis) @ year / np.maximum(n_like + n_dis, 1)
    struct = np.stack([np.log1p(n_like), np.log1p(n_dis)], 1)
    flat = np.concatenate([prof_like, prof_dis, struct, mean_year[:, None]], 1)
    item_x = np.concatenate([G, year[:, None]], 1)
    return Features(flat.astype(np.float32), struct.astype(np.float32), item_x.astype(np.float32))


def standardise_users(feats: Features, train_idx: np.ndarray) -> Features:
    """Fit user-feature scalers on TRAIN users only (features, not targets, but keep the discipline)."""
    out = Features(feats.flat.copy(), feats.user_struct.copy(), feats.item_x)
    for name in ("flat", "user_struct"):
        x = getattr(out, name)
        setattr(out, name, StandardScaler().fit(x[train_idx]).transform(x).astype(np.float32))
    return out


def build_hetero_graph(ratings: pd.DataFrame, user_x: np.ndarray, item_x: np.ndarray,
                       like_threshold: int = 4) -> HeteroData:
    """Bipartite user-item graph with two relation types: likes (rating >= thr) and dislikes (< thr)."""
    u = torch.as_tensor(ratings["user"].to_numpy(copy=True), dtype=torch.long)
    i = torch.as_tensor(ratings["item"].to_numpy(copy=True), dtype=torch.long)
    liked = torch.as_tensor(ratings["rating"].to_numpy(copy=True) >= like_threshold)
    g = HeteroData()
    g["user"].x = torch.as_tensor(user_x)
    g["item"].x = torch.as_tensor(item_x)
    for fwd, rev, mask in ((EDGE_TYPES[0], EDGE_TYPES[1], liked), (EDGE_TYPES[2], EDGE_TYPES[3], ~liked)):
        ei = torch.stack([u[mask], i[mask]])
        g[fwd].edge_index = ei
        g[rev].edge_index = ei.flip(0)
    return g


def shuffle_item_endpoints(ratings: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Structural control: permute item ids across all rating rows.

    This keeps each user's degree and rating multiset and the global item-degree
    sequence, but not relation-specific item degrees. Parallel user-item
    endpoints can be introduced. It randomises which item endpoints each user
    receives without pretending to be a fully constrained graph null model.
    """
    rng = np.random.default_rng(seed)
    out = ratings.copy()
    out["item"] = rng.permutation(out["item"].to_numpy())
    return out


def knn_user_graph(x: np.ndarray, k: int) -> Data:
    """Alternative construction: homogeneous user-user graph, k nearest neighbours by cosine similarity
    of the (standardised) behaviour features; symmetrised."""
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    sim = xn @ xn.T
    np.fill_diagonal(sim, -np.inf)
    nbrs = np.argpartition(-sim, k, axis=1)[:, :k]
    src = np.repeat(np.arange(len(x)), k)
    ei = torch.as_tensor(np.stack([nbrs.reshape(-1), src]), dtype=torch.long)  # neighbour -> node
    return Data(x=torch.as_tensor(x), edge_index=to_undirected(ei, num_nodes=len(x)))
