"""Denoiser and condition encoders. Deliberately small.

All condition encoders expose forward(inputs) -> [n_users, cond_dim] and end in a
parameter-free LayerNorm, so the denoiser sees embeddings of the same scale regardless of encoder.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

from .graphs import EDGE_TYPES


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of the integer timestep (as in Transformers / DDPM)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t.float().unsqueeze(-1) * freqs
    return torch.cat([ang.sin(), ang.cos()], -1)


class FiLMBlock(nn.Module):
    """Residual MLP block whose hidden activations are scaled/shifted by the conditioning vector (FiLM)."""
    def __init__(self, width: int, emb_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1, self.fc2 = nn.Linear(width, width), nn.Linear(width, width)
        self.film = nn.Linear(emb_dim, 2 * width)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, emb):
        scale, shift = self.film(emb).chunk(2, -1)
        z = self.norm(h) * (1 + scale) + shift
        return h + self.fc2(self.drop(F.silu(self.fc1(F.silu(z)))))


class Denoiser(nn.Module):
    """f_theta(x_t, t, c): outputs eps_hat for numeric columns and logits for each categorical column."""
    def __init__(self, n_num: int, cat_sizes: list[int], cond_dim: int = 0, width: int = 128,
                 depth: int = 3, t_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.t_dim, self.cond_dim = t_dim, cond_dim
        d_in = n_num + sum(cat_sizes)
        emb_dim = width
        self.in_proj = nn.Linear(d_in, width)
        # timestep and condition are mixed by an MLP so their interaction can be non-additive
        self.emb = nn.Sequential(nn.Linear(t_dim + cond_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.blocks = nn.ModuleList([FiLMBlock(width, emb_dim, dropout) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, n_num + sum(cat_sizes)))

    def forward(self, x_num, x_cat_onehot, t, cond=None):
        e = timestep_embedding(t, self.t_dim)
        if self.cond_dim > 0:
            if cond is None or cond.shape[-1] != self.cond_dim:
                raise ValueError(f"expected condition of width {self.cond_dim}")
            e = torch.cat([e, cond], -1)
        emb = self.emb(e)
        h = self.in_proj(torch.cat([x_num, x_cat_onehot], -1))
        for blk in self.blocks:
            h = blk(h, emb)
        return self.out(h)


class FlatEncoder(nn.Module):
    """Non-graph conditioning: MLP on the flat behaviour vector (what 'flattening' would give)."""
    def __init__(self, in_dim: int, cond_dim: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, cond_dim))
        self.norm = nn.LayerNorm(cond_dim, elementwise_affine=False)

    def forward(self, x):
        return self.norm(self.net(x))


class HeteroGNNEncoder(nn.Module):
    """Relational GraphSAGE on the user-item graph (likes / dislikes relations, both directions).
    User nodes start from label-free structural features only; the embedding of a user is a function
    of the items it is connected to (and, at depth 2, of the other users who rated those items)."""
    def __init__(self, user_in: int, item_in: int, cond_dim: int, hidden: int = 32, layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.lin = nn.ModuleDict({"user": nn.Linear(user_in, hidden), "item": nn.Linear(item_in, hidden)})
        self.convs = nn.ModuleList([
            HeteroConv({et: SAGEConv(hidden, hidden, aggr="mean") for et in EDGE_TYPES}, aggr="sum")
            for _ in range(layers)])
        self.out = nn.Linear(hidden, cond_dim)
        self.norm = nn.LayerNorm(cond_dim, elementwise_affine=False)
        self.dropout = dropout

    def forward(self, g):
        h = {k: F.relu(self.lin[k](g[k].x)) for k in ("user", "item")}
        for conv in self.convs:
            new = conv(h, g.edge_index_dict)
            h = {k: h[k] + F.dropout(F.relu(new[k]), self.dropout, self.training) for k in h}
        return self.norm(self.out(h["user"]))


class HomoGNNEncoder(nn.Module):
    """GraphSAGE on a homogeneous user-user kNN graph (alternative graph construction)."""
    def __init__(self, in_dim: int, cond_dim: int, hidden: int = 32, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([SAGEConv(hidden, hidden, aggr="mean") for _ in range(layers)])
        self.out = nn.Linear(hidden, cond_dim)
        self.norm = nn.LayerNorm(cond_dim, elementwise_affine=False)
        self.dropout = dropout

    def forward(self, g):
        h = F.relu(self.lin(g.x))
        for conv in self.convs:
            h = h + F.dropout(F.relu(conv(h, g.edge_index)), self.dropout, self.training)
        return self.norm(self.out(h))
