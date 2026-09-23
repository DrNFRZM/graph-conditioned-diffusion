"""Two figures: (1) node-level information carried by generated attributes, (2) fidelity + memorisation check.
Bars = mean over seeds, whiskers = sample std, dots = individual seeds."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# fixed colour per method (never re-assigned when a subset is plotted)
COLORS = {"marginal_bootstrap": "#9a9a95", "row_bootstrap": "#c9c9c3", "diff_uncond": "#52514e",
          "diff_flat": "#eb6834", "diff_gnn": "#2a78d6", "diff_gnn_shuffled": "#1baf7a", "diff_gnn_knn": "#4a3aa7"}
LABELS = {"marginal_bootstrap": "A1 independent marginals", "row_bootstrap": "A2 resampled real rows",
          "diff_uncond": "B  diffusion, no condition", "diff_flat": "C  diffusion + flat features",
          "diff_gnn": "D  diffusion + GNN (bipartite)", "diff_gnn_shuffled": "D-abl  GNN, endpoint shuffle",
          "diff_gnn_knn": "D-abl  GNN, user-user kNN"}


def _panel(ax, df, metric, title, methods, ref=None, ref_label=None, better="higher is better"):
    y = np.arange(len(methods))[::-1]
    for yi, m in zip(y, methods):
        v = df.loc[df.method == m, metric].dropna().to_numpy()
        if len(v) == 0:
            continue
        sd = v.std(ddof=1) if len(v) > 1 else 0.0
        ax.barh(yi, v.mean(), height=0.55, color=COLORS[m], alpha=0.9, xerr=sd,
                error_kw=dict(ecolor="#0b0b0b", lw=1, capsize=2))
        ax.scatter(v, np.full(len(v), yi), s=9, color="white", edgecolor="#0b0b0b", linewidth=0.6, zorder=3)
    if ref is not None:
        ax.axvline(ref[0], color="#0b0b0b", ls=":", lw=1)
        ax.text(ref[0], len(methods) - 0.4, ref_label + " ", fontsize=7, ha="right", va="bottom", color="#52514e")
    ax.set_title(f"{title}\n({better})", fontsize=9, loc="left")
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[m] for m in methods] if ax.get_subplotspec().is_first_col() else [], fontsize=8)
    ax.set_ylim(-0.6, len(methods) - 0.1 + 0.4)
    ax.grid(axis="x", color="#e5e5e0", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8)


def make_plots(df: pd.DataFrame, out: Path) -> None:
    out = Path(out)
    df = df.copy()
    methods = [m for m in LABELS if m in set(df.method)]
    n_seeds = df.seed.nunique()
    probe = df[df.method == "probe_real_labels"]
    # paired per-seed difference to the unconditional model (same split, same fixed noise)
    base = df[df.method == "diff_uncond"].set_index("seed")["test_loss"]
    df["test_loss_delta"] = df["test_loss"] - df["seed"].map(base)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7))
    specs = [("test_loss_delta", "Held-out denoising loss minus unconditional (paired per seed)", 0.0, "unconditional", "lower is better"),
             ("cond_age_r2", "Age R² of draw-mean vs. real (test users)", 0.0, "train-mean baseline", "higher is better"),
             ("cond_gender_auc", "Gender AUC of draws vs. real (test users)", 0.5, "chance", "higher is better"),
             ("tstr_node_gender_auc", "Train on synthetic labels, test on real: gender AUC", 0.5, "chance", "higher is better")]
    for ax, (k, t, ref, rl, better) in zip(axes.ravel(), specs):
        _panel(ax, df, k, t, methods, ref=(ref, ""), ref_label=rl, better=better)
        if len(probe) and k != "test_loss_delta":
            pk = probe[k].mean()
            ax.axvline(pk, color="#2a78d6", ls="--", lw=1)
            ax.text(pk, len(methods) - 0.4, " real-label probe", fontsize=7, color="#2a78d6", ha="left", va="bottom")
    fig.suptitle(f"Do generated attributes carry information about the user's graph?  (MovieLens 100K, {n_seeds} seeds; whiskers = std, dots = seeds)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out / "fig_node_information.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    _panel(axes[0], df, "marginal_score", "Marginal distance (mean KS/TV)", methods, better="lower is better")
    _panel(axes[1], df, "assoc_error", "Association error (all column pairs)", methods, better="lower is better")
    _panel(axes[2], df, "dcr_ratio", "Nearest-train-row distance: synthetic / new real users", methods,
           ref=(1.0, ""), ref_label="1 = as far as new real users", better="closer to 1 is better; near 0 = copying")
    fig.suptitle("Fidelity to held-out real users and a copying sanity check (not a privacy guarantee)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_fidelity.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":  # regenerate figures: python -m gcdiff.plots results/quick_10seeds
    import sys
    d = Path(sys.argv[1])
    make_plots(pd.read_csv(d / "metrics_per_seed.csv"), d)
