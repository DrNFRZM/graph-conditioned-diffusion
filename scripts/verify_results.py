"""Check committed aggregates and the README against per-seed CSVs, without training.

This verifies internal arithmetic and complete method/seed coverage. It does not
recreate predictions or prove that the training run generated the source CSV.
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [("diff_gnn", b) for b in ("diff_uncond", "diff_flat", "diff_gnn_shuffled", "diff_gnn_knn")]
PAIRS.append(("diff_flat", "diff_uncond"))
METRICS = ["test_loss", "test_loss_num", "test_loss_cat", "marginal_score", "assoc_error",
           "tstr_table_auc", "cond_gender_auc", "cond_age_r2", "cond_occ_acc",
           "tstr_node_gender_auc", "tstr_node_age_r2"]
LABELS = {"diff_uncond": "B unconditional", "diff_flat": "C flat features", "diff_gnn": "D GNN",
          "diff_gnn_shuffled": "D, endpoint shuffle", "diff_gnn_knn": "D, kNN graph"}


def compare(actual, expected, keys, label):
    if actual.duplicated(keys).any():
        raise ValueError(f"{label}: duplicate keys")
    a = actual.set_index(keys).sort_index()
    b = expected.set_index(keys).sort_index()
    pd.testing.assert_index_equal(a.index, b.index)
    if set(a.columns) != set(b.columns):
        raise ValueError(f"{label}: columns differ")
    np.testing.assert_allclose(a[b.columns].to_numpy(float), b.to_numpy(float),
                               rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=label)


def verify(root=ROOT):
    run = root / "results/quick_10seeds"
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    df = pd.read_csv(run / "metrics_per_seed.csv")
    if df.duplicated(["method", "seed"]).any():
        raise ValueError("duplicate (method, seed) result")
    expected_keys = {(m, s) for m in [*cfg["methods"], "probe_real_labels"] for s in cfg["seeds"]}
    if set(zip(df.method, df.seed)) != expected_keys:
        raise ValueError("method/seed coverage does not match the saved configuration")
    metrics = df.drop(columns="seed").select_dtypes("number").columns
    if np.isinf(df[metrics].to_numpy(float)).any():
        raise ValueError("infinite per-seed metric")
    means = df.groupby("method")[metrics].mean()
    stds = df.groupby("method")[metrics].std(ddof=1)
    summary = pd.DataFrame(index=means.index)
    for metric in metrics:
        summary[f"{metric}_mean"] = means[metric]
        summary[f"{metric}_std"] = stds[metric]
    compare(pd.read_csv(run / "summary.csv"), summary.reset_index(), ["method"], "summary")
    rows = []
    for a, b in PAIRS:
        for metric in METRICS:
            diff = (df[df.method == a].set_index("seed")[metric]
                    - df[df.method == b].set_index("seed")[metric]).dropna()
            rows.append(dict(a=a, b=b, metric=metric, mean_diff=diff.mean(), std_diff=diff.std(ddof=1),
                             n_seeds=len(diff), n_a_higher=int((diff > 0).sum())))
    compare(pd.read_csv(run / "paired_deltas.csv"), pd.DataFrame(rows), ["a", "b", "metric"], "paired deltas")
    readme = (root / "README.md").read_text(encoding="utf-8")
    for method, label in LABELS.items():
        cells = []
        for metric in ("test_loss", "cond_age_r2", "cond_gender_auc", "assoc_error"):
            cell = f"{means.loc[method, metric]:.3f} ± {stds.loc[method, metric]:.3f}".replace("-", "−")
            cells.append(cell)
        line = next((line for line in readme.splitlines() if line.startswith(f"| {label} |")), "")
        actual_cells = [cell.strip().replace("**", "") for cell in line.split("|")[2:-1]]
        if actual_cells != cells:
            raise ValueError(f"README table differs from per-seed metrics: {method}")
    print(f"Verified {len(df)} per-seed rows, every summary and paired-delta cell, and five README rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="repository or copied evidence directory")
    verify(parser.parse_args().root)
