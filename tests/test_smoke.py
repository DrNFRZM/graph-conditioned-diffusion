import copy
from pathlib import Path

import pandas as pd

from gcdiff.experiment import BASELINES, DIFFUSION_METHODS, load_config, run_experiment

ROOT = Path(__file__).resolve().parents[1]
TINY = {
    "name": "test", "device": "cpu", "seeds": [0],
    "data": {"root": "unused", "like_threshold": 4, "split": [0.6, 0.15, 0.25], "knn_k": 5},
    "diffusion": {"T": 8, "width": 16, "depth": 1, "t_dim": 8, "cond_dim": 4, "dropout": 0.0, "clip_x0": 4.0},
    "gnn": {"hidden": 8, "layers": 2, "dropout": 0.0},
    "train": {"epochs": 2, "batch_size": 16, "lr": 0.003, "weight_decay": 0.0, "eval_every": 1, "patience": 5},
    "eval": {"n_samples_per_node": 2, "knn_eval_k": 4},
    "methods": list(BASELINES) + list(DIFFUSION_METHODS),
}


def test_end_to_end_all_methods_and_reproducibility(tables, tmp_path):
    a = run_experiment(copy.deepcopy(TINY), tmp_path / "a", tables=tables, log=lambda *_: None)
    b = run_experiment(copy.deepcopy(TINY), tmp_path / "b", tables=tables, log=lambda *_: None)
    assert set(a.method) == set(TINY["methods"]) | {"probe_real_labels"}
    for f in ("config.yaml", "env.json", "metrics_per_seed.csv", "summary.csv", "paired_deltas.csv",
              "fig_node_information.png", "fig_fidelity.png"):
        assert (tmp_path / "a" / f).exists()
    summary = pd.read_csv(tmp_path / "a" / "summary.csv")
    assert {"method", "cond_age_r2_mean", "cond_age_r2_std"} <= set(summary.columns)
    cols = ["marginal_score", "assoc_error", "cond_gender_auc", "tstr_node_gender_auc", "dcr_ratio", "test_loss", "method"]
    x = a[a.method != "probe_real_labels"][cols].reset_index(drop=True)
    y = b[b.method != "probe_real_labels"][cols].reset_index(drop=True)
    assert x.drop(columns=["method", "test_loss"]).notna().all().all()
    assert x[x.method.str.startswith("diff_")]["test_loss"].notna().all()   # baselines have no denoising loss
    pd.testing.assert_frame_equal(x, y)          # same seed -> identical numbers (CPU)


def test_shipped_configs_are_consistent():
    for name in ("quick", "full"):
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        assert set(cfg["methods"]) <= set(BASELINES) | set(DIFFUSION_METHODS)
        assert abs(sum(cfg["data"]["split"]) - 1) < 1e-9
