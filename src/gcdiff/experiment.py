"""Runs all methods over several seeds (each seed = new node split, new initialisation) and evaluates them."""
from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from . import baselines, metrics
from .data import MovieLensTables, TableCodec, load_movielens, make_split
from .diffusion import MixedDiffusion, Schedule
from .graphs import (Features, behaviour_features, build_hetero_graph, knn_user_graph,
                     shuffle_item_endpoints, standardise_users)
from .models import Denoiser, FlatEncoder, HeteroGNNEncoder, HomoGNNEncoder
from .train import fit, set_seed, validation_loss

# method name -> conditioning kind
DIFFUSION_METHODS = {"diff_uncond": "none", "diff_flat": "flat", "diff_gnn": "gnn",
                     "diff_gnn_shuffled": "gnn_shuffled", "diff_gnn_knn": "gnn_knn"}
BASELINES = {"marginal_bootstrap": baselines.marginal_bootstrap, "row_bootstrap": baselines.row_bootstrap}
KEY_METRICS = ["test_loss", "test_loss_num", "test_loss_cat", "marginal_score", "assoc_error", "tstr_table_auc", "cond_gender_auc", "cond_age_r2",
               "cond_occ_acc", "tstr_node_gender_auc", "tstr_node_age_r2"]


@dataclass
class Synth:
    train_first: pd.DataFrame  # one draw per TRAIN user (used for train-on-synthetic tasks)
    test: pd.DataFrame         # S draws per TEST user, s-major order
    S: int


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _cond_setup(kind: str, feats: Features, graphs: dict, cfg: dict, device):
    """Returns (encoder | None, encoder input)."""
    d, g = cfg["diffusion"]["cond_dim"], cfg["gnn"]
    if kind == "none":
        return None, None
    if kind == "flat":
        return FlatEncoder(feats.flat.shape[1], d).to(device), torch.as_tensor(feats.flat).to(device)
    if kind in ("gnn", "gnn_shuffled"):
        enc = HeteroGNNEncoder(feats.user_struct.shape[1], feats.item_x.shape[1], d, g["hidden"], g["layers"], g["dropout"])
        return enc.to(device), graphs[kind].to(device)
    if kind == "gnn_knn":
        enc = HomoGNNEncoder(feats.flat.shape[1], d, g["hidden"], g["layers"], g["dropout"])
        return enc.to(device), graphs[kind].to(device)
    raise ValueError(kind)


def _generate(model, encoder, cond_input, codec, idx_train, idx_test, S, seed, cfg, device) -> Synth:
    gen = torch.Generator(device=device).manual_seed(20_000 + seed)
    clip = cfg["diffusion"].get("clip_x0", 4.0)
    if encoder is not None:
        encoder.eval()
        with torch.no_grad():
            emb = encoder(cond_input)
        c_tr, c_te = emb[idx_train], emb[idx_test].repeat(S, 1)     # repeat -> s-major: row s*n+i = draw s of user i
    else:
        c_tr = c_te = None
    xn, xc = model.sample(len(idx_train), c_tr, gen, clip)
    train_first = codec.decode(xn.cpu().numpy(), xc.cpu().numpy())
    xn, xc = model.sample(len(idx_test) * S, c_te, gen, clip)
    return Synth(train_first, codec.decode(xn.cpu().numpy(), xc.cpu().numpy()), S)


def _evaluate(syn: Synth, c: dict) -> dict[str, float]:
    real_train, real_test, n = c["real_train"], c["real_test"], len(c["real_test"])
    first = syn.test.iloc[:n].reset_index(drop=True)
    out = metrics.marginal_fidelity(real_test, syn.test)
    out["assoc_error"] = metrics.association_error(real_test, syn.test)
    out["tstr_table_auc"] = metrics.tstr_table_auc(
        syn.train_first, real_test, c["vocab"], c["age_mean"], c["age_std"])
    out.update(metrics.conditional_consistency(syn.test, syn.S, real_test, c["age_mean"]))
    node = metrics.node_probe_metrics(c["flat_train"], syn.train_first, c["flat_test"], real_test, c["age_mean"])
    out["tstr_node_gender_auc"], out["tstr_node_age_r2"] = node["gender_auc"], node["age_r2"]
    h = metrics.homophily(first, c["knn_test"])
    out["hom_gender_gap"] = abs(h["same_gender_rate"] - c["hom_real"]["same_gender_rate"])
    out["hom_age_gap"] = abs(h["age_assort"] - c["hom_real"]["age_assort"])
    out.update(metrics.memorisation_check(first, real_train, real_test))
    # per-user memorisation: does a draw for a TRAIN user match its real attributes better than for an unseen user?
    g_tr = (syn.train_first["gender"].to_numpy() == real_train["gender"].to_numpy()).mean()
    g_te = (first["gender"].to_numpy() == real_test["gender"].to_numpy()).mean()
    a_tr = np.abs(syn.train_first["age"].to_numpy() - real_train["age"].to_numpy()).mean()
    a_te = np.abs(first["age"].to_numpy() - real_test["age"].to_numpy()).mean()
    out["node_gap_gender_agree"], out["node_gap_age_mae"] = float(g_tr - g_te), float(a_te - a_tr)
    return out


def run_seed(tables: MovieLensTables, raw: Features, cfg: dict, seed: int, device, log=print):
    dcfg, tcfg = cfg["data"], cfg["train"]
    users = tables.users
    split = make_split((users["gender"] == "M").to_numpy(), tuple(dcfg["split"]), seed)
    real_train = users.iloc[split.train].reset_index(drop=True)
    real_test = users.iloc[split.test].reset_index(drop=True)
    codec = TableCodec.fit(real_train, tables.vocab)                    # statistics from TRAIN users only
    x_num, x_cat = (torch.as_tensor(a).to(device) for a in codec.encode(users))
    feats = standardise_users(raw, split.train)
    thr = dcfg["like_threshold"]
    graphs = {
        "gnn": build_hetero_graph(tables.ratings, feats.user_struct, feats.item_x, thr),
        "gnn_shuffled": build_hetero_graph(shuffle_item_endpoints(tables.ratings, seed), feats.user_struct, feats.item_x, thr),
        "gnn_knn": knn_user_graph(feats.flat, dcfg["knn_k"]),
    }
    idx = {k: torch.as_tensor(v).to(device) for k, v in (("train", split.train), ("val", split.val), ("test", split.test))}
    S = cfg["eval"]["n_samples_per_node"]
    knn_test = metrics.rating_knn(tables.ratings, split.test, tables.n_items, cfg["eval"]["knn_eval_k"])
    ctx = dict(real_train=real_train, real_test=real_test, vocab=tables.vocab,
               age_mean=float(real_train["age"].mean()), age_std=float(real_train["age"].std() + 1e-8),
               flat_train=feats.flat[split.train], flat_test=feats.flat[split.test], knn_test=knn_test,
               hom_real=metrics.homophily(real_test, knn_test))
    rows, logs = [], {}

    # reference: discriminative probe on REAL labels (not a generator)
    ref = metrics.node_probe_metrics(ctx["flat_train"], real_train, ctx["flat_test"], real_test, ctx["age_mean"])
    rows.append(dict(method="probe_real_labels", seed=seed, cond_gender_auc=ref["gender_auc"], cond_age_r2=ref["age_r2"],
                     tstr_node_gender_auc=ref["gender_auc"], tstr_node_age_r2=ref["age_r2"]))

    rng = np.random.default_rng(seed)
    for name in cfg["methods"]:
        t0 = time.time()
        if name in BASELINES:
            f = BASELINES[name]
            syn = Synth(f(real_train, len(real_train), rng), f(real_train, len(real_test) * S, rng), S)
            info, n_params = {}, 0
        else:
            set_seed(seed)
            d = cfg["diffusion"]
            kind = DIFFUSION_METHODS[name]
            den = Denoiser(len(codec.num_cols), codec.cat_sizes, 0 if kind == "none" else d["cond_dim"],
                           d["width"], d["depth"], d["t_dim"], d["dropout"])
            model = MixedDiffusion(den, Schedule(d["T"]), len(codec.num_cols), codec.cat_sizes).to(device)
            enc, cin = _cond_setup(kind, feats, graphs, cfg, device)
            n_params = sum(p.numel() for p in model.parameters()) + (0 if enc is None else sum(p.numel() for p in enc.parameters()))
            info = fit(model, enc, cin, x_num, x_cat, idx["train"], idx["val"], tcfg, seed)
            tl = validation_loss(model, enc, cin, x_num, x_cat, idx["test"], seed)   # reporting only, not used for selection
            info.update(test_loss=tl["loss"], test_loss_num=tl["num"], test_loss_cat=tl["cat"])
            syn = _generate(model, enc, cin, codec, idx["train"], idx["test"], S, seed, cfg, device)
        row = dict(method=name, seed=seed, n_params=n_params, seconds=round(time.time() - t0, 1),
                   best_epoch=info.get("best_epoch"), best_val=info.get("best_val"), test_loss=info.get("test_loss"),
                   test_loss_num=info.get("test_loss_num"), test_loss_cat=info.get("test_loss_cat"))
        row.update(_evaluate(syn, ctx))
        rows.append(row)
        logs[name] = info.get("history")
        log(f"  seed {seed} {name:20s} {row['seconds']:6.1f}s  best_ep={row['best_epoch']}  "
            f"gAUC={row['cond_gender_auc']:.3f} ageR2={row['cond_age_r2']:.3f} marg={row['marginal_score']:.3f} assoc={row['assoc_error']:.3f}")
    return rows, logs, {"hom_real": ctx["hom_real"]}


def summarise(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    num = df.drop(columns=["seed"]).select_dtypes("number").columns
    g = df.groupby("method", sort=False)[list(num)]
    means, stds = g.mean(), g.std(ddof=1)
    # Flat columns keep the CSV readable in a browser and unambiguous in pandas.
    summary = pd.DataFrame(index=means.index)
    for metric in num:
        summary[f"{metric}_mean"] = means[metric]
        summary[f"{metric}_std"] = stds[metric]
    pairs = [("diff_gnn", "diff_uncond"), ("diff_gnn", "diff_flat"), ("diff_gnn", "diff_gnn_shuffled"),
             ("diff_gnn", "diff_gnn_knn"), ("diff_flat", "diff_uncond")]
    piv = {m: df[df.method == m].set_index("seed") for m in df.method.unique()}
    recs = []
    for a, b in pairs:
        if a in piv and b in piv:
            for k in KEY_METRICS:
                diff = (piv[a][k] - piv[b][k]).dropna()
                recs.append(dict(a=a, b=b, metric=k, mean_diff=diff.mean(), std_diff=diff.std(ddof=1) if len(diff) > 1 else np.nan,
                                 n_seeds=len(diff), n_a_higher=int((diff > 0).sum())))
    return summary, pd.DataFrame(recs)


def env_info(device) -> dict:
    import sklearn, torch_geometric
    return dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
                torch_geometric=torch_geometric.__version__, sklearn=sklearn.__version__, numpy=np.__version__,
                pandas=pd.__version__, device=str(device),
                gpu=torch.cuda.get_device_name(0) if str(device).startswith("cuda") else None)


def run_experiment(cfg: dict, out_dir: str | Path, tables: MovieLensTables | None = None, log=print) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = cfg.get("device", "auto")
    device = torch.device("cuda" if dev == "auto" and torch.cuda.is_available() else ("cpu" if dev == "auto" else dev))
    tables = tables or load_movielens(cfg["data"]["root"])
    raw = behaviour_features(tables.ratings, tables.items, tables.genres, tables.n_users, cfg["data"]["like_threshold"])
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "env.json").write_text(json.dumps(env_info(device), indent=2))
    log(f"device={device}  users={tables.n_users} items={tables.n_items} ratings={len(tables.ratings)} seeds={cfg['seeds']}")
    all_rows, all_logs, extra = [], {}, {}
    for seed in cfg["seeds"]:
        rows, logs, ex = run_seed(tables, raw, cfg, seed, device, log)
        all_rows += rows
        all_logs[seed] = logs
        extra[seed] = ex
    df = pd.DataFrame(all_rows)
    summary, deltas = summarise(df)
    df.to_csv(out / "metrics_per_seed.csv", index=False)
    summary.to_csv(out / "summary.csv")
    deltas.to_csv(out / "paired_deltas.csv", index=False)
    (out / "training_curves.json").write_text(json.dumps({"curves": all_logs, "eval_graph_reference": extra}, default=float))
    from .plots import make_plots
    make_plots(df, out)
    return df
