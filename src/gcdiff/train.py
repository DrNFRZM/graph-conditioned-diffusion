"""Training loop: AdamW, early stopping on a deterministic validation diffusion loss."""
from __future__ import annotations

import copy
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cond(encoder, cond_input, idx):
    return None if encoder is None else encoder(cond_input)[idx]


@torch.no_grad()
def validation_loss(model, encoder, cond_input, x_num, x_cat, idx, seed: int, repeats: int = 4) -> dict[str, float]:
    """Diffusion loss on held-out users with FIXED noise and timesteps stratified over 1..T,
    so the number is comparable across epochs and across methods with the same seed."""
    model.eval()
    if encoder is not None:
        encoder.eval()
    gen = torch.Generator(device=x_num.device).manual_seed(10_000 + seed)
    cond = _cond(encoder, cond_input, idx)
    T, n = model.sch.T, len(idx)
    tot = {"loss": 0.0, "num": 0.0, "cat": 0.0}
    for r in range(repeats):
        t = (torch.arange(n, device=x_num.device) + r * (T // repeats + 1)) % T + 1
        out = model.loss(x_num[idx], x_cat[idx], cond, gen, t=t)
        for k in tot:
            tot[k] += out[k].item() / repeats
    return tot


def fit(model, encoder, cond_input, x_num, x_cat, idx_train, idx_val, cfg: dict, seed: int) -> dict:
    """Trains denoiser (and encoder end-to-end through the diffusion loss). Restores the best-validation weights."""
    params = list(model.parameters()) + ([] if encoder is None else list(encoder.parameters()))
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    dev = x_num.device
    noise_gen = torch.Generator(device=dev).manual_seed(seed)
    order_gen = torch.Generator().manual_seed(seed)
    n, bs = len(idx_train), cfg["batch_size"]
    best, best_state, best_epoch, bad = float("inf"), None, 0, 0
    hist = {"epoch": [], "train": [], "val": []}
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        if encoder is not None:
            encoder.train()
        perm = idx_train[torch.randperm(n, generator=order_gen).to(dev)]
        run, cnt = 0.0, 0
        for b in range(0, n, bs):
            idx = perm[b:b + bs]
            if len(idx) < 2:
                continue
            cond = _cond(encoder, cond_input, idx)      # full-graph pass, then pick the batch's users
            loss = model.loss(x_num[idx], x_cat[idx], cond, noise_gen)["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            run, cnt = run + loss.item() * len(idx), cnt + len(idx)
        if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]:
            val = validation_loss(model, encoder, cond_input, x_num, x_cat, idx_val, seed)["loss"]
            hist["epoch"].append(epoch); hist["train"].append(run / max(cnt, 1)); hist["val"].append(val)
            if val < best - 1e-4:
                best, best_epoch, bad = val, epoch, 0
                best_state = (copy.deepcopy(model.state_dict()),
                              None if encoder is None else copy.deepcopy(encoder.state_dict()))
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break
    model.load_state_dict(best_state[0])
    if encoder is not None:
        encoder.load_state_dict(best_state[1])
    return {"best_val": best, "best_epoch": best_epoch, "epochs_run": epoch, "history": hist}
