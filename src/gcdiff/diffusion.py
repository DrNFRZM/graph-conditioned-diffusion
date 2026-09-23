"""Mixed-type diffusion written out explicitly (see docs/diffusion_explained.md).

  numeric columns     -> Gaussian DDPM, epsilon-prediction, loss = MSE(eps, eps_hat)   [Ho et al. 2020]
  categorical columns -> multinomial diffusion, x0-parameterised, loss = KL(q post || p_theta post)
                                                                         [Hoogeboom et al. 2021]
Conventions: timesteps are t = 1..T; index 0 is clean data (alpha_bar_0 = 1).
Noise schedule: cosine [Nichol & Dhariwal 2021]. All per-timestep constants are tensors of length T+1.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Schedule:
    def __init__(self, T: int, s: float = 0.008, max_beta: float = 0.999):
        self.T = T
        k = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos((k / T + s) / (1 + s) * math.pi / 2) ** 2
        beta = (1 - f[1:] / f[:-1]).clamp(max=max_beta)                     # beta_1..beta_T
        alpha = 1 - beta
        ab = torch.cat([torch.ones(1, dtype=torch.float64), torch.cumprod(alpha, 0)])  # alpha_bar_0..T
        ab_prev, ab_t = ab[:-1], ab[1:]
        pad = lambda v, fill=0.0: torch.cat([torch.full((1,), fill, dtype=torch.float64), v]).float()
        self.beta = pad(beta)
        self.alpha = pad(alpha, 1.0)
        self.alpha_bar = ab.float()
        # Gaussian posterior q(x_{t-1} | x_t, x_0) = N(coef_x0 * x_0 + coef_xt * x_t, post_var)
        self.coef_x0 = pad(beta * ab_prev.sqrt() / (1 - ab_t))
        self.coef_xt = pad((1 - ab_prev) * alpha.sqrt() / (1 - ab_t))
        self.post_var = pad(beta * (1 - ab_prev) / (1 - ab_t))              # = 0 at t = 1 (ab_prev = 1)

    def to(self, device) -> "Schedule":
        for name in ("beta", "alpha", "alpha_bar", "coef_x0", "coef_xt", "post_var"):
            setattr(self, name, getattr(self, name).to(device))
        return self


def _col(v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Gather per-timestep constants -> shape [B, 1] so they broadcast over feature dims."""
    return v[t].unsqueeze(-1)


# ---------------------------------------------------------------- Gaussian (numeric columns)
def q_sample_gaussian(x0, t, noise, sch: Schedule):
    """Forward process in closed form: x_t = sqrt(ab_t) x_0 + sqrt(1 - ab_t) eps."""
    ab = _col(sch.alpha_bar, t)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


def predict_x0(xt, t, eps, sch: Schedule):
    ab = _col(sch.alpha_bar, t)
    return (xt - (1 - ab).sqrt() * eps) / ab.sqrt()


def gaussian_posterior(x0, xt, t, sch: Schedule):
    return _col(sch.coef_x0, t) * x0 + _col(sch.coef_xt, t) * xt, _col(sch.post_var, t)


# ---------------------------------------------------------------- multinomial (categorical columns)
def q_probs_categorical(x0_probs, t, sch: Schedule, K: int):
    """q(x_t | x_0) = Cat( ab_t * x_0 + (1 - ab_t) / K ): keep the category w.p. ab_t, else uniform."""
    ab = _col(sch.alpha_bar, t)
    return ab * x0_probs + (1 - ab) / K


def categorical_posterior(xt_onehot, x0_probs, t, sch: Schedule, K: int):
    """q(x_{t-1} | x_t, x_0) ∝ [alpha_t x_t + (1-alpha_t)/K] ⊙ [ab_{t-1} x_0 + (1-ab_{t-1})/K].
    With x0_probs = the network's softmax this is also the model's reverse step p_theta(x_{t-1} | x_t)."""
    a = _col(sch.alpha, t)
    ab_prev = _col(sch.alpha_bar, t - 1)
    post = (a * xt_onehot + (1 - a) / K) * (ab_prev * x0_probs + (1 - ab_prev) / K)
    return post / post.sum(-1, keepdim=True)


def categorical_kl(p, q, eps: float = 1e-30):
    return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(-1)


def _draw(probs, generator):
    idx = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    return idx, F.one_hot(idx, probs.shape[-1]).float()


# ---------------------------------------------------------------- full model wrapper
class MixedDiffusion(nn.Module):
    def __init__(self, denoiser: nn.Module, schedule: Schedule, n_num: int, cat_sizes: list[int]):
        super().__init__()
        self.denoiser, self.sch, self.n_num, self.cat_sizes = denoiser, schedule, n_num, list(cat_sizes)

    def to(self, *a, **k):
        super().to(*a, **k)
        self.sch.to(next(self.parameters()).device)
        return self

    def _split(self, out):
        eps = out[:, : self.n_num]
        logits = torch.split(out[:, self.n_num:], self.cat_sizes, dim=1)
        return eps, logits

    def loss(self, x_num, x_cat, cond, generator, t=None) -> dict[str, torch.Tensor]:
        """One Monte-Carlo estimate of the training loss (t ~ Uniform{1..T} unless given)."""
        B, dev, sch = x_num.shape[0], x_num.device, self.sch
        if t is None:
            t = torch.randint(1, sch.T + 1, (B,), generator=generator, device=dev)
        noise = torch.randn(x_num.shape, generator=generator, device=dev)
        xt_num = q_sample_gaussian(x_num, t, noise, sch)                        # forward process, numeric

        x0_oh = [F.one_hot(x_cat[:, j], K).float() for j, K in enumerate(self.cat_sizes)]
        xt_oh = [_draw(q_probs_categorical(x0, t, sch, K), generator)[1]        # forward process, categorical
                 for x0, K in zip(x0_oh, self.cat_sizes)]

        out = self.denoiser(xt_num, torch.cat(xt_oh, 1), t, cond)
        eps_hat, logits = self._split(out)

        l_num = F.mse_loss(eps_hat, noise)                                       # epsilon-prediction
        kls = []
        for x0, xt, lg, K in zip(x0_oh, xt_oh, logits, self.cat_sizes):
            true_post = categorical_posterior(xt, x0, t, sch, K)
            model_post = categorical_posterior(xt, lg.softmax(-1), t, sch, K)
            kls.append(categorical_kl(true_post, model_post))                    # t=1: reduces to -log p(x_0 | x_1)
        l_cat = torch.stack(kls, 1).mean()                                        # mean over columns and batch
        return {"loss": l_num + l_cat, "num": l_num.detach(), "cat": l_cat.detach()}

    @torch.no_grad()
    def sample(self, n: int, cond, generator, clip_x0: float | None = 4.0):
        """Ancestral sampling: start from pure noise / uniform categories and apply p_theta for t = T..1."""
        was_training = self.training
        self.eval()
        sch = self.sch
        dev = next(self.parameters()).device
        x_num = torch.randn((n, self.n_num), generator=generator, device=dev)
        idx0 = [torch.randint(0, K, (n,), generator=generator, device=dev) for K in self.cat_sizes]
        xt_oh = [F.one_hot(i, K).float() for i, K in zip(idx0, self.cat_sizes)]
        cat_idx = idx0
        for step in range(sch.T, 0, -1):
            t = torch.full((n,), step, dtype=torch.long, device=dev)
            eps_hat, logits = self._split(self.denoiser(x_num, torch.cat(xt_oh, 1), t, cond))
            x0_hat = predict_x0(x_num, t, eps_hat, sch)
            if clip_x0 is not None:
                x0_hat = x0_hat.clamp(-clip_x0, clip_x0)
            mean, var = gaussian_posterior(x0_hat, x_num, t, sch)
            x_num = mean + var.sqrt() * torch.randn(x_num.shape, generator=generator, device=dev)
            drawn = [_draw(categorical_posterior(oh, lg.softmax(-1), t, sch, K), generator)
                     for oh, lg, K in zip(xt_oh, logits, self.cat_sizes)]
            cat_idx = [d[0] for d in drawn]
            xt_oh = [d[1] for d in drawn]
        self.train(was_training)
        return x_num, torch.stack(cat_idx, 1)
