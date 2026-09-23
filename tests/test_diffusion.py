import math

import pytest
import torch
import torch.nn.functional as F

from gcdiff.diffusion import (MixedDiffusion, Schedule, categorical_kl, categorical_posterior, gaussian_posterior,
                              predict_x0, q_probs_categorical, q_sample_gaussian)
from gcdiff.models import Denoiser


def test_schedule_shapes_and_monotone():
    s = Schedule(50)
    assert s.alpha_bar.shape == (51,) and s.alpha_bar[0] == 1
    assert (s.alpha_bar[1:] < s.alpha_bar[:-1]).all()
    assert s.alpha_bar[-1] < 1e-3                       # ~pure noise at t = T
    assert s.post_var[1] == 0                           # no noise added on the last reverse step


def test_gaussian_forward_marginal_and_x0_inversion():
    s, g = Schedule(100), torch.Generator().manual_seed(0)
    x0 = torch.full((20000, 1), 2.0)
    t = torch.full((20000,), 40)
    noise = torch.randn(x0.shape, generator=g)
    xt = q_sample_gaussian(x0, t, noise, s)
    ab = s.alpha_bar[40].item()
    assert xt.mean().item() == pytest.approx(math.sqrt(ab) * 2.0, abs=0.03)
    assert xt.var().item() == pytest.approx(1 - ab, abs=0.03)
    assert torch.allclose(predict_x0(xt, t, noise, s), x0, atol=1e-4)   # exact inverse given the true noise


def test_gaussian_posterior_matches_conjugate_formula():
    s = Schedule(30)
    for t in (2, 10, 30):
        x0, xt = torch.tensor([[0.7]]), torch.tensor([[-0.4]])
        mean, var = gaussian_posterior(x0, xt, torch.tensor([t]), s)
        ab_prev, a, b = s.alpha_bar[t - 1].item(), s.alpha[t].item(), s.beta[t].item()
        # N(x_{t-1}; sqrt(ab_prev) x0, 1-ab_prev) * N(x_t; sqrt(a) x_{t-1}, b)  -> Gaussian in x_{t-1}
        prec = 1 / (1 - ab_prev) + a / b
        m = (math.sqrt(ab_prev) * 0.7 / (1 - ab_prev) + math.sqrt(a) * -0.4 / b) / prec
        assert mean.item() == pytest.approx(m, rel=1e-3) and var.item() == pytest.approx(1 / prec, rel=1e-3)
    m1, v1 = gaussian_posterior(torch.tensor([[0.7]]), torch.tensor([[5.0]]), torch.tensor([1]), s)
    assert m1.item() == pytest.approx(0.7, abs=1e-5) and v1.item() == 0   # t=1 returns x0


def _Q(alpha, K):
    return alpha * torch.eye(K) + (1 - alpha) / K * torch.ones(K, K)


def test_multinomial_forward_is_chapman_kolmogorov_consistent():
    s, K = Schedule(20), 5
    Q = torch.eye(K)
    for t in range(1, 8):
        Q = Q @ _Q(s.alpha[t].item(), K)                       # product of one-step kernels
    x0 = F.one_hot(torch.tensor([2]), K).float()
    closed = q_probs_categorical(x0, torch.tensor([7]), s, K)
    assert torch.allclose(x0 @ Q, closed, atol=1e-5)


def test_multinomial_posterior_equals_bayes_from_matrices():
    s, K, t = Schedule(20), 4, 6
    x0_i, xt_k = 1, 3
    Qprev = torch.eye(K)
    for u in range(1, t):
        Qprev = Qprev @ _Q(s.alpha[u].item(), K)
    prior = Qprev[x0_i]                                        # q(x_{t-1} | x0)
    lik = _Q(s.alpha[t].item(), K)[:, xt_k]                    # q(x_t = k | x_{t-1} = j)
    bayes = prior * lik / (prior * lik).sum()
    got = categorical_posterior(F.one_hot(torch.tensor([xt_k]), K).float(), F.one_hot(torch.tensor([x0_i]), K).float(),
                                torch.tensor([t]), s, K)[0]
    assert torch.allclose(got, bayes, atol=1e-5)


def test_categorical_kl_properties_and_t1_is_nll():
    s, K = Schedule(20), 4
    x0 = F.one_hot(torch.tensor([1, 2]), K).float()
    xt = F.one_hot(torch.tensor([0, 2]), K).float()
    t = torch.tensor([5, 9])
    p = categorical_posterior(xt, x0, t, s, K)
    assert torch.allclose(categorical_kl(p, p), torch.zeros(2), atol=1e-6)      # KL(p||p) = 0
    q = categorical_posterior(xt, torch.full((2, K), 1 / K), t, s, K)
    assert (categorical_kl(p, q) >= 0).all()
    # t=1: the true posterior is one-hot(x0), so KL = -log p_theta(x0 | x1), with
    # p_theta(j | x1) proportional to q(x1 | j) * p_hat(j)
    probs = torch.tensor([[0.1, 0.6, 0.2, 0.1]])
    p1 = categorical_posterior(xt[:1], x0[:1], torch.tensor([1]), s, K)
    q1 = categorical_posterior(xt[:1], probs, torch.tensor([1]), s, K)
    fact1 = s.alpha[1] * xt[:1] + (1 - s.alpha[1]) / K
    expected = -torch.log(fact1[0, 1] * probs[0, 1] / (fact1 * probs).sum()).item()
    assert categorical_kl(p1, q1).item() == pytest.approx(expected, rel=1e-3)


def _model(cond_dim=0, T=10):
    den = Denoiser(2, [3, 4], cond_dim=cond_dim, width=16, depth=2, t_dim=8)
    return MixedDiffusion(den, Schedule(T), 2, [3, 4])


def test_loss_and_sampling_shapes_ranges_and_determinism():
    m = _model(cond_dim=6)
    x_num = torch.randn(9, 2)
    x_cat = torch.stack([torch.randint(0, 3, (9,)), torch.randint(0, 4, (9,))], 1)
    cond = torch.randn(9, 6)
    out = m.loss(x_num, x_cat, cond, torch.Generator().manual_seed(0))
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    out["loss"].backward()
    a = m.sample(7, cond[:7], torch.Generator().manual_seed(1))
    b = m.sample(7, cond[:7], torch.Generator().manual_seed(1))
    assert a[0].shape == (7, 2) and a[1].shape == (7, 2) and torch.isfinite(a[0]).all()
    assert (a[1][:, 0] < 3).all() and (a[1][:, 1] < 4).all() and (a[1] >= 0).all()
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_condition_is_used_and_dimension_is_checked():
    m = _model(cond_dim=6).eval()
    x, oh, t = torch.randn(5, 2), torch.zeros(5, 7), torch.full((5,), 3)
    assert not torch.allclose(m.denoiser(x, oh, t, torch.randn(5, 6)), m.denoiser(x, oh, t, torch.randn(5, 6)))
    with pytest.raises(ValueError):
        m.denoiser(x, oh, t, torch.randn(5, 5))
    with pytest.raises(ValueError):
        m.denoiser(x, oh, t, None)


def test_optimisation_reduces_loss_and_learns_deterministic_categories():
    torch.manual_seed(0)
    m = _model(T=20)
    x_num = torch.randn(128, 2) * 0.3 + 1.0
    x_cat = torch.stack([torch.zeros(128, dtype=torch.long), torch.ones(128, dtype=torch.long)], 1)
    opt, g = torch.optim.Adam(m.parameters(), 3e-3), torch.Generator().manual_seed(0)
    first = last = None
    for i in range(300):
        l = m.loss(x_num, x_cat, None, g)["loss"]
        opt.zero_grad(); l.backward(); opt.step()
        first = l.item() if i == 0 else first
        last = l.item()
    assert last < 0.6 * first
    _, cs = m.sample(300, None, torch.Generator().manual_seed(0))
    assert (cs[:, 0] == 0).float().mean() > 0.9 and (cs[:, 1] == 1).float().mean() > 0.9
