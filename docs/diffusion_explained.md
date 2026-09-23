# Diffusion, explained for this repository

Goal: after reading this you can point at each line of [`src/gcdiff/diffusion.py`](../src/gcdiff/diffusion.py) and say which equation it implements. Prerequisites: Gaussian densities, Bayes' rule, KL divergence, conditional expectation.

Notation: $x_0$ is a clean data row, $x_t$ its noised version at step $t\in\{1,\dots,T\}$, $c$ the conditioning vector, $\epsilon\sim\mathcal N(0,I)$.

---

## 1. The idea in one paragraph

Pick a *fixed* way to destroy data gradually by adding noise (forward process $q$). Train a network to undo one small step of the destruction (reverse process $p_\theta$). To generate, start from pure noise and apply the learned reverse step $T$ times. Nothing about $q$ is learned; all learning is in the reverse step.

## 2. Forward process $q(x_t\mid x_0)$ (numeric columns)

One step adds a little Gaussian noise and shrinks the signal slightly:

$$q(x_t\mid x_{t-1})=\mathcal N\!\big(\sqrt{1-\beta_t}\,x_{t-1},\;\beta_t I\big),\qquad \alpha_t:=1-\beta_t,\qquad \bar\alpha_t:=\prod_{s\le t}\alpha_s .$$

Because a sum of independent Gaussians is Gaussian, composing $t$ steps collapses to a **closed form**:

$$\boxed{\,q(x_t\mid x_0)=\mathcal N\!\big(\sqrt{\bar\alpha_t}\,x_0,\;(1-\bar\alpha_t)I\big)\;\Longleftrightarrow\; x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon\,}$$

Why this matters for training: you never simulate $t$ steps. You draw a random $t$, one $\epsilon$, and get $x_t$ directly. In code: `q_sample_gaussian`. The test `test_gaussian_forward_marginal_and_x0_inversion` checks the mean and variance of this formula empirically.

Sanity picture: at $t$ small, $\bar\alpha_t\approx1$, $x_t\approx x_0$. At $t=T$, $\bar\alpha_T\approx0$, $x_T\approx\epsilon$ — pure noise, which is why sampling can *start* from $\mathcal N(0,I)$.

## 3. Noise schedule

The schedule is the sequence $\beta_1,\dots,\beta_T$ (equivalently $\bar\alpha_t$). It decides how much of the $t$-axis is spent at "mostly signal" versus "mostly noise". This repo uses the **cosine schedule** of Nichol & Dhariwal (2021), which retains signal more gradually than a common linear schedule. No schedule sweep was performed:

$$\bar\alpha_t=\frac{f(t)}{f(0)},\quad f(t)=\cos^2\!\Big(\frac{t/T+s}{1+s}\cdot\frac\pi2\Big),\quad s=0.008,$$

with $\beta_t=1-\bar\alpha_t/\bar\alpha_{t-1}$ clipped at 0.999. See `Schedule`. Index 0 is defined as clean data ($\bar\alpha_0=1$); steps are $1..T$ everywhere (a common off-by-one bug source, so the tests check $\bar\alpha_0=1$ and that the last reverse step adds no noise).

## 4. What the network predicts: $\epsilon$

We want the reverse step $p_\theta(x_{t-1}\mid x_t)$. If we knew $x_0$, the *true* reverse step is a Gaussian (Bayes' rule on two Gaussians):

$$q(x_{t-1}\mid x_t,x_0)=\mathcal N\!\big(\tilde\mu_t(x_t,x_0),\;\tilde\beta_t I\big),\quad
\tilde\mu_t=\underbrace{\tfrac{\beta_t\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_t}}_{\texttt{coef\_x0}}x_0+\underbrace{\tfrac{(1-\bar\alpha_{t-1})\sqrt{\alpha_t}}{1-\bar\alpha_t}}_{\texttt{coef\_xt}}x_t,\quad
\tilde\beta_t=\tfrac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t .$$

(`test_gaussian_posterior_matches_conjugate_formula` re-derives this by multiplying the two Gaussians.) We do not know $x_0$ at sampling time, but $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$ can be solved for $x_0$ if we can guess the noise:

$$\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon_\theta(x_t,t,c)}{\sqrt{\bar\alpha_t}}\quad(\texttt{predict\_x0}).$$

So **predicting $\epsilon$ is equivalent to predicting $x_0$**, just a better-conditioned target (unit variance at every $t$). Plug $\hat x_0$ into $\tilde\mu_t$ and you have the model's reverse mean.

**Training loss.** Sample $t\sim\mathrm{Unif}\{1..T\}$, $\epsilon\sim\mathcal N(0,I)$, form $x_t$, and minimise

$$\mathcal L_{\text{num}}=\mathbb E_{x_0,t,\epsilon}\,\big\|\epsilon-\hat\epsilon_\theta(x_t,t,c)\big\|^2 .$$

What does this optimise? The minimiser is $\hat\epsilon^\star(x_t,t,c)=\mathbb E[\epsilon\mid x_t,c]$. Since $\nabla_{x_t}\log q(x_t\mid x_0)=-\epsilon/\sqrt{1-\bar\alpha_t}$, this is (up to scale) the **score** of the noised data distribution: $\hat\epsilon^\star=-\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log p_t(x_t\mid c)$. Following the learned score field from noise back to the data is exactly what ancestral sampling does. The un-weighted MSE is a re-weighted version of the variational lower bound (Ho et al. 2020), which is why it is a valid density-modelling objective, not just a heuristic.

## 5. Categorical columns: multinomial diffusion

Gaussian noise makes no sense for a category such as *occupation*. Hoogeboom et al. (2021) replace it with a "keep or resample uniformly" corruption on the one-hot vector $x_0\in\{0,1\}^K$:

$$q(x_t\mid x_{t-1})=\mathrm{Cat}\big(\alpha_t x_{t-1}+(1-\alpha_t)/K\big),\qquad
q(x_t\mid x_0)=\mathrm{Cat}\big(\bar\alpha_t x_0+(1-\bar\alpha_t)/K\big).$$

Interpretation: with probability $\bar\alpha_t$ the category survived, otherwise it has been replaced by a uniformly random one. At $t=T$ the category is uniform noise — the counterpart of $\mathcal N(0,I)$. (`test_multinomial_forward_is_chapman_kolmogorov_consistent` multiplies the one-step kernels and checks they equal the closed form.)

Bayes' rule again gives the true reverse step

$$q(x_{t-1}\mid x_t,x_0)\;\propto\;\underbrace{\big[\alpha_t x_t+(1-\alpha_t)/K\big]}_{q(x_t\mid x_{t-1})\text{ as a function of }x_{t-1}}\;\odot\;\underbrace{\big[\bar\alpha_{t-1}x_0+(1-\bar\alpha_{t-1})/K\big]}_{q(x_{t-1}\mid x_0)}$$

(`test_multinomial_posterior_equals_bayes_from_matrices` verifies this against explicit transition matrices). Here the network outputs logits, $\hat x_0=\mathrm{softmax}(\text{logits})$, and the **model's** reverse step is the same formula with $\hat x_0$ (a probability vector) in place of $x_0$. The loss is the KL between the true and model reverse steps:

$$\mathcal L_{\text{cat}}=\mathbb E\;\mathrm{KL}\big(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t,c)\big)\quad(\text{mean over columns}).$$

At $t=1$, $\bar\alpha_0=1$ so the true posterior is one-hot on $x_0$ and the KL collapses to $-\log p_\theta(x_0\mid x_1)$, a cross-entropy — no special case needed in code.

**Total loss** $=\mathcal L_{\text{num}}+\mathcal L_{\text{cat}}$ (as in TabDDPM). The two terms are on different scales, so their relative weight is a modelling choice, not a law; this is recorded in [design_decisions.md](design_decisions.md).

## 6. Reverse process / sampling

`MixedDiffusion.sample` does, for $t=T,\dots,1$:

1. Numeric: $\hat\epsilon\to\hat x_0$ (clipped to $\pm4$ std for stability) $\to\tilde\mu_t$, then $x_{t-1}=\tilde\mu_t+\sqrt{\tilde\beta_t}\,z$, $z\sim\mathcal N(0,I)$. At $t=1$, $\tilde\beta_1=0$: the last step is deterministic.
2. Categorical: compute the model reverse step (formula in §5), **draw** a category.

Intuition: early steps (large $t$) decide coarse structure (which occupation cluster; roughly what age range), late steps refine details. Because every step is stochastic, repeated sampling with the same $c$ gives different rows — that is the conditional distribution $p(x\mid c)$, not a point prediction.

One limitation to keep in mind: within one reverse step, columns are sampled independently *given $x_t$*. Dependence between columns is created over many steps through the network's joint input, and it needs enough training to be learned well (see the diagnostics in the research note).

## 7. Conditional information: what $c$ does

Nothing in $q$ depends on $c$: we corrupt $x_0$ the same way regardless. Only the reverse network sees it: $\hat\epsilon_\theta(x_t,t,c)$ and the categorical logits. Mechanically ([`models.py`](../src/gcdiff/models.py)): $t$ and $c$ are concatenated and mapped by an MLP to an embedding, which produces a per-block scale and shift of the hidden activations (**FiLM**, Perez et al. 2018).

What is being learned statistically: the model of the *conditional* density $p(x_0\mid c)$. Unconditional NLL is $H(x_0)$; a perfect conditional model achieves $H(x_0\mid c)=H(x_0)-I(x_0;c)$. So the achievable improvement in (variational) likelihood from conditioning is bounded by the mutual information between the conditioning vector and the target. If $c$ carries no information about $x_0$, the optimal conditional network ignores it and equals the unconditional one. This is the theoretical basis for the falsifiable claim in the research note: *if graph embeddings do not carry information about the attributes beyond what non-graph features do, graph conditioning cannot help.*

## 8. What the GNN embedding changes

$c_u=g_\phi(\mathcal G)_u$ is the output of a graph neural network evaluated at user $u$'s node. Training is **end-to-end**: gradients of $\mathcal L$ flow through the denoiser into $\phi$, so the GNN is pushed to produce whatever summary of $u$'s neighbourhood best reduces the denoising loss of $u$'s attributes. There is no separate pre-training objective.

Compared with a flat feature vector $b_u$ (the non-graph condition, `diff_flat`):

* the GNN can aggregate over neighbours' features with *learned* weights, and at depth 2 it sees the other users who rated the same items;
* it can share statistical strength between users through shared item representations;
* it also has more freedom to overfit: with ~570 training users, an encoder can memorise user-specific patterns. This is why the repo early-stops on a validation loss, checks a per-user memorisation gap, and includes a *user-degree-preserving endpoint shuffle* as a structural control.

What it cannot do: use the target. User nodes carry only label-free structural inputs (log degree per relation); demographics are never node features, edge features or neighbours' features. `tests/test_data_graphs.py::test_no_target_leakage_into_graph_or_features` builds two worlds with identical ratings but different demographics and asserts identical conditioning inputs.

## 9. Reading list (verified references)

* Sohl-Dickstein et al. 2015, *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*, arXiv:1503.03585 — the original diffusion framework.
* Ho, Jain, Abbeel 2020, *Denoising Diffusion Probabilistic Models*, arXiv:2006.11239 — $\epsilon$-prediction and the simple loss.
* Nichol & Dhariwal 2021, *Improved Denoising Diffusion Probabilistic Models*, arXiv:2102.09672 — cosine schedule.
* Hoogeboom et al. 2021, *Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions*, arXiv:2102.05379.
* Kotelnikov et al. 2023, *TabDDPM: Modelling Tabular Data with Diffusion Models*, arXiv:2209.15421 — Gaussian + multinomial diffusion for tabular data.
* Perez et al. 2018, *FiLM: Visual Reasoning with a General Conditioning Layer*, arXiv:1709.07871.
* Ho & Salimans 2022, *Classifier-Free Diffusion Guidance*, arXiv:2207.12598 — not implemented here; the natural next step for stronger conditioning.
