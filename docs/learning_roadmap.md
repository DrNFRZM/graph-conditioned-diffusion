# Learning roadmap

A suggested order for someone with ML basics (MSc level) who wants to *own* this repository, not just run it. Each step names something to read, something to run and something to break.

## Week 1 - the diffusion core

1. Read [diffusion_explained.md](diffusion_explained.md) §1-§4 with pen and paper. Re-derive $q(x_t\mid x_0)$ from the one-step kernel, and the posterior mean coefficients by multiplying two Gaussians.
2. Read `Schedule`, `q_sample_gaussian`, `predict_x0`, `gaussian_posterior` in [`diffusion.py`](../src/gcdiff/diffusion.py) next to `tests/test_diffusion.py`. Every test is a claim from the maths.
3. **Break it:** in `Schedule`, change the cosine schedule to a linear one (`beta = torch.linspace(1e-4, 0.02, T)`). Which tests fail? Which don't, and why?
4. **Experiment:** train `diff_uncond` for 20 vs 300 epochs and plot samples' marginal age histogram against the real one.

## Week 2 - categorical diffusion

1. §5 of the explainer. Verify by hand that for $K=2$ the posterior formula gives sensible numbers at $t=1$ and $t=T$.
2. Read `categorical_posterior` and `categorical_kl`; then `test_multinomial_posterior_equals_bayes_from_matrices`.
3. **Break it:** swap the roles of `x_t` and `x_0` inside `categorical_posterior`. Which test catches it?
4. **Experiment:** compare the gender-occupation association in real vs synthetic data for 100 / 300 / 600 epochs (`association_matrix` in `metrics.py`). The research note shows why this matters.

## Week 3 - graphs and conditioning

1. Read `graphs.py` and `models.py`. Draw the tensor shapes: `[n_users, 41]` flat features, hetero graph node/edge dicts, encoder output `[n_users, 32]`.
2. Read PyG's `HeteroConv` and `SAGEConv` documentation (Fey & Lenssen 2019, arXiv:1903.02428; Hamilton et al. 2017, arXiv:1706.02216).
3. **Break it:** add the true gender as a user-node input feature and watch `cond_gender_auc` jump. Then run `tests/test_data_graphs.py::test_no_target_leakage_into_graph_or_features` to see what a passing leakage guard looks like, and write the equivalent failing test for your modification.
4. **Experiment:** change `like_threshold` to 3 or 5; add a third ablation (item features removed).

## Week 4 - evaluation and critical reading

1. Read [design_decisions.md](design_decisions.md), evaluation table. For each metric, write one sentence of what it would look like if the generator were (a) perfect, (b) a memoriser, (c) unconditional.
2. Read Stadler, Oprisanu, Troncoso 2022, *Synthetic Data - Anonymisation Groundhog Day* (USENIX Security), and Carlini et al. 2023, *Extracting Training Data from Diffusion Models* (USENIX Security). Then reread the memorisation section of the research note and list what it does not test.
3. Run the `full` config with 10 seeds (`python -m gcdiff.run --config configs/full.yaml --out results/full`) and compare with `results/quick_10seeds`.
4. Work through [interview_guide.md](interview_guide.md) aloud without looking at the answers.

## Extensions worth trying (in rough order of value)

* **Classifier-free guidance / conditioning dropout** (Ho & Salimans 2022, arXiv:2207.12598) to make the influence of $c$ tunable.
* **Inductive evaluation:** drop test users' edges at training time and add them only at generation time.
* **A second dataset** with a different graph (e.g. a citation network with numeric + categorical node attributes) to test whether any finding transfers.
* **A stronger non-graph baseline:** truncated SVD / matrix-factorisation user embeddings of the rating matrix as the flat condition.
* **A real privacy evaluation:** membership-inference against the generator with a shadow-model attack.

## Books / lectures that pay off

* Murphy, *Probabilistic Machine Learning: Advanced Topics*, for deep generative modelling and score-based methods.
* Hamilton, *Graph Representation Learning* (free book; message passing chapters).
* Lilian Weng's blog post *What are Diffusion Models?* for a second derivation.
