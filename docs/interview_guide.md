# Interview guide

Seventeen hard questions with model answers. Numbers refer to the executed 10-seed run in `results/quick_10seeds/` (details in [research_note.md](research_note.md)); ± denotes the sample standard deviation across seeds, not a confidence interval. Practise answering aloud; the strongest answers state a limitation before the interviewer finds it.

---

## A. Motivation and method choice

**1. Why diffusion rather than a VAE or a GAN?**
For this repo the honest answer is partly pedagogical and partly technical. Diffusion gives a direct, stable denoising objective, supports explicit Gaussian and multinomial corruption processes, and makes the conditioning path easy to inspect. TabDDPM provides precedent for this mixed-type choice. Against: sampling needs T network calls, and a carefully specified VAE or GAN could be competitive on only about 566 training users. I did **not** run either, so I make no claim that diffusion is superior. The comparison that matters here is controlled conditioning—unconditional, flat and graph—not a generative-family benchmark.

**2. Why graph conditioning at all? What is the argument?**
Ratings relate users to items. A flattened vector loses which specific items a user interacted with and, at depth two, who else interacted with them. A GNN aggregates neighbourhoods with learned weights and shares statistical strength across users through shared item representations. The testable claim: the resulting embedding carries information about a user's attributes that a flattened summary does not. The result: graph conditioning clearly helps age relative to no conditioning, but I found no reliable advantage over a strong flat summary. So the motivation is only partly borne out.

**3. How is the graph constructed, and why is that defensible?**
Nodes: 943 users, 1,682 items. Edges: each rating becomes a `likes` edge (rating ≥ 4) or a `dislikes` edge (≤ 3) in both directions, giving four relation types. Item features: 19 genre flags and standardised release year. User features: log-degree per relation only. The graph is *observed* (who rated what), not derived from the attributes being generated, which is what makes it defensible. The threshold 4 is a convention I did not tune; a sensitivity check on 3 or 5 is a natural extension. Two alternative constructions are in the ablations: a user-degree-preserving endpoint shuffle and a user-user kNN graph.

**4. Could the graph leak target information?**
Three routes were considered. (a) *Direct*: demographics as node features: not used; a test builds two worlds with identical ratings and different demographics and asserts identical conditioning inputs. (b) *Learned ID embeddings*: a free per-user embedding trained end-to-end could memorise the target; user nodes therefore carry no ID, only log-degrees. (c) *Transductive use of test users' edges*: their ratings are in the graph at training time. That is not target-label leakage (their attributes are never seen), but it does make the setting transductive: test users' behaviour influences item representations that train users' losses depend on. An inductive protocol is future work. The train-versus-test per-user age-error gaps have wide seed spreads and do not resolve memorisation; early stopping on validation users is still a useful guard.

**5. What does the GNN actually learn?**
It is trained only through the diffusion loss, so it learns the summary of a user's neighbourhood that best helps denoise that user's attributes. In practice the loss is dominated by the numeric term (about 0.45–0.48 vs 0.018 for the categorical KL), so it learns age-relevant structure. I did not inspect the embedding (e.g. probing it for genre-taste directions), so I cannot claim it learned "taste profiles." The endpoint-shuffle ablation shows sensitivity to which item endpoints users receive beyond the preserved activity statistics; it does not identify a specific learned concept.

**6. What does the diffusion loss optimise?**
For numeric columns, MSE between true and predicted noise, whose minimiser is E[ε | x_t, c]; equivalently (up to scale) the score of the noised conditional distribution, and a re-weighted variational bound on −log p(x | c). For categorical columns, the KL between the true posterior q(x_{t−1}|x_t,x_0) and the model's reverse step, i.e. the corresponding term of the multinomial-diffusion bound. Two caveats I would volunteer: the total loss adds two terms of different scale with unit weights, and sampling one t per example makes the categorical term a 1/T-scaled estimate of its ELBO contribution, which under-weights categorical learning relative to the numeric part.

## B. Evaluation validity

**7. Why are these metrics valid? What does each fail to show?**
Each metric answers a different question and I state what it does *not* show in [design_decisions.md](design_decisions.md). Short version: KS/TV tests marginals only; association error tests pairwise dependence only (the resampled-real floor is 0.071 ± 0.021); TSTR-table tests attribute-to-attribute transfer; held-out denoising loss is a re-weighted bound, not a proper NLL, but it is paired across methods with fixed noise; *conditional consistency* (AUC/R² of draws for user *i* against user *i*'s real attributes) tests whether draws depend usefully on the condition; TSTR-node tests transfer of synthetic labels to a real prediction task. Baselines A1 and A2 check metric behaviour: A1 has association error 0.206, while A2 has DCR ratio 0 and exact-match 1 for direct copying.

**8. Why is the "age R²" of the conditional draws credible? Isn't it just a regression in disguise?**
It is a regression readout of a generative model: the mean of 32 draws per user is a Monte-Carlo estimate of E[age | graph]. That is a fair way to ask "does p(x | c) depend on the graph in the right direction", and it is compared to the train-mean baseline (R² = 0). It does *not* show the draws' *variance* is right; marginal KS and association metrics address that separately. It also doesn't show causality, only predictive dependence.

**9. Your GNN-versus-flat comparison is a tie. What does a tie mean?**
It is a mixed comparison rather than a literal tie. GNN minus flat age R² is −0.034 ± 0.075 (5/10 seeds), and TSTR-node age R² favours flat in 7/10. GNN instead has lower held-out loss (−0.018 ± 0.013) and association error in 10/10 seeds. Among 55 paired comparisons in the CSV (correlated, uncorrected), I would not select whichever metric makes one method look best. At this scale, graph structure adds no demonstrated overall value *beyond* one-hop summaries. It could still help with more users, richer features or multi-hop structure that the flat vector cannot express.

**10. The seeds share the same 943 users. Are your standard deviations meaningful?**
They measure sensitivity to split and initialisation, not sampling variability of users. Different seeds overlap heavily in which users appear in train or test, so the runs are positively correlated and the across-seed std understates uncertainty about new data. Paired differences within a seed are more informative than the marginal stds, and I report both. A stronger design would use a bootstrap over users or an independent dataset.

**11. Why did gender not work, and why should I trust that it is a failure of the model, not of the signal?**
A linear probe on the same flat features gets gender AUC 0.706, so the signal exists in the conditioning inputs; every diffusion variant gets 0.478–0.490. The failure is in how the generator uses the information. My hypothesis, *untested*: unit loss weights and per-step KL make the categorical loss about 25 times smaller than the numeric one, so the end-to-end encoder is optimised mainly for age. Falsifiable follow-up: up-weight the categorical term (or train the encoder with an auxiliary classification loss) and see whether gender AUC rises; if it does not, the hypothesis is wrong.

**12. What would falsify the central hypothesis?**
Two levels. H1 asks whether structure beyond the preserved activity statistics helps: it is challenged if the endpoint-shuffled control performs as well as the observed graph. The control is not perfect—it preserves user rating multisets and global item degrees but not every relation-specific statistic—so I would not call it a proof of structural causality. H2 asks whether graph conditioning beats equally informed non-graph conditioning: it is falsified if, with adequate power, the GNN is no better than a strong flat or matrix-factorisation baseline on held-out loss and node-level utility. The current evidence does not support H2.

## C. Diffusion mathematics and implementation

**13. Why predict ε rather than x₀ or the mean?**
They are reparameterisations: x̂₀ = (x_t − √(1−ᾱ_t) ε̂)/√ᾱ_t. ε has unit variance at every t, so the regression target is well scaled, whereas x₀-prediction is trivial at small t and very noisy at large t. For categoricals I use x₀-parameterisation instead (the network outputs a distribution over the clean category), because there is no additive-noise interpretation and it makes the posterior in closed form. Both choices follow Ho et al. 2020 and Hoogeboom et al. 2021.

**14. How does multinomial diffusion work, and why not just add Gaussian noise to one-hot vectors?**
The forward kernel keeps the category with probability α_t or resamples uniformly, so q(x_t|x₀) = Cat(ᾱ_t x₀ + (1−ᾱ_t)/K); at t = T the category is uniform noise. The reverse posterior is Bayes' rule in closed form: ∝ [α_t x_t + (1−α_t)/K] ⊙ [ᾱ_{t−1} x₀ + (1−ᾱ_{t−1})/K], verified in a unit test against explicit transition matrices. Gaussian-on-one-hot with argmax decoding is simpler and often works, but it treats categories as points in Euclidean space and has no likelihood interpretation on the discrete variable. Cost of my choice: within a reverse step, columns are sampled independently given x_t; my diagnostic showed cross-column dependence (gender-occupation Cramér's V 0.075 after early stopping vs 0.342 real) is learned slowly.

**15. Why the cosine schedule, and how would you notice a bad schedule?**
The cosine schedule retains signal more gradually than a common linear schedule (Nichol & Dhariwal 2021). Symptoms of a poor schedule include x_T not being close to N(0, I), or useful timesteps being concentrated in a narrow range. A test checks ᾱ_T < 10⁻³. I did not compare schedules and make no claim that cosine is optimal here.

**16. What does the conditioning actually change mathematically?**
Nothing in q(x_t | x₀); only the reverse network sees c. The optimum is ε̂*(x_t, t, c) = E[ε | x_t, c], the score of p_t(x_t | c). A perfect conditional model reaches H(x|c) = H(x) − I(x; c) nats, so the improvement from conditioning is bounded by the mutual information between the embedding and the target. If an embedding is independent of x, the optimal conditional and unconditional networks coincide. The shuffled control does not guarantee independence because it retains activity statistics; it tests whether the observed user–item endpoint arrangement adds information beyond those retained statistics.

## D. Robustness, privacy, next steps

**17. Does your memorisation check show the synthetic data are private?**
No. For the GNN, the DCR ratio is 1.044 ± 0.127 and the exact-match rate is 0.108 ± 0.016 versus 0.136 ± 0.015 for genuinely new users. A2 shows the check detects direct copying (ratio 0, exact-match 1). But no membership-inference attack was run; exact matches are common by chance because the columns are low-cardinality; and the DCR metric covers only four coarse columns. Differences from training rows are not privacy (Stadler et al. 2022; Carlini et al. 2023). The claim is limited to "no evidence of gross copying."

---

### Bonus follow-ups a strong interviewer might add

* *Why transductive?* Behaviour of new users is observable at generation time in the intended use; inductive evaluation is a straightforward extension, but the encoder's neighbours-of-neighbours structure makes it non-trivial (test users' edges influence item embeddings).
* *Why is there only one numeric column?* Derived numerics (degree, mean rating) would be deterministic functions of the graph, making conditioning trivially informative and blurring the leakage argument. The code supports several.
* *What would you do with 10× compute?* Classifier-free guidance with conditioning dropout; per-column loss weights; inductive protocol; stronger non-graph baseline (SVD / matrix factorisation); a second dataset; a membership-inference evaluation.
* *If the GNN had won, would you believe it?* Only after checking that the flat baseline had comparable capacity and tuning effort, that the shuffle ablation removed the effect, and that it held on more seeds and a second dataset.
* *Are the encoder capacities matched?* Not exactly. The quick GNN-conditioned model has 255,683 trainable parameters and the flat-conditioned model 241,987, about a 6% difference. That is small but non-zero, so a marginal GNN advantage would need a parameter-matched control.
* *Was the experiment preregistered?* No. A seed-0 development run informed training length, patience and draw count, and its outputs were visible. The final ten seeds are disjoint (1000–1009), all methods share one configuration, and the limitation is disclosed; the study is still exploratory rather than confirmatory.
