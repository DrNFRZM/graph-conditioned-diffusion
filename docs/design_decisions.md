# Design decisions and threats to validity

Each row: what was chosen, what else was possible, why. Written so a reviewer can disagree with a specific line.

## Data and task

| Decision | Alternatives | Reason |
|---|---|---|
| **MovieLens 100K** (GroupLens; 943 users, 1,682 items, 100,000 ratings). Downloaded at run time from `files.grouplens.org`, SHA-256 checked, never committed. | Adult/Census (graph would be artificial); Cora/Planetoid (one modality); MovieLens 1M (age is binned) | Public, no credentials, ~5 MB. The user-item graph is *observed* (who rated what), not constructed from the target. Users have exact numeric age plus categorical gender / occupation / zip; items have genres and year. |
| **Generation target = user profile** (age; gender 2; occupation 21; zip region 11). **Condition = the user's position in the rating graph.** | Generate the ratings themselves; generate the graph | Keeps the model small and the question clean: does graph context change the distribution of node attributes? The target is mixed-type by construction. |
| Only **one numeric column** (age). | Add derived numerics (mean rating, degree) | Derived numerics are deterministic functions of the edges; conditioning on the graph would then "predict" them trivially and blur the leakage argument. The code supports any number of numeric columns. |
| `zip_region` = first character of zip if it is a digit, else `other`. | Full zip | 795 distinct zips in 943 users is not modelable. The first digit is a coarse US region. It is mostly noise with respect to behaviour, and that is fine: it tests that the model does not invent structure. |
| Node-level **60/15/25 split of users**, stratified by gender, redrawn per seed. | Edge-level split; single fixed split | The generated object is a user, so the unit of splitting is a user. Val is for early stopping only; test is touched only for reporting. Different seeds give different splits, so the seed spread includes split variance. |

## Graph

| Decision | Alternatives | Reason |
|---|---|---|
| **Heterogeneous bipartite graph** with relations `likes` (rating ≥ 4) and `dislikes` (≤ 3), both directions. | One undirected edge type with rating as edge weight | Polarity is the informative part of a rating; separate relations let a relational SAGE treat it explicitly. Threshold 4 is a convention, not tuned. |
| **User nodes carry only log-degree per relation.** Item nodes: genre multi-hot + standardised release year. | Learned ID embeddings per user | A free per-user embedding can memorise the user's attributes through the end-to-end loss, which is target leakage by another route. |
| **Transductive message passing**: all users' ratings are in the graph at train time; only *attributes* of train users enter the loss. | Inductive: train graph without test users' edges | Test users' *behaviour* is legitimately observable at generation time (that is the conditioning input); their *attributes* are never seen. Item features are standardised over all items (no user attributes involved). User-feature scalers are fit on train users only. |
| **Ablation graphs**: (i) user-degree-preserving endpoint shuffle, (ii) user-user kNN graph. "No edges" = remove conditioning = method B. | Feature-only GNN with self loops | The shuffle keeps each user's degree and rating multiset and the global item-degree sequence, but randomises *which* item endpoints each user receives. It is a useful activity-preserving control, not a perfectly simple graph: relation-specific item degrees are not fixed and parallel user-item endpoints can occur. kNN changes construction (homogeneous, similarity-based) while carrying the same information as the flat baseline in node features. |

## Model

| Decision | Alternatives | Reason |
|---|---|---|
| **Gaussian DDPM (ε-prediction) for numeric + multinomial diffusion (x₀-parameterised, KL loss) for categoricals**, hand-written. | One-hot relaxed into Gaussian noise and argmax-decoded; a diffusion library | Categories have no metric structure; multinomial diffusion is the principled counterpart. Writing it by hand is the point of the repo. Tests check the posterior against explicit Bayes. |
| **Loss = MSE + mean categorical KL, unit weights** (as TabDDPM). | Weighted / learned weights | A convention; weighting affects the trade-off between age fidelity and category fidelity. Not tuned. |
| Cosine schedule, T = 100 (quick) / 250 (full). | Linear | Better use of steps for low-dimensional data (Nichol & Dhariwal 2021). |
| **FiLM conditioning** on an MLP over `[t-embedding, c]`. | Concatenate c to input; cross-attention | Simple, standard, identical for every conditioned method, so differences come from the embedding, not the mechanism. |
| Encoders end in a parameter-free LayerNorm; `cond_dim = 32` for flat and GNN. | Unnormalised outputs | Removes trivial scale differences between encoders. Parameter counts are recorded in `metrics_per_seed.csv` (`n_params`). |
| **No EMA, no classifier-free guidance, no conditioning dropout.** | Add them | Each would add a hyper-parameter and a comparison confound. Listed as future work. |
| Early stopping on a **deterministic** validation loss (fixed noise, timesteps stratified over 1..T). | Fixed epochs | Prevents the GNN/flat encoders from memorising 570 rows; deterministic so curves are comparable. Note that this loss is a re-weighted bound, not a proper NLL. |
| x̂₀ clipped to ±4 (standardised units) while sampling; ages rounded and clipped to the training range. | No clipping | Standard stabilisation. The clipping is an assumption: an out-of-range age can never be generated. |

## Evaluation (what each number can and cannot say)

| Metric | Shows | Does **not** show |
|---|---|---|
| KS / TV marginals | one-column distribution match | dependence |
| Association error (|Pearson|, η, bias-corrected Cramér's V) | pairwise dependence between attribute columns | higher-order structure, any relation to the graph |
| TSTR on the attribute table (gender from age/occupation/zip), with age scaled from the real training partition | attribute→attribute structure transfers to real users | value of the graph |
| **Held-out denoising loss** (test users, fixed noise, paired across methods) | whether the conditioning vector reduces the training objective on unseen users | a proper likelihood; sample quality |
| **Conditional consistency** (AUC / R² / accuracy of S draws per user vs that user's real attributes) | draws given user *i*'s graph carry information about user *i*'s real attributes | marginal fidelity; causation |
| **TSTR-node**: train a probe on synthetic labels of train users, test on real test users | whether synthetic labels contain graph-dependent signal that transfers | that the generator is faithful in other respects |
| Real-label probe (logistic / ridge on the flat features) | a simple, untuned, non-generative reference for what a linear model on the flat features predicts | that it is an upper bound (it is not: it is a regularised linear model, and it can be beaten) |
| Homophily gap (same-gender rate, age assortativity over a rating-based kNN graph among test users) | attribute arrangement over the graph resembles real | — (secondary; noisy at 236 nodes) |
| DCR ratio, exact-match rate, per-user train-vs-test gap | absence of gross copying / per-user memorisation | **privacy**. No membership-inference attack, no differential privacy. Low-cardinality columns make coincidental exact matches common. |

## Threats to validity (read before believing any table)

1. **Small n.** 943 users; ~236 test users per seed. Gender AUC has a null standard deviation of roughly 0.04 per seed. The reported runs use 10 seeds. Seeds re-draw the split and initialisation but reuse the same 943 users, so seeds are **not independent samples of the population**; the across-seed std understates uncertainty about new users or datasets.
2. **Weak target signal.** A simple linear probe on real labels reaches only modest AUC / R² from behaviour features (see the research note); the information available to any generator here is limited.
3. **Same information in flat and GNN conditions at depth 1.** The bipartite GNN's extra information is the 2-hop neighbourhood and learned aggregation. A null difference between C and D is therefore an expected outcome, not a failure of the code.
4. **Homogeneous evaluation graph.** The homophily metric uses a rating-based kNN graph, which is related to (not identical with) the conditioning graphs.
5. **The final configuration was not chosen blind.** It is shared across methods and was not tuned per method, but the development seed's test outputs were visible when training length, patience and draw count were adjusted. The reported run therefore uses disjoint evaluation seeds and is framed as exploratory; it is not a preregistered confirmatory test.
6. **Single dataset.** Nothing here generalises beyond MovieLens 100K without replication.
7. **Not a privacy study.** See the metric table.
8. **Unit loss weights and per-step KL.** Sampling one timestep per example and averaging the categorical KL (rather than summing it over the T steps of the bound) makes the categorical term small next to the epsilon-MSE. In the reported runs the categorical part of the held-out loss is about 0.018 for every method while the numeric part is about 0.45–0.48, so end-to-end training of the encoder is driven mainly by age. This is a likely reason no method recovers gender information (see research note). It was **not** tuned away.
9. **Early stopping picks different epochs per method** (mean best epoch: 97 unconditional, 34.5 flat, 72.5 bipartite GNN, 86 shuffled GNN and 18 kNN GNN). Methods therefore see different amounts of training, which can affect columns that are learned slowly (categorical dependencies), e.g. association error.
10. **Encoder capacities are not exactly matched.** In the quick configuration the GNN-conditioned model has 255,683 trainable parameters and the flat-conditioned model 241,987 (about 6% fewer). Equal conditioning width does not remove this capacity difference, so a small GNN advantage could not be attributed to graph structure alone.
