"""One-off diagnostic (single seed, unconditional model): how much of the real gender-occupation dependence
does the diffusion model reproduce after 200 (early-stopped) vs 600 epochs?  Run from the repo root:
    python scripts/diagnose_categorical_dependence.py > results/diagnostics/categorical_dependence.txt
"""
import warnings

import torch

from gcdiff import metrics
from gcdiff.data import TableCodec, load_movielens, make_split
from gcdiff.diffusion import MixedDiffusion, Schedule
from gcdiff.models import Denoiser
from gcdiff.train import fit, set_seed

warnings.filterwarnings("ignore")
t = load_movielens("data")
u = t.users
sp = make_split((u.gender == "M").to_numpy(), (0.6, 0.15, 0.25), 0)
tr, te = u.iloc[sp.train].reset_index(drop=True), u.iloc[sp.test].reset_index(drop=True)
codec = TableCodec.fit(tr, t.vocab)
age_mean, age_std = float(tr.age.mean()), float(tr.age.std() + 1e-8)
xn, xc = (torch.as_tensor(a) for a in codec.encode(u))
print("seed 0 split; unconditional model; T=100; same settings as configs/quick.yaml except epochs/patience")
print("real train: gender-occupation Cramer's V = %.3f; real test = %.3f" % (
    metrics._cramers_v(tr.gender, tr.occupation), metrics._cramers_v(te.gender, te.occupation)))
for epochs, patience in [(200, 8), (600, 1000)]:
    set_seed(0)
    model = MixedDiffusion(Denoiser(1, codec.cat_sizes, 0, 128, 3, 64, 0.1), Schedule(100), 1, codec.cat_sizes)
    cfg = dict(lr=1e-3, weight_decay=1e-4, batch_size=64, epochs=epochs, eval_every=5, patience=patience)
    info = fit(model, None, None, xn, xc, torch.as_tensor(sp.train), torch.as_tensor(sp.val), cfg, 0)
    a, b = model.sample(3000, None, torch.Generator().manual_seed(1))
    s = codec.decode(a.numpy(), b.numpy())
    print(f"epochs<={epochs}: best_epoch={info['best_epoch']} best_val={info['best_val']:.3f} | "
          f"gender-occ V synth={metrics._cramers_v(s.gender, s.occupation):.3f} | "
          f"association error vs real train={metrics.association_error(tr, s):.3f} | "
          f"TSTR-table AUC synth={metrics.tstr_table_auc(s.iloc[:len(tr)], te, t.vocab, age_mean, age_std):.3f} "
          f"(real train rows: {metrics.tstr_table_auc(tr, te, t.vocab, age_mean, age_std):.3f})")
