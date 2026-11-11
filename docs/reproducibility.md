# Reproduction check

## 11 November 2026 maintenance verification

The AI-assisted maintenance pass used Python 3.12.16 on Windows, CPU only,
with NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3, scikit-learn 1.8.0,
PyTorch 2.15.0+cpu, PyG 2.8.1, and Matplotlib 3.10.8. The audit environment
passed its dependency consistency check.

All 33 tests passed. `python scripts/verify_results.py` independently recomputed
all saved summary and paired-delta cells from the 80 per-seed result rows
(tolerance 1e-12), checked method/seed coverage, and matched the five README
comparison rows. Corruption tests cover duplicate/missing rows, changed source
metrics, and a changed README claim. The checker imports no `gcdiff` code.
It checks arithmetic consistency, not the provenance of the original training.

A separate MovieLens 100K execution check downloaded and SHA-256-verified the
archive, then completed all seven configured methods plus the real-label probe
on the 943 users. This was a two-epoch, one-seed plumbing check with a reduced
model (T=8, width=32, depth=1, condition dimension=8, GNN hidden size=8,
two generated draws per test user, seed 0). Its outputs were kept outside the
committed result directory and are not evidence for a scientific comparison.
The standalone Python runtime used the installed CA bundle for HTTPS verification.

The ten-seed study and the full configuration were not retrained. Original
numerical artifacts remain unchanged. An upstream `torch.jit.script`
deprecation warning did not cause a failure. MovieLens experiments remain
separate from the author's ongoing biomedical MSc thesis.

## Earlier verification record

The project was installed in a fresh Python 3.12.14 virtual environment on
Linux, CPU only. PyTorch came from the CPU wheel index, followed by
`pip install -e ".[dev]"`. `pip check` found no dependency conflicts,
and all 28 synthetic-data tests passed.

The tested stack was NumPy 2.5.4, SciPy 1.18.1, pandas 3.0.6, scikit-learn
1.9.1, PyTorch 2.14.1+cpu, PyG 2.8.0.post1, and Matplotlib 3.11.2.
Tests used one BLAS/OpenMP thread; the experiment used two.

The MovieLens download and the complete three-seed quick run were checked:

```bash
python -m gcdiff.run --config configs/quick.yaml --out results/verification-quick --device cpu
```

This completed the unconditional, flat-conditioned, graph-conditioned,
endpoint-shuffled and kNN diffusion variants, the resampling baselines, and
the real-label probe reference. It produced per-seed metrics, summaries, paired
differences, training curves, software versions, and figures in its own output
directory.

Separately, the committed ten-seed summary and paired differences were
recomputed from `results/quick_10seeds/metrics_per_seed.csv` and agreed within
relative and absolute tolerances of 1e-12. Those 80 result rows have no duplicate
`(method, seed)` pairs. The ten-seed training run was not repeated, and the
unexecuted full configuration remains unexecuted in this check.

The dependency versions differ from the original result environment, so this
check does not promise bit-identical retraining. Original result files remain
unchanged. An upstream PyTorch/PyG deprecation warning was emitted but did not
cause tests or the experiment to fail.
