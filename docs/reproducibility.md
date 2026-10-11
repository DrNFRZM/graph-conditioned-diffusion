# Reproduction check

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
