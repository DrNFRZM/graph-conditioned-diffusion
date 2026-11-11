"""Ensure result drift, missing seeds and duplicate observations fail verification."""
import importlib.util
import shutil
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_results", ROOT / "scripts/verify_results.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


@pytest.fixture
def evidence(tmp_path):
    run = tmp_path / "results/quick_10seeds"
    run.mkdir(parents=True)
    for name in ("config.yaml", "metrics_per_seed.csv", "summary.csv", "paired_deltas.csv"):
        shutil.copy2(ROOT / "results/quick_10seeds" / name, run / name)
    shutil.copy2(ROOT / "README.md", tmp_path / "README.md")
    return tmp_path


def test_committed_evidence_is_consistent():
    verifier.verify(ROOT)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "metric", "readme"])
def test_corrupted_evidence_fails(evidence, mutation):
    path = evidence / "results/quick_10seeds/metrics_per_seed.csv"
    df = pd.read_csv(path)
    if mutation == "duplicate":
        pd.concat([df, df.iloc[:1]]).to_csv(path, index=False)
    elif mutation == "missing":
        df.iloc[1:].to_csv(path, index=False)
    elif mutation == "metric":
        df.loc[df.method == "diff_gnn", "cond_age_r2"] += 0.1
        df.to_csv(path, index=False)
    else:
        readme = evidence / "README.md"
        readme.write_text(readme.read_text(encoding="utf-8").replace("0.151 ± 0.061", "0.951 ± 0.061"), encoding="utf-8")
    with pytest.raises((ValueError, AssertionError)):
        verifier.verify(evidence)
