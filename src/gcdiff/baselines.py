"""Non-generative baselines. They ignore the graph entirely."""
from __future__ import annotations

import numpy as np
import pandas as pd


def marginal_bootstrap(train: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Resample every column independently from its empirical train marginal.
    Perfect univariate marginals by construction, zero cross-column dependence."""
    return pd.DataFrame({c: rng.choice(train[c].to_numpy(), n, replace=True) for c in train.columns})


def row_bootstrap(train: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Resample whole train rows. Reference for 'what a pure copy of the training data scores':
    fidelity at the real-data noise floor, memorisation checks at their maximum."""
    return train.iloc[rng.integers(0, len(train), n)].reset_index(drop=True)
