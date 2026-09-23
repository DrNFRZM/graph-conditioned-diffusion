"""CLI: python -m gcdiff.run --config configs/quick.yaml --out results/my_quick"""
from __future__ import annotations

import argparse

from .experiment import load_config, run_experiment


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Graph-conditioned diffusion experiment on MovieLens 100K")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None, help="cpu | cuda | auto (default: from config)")
    p.add_argument("--seeds", type=int, nargs="*", default=None, help="override seeds in the config")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    if a.device:
        cfg["device"] = a.device
    if a.seeds:
        cfg["seeds"] = a.seeds
    run_experiment(cfg, a.out)


if __name__ == "__main__":
    main()
