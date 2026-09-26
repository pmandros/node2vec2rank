"""Is the gain from subsampling the bigger group due to balance or to averaging?

Follow-up to ``temporal_imbalance.py``. For each group-size pair, distances are
averaged over 10 draws of

- balanced: both groups subsampled to the smaller size (the bigger group varies),
- bagged 80%: both groups subsampled to 80% of their own size (stays unbalanced),

and compared with the distances on all cells.

    python benchmarks/temporal_bagging.py   # writes benchmarks/results/temporal_bagging.csv
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from temporal import RESULTS_DIR  # noqa: E402
from temporal_imbalance import NETWORKS, distance_columns, evaluate, network, simulate_expression  # noqa: E402


def draws(expression, sizes, transform, rng, repeats=10):
    total = 0
    for _ in range(repeats):
        total = total + distance_columns(
            [network(x[rng.choice(len(x), s, replace=False)], transform) for x, s in zip(expression, sizes)])
    return total / repeats


def main():
    rows = []
    for name, transform in NETWORKS.items():
        for sizes in [(190, 190), (840, 190), (840, 260)]:
            for seed in range(10):
                rng = np.random.default_rng(seed)
                sim = simulate_expression(num_genes=600, num_samples=sizes, num_modules=5, random_state=rng)
                expression = [np.asarray(x) for x in sim.expression]
                expression = [(x - x.mean(axis=0)) / x.std(axis=0) for x in expression]
                rewired = sim.rewired.to_numpy()
                graphs = [network(x, transform) for x in expression]
                degree = np.mean([g.sum(axis=0) for g in graphs], axis=0)
                small = min(sizes)
                variants = {"all cells": distance_columns(graphs),
                            "bagged 80%": draws(expression, [int(0.8 * s) for s in sizes], transform, rng)}
                if sizes[0] != sizes[1]:
                    variants["balanced"] = draws(expression, [small, small], transform, rng)
                    variants["balanced, one draw"] = draws(expression, [small, small], transform, rng, repeats=1)
                for method, columns in variants.items():
                    rows.append(dict(network=name, sizes=f"{sizes[0]} vs {sizes[1]}", seed=seed, method=method,
                                     **evaluate(columns, degree, rewired)))
    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(RESULTS_DIR, "temporal_bagging.csv"), index=False)
    pd.set_option("display.width", 200)
    print(results.drop(columns="seed").groupby(["network", "sizes", "method"], sort=False).mean().round(3)
          .to_string())


if __name__ == "__main__":
    main()
