"""Calibration of the averaged-subsample distances when nothing is rewired.

Same designs as ``temporal_bagging.py`` with no rewired genes (differentially
expressed decoys only). Reports false calls at q < 0.1 and the fraction of
p-values below 0.05.

    python benchmarks/temporal_bagging_null.py  # writes benchmarks/results/temporal_bagging_null.csv
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from temporal import RESULTS_DIR  # noqa: E402
from temporal_bagging import draws  # noqa: E402
from temporal_imbalance import NETWORKS, distance_columns, network, simulate_expression  # noqa: E402
from node2vec2rank.significance import empirical_null_test  # noqa: E402


def main():
    rows = []
    for name, transform in NETWORKS.items():
        for sizes in [(190, 190), (840, 190)]:
            for seed in range(10):
                rng = np.random.default_rng(100 + seed)
                sim = simulate_expression(num_genes=600, num_samples=sizes, num_modules=5, frac_rewired=0,
                                          random_state=rng)
                expression = [np.asarray(x) for x in sim.expression]
                expression = [(x - x.mean(axis=0)) / x.std(axis=0) for x in expression]
                graphs = [network(x, transform) for x in expression]
                degree = np.mean([g.sum(axis=0) for g in graphs], axis=0)
                small = min(sizes)
                variants = {"all cells": distance_columns(graphs),
                            "bagged 80%": draws(expression, [int(0.8 * s) for s in sizes], transform, rng)}
                if sizes[0] != sizes[1]:
                    variants["balanced"] = draws(expression, [small, small], transform, rng)
                for method, columns in variants.items():
                    _, p, q = empirical_null_test(columns, degree)
                    rows.append(dict(network=name, sizes=f"{sizes[0]} vs {sizes[1]}", seed=seed, method=method,
                                     false_calls=int(np.sum(q < 0.1)), frac_p_below_05=float(np.nanmean(p < 0.05))))
    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(RESULTS_DIR, "temporal_bagging_null.csv"), index=False)
    print(results.drop(columns="seed").groupby(["network", "sizes", "method"], sort=False)
          .agg(["mean", "max"]).round(3).to_string())


if __name__ == "__main__":
    main()
