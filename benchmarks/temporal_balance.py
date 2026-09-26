"""Ordered conditions of unequal size: can subsampling or bootstrap standardisation fix them?

Condition 3 of 4 has few samples (30), the others many (150), in the simulation of
``benchmarks/temporal.py``. Compared, for every sequential step:

- unbalanced: the sequential comparison on the networks as they are,
- balanced control: every condition has 150 samples (the best case),
- subsampled: every condition subsampled to the smallest size, networks rebuilt,
  distances averaged over repeats,
- bootstrap-standardised: cells resampled within each condition, the resampled
  networks projected onto the fixed UASE, and every gene's move divided by its
  bootstrap standard deviation (a per-gene, per-condition noise scale).

Scores: AUROC per step of genes that switch module abruptly against stable module
genes, AUROC pooled over all steps (are the scores of different steps on one
scale?), and true and false calls at q < 0.1.

    python benchmarks/temporal_balance.py   # writes benchmarks/results/temporal_balance.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporal import (DIMS, RESULTS_DIR, auroc, changed_at_step,  # noqa: E402
                      sequential_borda, simulate_trajectories, uase_both)
from node2vec2rank.significance import empirical_null_test  # noqa: E402
from node2vec2rank.simulate import coexpression_network  # noqa: E402

D = max(DIMS)


def sequential_columns(right, K):
    return [sequential_borda(right, t)[1] for t in range(K - 1)]


def subsampled_columns(expression, K, repeats, rng):
    size = min(len(x) for x in expression)
    total = None
    for _ in range(repeats):
        graphs = [coexpression_network(x[rng.choice(len(x), size, replace=False)]) for x in expression]
        _, right = uase_both(graphs, D)
        columns = sequential_columns(right, K)
        total = columns if total is None else [a + b for a, b in zip(total, columns)]
    return [c / repeats for c in total]


def bootstrap_standardised(expression, graphs, K, repeats, rng):
    """Per step, an (n x len(DIMS)) array of |move| / bootstrap sd, one column per dimension."""
    left, right = uase_both(graphs, D)
    # right = A^T left S^{-1}, with S = left^T left diagonal
    projector = left / np.sum(left ** 2, axis=0)
    boot = np.empty((repeats,) + right.shape)
    for b in range(repeats):
        for t, x in enumerate(expression):
            resampled = x[rng.integers(0, len(x), len(x))]
            boot[b, t] = coexpression_network(resampled).T @ projector
    variance = boot.var(axis=0)  # K x n x D
    columns = []
    for t in range(K - 1):
        move = right[t + 1] - right[t]
        cols = []
        for dim in DIMS:
            noise = variance[t, :, :dim].sum(axis=1) + variance[t + 1, :, :dim].sum(axis=1)
            cols.append(np.linalg.norm(move[:, :dim], axis=1) / np.sqrt(noise))
        columns.append(np.column_stack(cols))
    return columns


def score(columns, graphs, changed, abrupt, keep):
    """AUROC per step, pooled AUROC and calls, from per-step distance columns."""
    rows, pooled_score, pooled_label = [], [], []
    for t, cols in enumerate(columns):
        degree = np.mean([np.abs(graphs[t]).sum(axis=0), np.abs(graphs[t + 1]).sum(axis=0)], axis=0)
        z, _, q = empirical_null_test(cols, degree)
        label = changed[t] & abrupt
        raw = cols.mean(axis=1)  # average over columns, on the method's own scale
        rows.append(dict(step=f"{t + 1}->{t + 2}", auroc=auroc(raw[keep], label[keep]),
                         true_calls=int(np.sum((q < 0.1) & label)),
                         false_calls=int(np.sum((q < 0.1) & ~changed[t]))))
        pooled_score.append(raw[keep])
        pooled_label.append(label[keep])
    pooled = auroc(np.concatenate(pooled_score), np.concatenate(pooled_label))
    return [dict(r, pooled_auroc=pooled) for r in rows]


def run(seed, small=30, large=150, repeats=10):
    K = 4
    rows = []
    for setting in ("unbalanced", "balanced control"):
        samples = [large] * K
        if setting == "unbalanced":
            samples[2] = small
        rng = np.random.default_rng(seed)
        expression, weight, kind, home, *_ = simulate_trajectories(K, samples=samples, rng=rng)
        expression = [np.asarray(x) for x in expression]
        graphs = [coexpression_network(x) for x in expression]
        changed = changed_at_step(weight, kind, K)
        abrupt = np.isin(kind, ("persistent", "transient"))
        # negatives are stable module genes; genes without a module barely move and are easy
        keep = abrupt | ((kind == "stable") & (home < home.max()))
        _, right = uase_both(graphs, D)
        methods = {"as is": sequential_columns(right, K)}
        if setting == "unbalanced":
            methods["subsampled"] = subsampled_columns(expression, K, repeats, rng)
            methods["bootstrap-standardised"] = bootstrap_standardised(expression, graphs, K, repeats * 2, rng)
        for method, columns in methods.items():
            for row in score(columns, graphs, changed, abrupt, keep):
                rows.append(dict(seed=seed, setting=setting, method=method, **row))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=10)
    args = parser.parse_args(argv)
    results = pd.DataFrame([row for seed in range(args.replicates) for row in run(seed)])
    results.to_csv(os.path.join(RESULTS_DIR, "temporal_balance.csv"), index=False)
    pd.set_option("display.width", 200)
    summary = results.groupby(["setting", "method", "step"])[
        ["auroc", "true_calls", "false_calls", "pooled_auroc"]].mean().round(3)
    print(summary)


if __name__ == "__main__":
    main()
