"""Calibration of N2V2R.significance() under no change, for the combinations of
degree trend and combination rule that node2vec2rank has used: the quadratic
trend with the mean over dimensions, the natural-spline trend with the mean,
and the natural-spline trend with the Cauchy combination of dimension windows
(the default).

The null scenarios follow the independent review of the significance test:
degree-corrected SBMs with Pareto or lognormal degrees, a weighted
degree-corrected SBM, and co-expression networks (|cor|^6) from 60 or 150
samples in which 40% of genes belong to no module. Both graphs of a pair are
independent draws of the same model, so every call is a false one. Run from
the repository root (about 3 minutes on 4 cores):

    python benchmarks/calibration.py      # -> results/calibration.csv
"""

import argparse
import os
import sys
import zlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from node2vec2rank.embedding import embed, select_dimension  # noqa: E402
from node2vec2rank.model_utils import compute_pairwise_distances  # noqa: E402
from node2vec2rank.significance import empirical_null_test  # noqa: E402

DEFAULT_DIMS = list(range(4, 25, 2))
METRICS = ("euclidean", "cosine")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SETTINGS = (("polynomial", "mean"), ("spline", "mean"), ("spline", "cauchy"))


def dcsbm(rng, theta, weighted=False, num_blocks=4, p_in=0.1, p_out=0.02):
    blocks = np.arange(len(theta)) % num_blocks
    probs = np.where(blocks[:, None] == blocks[None, :], p_in, p_out) * np.outer(theta, theta)
    upper = np.triu((rng.random(probs.shape) < np.clip(probs, 0, 1)).astype(float), 1)
    if weighted:
        upper *= np.abs(1 + 0.3 * rng.standard_normal(upper.shape))
    return upper + upper.T


def coexpression(rng, num_samples, loadings, modules, num_modules=6):
    factors = rng.standard_normal((num_samples, num_modules))
    expression = rng.standard_normal((num_samples, len(modules)))
    member = modules >= 0
    expression[:, member] = (factors[:, modules[member]] * loadings[member]
                             + expression[:, member] * np.sqrt(1 - loadings[member] ** 2))
    adjacency = np.abs(np.corrcoef(expression, rowvar=False)) ** 6
    np.fill_diagonal(adjacency, 0)
    return adjacency


def null_pair(scenario, rng):
    if scenario.startswith("dcsbm") or scenario == "weighted dcsbm":
        theta = rng.lognormal(0, 1, 1000) if scenario == "dcsbm lognormal" else rng.pareto(2.5, 1000) + 1
        theta /= theta.mean()
        return [dcsbm(rng, theta, weighted=scenario == "weighted dcsbm") for _ in range(2)]
    num_samples = int(scenario.split()[1])
    modules = np.where(rng.random(800) < 0.6, rng.integers(0, 6, 800), -1)
    loadings = rng.uniform(0.2, 0.9, 800)
    return [coexpression(rng, num_samples, loadings, modules) for _ in range(2)]


SCENARIOS = ("dcsbm pareto", "dcsbm lognormal", "weighted dcsbm", "coexpression 60", "coexpression 150")


def replicate(scenario, rep):
    rng = np.random.default_rng([rep, zlib.crc32(scenario.encode())])
    graphs = null_pair(scenario, rng)
    degree = (graphs[0].sum(axis=0) + graphs[1].sum(axis=0)) / 2
    embeddings, singular_values = embed(graphs, max(DEFAULT_DIMS), "uase", random_state=rep,
                                        return_singular_values=True)
    elbow = select_dimension(singular_values, min_dimension=2)
    records = []
    for dims_name, dims in (("default", DEFAULT_DIMS), ("elbow", [elbow])):
        distances = np.column_stack([compute_pairwise_distances(embeddings[0][:, :d], embeddings[1][:, :d], m)
                                     for d in dims for m in METRICS])
        column_dimensions = np.repeat(dims, len(METRICS))
        for trend, combine in SETTINGS:
            _, pvalues, qvalues = empirical_null_test(distances, degree, trend=trend, combine=combine,
                                                      dimensions=column_dimensions)
            valid = ~np.isnan(pvalues)
            records.append({"scenario": scenario, "replicate": rep, "dimensions": dims_name, "trend": trend,
                            "combine": combine,
                            "frac_p_below_0.05": np.mean(pvalues[valid] < 0.05),
                            "frac_p_below_0.001": np.mean(pvalues[valid] < 0.001),
                            "num_called_q_0.1": int(np.sum(qvalues < 0.1))})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replicates", type=int, default=30)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = Parallel(n_jobs=-1)(delayed(replicate)(s, r) for s in SCENARIOS for r in range(args.replicates))
    results = pd.DataFrame([r for records in out for r in records])
    results.to_csv(os.path.join(RESULTS_DIR, "calibration.csv"), index=False)
    results["any_call"] = results["num_called_q_0.1"] > 0
    summary = results.groupby(["scenario", "dimensions", "trend", "combine"])[
        ["frac_p_below_0.05", "frac_p_below_0.001", "any_call", "num_called_q_0.1"]].mean()
    print(summary.round(3).to_string())
