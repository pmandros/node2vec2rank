"""Simulation benchmarks for node2vec2rank.

Compares ranking and testing strategies on simulated pairs of graphs with a
known set of changed nodes, and under no change (to check calibration and
degree bias). Run from the repository root:

    python benchmarks/simulations.py            # writes benchmarks/results/*.csv
    python benchmarks/figures.py                # writes benchmarks/results/*.png

Scenarios
---------
dcsbm       binary degree-corrected SBM (Pareto degrees), changed nodes switch block
sbm         binary SBM without degree heterogeneity
weighted    degree-corrected SBM with Gaussian edge weights
coexpression  WGCNA-style networks (|correlation|^6) from simulated expression
              with gene modules; changed genes switch module
"""

import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from node2vec2rank.embedding import embed, select_dimension  # noqa: E402
from node2vec2rank.model_utils import borda_aggregate, compute_pairwise_distances  # noqa: E402
from node2vec2rank.significance import empirical_null_test  # noqa: E402

DEFAULT_DIMS = list(range(4, 25, 2))
METRICS = ("euclidean", "cosine")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ---------------------------------------------------------------- simulators

def _changed_blocks(blocks, num_blocks, frac_changed, rng):
    after = blocks.copy()
    changed = np.zeros(len(blocks), bool)
    if frac_changed > 0:
        moved = rng.choice(len(blocks), int(round(frac_changed * len(blocks))), replace=False)
        after[moved] = (blocks[moved] + 1) % num_blocks
        changed[moved] = True
    return after, changed


def simulate_sbm(rng, num_nodes=1000, frac_changed=0.1, heterogeneous=True, weighted=False,
                 num_blocks=4, p_in=0.2, p_out=0.03, weight_noise=0.1):
    blocks = rng.integers(0, num_blocks, num_nodes)
    after, changed = _changed_blocks(blocks, num_blocks, frac_changed, rng)
    theta = rng.pareto(2.5, num_nodes) + 1 if heterogeneous else np.ones(num_nodes)
    theta /= theta.mean()
    block_probs = np.full((num_blocks, num_blocks), p_out) + np.eye(num_blocks) * (p_in - p_out)
    graphs = []
    for z in (blocks, after):
        probs = np.clip(np.outer(theta, theta) * block_probs[z][:, z], 0, 1)
        if weighted:
            upper = np.triu(probs + weight_noise * rng.standard_normal(probs.shape), 1)
        else:
            upper = np.triu(rng.random(probs.shape) < probs, 1).astype(float)
        graphs.append(upper + upper.T)
    return graphs, changed


def simulate_coexpression(rng, num_genes=1000, frac_changed=0.1, num_modules=5, num_samples=150,
                          frac_in_modules=0.7, soft_power=6):
    """Genes load on one latent module factor (or none); changed module genes
    load on the next module in the second condition."""
    in_module = rng.random(num_genes) < frac_in_modules
    modules = np.where(in_module, rng.integers(0, num_modules, num_genes), -1)
    after = modules.copy()
    changed = np.zeros(num_genes, bool)
    candidates = np.flatnonzero(in_module)
    moved = rng.choice(candidates, int(round(frac_changed * num_genes)), replace=False)
    after[moved] = (modules[moved] + 1) % num_modules
    changed[moved] = True
    # heterogeneous loadings give heterogeneous connectivity, as in real data
    loadings = rng.uniform(0.3, 0.9, num_genes)
    graphs = []
    for z in (modules, after):
        factors = rng.standard_normal((num_samples, num_modules))
        expression = rng.standard_normal((num_samples, num_genes))
        member = z >= 0
        expression[:, member] = (loadings[member] * factors[:, z[member]]
                                 + np.sqrt(1 - loadings[member] ** 2) * expression[:, member])
        adjacency = np.abs(np.corrcoef(expression, rowvar=False)) ** soft_power
        np.fill_diagonal(adjacency, 0)
        graphs.append(adjacency)
    return graphs, changed


SCENARIOS = {
    "dcsbm": lambda rng, frac: simulate_sbm(rng, frac_changed=frac),
    "sbm": lambda rng, frac: simulate_sbm(rng, frac_changed=frac, heterogeneous=False),
    "weighted": lambda rng, frac: simulate_sbm(rng, frac_changed=frac, weighted=True),
    "coexpression": lambda rng, frac: simulate_coexpression(rng, frac_changed=frac),
}


# ---------------------------------------------------------------- methods

def distances(embeddings, dims, metrics):
    return np.column_stack([compute_pairwise_distances(embeddings[0][:, :d], embeddings[1][:, :d], m)
                            for d in dims for m in metrics])


TESTS = {"n2v2r degree-adjusted z": "default", "n2v2r degree-adjusted z (elbow)": "elbow"}


def score_methods(graphs, seed=0):
    """Returns ({method: score}, degree, {test method: (p, q)}, elbow dimension)."""
    degree = (np.abs(graphs[0]).sum(axis=0) + np.abs(graphs[1]).sum(axis=0)) / 2
    uase_embeddings, singular_values = embed(graphs, max(DEFAULT_DIMS), "uase", random_state=seed,
                                             return_singular_values=True)
    ulse_embeddings = embed(graphs, max(DEFAULT_DIMS), "ulse", random_state=seed)
    elbow = select_dimension(singular_values, min_dimension=2)

    all_distances = distances(uase_embeddings, DEFAULT_DIMS, METRICS)
    elbow_distances = distances(uase_embeddings, [elbow], METRICS)
    z_all, p_all, q_all = empirical_null_test(all_distances, degree)
    z_elbow, p_elbow, q_elbow = empirical_null_test(elbow_distances, degree)

    scores = {
        "DeDi": np.abs(graphs[0].sum(axis=0) - graphs[1].sum(axis=0)),
        "n2v2r (default)": borda_aggregate(all_distances),
        "n2v2r euclidean only": borda_aggregate(all_distances[:, 0::2]),
        "n2v2r cosine only": borda_aggregate(all_distances[:, 1::2]),
        "n2v2r ULSE": borda_aggregate(distances(ulse_embeddings, DEFAULT_DIMS, METRICS)),
        "n2v2r elbow dimension": borda_aggregate(elbow_distances),
        "n2v2r degree-adjusted z": np.nan_to_num(z_all, nan=-np.inf),
        "n2v2r degree-adjusted z (elbow)": np.nan_to_num(z_elbow, nan=-np.inf),
    }
    tests = {"n2v2r degree-adjusted z": (p_all, q_all),
             "n2v2r degree-adjusted z (elbow)": (p_elbow, q_elbow)}
    return scores, degree, tests, elbow


def auroc(labels, scores):
    ranks = rankdata(scores)
    positives = labels.sum()
    negatives = len(labels) - positives
    return (ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives)


def precision_at_k(labels, scores):
    k = labels.sum()
    top = np.argsort(-scores, kind="stable")[:k]
    return labels[top].mean()


def spearman(a, b):
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]


# ---------------------------------------------------------------- runner

PVALUE_BINS = np.linspace(0, 1, 21)


def run(num_replicates=5, fractions=(0.0, 0.1, 0.3)):
    """Returns (per-replicate metrics, histogram counts of the null p-values)."""
    records = []
    null_histograms = []
    for scenario, frac, replicate in itertools.product(SCENARIOS, fractions, range(num_replicates)):
        rng = np.random.default_rng([replicate, int(frac * 100), list(SCENARIOS).index(scenario)])
        graphs, changed = SCENARIOS[scenario](rng, frac)
        tic = time.time()
        scores, degree, tests, elbow = score_methods(graphs, seed=replicate)
        for method, score in scores.items():
            record = {"scenario": scenario, "frac_changed": frac, "replicate": replicate,
                      "method": method, "elbow_dimension": elbow,
                      "spearman_with_degree": spearman(score, degree)}
            if frac > 0:
                record["auroc"] = auroc(changed, score)
                record["precision_at_k"] = precision_at_k(changed, score)
            if method in tests:
                pvalues, qvalues = tests[method]
                valid = ~np.isnan(pvalues)
                record["frac_p_below_0.05"] = np.mean(pvalues[valid] < 0.05)
                record["num_called_q_0.1"] = int(np.sum(qvalues < 0.1))
                called = qvalues < 0.1
                if frac > 0:
                    record["power_q_0.1"] = called[changed].mean()
                    record["fdp_q_0.1"] = (called & ~changed).sum() / max(called.sum(), 1)
                else:
                    record["fdp_q_0.1"] = float(called.any())
                    counts, _ = np.histogram(pvalues[valid], bins=PVALUE_BINS)
                    null_histograms.extend(
                        {"scenario": scenario, "method": method, "replicate": replicate,
                         "bin_start": start, "count": count}
                        for start, count in zip(PVALUE_BINS[:-1], counts))
            records.append(record)
        print(f"{scenario:13s} frac={frac:.1f} rep={replicate} elbow={elbow} "
              f"({time.time() - tic:.1f}s)", flush=True)
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(null_histograms)


def run_demo():
    """Recall of the changed community on the repository's demo graphs."""
    from node2vec2rank.dataloader import DataLoader
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    demo_dir = os.path.join(repo, "data", "networks", "demo")
    loader = DataLoader(data_dir=demo_dir, graph_filenames=["adj_matrix_1.csv", "adj_matrix_2.csv"],
                        separator=",", verbose=-1)
    nodes = pd.Index(loader.get_nodes())
    communities = pd.read_csv(os.path.join(demo_dir, "comm_asiggnments.csv"), index_col=0)
    communities.index = communities.index.astype(str)
    changed = (communities.loc[nodes, "0"] == 0).to_numpy()
    scores, _, tests, elbow = score_methods(loader.get_graphs(), seed=0)
    records = []
    for method, score in scores.items():
        record = {"method": method, "recall_at_k": precision_at_k(changed, score),
                  "auroc": auroc(changed, score), "elbow_dimension": elbow}
        if method in tests:
            called = tests[method][1] < 0.1
            record["power_q_0.1"] = called[changed].mean()
            record["fdp_q_0.1"] = (called & ~changed).sum() / max(called.sum(), 1)
        records.append(record)
    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replicates", type=int, default=20)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    demo = run_demo()
    demo.to_csv(os.path.join(RESULTS_DIR, "demo.csv"), index=False)
    print(demo.round(3).to_string(index=False))
    results, null_histograms = run(num_replicates=args.replicates)
    results.to_csv(os.path.join(RESULTS_DIR, "simulations.csv"), index=False)
    null_histograms.to_csv(os.path.join(RESULTS_DIR, "null_pvalue_histograms.csv"), index=False)
