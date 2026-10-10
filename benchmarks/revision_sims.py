"""Simulations for the revision: community splits and merges, and one
ablation table.

Parts:

1. splitmerge: changes other than single nodes switching community.
   - "split": one community splits in two in the second network (half of its
     nodes form a new community). Changed: every node of that community (each
     loses half its partners).
   - "merge": two communities merge into one. Changed: every node of the two.
   - "switch": 10% of the nodes switch community (the paper's setting), for
     reference.
   On the degree-corrected block model (1,000 nodes, 8 blocks) and the
   co-expression simulation (1,000 genes, 8 modules), 10 replicates each.
   Methods: n2v2r default, elbow dimension, euclidean only, cosine only, DeDi,
   and the DGCA-style score on co-expression.
2. ablation: the paper's two simulations (dcsbm and coexpression of
   benchmarks/simulations.py, 10% switched), 10 replicates, every component of
   n2v2r varied one at a time:
   - embedding: joint UASE (default), ULSE, omnibus embedding (OMNI),
     separate adjacency spectral embeddings aligned by orthogonal Procrustes,
     and the raw adjacency rows (no embedding);
   - dimension: each single dimension 4..24, the elbow, and the Borda over all;
   - distance: euclidean, cosine, both;
   - aggregation of the 22 rankings: Borda (= mean rank), median rank;
   - DeDi.

Run from the repository root:

    OMP_NUM_THREADS=1 python benchmarks/revision_sims.py --part splitmerge
    OMP_NUM_THREADS=1 python benchmarks/revision_sims.py --part ablation
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh, orthogonal_procrustes
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import competitor_seeds as cs  # noqa: E402
import simulations as sims  # noqa: E402
from node2vec2rank.embedding import embed, select_dimension  # noqa: E402
from node2vec2rank.model_utils import borda_aggregate, compute_pairwise_distances  # noqa: E402

DIMS = sims.DEFAULT_DIMS
METRICS = sims.METRICS
NUM_REPLICATES = 10


# ---------------------------------------------------------------- split / merge simulators

def _communities(rng, num_nodes, num_blocks, kind, frac_switch=0.1):
    """Community labels before and after, and the changed nodes."""
    before = rng.integers(0, num_blocks, num_nodes)
    after = before.copy()
    if kind == "split":
        members = np.flatnonzero(before == 0)
        after[rng.choice(members, len(members) // 2, replace=False)] = num_blocks
        changed = before == 0
    elif kind == "merge":
        after[before == 1] = 0
        changed = np.isin(before, [0, 1])
    elif kind == "switch":
        after, changed = sims._changed_blocks(before, num_blocks, frac_switch, rng)
    else:
        raise ValueError(kind)
    return before, after, changed


def sbm_pair(rng, kind, num_nodes=1000, num_blocks=8, p_in=0.2, p_out=0.03):
    """Degree-corrected block model, as simulations.simulate_sbm."""
    before, after, changed = _communities(rng, num_nodes, num_blocks, kind)
    theta = rng.pareto(2.5, num_nodes) + 1
    theta /= theta.mean()
    graphs = []
    for z in (before, after):
        same = z[:, None] == z[None, :]
        probs = np.clip(np.outer(theta, theta) * np.where(same, p_in, p_out), 0, 1)
        upper = np.triu(rng.random(probs.shape) < probs, 1).astype(float)
        graphs.append(upper + upper.T)
    return graphs, changed, None


def coexpression_pair(rng, kind, num_genes=1000, num_modules=8, num_samples=150, frac_in_modules=0.7,
                      soft_power=6):
    """Co-expression simulation, as simulations.simulate_coexpression; genes
    outside modules (label -1) never change."""
    in_module = rng.random(num_genes) < frac_in_modules
    before, after, changed = _communities(rng, num_genes, num_modules, kind)
    if kind == "switch":
        # switch only module genes, as the paper's simulator
        after, changed = before.copy(), np.zeros(num_genes, bool)
        moved = rng.choice(np.flatnonzero(in_module), int(round(0.1 * num_genes)), replace=False)
        after[moved] = (before[moved] + 1) % num_modules
        changed[moved] = True
    before, after = np.where(in_module, before, -1), np.where(in_module, after, -1)
    changed &= in_module
    loadings = rng.uniform(0.3, 0.9, num_genes)
    graphs, expressions = [], []
    for z in (before, after):
        factors = rng.standard_normal((num_samples, num_modules + 1))
        expression = rng.standard_normal((num_samples, num_genes))
        member = z >= 0
        expression[:, member] = (loadings[member] * factors[:, z[member]]
                                 + np.sqrt(1 - loadings[member] ** 2) * expression[:, member])
        adjacency = np.abs(np.corrcoef(expression, rowvar=False)) ** soft_power
        np.fill_diagonal(adjacency, 0)
        graphs.append(adjacency)
        expressions.append(expression)
    return graphs, changed, expressions


# ---------------------------------------------------------------- embeddings

def ase(graph, d):
    """Adjacency spectral embedding: top-d eigenvectors by magnitude, scaled
    by the square root of the absolute eigenvalues."""
    values, vectors = eigh(graph)
    top = np.argsort(-np.abs(values))[:d]
    return vectors[:, top] * np.sqrt(np.abs(values[top]))


def omni(graphs, d):
    """Omnibus embedding (Levin et al. 2017) of two graphs."""
    mean = (graphs[0] + graphs[1]) / 2
    matrix = np.block([[graphs[0], mean], [mean, graphs[1]]])
    x = ase(matrix, d)
    n = len(graphs[0])
    return x[:n], x[n:]


def distance_columns(pairs_by_dim, metrics=METRICS):
    """pairs_by_dim: {d: (X1, X2)} -> {(d, metric): distances}."""
    return {(d, m): compute_pairwise_distances(x1, x2, m) for d, (x1, x2) in pairs_by_dim.items() for m in metrics}


def median_rank(columns):
    columns = np.where(np.isnan(columns), -np.inf, columns)
    return np.median(rankdata(columns, axis=0), axis=1)


def ablation_scores(graphs, seed):
    uase_x, singular_values = embed(graphs, max(DIMS), "uase", random_state=seed, return_singular_values=True)
    ulse_x = embed(graphs, max(DIMS), "ulse", random_state=seed)
    elbow = select_dimension(singular_values, min_dimension=2)
    omni_x = omni(graphs, max(DIMS))
    ase_x = [ase(g, max(DIMS)) for g in graphs]

    def sliced(x):
        return {d: (x[0][:, :d], x[1][:, :d]) for d in DIMS}

    def procrustes(d):
        first, second = ase_x[0][:, :d], ase_x[1][:, :d]
        rotation, _ = orthogonal_procrustes(second, first)
        return first, second @ rotation

    uase_cols = distance_columns(sliced(uase_x))
    all_uase = np.column_stack(list(uase_cols.values()))
    scores = {
        "embedding: joint UASE (default)": borda_aggregate(all_uase),
        "embedding: ULSE": borda_aggregate(np.column_stack(list(distance_columns(sliced(ulse_x)).values()))),
        "embedding: OMNI": borda_aggregate(np.column_stack(list(distance_columns(sliced(omni_x)).values()))),
        "embedding: separate ASE + Procrustes": borda_aggregate(np.column_stack(list(
            distance_columns({d: procrustes(d) for d in DIMS}).values()))),
        "embedding: none (raw adjacency rows)": borda_aggregate(np.column_stack(
            [compute_pairwise_distances(graphs[0], graphs[1], m) for m in METRICS])),
        "distance: euclidean only": borda_aggregate(np.column_stack([uase_cols[d, "euclidean"] for d in DIMS])),
        "distance: cosine only": borda_aggregate(np.column_stack([uase_cols[d, "cosine"] for d in DIMS])),
        "aggregation: median rank": median_rank(all_uase),
        "dimension: elbow": borda_aggregate(np.column_stack(list(distance_columns(
            {elbow: (uase_x[0][:, :elbow], uase_x[1][:, :elbow])}).values()))),
        "DeDi": cs.degree_difference(graphs),
    }
    for d in DIMS:
        scores[f"dimension: {d} only"] = borda_aggregate(np.column_stack([uase_cols[d, m] for m in METRICS]))
    return scores, elbow


def basic_scores(graphs, expressions, seed):
    uase_x, singular_values = embed(graphs, max(DIMS), "uase", random_state=seed, return_singular_values=True)
    elbow = select_dimension(singular_values, min_dimension=2)
    cols = distance_columns({d: (uase_x[0][:, :d], uase_x[1][:, :d]) for d in DIMS + [elbow]})
    scores = {
        "n2v2r (default)": borda_aggregate(np.column_stack([cols[d, m] for d in DIMS for m in METRICS])),
        "n2v2r elbow dimension": borda_aggregate(np.column_stack([cols[elbow, m] for m in METRICS])),
        "n2v2r euclidean only": borda_aggregate(np.column_stack([cols[d, "euclidean"] for d in DIMS])),
        "n2v2r cosine only": borda_aggregate(np.column_stack([cols[d, "cosine"] for d in DIMS])),
        "DeDi": cs.degree_difference(graphs),
    }
    if expressions is not None:
        scores["DGCA-style"] = cs.dgca_score(expressions)
    return scores, elbow


def metrics(labels, score):
    return dict(auroc=sims.auroc(labels, score), precision_at_k=sims.precision_at_k(labels, score))


def part_splitmerge(replicates):
    rows = []
    for setting, simulate in (("block model", sbm_pair), ("co-expression", coexpression_pair)):
        for kind in ("split", "merge", "switch"):
            for replicate in range(replicates):
                rng = np.random.default_rng(1000 * replicate + {"split": 1, "merge": 2, "switch": 3}[kind])
                graphs, changed, expressions = simulate(rng, kind)
                degree = [g.sum(axis=0) for g in graphs]
                delta = (degree[1] - degree[0]) / np.maximum(degree[0], 1e-12)
                scores, elbow = basic_scores(graphs, expressions, replicate)
                for name, score in scores.items():
                    rows.append(dict(setting=setting, change=kind, replicate=replicate, method=name, elbow=elbow,
                                     changed=int(changed.sum()),
                                     changed_relative_degree_change=delta[changed].mean(),
                                     **metrics(changed, score)))
                print(setting, kind, replicate, {k: round(sims.auroc(changed, v), 3) for k, v in scores.items()},
                      flush=True)
    return pd.DataFrame(rows)


def part_ablation(replicates):
    rows = []
    for scenario in ("dcsbm", "coexpression"):
        for replicate in range(replicates):
            graphs, changed = sims.SCENARIOS[scenario](np.random.default_rng(2000 + replicate), 0.1)
            scores, elbow = ablation_scores(graphs, replicate)
            for name, score in scores.items():
                rows.append(dict(scenario=scenario, replicate=replicate, variant=name, elbow=elbow,
                                 **metrics(changed, score)))
            print(scenario, replicate, elbow, {k: round(sims.auroc(changed, v), 3) for k, v in scores.items()},
                  flush=True)
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=["splitmerge", "ablation"], required=True)
    parser.add_argument("--replicates", type=int, default=NUM_REPLICATES)
    args = parser.parse_args(argv)
    result = {"splitmerge": part_splitmerge, "ablation": part_ablation}[args.part](args.replicates)
    os.makedirs(cs.RESULTS_DIR, exist_ok=True)
    result.to_csv(os.path.join(cs.RESULTS_DIR, f"revision_{args.part}.csv"), index=False)
    keys = ["setting", "change", "method"] if args.part == "splitmerge" else ["scenario", "variant"]
    summary = result.groupby(keys, sort=False)[["auroc", "precision_at_k"]].agg(["mean", "std"]).round(3)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summary, flush=True)


if __name__ == "__main__":
    main()
