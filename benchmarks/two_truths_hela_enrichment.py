"""Pathway enrichment on HeLa for each "truth" ranking: community (cosine), hub (radial), euclidean,
the paper's default (Borda of euclidean and cosine) and the degree difference, on the adjacency (UASE)
and the Laplacian (ULSE) embedding.

Uses the pipeline of benchmarks/cell_cycle.py (Revelio HeLa S3 cells, 4 phases, 2,000 most variable
genes, metacells, hdWGCNA signed networks) and the fast competitive gene-set test of
node2vec2rank/fast_gene_sets.py, both from PR #5. Every ranking is turned into a per-gene
degree-adjusted z-score (the empirical null of node2vec2rank.significance over the ranking's columns:
dims 4-24, and the metric(s) of the ranking); the degree difference uses normal scores. A set's
statistic is its mean z minus the mean over all genes. Its null mean and variance come from
``--num-permutations`` cell-label shuffles within batch, shared by all rankings (each shuffle builds
and embeds the networks once).

Comparisons: G1 -> S, S -> G2, G2 -> M, and random halves of the G1 cells (null, nothing should be
called). Library: Reactome; cell-cycle pathways (by name) are the expected positives.

    OMP_NUM_THREADS=2 python benchmarks/two_truths_hela_enrichment.py --revelio-dir path/to/Revelio/data
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import cell_cycle as cc  # noqa: E402
from node2vec2rank.embedding import embed  # noqa: E402
from node2vec2rank.fast_gene_sets import _correlation_model, _quadratic_forms  # noqa: E402
from node2vec2rank.model_utils import compute_pairwise_distances  # noqa: E402
from node2vec2rank.permutation import _stratified_orders  # noqa: E402
from node2vec2rank.significance import benjamini_hochberg, empirical_null_test  # noqa: E402

RANKINGS = ("community (cosine)", "hub (radial)", "euclidean", "default (euclidean + cosine)")


def node_scores(group_a, group_b, build):
    """Degree-adjusted z-scores of every ranking, for one pair of groups."""
    graphs = [np.asarray(build(group_a), dtype=float), np.asarray(build(group_b), dtype=float)]
    degree = np.mean([g.sum(axis=0) for g in graphs], axis=0)
    scores = {}
    for method in ("uase", "ulse"):
        embeddings = embed(graphs, max(cc.DIMENSIONS), method=method, random_state=42)
        radial, angular, euclid = [], [], []
        for d in cc.DIMENSIONS:
            one, two = embeddings[0, :, :d], embeddings[1, :, :d]
            radial.append(np.abs(np.linalg.norm(one, axis=1) - np.linalg.norm(two, axis=1)))
            angular.append(compute_pairwise_distances(one, two, "cosine"))
            euclid.append(compute_pairwise_distances(one, two, "euclidean"))
        radial, angular, euclid = map(np.column_stack, (radial, angular, euclid))
        columns = {"community (cosine)": angular, "hub (radial)": radial, "euclidean": euclid,
                   "default (euclidean + cosine)": np.hstack([euclid, angular])}
        for name, values in columns.items():
            z, _, _ = empirical_null_test(values, degree)
            scores[(method.upper(), name)] = np.nan_to_num(z)
    dedi = np.abs(graphs[1].sum(axis=0) - graphs[0].sum(axis=0))
    scores[("-", "degree difference")] = norm.ppf((rankdata(dedi) - 0.5) / len(dedi))
    return scores


def set_tests(group_a, group_b, strata, build, members, num_permutations, rng):
    observed = node_scores(group_a, group_b, build)
    pooled = pd.concat([group_a, group_b], axis=0, ignore_index=True)
    size_a = len(group_a)
    orders = _stratified_orders(np.r_[strata[0], strata[1]], size_a, num_permutations, rng)
    null = [node_scores(pooled.iloc[o[:size_a]], pooled.iloc[o[size_a:]], build) for o in orders]
    values = pooled.to_numpy(dtype=np.float64)
    spread = values.std(axis=0, ddof=1)
    scaled = (values - values.mean(axis=0)) / np.where(spread > 0, spread, 1.0)

    def statistic(z):
        return np.array([z[index].mean() - z.mean() for index in members.values()])

    results = {}
    for key, z_observed in observed.items():
        null_z = np.stack([n[key] for n in null])
        centres, correlations = _correlation_model(null_z, scaled, 400000, 40, rng)
        variance = _quadratic_forms(scaled, members.values(), centres, correlations, null_z.std(axis=0, ddof=1))
        null_mean = np.mean([statistic(z) for z in null_z], axis=0)
        z = (statistic(z_observed) - null_mean) / np.sqrt(np.maximum(variance * (1 + 1 / num_permutations),
                                                                      np.finfo(float).tiny))
        p = norm.sf(z)
        results[key] = pd.DataFrame({"z": z, "pvalue": p, "qvalue": benjamini_hochberg(p)},
                                    index=pd.Index(list(members), name="gene_set"))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--revelio-dir", required=True)
    parser.add_argument("--num-permutations", type=int, default=20)
    parser.add_argument("--num-nulls", type=int, default=2)
    args = parser.parse_args(argv)

    cells_table, groups, batches, build = cc.prepare(args.revelio_dir)
    genes = pd.Index(groups["G1"].columns)
    reactome = cc.read_gmt(cc.REACTOME)
    members = {}
    for name, gs in reactome.items():
        index = np.unique(genes.get_indexer(list(gs)))
        index = index[index >= 0]
        if 5 <= len(index) <= 500:
            members[name] = index
    names = pd.Index(list(members))
    is_cell_cycle = np.asarray(names.str.contains(cc.CELL_CYCLE))
    print(f"{len(members)} Reactome sets tested, {is_cell_cycle.sum()} cell-cycle", flush=True)

    comparisons = [(f"{a} -> {b}", groups[a], groups[b], (batches[a], batches[b])) for a, b in cc.COMPARISONS]
    for seed in range(args.num_nulls):
        half = cc.random_halves(batches["G1"], np.random.default_rng(seed))
        comparisons.append((f"null: G1 halves (seed {seed})", groups["G1"][half], groups["G1"][~half],
                            (batches["G1"][half], batches["G1"][~half])))

    rows, tops = [], []
    rng = np.random.default_rng(0)
    for label, a, b, strata in comparisons:
        tic = time.time()
        for (embedding, ranking), result in set_tests(a, b, strata, build, members, args.num_permutations,
                                                      rng).items():
            called = result["qvalue"].to_numpy() < 0.1
            rows.append(dict(comparison=label, embedding=embedding, ranking=ranking, called=int(called.sum()),
                             cell_cycle_called=int((called & is_cell_cycle).sum()),
                             auroc_cell_cycle=cc.auroc(is_cell_cycle, result["z"].to_numpy())))
            top = result[result["qvalue"] < 0.1].sort_values("z", ascending=False).head(15)
            tops.append(pd.DataFrame({"comparison": label, "embedding": embedding, "ranking": ranking,
                                      "gene_set": top.index, "z": top["z"].to_numpy(),
                                      "qvalue": top["qvalue"].to_numpy(),
                                      "cell_cycle": np.asarray(top.index.str.contains(cc.CELL_CYCLE))}))
        print(f"{label}: {time.time() - tic:.0f}s", flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(cc.RESULTS_DIR, "two_truths_hela_enrichment.csv"), index=False)
    pd.concat(tops).to_csv(os.path.join(cc.RESULTS_DIR, "two_truths_hela_enrichment_top_sets.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
