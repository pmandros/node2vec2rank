"""HeLa sensitivity runs for the revision (hdWGCNA pipeline of
benchmarks/cell_cycle.py, G1 -> S, always next to a G1-vs-G1 null split).

Parts:

1. dimensions: the singular values of the joint embedding and the elbow;
   for each single dimension 2..24 (and the elbow), its ranking's Spearman
   correlation and top-100 overlap with the default ranking (Borda over
   dimensions 4..24, euclidean and cosine) and its cell-cycle gene AUROC; and
   the calibrated fast set test (Reactome) with single dimensions against the
   default, for G1 -> S and the null split.
2. grid: metacell size k in {10, 25, 50} x signed soft power in {6, 8, 10, 12}.
   For each, the default ranking's agreement with the paper's setting (k = 25,
   power picked by scale-free fit, 10), cell-cycle gene AUROC, and the fast set
   test for G1 -> S and the null split.

The null split is the one of benchmarks/cell_cycle.py (random halves of the G1
cells within batch, seed 0).

    OMP_NUM_THREADS=1 python benchmarks/revision_hela.py --part dimensions --revelio-dir path/to/Revelio/data
"""

import argparse
import functools
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import cell_cycle as cc  # noqa: E402
import competitor_seeds as cs  # noqa: E402
from node2vec2rank.embedding import embed, select_dimension  # noqa: E402
from node2vec2rank.fast_gene_sets import fast_gene_set_test  # noqa: E402
from node2vec2rank.model_utils import borda_aggregate, compute_pairwise_distances  # noqa: E402
from node2vec2rank.permutation import read_gmt  # noqa: E402
from node2vec2rank.singlecell import metacell_network  # noqa: E402

TOP_K = 100
SPECTRUM_SIZE = 60


def top(score, k=TOP_K):
    return set(np.argsort(-score)[:k])


def borda(embeddings, dims):
    return borda_aggregate(np.column_stack([compute_pairwise_distances(embeddings[0][:, :d], embeddings[1][:, :d], m)
                                            for d in dims for m in ("euclidean", "cosine")]))


def set_test(comparisons, build, gene_sets, is_cell_cycle, dims):
    rows = []
    for label, (group_a, group_b, strata) in comparisons.items():
        begin = time.perf_counter()
        result = fast_gene_set_test(group_a, group_b, gene_sets, build_network=build, num_permutations=10,
                                    min_size=5, max_size=500, strata=strata, random_state=0, ranking="n2v2r",
                                    embed_dimensions=dims, distance_metrics=["euclidean", "cosine"], seed=42)
        result = result.drop(columns="score").rename(columns={"z": "score"})
        rows.append({**cc.summarise("fast set test, n2v2r", label, result, is_cell_cycle),
                     "seconds": time.perf_counter() - begin})
    return rows


def setup(revelio_dir):
    cells_table, groups, batches, build = cc.prepare(revelio_dir)
    gene_sets = read_gmt(cc.REACTOME)
    names = pd.Index(list(gene_sets))
    is_cell_cycle = pd.Series(np.asarray(names.str.contains(cc.CELL_CYCLE)), index=names)
    genes = groups["G1"].columns
    cell_cycle_genes = set().union(*(m for name, m in gene_sets.items() if cc.CELL_CYCLE.search(name)))
    gene_labels = np.asarray(genes.isin(list(cell_cycle_genes)))
    half = cc.random_halves(batches["G1"], np.random.default_rng(0))
    comparisons = {"G1 -> S": (groups["G1"], groups["S"], (batches["G1"], batches["S"])),
                   "null: G1 halves": (groups["G1"][half], groups["G1"][~half],
                                       (batches["G1"][half], batches["G1"][~half]))}
    return comparisons, build, gene_sets, is_cell_cycle, gene_labels


def part_dimensions(revelio_dir):
    comparisons, build, gene_sets, is_cell_cycle, gene_labels = setup(revelio_dir)
    rankings, spectra = [], []
    for label, (group_a, group_b, _) in comparisons.items():
        graphs = [np.asarray(build(g)) for g in (group_a, group_b)]
        embeddings, singular_values = embed(graphs, SPECTRUM_SIZE, "uase", random_state=42,
                                            return_singular_values=True)
        elbow = select_dimension(singular_values, min_dimension=2)
        spectra.append(pd.DataFrame({"comparison": label, "dimension": np.arange(1, SPECTRUM_SIZE + 1),
                                     "singular_value": np.sort(singular_values)[::-1], "elbow": elbow}))
        default = borda(embeddings, cc.DIMENSIONS)
        for name, dims in [("default (Borda 4-24)", cc.DIMENSIONS), (f"elbow ({elbow})", [elbow])] + \
                [(f"{d} only", [d]) for d in range(2, 25, 2)]:
            score = borda(embeddings, dims)
            rankings.append(dict(comparison=label, ranking=name, spearman_with_default=spearmanr(score, default)[0],
                                 top100_shared_with_default=len(top(score) & top(default)),
                                 cell_cycle_gene_auroc=cs.auroc(gene_labels, score)))
            print(rankings[-1], flush=True)
    elbow = int(spectra[0]["elbow"].iloc[0])
    tests = []
    for name, dims in [("default (Borda 4-24)", cc.DIMENSIONS), (f"elbow ({elbow})", [elbow]),
                       ("4 only", [4]), ("8 only", [8]), ("16 only", [16]), ("24 only", [24])]:
        for row in set_test(comparisons, build, gene_sets, is_cell_cycle, dims):
            tests.append({"dimensions": name, **row})
            print(tests[-1], flush=True)
    return {"spectrum": pd.concat(spectra), "rankings": pd.DataFrame(rankings), "set_test": pd.DataFrame(tests)}


def part_grid(revelio_dir):
    comparisons, paper_build, gene_sets, is_cell_cycle, gene_labels = setup(revelio_dir)
    paper_power = paper_build.keywords["power"]
    group_a, group_b, _ = comparisons["G1 -> S"]
    reference = None
    rows = []
    settings = [(25, paper_power)] + [(k, p) for k in (10, 25, 50) for p in (6, 8, 10, 12) if (k, p) != (25, paper_power)]
    for k, power in settings:
        build = functools.partial(metacell_network, k=k, power=power, signed=True)
        graphs = [np.asarray(build(g)) for g in (group_a, group_b)]
        score = cs.n2v2r(graphs, 42)
        if reference is None:
            reference = score
        row = dict(metacell_k=k, soft_power=power, spearman_with_paper=spearmanr(score, reference)[0],
                   top100_shared_with_paper=len(top(score) & top(reference)),
                   cell_cycle_gene_auroc=cs.auroc(gene_labels, score))
        for test in set_test(comparisons, build, gene_sets, is_cell_cycle, cc.DIMENSIONS):
            prefix = "null" if test["comparison"].startswith("null") else "G1_S"
            row.update({f"{prefix}_called": test["called"], f"{prefix}_cell_cycle_called": test["cell_cycle_called"],
                        f"{prefix}_precision": test["precision"], f"{prefix}_set_auroc": test["auroc_cell_cycle"]})
        rows.append(row)
        print(row, flush=True)
    return {"grid": pd.DataFrame(rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=["dimensions", "grid"], required=True)
    parser.add_argument("--revelio-dir", required=True)
    args = parser.parse_args(argv)
    results = {"dimensions": part_dimensions, "grid": part_grid}[args.part](args.revelio_dir)
    os.makedirs(cs.RESULTS_DIR, exist_ok=True)
    for name, table in results.items():
        table.to_csv(os.path.join(cs.RESULTS_DIR, f"revision_hela_{name}.csv"), index=False)
        with pd.option_context("display.width", 200, "display.max_columns", 30, "display.max_rows", 200):
            print(table.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
