"""Two truths on the paper's HeLa cell-cycle networks: which genes does each truth pick?

Uses the pipeline of benchmarks/cell_cycle.py (Revelio HeLa S3 cells, 4 phases, 2,000 most variable
genes, metacells per phase, hdWGCNA signed networks ((1 + cor) / 2)^power), so it needs that script and
node2vec2rank.singlecell from PR #5.

For each sequential comparison, and for random halves of the G1 cells (null), the joint UASE (dims
4-24) is split per gene into
- the radial part, the change of the embedding norm: hub status (the "core-periphery" truth);
- the angular part, the cosine distance: direction, i.e. module membership (the "affinity" truth);
and compared with the paper's default (Borda of euclidean and cosine) and the degree difference (DeDi).

Reported per ranking: Spearman with degree, top-100 overlap with the default, and the AUROC of genes
in Reactome cell-cycle pathways (the expected biology). The null splits give the AUROC that degree or
module structure alone produce, which is the baseline to beat.

    OMP_NUM_THREADS=1 python benchmarks/two_truths_hela.py --revelio-dir path/to/Revelio/data
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import cell_cycle as cc  # noqa: E402
from node2vec2rank.embedding import embed  # noqa: E402
from node2vec2rank.model_utils import borda_aggregate, compute_pairwise_distances  # noqa: E402

TOP = 100


def spearman(a, b):
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]


def rankings(graphs, method="uase"):
    embeddings = embed(graphs, max(cc.DIMENSIONS), method=method, random_state=42)
    radial, angular, euclid = [], [], []
    for d in cc.DIMENSIONS:
        one, two = embeddings[0, :, :d], embeddings[1, :, :d]
        radial.append(np.abs(np.linalg.norm(one, axis=1) - np.linalg.norm(two, axis=1)))
        angular.append(compute_pairwise_distances(one, two, "cosine"))
        euclid.append(compute_pairwise_distances(one, two, "euclidean"))
    radial, angular, euclid = map(np.column_stack, (radial, angular, euclid))
    return {"default (euclidean + cosine)": borda_aggregate(np.hstack([euclid, angular])),
            "euclidean": borda_aggregate(euclid),
            "cosine (community)": borda_aggregate(angular),
            "radial (hub)": borda_aggregate(radial),
            "radial + cosine": borda_aggregate(np.hstack([radial, angular])),
            "DeDi": np.abs(graphs[1].sum(axis=0) - graphs[0].sum(axis=0))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--revelio-dir", required=True)
    parser.add_argument("--num-nulls", type=int, default=3)
    args = parser.parse_args(argv)

    cells_table, groups, batches, build = cc.prepare(args.revelio_dir)
    genes = pd.Index(groups["G1"].columns)
    reactome = cc.read_gmt(cc.REACTOME)
    cell_cycle_genes = set().union(*(members for name, members in reactome.items() if cc.CELL_CYCLE.search(name)))
    positive = np.asarray(genes.isin(list(cell_cycle_genes)))
    print(f"{positive.sum()} of {len(genes)} genes are in Reactome cell-cycle pathways", flush=True)

    comparisons = [(f"{a} -> {b}", groups[a], groups[b]) for a, b in cc.COMPARISONS]
    for seed in range(args.num_nulls):
        half = cc.random_halves(batches["G1"], np.random.default_rng(seed))
        comparisons.append((f"null: G1 halves (seed {seed})", groups["G1"][half], groups["G1"][~half]))

    rows = []
    for label, a, b in comparisons:
        graphs = [np.asarray(build(a), dtype=float), np.asarray(build(b), dtype=float)]
        degree = np.mean([g.sum(axis=0) for g in graphs], axis=0)
        dedi = np.abs(graphs[1].sum(axis=0) - graphs[0].sum(axis=0))
        for method in ("uase", "ulse"):
            scores = rankings(graphs, method)
            if method == "ulse":
                scores = {k: v for k, v in scores.items() if k != "DeDi"}
            default_top = set(np.argsort(-scores["default (euclidean + cosine)"])[:TOP])
            for name, score in scores.items():
                top = set(np.argsort(-score)[:TOP])
                rows.append(dict(comparison=label, embedding=method.upper(), ranking=name,
                                 auroc_cell_cycle=cc.auroc(positive, score),
                                 cell_cycle_in_top100=int(positive[list(top)].sum()),
                                 spearman_degree=spearman(score, degree),
                                 spearman_dedi=spearman(score, dedi),
                                 overlap_default_top100=len(top & default_top)))
        print(label, flush=True)

    results = pd.DataFrame(rows)
    os.makedirs(cc.RESULTS_DIR, exist_ok=True)
    results.to_csv(os.path.join(cc.RESULTS_DIR, "two_truths_hela.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.max_columns", 20):
        print(results.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
