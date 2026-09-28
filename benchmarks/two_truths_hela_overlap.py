"""Do the euclidean, cosine and default (Borda) rankings share genes and pathways on HeLa?

For each comparison of benchmarks/two_truths_hela_enrichment.py (G1 -> S, S -> G2, G2 -> M and two
G1-vs-G1 null splits), on the adjacency embedding (UASE), with radial as the pure hub ranking:

- genes: how many of the top ``--top`` genes of each degree-adjusted ranking are shared by each pair
  (and all three of euclidean, cosine, default), against the overlap expected by chance, with the
  number of Reactome cell-cycle genes and the median degree percentile of the shared genes. Also
  the Spearman correlation between the rankings over all genes, and an "overlap" ranking
  min(z_cosine, z_radial) that is high only for genes changing on both axes;
- pathways: the fast competitive set test (as in the enrichment script) for every ranking, and how
  many called pathways (FDR 0.1) each pair and all three share, and which only one of them calls.

Needs PR #5 (cell_cycle.py, singlecell.py, fast_gene_sets.py):

    OMP_NUM_THREADS=2 python benchmarks/two_truths_hela_overlap.py --revelio-dir path/to/Revelio/data
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
from two_truths_hela_enrichment import set_tests  # noqa: E402
from node2vec2rank.embedding import embed  # noqa: E402
from node2vec2rank.model_utils import compute_pairwise_distances  # noqa: E402
from node2vec2rank.significance import empirical_null_test  # noqa: E402

OVERLAP = "overlap: min(cosine, radial)"


def overlap_scores(group_a, group_b, build):
    graphs = [np.asarray(build(group_a), dtype=float), np.asarray(build(group_b), dtype=float)]
    degree = np.mean([g.sum(axis=0) for g in graphs], axis=0)
    embeddings = embed(graphs, max(cc.DIMENSIONS), method="uase", random_state=42)
    radial, angular, euclid = [], [], []
    for d in cc.DIMENSIONS:
        one, two = embeddings[0, :, :d], embeddings[1, :, :d]
        radial.append(np.abs(np.linalg.norm(one, axis=1) - np.linalg.norm(two, axis=1)))
        angular.append(compute_pairwise_distances(one, two, "cosine"))
        euclid.append(compute_pairwise_distances(one, two, "euclidean"))
    radial, angular, euclid = map(np.column_stack, (radial, angular, euclid))
    columns = {"cosine": angular, "radial": radial, "euclidean": euclid,
               "default": np.hstack([euclid, angular])}
    scores = {}
    for name, values in columns.items():
        z, _, _ = empirical_null_test(values, degree)
        scores[("UASE", name)] = np.nan_to_num(z)
    scores[("UASE", OVERLAP)] = np.minimum(scores[("UASE", "cosine")], scores[("UASE", "radial")])
    dedi = np.abs(graphs[1].sum(axis=0) - graphs[0].sum(axis=0))
    scores[("-", "degree difference")] = norm.ppf((rankdata(dedi) - 0.5) / len(dedi))
    scores[("-", "degree")] = degree  # not a ranking; used for the degree percentile of each region
    return scores


LISTS = ("euclidean", "cosine", "default", "radial")
GROUPS = [("euclidean", "cosine"), ("euclidean", "default"), ("cosine", "default"),
          ("euclidean", "cosine", "default"), ("radial", "cosine"), ("radial", "euclidean"),
          ("radial", "default")]


def overlaps(lists, universe_size, top=None):
    """Shared members of each group of lists, with the count expected by chance for gene lists."""
    rows = []
    for group in GROUPS:
        shared = set.intersection(*(lists[name] for name in group))
        expected = universe_size * np.prod([len(lists[name]) / universe_size for name in group])
        rows.append(dict(lists=" & ".join(group), shared=len(shared), chance=round(expected, 1),
                         members=shared))
    for name in ("euclidean", "cosine", "default"):
        others = set.union(*(lists[o] for o in ("euclidean", "cosine", "default") if o != name))
        rows.append(dict(lists=f"{name} only (of the three)", shared=len(lists[name] - others), chance=np.nan,
                         members=lists[name] - others))
    return rows


def gene_overlaps(scores, positive, top):
    degree_percentile = rankdata(scores[("-", "degree")]) / len(positive) * 100
    lists = {name: set(np.argsort(-scores[("UASE", name)])[:top]) for name in LISTS + (OVERLAP,)}
    rows = overlaps(lists, len(positive))
    for row in rows:
        members = list(row.pop("members"))
        row["cell_cycle"] = int(positive[members].sum()) if members else 0
        row["median_degree_percentile"] = float(np.median(degree_percentile[members])) if members else np.nan
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--revelio-dir", required=True)
    parser.add_argument("--num-permutations", type=int, default=20)
    parser.add_argument("--num-nulls", type=int, default=2)
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args(argv)

    cells_table, groups, batches, build = cc.prepare(args.revelio_dir)
    genes = pd.Index(groups["G1"].columns)
    reactome = cc.read_gmt(cc.REACTOME)
    cell_cycle_genes = set().union(*(m for name, m in reactome.items() if cc.CELL_CYCLE.search(name)))
    positive = np.asarray(genes.isin(list(cell_cycle_genes)))
    members = {}
    for name, gs in reactome.items():
        index = np.unique(genes.get_indexer(list(gs)))
        index = index[index >= 0]
        if 5 <= len(index) <= 500:
            members[name] = index
    names = pd.Index(list(members))
    is_cell_cycle = np.asarray(names.str.contains(cc.CELL_CYCLE))

    comparisons = [(f"{a} -> {b}", groups[a], groups[b], (batches[a], batches[b])) for a, b in cc.COMPARISONS]
    for seed in range(args.num_nulls):
        half = cc.random_halves(batches["G1"], np.random.default_rng(seed))
        comparisons.append((f"null: G1 halves (seed {seed})", groups["G1"][half], groups["G1"][~half],
                            (batches["G1"][half], batches["G1"][~half])))

    region_rows, set_rows, called_sets = [], [], []
    rng = np.random.default_rng(0)
    for label, a, b, strata in comparisons:
        tic = time.time()
        observed = overlap_scores(a, b, build)
        region_rows += [dict(comparison=label, **row) for row in gene_overlaps(observed, positive, args.top)]
        spearman = pd.DataFrame({name: observed[("UASE", name)] for name in LISTS}).corr(method="spearman")
        for one, two in GROUPS[:3] + GROUPS[4:5]:
            region_rows.append(dict(comparison=label, lists=f"Spearman {one} vs {two}",
                                    shared=round(spearman.loc[one, two], 2)))
        results = set_tests(a, b, strata, build, members, args.num_permutations, rng,
                            scorer=lambda x, y, f: {k: v for k, v in overlap_scores(x, y, f).items()
                                                    if k != ("-", "degree")})
        called = {key[1]: set(names[r["qvalue"].to_numpy() < 0.1]) for key, r in results.items()}
        for ranking, sets in called.items():
            set_rows.append(dict(comparison=label, lists=f"{ranking} (all called)", shared=len(sets),
                                 cell_cycle=int(np.isin(names, list(sets))[is_cell_cycle].sum())))
            called_sets += [dict(comparison=label, ranking=ranking, gene_set=s) for s in sorted(sets)]
        for row in overlaps(called, len(names)):
            sets = row.pop("members")
            row.pop("chance")
            set_rows.append(dict(comparison=label, **row,
                                 cell_cycle=int(np.isin(names, list(sets))[is_cell_cycle].sum())))
        print(f"{label}: {time.time() - tic:.0f}s", flush=True)

    regions, sets = pd.DataFrame(region_rows), pd.DataFrame(set_rows)
    regions.to_csv(os.path.join(cc.RESULTS_DIR, "two_truths_hela_overlap_genes.csv"), index=False)
    sets.to_csv(os.path.join(cc.RESULTS_DIR, "two_truths_hela_overlap_sets.csv"), index=False)
    pd.DataFrame(called_sets).to_csv(os.path.join(cc.RESULTS_DIR, "two_truths_hela_overlap_called.csv"),
                                     index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        for frame, title in ((regions, "top genes"), (sets, "pathways called at FDR 0.1")):
            for value in ("shared", "cell_cycle"):
                print(f"\n{title}: {value}")
                print(frame.pivot_table(index="lists", columns="comparison", values=value, sort=False).to_string())
        print("\ntop genes: chance overlap")
        print(regions.pivot_table(index="lists", columns="comparison", values="chance", sort=False).to_string())


if __name__ == "__main__":
    main()
