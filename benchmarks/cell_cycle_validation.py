"""Ground-truth checks for the pathway tests on the HeLa cells of
``cell_cycle.py``: which calls are true, and are they reproducible?

1. Planted changes in real cells. The G1 cells of each batch are split at
   random into two halves, so no gene differs between them. In the second
   half, a known fraction of the genes of a few Reactome sets (15-100 genes in
   the data, no gene shared between planted sets) lose their co-expression:
   each chosen gene's values are shuffled across the cells of a batch, which
   keeps its expression distribution and breaks its correlation with every
   other gene. The planted sets are the true positives; the sets that contain
   none of the shuffled genes are the true negatives (sets that share some
   shuffled genes are left out). For every method we report the recall of
   planted sets at FDR 0.1, the false calls among the negatives, and the
   AUROC of planted sets against negatives by the method's score.
2. Reproducibility. The cells of each phase are split into two independent
   halves (within batch), and the G1 -> S and G2 -> M comparisons are run on
   each half, with smaller metacells (10 cells, at most 5 shared), since half a
   G2 or M phase (about 90 cells) gives only two metacells of 25. Without any ground truth, a method whose results mean something
   should give similar results on independent cells: we report the Spearman
   correlation of set scores between halves, the Jaccard index of the sets
   called at FDR 0.1, and the overlap of the top 20 sets.

Run from the repository root (about 75 minutes on 4 cores):

    OMP_NUM_THREADS=1 python benchmarks/cell_cycle_validation.py --revelio-dir path/to/Revelio/data
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
sys.path.insert(0, HERE)

from cell_cycle import (METHODS, REACTOME, RESULTS_DIR, auroc, prepare, random_halves,  # noqa: E402
                        read_gmt, score_methods)
from node2vec2rank.singlecell import metacell_network  # noqa: E402

Q_THRESHOLD = 0.1


def plant(group, strata, gene_sets, rng, num_sets, fraction, min_size=15, max_size=100):
    """Shuffles a fraction of the genes of ``num_sets`` disjoint gene sets
    across the cells of every stratum. Returns the changed copy, the planted
    set names and the shuffled genes."""
    columns = pd.Index(group.columns)
    in_data = {name: [g for g in pd.unique(np.asarray(list(genes), dtype=object)) if g in columns]
               for name, genes in gene_sets.items()}
    candidates = [name for name, genes in in_data.items() if min_size <= len(genes) <= max_size]
    planted, used = [], set()
    for name in rng.permutation(candidates):
        if not used.intersection(in_data[name]):
            planted.append(name)
            used.update(in_data[name])
        if len(planted) == num_sets:
            break
    shuffled = []
    for name in planted:
        genes = in_data[name]
        shuffled += list(rng.choice(genes, max(1, round(fraction * len(genes))), replace=False))
    changed = group.copy()
    for value in np.unique(strata):
        rows = np.flatnonzero(strata == value)
        for gene in shuffled:
            position = columns.get_loc(gene)
            changed.iloc[rows, position] = changed.iloc[rows, position].to_numpy()[rng.permutation(len(rows))]
    return changed, planted, set(shuffled), in_data


def spike_in(groups, batches, gene_sets, build, args):
    records = []
    for fraction in args.fractions:
        for replicate in range(args.replicates):
            tic = time.time()
            rng = np.random.default_rng(1000 * replicate + int(100 * fraction))
            half = random_halves(batches["G1"], rng)
            group_a, group_b = groups["G1"][half], groups["G1"][~half]
            strata = (batches["G1"][half], batches["G1"][~half])
            group_b, planted, shuffled, in_data = plant(group_b, strata[1], gene_sets, rng, args.num_planted,
                                                        fraction)
            results = score_methods(group_a, group_b, strata, gene_sets, build, args.num_permutations,
                                    args.n_jobs, args.methods)
            for method, result in results.items():
                positive = result.index.isin(planted)
                negative = np.array([not shuffled.intersection(in_data.get(name, ())) for name in result.index])
                keep = positive | negative
                called = (result["qvalue"] <= Q_THRESHOLD).to_numpy()
                ranks = result.loc[keep, "score"].rank(ascending=False)
                records.append({"fraction_shuffled": fraction, "replicate": replicate, "method": method,
                                "planted": int(positive.sum()), "negatives": int(negative.sum()),
                                "recall": called[positive].mean(),
                                "false_calls": int(called[negative].sum()),
                                "auroc": auroc(positive[keep], result.loc[keep, "score"].to_numpy()),
                                "median_rank_of_planted": float(ranks[positive[keep]].median())})
            print(f"spike-in fraction {fraction} replicate {replicate}: {time.time() - tic:.0f}s", flush=True)
    return pd.DataFrame(records)


def reproducibility(groups, batches, gene_sets, build, args):
    # half of a G2 or M phase is about 90 cells, which gives only a couple of
    # metacells with k = 25 and 10 shared cells; smaller metacells keep ~50
    build = functools.partial(metacell_network, power=build.keywords["power"], signed=True,
                              k=args.repro_k, max_shared=args.repro_max_shared)
    records = []
    rng = np.random.default_rng(7)
    halves = {phase: random_halves(batches[phase], rng) for phase in groups}
    for a, b in (("G1", "S"), ("G2", "M")):
        tic = time.time()
        runs = []
        for side in (True, False):
            mask_a, mask_b = halves[a] == side, halves[b] == side
            runs.append(score_methods(groups[a][mask_a], groups[b][mask_b],
                                      (batches[a][mask_a], batches[b][mask_b]), gene_sets, build,
                                      args.num_permutations, args.n_jobs, args.methods))
        for method in runs[0]:
            first, second = runs[0][method], runs[1][method].reindex(runs[0][method].index)
            calls = [set(r.index[r["qvalue"] <= Q_THRESHOLD]) for r in (first, second)]
            tops = [set(r["score"].sort_values(ascending=False).index[:20]) for r in (first, second)]
            union = calls[0] | calls[1]
            records.append({"comparison": f"{a} -> {b}", "method": method,
                            "spearman_scores": spearmanr(first["score"], second["score"], nan_policy="omit")[0],
                            "called_half_1": len(calls[0]), "called_half_2": len(calls[1]),
                            "called_both": len(calls[0] & calls[1]),
                            "jaccard_calls": len(calls[0] & calls[1]) / len(union) if union else np.nan,
                            "top20_overlap": len(tops[0] & tops[1])})
        print(f"reproducibility {a} -> {b}: {time.time() - tic:.0f}s", flush=True)
    return pd.DataFrame(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--revelio-dir", required=True, help="the data folder of the Revelio repository")
    parser.add_argument("--num-genes", type=int, default=2000)
    parser.add_argument("--num-permutations", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--num-planted", type=int, default=8)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.3, 0.6])
    parser.add_argument("--skip-reproducibility", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--suffix", default="", help="appended to the result file names")
    parser.add_argument("--repro-k", type=int, default=10, help="cells per metacell for the split halves")
    parser.add_argument("--repro-max-shared", type=int, default=5)
    args = parser.parse_args(argv)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    _, groups, batches, build = prepare(args.revelio_dir, args.num_genes)
    gene_sets = read_gmt(REACTOME)

    spikes = spike_in(groups, batches, gene_sets, build, args)
    spikes.to_csv(os.path.join(RESULTS_DIR, f"cell_cycle_spike_in{args.suffix}.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(spikes.groupby(["fraction_shuffled", "method"])[
            ["recall", "false_calls", "auroc", "median_rank_of_planted"]].mean().round(3))
    if not args.skip_reproducibility:
        repro = reproducibility(groups, batches, gene_sets, build, args)
        repro.to_csv(os.path.join(RESULTS_DIR, f"cell_cycle_reproducibility{args.suffix}.csv"), index=False)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(repro.round(3))


if __name__ == "__main__":
    main()
