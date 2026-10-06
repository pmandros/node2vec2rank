"""n2v2r against competitors on real cells: planted changes, cell subsamples
and null splits (HeLa, the hdWGCNA pipeline of benchmarks/cell_cycle.py).

Methods (from benchmarks/competitor_seeds.py): n2v2r (paper default), node2vec
and DeepWalk (separate embeddings + Procrustes), PLEX.I (Python port, 50
trainings), the DGCA-style correlation-difference score, and degree
difference (DeDi). Stochastic methods get a new seed in every run.

Parts:

1. planted: the G1 cells of each batch are split at random into two halves,
   so nothing differs between them; then 100 genes are changed in the second
   half. Candidates are the genes whose G1 degree is above the median (genes
   with no partners cannot change their network). Two kinds of change:
   - "loss": each planted gene's values are shuffled across the cells of a
     batch, which keeps its expression and breaks its co-expression (as in
     benchmarks/cell_cycle_validation.py). Its degree drops.
   - "partner": each planted gene's values are replaced by those of a donor
     gene (another candidate, |cor| < 0.1 with it) plus independent noise of
     the same variance, so the gene takes on the donor's partners while
     staying similarly connected (a module switch).
   Score: AUROC of planted genes against the rest, planted genes in the top
   100, and the planted genes' mean degree change (to see how degree-like each
   change is). 3 replicates per kind.
2. subsample: the G1 -> S comparison on 5 subsamples of 80% of the cells of
   each phase and batch. Score: mean Spearman and top-100 overlap between
   subsamples (data noise plus, for random methods, seed noise).
3. null: 3 random G1-vs-G1 splits. Score: Spearman of each ranking with
   degree, and how many of its top 100 are also in its own top 100 for the
   real G1 -> S comparison (a ranking that gives the same list when nothing
   changed is driven by network structure, not by change).

Run from the repository root, one part per process:

    OMP_NUM_THREADS=1 python benchmarks/competitor_hela.py --part planted --revelio-dir path/to/Revelio/data
"""

import argparse
import itertools
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
from node2vec2rank.simulate import coexpression_network  # noqa: E402
from node2vec2rank.singlecell import metacells  # noqa: E402

TOP_K = 100
NUM_PLANTED = 100


def networks(group_a, group_b, power):
    expressions = [np.asarray(metacells(g)) for g in (group_a, group_b)]
    graphs = [np.asarray(coexpression_network(e, power=power, signed=True)) for e in expressions]
    return graphs, expressions


def all_methods(graphs, expressions, seed):
    """{method: (score, seconds)}"""
    methods = {
        "n2v2r (default)": lambda: cs.n2v2r(graphs, seed),
        "node2vec + Procrustes": lambda: cs.walk_then_rank(graphs, seed, p=1, q=0.5),
        "DeepWalk + Procrustes": lambda: cs.walk_then_rank(graphs, seed, p=1, q=1),
        "PLEX.I (50 trainings)": lambda: cs.plexi(graphs, seed)[0],
        "DGCA-style": lambda: cs.dgca_score(expressions),
        "DeDi": lambda: cs.degree_difference(graphs),
    }
    out = {}
    for name, method in methods.items():
        begin = time.perf_counter()
        out[name] = (np.asarray(method(), dtype=float), time.perf_counter() - begin)
    return out


def top(score, k=TOP_K):
    return set(np.argsort(-score)[:k])


def plant(group, strata, candidates, rng, kind):
    changed = group.copy()
    values = changed.to_numpy().copy()
    planted = rng.choice(candidates, NUM_PLANTED, replace=False)
    if kind == "loss":
        for stratum in np.unique(strata):
            rows = np.flatnonzero(strata == stratum)
            for gene in planted:
                values[rows, gene] = values[rows[rng.permutation(len(rows))], gene]
    elif kind == "partner":
        original = group.to_numpy()
        correlation = np.corrcoef(original, rowvar=False)
        donors_pool = np.setdiff1d(candidates, planted)
        for gene in planted:
            options = donors_pool[np.abs(correlation[gene, donors_pool]) < 0.1]
            donor = rng.choice(options)
            mixed = original[:, donor] + original[:, donor].std() * rng.standard_normal(len(original))
            values[:, gene] = (mixed - mixed.mean()) / mixed.std() * original[:, gene].std() + original[:, gene].mean()
    else:
        raise ValueError(kind)
    changed = pd.DataFrame(values, index=group.index, columns=group.columns)
    labels = np.zeros(group.shape[1], dtype=bool)
    labels[planted] = True
    return changed, labels


def part_planted(groups, batches, power, replicates):
    g1, strata = groups["G1"], batches["G1"]
    full_graph = np.asarray(coexpression_network(np.asarray(metacells(g1)), power=power, signed=True))
    degree = full_graph.sum(axis=0)
    candidates = np.flatnonzero(degree > np.median(degree))
    rows = []
    for kind in ("loss", "partner"):
        for replicate in range(replicates):
            rng = np.random.default_rng(100 * replicate + (kind == "partner"))
            half = cc.random_halves(strata, rng)
            group_b, labels = plant(g1[~half], strata[~half], candidates, rng, kind)
            graphs, expressions = networks(g1[half], group_b, power)
            delta = (graphs[1].sum(axis=0) - graphs[0].sum(axis=0)) / graphs[0].sum(axis=0)
            for name, (score, seconds) in all_methods(graphs, expressions, replicate).items():
                rows.append(dict(kind=kind, replicate=replicate, method=name, auroc=cs.auroc(labels, score),
                                 planted_in_top100=len(top(score) & set(np.flatnonzero(labels))),
                                 planted_relative_degree_change=delta[labels].mean(),
                                 others_relative_degree_change=delta[~labels].mean(), seconds=seconds))
                print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def subsample(group, strata, rng, fraction=0.8):
    keep = np.zeros(len(strata), dtype=bool)
    for stratum in np.unique(strata):
        members = np.flatnonzero(strata == stratum)
        keep[rng.choice(members, int(round(fraction * len(members))), replace=False)] = True
    return group[keep]


def part_subsample(groups, batches, power, replicates):
    scores = {}
    for replicate in range(replicates):
        rng = np.random.default_rng(500 + replicate)
        graphs, expressions = networks(subsample(groups["G1"], batches["G1"], rng),
                                       subsample(groups["S"], batches["S"], rng), power)
        for name, (score, seconds) in all_methods(graphs, expressions, replicate).items():
            scores.setdefault(name, []).append(score)
            print(replicate, name, round(seconds, 1), flush=True)
    rows = []
    for name, runs in scores.items():
        pairs = list(itertools.combinations(range(len(runs)), 2))
        rhos = [spearmanr(runs[i], runs[j])[0] for i, j in pairs]
        overlaps = [len(top(runs[i]) & top(runs[j])) / TOP_K for i, j in pairs]
        rows.append(dict(method=name, subsamples=len(runs), spearman_mean=np.mean(rhos), spearman_min=np.min(rhos),
                         top100_overlap_mean=np.mean(overlaps), top100_overlap_min=np.min(overlaps)))
    return pd.DataFrame(rows)


def part_null(groups, batches, power, replicates):
    real_graphs, real_expressions = networks(groups["G1"], groups["S"], power)
    real = {name: score for name, (score, _) in all_methods(real_graphs, real_expressions, 0).items()}
    rows = []
    for replicate in range(replicates):
        rng = np.random.default_rng(900 + replicate)
        half = cc.random_halves(batches["G1"], rng)
        graphs, expressions = networks(groups["G1"][half], groups["G1"][~half], power)
        degree = (graphs[0].sum(axis=0) + graphs[1].sum(axis=0)) / 2
        for name, (score, seconds) in all_methods(graphs, expressions, replicate + 1).items():
            rows.append(dict(replicate=replicate, method=name, spearman_with_degree=spearmanr(score, degree)[0],
                             shared_with_real_top100=len(top(score) & top(real[name]))))
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=["planted", "subsample", "null"], required=True)
    parser.add_argument("--revelio-dir", required=True)
    parser.add_argument("--replicates", type=int)
    args = parser.parse_args(argv)
    cells_table, groups, batches, build = cc.prepare(args.revelio_dir)
    power = build.keywords["power"]
    replicates = args.replicates or {"planted": 3, "subsample": 5, "null": 3}[args.part]
    result = {"planted": part_planted, "subsample": part_subsample, "null": part_null}[args.part](
        groups, batches, power, replicates)
    os.makedirs(cs.RESULTS_DIR, exist_ok=True)
    result.to_csv(os.path.join(cs.RESULTS_DIR, f"competitor_hela_{args.part}.csv"), index=False)
    print(result.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
