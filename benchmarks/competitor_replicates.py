"""Accuracy of n2v2r against the competitors over many replicates.

The seed test (benchmarks/competitor_seeds.py) ranked one simulated network
pair per setting, and the HeLa benchmark (benchmarks/competitor_hela.py) used
3 planted-change replicates. This script repeats both with more replicates and
compares methods replicate by replicate (paired), so that differences can be
told apart from replicate noise.

Parts:

1. sims: 10 replicates (a new network pair each, and a new seed for every
   random method) of six settings:
   - "block switch": degree-corrected block model, 1,000 nodes, 4 blocks, 10%
     of nodes switch block (benchmarks/simulations.py; the seed test's setting);
   - "co-expression switch": 1,000 genes, 5 modules, 10% of genes switch
     module (the seed test's setting);
   - "block split" / "block merge" / "co-expression split" / "co-expression
     merge": 8 communities, one splits or two merge (benchmarks/revision_sims.py).
   Methods: n2v2r default, n2v2r at the number of communities (an oracle
   dimension, for reference only), node2vec and DeepWalk + Procrustes, PLEX.I
   (50 trainings), DGCA-style (co-expression only) and DeDi.
2. planted: more replicates of competitor_hela.py's planted changes (same
   seeds, so replicates 0-2 reproduce the earlier ones), one kind per process.

Summaries (mean AUROC, and for every method against n2v2r default: mean
paired difference, its standard error, replicates won, and the Wilcoxon
signed-rank p-value):

    OMP_NUM_THREADS=1 python benchmarks/competitor_replicates.py --part sims --settings "block switch" "block split"

Replicates already in the result files are skipped. With --max-new 1 a process runs one replicate and
exits with status 3 while more remain, so a shell loop keeps memory down:

    while OMP_NUM_THREADS=1 python benchmarks/competitor_replicates.py --part sims --max-new 1; [ $? -eq 3 ]; do :; done
    OMP_NUM_THREADS=1 python benchmarks/competitor_replicates.py --part planted --kind partner --replicates 10 --revelio-dir ...
    python benchmarks/competitor_replicates.py --summarise
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import competitor_seeds as cs  # noqa: E402
import revision_sims as rs  # noqa: E402
import simulations as sims  # noqa: E402
from node2vec2rank.embedding import embed  # noqa: E402
from node2vec2rank.model_utils import borda_aggregate, compute_pairwise_distances  # noqa: E402

NUM_REPLICATES = 10
REFERENCE = "n2v2r (default)"


def block_switch(rng):
    graphs, changed = sims.simulate_sbm(rng, frac_changed=0.1)
    return graphs, changed, None, 4


def coexpression_switch(rng):
    graphs, changed, expressions = cs.simulate_coexpression(rng)
    return graphs, changed, expressions, 5


SETTINGS = {
    "block switch": block_switch,
    "co-expression switch": coexpression_switch,
    "block split": lambda rng: (*rs.sbm_pair(rng, "split"), 9),
    "block merge": lambda rng: (*rs.sbm_pair(rng, "merge"), 8),
    "co-expression split": lambda rng: (*rs.coexpression_pair(rng, "split"), 9),
    "co-expression merge": lambda rng: (*rs.coexpression_pair(rng, "merge"), 8),
}


def n2v2r_at(graphs, d, seed):
    embeddings = embed(graphs, d, "uase", random_state=seed)
    return borda_aggregate(np.column_stack([compute_pairwise_distances(embeddings[0], embeddings[1], m)
                                            for m in ("euclidean", "cosine")]))


def methods(graphs, expressions, communities, seed):
    out = {
        REFERENCE: lambda: cs.n2v2r(graphs, seed),
        "n2v2r at the number of communities (oracle)": lambda: n2v2r_at(graphs, communities, seed),
        "node2vec + Procrustes": lambda: cs.walk_then_rank(graphs, seed, p=1, q=0.5),
        "DeepWalk + Procrustes": lambda: cs.walk_then_rank(graphs, seed, p=1, q=1),
        "PLEX.I (50 trainings)": lambda: cs.plexi(graphs, seed)[0],
        "DeDi": lambda: cs.degree_difference(graphs),
    }
    if expressions is not None:
        out["DGCA-style"] = lambda: cs.dgca_score(expressions)
    return out


def _existing(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["replicate"])


def part_sims(settings, replicates, max_new):
    """Runs up to max_new replicates not yet in the result files (one process
    per replicate keeps memory down: the walk embeddings do not free all of
    theirs). Returns the number of replicates still missing."""
    done = 0
    for setting in settings:
        path = os.path.join(cs.RESULTS_DIR, "competitor_replicates_sims_" + setting.replace(" ", "_") + ".csv")
        for replicate in range(replicates):
            table = _existing(path)
            if replicate in set(table["replicate"]):
                continue
            if done == max_new:
                return 1
            rng = np.random.default_rng(7000 + 100 * list(SETTINGS).index(setting) + replicate)
            graphs, changed, expressions, communities = SETTINGS[setting](rng)
            rows = []
            for name, method in methods(graphs, expressions, communities, replicate).items():
                begin = time.perf_counter()
                score = np.asarray(method(), dtype=float)
                rows.append(dict(setting=setting, replicate=replicate, method=name,
                                 auroc=cs.auroc(changed, score),
                                 precision_at_k=sims.precision_at_k(changed, score),
                                 seconds=time.perf_counter() - begin))
            print(setting, replicate, {r["method"]: round(r["auroc"], 3) for r in rows}, flush=True)
            pd.concat([table, pd.DataFrame(rows)]).to_csv(path, index=False)
            done += 1
    return 0


def part_planted(revelio_dir, kind, replicates, max_new):
    import cell_cycle as cc
    import competitor_hela as ch
    from node2vec2rank.simulate import coexpression_network
    from node2vec2rank.singlecell import metacells
    cells_table, groups, batches, build = cc.prepare(revelio_dir)
    power = build.keywords["power"]
    g1, strata = groups["G1"], batches["G1"]
    full_graph = np.asarray(coexpression_network(np.asarray(metacells(g1)), power=power, signed=True))
    degree = full_graph.sum(axis=0)
    candidates = np.flatnonzero(degree > np.median(degree))
    path = os.path.join(cs.RESULTS_DIR, f"competitor_replicates_planted_{kind}.csv")
    done = 0
    for replicate in range(replicates):
        table = _existing(path)
        if replicate in set(table["replicate"]):
            continue
        if done == max_new:
            return 1
        # the seeds of competitor_hela.part_planted
        rng = np.random.default_rng(100 * replicate + (kind == "partner"))
        half = cc.random_halves(strata, rng)
        group_b, labels = ch.plant(g1[~half], strata[~half], candidates, rng, kind)
        graphs, expressions = ch.networks(g1[half], group_b, power)
        rows = []
        for name, (score, seconds) in ch.all_methods(graphs, expressions, replicate).items():
            rows.append(dict(setting=f"HeLa planted {kind}", replicate=replicate, method=name,
                             auroc=cs.auroc(labels, score),
                             precision_at_k=len(ch.top(score) & set(np.flatnonzero(labels))) / ch.NUM_PLANTED,
                             seconds=seconds))
        print(kind, replicate, {r["method"]: round(r["auroc"], 3) for r in rows}, flush=True)
        pd.concat([table, pd.DataFrame(rows)]).to_csv(path, index=False)
        done += 1
    return 0


def paired_summary(table):
    rows = []
    for setting, block in table.groupby("setting", sort=False):
        wide = block.pivot(index="replicate", columns="method", values="auroc").dropna()
        reference = wide[REFERENCE]
        for method in block["method"].unique():
            values = wide[method]
            difference = values - reference
            row = dict(setting=setting, method=method, replicates=len(values), auroc_mean=values.mean(),
                       auroc_sd=values.std(), seconds_mean=block.loc[block.method == method, "seconds"].mean())
            if method != REFERENCE:
                row.update(minus_n2v2r_mean=difference.mean(),
                           minus_n2v2r_se=difference.std() / np.sqrt(len(difference)),
                           replicates_better_than_n2v2r=int((difference > 0).sum()),
                           wilcoxon_p=wilcoxon(difference).pvalue if (difference != 0).any() else 1.0)
            rows.append(row)
    return pd.DataFrame(rows)


def summarise():
    files = sorted(glob.glob(os.path.join(cs.RESULTS_DIR, "competitor_replicates_*.csv")))
    files = [f for f in files if not f.endswith("_summary.csv")]
    table = pd.concat([pd.read_csv(f) for f in files])
    summary = paired_summary(table)
    summary.to_csv(os.path.join(cs.RESULTS_DIR, "competitor_replicates_summary.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.max_columns", 20):
        print(summary.round(3).to_string())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=["sims", "planted"])
    parser.add_argument("--settings", nargs="+", choices=list(SETTINGS), default=list(SETTINGS))
    parser.add_argument("--kind", choices=["loss", "partner"])
    parser.add_argument("--replicates", type=int, default=NUM_REPLICATES)
    parser.add_argument("--revelio-dir")
    parser.add_argument("--summarise", action="store_true")
    parser.add_argument("--max-new", type=int, default=-1,
                        help="stop after this many new replicates and exit with status 3 if more remain")
    args = parser.parse_args(argv)
    os.makedirs(cs.RESULTS_DIR, exist_ok=True)
    remaining = 0
    if args.part == "sims":
        remaining = part_sims(args.settings, args.replicates, args.max_new)
    elif args.part == "planted":
        remaining = part_planted(args.revelio_dir, args.kind, args.replicates, args.max_new)
    if args.summarise:
        summarise()
    sys.exit(3 if remaining else 0)


if __name__ == "__main__":
    main()
