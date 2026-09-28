"""Robustness check of the overlap claims between the euclidean, cosine and default rankings.

Claims checked (from benchmarks/two_truths_hela_overlap.py and the enrichment simulation):

1. At gene level the euclidean and cosine top lists share few genes, and the default takes about
   half of its top genes from each.
2. The default behaves like an average of the euclidean and cosine scores (its top list is filled by
   genes that are high in one and at least moderate in the other), not like an intersection.
3. The genes shared by euclidean and cosine are mostly genes that change on both axes (partners and
   strength), and min(cosine, radial) singles those out.
4. At pathway level the three lists call mostly the same pathways; the pathways only the default
   calls are few and not stable.

Parts (all rankings are UASE, dims 4-24, degree-adjusted z as in the other two-truths scripts):

- ``synthetic``: gene level, 10 replicates each of the co-expression factor model of
  two_truths_enrichment_sim.py (150 and 400 samples) and the mixed degree-corrected block model of
  two_truths.py (binary and Poisson weights).
- ``hela``: gene level on 10 random G1-vs-G1 splits and on 5 subsamples of 80% of the cells of each
  transition; pathway level for the three transitions and two null splits, repeated with 3
  shuffle seeds (20 shuffles each), to see which calls are stable.
- ``sim-sets``: set level in the enrichment simulation, from its per-set calls
  (results/two_truths_enrichment_sim_calls_8reps.csv, written by
  ``two_truths_enrichment_sim.py --reps 8 --suffix _8reps``).

Needs PR #5 (cell_cycle.py, singlecell.py, fast_gene_sets.py):

    OMP_NUM_THREADS=1 python benchmarks/two_truths_overlap_check.py --part synthetic
    OMP_NUM_THREADS=4 python benchmarks/two_truths_overlap_check.py --part hela --revelio-dir path/to/data
    python benchmarks/two_truths_overlap_check.py --part sim-sets
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import two_truths as tt  # noqa: E402
import two_truths_enrichment_sim as sim  # noqa: E402
from node2vec2rank.embedding import embed  # noqa: E402
from node2vec2rank.significance import empirical_null_test  # noqa: E402

RESULTS = os.path.join(HERE, "results")
DIMS = list(range(4, 25, 2))


def combined(z):
    """The four base rankings plus average-, union- and intersection-like combinations."""
    z = dict(z)
    z["mean(euclidean, cosine)"] = (z["euclidean"] + z["cosine"]) / 2
    z["max(euclidean, cosine)"] = np.maximum(z["euclidean"], z["cosine"])
    z["min(euclidean, cosine)"] = np.minimum(z["euclidean"], z["cosine"])
    z["min(cosine, radial)"] = np.minimum(z["cosine"], z["radial"])
    return z


def gene_metrics(z, kind=None, tops=(50, 100, 200)):
    """Overlaps of the top lists, how the default relates to the combinations, and (when the truth is
    known) which change kinds sit in each overlap and how well each ranking finds 'both' genes."""
    z = combined(z)
    n = len(z["default"])
    rows = {}
    for top in tops:
        best = {name: set(np.argsort(-values)[:top]) for name, values in z.items()}
        e, c, d = best["euclidean"], best["cosine"], best["default"]
        rows[f"top{top}: euclidean & cosine"] = len(e & c)
        rows[f"top{top}: euclidean & default"] = len(e & d)
        rows[f"top{top}: cosine & default"] = len(c & d)
        rows[f"top{top}: default in neither"] = len(d - e - c)
        rows[f"top{top}: chance pair"] = round(top * top / n, 1)
        for other in ("mean(euclidean, cosine)", "max(euclidean, cosine)", "min(euclidean, cosine)"):
            rows[f"top{top}: default & {other}"] = len(d & best[other])
        if top == 100:
            rank_e, rank_c = rankdata(-z["euclidean"]), rankdata(-z["cosine"])
            members = np.array(sorted(d))
            # default top genes that are in the bottom half of one of the two lists
            rows["top100: default genes in bottom half of euclidean or cosine"] = int(
                np.sum((rank_e[members] > n / 2) | (rank_c[members] > n / 2)))
            if kind is not None:
                for group, members in (("euclidean & cosine", e & c), ("euclidean only", e - c),
                                       ("cosine only", c - e), ("default in neither", d - e - c)):
                    for k in ("both", "hub", "switch", "none"):
                        rows[f"top100 kinds: {group}: {k}"] = int(np.sum(kind[list(members)] == k))
    for other in ("mean(euclidean, cosine)", "max(euclidean, cosine)", "min(euclidean, cosine)"):
        rows[f"Spearman: default vs {other}"] = round(spearmanr(z["default"], z[other])[0], 3)
    rows["Spearman: euclidean vs cosine"] = round(spearmanr(z["euclidean"], z["cosine"])[0], 3)
    if kind is not None and (kind == "both").any():
        for name in ("default", "euclidean", "cosine", "radial", "mean(euclidean, cosine)",
                     "max(euclidean, cosine)", "min(euclidean, cosine)", "min(cosine, radial)"):
            rows[f"AUROC both vs all others: {name}"] = round(tt.auroc(kind == "both", z[name]), 3)
    return rows


# --- synthetic ---------------------------------------------------------------------------------------

SIM_SCENARIOS = ["mixed (5% switch, 5% hub, 5% both)", "module switches only (10%)", "hub changes only (10%)",
                 "null (nothing changes)"]
SIM_NAMES = {"euclidean": "euclidean, 4-24", "cosine": "cosine, 4-24", "radial": "radial, 4-24",
             "default": "default (euclidean + cosine, 4-24)"}


def coexpression_job(job):
    scenario, samples, rep = job
    rng = np.random.default_rng([samples, rep, 1000 + SIM_SCENARIOS.index(scenario)])
    data, kind, _, _ = sim.expression(sim.SCENARIOS[scenario], samples, rng)
    scores = sim.node_scores(*data)
    z = {short: scores[name][0] for short, name in SIM_NAMES.items()}
    return dict(model="co-expression", scenario=scenario, size=samples, rep=rep, **gene_metrics(z, kind))


def block_job(job):
    weights, rep = job
    rng = np.random.default_rng([rep, 2000 + (weights == "poisson")])
    graphs, kind = tt.mixed_graphs(2000, rng, weights)
    kind = np.where(kind == "community", "switch", kind).astype(object)
    degree = np.mean([g.sum(axis=0) for g in graphs], axis=0)
    embeddings = embed(graphs, max(DIMS), method="uase", random_state=rep)
    radial, angular, euclid = tt.radial_angular(embeddings, DIMS)
    columns = {"euclidean": euclid, "cosine": angular, "radial": radial, "default": np.hstack([euclid, angular])}
    z = {name: np.nan_to_num(empirical_null_test(values, degree)[0]) for name, values in columns.items()}
    return dict(model="degree-corrected blocks", scenario=f"mixed, {weights}", size=2000, rep=rep,
                **gene_metrics(z, kind))


def synthetic(reps, workers):
    jobs = [(s, n, r) for s in SIM_SCENARIOS for n in (150, 400) for r in range(reps)]
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(coexpression_job, jobs))
        rows += list(pool.map(block_job, [(w, r) for w in ("binary", "poisson") for r in range(reps)]))
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS, "two_truths_overlap_check_synthetic.csv"), index=False)
    summary = table.drop(columns=["rep"]).groupby(["model", "scenario", "size"]).agg(["mean", "min", "max"])
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 20):
        print(summary.T.round(2).to_string())


# --- HeLa --------------------------------------------------------------------------------------------

def hela(revelio_dir, num_permutations, num_seeds, num_splits, num_subsamples):
    import cell_cycle as cc
    from two_truths_hela_enrichment import set_tests
    from two_truths_hela_overlap import overlap_scores

    _, groups, batches, build = cc.prepare(revelio_dir)
    genes = pd.Index(groups["G1"].columns)
    reactome = cc.read_gmt(cc.REACTOME)
    members = {}
    for name, gs in reactome.items():
        index = np.unique(genes.get_indexer(list(gs)))
        index = index[index >= 0]
        if 5 <= len(index) <= 500:
            members[name] = index
    names = pd.Index(list(members))
    is_cell_cycle = pd.Series(np.asarray(names.str.contains(cc.CELL_CYCLE)), index=names)

    def base_z(scores):
        return {name: scores[("UASE", name)] for name in ("euclidean", "cosine", "radial", "default")}

    gene_rows = []
    for seed in range(num_splits):
        half = cc.random_halves(batches["G1"], np.random.default_rng(seed))
        scores = overlap_scores(groups["G1"][half], groups["G1"][~half], build)
        gene_rows.append(dict(comparison="null: G1 halves", replicate=seed, **gene_metrics(base_z(scores))))
    for a, b in cc.COMPARISONS:
        for seed in range(num_subsamples):
            rng = np.random.default_rng(100 + seed)
            pick = [rng.random(len(groups[p])) < 0.8 for p in (a, b)]
            scores = overlap_scores(groups[a][pick[0]], groups[b][pick[1]], build)
            gene_rows.append(dict(comparison=f"{a} -> {b}", replicate=seed, **gene_metrics(base_z(scores))))
        print(f"gene level {a} -> {b} done", flush=True)
    genes_table = pd.DataFrame(gene_rows)
    genes_table.to_csv(os.path.join(RESULTS, "two_truths_overlap_check_hela_genes.csv"), index=False)

    def scorer(x, y, f):
        scores = overlap_scores(x, y, f)
        z = combined(base_z(scores))
        return {("UASE", name): values for name, values in z.items()}

    comparisons = [(f"{a} -> {b}", groups[a], groups[b], (batches[a], batches[b])) for a, b in cc.COMPARISONS]
    for seed in range(2):
        half = cc.random_halves(batches["G1"], np.random.default_rng(seed))
        comparisons.append((f"null: G1 halves (seed {seed})", groups["G1"][half], groups["G1"][~half],
                            (batches["G1"][half], batches["G1"][~half])))
    call_rows = []
    for label, a, b, strata in comparisons:
        for seed in range(num_seeds):
            tic = time.time()
            results = set_tests(a, b, strata, build, members, num_permutations, np.random.default_rng(seed),
                                scorer=scorer)
            for (_, ranking), result in results.items():
                for name in names[result["qvalue"].to_numpy() < 0.1]:
                    call_rows.append(dict(comparison=label, seed=seed, ranking=ranking, gene_set=name,
                                          cell_cycle=bool(is_cell_cycle[name])))
            print(f"sets {label} seed {seed}: {time.time() - tic:.0f}s", flush=True)
    calls = pd.DataFrame(call_rows, columns=["comparison", "seed", "ranking", "gene_set", "cell_cycle"])
    calls.to_csv(os.path.join(RESULTS, "two_truths_overlap_check_hela_calls.csv"), index=False)

    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 20):
        print(genes_table.drop(columns="replicate").groupby("comparison").agg(["mean", "min", "max"]).T
              .round(2).to_string())
        print(summarize_calls(calls, [c[0] for c in comparisons], num_seeds).to_string())


def summarize_calls(calls, comparisons, num_seeds):
    """Per comparison: calls per ranking and seed, and the shared / list-specific pathways, both per
    seed and for 'stable' calls (called with every seed)."""
    rows = []
    for label in comparisons:
        these = calls[calls["comparison"] == label]
        per_seed = [{r: set(g["gene_set"]) for r, g in these[these["seed"] == s].groupby("ranking")}
                    for s in range(num_seeds)]
        stable = {}
        for ranking in set(these["ranking"]):
            stable[ranking] = set.intersection(*(p.get(ranking, set()) for p in per_seed))
        for version, lists in [(f"seed {s}", p) for s, p in enumerate(per_seed)] + [("stable", stable)]:
            e, c, d = (lists.get(k, set()) for k in ("euclidean", "cosine", "default"))
            row = dict(comparison=label, calls=version)
            for ranking in ("euclidean", "cosine", "default", "radial", "min(cosine, radial)",
                            "mean(euclidean, cosine)", "max(euclidean, cosine)", "min(euclidean, cosine)"):
                row[ranking] = len(lists.get(ranking, set()))
            row.update({"euclidean & cosine & default": len(e & c & d), "euclidean & default": len(e & d),
                        "cosine & default": len(c & d), "default only": len(d - e - c),
                        "euclidean or cosine, not default": len((e | c) - d),
                        "radial not in default": len(lists.get("radial", set()) - d)})
            rows.append(row)
    return pd.DataFrame(rows)


# --- set level in the simulation ------------------------------------------------------------------------

def sim_sets(path):
    calls = pd.read_csv(path, dtype={"called": str})
    short = {"euclidean, 4-24": "euclidean", "cosine, 4-24": "cosine", "radial, 4-24": "radial",
             "default (euclidean + cosine, 4-24)": "default"}
    rows = []
    for (scenario, samples, rep), group in calls.groupby(["scenario", "samples", "rep"]):
        labels = np.array(group["labels"].iloc[0].split("|"))
        called = {short.get(r, r): np.array([ch == "1" for ch in s]) for r, s in zip(group["ranking"], group["called"])}
        e, c, d = called["euclidean"], called["cosine"], called["default"]
        for kind in pd.unique(labels):
            m = labels == kind
            rows.append(dict(scenario=scenario, samples=samples, rep=rep,
                             sets="null" if kind.startswith("null") else kind, total=int(m.sum()),
                             default=int((d & m).sum()), euclidean=int((e & m).sum()), cosine=int((c & m).sum()),
                             all_three=int((e & c & d & m).sum()), default_only=int((d & ~e & ~c & m).sum()),
                             euclidean_or_cosine_not_default=int(((e | c) & ~d & m).sum()),
                             union_max=int((called["max(euclidean, cosine)"] & m).sum()),
                             intersection_min=int((called["min(euclidean, cosine)"] & m).sum())))
    table = pd.DataFrame(rows).groupby(["scenario", "sets"]).sum(numeric_only=True).drop(columns=["samples", "rep"])
    table.to_csv(os.path.join(RESULTS, "two_truths_overlap_check_sim_sets.csv"))
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(table.to_string())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=["synthetic", "hela", "sim-sets"], required=True)
    parser.add_argument("--revelio-dir")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-permutations", type=int, default=20)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-splits", type=int, default=10)
    parser.add_argument("--num-subsamples", type=int, default=5)
    args = parser.parse_args(argv)
    if args.part == "synthetic":
        synthetic(args.reps, args.workers)
    elif args.part == "hela":
        hela(args.revelio_dir, args.num_permutations, args.num_seeds, args.num_splits, args.num_subsamples)
    else:
        sim_sets(os.path.join(RESULTS, "two_truths_enrichment_sim_calls_8reps.csv"))


if __name__ == "__main__":
    main()
