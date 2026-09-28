"""Does the HeLa picture hold in simulation? Gene- and pathway-level results for each "truth" ranking.

On HeLa (benchmarks/two_truths_hela_enrichment.py) the default ranking (Borda of euclidean and cosine
over dims 4-24) was the best single list, the community ranking (cosine) found the most pathways, and
a hub ranking (radial or degree difference) told a second story at G1 -> S. This script checks the
same claim where the truth is known.

Expression follows a factor model like benchmarks/two_truths.py: 1,000 genes, 8 modules, 80% of
genes in a module, hub genes load 0.85 on their module and the rest 0.4. Between the conditions some
module genes change:

- switch: the gene moves to another module (community truth);
- hub: the gene's loading flips between 0.4 and 0.85 (hub truth);
- both: the gene does both;
- strengthen: every non-hub gene of one module goes from 0.4 to 0.75 (a pathway becoming coordinated).

Networks are the paper's signed hdWGCNA ((1 + cor) / 2)^10. Each ranking gives a degree-adjusted
z-score per gene (the empirical null of node2vec2rank.significance over the ranking's columns); the
degree difference uses normal scores. "Pathways" are gene sets drawn from the modules: positive sets
are half changed genes of one kind, null sets hold no changed gene (some are hub-rich, since hubs are
where degree-biased rankings go wrong). Sets are tested with the fast competitive test of PR #5
(set mean z minus the overall mean, null mean and variance from ``--num-permutations`` sample-label
shuffles, shared by all rankings).

Needs node2vec2rank/fast_gene_sets.py from PR #5; run it from a checkout of that branch:

    OMP_NUM_THREADS=1 python benchmarks/two_truths_enrichment_sim.py

Results: results/two_truths_enrichment_sim_genes.csv and results/two_truths_enrichment_sim_sets.csv.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from node2vec2rank.embedding import embed  # noqa: E402
from node2vec2rank.fast_gene_sets import _correlation_model, _quadratic_forms  # noqa: E402
from node2vec2rank.model_utils import compute_pairwise_distances  # noqa: E402
from node2vec2rank.significance import benjamini_hochberg, empirical_null_test  # noqa: E402
from node2vec2rank.simulate import coexpression_network  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DIMS = list(range(4, 25, 2))
NUM_MODULES = 8
KINDS = ("switch", "hub", "both", "strengthen")
SCENARIOS = {  # fraction of genes per kind; "strengthen" is a whole module
    "mixed (5% switch, 5% hub, 5% both)": dict(switch=0.05, hub=0.05, both=0.05),
    "module switches only (10%)": dict(switch=0.10),
    "hub changes only (10%)": dict(hub=0.10),
    "one module strengthens": dict(strengthen=True),
    "mixed + one module strengthens": dict(switch=0.05, hub=0.05, strengthen=True),
    "null (nothing changes)": dict(),
}


def auroc(labels, scores):
    labels = np.asarray(labels, bool)
    ranks = rankdata(np.nan_to_num(scores, nan=-np.inf))
    positives = labels.sum()
    return (ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * (len(labels) - positives))


def expression(scenario, samples, rng, num_genes=1000):
    modules = np.where(rng.random(num_genes) < 0.8, rng.integers(0, NUM_MODULES, num_genes), -1)
    hub = (rng.random(num_genes) < 0.3) & (modules >= 0)
    loadings = np.where(hub, 0.85, 0.4) * (modules >= 0)
    kind = np.full(num_genes, "none", dtype=object)
    free = np.flatnonzero(modules >= 0)
    if scenario.get("strengthen"):
        target = np.flatnonzero((modules == 0) & ~hub)
        kind[target] = "strengthen"
        free = np.flatnonzero((modules > 0))
    for k in ("switch", "hub", "both"):
        if k in scenario:
            chosen = rng.choice(free, int(scenario[k] * num_genes), replace=False)
            kind[chosen] = k
            free = np.setdiff1d(free, chosen)
    modules_after, loadings_after = modules.copy(), loadings.copy()
    moved = np.isin(kind, ["switch", "both"])
    modules_after[moved] = (modules[moved] + rng.integers(1, NUM_MODULES, moved.sum())) % NUM_MODULES
    flipped = np.isin(kind, ["hub", "both"])
    loadings_after[flipped] = np.where(hub[flipped], 0.4, 0.85)
    loadings_after[kind == "strengthen"] = 0.75
    data = []
    for mods, load in ((modules, loadings), (modules_after, loadings_after)):
        factors = rng.standard_normal((samples, NUM_MODULES))
        signal = np.where(mods >= 0, factors[:, np.maximum(mods, 0)] * load, 0.0)
        data.append(signal + rng.standard_normal((samples, num_genes)) * np.sqrt(1 - load ** 2))
    return data, kind, modules, hub


def gene_sets(kind, modules, hub, rng, per_kind=15, num_null=60):
    """Positive sets: half genes of one change kind, half unchanged genes of their (old) module.
    Null sets: unchanged genes of one module (half of them hub-rich), plus random unchanged genes."""
    unchanged = kind == "none"
    sets, labels = [], []
    for k in KINDS:
        changed = np.flatnonzero(kind == k)
        if len(changed) < 10:
            continue
        for _ in range(per_kind):
            size = rng.integers(16, 41)
            core = rng.choice(changed, size // 2, replace=False)
            module = np.bincount(modules[core][modules[core] >= 0], minlength=NUM_MODULES).argmax()
            pool = np.flatnonzero(unchanged & (modules == module))
            rest = rng.choice(pool, min(size - size // 2, len(pool)), replace=False)
            sets.append(np.r_[core, rest])
            labels.append(k)
    for i in range(num_null):
        size = rng.integers(16, 41)
        if i < num_null // 3:
            pool = np.flatnonzero(unchanged)
            sets.append(rng.choice(pool, size, replace=False))
            labels.append("null: random")
            continue
        module = rng.integers(0, NUM_MODULES)
        pool = np.flatnonzero(unchanged & (modules == module))
        if i % 2:  # hub-rich: 3/4 hubs
            hubs, others = pool[hub[pool]], pool[~hub[pool]]
            chosen = np.r_[rng.choice(hubs, min(3 * size // 4, len(hubs)), replace=False),
                           rng.choice(others, min(size // 4, len(others)), replace=False)]
            labels.append("null: hub-rich module")
        else:
            chosen = rng.choice(pool, min(size, len(pool)), replace=False)
            labels.append("null: module")
        sets.append(chosen)
    return sets, np.array(labels)


def node_scores(data_a, data_b):
    graphs = [coexpression_network(x, power=10, signed=True) for x in (data_a, data_b)]
    degree = np.mean([g.sum(axis=0) for g in graphs], axis=0)
    embeddings = embed(graphs, max(DIMS), method="uase", random_state=0)
    radial, angular, euclid = {}, {}, {}
    for d in DIMS:
        one, two = embeddings[0, :, :d], embeddings[1, :, :d]
        radial[d] = np.abs(np.linalg.norm(one, axis=1) - np.linalg.norm(two, axis=1))
        angular[d] = compute_pairwise_distances(one, two, "cosine")
        euclid[d] = compute_pairwise_distances(one, two, "euclidean")
    stack = lambda part, dims: np.column_stack([part[d] for d in dims])  # noqa: E731
    columns = {
        "default (euclidean + cosine, 4-24)": np.hstack([stack(euclid, DIMS), stack(angular, DIMS)]),
        "cosine, 4-24": stack(angular, DIMS),
        "radial, 4-24": stack(radial, DIMS),
        "euclidean, 4-24": stack(euclid, DIMS),
        "radial + cosine, 4-24": np.hstack([stack(radial, DIMS), stack(angular, DIMS)]),
        "cosine, d=8 (true rank)": stack(angular, [8]),
        "radial, d=8 (true rank)": stack(radial, [8]),
    }
    scores = {}
    for name, values in columns.items():
        z, _, q = empirical_null_test(values, degree)
        scores[name] = (np.nan_to_num(z), q)
    dedi = np.abs(graphs[1].sum(axis=0) - graphs[0].sum(axis=0))
    z = norm.ppf((rankdata(dedi) - 0.5) / len(dedi))
    scores["degree difference"] = (z, benjamini_hochberg(norm.sf(z)))
    return scores


def run(job):
    scenario_name, samples, rep, num_permutations = job
    rng = np.random.default_rng([samples, rep, list(SCENARIOS).index(scenario_name)])
    data, kind, modules, hub = expression(SCENARIOS[scenario_name], samples, rng)
    sets, set_labels = gene_sets(kind, modules, hub, rng)
    observed = node_scores(*data)
    pooled = np.vstack(data)
    null = []
    for _ in range(num_permutations):
        order = rng.permutation(len(pooled))
        null.append(node_scores(pooled[order[:samples]], pooled[order[samples:]]))
    scaled = (pooled - pooled.mean(axis=0)) / pooled.std(axis=0, ddof=1)

    def statistic(z):
        return np.array([z[index].mean() - z.mean() for index in sets])

    gene_rows, set_rows = [], []
    base = dict(scenario=scenario_name, samples=samples, rep=rep)
    for ranking, (z_obs, q_obs) in observed.items():
        for k in KINDS:
            if (kind == k).any():
                keep = np.isin(kind, [k, "none"])
                gene_rows.append(dict(base, ranking=ranking, truth=k, auroc=auroc(kind[keep] == k, z_obs[keep]),
                                      called=int(np.sum((q_obs < 0.1) & (kind == k))), total=int((kind == k).sum())))
        gene_rows.append(dict(base, ranking=ranking, truth="none (false calls)", auroc=np.nan,
                              called=int(np.sum((q_obs < 0.1) & (kind == "none"))), total=int((kind == "none").sum())))
        null_z = np.stack([n[ranking][0] for n in null])
        centres, correlations = _correlation_model(null_z, scaled, 200000, 40, rng)
        variance = _quadratic_forms(scaled, sets, centres, correlations, null_z.std(axis=0, ddof=1))
        null_mean = np.mean([statistic(z) for z in null_z], axis=0)
        set_z = (statistic(z_obs) - null_mean) / np.sqrt(np.maximum(variance * (1 + 1 / num_permutations),
                                                                    np.finfo(float).tiny))
        called = benjamini_hochberg(norm.sf(set_z)) < 0.1
        is_null = np.char.startswith(set_labels.astype(str), "null")
        for label in pd.unique(set_labels):
            members = set_labels == label
            set_rows.append(dict(base, ranking=ranking, sets=label, called=int(called[members].sum()),
                                 total=int(members.sum()),
                                 auroc=np.nan if label.startswith("null") else
                                 auroc(members[members | is_null], set_z[members | is_null])))
    return gene_rows, set_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--samples", type=int, nargs="+", default=[150, 400])
    parser.add_argument("--num-permutations", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    jobs = [(s, n, r, args.num_permutations) for s in SCENARIOS for n in args.samples for r in range(args.reps)]
    tic = time.time()
    gene_rows, set_rows = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for genes, sets in pool.map(run, jobs):
            gene_rows += genes
            set_rows += sets
            print(f"{len(set_rows)} set rows, {time.time() - tic:.0f}s", flush=True)
    genes, sets = pd.DataFrame(gene_rows), pd.DataFrame(set_rows)
    genes.to_csv(os.path.join(RESULTS, "two_truths_enrichment_sim_genes.csv"), index=False)
    sets.to_csv(os.path.join(RESULTS, "two_truths_enrichment_sim_sets.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 30):
        print(genes.pivot_table(index=["samples", "scenario", "ranking"], columns="truth",
                                values=["auroc", "called"], aggfunc="mean").round(2).to_string())
        print(sets.pivot_table(index=["samples", "scenario", "ranking"], columns="sets",
                               values="called", aggfunc="mean").round(1).to_string())
        print(sets.pivot_table(index=["samples", "scenario", "ranking"], columns="sets",
                               values="auroc", aggfunc="mean").round(2).dropna(axis=1, how="all").to_string())


if __name__ == "__main__":
    main()
