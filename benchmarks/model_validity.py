"""Does n2v2r recover the population ranking? Checks the model assumptions behind the claim.

The consistency results for the unfolded adjacency spectral embedding (UASE) say that, when the
graphs' entries are independent with a low-rank mean, every node's estimated per-graph position
converges uniformly to its population position, up to one orthogonal matrix shared by all graphs.
Euclidean and cosine distances between a node's positions are invariant to that matrix, so the
ranking by estimated distance converges to the ranking by population distance. This script checks:

1. weighted graphs with independent entries (Gaussian and Poisson weights): the claim holds, but
   among nodes whose population distance is tied (for example every unchanged node) the order
   follows each node's noise level, and dimensions beyond the true rank add noise;
2. co-expression networks (|cor|^6), which break those assumptions (dependent entries, a target that
   is the powered correlation matrix rather than a low-rank mean): whether the ranking of rewired
   genes still converges to the ranking on the population network as the number of samples grows,
   and whether, with nothing rewired, the ranking is systematic (reproducible across independent
   replicates) rather than random, including when the two conditions have different sample sizes.

Run with ``python benchmarks/model_validity.py`` (about a minute on 4 cores); the numbers are
written to ``results/model_validity_*.csv``.
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from node2vec2rank.embedding import uase
from node2vec2rank.model_utils import borda_aggregate, compute_pairwise_distances

RESULTS = os.path.join(os.path.dirname(__file__), "results")
DIMS = list(range(4, 25, 2))


def roc_auc_score(labels, scores):
    labels = np.asarray(labels, bool)
    ranks = rankdata(scores)
    positives = labels.sum()
    return (ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * (len(labels) - positives))


def distances(graphs, dims, metrics=("euclidean", "cosine"), seed=0):
    embeddings = uase(graphs, max(dims), random_state=seed)
    columns = {}
    for d in dims:
        for metric in metrics:
            columns[(d, metric)] = compute_pairwise_distances(
                embeddings[0, :, :d], embeddings[1, :, :d], metric)
    return pd.DataFrame(columns)


def population_distances(means, rank):
    return distances(means, [rank])


# --- 1. weighted graphs with independent entries -----------------------------------------------

def latent_positions(n, rank, rng, frac_changed=0.2):
    """Mixed-membership positions with degree heterogeneity; changed nodes move by a random amount."""
    centres = np.eye(rank) * 0.6 + 0.1
    weights = rng.dirichlet(np.full(rank, 0.3), n)
    degree = rng.uniform(0.4, 1.0, n)
    before = degree[:, None] * weights @ centres
    changed = rng.random(n) < frac_changed
    amount = np.where(changed, rng.uniform(0.1, 1.0, n), 0.0)
    target = degree[:, None] * rng.dirichlet(np.full(rank, 0.3), n) @ centres
    after = (1 - amount[:, None]) * before + amount[:, None] * target
    return before, after, changed, amount, degree


def symmetric_noise(mean, kind, rng, scale):
    n = mean.shape[0]
    if kind == "gaussian":
        noise = rng.normal(0, scale, (n, n))
        upper = np.triu(mean + noise, 1)
    else:  # poisson: counts with mean scale * P, rescaled back to P's scale
        upper = np.triu(rng.poisson(scale * mean) / scale, 1)
    return upper + upper.T


def weighted_experiment(rng, reps=5):
    rows = []
    rank = 4
    for n in (300, 1000, 3000):
        for kind, scale in (("gaussian", 0.3), ("poisson", 2.0)):
            for rep in range(reps):
                before, after, changed, amount, degree = latent_positions(n, rank, rng)
                means = [before @ before.T, after @ after.T]
                for mean in means:
                    np.fill_diagonal(mean, 0)
                graphs = [symmetric_noise(m, kind, rng, scale) for m in means]
                pop = population_distances(means, rank)
                est = distances(graphs, DIMS, seed=rep)
                borda = borda_aggregate(est.to_numpy())
                for metric in ("euclidean", "cosine"):
                    p, e = pop[(rank, metric)].to_numpy(), est[(rank, metric)].to_numpy()
                    rows.append(dict(
                        noise=kind, n=n, rep=rep, metric=metric,
                        spearman_all=spearmanr(p, e)[0],
                        spearman_changed=spearmanr(p[changed], e[changed])[0],
                        max_abs_error=np.max(np.abs(p - e)),
                        auroc_true_rank=roc_auc_score(changed, e),
                        unchanged_order_vs_degree=spearmanr(e[~changed], degree[~changed])[0]))
                rows.append(dict(
                    noise=kind, n=n, rep=rep, metric="borda 4..24",
                    spearman_all=spearmanr(pop[(rank, "euclidean")], borda)[0],
                    spearman_changed=spearmanr(pop[(rank, "euclidean")].to_numpy()[changed],
                                               borda[changed])[0],
                    max_abs_error=np.nan,
                    auroc_true_rank=roc_auc_score(changed, borda),
                    unchanged_order_vs_degree=spearmanr(borda[~changed], degree[~changed])[0]))
    return pd.DataFrame(rows)


# --- 2. co-expression networks -------------------------------------------------------------------

def module_correlation(n, num_modules, rng, frac_in_modules=0.7, frac_rewired=0.1):
    """Correlation matrices of the factor model in simulate.py, before and after rewiring."""
    in_module = rng.random(n) < frac_in_modules
    before = np.where(in_module, rng.integers(0, num_modules, n), -1)
    after = before.copy()
    rewired = np.zeros(n, bool)
    pool = np.flatnonzero(in_module)
    chosen = rng.choice(pool, int(frac_rewired * n), replace=False)
    after[chosen] = (before[chosen] + rng.integers(1, num_modules, len(chosen))) % num_modules
    rewired[chosen] = True
    loadings = rng.uniform(0.3, 0.9, n)

    def loading_matrix(modules):
        load = np.zeros((n, num_modules))
        member = modules >= 0
        load[np.flatnonzero(member), modules[member]] = loadings[member]
        return load

    return loading_matrix(before), loading_matrix(after), rewired, loadings * in_module


def sample_network(load, m, rng, power=6):
    n, k = load.shape
    values = rng.standard_normal((m, k)) @ load.T \
        + rng.standard_normal((m, n)) * np.sqrt(1 - (load ** 2).sum(axis=1))
    adjacency = np.abs(np.corrcoef(values, rowvar=False)) ** power
    np.fill_diagonal(adjacency, 0)
    return adjacency


def population_network(load, power=6):
    correlation = load @ load.T
    adjacency = np.abs(correlation) ** power
    np.fill_diagonal(adjacency, 0)
    return adjacency


def coexpression_experiment(rng, reps=5, n=1000, num_modules=5):
    rows, spectra = [], []
    for rep in range(reps):
        load_before, load_after, rewired, strength = module_correlation(n, num_modules, rng)
        target = [population_network(load_before), population_network(load_after)]
        pop = population_distances(target, num_modules)
        if rep == 0:
            singular_values = np.linalg.svd(np.hstack(target), compute_uv=False)[:30]
            spectra.append(pd.DataFrame({"index": np.arange(1, 31), "singular_value": singular_values}))

        # (a) rewiring, equal sample sizes: how close is the ranking to the intended target?
        for m in (50, 150, 500, 2000):
            graphs = [sample_network(load_before, m, rng), sample_network(load_after, m, rng)]
            est = distances(graphs, sorted(set(DIMS) | {num_modules}), seed=rep)
            borda = borda_aggregate(est[DIMS].to_numpy())
            p = pop[(num_modules, "euclidean")].to_numpy()
            e = est[(num_modules, "euclidean")].to_numpy()
            rows.append(dict(experiment="rewiring", samples=f"{m} vs {m}", rep=rep,
                             spearman_vs_target=spearmanr(p, borda)[0],
                             spearman_vs_target_rewired=spearmanr(p[rewired], borda[rewired])[0],
                             spearman_vs_target_rewired_rank_d=spearmanr(p[rewired], e[rewired])[0],
                             auroc_rewired=roc_auc_score(rewired, borda),
                             null_order_vs_strength=spearmanr(borda[~rewired], strength[~rewired])[0]))

        # (b) no rewiring at all: identical correlations, equal or unequal sample sizes
        for m1, m2 in ((150, 150), (150, 600), (50, 50), (50, 500)):
            bordas = []
            for _ in range(2):
                graphs = [sample_network(load_before, m1, rng), sample_network(load_before, m2, rng)]
                bordas.append(borda_aggregate(distances(graphs, DIMS, seed=rep).to_numpy()))
            top = lambda b: rankdata(-b) <= 0.05 * n  # noqa: E731
            rows.append(dict(experiment="no rewiring", samples=f"{m1} vs {m2}", rep=rep,
                             # a reproducible ranking under no change means a systematic artefact
                             replicate_agreement=spearmanr(bordas[0], bordas[1])[0],
                             null_order_vs_strength=spearmanr(bordas[0], strength)[0],
                             top5pct_overlap=np.mean(top(bordas[0])[top(bordas[1])])))
    return pd.DataFrame(rows), spectra[0]


def main():
    rng = np.random.default_rng(2026)
    weighted = weighted_experiment(rng)
    coexpression, spectrum = coexpression_experiment(rng)
    weighted.to_csv(os.path.join(RESULTS, "model_validity_weighted.csv"), index=False)
    coexpression.to_csv(os.path.join(RESULTS, "model_validity_coexpression.csv"), index=False)
    spectrum.to_csv(os.path.join(RESULTS, "model_validity_coexpression_spectrum.csv"), index=False)

    pd.set_option("display.width", 200)
    print(weighted.groupby(["noise", "metric", "n"]).mean(numeric_only=True).drop(columns="rep").round(3))
    print(coexpression.groupby(["experiment", "samples"]).mean(numeric_only=True)
          .drop(columns="rep").round(3))
    print(spectrum.round(2).T)


if __name__ == "__main__":
    main()
