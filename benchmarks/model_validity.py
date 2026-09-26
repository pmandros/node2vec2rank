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
   Checked for the paper's constructions: WGCNA |cor|^6 and |cor|^10, and hdWGCNA's TOM;
3. what the population ranking measures: noise-free, unchanged nodes still move when other nodes
   change (Euclidean spill-over grows with degree), and in the paper's SBM design the population
   ranking does not always put the altered community first, never under degree correction for
   Euclidean distances;
4. the error norms: the largest row error shrinks with n while the spectral and Frobenius errors
   do not, so a per-node guarantee needs a row-wise (two-to-infinity) argument.

Run with ``python benchmarks/model_validity.py`` (about two minutes on 4 cores); the numbers are
written to ``results/model_validity_*.csv``.
"""

import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import chebyshev, cosine, euclidean
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


def build_network(correlation, kind):
    """The paper's network constructions: WGCNA |cor|^power, or hdWGCNA's TOM of |cor|^6."""
    power = {"power 6": 6, "power 10": 10, "TOM (power 6)": 6}[kind]
    adjacency = np.abs(correlation) ** power
    np.fill_diagonal(adjacency, 0)
    if kind.startswith("TOM"):
        degree = adjacency.sum(axis=1)
        shared = adjacency @ adjacency
        adjacency = (shared + adjacency) / (np.minimum.outer(degree, degree) + 1 - adjacency)
        np.fill_diagonal(adjacency, 0)
    return adjacency


def sample_network(load, m, rng, kind="power 6"):
    n, k = load.shape
    values = rng.standard_normal((m, k)) @ load.T \
        + rng.standard_normal((m, n)) * np.sqrt(1 - (load ** 2).sum(axis=1))
    return build_network(np.corrcoef(values, rowvar=False), kind)


def population_network(load, kind="power 6"):
    correlation = load @ load.T
    np.fill_diagonal(correlation, 1)
    return build_network(correlation, kind)


def coexpression_experiment(rng, reps=5, n=1000, num_modules=5,
                            kinds=("power 6", "power 10", "TOM (power 6)")):
    rows, spectra = [], []
    for rep in range(reps):
        load_before, load_after, rewired, strength = module_correlation(n, num_modules, rng)
        for kind in kinds:
            target = [population_network(load_before, kind), population_network(load_after, kind)]
            pop = population_distances(target, num_modules)
            if rep == 0:
                singular_values = np.linalg.svd(np.hstack(target), compute_uv=False)[:30]
                spectra.append(pd.DataFrame({"network": kind, "index": np.arange(1, 31),
                                             "singular_value": singular_values}))

            # (a) rewiring, equal sample sizes: how close is the ranking to the intended target?
            for m in (50, 150, 500, 2000):
                graphs = [sample_network(load_before, m, rng, kind),
                          sample_network(load_after, m, rng, kind)]
                est = distances(graphs, sorted(set(DIMS) | {num_modules}), seed=rep)
                borda = borda_aggregate(est[DIMS].to_numpy())
                p = pop[(num_modules, "euclidean")].to_numpy()
                e = est[(num_modules, "euclidean")].to_numpy()
                rows.append(dict(network=kind, experiment="rewiring", samples=f"{m} vs {m}", rep=rep,
                                 spearman_vs_target_rewired=spearmanr(p[rewired], borda[rewired])[0],
                                 spearman_vs_target_rewired_rank_d=spearmanr(p[rewired], e[rewired])[0],
                                 auroc_rewired=roc_auc_score(rewired, borda),
                                 null_order_vs_strength=spearmanr(borda[~rewired], strength[~rewired])[0]))

            # (b) no rewiring at all: identical correlations, equal or unequal sample sizes
            for m1, m2 in ((150, 150), (150, 600), (50, 50), (50, 500)):
                bordas = []
                for _ in range(2):
                    graphs = [sample_network(load_before, m1, rng, kind),
                              sample_network(load_before, m2, rng, kind)]
                    bordas.append(borda_aggregate(distances(graphs, DIMS, seed=rep).to_numpy()))
                top = lambda b: rankdata(-b) <= 0.05 * n  # noqa: E731
                rows.append(dict(network=kind, experiment="no rewiring", samples=f"{m1} vs {m2}", rep=rep,
                                 # a reproducible ranking under no change means a systematic artefact
                                 replicate_agreement=spearmanr(bordas[0], bordas[1])[0],
                                 null_order_vs_strength=spearmanr(bordas[0], strength)[0],
                                 top5pct_overlap=np.mean(top(bordas[0])[top(bordas[1])])))
    return pd.DataFrame(rows), pd.concat(spectra)


# --- 3. what the population ranking itself measures ---------------------------------------------

def population_target_experiment(rng, n=2000):
    """Noise-free distances: nodes whose own position is unchanged still move when others change.

    A node's UASE position is its column of P projected on the shared left embedding, so when
    other nodes change, the unchanged node's column changes too. For Euclidean distances this
    spill-over scales with the node's degree.
    """
    rows = []
    for frac in (0.05, 0.2, 0.5):
        before, after, changed, amount, degree = latent_positions(n, 4, rng, frac)
        means = [before @ before.T, after @ after.T]
        for mean in means:
            np.fill_diagonal(mean, 0)
        pop = population_distances(means, 4)
        for metric in ("euclidean", "cosine"):
            d = pop[(4, metric)].to_numpy()
            rows.append(dict(design="mixed membership", changed=frac, metric=metric,
                             median_unchanged=np.median(d[~changed]), max_unchanged=d[~changed].max(),
                             median_changed=np.median(d[changed]),
                             changed_below_max_unchanged=np.mean(d[changed] < d[~changed].max()),
                             unchanged_vs_degree=spearmanr(d[~changed], degree[~changed])[0],
                             auroc=roc_auc_score(changed, d)))

    return pd.DataFrame(rows)


def paper_design_experiment(rng, n=600, accepted=100):
    """The paper's SBM design with infinite data: is the altered community on top of the target?

    One community's row and column of B are resampled, and pairs are kept only when the
    community-wise Borda of Euclidean, cosine and Chebyshev distances between the rows of B ranks
    the altered community first. The UASE population distances are then computed from the mean
    matrices, so any miss is a property of the target, not of estimation error.
    """
    rows = []
    for k in (4, 10):
        kept = 0
        while kept < accepted:
            b = rng.uniform(0.05, 0.5, (k, k))
            b = (b + b.T) / 2
            b_after = b.copy()
            b_after[0, :] = b_after[:, 0] = rng.uniform(0.05, 0.5, k)
            row_distances = np.array([[euclidean(b[c], b_after[c]), cosine(b[c], b_after[c]),
                                       chebyshev(b[c], b_after[c])] for c in range(k)])
            if np.argmax(borda_aggregate(row_distances)) != 0:
                continue
            kept += 1
            z = np.repeat(np.arange(k), n // k)
            changed = z == 0
            for degree_corrected in (False, True):
                theta = rng.pareto(2.5, len(z)) + 1 if degree_corrected else np.ones(len(z))
                theta /= theta.mean()
                pop = population_distances([np.outer(theta, theta) * m[z][:, z] for m in (b, b_after)], k)
                scores = {"euclidean": pop[(k, "euclidean")], "cosine": pop[(k, "cosine")],
                          "borda (both)": borda_aggregate(pop.to_numpy())}
                for metric, d in scores.items():
                    rows.append(dict(communities=k, design="DC-SBM" if degree_corrected else "SBM",
                                     metric=metric, population_auroc=roc_auc_score(changed, d)))
    return pd.DataFrame(rows)


def error_norm_experiment(rng):
    """Row-wise error shrinks with n while the Frobenius and spectral errors do not.

    So a per-node guarantee cannot be obtained by bounding the largest row error by a spectral or
    Frobenius norm; it needs a row-wise (two-to-infinity) argument.
    """
    rows = []
    for n in (500, 1000, 2000, 4000):
        before, after, _, _, _ = latent_positions(n, 4, rng)
        means = [before @ before.T, after @ after.T]
        for mean in means:
            np.fill_diagonal(mean, 0)
        graphs = [symmetric_noise(m, "gaussian", rng, 0.3) for m in means]
        population = uase(means, 4, random_state=0).reshape(-1, 4)
        estimate = uase(graphs, 4, random_state=0).reshape(-1, 4)
        u, _, vt = np.linalg.svd(estimate.T @ population)
        error = estimate @ (u @ vt) - population
        rows.append(dict(n=n, frobenius=np.linalg.norm(error), spectral=np.linalg.norm(error, 2),
                         max_row=np.linalg.norm(error, axis=1).max()))
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(2026)
    weighted = weighted_experiment(rng)
    coexpression, spectrum = coexpression_experiment(rng)
    target = population_target_experiment(rng)
    norms = error_norm_experiment(rng)
    design = paper_design_experiment(rng)
    design.to_csv(os.path.join(RESULTS, "model_validity_paper_design.csv"), index=False)
    target.to_csv(os.path.join(RESULTS, "model_validity_population_target.csv"), index=False)
    norms.to_csv(os.path.join(RESULTS, "model_validity_error_norms.csv"), index=False)
    weighted.to_csv(os.path.join(RESULTS, "model_validity_weighted.csv"), index=False)
    coexpression.to_csv(os.path.join(RESULTS, "model_validity_coexpression.csv"), index=False)
    spectrum.to_csv(os.path.join(RESULTS, "model_validity_coexpression_spectrum.csv"), index=False)

    pd.set_option("display.width", 200)
    print(weighted.groupby(["noise", "metric", "n"]).mean(numeric_only=True).drop(columns="rep").round(3))
    print(coexpression.groupby(["network", "experiment", "samples"]).mean(numeric_only=True)
          .drop(columns="rep").round(3))
    print(spectrum.pivot(index="index", columns="network", values="singular_value").head(12).round(3))
    print(target.groupby(["design", "metric", "changed"]).mean(numeric_only=True).round(3))
    print(norms.round(3))
    print(design.groupby(["communities", "design", "metric"]).population_auroc
          .agg(mean="mean", perfect=lambda a: np.mean(a > 1 - 1e-9)).round(3))


if __name__ == "__main__":
    main()
