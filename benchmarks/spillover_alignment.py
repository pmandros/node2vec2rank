"""Spill-over: the UASE target versus the nodes whose own latent position changed.

The consistency theorem is about the distance between a node's population UASE positions. Under a
latent position model P^(k) = Phi^(k) Lambda_k Phi^(k)^T, those positions are
Y^(k) = Phi^(k) M_k for a d x d matrix M_k that depends on every node of layer k. So when some nodes
change (or the layer's kernel Lambda_k changes), M_1 != M_2 and *every* node moves, even nodes whose
own latent position is unchanged ("spill-over"). With Euclidean distance the spill-over of node i is
||Phi_i (M_1 - M_2)||, which grows with degree.

Spill-over is a single linear map per layer, so it can be removed: fit G with Y^(1) G ~= Y^(2) on the
nodes that did not change and measure distances after aligning. The unchanged nodes are unknown, so
G is fitted robustly (least trimmed squares), which works when most nodes are unchanged, the same
assumption as between-sample normalisation in differential expression. This script compares the
default distances with the aligned ones in:

1. mixed-membership latent positions with Poisson weights, 5-40% of nodes changing;
2. the same with a global change of the kernel (every module's strength changes);
3. a degree-corrected SBM where nodes switch community;
4. co-expression networks (hdWGCNA's signed ((1+cor)/2)^10 and WGCNA's |cor|^6);
5. co-expression with no change (false calls), and with one whole module weakening, a change of
   the kernel that alignment absorbs by design;
6. signed correlation networks where 10% of genes turn anti-correlated with their module.

Truth is "the node's own latent position changed". Reported: AUROC of the Borda over the default
dimensions (4-24) and metrics, and calls at q < 0.1 from the degree-adjusted empirical null test.
Run with ``python benchmarks/spillover_alignment.py``; results go to ``results/spillover_alignment.csv``.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from node2vec2rank.embedding import uase  # noqa: E402
from node2vec2rank.model_utils import borda_aggregate, compute_pairwise_distances  # noqa: E402
from node2vec2rank.significance import empirical_null_test  # noqa: E402
from node2vec2rank.simulate import coexpression_network, simulate_expression  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")
DIMS = list(range(4, 25, 2))
METRICS = ("euclidean", "cosine")


def auroc(labels, scores):
    labels = np.asarray(labels, bool)
    ranks = rankdata(np.nan_to_num(scores, nan=-np.inf))
    positives = labels.sum()
    return (ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * (len(labels) - positives))


def robust_linear_alignment(source, target, keep=0.5, max_iterations=50):
    """Least trimmed squares fit of a d x d matrix G with source @ G ~= target.

    Starts from ordinary least squares and from the identity, and alternates between fitting G on
    the ``keep`` fraction of rows with the smallest residuals and recomputing the residuals
    (concentration steps), keeping the start with the smaller trimmed loss.
    """
    n, d = source.shape
    h = max(d + 1, int(keep * n))
    best, best_loss = None, np.inf
    for start in (np.linalg.lstsq(source, target, rcond=None)[0], np.eye(d)):
        g = start
        chosen = None
        for _ in range(max_iterations):
            residuals = np.linalg.norm(source @ g - target, axis=1)
            new = np.sort(np.argpartition(residuals, h - 1)[:h])
            if chosen is not None and np.array_equal(new, chosen):
                break
            chosen = new
            g = np.linalg.lstsq(source[chosen], target[chosen], rcond=None)[0]
        loss = np.sort(np.linalg.norm(source @ g - target, axis=1) ** 2)[:h].sum()
        if loss < best_loss:
            best, best_loss = g, loss
    return best


def distance_table(embeddings, align):
    columns = {}
    for d in DIMS:
        one, two = embeddings[0, :, :d], embeddings[1, :, :d]
        if align:
            one = one @ robust_linear_alignment(one, two)
        for metric in METRICS:
            columns[(d, metric)] = compute_pairwise_distances(one, two, metric)
    return pd.DataFrame(columns)


def evaluate(graphs, changed, scenario, setting, rep, rows):
    embeddings = uase(graphs, max(DIMS), random_state=rep)
    degree = np.mean([np.abs(g).sum(axis=0) for g in graphs], axis=0)
    for align in (False, True):
        table = distance_table(embeddings, align)
        variants = {"Borda (both)": table, "euclidean": table.xs("euclidean", axis=1, level=1),
                    "cosine": table.xs("cosine", axis=1, level=1)}
        for metric, frame in variants.items():
            values = frame.to_numpy()
            borda = borda_aggregate(values)
            _, _, q = empirical_null_test(values, degree)
            calls = q < 0.1
            top = rankdata(-borda, method="ordinal") <= changed.sum()
            rows.append(dict(scenario=scenario, setting=setting, rep=rep,
                             distances="aligned" if align else "default", metric=metric,
                             auroc=auroc(changed, borda) if changed.any() else np.nan,
                             top_k_precision=changed[top].mean() if changed.any() else np.nan,
                             true_calls=int(np.sum(calls & changed)),
                             false_calls=int(np.sum(calls & ~changed)),
                             unchanged_vs_degree=pd.Series(borda[~changed]).corr(
                                 pd.Series(degree[~changed]), method="spearman")))


def poisson_graph(mean, rng, scale=2.0):
    upper = np.triu(rng.poisson(scale * mean) / scale, 1)
    return upper + upper.T


def mixed_membership(n, rank, rng, frac_changed, kernel_change):
    centres = np.eye(rank) * 0.6 + 0.1
    degree = rng.uniform(0.3, 1.0, n)
    before = degree[:, None] * rng.dirichlet(np.full(rank, 0.3), n) @ centres
    changed = rng.random(n) < frac_changed
    after = before.copy()
    after[changed] = degree[changed, None] * rng.dirichlet(np.full(rank, 0.3), changed.sum()) @ centres
    kernel = np.diag(rng.uniform(0.6, 1.4, rank)) if kernel_change else np.eye(rank)
    means = [before @ before.T, after @ kernel @ after.T]
    for mean in means:
        np.fill_diagonal(mean, 0)
    return means, changed


def dcsbm(n, k, rng, frac_changed):
    b = np.full((k, k), 0.05) + np.eye(k) * 0.25
    z = rng.integers(0, k, n)
    changed = rng.random(n) < frac_changed
    z_after = z.copy()
    z_after[changed] = (z[changed] + rng.integers(1, k, changed.sum())) % k
    theta = rng.pareto(2.5, n) + 1
    theta /= theta.mean()
    means = [np.outer(theta, theta) * b[a][:, a] for a in (z, z_after)]
    for mean in means:
        np.fill_diagonal(mean, 0)
    return means, changed


def module_expression(num_genes, samples, rng, num_modules=5, weaken=None, num_flipped=0):
    """Factor-model expression.

    ``weaken`` scales the loadings of module 0 in the second condition; ``num_flipped`` module genes
    turn anti-correlated with their module in the second condition. Returns the expression of both
    conditions and the genes of module 0 (or the flipped genes, if any).
    """
    modules = np.where(rng.random(num_genes) < 0.7, rng.integers(0, num_modules, num_genes), -1)
    loadings = rng.uniform(0.3, 0.9, num_genes) * (modules >= 0)
    flipped = np.zeros(num_genes, bool)
    flipped[rng.choice(np.flatnonzero(modules >= 0), num_flipped, replace=False)] = True
    expression = []
    for condition in range(2):
        scaled = loadings * (weaken if condition == 1 and weaken is not None else 1.0) ** (modules == 0)
        if condition == 1:
            scaled = np.where(flipped, -scaled, scaled)
        factors = rng.standard_normal((samples, num_modules))
        signal = np.where(modules >= 0, factors[:, np.maximum(modules, 0)] * scaled, 0.0)
        expression.append(signal + rng.standard_normal((samples, num_genes)) * np.sqrt(1 - scaled ** 2))
    return expression, flipped if num_flipped else modules == 0


def main(reps=5):
    rng = np.random.default_rng(2027)
    rows = []
    for rep in range(reps):
        for frac in (0.05, 0.2, 0.4):
            for kernel_change in (False, True):
                means, changed = mixed_membership(1500, 4, rng, frac, kernel_change)
                scenario = "mixed membership + kernel change" if kernel_change else "mixed membership"
                evaluate([poisson_graph(m, rng) for m in means], changed, scenario, f"{frac:.0%} changed",
                         rep, rows)
                evaluate(means, changed, scenario + " (noise-free)", f"{frac:.0%} changed", rep, rows)
        for frac in (0.1, 0.3):
            means, changed = dcsbm(1500, 6, rng, frac)
            evaluate([poisson_graph(m, rng, 5.0) for m in means], changed, "DC-SBM switch",
                     f"{frac:.0%} changed", rep, rows)
        for power, signed in ((10, True), (6, False)):
            for samples in (150, 500):
                sim = simulate_expression(num_genes=1000, num_samples=samples, frac_rewired=0.1,
                                          frac_differentially_expressed=0.0, random_state=rng)
                graphs = [coexpression_network(e, power=power, signed=signed) for e in sim.expression]
                name = "co-expression ((1+cor)/2)^10" if signed else "co-expression |cor|^6"
                evaluate(graphs, sim.rewired.to_numpy(), name, f"{samples} samples", rep, rows)
        for power, signed in ((10, True), (6, False)):
            name = "((1+cor)/2)^10" if signed else "|cor|^6"
            # nothing changes: false calls only
            expression, _ = module_expression(1000, 150, rng)
            graphs = [coexpression_network(e, power=power, signed=signed) for e in expression]
            evaluate(graphs, np.zeros(1000, bool), f"co-expression {name}, no change", "150 samples",
                     rep, rows)
            # a whole module weakens: a change of the kernel, which alignment absorbs by design
            expression, module = module_expression(1000, 150, rng, weaken=0.6)
            graphs = [coexpression_network(e, power=power, signed=signed) for e in expression]
            evaluate(graphs, module, f"co-expression {name}, one module weakens", "150 samples",
                     rep, rows)
        # genes turning anti-correlated, in the signed correlation network (zero diagonal)
        expression, flipped = module_expression(1000, 500, rng, num_flipped=100)
        graphs = [np.corrcoef(e, rowvar=False) - np.eye(1000) for e in expression]
        evaluate(graphs, flipped, "signed correlation, sign flips", "500 samples", rep, rows)

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(RESULTS, "spillover_alignment.csv"), index=False)
    summary = (results.groupby(["scenario", "setting", "metric", "distances"])
               [["auroc", "top_k_precision", "true_calls", "false_calls", "unchanged_vs_degree"]]
               .mean().round(3))
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    print(summary)


if __name__ == "__main__":
    main()
