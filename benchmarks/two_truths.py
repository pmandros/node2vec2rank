"""The two-truths phenomenon (Priebe et al., PNAS 2019) for differential ranking.

On the same graph, adjacency spectral embedding (ASE) tends to reveal core-periphery structure
(hubs versus the rest), while Laplacian spectral embedding (LSE) tends to reveal affinity structure
(communities). n2v2r embeds with the unfolded adjacency (UASE) by default; the regularised unfolded
Laplacian (ULSE) is an option. This script asks which kind of *change* each captures:

- an affinity change: a node moves to another community (hemisphere), keeping its core/periphery
  status;
- a core-periphery change: a node moves between core and periphery, keeping its community.

It uses a 4-block SBM (2 communities x core/periphery, as in Priebe et al.'s connectome example), with
and without degree heterogeneity, and co-expression networks where a gene either switches module
(affinity) or changes how strongly it loads on its module (hub status). For each embedding (UASE,
ULSE), metric (euclidean, cosine, both) and dimension choice (2, the true rank, the default Borda over
4-24), it reports the AUROC for the changed nodes. "UASE + ULSE" is the Borda of both embeddings'
default rankings; the absolute degree difference (DeDi) is shown in the same column for reference.

Run with ``python benchmarks/two_truths.py``; results go to ``results/two_truths.csv``.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from node2vec2rank.embedding import embed  # noqa: E402
from node2vec2rank.model_utils import borda_aggregate, compute_pairwise_distances  # noqa: E402
from node2vec2rank.simulate import coexpression_network  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")
DEFAULT_DIMS = list(range(4, 25, 2))
METRICS = {"euclidean": ("euclidean",), "cosine": ("cosine",), "both": ("euclidean", "cosine")}

# blocks: (community 0, core), (community 0, periphery), (community 1, core), (community 1, periphery)
TWO_TRUTHS_B = np.array([[0.30, 0.05, 0.15, 0.03],
                         [0.05, 0.02, 0.03, 0.01],
                         [0.15, 0.03, 0.30, 0.05],
                         [0.03, 0.01, 0.05, 0.02]])


def auroc(labels, scores):
    labels = np.asarray(labels, bool)
    ranks = rankdata(np.nan_to_num(scores, nan=-np.inf))
    positives = labels.sum()
    return (ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * (len(labels) - positives))


def scores(embeddings, dims, metrics):
    columns = [compute_pairwise_distances(embeddings[0, :, :d], embeddings[1, :, :d], metric)
               for d in dims for metric in metrics]
    return borda_aggregate(np.column_stack(columns))


def evaluate(graphs, changed, scenario, change, rank, rep, rows):
    degree_difference = np.abs(np.abs(np.asarray(graphs[1])).sum(axis=0)
                               - np.abs(np.asarray(graphs[0])).sum(axis=0))
    rows.append(dict(scenario=scenario, change=change, rep=rep, embedding="DeDi (degree difference)",
                     dims="default 4-24", metric="both", auroc=auroc(changed, degree_difference)))
    default_scores = []
    for method in ("uase", "ulse"):
        embeddings = embed(graphs, max(DEFAULT_DIMS), method=method, random_state=rep)
        for dims_name, dims in (("d=2", [2]), ("true rank", [rank]), ("default 4-24", DEFAULT_DIMS)):
            for metric_name, metrics in METRICS.items():
                s = scores(embeddings, dims, metrics)
                if dims_name == "default 4-24" and metric_name == "both":
                    default_scores.append(s)
                rows.append(dict(scenario=scenario, change=change, rep=rep, embedding=method.upper(),
                                 dims=dims_name, metric=metric_name, auroc=auroc(changed, s)))
    rows.append(dict(scenario=scenario, change=change, rep=rep, embedding="UASE + ULSE",
                     dims="default 4-24", metric="both",
                     auroc=auroc(changed, borda_aggregate(np.column_stack(default_scores)))))


def bernoulli_graph(mean, rng):
    upper = np.triu(rng.random(mean.shape) < mean, 1).astype(float)
    return upper + upper.T


def block_graphs(n, rng, change, degree_corrected, frac_changed=0.1):
    blocks = rng.integers(0, 4, n)
    community, core = blocks // 2, blocks % 2
    changed = rng.random(n) < frac_changed
    if change == "affinity":
        community = np.where(changed, 1 - community, community)
    else:
        core = np.where(changed, 1 - core, core)
    blocks_after = 2 * community + core
    theta = rng.pareto(3, n) + 1 if degree_corrected else np.ones(n)
    theta = np.minimum(theta / theta.mean(), 4.0)
    means = [np.clip(np.outer(theta, theta) * TWO_TRUTHS_B[b][:, b], 0, 1) for b in (blocks, blocks_after)]
    return [bernoulli_graph(m, rng) for m in means], changed


def expression(num_genes, samples, rng, change, num_modules=5, frac_changed=0.1):
    """Factor-model expression; changed genes switch module (affinity) or hub status (loading)."""
    modules = np.where(rng.random(num_genes) < 0.8, rng.integers(0, num_modules, num_genes), -1)
    hub = rng.random(num_genes) < 0.3
    loadings = np.where(hub, 0.85, 0.4) * (modules >= 0)
    member = np.flatnonzero(modules >= 0)
    changed = np.zeros(num_genes, bool)
    changed[rng.choice(member, int(frac_changed * num_genes), replace=False)] = True
    modules_after, loadings_after = modules.copy(), loadings.copy()
    if change == "affinity":
        modules_after[changed] = (modules[changed] + rng.integers(1, num_modules, changed.sum())) % num_modules
    else:
        loadings_after[changed] = np.where(hub[changed], 0.4, 0.85)
    data = []
    for mods, load in ((modules, loadings), (modules_after, loadings_after)):
        factors = rng.standard_normal((samples, num_modules))
        signal = np.where(mods >= 0, factors[:, np.maximum(mods, 0)] * load, 0.0)
        data.append(signal + rng.standard_normal((samples, num_genes)) * np.sqrt(1 - load ** 2))
    return data, changed


def main(reps=5):
    rng = np.random.default_rng(2019)
    rows = []
    for rep in range(reps):
        for change in ("affinity", "core-periphery"):
            for degree_corrected in (False, True):
                graphs, changed = block_graphs(2000, rng, change, degree_corrected)
                scenario = "4-block SBM" + (", degree-corrected" if degree_corrected else "")
                evaluate(graphs, changed, scenario, change, 4, rep, rows)
            for power, signed in ((10, True), (6, False)):
                data, changed = expression(1000, 150, rng, change)
                graphs = [coexpression_network(x, power=power, signed=signed) for x in data]
                name = "co-expression ((1+cor)/2)^10" if signed else "co-expression |cor|^6"
                evaluate(graphs, changed, name, "module switch" if change == "affinity" else "hub status",
                         5, rep, rows)

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(RESULTS, "two_truths.csv"), index=False)
    table = results.pivot_table(index=["scenario", "change", "embedding"], columns=["dims", "metric"],
                                values="auroc", aggfunc="mean").round(2)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(table.to_string())


if __name__ == "__main__":
    main()
