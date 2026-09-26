"""Two groups of unequal size: why the test loses power, and which fix works.

Mechanism checked here: a soft-thresholded co-expression network has a noise
floor, the mean edge weight between unrelated genes, that grows as the number of
samples falls (for |cor|^6, E|r|^6 is about 15 / n^3). The smaller group's network
therefore has a larger constant added to every edge. In UASE that is a
rank-one, all-ones component with a different weight in each network, so every
gene moves along the same axis. The move is largest relative to the position of
weakly connected genes, which is what the degree-adjusted test then sees.

Two-group simulations from :func:`node2vec2rank.simulate.simulate_expression`
(600 genes, 10% rewired, 10% differentially expressed decoys), at group sizes
from balanced to HeLa-like (G1 about 840 cells, S about 260, G2 and M about 190),
with the unsigned |cor|^6 network and the signed ((1 + cor) / 2)^10 network used
for the HeLa phases. Fixes compared:

- as is: the degree-adjusted test on all dimensions and metrics,
- cosine only: the same test on the cosine distances,
- subsampled: both groups subsampled to the smaller size, 10 times, distances averaged,
- debiased: the expected edge weight between unrelated genes, for that group's
  size, subtracted from every edge (clipped at 0),
- layer-scaled: every network divided by its mean edge weight,
- permutation: the sample-label permutation test (100 permutations), whose null
  keeps the two group sizes.

    python benchmarks/temporal_imbalance.py  # writes benchmarks/results/temporal_imbalance.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporal import DIMS, METRICS, RESULTS_DIR, auroc, uase_both  # noqa: E402
from node2vec2rank.model_utils import compute_pairwise_distances  # noqa: E402
from node2vec2rank.permutation import permutation_test  # noqa: E402
from node2vec2rank.significance import empirical_null_test  # noqa: E402
from node2vec2rank.simulate import simulate_expression  # noqa: E402

D = max(DIMS)
SIZES = [(190, 190), (30, 30), (150, 30), (840, 260), (840, 190)]
NETWORKS = {
    "unsigned |cor|^6": lambda r: np.abs(r) ** 6,
    "signed ((1+cor)/2)^10": lambda r: ((1 + r) / 2) ** 10,
}


def correlation(values):
    values = np.asarray(values, dtype=np.float64)
    values = (values - values.mean(axis=0)) / values.std(axis=0)
    return values.T @ values / len(values)


def network(values, transform, debias=False, _floor={}):
    adjacency = transform(correlation(values))
    if debias:
        key = (id(transform), len(values))
        if key not in _floor:  # expected edge between unrelated genes, for this sample size
            rng = np.random.default_rng(0)
            null = correlation(rng.standard_normal((len(values), 400)))
            _floor[key] = transform(null[np.triu_indices(400, 1)]).mean()
        adjacency = np.maximum(adjacency - _floor[key], 0)
    np.fill_diagonal(adjacency, 0)
    return adjacency


def distance_columns(graphs, metrics=METRICS):
    _, right = uase_both(graphs, D)
    return np.column_stack([compute_pairwise_distances(right[0, :, :dim], right[1, :, :dim], metric)
                            for dim in DIMS for metric in metrics])


def evaluate(columns, degree, rewired):
    z, _, q = empirical_null_test(columns, degree)
    return dict(auroc=auroc(columns.mean(axis=1) if columns.shape[1] > 1 else columns[:, 0], rewired),
                auroc_z=auroc(np.nan_to_num(z, nan=-np.inf), rewired),
                true_calls=int(np.sum((q < 0.1) & rewired)), false_calls=int(np.sum((q < 0.1) & ~rewired)))


def run(seed, sizes, network_name, repeats=10, permutations=0):
    transform = NETWORKS[network_name]
    rng = np.random.default_rng(seed)
    sim = simulate_expression(num_genes=600, num_samples=sizes, num_modules=5, random_state=rng)
    expression = [np.asarray(x) for x in sim.expression]
    # standardise within group, as the HeLa pipeline does, so DE decoys are pure decoys
    expression = [(x - x.mean(axis=0)) / x.std(axis=0) for x in expression]
    rewired = sim.rewired.to_numpy()
    graphs = [network(x, transform) for x in expression]
    degree = np.mean([g.sum(axis=0) for g in graphs], axis=0)
    base = dict(seed=seed, sizes=f"{sizes[0]} vs {sizes[1]}", network=network_name,
                degree_gap=float(np.mean(graphs[1].sum(0) - graphs[0].sum(0))))
    rows = []

    columns = distance_columns(graphs)
    rows.append(dict(base, method="as is", **evaluate(columns, degree, rewired)))
    cosine = columns[:, [i for i, (d, m) in enumerate((d, m) for d in DIMS for m in METRICS) if m == "cosine"]]
    rows.append(dict(base, method="cosine only", **evaluate(cosine, degree, rewired)))

    size = min(sizes)
    if sizes[0] != sizes[1]:
        total = 0
        for _ in range(repeats):
            sub = [network(x[rng.choice(len(x), size, replace=False)], transform) for x in expression]
            total = total + distance_columns(sub)
        rows.append(dict(base, method="subsampled", **evaluate(total / repeats, degree, rewired)))

    debiased = [network(x, transform, debias=True) for x in expression]
    rows.append(dict(base, method="debiased", **evaluate(
        distance_columns(debiased), np.mean([g.sum(0) for g in debiased], axis=0), rewired)))

    scaled = [g / g.mean() for g in graphs]
    rows.append(dict(base, method="layer-scaled", **evaluate(
        distance_columns(scaled), np.mean([g.sum(0) for g in scaled], axis=0), rewired)))

    if permutations:
        frames = [pd.DataFrame(x, columns=sim.expression[0].columns) for x in expression]
        result = permutation_test(frames[0], frames[1], build_network=lambda x: network(x, transform),
                                  num_permutations=permutations, random_state=seed,
                                  embed_dimensions=DIMS, distance_metrics=list(METRICS), seed=seed)
        q = result["qvalue"].to_numpy()
        rows.append(dict(base, method="permutation", auroc=auroc(result["borda_ranks"].to_numpy(), rewired),
                         auroc_z=auroc(result["z"].to_numpy(), rewired),
                         true_calls=int(np.sum((q < 0.1) & rewired)),
                         false_calls=int(np.sum((q < 0.1) & ~rewired))))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--permutation-replicates", type=int, default=3,
                        help="replicates that also run the (slow) permutation test")
    args = parser.parse_args(argv)
    rows = []
    for network_name in NETWORKS:
        for sizes in SIZES:
            for seed in range(args.replicates):
                perms = args.permutations if seed < args.permutation_replicates else 0
                rows.extend(run(seed, sizes, network_name, permutations=perms))
                print(network_name, sizes, seed, flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(RESULTS_DIR, "temporal_imbalance.csv"), index=False)
    pd.set_option("display.width", 250)
    print(results.drop(columns="seed").groupby(["network", "sizes", "method"], sort=False).mean().round(3)
          .to_string())


if __name__ == "__main__":
    main()
