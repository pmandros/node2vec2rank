"""The paper's simulation design, ported from the authors' original script
(scripts/sim_network_generation_and_comparison/sim_experiment_vs_dedi.py, sent
by Pipi on 2026-10-13), with the competitors added.

Generator (as in the original script, run with sample_uni False and noise
denominator 10):
- m communities; community proportions pi drawn at random (uniform weights,
  normalised), every community must get at least one node.
- B1 ~ U[0, 0.5] (symmetric). "Degree-naive": B2 reverses the first row and
  column of B1 (keeping B1[0, 0]). "Random": B2 re-samples the first row and
  column from U[0, 0.5]. Then every entry outside the first row and column
  gets N(0, 0.05) noise with probability 0.5, and B2 is clipped to [0, 0.5].
- A1 ~ Bernoulli(B1); A2 is A1 with edges deleted with probability 1 - B2/B1
  where B2 < B1, and added with probability 1 - (1 - B2)/(1 - B1) where B2 > B1.
- Ground truth: communities ranked by the Borda of the Euclidean, cosine,
  city-block and Chebyshev distances between the rows of B1 and B2; the pair
  is kept only if community 1 comes first.
- Score: recall of community 1 among the top |community 1| nodes.

Two details differ from the manuscript's Methods text, which says nodes are
assigned uniformly and the ground truth uses three distances: the script (and
so the published figures) uses random proportions and four distances.

Methods: n2v2r_b (dimensions 4-24, euclidean and cosine, Borda), n2v2r_e1
(dimension 1, Euclidean), DeDi (the paper's three), plus node2vec, DeepWalk
(+ Procrustes) and PLEX.I (50 trainings) from benchmarks/competitor_seeds.py.

The paper used 2,000 pairs per setting; the competitors take up to minutes per
pair, so this runs fewer pairs (default 50) for 6 communities and 50-1,000
nodes, the paper's main-text figure. Pairs already in the result file are
skipped; with --max-new 1 a process runs one pair and exits with status 3
while more remain (see competitor_replicates.py).

    while OMP_NUM_THREADS=1 python benchmarks/paper_sims.py --regime naive --nodes 1000 --max-new 1; [ $? -eq 3 ]; do :; done
    python benchmarks/paper_sims.py --summarise

The summary (paper_sims_summary.csv) gives, per setting and method, the mean
and median recall and AUROC, and against n2v2r_b the mean paired recall
difference, the pairs won and lost, and the Wilcoxon signed-rank p-value.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import chebyshev, cityblock, cosine, euclidean
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import competitor_seeds as cs  # noqa: E402
from node2vec2rank.embedding import embed  # noqa: E402
from node2vec2rank.model_utils import borda_aggregate, compute_pairwise_distances  # noqa: E402

NUM_PAIRS = 50
NOISE_SD = 0.5 / 10


def _symmetric_uniform(rng, m, low=0.0, high=0.5):
    upper = np.triu(rng.uniform(low, high, (m, m)))
    return upper + np.triu(upper, 1).T


def _borda_sort(lists):
    """The original script's borda_sort: highest total of reversed positions first."""
    scores = {}
    for ordering in lists:
        for position, element in enumerate(reversed(ordering)):
            scores[element] = scores.get(element, 0) + position
    return sorted(scores, key=lambda element: scores[element], reverse=True)


def most_changed_community(b1, b2):
    distances = [
        [euclidean(r1, r2) for r1, r2 in zip(b1, b2)],
        [cosine(r1, r2) for r1, r2 in zip(b1, b2)],
        [cityblock(r1, r2) for r1, r2 in zip(b1, b2)],
        [chebyshev(r1, r2) for r1, r2 in zip(b1, b2)],
    ]
    return _borda_sort([np.flip(np.argsort(d)) for d in distances])[0]


def sample_probabilities(rng, m, regime):
    b1 = _symmetric_uniform(rng, m)
    noise = np.zeros((m, m))
    for i in range(1, m):
        for j in range(i, m):
            value = rng.normal(0, NOISE_SD) if rng.random() < 0.5 else 0.0
            noise[i, j] = noise[j, i] = value
    b2 = b1.copy()
    if regime == "naive":
        b2[0, 1:] = b2[0, 1:][::-1]
        b2[1:, 0] = b2[1:, 0][::-1]
    else:
        b2[0, :] = rng.uniform(0, 0.5, m)
        b2[:, 0] = b2[0, :]
    return b1, np.clip(b2 + noise, 0, 0.5)


def sample_graphs(rng, n, m, regime):
    while True:
        b1, b2 = sample_probabilities(rng, m, regime)
        pi = rng.random(m)
        z = rng.choice(m, n, p=pi / pi.sum())
        if len(np.unique(z)) < m or most_changed_community(b1, b2) != 0:
            continue
        p1, p2 = b1[z][:, z], b2[z][:, z]
        iu = np.triu_indices(n, 1)
        q1, q2 = p1[iu], p2[iu]
        a1 = rng.random(len(q1)) < q1
        u = rng.random(len(q1))
        with np.errstate(divide="ignore", invalid="ignore"):
            delete = a1 & (q2 < q1) & (u >= q2 / q1)
            add = ~a1 & (q2 > q1) & (u < 1 - (1 - q2) / (1 - q1))
        a2 = (a1 & ~delete) | add
        graphs = []
        for upper in (a1, a2):
            graph = np.zeros((n, n))
            graph[iu] = upper
            graphs.append(graph + graph.T)
        return graphs, z == 0


def n2v2r_e1(graphs, seed):
    embeddings = embed(graphs, 1, "uase", random_state=seed)
    return compute_pairwise_distances(embeddings[0], embeddings[1], "euclidean")


def methods(graphs, seed):
    return {
        "n2v2r_b": lambda: cs.n2v2r(graphs, seed),
        "n2v2r_e1": lambda: n2v2r_e1(graphs, seed),
        "DeDi": lambda: cs.degree_difference(graphs),
        "node2vec": lambda: cs.walk_then_rank(graphs, seed, p=1, q=0.5),
        "DeepWalk": lambda: cs.walk_then_rank(graphs, seed, p=1, q=1),
        "PLEX.I": lambda: cs.plexi(graphs, seed)[0],
    }


def recall(labels, score):
    k = labels.sum()
    return labels[np.argsort(-score, kind="stable")[:k]].mean()


def run(regime, n, m, pairs, max_new):
    path = os.path.join(cs.RESULTS_DIR, f"paper_sims_{regime}_{n}-nodes_{m}-com.csv")
    done = 0
    for pair in range(pairs):
        table = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["pair"])
        if pair in set(table["pair"]):
            continue
        if done == max_new:
            return 1
        rng = np.random.default_rng([{"naive": 1, "random": 2}[regime], n, m, pair])
        graphs, labels = sample_graphs(rng, n, m, regime)
        rows = []
        for name, method in methods(graphs, pair).items():
            begin = time.perf_counter()
            score = np.nan_to_num(np.asarray(method(), dtype=float), nan=-np.inf)
            rows.append(dict(regime=regime, nodes=n, communities=m, pair=pair, method=name,
                             recall=recall(labels, score), auroc=cs.auroc(labels, score),
                             altered=int(labels.sum()), seconds=time.perf_counter() - begin))
        print(regime, n, m, pair, {r["method"]: round(r["recall"], 2) for r in rows}, flush=True)
        pd.concat([table, pd.DataFrame(rows)]).to_csv(path, index=False)
        done += 1
    return 0


def summarise():
    files = [f for f in sorted(glob.glob(os.path.join(cs.RESULTS_DIR, "paper_sims_*-com.csv")))]
    rows = []
    for path in files:
        table = pd.read_csv(path)
        wide = table.pivot(index="pair", columns="method", values="recall").dropna()
        setting = dict(regime=table.regime.iloc[0], nodes=int(table.nodes.iloc[0]),
                       communities=int(table.communities.iloc[0]), pairs=len(wide))
        for method, block in table.groupby("method", sort=False):
            difference = wide[method] - wide["n2v2r_b"]
            row = dict(**setting, method=method, recall_mean=block.recall.mean(), recall_median=block.recall.median(),
                       auroc_mean=block.auroc.mean(), seconds_median=block.seconds.median())
            if method != "n2v2r_b":
                row.update(minus_n2v2r_b=difference.mean(), pairs_better=int((difference > 0).sum()),
                           pairs_worse=int((difference < 0).sum()),
                           wilcoxon_p=wilcoxon(difference).pvalue if (difference != 0).any() else 1.0)
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(cs.RESULTS_DIR, "paper_sims_summary.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(summary.round(3).to_string())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regime", choices=["naive", "random"])
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--communities", type=int, default=6)
    parser.add_argument("--pairs", type=int, default=NUM_PAIRS)
    parser.add_argument("--max-new", type=int, default=-1)
    parser.add_argument("--summarise", action="store_true")
    args = parser.parse_args(argv)
    os.makedirs(cs.RESULTS_DIR, exist_ok=True)
    if args.summarise:
        return summarise()
    sys.exit(3 if run(args.regime, args.nodes, args.communities, args.pairs, args.max_new) else 0)


if __name__ == "__main__":
    main()
