"""The two-truths phenomenon on Zachary's karate club, statically and for differential ranking.

Static: embed the graph with ASE and LSE (d = 2) and ask which truth each shows: the two factions
(community) or the hubs (degree). Measured by how well a 2-means clustering of the embedding (or of
its row-normalised version, which is what cosine distance sees) matches the factions, and by how well
the embedding's row norm tracks degree.

Differential: for every non-hub member, make a second graph where the member either
- partly switches faction: 2 of its edges are moved to random members of the other faction (degree
  kept), or
- becomes more of a hub: it gains 2 new edges to random members of its own faction,
and record where the member lands in each ranking (1 = top of 34), averaged over members and 20
random rewirings each. Rankings: euclidean, cosine, the radial part (change of the row norm), the
default Borda of euclidean and cosine, and the degree difference (DeDi). Dimensions 2 and 4.

Run with ``python benchmarks/two_truths_karate.py``; results go to ``results/two_truths_karate.txt``.
"""

import os
import sys

import networkx as nx
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from node2vec2rank.embedding import embed  # noqa: E402
from node2vec2rank.model_utils import borda_aggregate, compute_pairwise_distances  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")
EDGES_CHANGED = 2


def two_means_agreement(points, labels, restarts=20, seed=0):
    """Best 2-means (Lloyd) clustering over random restarts; agreement with labels up to label swap."""
    rng = np.random.default_rng(seed)
    best, best_loss = None, np.inf
    for _ in range(restarts):
        centres = points[rng.choice(len(points), 2, replace=False)]
        for _ in range(100):
            assign = np.argmin(((points[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
            new = np.array([points[assign == k].mean(0) if np.any(assign == k) else centres[k] for k in range(2)])
            if np.allclose(new, centres):
                break
            centres = new
        loss = ((points - centres[assign]) ** 2).sum()
        if loss < best_loss:
            best, best_loss = assign, loss
    agreement = np.mean(best == labels)
    return max(agreement, 1 - agreement)


def rankings(graphs, d, method):
    embeddings = embed(graphs, d, method=method, random_state=0)
    one, two = embeddings[0], embeddings[1]
    euclid = compute_pairwise_distances(one, two, "euclidean")
    cosine = np.nan_to_num(compute_pairwise_distances(one, two, "cosine"))
    radial = np.abs(np.linalg.norm(one, axis=1) - np.linalg.norm(two, axis=1))
    return {"euclidean": euclid, "cosine": cosine, "radial": radial,
            "default Borda (euclidean + cosine)": borda_aggregate(np.column_stack([euclid, cosine]))}


def main():
    graph = nx.karate_club_graph()
    adjacency = nx.to_numpy_array(graph, weight=None)
    n = len(adjacency)
    faction = np.array([graph.nodes[i]["club"] == "Officer" for i in range(n)])
    degree = adjacency.sum(axis=0)
    lines = ["Static embedding (d = 2): agreement of 2-means with the factions; Spearman(row norm, degree)"]
    for method in ("uase", "ulse"):
        x = embed([adjacency], 2, method=method, random_state=0)[0]
        norm = np.linalg.norm(x, axis=1)
        unit = x / norm[:, None]
        rho = np.corrcoef(rankdata(norm), rankdata(degree))[0, 1]
        lines.append(f"  {method.upper()}: raw {two_means_agreement(x, faction):.2f}, "
                     f"row-normalised {two_means_agreement(unit, faction):.2f}, norm vs degree {rho:.2f}")
    lines.append("Per dimension: |Spearman| with degree / point-biserial |correlation| with faction")
    for method in ("uase", "ulse"):
        x = embed([adjacency], 4, method=method, random_state=0)[0]
        cells = []
        for j in range(4):
            with_degree = abs(np.corrcoef(rankdata(x[:, j]), rankdata(degree))[0, 1])
            with_faction = abs(np.corrcoef(x[:, j], faction)[0, 1])
            cells.append(f"dim {j + 1}: {with_degree:.2f} / {with_faction:.2f}")
        lines.append(f"  {method.upper()}: " + "; ".join(cells))

    rng = np.random.default_rng(34)
    members = [i for i in range(n) if degree[i] <= 6]
    results = {}
    for change in ("switch faction", "become a hub"):
        for method in ("uase", "ulse"):
            for d in (2, 4):
                positions = {}
                for i in members:
                    for _ in range(20):
                        after = adjacency.copy()
                        k = EDGES_CHANGED
                        if change == "switch faction":
                            # move k of the member's edges to the other faction (degree kept)
                            dropped = rng.choice(np.flatnonzero(adjacency[i]), min(k, int(degree[i])),
                                                 replace=False)
                            after[i, dropped] = after[dropped, i] = 0
                            k = len(dropped)
                            pool = np.flatnonzero((faction != faction[i]) & (adjacency[i] == 0)
                                                  & (np.arange(n) != i))
                        else:
                            pool = np.flatnonzero((faction == faction[i]) & (adjacency[i] == 0)
                                                  & (np.arange(n) != i))
                            k = min(k, len(pool))
                        targets = rng.choice(pool, k, replace=False)
                        after[i, targets] = after[targets, i] = 1
                        scores = rankings([adjacency, after], d, method)
                        scores["DeDi"] = np.abs(after.sum(0) - degree)
                        for name, s in scores.items():
                            positions.setdefault(name, []).append(rankdata(-s)[i])
                results[(change, method, d)] = {k: np.mean(v) for k, v in positions.items()}
    lines.append("")
    lines.append(f"Differential: mean rank of the changed member among {n} (1 = top), "
                 f"{len(members)} members x 20 rewirings")
    names = list(next(iter(results.values())))
    lines.append("  " + " | ".join(["change, embedding, d"] + names))
    for (change, method, d), row in results.items():
        lines.append("  " + " | ".join([f"{change}, {method.upper()}, {d}"] + [f"{row[k]:.1f}" for k in names]))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(RESULTS, "two_truths_karate.txt"), "w") as handle:
        handle.write(text + "\n")


if __name__ == "__main__":
    main()
