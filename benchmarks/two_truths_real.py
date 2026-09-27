"""Which truth drives n2v2r's ranking on real networks (locCSN ASD vs control)?

Splits every node's change into its radial part (change of the embedding norm: hub status) and its
angular part (cosine distance: direction, i.e. community), over the default dimensions 4-24 of the
joint UASE, and compares the default ranking (Borda of euclidean and cosine) with rankings by each
part and by the degree difference (DeDi). Reported: Spearman correlations between rankings, the
overlap of top-100 lists, and each ranking's Spearman correlation with degree.

Run with ``python benchmarks/two_truths_real.py``; results go to ``results/two_truths_real.txt``.
"""

import os
import sys

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from node2vec2rank.embedding import embed  # noqa: E402
from node2vec2rank.model_utils import borda_aggregate  # noqa: E402
from real_data import load_networks  # noqa: E402
from two_truths import DEFAULT_DIMS, radial_angular  # noqa: E402

TOP = 100


def spearman(a, b):
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]


def main():
    genes, graphs = load_networks()
    degree = np.mean([g.sum(axis=0) for g in graphs], axis=0)
    rankings = {"DeDi": np.abs(graphs[1].sum(axis=0) - graphs[0].sum(axis=0))}
    for method in ("uase", "ulse"):
        embeddings = embed(graphs, max(DEFAULT_DIMS), method=method, random_state=0)
        radial, angular, euclid = radial_angular(embeddings, DEFAULT_DIMS)
        tag = "" if method == "uase" else " (ULSE)"
        rankings["default" + tag] = borda_aggregate(np.hstack([euclid, angular]))
        rankings["euclidean" + tag] = borda_aggregate(euclid)
        rankings["cosine (angular)" + tag] = borda_aggregate(angular)
        rankings["radial" + tag] = borda_aggregate(radial)
    names = list(rankings)
    top = {k: set(np.argsort(-v)[:TOP]) for k, v in rankings.items()}
    lines = [f"locCSN ASD vs CTL, {len(degree)} genes, dims {DEFAULT_DIMS[0]}-{DEFAULT_DIMS[-1]}",
             "", "Spearman with degree:"]
    lines += [f"  {k}: {spearman(v, degree):.2f}" for k, v in rankings.items()]
    lines += ["", f"Spearman between rankings / top-{TOP} overlap:"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            lines.append(f"  {a} vs {b}: {spearman(rankings[a], rankings[b]):.2f} / "
                         f"{len(top[a] & top[b])}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "results", "two_truths_real.txt"), "w") as handle:
        handle.write(text + "\n")


if __name__ == "__main__":
    main()
