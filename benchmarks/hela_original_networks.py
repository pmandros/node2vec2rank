"""The paper's own HeLa networks against the networks rebuilt in Python.

The paper's four hdWGCNA TOM networks (one per phase, from the authors; each
phase has its own genes, and the 2,420 genes common to all four are the
paper's networks) are ranked with the default n2v2r (joint UASE of the four
networks, sequential comparisons, dimensions 4..24, Euclidean and cosine,
Borda), as in the paper. The networks rebuilt by cell_cycle.prepare (setting
"paper", soft power 6 or 5) are ranked the same way. For G1 -> S and G2 -> M
we report, over the genes in both gene sets, the Spearman correlation of the
two rankings and the overlap of their top 100 genes, and we write the
rankings of the paper's networks. On the paper's networks we also count the
Revelio phase-marker genes (Whitfield et al. 2002) among the top 20 genes of
n2v2r and of DeDi. With --dimensions, the dimension check of
revision_hela.py runs on the paper's networks instead (joint UASE of the four
networks; G1 -> S and G2 -> M).

    OMP_NUM_THREADS=1 python benchmarks/hela_original_networks.py \
        --networks-dir /mnt/project-files/revision/hdWGCNA_metacells --revelio-dir path/to/Revelio/data
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import cell_cycle as cc  # noqa: E402
from node2vec2rank.model import N2V2R  # noqa: E402

PHASES = ("G1", "S", "G2", "M")
TRANSITIONS = {"1": "G1 -> S", "3": "G2 -> M"}  # keys of the sequential comparisons
TOP_K = 100
TOP_MARKERS = 20


def rank(networks, genes, dedi=False):
    model = N2V2R([networks[p].loc[genes, genes].to_numpy() for p in PHASES], nodes=genes, seed=0, verbose=0)
    model.fit_transform_rank()
    borda = model.aggregate_transform()
    ranks = {label: borda[key]["borda_ranks"] for key, label in TRANSITIONS.items()}
    if dedi:
        degree = model.degree_difference_ranking()
        return ranks, {label: degree[key]["absDeDi"] for key, label in TRANSITIONS.items()}
    return ranks


def dimensions(original, genes):
    """Singular values of the joint embedding of the four networks, its elbow, and for G1 -> S and G2 -> M
    each single dimension's ranking against the default (Borda over 4..24, Euclidean and cosine)."""
    import revision_hela as rh
    from node2vec2rank.embedding import embed, select_dimension
    graphs = [original[p].loc[genes, genes].to_numpy() for p in PHASES]
    embeddings, singular_values = embed(graphs, rh.SPECTRUM_SIZE, "uase", random_state=42, return_singular_values=True)
    elbow = select_dimension(singular_values, min_dimension=2)
    spectrum = pd.DataFrame({"comparison": "G1, S, G2 and M", "dimension": np.arange(1, rh.SPECTRUM_SIZE + 1),
                             "singular_value": np.sort(singular_values)[::-1], "elbow": elbow})
    rankings = []
    for label, (a, b) in {"G1 -> S": (0, 1), "G2 -> M": (2, 3)}.items():
        pair = [embeddings[a], embeddings[b]]
        default = rh.borda(pair, cc.DIMENSIONS)
        for name, dims in [("default (Borda 4-24)", cc.DIMENSIONS), (f"elbow ({elbow})", [elbow])] + \
                [(f"{d} only", [d]) for d in range(2, 25, 2)]:
            score = rh.borda(pair, dims)
            rankings.append(dict(comparison=label, ranking=name, spearman_with_default=spearmanr(score, default)[0],
                                 top100_shared_with_default=len(rh.top(score) & rh.top(default))))
            print(rankings[-1], flush=True)
    os.makedirs(cc.RESULTS_DIR, exist_ok=True)
    spectrum.to_csv(os.path.join(cc.RESULTS_DIR, "revision_hela_spectrum_original.csv"), index=False)
    pd.DataFrame(rankings).to_csv(os.path.join(cc.RESULTS_DIR, "revision_hela_rankings_original.csv"), index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--networks-dir", required=True, help="folder with G1.txt, S.txt, G2.txt and M.txt")
    parser.add_argument("--revelio-dir", required=True)
    parser.add_argument("--powers", nargs="+", type=int, default=[6, 5], help="soft powers of the rebuilt networks")
    parser.add_argument("--same-genes", action="store_true",
                        help="rebuild the networks on the genes of the paper's networks")
    parser.add_argument("--suffix", default="", help="appended to the agreement file name")
    parser.add_argument("--dimensions", action="store_true", help="only the dimension check on the paper's networks")
    args = parser.parse_args(argv)

    original = {p: pd.read_csv(os.path.join(args.networks_dir, f"{p}.txt"), sep=" ", index_col=0) for p in PHASES}
    genes = [g for g in original["G1"].index if all(g in original[p].index for p in PHASES)]
    print(f"paper's networks: {len(genes)} genes in all four phases", flush=True)
    if args.dimensions:
        dimensions(original, genes)
        return
    paper_ranks, paper_dedi = rank(original, genes, dedi=True)

    _, _, _, markers = cc.read_revelio(args.revelio_dir)
    marker_genes = set().union(*(set(markers[c].dropna().astype(str)) for c in markers.columns)) - {""}
    in_markers = [g for g in genes if g in marker_genes]
    marker_records = []
    for label in TRANSITIONS.values():
        for method, score in (("n2v2r", paper_ranks[label]), ("DeDi", paper_dedi[label])):
            top = list(score.sort_values(ascending=False).index[:TOP_MARKERS])
            marker_records.append({"comparison": label, "method": method, f"markers in top {TOP_MARKERS}":
                                   sum(g in marker_genes for g in top), "markers among genes": len(in_markers),
                                   "genes": len(genes), "expected by chance": TOP_MARKERS * len(in_markers) / len(genes),
                                   "top genes": " ".join(top)})
            print(marker_records[-1], flush=True)
    os.makedirs(cc.RESULTS_DIR, exist_ok=True)
    pd.DataFrame(marker_records).to_csv(os.path.join(cc.RESULTS_DIR, "hela_original_networks_markers.csv"), index=False)

    rankings = pd.DataFrame({f"{label} borda": r for label, r in paper_ranks.items()})
    for label, r in paper_ranks.items():
        rankings[f"{label} rank"] = r.rank(ascending=False, method="first").astype(int)
    rankings.index.name = "gene"
    os.makedirs(cc.RESULTS_DIR, exist_ok=True)
    rankings.sort_values("G1 -> S rank").to_csv(os.path.join(cc.RESULTS_DIR, "hela_original_networks_rankings.csv"))
    for label, r in paper_ranks.items():
        print(f"{label}, top 20 on the paper's networks: {', '.join(r.sort_values(ascending=False).index[:20])}",
              flush=True)

    records = []
    for power in args.powers:
        _, groups, _, build = cc.prepare(args.revelio_dir, setting="paper", paper_power=power,
                                         paper_genes=genes if args.same_genes else None)
        rebuilt = {p: build(groups[p]) for p in PHASES}
        rebuilt_genes = list(rebuilt["G1"].columns)
        rebuilt_ranks = rank(rebuilt, rebuilt_genes)
        shared = [g for g in genes if g in set(rebuilt_genes)]
        for label in TRANSITIONS.values():
            a, b = paper_ranks[label].loc[shared], rebuilt_ranks[label].loc[shared]
            top_a = set(a.sort_values(ascending=False).index[:TOP_K])
            top_b = set(b.sort_values(ascending=False).index[:TOP_K])
            records.append({"rebuilt power": power, "same genes": args.same_genes, "comparison": label,
                            "paper genes": len(genes),
                            "rebuilt genes": len(rebuilt_genes), "shared genes": len(shared),
                            "spearman": spearmanr(a, b)[0], f"top {TOP_K} shared": len(top_a & top_b)})
            print(records[-1], flush=True)
    pd.DataFrame(records).to_csv(os.path.join(cc.RESULTS_DIR, f"hela_original_networks_agreement{args.suffix}.csv"), index=False)


if __name__ == "__main__":
    main()
