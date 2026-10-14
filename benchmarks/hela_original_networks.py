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
n2v2r and of DeDi.

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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--networks-dir", required=True, help="folder with G1.txt, S.txt, G2.txt and M.txt")
    parser.add_argument("--revelio-dir", required=True)
    parser.add_argument("--powers", nargs="+", type=int, default=[6, 5], help="soft powers of the rebuilt networks")
    parser.add_argument("--same-genes", action="store_true",
                        help="rebuild the networks on the genes of the paper's networks")
    parser.add_argument("--suffix", default="", help="appended to the agreement file name")
    args = parser.parse_args(argv)

    original = {p: pd.read_csv(os.path.join(args.networks_dir, f"{p}.txt"), sep=" ", index_col=0) for p in PHASES}
    genes = [g for g in original["G1"].index if all(g in original[p].index for p in PHASES)]
    print(f"paper's networks: {len(genes)} genes in all four phases", flush=True)
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
