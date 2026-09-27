"""HeLa cell cycle: signed correlation vs soft-powered co-expression networks as n2v2r input.

Companion to benchmarks/correlation_uase.py, on real data. It reuses the pipeline of
benchmarks/cell_cycle.py (Revelio HeLa S3 cells, Revelio phases, 2,000 most variable genes, metacells
per phase, genes z-scored within phase and batch), so it needs that script and
node2vec2rank.singlecell / fast_gene_sets from PR #5. Only the network built on the metacells changes:

* ``hdWGCNA signed``: ((1 + cor) / 2)^power with the scale-free power, the paper's HeLa networks;
* ``unsigned |cor|^power``: WGCNA's unsigned network with the same power;
* ``signed correlation``: the Pearson correlation of the metacells, zero diagonal, no power.

For the sequential comparisons and a null split of the G1 cells, it reports n2v2r's gene calls
(``significance()``, q < 0.1) and the fast gene-set test on Reactome (cell-cycle pathways at FDR 0.1,
their AUROC).

    OMP_NUM_THREADS=1 python benchmarks/cell_cycle_signed.py --revelio-dir path/to/Revelio/data
"""

import argparse
import functools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import cell_cycle as cc  # noqa: E402
from node2vec2rank import N2V2R  # noqa: E402
from node2vec2rank.fast_gene_sets import fast_gene_set_test  # noqa: E402
from node2vec2rank.singlecell import metacells  # noqa: E402


def correlation_network(expression, kind, power, k=25, max_shared=10):
    values = np.asarray(metacells(expression, k=k, max_shared=max_shared), dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.nan_to_num(np.corrcoef(values, rowvar=False), nan=0.0)
    if kind == "hdWGCNA signed":
        adjacency = ((1 + correlation) / 2) ** power
    elif kind == "unsigned |cor|^power":
        adjacency = np.abs(correlation) ** power
    elif kind == "signed correlation":
        adjacency = correlation
    else:
        raise ValueError(kind)
    np.fill_diagonal(adjacency, 0)
    return adjacency


KINDS = ("hdWGCNA signed", "unsigned |cor|^power", "signed correlation")


def evaluate(label, kind, group_a, group_b, strata, build, gene_sets, is_cell_cycle):
    model = N2V2R([build(group_a), build(group_b)], nodes=list(group_a.columns), verbose=-1,
                  **cc.PAPER_PARAMS)
    model.fit_transform_rank()
    genes = next(iter(model.significance().values()))
    sets = fast_gene_set_test(group_a, group_b, gene_sets, build_network=build, num_permutations=10,
                              min_size=5, max_size=500, strata=strata, random_state=0, ranking="n2v2r",
                              **cc.PAPER_PARAMS)
    sets = sets.drop(columns="score").rename(columns={"z": "score"})
    row = cc.summarise(kind, label, sets, is_cell_cycle)
    row["genes_q<0.1"] = int((genes["qvalue"] < 0.1).sum())
    top = sets.sort_values(["qvalue", "score"], ascending=[True, False]).head(10)
    tops = pd.DataFrame({"comparison": label, "network": kind, "gene_set": top.index,
                         "z": top["score"].to_numpy(), "qvalue": top["qvalue"].to_numpy()})
    print(label, kind, {k: row[k] for k in ("called", "cell_cycle_called", "auroc_cell_cycle", "genes_q<0.1")},
          flush=True)
    return row, tops


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--revelio-dir", required=True)
    parser.add_argument("--phase-merge", choices=list(cc.PHASE_MERGES), default="earlier")
    args = parser.parse_args(argv)

    cells_table, groups, batches, hdwgcna = cc.prepare(args.revelio_dir, merge=args.phase_merge)
    power = hdwgcna.keywords["power"]
    gene_sets = cc.read_gmt(cc.REACTOME)
    names = pd.Index(list(gene_sets))
    is_cell_cycle = pd.Series(np.asarray(names.str.contains(cc.CELL_CYCLE)), index=names)

    half = cc.random_halves(batches["G1"], np.random.default_rng(0))
    comparisons = [(f"{a} -> {b}", groups[a], groups[b], (batches[a], batches[b])) for a, b in cc.COMPARISONS]
    comparisons.append(("null: G1 halves", groups["G1"][half], groups["G1"][~half],
                        (batches["G1"][half], batches["G1"][~half])))

    rows, tops = [], []
    for kind in KINDS:
        build = functools.partial(correlation_network, kind=kind, power=power)
        for label, a, b, strata in comparisons:
            row, top = evaluate(label, kind, a, b, strata, build, gene_sets, is_cell_cycle)
            row["power"] = power if kind != "signed correlation" else 1
            rows.append(row)
            tops.append(top)

    os.makedirs(cc.RESULTS_DIR, exist_ok=True)
    suffix = f"_{args.phase_merge}"
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(cc.RESULTS_DIR, f"cell_cycle_signed{suffix}.csv"), index=False)
    pd.concat(tops).to_csv(os.path.join(cc.RESULTS_DIR, f"cell_cycle_signed_top_sets{suffix}.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(3))


if __name__ == "__main__":
    main()
