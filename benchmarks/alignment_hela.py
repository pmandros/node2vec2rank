"""Robust alignment on real cells: the HeLa cell-cycle data of cell_cycle.py.

For the three phase transitions and three random splits of the G1 cells (where
nothing differs), with and without ``align="robust"``:

- gene level: genes called by ``significance()`` at q < 0.1, and Reactome
  cell-cycle genes among the 100 top-ranked genes of the degree-adjusted z;
- pathway level: Reactome pathways called at FDR 0.1 by the fast gene-set
  test (20 label shuffles within batch), all and cell-cycle pathways.

    OMP_NUM_THREADS=4 python benchmarks/alignment_hela.py --revelio-dir path/to/Revelio/data

Results: results/alignment_hela.csv.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import cell_cycle as cc  # noqa: E402
from node2vec2rank import N2V2R  # noqa: E402
from node2vec2rank.fast_gene_sets import fast_gene_set_test  # noqa: E402
from node2vec2rank.permutation import read_gmt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--revelio-dir", required=True)
    parser.add_argument("--null-splits", type=int, default=3)
    parser.add_argument("--num-permutations", type=int, default=20)
    args = parser.parse_args()

    _, groups, batches, build = cc.prepare(args.revelio_dir)
    gene_sets = read_gmt(cc.REACTOME)
    names = pd.Index(list(gene_sets))
    cell_cycle_sets = set(names[names.str.contains(cc.CELL_CYCLE)])
    genes = groups["G1"].columns
    cell_cycle_genes = np.isin(genes, sorted({g for name in cell_cycle_sets for g in gene_sets[name]}))

    comparisons = [(f"{a} -> {b}", groups[a], groups[b], (batches[a], batches[b])) for a, b in cc.COMPARISONS]
    for split in range(args.null_splits):
        half = cc.random_halves(batches["G1"], np.random.default_rng(split))
        comparisons.append((f"null: G1 halves {split}", groups["G1"][half], groups["G1"][~half],
                            (batches["G1"][half], batches["G1"][~half])))

    rows = []
    for label, group_a, group_b, strata in comparisons:
        for align in (None, "robust"):
            model = N2V2R([np.asarray(build(group_a)), np.asarray(build(group_b))], nodes=list(genes),
                          align=align, seed=42, verbose=-1)
            model.fit_transform_rank()
            test = model.significance()["1"]
            top = np.argsort(-np.nan_to_num(test["z"].to_numpy(), nan=-np.inf))[:100]
            sets = fast_gene_set_test(group_a, group_b, gene_sets, build_network=build,
                                      num_permutations=args.num_permutations, strata=strata, random_state=0,
                                      align=align, seed=42)
            called = sets.index[sets["qvalue"] < 0.1]
            rows.append(dict(comparison=label, align=align or "none",
                             genes_called=int((test["qvalue"] < 0.1).sum()),
                             cell_cycle_genes_in_top_100=int(cell_cycle_genes[top].sum()),
                             pathways_called=len(called),
                             cell_cycle_pathways_called=int(sum(name in cell_cycle_sets for name in called))))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(cc.RESULTS_DIR, "alignment_hela.csv"), index=False)


if __name__ == "__main__":
    main()
