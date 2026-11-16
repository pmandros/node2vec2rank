"""Which module changed, on real cells: the HeLa cell-cycle data of cell_cycle.py.

For the three phase transitions and three random splits of the G1 cells (where
nothing differs), ``module_permutation_test`` finds the modules in the network
of all cells of the comparison and tests each one by shuffling the cells
between the groups within batch. Recorded per module: its size, the
description of its change, ``move``, ``rotation``, ``strength_change`` and the
permutation p-values; the chi-square p-value that two networks alone would
give (``module_changes(test=True)``, not valid for these networks), with its
inflation; and, to name the module, the share of its genes in Reactome
cell-cycle pathways and its most over-represented Reactome pathway
(hypergeometric test among the genes of the networks).

    OMP_NUM_THREADS=4 python benchmarks/modules_hela.py --revelio-dir path/to/Revelio/data
    ... --num-modules 6 --null-splits 0 --output modules_hela_6.csv     # the transitions with six modules

Results: results/modules_hela.csv, results/modules_hela_6.csv.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import cell_cycle as cc  # noqa: E402
from node2vec2rank import module_changes, module_permutation_test  # noqa: E402
from node2vec2rank.permutation import read_gmt  # noqa: E402


def top_pathway(members, memberships, set_names):
    """The most over-represented gene set among the member genes."""
    in_module = memberships[members].sum(axis=0)
    in_all = memberships.sum(axis=0)
    pvalues = hypergeom.sf(in_module - 1, len(members), in_all, members.sum())
    best = int(np.argmin(pvalues))
    return set_names[best], int(in_module[best]), int(in_all[best]), float(min(1.0, pvalues[best] * len(set_names)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--revelio-dir", required=True)
    parser.add_argument("--null-splits", type=int, default=3)
    parser.add_argument("--num-permutations", type=int, default=199)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--num-modules", type=int, default=None,
                        help="number of modules instead of the number chosen from the data")
    parser.add_argument("--output", default="modules_hela.csv")
    args = parser.parse_args()

    _, groups, batches, build = cc.prepare(args.revelio_dir)
    genes = groups["G1"].columns
    gene_sets = {name: [g for g in members if g in set(genes)] for name, members in read_gmt(cc.REACTOME).items()}
    gene_sets = {name: members for name, members in gene_sets.items() if 5 <= len(members) <= 500}
    set_names = list(gene_sets)
    memberships = np.zeros((len(genes), len(set_names)), dtype=bool)
    for column, members in enumerate(gene_sets.values()):
        memberships[genes.get_indexer(members), column] = True
    cell_cycle = memberships[:, [bool(cc.CELL_CYCLE.search(name)) for name in set_names]].any(axis=1)

    comparisons = [(f"{a} -> {b}", groups[a], groups[b], (batches[a], batches[b])) for a, b in cc.COMPARISONS]
    for split in range(args.null_splits):
        half = cc.random_halves(batches["G1"], np.random.default_rng(split))
        comparisons.append((f"null: G1 halves {split}", groups["G1"][half], groups["G1"][~half],
                            (batches["G1"][half], batches["G1"][~half])))

    tables = []
    for label, group_a, group_b, strata in comparisons:
        changes, nodes = module_permutation_test(group_a, group_b, build_network=build, strata=strata,
                                                 num_permutations=args.num_permutations, random_state=0,
                                                 num_modules=args.num_modules, n_jobs=args.jobs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # what two networks alone would give; the test standardises within the groups, as here
            chi_square = module_changes(np.asarray(build(group_a)), np.asarray(build(group_b)), nodes["module"],
                                        test=True)
        changes["chi_square_pvalue"] = chi_square["pvalue"].reindex(changes.index)
        changes["chi_square_inflation"] = chi_square.attrs["inflation"]
        annotation = []
        for module in changes.index:
            members = (nodes["module"] == module).to_numpy()
            annotation.append((cell_cycle[members].mean(), *top_pathway(members, memberships, set_names)))
        changes[["cell_cycle_share", "top_pathway", "top_pathway_genes", "top_pathway_size",
                 "top_pathway_pvalue"]] = annotation
        changes.insert(0, "comparison", label)
        changes.insert(1, "unassigned_genes", int((nodes["module"] == 0).sum()))
        tables.append(changes.reset_index())
        print(tables[-1].drop(columns=["comparison", "within_before", "within_after", "partner_before",
                                       "partner_after"]).round(3).to_string(index=False), flush=True)
    pd.concat(tables).to_csv(os.path.join(cc.RESULTS_DIR, args.output), index=False)


if __name__ == "__main__":
    pd.set_option("display.width", 320)
    main()
