"""A p-value per module from the samples: ``module_permutation_test``.

Two networks inferred from samples cannot say whether a module changed more
than sampling noise would make it (benchmarks/modules.py: the chi-square test
calls every unchanged co-expression module). With the samples, shuffling their
labels gives the null. Checked here on the co-expression simulations of
benchmarks/modules.py (1,000 genes, 5 modules, 150 samples per condition), as
((1 + cor) / 2)^10 and as |cor|^6 networks, without a change and with one:

none             nothing changes
weakens          the genes of one module load 0.6 times as strongly on its factor
weakens slightly the same with 0.85
merge            two modules become one
split            half of a module separates from the other half

for three ways of choosing the modules:

all samples      the default: modules found in the network of all samples
twice as many    the same, forced to twice the true number of modules
two networks     modules found in the average of the networks of the two groups
                 (``detect_modules``, as ``N2V2R.module_changes`` does without
                 samples) and passed in. These depend on the labels, so the
                 test is not exact by construction; this measures by how much.

Recorded: the share of unchanged and changed modules with p < 0.05 and with
q < 0.1, for the main test (``move``) and for the rotation and strength tests.

    OMP_NUM_THREADS=1 python benchmarks/module_permutations.py     # about 20 minutes on 4 cores

Results: results/module_permutations.csv (per module), results/module_permutations_summary.csv.
"""

import argparse
import functools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import modules as simulations  # noqa: E402
from node2vec2rank import coexpression_network, detect_modules, module_permutation_test  # noqa: E402

RESULTS = os.path.join(HERE, "results")
NETWORKS = {"co-expression ((1+cor)/2)^10": dict(power=10, signed=True),
            "co-expression |cor|^6": dict(power=6, signed=False)}
CHANGES = {"none": ("none", {}), "weakens": ("weakens", {}), "weakens slightly": ("weakens", dict(weaker=0.85)),
           "merge": ("merge", {}), "split": ("split", {})}
SETTINGS = [(network, change) for network in NETWORKS for change in CHANGES]


def run_one(task, num_permutations):
    index, replicate = task
    network, change = SETTINGS[index]
    kind, options = CHANGES[change]
    samples, truth, affected, _ = simulations.coexpression_samples(np.random.default_rng([replicate, index]),
                                                                    kind, **options)
    group_a, group_b = (pd.DataFrame(expression) for expression in samples)
    build = functools.partial(coexpression_network, **NETWORKS[network])
    num_true = len(set(truth[truth >= 0]))
    from_networks = detect_modules([np.asarray(build(group_a)), np.asarray(build(group_b))],
                                   random_state=replicate)["module"].to_numpy()
    rows = []
    for mode, options in (("all samples", {}), ("twice as many", dict(num_modules=2 * num_true)),
                          ("two networks", dict(modules=from_networks))):
        changes, nodes = module_permutation_test(group_a, group_b, build_network=build,
                                                 num_permutations=num_permutations, random_state=replicate, **options)
        found = set()
        for module, row in changes.iterrows():
            members = nodes["module"].to_numpy() == module
            # a module counts as changed if most of its nodes are in a changed module
            changed = bool(np.isin(truth[members], list(affected)).mean() > 0.5)
            if changed:
                found.update(set(truth[members]) & affected)
            rows.append(dict(network=network, change=change, replicate=replicate, modules=mode, num_true=num_true,
                             num_found=len(changes), module=module, size=row["size"], changed=changed,
                             description=row["change"], move=row["move"], rotation=row["rotation"],
                             strength_change=row["strength_change"], z=row["z"], pvalue=row["pvalue"],
                             qvalue=row["qvalue"], pvalue_rotation=row["pvalue_rotation"],
                             pvalue_strength=row["pvalue_strength"]))
        for row in rows[-len(changes):]:
            row["changed_modules_found"] = len(found) / len(affected) if affected else np.nan
    return rows


def summarise(results):
    rows = []
    for (network, change, mode), group in results.groupby(["network", "change", "modules"], sort=False):
        replicates = group.groupby("replicate")
        changed, unchanged = group[group["changed"]], group[~group["changed"]]
        row = dict(network=network, change=change, modules=mode,
                   num_found=" ".join(str(n) for n in sorted(replicates["num_found"].first().unique())),
                   unchanged_modules=len(unchanged),
                   unchanged_p_below_05=(unchanged["pvalue"] < 0.05).mean(),
                   unchanged_called=(unchanged["qvalue"] < 0.1).mean(),
                   unchanged_rotation_p_below_05=(unchanged["pvalue_rotation"] < 0.05).mean(),
                   unchanged_strength_p_below_05=(unchanged["pvalue_strength"] < 0.05).mean())
        if change != "none":
            row.update(changed_modules_found=replicates["changed_modules_found"].first().mean(),
                       changed_p_below_05=(changed["pvalue"] < 0.05).mean() if len(changed) else np.nan,
                       changed_called=(changed["qvalue"] < 0.1).mean() if len(changed) else np.nan,
                       changed_rotation_p_below_05=(changed["pvalue_rotation"] < 0.05).mean() if len(changed) else np.nan,
                       changed_strength_p_below_05=(changed["pvalue_strength"] < 0.05).mean() if len(changed) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--num-permutations", type=int, default=99)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()
    tasks = [(index, replicate) for index in range(len(SETTINGS)) for replicate in range(args.replicates)]
    with ProcessPoolExecutor(args.jobs) as pool:
        results = pd.DataFrame([row for rows in pool.map(functools.partial(run_one, num_permutations=args.num_permutations),
                                                         tasks, chunksize=1) for row in rows])
    results.to_csv(os.path.join(RESULTS, "module_permutations.csv"), index=False)
    summary = summarise(results).round(3)
    summary.to_csv(os.path.join(RESULTS, "module_permutations_summary.csv"), index=False)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_rows", 1000)
    pd.set_option("display.max_columns", 50)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
