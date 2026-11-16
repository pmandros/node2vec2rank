"""Which module changed: detection of modules and the report of their changes
(``N2V2R.module_changes``, node2vec2rank/modules.py).

Two graphs with known modules are simulated, with one change of a whole
module or none:

none         nothing changes
weakens      the edges within module 0 are 0.6 times as likely (block models),
             or its genes load 0.6 times as strongly on its factor (co-expression)
strengthens  the reverse (1.5 times; co-expression: weaker in the first condition)
merge        modules 0 and 1 connect to each other as they do within
split        half of module 0 separates from the other half
reconnects   modules 0 and 1 connect more to each other, without merging (block models)
switch       10% of the nodes move to another module one by one; no module changes as a whole

Models: a binary block model with heavy-tailed degrees (1,000 nodes, 6
communities), the same with weights, with 12 communities and 2,000 nodes, a
small one (300 nodes, 4 communities, weak structure) and a sparse one; and
co-expression networks from a factor model with 150 samples per condition
(1,000 genes, 5 modules, 30% of the genes in no module), as ((1 + cor) / 2)^10
and as |cor|^6.

Recorded per replicate: the number of modules found and their agreement with
the truth (adjusted Rand index); whether the modules with the largest ``move``
are the changed ones and whether their change is described correctly; the
rotation and strength change of changed and unchanged modules; and, for the
chi-square test per module, the share of unchanged and changed modules with
p < 0.05, with the modules found, with the true number of modules and with
twice as many (to check that choosing the modules on the same data does not
bias the test).

    OMP_NUM_THREADS=1 python benchmarks/modules.py            # about 5 minutes on 4 cores

Results: results/modules.csv (per module), results/modules_summary.csv.
"""

import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.special import comb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from node2vec2rank import coexpression_network  # noqa: E402
from node2vec2rank.modules import detect_modules, module_changes  # noqa: E402

RESULTS = os.path.join(HERE, "results")
# the description of a changed module: exact, and what it reads when two modules
# that merge, or the halves of one that splits, are found as a single module
EXPECTED = {"weakens": ("weaker",), "strengthens": ("stronger",), "merge": ("merges with", "stronger"),
            "split": ("splits from", "weaker"), "reconnects": ("connects more to", "stronger")}


# ---------------------------------------------------------------- simulators

def block_pair(rng, change, num_nodes=1000, num_blocks=6, p_in=0.2, p_out=0.03, weighted=False):
    """Two degree-corrected block model graphs. Returns the graphs, one label
    per node (the same label means the same community in both graphs), the
    labels of the changed communities and the nodes that moved one by one."""
    before = rng.integers(0, num_blocks, num_nodes)
    joint, after = before.copy(), before.copy()
    theta = rng.pareto(2.5, num_nodes) + 1
    theta /= theta.mean()
    block_before = np.full((num_blocks, num_blocks), p_out)
    np.fill_diagonal(block_before, p_in)
    block_after = block_before.copy()
    affected, moved = set(), np.zeros(num_nodes, bool)
    if change == "weakens":
        block_after[0, 0], affected = 0.6 * p_in, {0}
    elif change == "strengthens":
        block_after[0, 0], affected = 1.5 * p_in, {0}
    elif change == "merge":
        block_after[0, 1] = block_after[1, 0] = p_in
        affected = {0, 1}
    elif change == "reconnects":
        block_after[0, 1] = block_after[1, 0] = 0.5 * p_in
        affected = {0, 1}
    elif change == "split":
        joint[(before == 0) & (rng.random(num_nodes) < 0.5)] = num_blocks
        after = joint.copy()
        block_after = np.full((num_blocks + 1, num_blocks + 1), p_out)
        np.fill_diagonal(block_after, p_in)
        affected = {0, num_blocks}
    elif change == "switch":
        movers = rng.choice(num_nodes, num_nodes // 10, replace=False)
        after[movers] = (before[movers] + 1 + rng.integers(0, num_blocks - 1, len(movers))) % num_blocks
        moved[movers] = True
    graphs = []
    for labels, block in ((before, block_before), (after, block_after)):
        probs = np.clip(np.outer(theta, theta) * block[labels][:, labels], 0, 1)
        if weighted:
            upper = np.triu(rng.poisson(5 * probs) * rng.gamma(2.0, 0.5, probs.shape), 1)
        else:
            upper = np.triu(rng.random(probs.shape) < probs, 1).astype(float)
        graphs.append(upper + upper.T)
    return graphs, joint, affected, moved


def coexpression_samples(rng, change, num_genes=1000, num_modules=5, num_samples=150, weaker=0.6):
    """Expression of two conditions from a factor model; label -1 is no module."""
    before = np.where(rng.random(num_genes) < 0.7, rng.integers(0, num_modules, num_genes), -1)
    joint, after = before.copy(), before.copy()
    loadings = rng.uniform(0.3, 0.9, num_genes) * (before >= 0)
    scale = [np.ones(num_genes), np.ones(num_genes)]
    affected, moved, merge = set(), np.zeros(num_genes, bool), False
    if change == "weakens":
        scale[1], affected = np.where(before == 0, weaker, 1.0), {0}
    elif change == "strengthens":
        scale[0], affected = np.where(before == 0, weaker, 1.0), {0}
    elif change == "merge":
        merge, affected = True, {0, 1}
    elif change == "split":
        joint[(before == 0) & (rng.random(num_genes) < 0.5)] = num_modules
        after, affected = joint.copy(), {0, num_modules}
    elif change == "switch":
        movers = rng.choice(np.flatnonzero(before >= 0), num_genes // 10, replace=False)
        after[movers] = (before[movers] + 1 + rng.integers(0, num_modules - 1, len(movers))) % num_modules
        moved[movers] = True
    samples = []
    for condition, labels in enumerate((before, after)):
        factors = rng.standard_normal((num_samples, num_modules + 1))
        if merge and condition == 1:
            factors[:, 1] = factors[:, 0]
        load = loadings * scale[condition]
        samples.append(np.where(labels >= 0, factors[:, np.maximum(labels, 0)] * load, 0.0)
                       + rng.standard_normal((num_samples, num_genes)) * np.sqrt(1 - load ** 2))
    return samples, joint, affected, moved


def coexpression_pair(rng, change, power=10, signed=True, **kwargs):
    """Two co-expression networks from the factor model."""
    samples, joint, affected, moved = coexpression_samples(rng, change, **kwargs)
    graphs = [coexpression_network(expression, power=power, signed=signed) for expression in samples]
    return graphs, joint, affected, moved


BLOCK_CHANGES = ("none", "weakens", "strengthens", "merge", "split", "reconnects", "switch")
COEXPRESSION_CHANGES = ("none", "weakens", "strengthens", "merge", "split", "switch")
MODELS = {
    "binary block model": (lambda rng, change: block_pair(rng, change), BLOCK_CHANGES, True),
    "weighted block model": (lambda rng, change: block_pair(rng, change, weighted=True), BLOCK_CHANGES, True),
    "binary, 12 communities": (lambda rng, change: block_pair(rng, change, num_nodes=2000, num_blocks=12),
                               ("none", "weakens", "merge"), True),
    "binary, small": (lambda rng, change: block_pair(rng, change, num_nodes=300, num_blocks=4, p_in=0.1),
                      ("none", "weakens", "merge"), True),
    "binary, sparse": (lambda rng, change: block_pair(rng, change, p_in=0.04, p_out=0.008),
                       ("none", "weakens", "merge"), True),
    "co-expression ((1+cor)/2)^10": (lambda rng, change: coexpression_pair(rng, change), COEXPRESSION_CHANGES, False),
    "co-expression |cor|^6": (lambda rng, change: coexpression_pair(rng, change, power=6, signed=False),
                              COEXPRESSION_CHANGES, False),
}
SETTINGS = [(model, change) for model, (_, changes, _) in MODELS.items() for change in changes]


def adjusted_rand_index(first, second):
    first, second = np.unique(first, return_inverse=True)[1], np.unique(second, return_inverse=True)[1]
    table = np.zeros((first.max() + 1, second.max() + 1))
    np.add.at(table, (first, second), 1)
    together = comb(table, 2).sum()
    rows, columns = comb(table.sum(axis=1), 2).sum(), comb(table.sum(axis=0), 2).sum()
    expected = rows * columns / comb(len(first), 2)
    return (together - expected) / (0.5 * (rows + columns) - expected)


def run_one(task):
    index, replicate = task
    model, change = SETTINGS[index]
    simulate, _, independent = MODELS[model]
    graphs, truth, affected, moved = simulate(np.random.default_rng([replicate, index]), change)
    num_true = len(set(truth[truth >= 0]))
    rows = []
    for mode, num_modules in (("found", None), ("true number", num_true), ("twice as many", 2 * num_true)):
        nodes = detect_modules(graphs, num_modules=num_modules, random_state=replicate)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modules = module_changes(graphs[0], graphs[1], nodes["module"], test=True)
        scored = (truth >= 0) & ~moved & (nodes["module"].to_numpy() > 0)
        agreement = adjusted_rand_index(nodes["module"].to_numpy()[scored], truth[scored])
        in_module = truth >= 0
        for rank, (module, row) in enumerate(modules.iterrows()):
            members = nodes["module"].to_numpy() == module
            rows.append(dict(
                model=model, change=change, replicate=replicate, modules=mode, independent_edges=independent,
                num_true=num_true, num_found=len(modules), adjusted_rand=agreement,
                assigned_module_nodes=(nodes["module"].to_numpy()[in_module] > 0).mean(),
                assigned_other_nodes=((nodes["module"].to_numpy()[~in_module] > 0).mean() if (~in_module).any()
                                      else np.nan),
                module=module, rank=rank + 1, size=row["size"],
                # a module counts as changed if most of its nodes are in a changed module
                changed=bool(np.isin(truth[members], list(affected)).mean() > 0.5),
                description=row["change"], move=row["move"], rotation=row["rotation"],
                strength_change=row["strength_change"], pvalue=row["pvalue"], qvalue=row["qvalue"],
                inflation=modules.attrs["inflation"]))
    return rows


def summarise(results):
    rows = []
    for (model, change, mode), group in results.groupby(["model", "change", "modules"], sort=False):
        replicates = group.groupby("replicate")
        changed, unchanged = group[group["changed"]], group[~group["changed"]]
        row = dict(model=model, change=change, modules=mode, num_true=group["num_true"].iloc[0],
                   num_found=" ".join(str(n) for n in sorted(replicates["num_found"].first().unique())),
                   adjusted_rand=replicates["adjusted_rand"].first().mean(),
                   assigned_module_nodes=replicates["assigned_module_nodes"].first().mean(),
                   assigned_other_nodes=replicates["assigned_other_nodes"].first().mean(),
                   inflation=replicates["inflation"].first().median(),
                   unchanged_p_below_05=(unchanged["pvalue"] < 0.05).mean() if len(unchanged) else np.nan,
                   unchanged_called=(unchanged["qvalue"] < 0.1).mean() if len(unchanged) else np.nan,
                   unchanged_max_rotation=unchanged["rotation"].max() if len(unchanged) else np.nan,
                   unchanged_max_abs_strength=unchanged["strength_change"].abs().max() if len(unchanged) else np.nan,
                   unchanged_max_move=unchanged["move"].max() if len(unchanged) else np.nan)
        if len(changed):
            def top_is_changed(one):
                return bool(one.sort_values("rank").head(int(one["changed"].sum()))["changed"].all())
            with_change = [one for _, one in replicates if one["changed"].any()]
            row.update(
                replicates_with_changed_module=len(with_change) / replicates.ngroups,
                changed_ranked_first=np.mean([top_is_changed(one) for one in with_change]),
                described_exactly=changed["description"].str.startswith(EXPECTED[change][0]).mean(),
                described_correctly=changed["description"].str.startswith(EXPECTED[change]).mean(),
                changed_p_below_05=(changed["pvalue"] < 0.05).mean(),
                changed_called=(changed["qvalue"] < 0.1).mean(),
                changed_min_rotation=changed["rotation"].min(),
                changed_median_strength=changed["strength_change"].median(),
                changed_min_move=changed["move"].min())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()
    tasks = [(index, replicate) for index in range(len(SETTINGS)) for replicate in range(args.replicates)]
    with ProcessPoolExecutor(args.jobs) as pool:
        results = pd.DataFrame([row for rows in pool.map(run_one, tasks, chunksize=1) for row in rows])
    results.to_csv(os.path.join(RESULTS, "modules.csv"), index=False)
    summary = summarise(results).round(3)
    summary.to_csv(os.path.join(RESULTS, "modules_summary.csv"), index=False)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_rows", 1000)
    pd.set_option("display.max_columns", 50)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
