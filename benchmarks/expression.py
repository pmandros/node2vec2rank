"""Benchmark on simulated gene expression: co-expression networks built from
samples, as in the paper, with a known set of rewired genes.

For every replicate, two conditions of expression are simulated with
node2vec2rank.simulate (1,000 genes, 100 samples per condition, 5 modules,
10% of genes rewired, 10% of the others differentially expressed as decoys),
and WGCNA-style networks |cor|^6 are built per condition. Three tests are compared:

- empirical null: N2V2R.significance() on the networks alone
- empirical null (elbow): N2V2R.significance(dimensions="elbow")
- permutation: node2vec2rank.permutation.permutation_test on the expression

Two analyses:

1. calibration: the samples of both conditions are pooled and split at random,
   so no gene truly differs between the two groups;
2. power: the true condition split, scored against the ground truth.

Run from the repository root (about 25 minutes on 4 cores):

    python benchmarks/expression.py
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from node2vec2rank import N2V2R  # noqa: E402
from node2vec2rank.permutation import permutation_test  # noqa: E402
from node2vec2rank.simulate import coexpression_network, simulate_expression  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PVALUE_BINS = np.linspace(0, 1, 21)
TESTS = ("empirical null", "empirical null (elbow)", "permutation")


def auroc(labels, scores):
    from scipy.stats import rankdata
    ranks = rankdata(scores)
    positives = labels.sum()
    return (ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * (len(labels) - positives))


def run_tests(expression_a, expression_b, num_permutations, seed):
    networks = [coexpression_network(expression_a), coexpression_network(expression_b)]
    model = N2V2R(networks, seed=seed, verbose=-1)
    model.fit_transform_rank()
    borda = model.aggregate_transform()["1"]["borda_ranks"]
    results = {"empirical null": model.significance()["1"],
               "empirical null (elbow)": model.significance(dimensions="elbow")["1"],
               "permutation": permutation_test(expression_a, expression_b,
                                               num_permutations=num_permutations,
                                               random_state=seed, seed=seed)}
    return results, borda


def gene_classes(sim):
    moved = sim.rewired.to_numpy()
    isolated = ((sim.modules.condition_1 == -1) & (sim.modules.condition_2 == -1)).to_numpy() & ~moved
    return {
        "rewired": moved,
        "module partners": ~moved & ~isolated,
        "never in a module": isolated,
        # decoys whose co-expression is unchanged even at the network level
        "differential expression only": sim.differentially_expressed.to_numpy() & isolated,
    }


def run(num_replicates, num_permutations, num_genes=1000, num_samples=100):
    records, histograms = [], []
    for replicate in range(num_replicates):
        tic = time.time()
        sim = simulate_expression(num_genes=num_genes, num_samples=num_samples, random_state=replicate)

        # 1. calibration: shuffle the samples between the groups
        rng = np.random.default_rng(1000 + replicate)
        pooled = pd.concat(sim.expression, axis=0)
        order = rng.permutation(len(pooled))
        shuffled_a = pooled.iloc[order[:num_samples]]
        shuffled_b = pooled.iloc[order[num_samples:]]
        null_results, _ = run_tests(shuffled_a, shuffled_b, num_permutations, seed=replicate)
        for test, result in null_results.items():
            p = result["pvalue"].dropna().to_numpy()
            records.append({"analysis": "shuffled labels", "replicate": replicate, "test": test,
                            "frac_p_below_0.05": np.mean(p < 0.05),
                            "num_called_q_0.1": int((result["qvalue"] < 0.1).sum())})
            counts, _ = np.histogram(p, bins=PVALUE_BINS)
            histograms.extend({"test": test, "replicate": replicate, "bin_start": start, "count": count}
                              for start, count in zip(PVALUE_BINS[:-1], counts))

        # 2. power: the true condition split
        results, borda = run_tests(*sim.expression, num_permutations, seed=replicate)
        classes = gene_classes(sim)
        rewired = classes["rewired"]
        for test, result in results.items():
            called = (result["qvalue"] < 0.1).to_numpy()
            record = {"analysis": "true conditions", "replicate": replicate, "test": test,
                      "auroc_rewired": auroc(rewired, result["z"].fillna(-np.inf).to_numpy()),
                      "num_called_q_0.1": int(called.sum())}
            record.update({f"called {name}": called[mask].mean() for name, mask in classes.items()})
            records.append(record)
        records.append({"analysis": "true conditions", "replicate": replicate, "test": "n2v2r Borda (no test)",
                        "auroc_rewired": auroc(rewired, borda.to_numpy())})
        print(f"replicate {replicate} done in {time.time() - tic:.0f}s", flush=True)
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(histograms)


def figures(results, histograms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, ink, muted, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d9d8d4"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "xtick.color": muted,
                         "ytick.color": muted, "text.color": ink, "axes.labelcolor": ink,
                         "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    for ax, test in zip(axes, TESTS):
        counts = histograms[histograms.test == test].groupby("bin_start")["count"].sum()
        density = counts / counts.sum() / 0.05
        ax.bar(counts.index + 0.025, density, width=0.045, color=blue)
        ax.axhline(1, color=orange, linestyle="--", linewidth=1.5)
        ax.set_title(test, fontsize=9)
        ax.set_xlabel("p-value")
    axes[0].set_ylabel("density")
    fig.suptitle("Shuffled sample labels (no true difference); dashed: uniform", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "expression_calibration.png"), dpi=150)

    classes = ["rewired", "module partners", "never in a module", "differential expression only"]
    power = results[(results.analysis == "true conditions") & results.test.isin(TESTS)]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
    y = np.arange(len(TESTS))[::-1]
    for ax, name in zip(axes, classes):
        values = power.groupby("test")[f"called {name}"].mean().reindex(TESTS)
        ax.barh(y, values, color=blue, height=0.6)
        for yi, value in zip(y, values):
            ax.annotate(f"{value:.2f}", (value, yi), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=8)
        ax.set_xlim(0, 1.15)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("fraction called (q < 0.1)")
        ax.grid(axis="x", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(TESTS)
    fig.suptitle("True conditions: which genes each test calls", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "expression_calls.png"), dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "expression.csv")
    histograms_path = os.path.join(RESULTS_DIR, "expression_null_pvalue_histograms.csv")
    if not args.figures_only:
        results, histograms = run(args.replicates, args.permutations)
        results.to_csv(results_path, index=False)
        histograms.to_csv(histograms_path, index=False)
    results, histograms = pd.read_csv(results_path), pd.read_csv(histograms_path)
    figures(results, histograms)
    summary = results.groupby(["analysis", "test"]).mean(numeric_only=True).drop(columns="replicate")
    print(summary.round(3).to_string())
