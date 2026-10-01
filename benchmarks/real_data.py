"""Benchmark on real single-cell networks: the locCSN autism (ASD) and control
(CTL) gene networks that used to live in ``data/networks/locscn``.

Each network is an average of cell-specific networks (locCSN): the weight of an
edge is the fraction of cells in which the edge is present, over 211 control
and 238 ASD cells (the denominators can be read off the weights). There are
942 genes. The per-cell networks are not available, so the sample-label
permutation test cannot be run, and a null has to be built from the averaged
networks. Three analyses:

1. observed: ASD vs CTL as in the paper. The spectrum and elbow dimension, the
   degree bias of every ranking, how much the default (dimensions 4-24) and
   elbow rankings agree, and the significance calls of both.
2. null: both networks are redrawn from the pooled edge frequencies, as
   Binomial(num_cells, pooled frequency) / num_cells per edge and group. No
   gene differs between the two groups, while the degree structure and the
   edge-level sampling noise of the real data are kept. This checks the
   calibration and degree bias of the tests. Cells are treated as
   independent per edge, which ignores that the edges of a gene co-vary across
   cells, so the null is less noisy than a cell-level resampling would be.
   A naive alternative, swapping every edge's weight between the two networks
   at random, is also reported to show why it is not a valid node-level null.
3. spike-in: in the second network, pairs of genes swap their neighbourhoods
   (their rows and columns). The pairs are either adjacent in degree (a
   degree-preserving rewiring, as in the demo network) or random. The spike-in
   is added on top of the resampled null and on top of the real ASD network.

Run from the repository root (about 3 minutes):

    python benchmarks/real_data.py

The networks are read from ``--data-dir`` if given, otherwise from the git
history of this repository (they were removed in commit b7bba41).
"""

import argparse
import io
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from node2vec2rank.embedding import embed  # noqa: E402
from simulations import (DEFAULT_DIMS, PVALUE_BINS, RESULTS_DIR, auroc, score_methods,  # noqa: E402
                         spearman)

HISTORY_COMMIT = "b7bba41^"
FILES = {"CTL": "avg_csn_ctl.csv", "ASD": "avg_csn_asd.csv"}
NUM_CELLS = {"CTL": 211, "ASD": 238}
TESTED = ("n2v2r degree-adjusted z", "n2v2r degree-adjusted z (elbow)")


def load_networks(data_dir=None):
    """Returns (genes, [CTL, ASD]) as dense arrays."""
    tables = []
    for filename in FILES.values():
        if data_dir:
            source = os.path.join(data_dir, filename)
        else:
            blob = subprocess.run(["git", "-C", REPO, "show", f"{HISTORY_COMMIT}:data/networks/locscn/{filename}"],
                                  check=True, capture_output=True).stdout
            source = io.BytesIO(blob)
        tables.append(pd.read_csv(source, index_col=0))
    genes = tables[0].index
    if not all(t.index.equals(genes) and t.columns.equals(genes) for t in tables):
        raise ValueError("Both networks must have the same genes in the same order")
    return genes, [t.to_numpy(dtype=float) for t in tables]


def _symmetric(upper_values, num_nodes, upper):
    matrix = np.zeros((num_nodes, num_nodes))
    matrix[upper] = upper_values
    return matrix + matrix.T


def binomial_null(graphs, rng):
    num_nodes = len(graphs[0])
    upper = np.triu_indices(num_nodes, 1)
    cells = np.array(list(NUM_CELLS.values()))
    pooled = sum(g[upper] * n for g, n in zip(graphs, cells)) / cells.sum()
    return [_symmetric(rng.binomial(n, pooled) / n, num_nodes, upper) for n in cells]


def edge_swap_null(graphs, rng):
    num_nodes = len(graphs[0])
    upper = np.triu_indices(num_nodes, 1)
    first, second = graphs[0][upper].copy(), graphs[1][upper].copy()
    swap = rng.random(len(first)) < 0.5
    first[swap], second[swap] = second[swap], graphs[0][upper][swap]
    return [_symmetric(first, num_nodes, upper), _symmetric(second, num_nodes, upper)]


def spike_in(graph, rng, frac=0.05, degree_matched=True):
    """Swaps the neighbourhoods of pairs of genes; returns (graph, changed)."""
    num_nodes = len(graph)
    degree = graph.sum(axis=0)
    connected = np.flatnonzero(degree > 0)
    num_pairs = int(round(frac * num_nodes / 2))
    permutation = np.arange(num_nodes)
    if degree_matched:
        by_degree = connected[np.argsort(degree[connected], kind="stable")]
        used = np.zeros(num_nodes, bool)
        pairs = 0
        for position in rng.permutation(len(by_degree) - 1):
            i, j = by_degree[position], by_degree[position + 1]
            if used[i] or used[j]:
                continue
            used[[i, j]] = True
            permutation[[i, j]] = j, i
            pairs += 1
            if pairs == num_pairs:
                break
    else:
        chosen = rng.choice(connected, 2 * num_pairs, replace=False)
        permutation[chosen[:num_pairs]], permutation[chosen[num_pairs:]] = chosen[num_pairs:], chosen[:num_pairs]
    return graph[np.ix_(permutation, permutation)], permutation != np.arange(num_nodes)


def _null_records(scores, degree, tests, **labels):
    connected = degree > 0
    records, histograms = [], []
    for method, score in scores.items():
        record = {**labels, "method": method,
                  "spearman_with_degree": spearman(score[connected], degree[connected])}
        if method in tests:
            pvalues, qvalues = tests[method]
            valid = ~np.isnan(pvalues)
            record["frac_p_below_0.05"] = np.mean(pvalues[valid] < 0.05)
            record["num_called_q_0.1"] = int(np.sum(qvalues < 0.1))
            counts, _ = np.histogram(pvalues[valid], bins=PVALUE_BINS)
            histograms.extend({**labels, "method": method, "bin_start": start, "count": count}
                              for start, count in zip(PVALUE_BINS[:-1], counts))
        records.append(record)
    return records, histograms


def run_observed(genes, graphs):
    _, singular_values = embed(graphs, 60, "uase", random_state=0, return_singular_values=True)
    scores, degree, tests, elbow = score_methods(graphs, seed=0)
    records, histograms = _null_records(scores, degree, tests, analysis="observed")
    default, elbow_ranking = scores["n2v2r (default)"], scores["n2v2r elbow dimension"]
    connected = degree > 0
    agreement = {"elbow_dimension": elbow,
                 "spearman_default_vs_elbow": spearman(default[connected], elbow_ranking[connected])}
    for k in (50, 100):
        top_default = set(np.argsort(-default, kind="stable")[:k])
        top_elbow = set(np.argsort(-elbow_ranking, kind="stable")[:k])
        agreement[f"top_{k}_overlap"] = len(top_default & top_elbow)
    top = pd.DataFrame({"gene": genes, "degree_ctl": graphs[0].sum(axis=0), "degree_asd": graphs[1].sum(axis=0),
                        "borda_default": default, "borda_elbow": elbow_ranking,
                        "z": scores["n2v2r degree-adjusted z"], "pvalue": tests[TESTED[0]][0],
                        "qvalue": tests[TESTED[0]][1], "z_elbow": scores["n2v2r degree-adjusted z (elbow)"],
                        "qvalue_elbow": tests[TESTED[1]][1]})
    top = top.sort_values("borda_default", ascending=False).head(30)
    spectrum = pd.DataFrame({"dimension": np.arange(1, len(singular_values) + 1),
                             "singular_value": singular_values})
    return records, histograms, agreement, top, spectrum


def run_nulls(graphs, num_replicates):
    records, histograms = [], []
    for null_name, make_null in (("binomial", binomial_null), ("edge swap", edge_swap_null)):
        for replicate in range(num_replicates):
            tic = time.time()
            rng = np.random.default_rng([replicate, len(null_name)])
            scores, degree, tests, elbow = score_methods(make_null(graphs, rng), seed=replicate)
            rec, hist = _null_records(scores, degree, tests, analysis=f"null: {null_name}",
                                      replicate=replicate, elbow_dimension=elbow)
            records.extend(rec)
            histograms.extend(hist)
            print(f"null {null_name:9s} rep={replicate} elbow={elbow} ({time.time() - tic:.1f}s)", flush=True)
    return records, histograms


def run_spike_ins(graphs, num_replicates, frac=0.05):
    records = []
    for background in ("resampled null", "real"):
        for degree_matched in (True, False):
            for replicate in range(num_replicates):
                rng = np.random.default_rng([replicate, int(degree_matched), len(background)])
                pair = binomial_null(graphs, rng) if background == "resampled null" else graphs
                spiked, changed = spike_in(pair[1], rng, frac=frac, degree_matched=degree_matched)
                scores, _, tests, elbow = score_methods([pair[0], spiked], seed=replicate)
                labels = {"background": background,
                          "spike_in": "degree-matched swap" if degree_matched else "random swap",
                          "replicate": replicate, "elbow_dimension": elbow}
                for method, score in scores.items():
                    record = {**labels, "method": method, "auroc": auroc(changed, score)}
                    if method in tests:
                        called = tests[method][1] < 0.1
                        record["power_q_0.1"] = called[changed].mean()
                        record["false_calls_q_0.1"] = int((called & ~changed).sum())
                    records.append(record)
                print(f"spike-in {background:14s} {labels['spike_in']:19s} rep={replicate} elbow={elbow}",
                      flush=True)
    return records


def figures(observed, nulls, histograms, spikes, spectrum, agreement):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figures import BLUE, GRID, METHODS, MUTED_INK, ORANGE, Y_POSITIONS, _method_axis

    # spectrum, degree bias and calibration
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
    ax = axes[0]
    shown = spectrum.head(40)
    ax.plot(shown.dimension, shown.singular_value, "o-", color=BLUE, markersize=3, linewidth=1)
    ax.set_yscale("log")
    ax.axvline(agreement["elbow_dimension"], color=ORANGE, linestyle="--", linewidth=1.2)
    ax.axvspan(min(DEFAULT_DIMS), max(DEFAULT_DIMS), color=GRID, alpha=0.5, linewidth=0)
    ax.set_xlabel("dimension")
    ax.set_ylabel("singular value (log)")
    ax.set_title(f"Spectrum: elbow at {agreement['elbow_dimension']} (dashed),\ndefault dimensions shaded",
                 fontsize=9, loc="left")

    ax = axes[1]
    observed_bias = observed.set_index("method").spearman_with_degree.reindex(METHODS)
    null_bias = (nulls[nulls.analysis == "null: binomial"].groupby("method").spearman_with_degree.mean()
                 .reindex(METHODS))
    ax.barh(Y_POSITIONS + 0.18, observed_bias, height=0.34, color=BLUE, label="ASD vs CTL")
    ax.barh(Y_POSITIONS - 0.18, null_bias, height=0.34, color=ORANGE, label="resampled null")
    ax.axvline(0, color=MUTED_INK, linewidth=1)
    _method_axis(ax, True)
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("Spearman with degree")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Degree bias", fontsize=9, loc="left")

    ax = axes[2]
    for method, color, label in ((TESTED[0], BLUE, "all dimensions"), (TESTED[1], ORANGE, "elbow")):
        subset = histograms[(histograms.analysis == "null: binomial") & (histograms.method == method)]
        counts = subset.groupby("bin_start")["count"].sum()
        density = counts / counts.sum() / 0.05
        ax.step(np.append(counts.index, 1), np.append(density, density.iloc[-1]), where="post",
                color=color, linewidth=1.5, label=label)
    ax.axhline(1, color=MUTED_INK, linewidth=1, linestyle=":")
    ax.set_ylim(0, 2)
    ax.set_xlabel("p-value")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("p-values under the resampled null,\n20 replicates (dotted: uniform)", fontsize=9, loc="left")
    fig.savefig(os.path.join(RESULTS_DIR, "real_data.png"), dpi=150)

    # spike-in accuracy
    settings = [("resampled null", "degree-matched swap"), ("resampled null", "random swap"),
                ("real", "degree-matched swap"), ("real", "random swap")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True, sharex=True)
    for ax, (background, kind) in zip(axes, settings):
        subset = spikes[(spikes.background == background) & (spikes.spike_in == kind)]
        summary = subset.groupby("method").auroc.agg(["mean", "std"]).reindex(METHODS)
        ax.hlines(Y_POSITIONS, 0.4, summary["mean"], color=GRID, linewidth=2)
        ax.errorbar(summary["mean"], Y_POSITIONS, xerr=summary["std"], fmt="o", color=BLUE,
                    ecolor=MUTED_INK, elinewidth=1, markersize=5)
        _method_axis(ax, ax is axes[0])
        ax.set_xlim(0.4, 1.01)
        ax.set_xlabel("AUROC for the swapped genes")
        ax.set_title(f"{kind}\non the {background} pair", fontsize=9)
    fig.suptitle("Spike-in on the locCSN networks (5% of genes, mean and sd)", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "real_data_spike_in.png"), dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", help="directory with avg_csn_ctl.csv and avg_csn_asd.csv")
    parser.add_argument("--replicates", type=int, default=20)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    genes, graphs = load_networks(args.data_dir)
    observed, observed_histograms, agreement, top, spectrum = run_observed(genes, graphs)
    null_records, null_histograms = run_nulls(graphs, args.replicates)
    spike_records = run_spike_ins(graphs, args.replicates)

    observed = pd.DataFrame.from_records(observed)
    nulls = pd.DataFrame.from_records(null_records)
    histograms = pd.DataFrame.from_records(observed_histograms + null_histograms)
    spikes = pd.DataFrame.from_records(spike_records)
    pd.concat([observed, nulls]).to_csv(os.path.join(RESULTS_DIR, "real_data.csv"), index=False)
    histograms.to_csv(os.path.join(RESULTS_DIR, "real_data_pvalue_histograms.csv"), index=False)
    spikes.to_csv(os.path.join(RESULTS_DIR, "real_data_spike_in.csv"), index=False)
    top.to_csv(os.path.join(RESULTS_DIR, "real_data_top_genes.csv"), index=False)
    spectrum.to_csv(os.path.join(RESULTS_DIR, "real_data_spectrum.csv"), index=False)
    pd.Series(agreement).to_csv(os.path.join(RESULTS_DIR, "real_data_agreement.csv"), header=["value"])

    print("\nASD vs CTL:", agreement)
    print(observed.round(3).to_string(index=False))
    print(top.head(15).round(3).to_string(index=False))
    print("\nNulls (mean over replicates):")
    print(nulls.groupby(["analysis", "method"])[["spearman_with_degree", "frac_p_below_0.05", "num_called_q_0.1"]]
          .agg(["mean", "max"]).round(3).to_string())
    print("\nSpike-ins (mean over replicates):")
    print(spikes.groupby(["background", "spike_in", "method"])[["auroc", "power_q_0.1", "false_calls_q_0.1"]]
          .mean().round(3).to_string())
    figures(observed, nulls, histograms, spikes, spectrum, agreement)
