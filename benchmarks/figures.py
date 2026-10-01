"""Draws the benchmark figures from benchmarks/results/*.csv (run simulations.py first)."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
BLUE, ORANGE, INK, MUTED_INK, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d9d8d4"

METHODS = ["DeDi", "n2v2r (default)", "n2v2r euclidean only", "n2v2r cosine only", "n2v2r ULSE",
           "n2v2r elbow dimension", "n2v2r degree-adjusted z", "n2v2r degree-adjusted z (elbow)"]
SCENARIOS = {"dcsbm": "Binary, degree-corrected SBM", "sbm": "Binary SBM",
             "weighted": "Weighted, degree-corrected SBM", "coexpression": "Co-expression (WGCNA-style)"}

Y_POSITIONS = np.arange(len(METHODS))[::-1]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED_INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED_INK, "ytick.color": MUTED_INK, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})


def _method_axis(ax, show_labels):
    # the first method is drawn at the top (see Y_POSITIONS)
    if show_labels:
        ax.set_yticks(Y_POSITIONS)
        ax.set_yticklabels(METHODS)
    else:
        ax.tick_params(labelleft=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def auroc_figure(results):
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5), sharex=True, sharey=True)
    for row, frac in enumerate((0.1, 0.3)):
        for col, (scenario, title) in enumerate(SCENARIOS.items()):
            ax = axes[row, col]
            subset = results[(results.scenario == scenario) & (results.frac_changed == frac)]
            summary = subset.groupby("method").auroc.agg(["mean", "std"]).reindex(METHODS)
            y = Y_POSITIONS
            ax.hlines(y, 0.5, summary["mean"], color=GRID, linewidth=2)
            ax.errorbar(summary["mean"], y, xerr=summary["std"], fmt="o", color=BLUE,
                        ecolor=MUTED_INK, elinewidth=1, markersize=5)
            _method_axis(ax, col == 0)
            ax.set_xlim(0.5, 1.01)
            if row == 0:
                ax.set_title(title, fontsize=9)
            if row == 1:
                ax.set_xlabel("AUROC for the changed nodes")
        axes[row, 0].annotate(f"{int(frac * 100)}% of nodes changed", (0, 1.02), xycoords="axes fraction",
                              fontsize=9, color=MUTED_INK, ha="right", va="bottom")
    fig.suptitle("Ranking accuracy (mean and sd over 20 replicates)", x=0.02, ha="left")
    fig.tight_layout()
    return fig


def degree_bias_figure(results):
    null = results[results.frac_changed == 0]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True, sharex=True)
    for col, (scenario, title) in enumerate(SCENARIOS.items()):
        ax = axes[col]
        summary = null[null.scenario == scenario].groupby("method").spearman_with_degree.mean().reindex(METHODS)
        y = Y_POSITIONS
        colors = [ORANGE if abs(v) > 0.2 else BLUE for v in summary]
        ax.barh(y, summary, color=colors, height=0.6)
        ax.axvline(0, color=MUTED_INK, linewidth=1)
        _method_axis(ax, col == 0)
        ax.set_xlim(-1, 1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Spearman with degree")
    fig.suptitle("Degree bias when nothing changes (orange: |rho| > 0.2)", x=0.02, ha="left")
    fig.tight_layout()
    return fig


def calibration_figure(histograms):
    methods = ["n2v2r degree-adjusted z", "n2v2r degree-adjusted z (elbow)"]
    fig, axes = plt.subplots(2, 4, figsize=(12, 4.8), sharex=True)
    for row, method in enumerate(methods):
        for col, (scenario, title) in enumerate(SCENARIOS.items()):
            ax = axes[row, col]
            subset = histograms[(histograms.scenario == scenario) & (histograms.method == method)]
            counts = subset.groupby("bin_start")["count"].sum()
            density = counts / counts.sum() / 0.05
            ax.bar(counts.index + 0.025, density, width=0.045, color=BLUE)
            ax.axhline(1, color=ORANGE, linewidth=1.5, linestyle="--")
            ax.set_ylim(0, max(2, density.max() * 1.1))
            if row == 0:
                ax.set_title(title, fontsize=9)
            if row == 1:
                ax.set_xlabel("p-value")
            if col == 0:
                ax.set_ylabel(("all dimensions" if row == 0 else "elbow dimension") + "\ndensity")
    fig.suptitle("Null p-values (no change, 20 replicates); dashed: uniform", x=0.02, ha="left")
    fig.tight_layout()
    return fig


def demo_figure(demo):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    summary = demo.set_index("method").reindex(METHODS)
    y = Y_POSITIONS
    ax.barh(y, summary["recall_at_k"], color=BLUE, height=0.6)
    for yi, value in zip(y, summary["recall_at_k"]):
        ax.annotate(f"{value:.2f}", (value, yi), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    _method_axis(ax, True)
    ax.set_xlim(0, 1)
    ax.set_xlabel("recall of the changed community in the top 106 nodes")
    ax.set_title(f"Demo network (elbow dimension = {int(demo.elbow_dimension.iloc[0])})", fontsize=9, loc="left")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    results = pd.read_csv(os.path.join(RESULTS_DIR, "simulations.csv"))
    histograms = pd.read_csv(os.path.join(RESULTS_DIR, "null_pvalue_histograms.csv"))
    demo = pd.read_csv(os.path.join(RESULTS_DIR, "demo.csv"))
    for name, fig in (("auroc", auroc_figure(results)), ("degree_bias", degree_bias_figure(results)),
                      ("calibration", calibration_figure(histograms)), ("demo", demo_figure(demo))):
        fig.savefig(os.path.join(RESULTS_DIR, f"{name}.png"), dpi=150)
        print(f"wrote {name}.png")
