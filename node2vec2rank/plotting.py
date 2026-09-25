"""Diagnostic plots for n2v2r (requires matplotlib: ``pip install node2vec2rank[plot]``)."""

import numpy as np
from scipy.stats import rankdata

# a colour-blind-safe palette: blue, orange, aqua
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED_INK, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def _axes(ax):
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(figsize=(5, 3.5))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED_INK)
    ax.tick_params(colors=MUTED_INK, labelcolor=INK)
    return ax


def plot_scree(singular_values, selected_dimension=None, ax=None):
    """Plots the singular values of the joint embedding, marking the selected
    (elbow) dimension."""
    ax = _axes(ax)
    values = np.asarray(singular_values)
    dims = np.arange(1, len(values) + 1)
    ax.plot(dims, values, color=BLUE, linewidth=2, marker="o", markersize=4)
    if selected_dimension is not None:
        ax.axvline(selected_dimension, color=MUTED_INK, linestyle="--", linewidth=1)
        ax.annotate(f"elbow d = {selected_dimension}", (selected_dimension, values.max()),
                    xytext=(6, 0), textcoords="offset points", color=INK, va="top")
    ax.set_xlabel("dimension")
    ax.set_ylabel("singular value")
    return ax


def plot_ranking_agreement(agreement, ax=None):
    """Heatmap of the Spearman agreement between rankings, as returned by
    :func:`node2vec2rank.diagnostics.ranking_agreement`."""
    import matplotlib.pyplot as plt
    ax = _axes(ax)
    image = ax.imshow(agreement.to_numpy(), cmap="Blues", vmin=min(0, agreement.to_numpy().min()), vmax=1)
    labels = [str(c).replace("dim-", "d").replace("_distance-", " ") for c in agreement.columns]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    plt.colorbar(image, ax=ax, label="Spearman correlation")
    return ax


def plot_degree_bias(score, degree, ax=None, label="score"):
    """Plots the rank of a score against the rank of node degree, with a
    running median; a flat median means no degree bias."""
    ax = _axes(ax)
    score = np.asarray(score, dtype=np.float64)
    degree = np.asarray(degree, dtype=np.float64)
    valid = np.isfinite(score) & np.isfinite(degree)
    n = valid.sum()
    x = rankdata(degree[valid]) / n
    y = rankdata(score[valid]) / n
    ax.scatter(x, y, s=4, color=BLUE, alpha=0.3, linewidths=0)
    order = np.argsort(x)
    window = max(n // 20, 1)
    running = np.convolve(y[order], np.ones(window) / window, mode="valid")
    ax.plot(x[order][window // 2: window // 2 + len(running)], running, color=ORANGE, linewidth=2)
    rho = np.corrcoef(x, y)[0, 1]
    ax.set_title(f"Spearman with degree: {rho:.2f}", color=INK, fontsize=10)
    ax.set_xlabel("degree (rank quantile)")
    ax.set_ylabel(f"{label} (rank quantile)")
    return ax


def plot_significance(significance, q_threshold=0.05, ax=None):
    """Plots the z-score of every node against its degree, highlighting the
    nodes with q-value below the threshold.

    Args:
        significance: a DataFrame from :meth:`node2vec2rank.model.N2V2R.significance`.
    """
    ax = _axes(ax)
    z = significance["z"].to_numpy()
    degree = significance["degree"].to_numpy()
    significant = significance["qvalue"].to_numpy() < q_threshold
    x = rankdata(degree) / len(degree)
    ax.scatter(x[~significant], z[~significant], s=4, color=MUTED_INK, alpha=0.3, linewidths=0,
               label="not significant")
    ax.scatter(x[significant], z[significant], s=8, color=ORANGE, linewidths=0,
               label=f"q < {q_threshold} ({significant.sum()})")
    ax.axhline(0, color=GRID, linewidth=1, zorder=0)
    ax.set_xlabel("degree (rank quantile)")
    ax.set_ylabel("degree-adjusted z")
    ax.legend(frameon=False, fontsize=8)
    return ax
