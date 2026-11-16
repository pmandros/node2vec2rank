"""A chi-square test per node for a change of its position between two graphs.

The unfolded embedding is a regression. With the singular value decomposition
``[A_1 | ... | A_K] = U S V^T`` truncated at dimension ``d``, the position of
node ``i`` in graph ``k`` is exactly

    Y_k[i] = sum_j A_k[j, i] * w_j,        w_j = U[j] / sqrt(S),

a weighted sum of the node's own edges, with weights that are shared by all
nodes and graphs. If the edges of a node are independent given the latent
positions (the model of the paper's consistency theorem; binary, count or
continuous weights), this sum is asymptotically normal around its population
value with covariance

    C_k[i] = sum_j Var(A_k[j, i]) * w_j w_j^T,

which is the central limit theorem of UASE (Jones & Rubin-Delanchy, 2021) in
regression form. For a node whose expected edges are the same in both graphs,
the difference of its two positions has mean zero, so

    T_i = (Y_1[i] - Y_2[i])^T (C_1[i] + C_2[i])^{-1} (Y_1[i] - Y_2[i])

follows a chi-square distribution with ``d`` degrees of freedom. The edge
variances are estimated from the rank-``d`` fit ``P = U S V^T``: as
``P (1 - P)`` for binary graphs, or as the squared residual ``(A - P)^2`` for
any weights (the heteroskedasticity-robust estimate of Eicker and White).
Unlike a raw distance, ``T_i`` measures the change of a node in units of its
own noise, so hubs (large norm, small relative noise) and low-degree nodes
(noisy direction) are compared fairly, and the statistic does not change under
any invertible linear map of the embedding space, not only rotations.

The same construction for two separately embedded binary graphs is the test of
Zheng (2025, arXiv:2502.01947), which builds on Du & Tang (2023). Here the
embedding is joint, so no Procrustes step is needed, and weighted graphs are
covered by the robust variance.

Three points decide when the p-values can be trusted:

- Independent edges. Networks inferred from samples (co-expression, TOM,
  PANDA) have dependent entries, and the variances above are then far too
  small. Use :meth:`node2vec2rank.model.N2V2R.significance` for those.
- The dimension must not exceed the rank of the signal. A singular vector of
  the noise is fitted to the very edges it is multiplied with, so beyond the
  rank the statistic is inflated for every node. The default is the elbow of
  the singular values.
- Spill-over. A node also moves when its partners change. With ``align`` the
  first embedding is first mapped onto the second by a linear map fitted on
  the half of the nodes that changed least, which removes it when fewer than
  half of the nodes changed (see :mod:`node2vec2rank.alignment`).

``genomic_control`` divides all statistics by their median over the median of
the chi-square distribution when that ratio is above one (Devlin & Roeder,
1999). It assumes that most nodes do not change, as alignment does, and
corrects a variance estimate that is somewhat too small for all nodes. The
ratio is reported; a value far above 1 says the model does not fit, typically
because the edges are dependent or the dimension is too large.
"""

import numpy as np
from scipy import sparse
from scipy.stats import chi2

from node2vec2rank.significance import benjamini_hochberg

VARIANCES = ("auto", "bernoulli", "robust")
COLUMN_CHUNK = 1024


def is_binary(graph):
    values = graph.data if sparse.issparse(graph) else np.asarray(graph)
    return bool(np.all((values == 0) | (values == 1)))


def node_covariances(graph, left, singular_values, embedding, variance="robust"):
    """Estimated covariance of every node's embedding position in one graph.

    Args:
        graph: (n, n) adjacency matrix, dense or sparse.
        left: (n, d) left singular vectors of the unfolded matrix.
        singular_values: the d singular values.
        embedding: (n, d) positions of the nodes in this graph.
        variance: "bernoulli" for ``P (1 - P)`` (binary graphs) or "robust"
            for the squared residuals ``(A - P)^2`` (any weights).

    Returns:
        np.ndarray of shape (n, d, d).
    """
    if variance not in ("bernoulli", "robust"):
        raise ValueError(f'variance must be "bernoulli" or "robust", got {variance!r}')
    left = np.asarray(left, dtype=np.float64)
    num_nodes, dimension = left.shape
    root = np.sqrt(np.asarray(singular_values, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(root > 0, left / root, 0.0)
    scaled_left = left * root
    upper = np.triu_indices(dimension)
    weight_products = weights[:, upper[0]] * weights[:, upper[1]]

    diagonal = graph.diagonal() if sparse.issparse(graph) else np.diagonal(graph)
    # without self-loops a node's position is a sum over the other nodes only
    hollow = not np.any(diagonal)
    packed = np.empty((num_nodes, len(upper[0])))
    for start in range(0, num_nodes, COLUMN_CHUNK):
        columns = slice(start, min(start + COLUMN_CHUNK, num_nodes))
        fitted = scaled_left @ embedding[columns].T
        if variance == "bernoulli":
            fitted = np.clip(fitted, 0, 1)
            edge_variance = fitted * (1 - fitted)
        else:
            observed = graph[:, columns]
            observed = observed.toarray() if sparse.issparse(observed) else np.asarray(observed)
            edge_variance = (observed - fitted) ** 2
        if hollow:
            edge_variance[np.arange(columns.start, columns.stop), np.arange(columns.stop - columns.start)] = 0
        packed[columns] = edge_variance.T @ weight_products

    covariances = np.empty((num_nodes, dimension, dimension))
    covariances[:, upper[0], upper[1]] = packed
    covariances[:, upper[1], upper[0]] = packed
    return covariances


def wald_statistics(differences, covariances):
    """``x^T C^{-1} x`` for every row ``x`` of ``differences`` with its own
    covariance; NaN for nodes without any variance (e.g., isolated nodes)."""
    dimension = differences.shape[1]
    scale = np.trace(covariances, axis1=1, axis2=2) / dimension
    testable = np.isfinite(scale) & (scale > 0)
    statistics = np.full(len(differences), np.nan)
    if not testable.any():
        return statistics
    # a tiny ridge keeps the solve stable when a node's covariance is singular
    regularised = covariances[testable] + (1e-10 * scale[testable])[:, None, None] * np.eye(dimension)
    solved = np.linalg.solve(regularised, differences[testable][:, :, None])[:, :, 0]
    statistics[testable] = np.einsum("ij,ij->i", differences[testable], solved)
    return statistics


def _aligned(reference, target, reference_cov, target_cov, keep, max_iterations=50):
    """Maps the reference onto the target with a linear map fitted on the
    ``keep`` fraction of nodes with the smallest statistics."""
    differences, covariances = reference - target, reference_cov + target_cov
    statistics = wald_statistics(differences, covariances)
    dimension = reference.shape[1]
    num_testable = int(np.isfinite(statistics).sum())
    num_kept = max(dimension + 1, int(np.ceil(keep * num_testable)))
    if num_testable <= num_kept:
        return differences, covariances, statistics
    kept = None
    for _ in range(max_iterations):
        order = np.argsort(np.where(np.isfinite(statistics), statistics, np.inf))
        new_kept = np.sort(order[:num_kept])
        if kept is not None and np.array_equal(new_kept, kept):
            break
        kept = new_kept
        transform = np.linalg.lstsq(reference[kept], target[kept], rcond=None)[0]
        differences = reference @ transform - target
        covariances = transform.T @ reference_cov @ transform + target_cov
        statistics = wald_statistics(differences, covariances)
    return differences, covariances, statistics


def wald_test(reference, target, reference_cov, target_cov, align=True, genomic_control=True, keep=0.5):
    """Tests every node for a change of position between two embeddings.

    Args:
        reference, target: (n, d) positions of the nodes in the two graphs.
        reference_cov, target_cov: (n, d, d) covariances of those positions,
            from :func:`node_covariances`.
        align: whether to first map the reference onto the target with a
            robust linear fit, to remove spill-over.
        genomic_control: whether to divide the statistics by their inflation
            (median over the chi-square median) when it is above one.
        keep: fraction of nodes the alignment is fitted on.

    Returns:
        dict with arrays of shape (n,): ``statistic`` (chi-square with d
        degrees of freedom under no change, before genomic control),
        ``pvalue``, ``qvalue`` (Benjamini-Hochberg), ``distance`` (Euclidean
        distance after alignment, the size of the change), and the scalar
        ``inflation``.
    """
    reference, target = np.asarray(reference, dtype=np.float64), np.asarray(target, dtype=np.float64)
    dimension = reference.shape[1]
    if align:
        differences, _, statistics = _aligned(reference, target, reference_cov, target_cov, keep)
    else:
        differences = reference - target
        statistics = wald_statistics(differences, reference_cov + target_cov)

    finite = np.isfinite(statistics)
    inflation = np.median(statistics[finite]) / chi2.ppf(0.5, dimension) if finite.any() else np.nan
    scale = max(inflation, 1.0) if genomic_control and np.isfinite(inflation) else 1.0
    pvalues = chi2.sf(statistics / scale, dimension)
    return {"statistic": statistics, "pvalue": pvalues, "qvalue": benjamini_hochberg(pvalues),
            "distance": np.linalg.norm(differences, axis=1), "inflation": float(inflation)}
