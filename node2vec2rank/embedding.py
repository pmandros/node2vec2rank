"""Unfolded adjacency spectral embedding (UASE).

UASE (Jones & Rubin-Delanchy, 2021; Gallagher, Jones & Rubin-Delanchy, 2021)
jointly embeds K graphs on the same node set by taking a truncated SVD of the
column-concatenated (unfolded) adjacency matrix ``A = [A_1 | A_2 | ... | A_K]``.
The right singular vectors, scaled by the square root of the singular values,
give one ``d``-dimensional embedding per node and per graph. These per-graph
embeddings are cross-sectionally and longitudinally stable, which is what makes
comparing a node's position across graphs meaningful.

The regularised unfolded Laplacian spectral embedding (ULSE) from the same line
of work is available as an alternative, together with an automatic choice of
the embedding dimension.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, svds
from scipy.stats import norm

EMBEDDING_METHODS = ("uase", "ulse")


def uase(graphs, d, random_state=None, return_singular_values=False):
    """Computes the unfolded adjacency spectral embedding of a list of graphs.

    Args:
        graphs: list of K (n x n) adjacency matrices (dense numpy arrays or
            scipy sparse matrices) over the same, identically ordered nodes.
        d: embedding dimension.
        random_state: seed or numpy Generator for the SVD starting vector,
            making the result reproducible.
        return_singular_values: whether to also return the singular values.

    Returns:
        np.ndarray of shape (K, n, d) with the per-graph node embeddings, with
        dimensions ordered by decreasing singular value. If
        ``return_singular_values`` is True, a tuple ``(embeddings, singular_values)``.
    """
    unfolded, num_graphs, n = _unfold(graphs, d)
    return _right_embedding(unfolded, num_graphs, n, d, random_state, return_singular_values)


def ulse(graphs, d, regularisation=None, random_state=None, return_singular_values=False):
    """Computes the regularised unfolded Laplacian spectral embedding.

    Like :func:`uase`, but embeds the degree-normalised unfolded matrix
    ``(D_row + tau I)^{-1/2} A (D_col + tau I)^{-1/2}``, where the degrees are
    computed on absolute edge weights. This down-weights hub nodes, which can
    otherwise dominate the leading dimensions of weighted networks.

    Args:
        graphs: list of K (n x n) adjacency matrices over the same nodes.
        d: embedding dimension.
        regularisation: the regulariser tau; defaults to the mean row degree
            of the unfolded matrix.
        random_state: seed or numpy Generator for the SVD starting vector.
        return_singular_values: whether to also return the singular values.

    Returns:
        Same as :func:`uase`.
    """
    unfolded, num_graphs, n = _unfold(graphs, d)
    magnitude = abs(unfolded)
    row_degrees = np.asarray(magnitude.sum(axis=1)).ravel()
    col_degrees = np.asarray(magnitude.sum(axis=0)).ravel()
    tau = row_degrees.mean() if regularisation is None else regularisation
    if tau <= 0 and (row_degrees.min() <= 0 or col_degrees.min() <= 0):
        raise ValueError("regularisation must be positive when some nodes have no edges")
    row_scale = 1.0 / np.sqrt(row_degrees + tau)
    col_scale = 1.0 / np.sqrt(col_degrees + tau)
    if sparse.issparse(unfolded):
        normalised = sparse.diags(row_scale) @ unfolded @ sparse.diags(col_scale)
    else:
        normalised = unfolded * row_scale[:, None] * col_scale[None, :]
    return _right_embedding(normalised, num_graphs, n, d, random_state, return_singular_values)


def embed(graphs, d, method="uase", random_state=None, return_singular_values=False):
    """Embeds graphs jointly with the given method ('uase' or 'ulse')."""
    method = method.casefold()
    if method == "uase":
        return uase(graphs, d, random_state=random_state,
                    return_singular_values=return_singular_values)
    if method == "ulse":
        return ulse(graphs, d, random_state=random_state,
                    return_singular_values=return_singular_values)
    raise ValueError(f"Unknown embedding method {method!r}, options are {EMBEDDING_METHODS}")


def select_dimension(singular_values, min_dimension=1):
    """Selects an embedding dimension at the elbow of the singular values.

    Uses the profile likelihood method of Zhu & Ghodsi (2006): the singular
    values are split into a leading and a trailing group, each modelled as
    Gaussian with its own mean and a common variance, and the split with the
    highest likelihood is the dimension.

    Args:
        singular_values: the singular values (any order).
        min_dimension: the smallest dimension that may be returned.

    Returns:
        int: the selected dimension.
    """
    values = np.sort(np.asarray(singular_values, dtype=np.float64))[::-1]
    num_values = len(values)
    if num_values < 3:
        return max(min_dimension, 1)

    best_dimension, best_likelihood = 1, -np.inf
    for q in range(1, num_values):
        head, tail = values[:q], values[q:]
        pooled_var = (np.sum((head - head.mean()) ** 2) + np.sum((tail - tail.mean()) ** 2)) \
            / (num_values - 2)
        sd = np.sqrt(max(pooled_var, np.finfo(float).tiny))
        likelihood = norm.logpdf(head, head.mean(), sd).sum() + norm.logpdf(tail, tail.mean(), sd).sum()
        if likelihood > best_likelihood:
            best_dimension, best_likelihood = q, likelihood
    return max(best_dimension, min_dimension)


def _unfold(graphs, d):
    if len(graphs) == 0:
        raise ValueError("At least one graph is required")

    n = graphs[0].shape[0]
    for i, graph in enumerate(graphs):
        if graph.shape != (n, n):
            raise ValueError(
                f"All graphs must be square with the same shape; graph {i} has shape "
                f"{graph.shape}, expected {(n, n)}")

    num_graphs = len(graphs)
    if not 1 <= d < min(n, n * num_graphs):
        raise ValueError(
            f"Embedding dimension must be between 1 and {n - 1}, got {d}")

    if all(sparse.issparse(graph) for graph in graphs):
        unfolded = sparse.hstack(graphs, format="csr").astype(np.float64)
    else:
        unfolded = np.hstack([graph.toarray() if sparse.issparse(graph) else np.asarray(graph)
                              for graph in graphs]).astype(np.float64)
    return unfolded, num_graphs, n


ARPACK_MAXITER = 300


def _dense_top_right_singular(unfolded, d):
    matrix = unfolded.toarray() if sparse.issparse(unfolded) else np.asarray(unfolded)
    eigenvalues, left = np.linalg.eigh(matrix @ matrix.T)
    top = np.argsort(eigenvalues)[::-1][:d]
    singular_values = np.sqrt(np.clip(eigenvalues[top], 0, None))
    left = left[:, top]
    with np.errstate(invalid="ignore", divide="ignore"):
        vt = (matrix.T @ left / np.where(singular_values > 0, singular_values, np.inf)).T
    return singular_values, vt


def _right_embedding(unfolded, num_graphs, n, d, random_state, return_singular_values):
    rng = np.random.default_rng(random_state)
    v0 = rng.standard_normal(min(unfolded.shape))
    try:
        _, singular_values, vt = svds(unfolded, k=d, v0=v0, maxiter=ARPACK_MAXITER)
    except ArpackNoConvergence:
        # near-degenerate spectra (e.g., a network built from a handful of
        # samples, whose rank is below d) can stall ARPACK; the exact SVD from
        # the smaller Gram matrix gives the same embedding up to sign
        singular_values, vt = _dense_top_right_singular(unfolded, d)

    # svds does not guarantee an order; sort by decreasing singular value so
    # that embeddings[..., :k] is always the top-k embedding
    order = np.argsort(singular_values)[::-1]
    singular_values = singular_values[order]
    vt = vt[order, :]

    right_embedding = vt.T * np.sqrt(singular_values)
    embeddings = right_embedding.reshape(num_graphs, n, d)

    if return_singular_values:
        return embeddings, singular_values
    return embeddings
