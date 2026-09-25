"""Unfolded adjacency spectral embedding (UASE).

UASE (Jones & Rubin-Delanchy, 2021; Gallagher, Jones & Rubin-Delanchy, 2021)
jointly embeds K graphs on the same node set by taking a truncated SVD of the
column-concatenated (unfolded) adjacency matrix ``A = [A_1 | A_2 | ... | A_K]``.
The right singular vectors, scaled by the square root of the singular values,
give one ``d``-dimensional embedding per node and per graph. These per-graph
embeddings are cross-sectionally and longitudinally stable, which is what makes
comparing a node's position across graphs meaningful.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds


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

    rng = np.random.default_rng(random_state)
    v0 = rng.standard_normal(min(unfolded.shape))
    _, singular_values, vt = svds(unfolded, k=d, v0=v0)

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
