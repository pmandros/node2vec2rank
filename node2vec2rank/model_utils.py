import pandas as pd
import numpy as np
from scipy.stats import rankdata


def signed_transform_single(ranks: pd.Series, prior_signed_ranks: pd.Series):
    """Gives each rank the sign of the node's value in a prior signed ranking.

    Nodes missing from the prior are dropped; nodes with a non-positive prior
    value get a negative sign.
    """
    common = ranks.index[ranks.index.isin(prior_signed_ranks.index)]
    signs = np.where(prior_signed_ranks.loc[common].to_numpy() > 0, 1, -1)
    return pd.Series(ranks.loc[common].to_numpy() * signs, index=common)


def borda_aggregate(scores) -> np.ndarray:
    """Aggregates several rankings into one with the Borda count.

    Every column of ``scores`` ranks the nodes (rows) by decreasing value. In
    each column the top node gets ``n`` points, the next ``n - 1`` and so on
    down to 1, and the Borda score of a node is the sum of its points over all
    columns. Tied nodes share the average of their points and NaN values are
    ranked last, so the result does not depend on the order of the nodes.

    Args:
        scores: array-like of shape (n_nodes, n_rankings).

    Returns:
        np.ndarray of shape (n_nodes,) with the Borda scores (higher is more
        differential).
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[:, None]
    scores = np.where(np.isnan(scores), -np.inf, scores)
    return rankdata(scores, method="average", axis=0).sum(axis=1)


def borda_aggregate_parallel(rankings: list):
    """Aggregates rankings given as lists of node names, best first, with Borda.

    Kept for backwards compatibility; see :func:`borda_aggregate`.

    Returns:
        pd.DataFrame indexed like the first ranking with a ``borda_ranks`` column.
    """
    index = list(rankings[0])
    num_nodes = len(index)
    position = {node: i for i, node in enumerate(index)}
    points = np.zeros(num_nodes, dtype=np.int64)
    for ranking in rankings:
        ranked_positions = np.fromiter((position[node] for node in ranking),
                                       dtype=np.int64, count=num_nodes)
        points[ranked_positions] += np.arange(num_nodes, 0, -1)
    return pd.DataFrame(points, index=index, columns=['borda_ranks'])


def compute_pairwise_distances(mat1, mat2, distance='cosine'):
    """Computes the row-wise distance between two embedding matrices.

    Args:
        mat1: first embedding matrix (n_nodes, d_dimensions)
        mat2: second embedding matrix (n_nodes, d_dimensions)
        distance: 'cosine', 'euclidean', 'radial' or 'correlation'. Defaults
            to 'cosine'. The squared Euclidean distance splits exactly into a
            radial and an angular part,
            ``|x - y|^2 = (|x| - |y|)^2 + 2 |x| |y| (1 - cos(x, y))``.
            'radial' is the first, ``| |x| - |y| |``: how much the node's
            weight in the leading structure changed (it became more or less
            of a hub). 'cosine' is the second without its weight: how much
            the direction changed (different partners). Euclidean distance
            mixes the two and weighs the change of direction by the norms,
            so it favours hubs.

    Raises:
        NotImplementedError: for an unsupported distance

    Returns:
        np.ndarray of shape (n_nodes,) with the distance of every node between
        the two embeddings. Cosine and correlation distances are NaN for nodes
        with an all-zero (or constant, for correlation) embedding.
    """
    mat1 = np.asarray(mat1, dtype=np.float64)
    mat2 = np.asarray(mat2, dtype=np.float64)
    if mat1.shape != mat2.shape:
        raise ValueError(
            f"Embeddings must have the same shape, got {mat1.shape} and {mat2.shape}")

    distance = distance.casefold()
    if distance == "euclidean":
        return np.linalg.norm(mat1 - mat2, axis=1)
    if distance == "radial":
        return np.abs(np.linalg.norm(mat1, axis=1) - np.linalg.norm(mat2, axis=1))

    if distance == "correlation":
        mat1 = mat1 - mat1.mean(axis=1, keepdims=True)
        mat2 = mat2 - mat2.mean(axis=1, keepdims=True)
    elif distance != "cosine":
        raise NotImplementedError(
            f"Unsupported metric '{distance}'. Available: cosine, euclidean, radial, correlation")

    norms = np.linalg.norm(mat1, axis=1) * np.linalg.norm(mat2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        similarity = np.einsum("ij,ij->i", mat1, mat2) / norms
    similarity = np.where(norms > 0, similarity, np.nan)
    # clip rounding errors so the distance stays within [0, 2], as in scipy
    return 1.0 - np.clip(similarity, -1.0, 1.0)
