"""Robust linear alignment of two embeddings, to remove spill-over.

Under a latent position model the mean of graph ``k`` is
``P_k = X_k L_k X_k^T``, with one latent position per node in the rows of
``X_k``. The UASE position of node ``i`` in graph ``k`` is then
``Y_k[i] = M_k^T X_k[i]``, where the ``d x d`` matrix ``M_k`` depends on *all*
nodes of the graph and on ``L_k``. When some nodes change, ``M_1`` and ``M_2``
differ, so every node moves between the two embeddings, also the nodes whose
own latent position is the same ("spill-over"); with Euclidean distance that
movement grows with the norm of the node, roughly its degree.

Spill-over is one linear map shared by all nodes: for an unchanged node,
``Y_2[i] = G^T Y_1[i]`` with ``G = M_1^{-1} M_2``. If fewer than half of the
nodes changed and the unchanged ones span the embedding space, ``G`` is
identified by a least trimmed squares fit, and after mapping the first
embedding with ``G`` the population distance is exactly zero for every
unchanged node. This is the same assumption, and the same idea, as
between-sample normalisation in differential expression: most nodes do not
change, and what the majority shares is not a difference.

A change shared by most nodes through a linear map, such as every community
getting denser or one large module getting stronger, is absorbed by ``G`` as
well. Alignment therefore ranks the nodes that changed relative to the rest,
not every node whose connections changed.
"""

import numpy as np

MAD_TO_SD = 1.4826


def _trimmed_fit(source, target, start, num_kept, max_iterations):
    transform, kept = start, None
    for _ in range(max_iterations):
        residuals = np.linalg.norm(source @ transform - target, axis=1)
        new_kept = np.sort(np.argpartition(residuals, num_kept - 1)[:num_kept])
        if kept is not None and np.array_equal(new_kept, kept):
            break
        kept = new_kept
        transform = np.linalg.lstsq(source[kept], target[kept], rcond=None)[0]
    residuals = np.linalg.norm(source @ transform - target, axis=1)
    return transform, residuals, np.sort(residuals ** 2)[:num_kept].sum()


def robust_linear_alignment(source, target, keep=0.5, max_iterations=50, reweight=False):
    """Fits a ``d x d`` matrix ``G`` with ``source @ G ~= target`` on the
    majority of the rows, by least trimmed squares.

    Starting from the ordinary least squares fit and from the identity, the
    fit alternates between solving for ``G`` on the ``keep`` fraction of rows
    with the smallest residuals and recomputing the residuals (concentration
    steps, as in FAST-LTS of Rousseeuw & Van Driessen, 2006), and the start
    with the smaller trimmed loss is kept. With ``reweight``, ``G`` is then
    fitted once more on every row that is not an outlier of that fit (the
    usual reweighting step of least trimmed squares). It is off by default:
    in co-expression networks the residuals of unchanged genes vary so much
    with their connectivity that changed genes are not outliers of the
    trimmed fit, the refit takes them back in, and the spill-over returns
    (benchmarks/alignment_wald.py).

    Args:
        source, target: arrays of shape (n, d), the same nodes in the same order.
        keep: fraction of rows the trimmed fit uses, in (0.5, 1]; 0.5 tolerates
            the most changed rows (just under half).
        max_iterations: maximum number of concentration steps per start.
        reweight: whether to refit on all rows that are not outliers of the trimmed fit.

    Returns:
        np.ndarray of shape (d, d).
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError(f"Embeddings must be 2-D with the same shape, got {source.shape} and {target.shape}")
    if not 0.5 <= keep <= 1:
        raise ValueError(f"keep must be between 0.5 and 1, got {keep}")
    num_rows, dimension = source.shape
    num_kept = min(num_rows, max(dimension + 1, int(np.ceil(keep * num_rows))))

    starts = (np.linalg.lstsq(source, target, rcond=None)[0], np.eye(dimension))
    transform, residuals, _ = min((_trimmed_fit(source, target, start, num_kept, max_iterations)
                                   for start in starts), key=lambda fit: fit[2])
    if not reweight:
        return transform

    positive = residuals > 0
    if positive.sum() <= dimension:
        return transform
    log_residuals = np.log(residuals[positive])
    centre = np.median(log_residuals)
    spread = MAD_TO_SD * np.median(np.abs(log_residuals - centre))
    inliers = residuals <= np.exp(centre + 2.5 * spread)
    if inliers.sum() <= dimension:
        return transform
    return np.linalg.lstsq(source[inliers], target[inliers], rcond=None)[0]
