"""Helpers to build co-expression networks from single-cell data, hdWGCNA style.

Single-cell counts are sparse and noisy, so correlations between single cells
are weak. hdWGCNA (Morabito et al., 2023) first aggregates similar cells into
metacells (each the average of a cell and its nearest neighbours, with a cap on
how many cells two metacells may share) and builds the network on those. The
functions here do the same with numpy and scipy only, and keep the counts
sparse until a small gene subset is selected:

    counts = scipy.sparse.load_npz(...)            # cells x genes, raw UMIs
    logged = normalize_log1p(counts)               # still sparse
    keep = highly_variable_genes(logged, counts, num_genes=2000)
    expression = pd.DataFrame(logged[:, keep].toarray(), columns=gene_names[keep])

``metacell_network`` can then be passed as ``build_network`` to
:func:`node2vec2rank.permutation.permutation_test` or
:func:`node2vec2rank.permutation.gene_set_test` (use ``functools.partial`` to
change its parameters), so that every permuted group gets its own metacells.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

from node2vec2rank.simulate import coexpression_network


def normalize_log1p(counts, scale=None):
    """Library-size normalisation and log1p of a cells x genes count matrix.

    Args:
        counts: sparse or dense matrix (cells x genes) of raw counts.
        scale: counts per cell after normalisation; defaults to the median
            library size.

    Returns:
        A matrix of the same kind (a CSR matrix for sparse input).
    """
    library = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    if scale is None:
        scale = np.median(library)
    factors = scale / np.where(library > 0, library, 1.0)
    if sp.issparse(counts):
        normalized = sp.csr_matrix(counts.multiply(factors[:, None]), dtype=np.float64)
        normalized.data = np.log1p(normalized.data)
        return normalized
    return np.log1p(np.asarray(counts, dtype=np.float64) * factors[:, None])


def highly_variable_genes(logged, counts=None, num_genes=2000, min_detected=0.05, num_bins=20):
    """Indices of the most variable genes, Seurat-style: the log dispersion
    (variance / mean) z-scored within bins of mean expression.

    Args:
        logged: cells x genes matrix from :func:`normalize_log1p` (sparse or dense).
        counts: the raw counts, used for the detection filter; defaults to ``logged``.
        num_genes: number of genes to keep.
        min_detected: minimum fraction of cells in which a gene is detected
            (hdWGCNA's default is 5%).
        num_bins: number of bins of mean expression.

    Returns:
        np.ndarray of column indices, most variable first.
    """
    counts = logged if counts is None else counts
    detected = np.asarray((counts > 0).mean(axis=0)).ravel()
    mean = np.asarray(logged.mean(axis=0)).ravel()
    squared = logged.multiply(logged) if sp.issparse(logged) else np.asarray(logged) ** 2
    variance = np.asarray(squared.mean(axis=0)).ravel() - mean ** 2
    candidates = np.flatnonzero((detected >= min_detected) & (mean > 0))
    dispersion = np.log(np.maximum(variance[candidates], 1e-12) / mean[candidates])
    bins = pd.qcut(mean[candidates], num_bins, labels=False, duplicates="drop")
    z = pd.Series(dispersion).groupby(bins).transform(
        lambda x: (x - x.mean()) / x.std() if len(x) > 1 and x.std() > 0 else x * 0).to_numpy()
    return candidates[np.argsort(-z, kind="stable")[:num_genes]]


def metacells(expression, k=25, max_shared=10, num_pcs=30, random_state=0):
    """Aggregates cells into metacells: every metacell is the mean of a cell
    and its k - 1 nearest neighbours (in PCA space), and two metacells share
    at most ``max_shared`` cells (hdWGCNA's defaults are k=25, max_shared=10).

    Args:
        expression: cells x genes DataFrame or array (e.g., log-normalised).
        k: cells per metacell.
        max_shared: maximum number of cells two metacells may share.
        num_pcs: principal components used to find neighbours.
        random_state: seed for the order in which cells are tried as centres.

    Returns:
        DataFrame (metacells x genes) if ``expression`` is a DataFrame, else an array.
    """
    values = np.asarray(expression, dtype=np.float64)
    num_cells = len(values)
    if num_cells < k:
        raise ValueError(f"Need at least k={k} cells, got {num_cells}")
    centred = values - values.mean(axis=0)
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    num_pcs = min(num_pcs, len(s))
    coordinates = u[:, :num_pcs] * s[:num_pcs]
    _, neighbours = cKDTree(coordinates).query(coordinates, k)

    rng = np.random.default_rng(random_state)
    membership = np.zeros((0, num_cells), dtype=np.int16)
    chosen = []
    for cell in rng.permutation(num_cells):
        members = neighbours[cell]
        if len(chosen) and membership[:, members].sum(axis=1).max() > max_shared:
            continue
        row = np.zeros((1, num_cells), dtype=np.int16)
        row[0, members] = 1
        membership = np.vstack([membership, row])
        chosen.append(members)
    aggregated = np.stack([values[members].mean(axis=0) for members in chosen])
    if isinstance(expression, pd.DataFrame):
        return pd.DataFrame(aggregated, columns=expression.columns)
    return aggregated


def metacell_network(expression, k=25, max_shared=10, num_pcs=30, power=6, signed=True, random_state=0):
    """hdWGCNA-style network: metacells (see :func:`metacells`), then the
    WGCNA soft-thresholded correlation network of the metacells (see
    :func:`node2vec2rank.simulate.coexpression_network`)."""
    return coexpression_network(metacells(expression, k=k, max_shared=max_shared, num_pcs=num_pcs,
                                          random_state=random_state),
                                power=power, signed=signed)
