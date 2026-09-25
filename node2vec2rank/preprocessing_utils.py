import numpy as np
import pandas as pd


def bipartite_to_unipartite_projection(graph, project_unipartite_on='columns'):
    """Projects an m x n bipartite matrix W to a symmetric unipartite matrix.

    Args:
        graph: an m x n matrix, e.g., a TF x gene adjacency matrix
        project_unipartite_on: 'columns' computes the n x n matrix W^T W,
            'rows' computes the m x m matrix W W^T

    Returns:
        The symmetric projected matrix
    """
    [r, c] = graph.shape

    if r == c:
        raise ValueError('Graph is square, it is not a bipartite adjacency matrix')

    if project_unipartite_on is None:
        raise ValueError(
            "The graphs are not square (bipartite); set project_unipartite_on to "
            "'rows' or 'columns' to project them to unipartite graphs")

    if project_unipartite_on.casefold() == 'columns':
        return graph.T @ graph
    if project_unipartite_on.casefold() == 'rows':
        return graph @ graph.T
    raise ValueError('Unknown projection type, options are columns or rows')


def symmetrize_matrix(matrix):
    """Represents an m x n bipartite matrix as an (m+n) x (m+n) symmetric
    adjacency matrix by zero-padding."""
    [r, c] = np.shape(matrix)
    if r == c:
        return matrix

    matrix = np.asarray(matrix)
    sym_matrix = np.zeros((r + c, r + c), dtype=matrix.dtype)
    sym_matrix[:r, r:] = matrix
    sym_matrix[r:, :r] = matrix.T
    return sym_matrix


def network_transform(network, threshold=None, top_percent_keep=100, binarize=False, absolute=False, project_unipartite_on='columns'):
    """Applies preprocessing transformations to a network, in this order:

    1. absolute: take the absolute value of every edge weight
    2. threshold: set every edge weight below this value to 0
    3. project_unipartite_on: if the matrix is m x n (bipartite), project it to
       m x m ('rows') or n x n ('columns')
    4. top_percent_keep: keep only the top percent of the non-zero edge weights
    5. binarize: set every non-zero edge weight to 1

    Use with caution, the order matters.

    Returns:
        The transformed symmetric matrix as a float32 numpy array
    """
    [r, c] = np.shape(network)

    if isinstance(network, pd.DataFrame):
        network_copy = network.to_numpy(dtype=np.float64, copy=True)
    else:
        network_copy = np.array(network, dtype=np.float64, copy=True)

    if absolute:
        network_copy = np.abs(network_copy)

    if threshold is not None:
        network_copy[network_copy < threshold] = 0

    if r != c:
        network_copy = bipartite_to_unipartite_projection(
            network_copy, project_unipartite_on)

    if top_percent_keep < 100:
        non_zero = network_copy[network_copy != 0]
        if non_zero.size > 0:
            cut_off = np.percentile(non_zero, 100 - top_percent_keep)
            network_copy[network_copy < cut_off] = 0

    if binarize:
        network_copy[network_copy != 0] = 1

    return np.float32(network_copy)


def match_networks(graphs):
    """Restricts a list of adjacency DataFrames to their common row and column
    nodes.

    Nodes keep the order in which they appear in the first graph, so the result
    is deterministic. For square graphs with the same row and column labels,
    rows and columns end up in the same order.
    """
    common_rows = graphs[0].index
    common_cols = graphs[0].columns
    for graph in graphs[1:]:
        common_rows = common_rows[common_rows.isin(graph.index)]
        common_cols = common_cols[common_cols.isin(graph.columns)]

    if len(common_rows) == 0 or len(common_cols) == 0:
        raise ValueError("The graphs have no nodes in common")

    # keep rows and columns aligned when they hold the same nodes
    if len(common_rows) == len(common_cols) and set(common_rows) == set(common_cols):
        common_rows = common_cols

    return [graph.loc[common_rows, common_cols] for graph in graphs]
