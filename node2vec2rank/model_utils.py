import pandas as pd
import numpy as np
import scipy.spatial.distance
from joblib import Parallel, delayed


def signed_transform_single(ranks: pd.Series, prior_signed_ranks: pd.Series):
    node_names_list = []
    ranks_list = []
    for index, rank in ranks.items():
        if index in prior_signed_ranks.index:
            node_names_list.append(index)
            value = prior_signed_ranks.loc[index]
            if value > 0:
                ranks_list.append(rank)
            else:
                ranks_list.append(-rank)

    return pd.Series(ranks_list, index=node_names_list)


def _get_ranking(ranking: list, index: list):
    num_candidates = len(index)

    return [num_candidates-ranking.index(node) for node in index]


def borda_aggregate_parallel(rankings: list):
    index = rankings[0]

    results = np.asarray(Parallel(
        n_jobs=-2)(delayed(_get_ranking)(ranking, index) for ranking in rankings))
    borda_ranks = np.sum(results, axis=0)
    to_return = pd.DataFrame(borda_ranks, index=index, columns=['borda_ranks'])

    return to_return


def compute_pairwise_distances(mat1, mat2, distance='cosine'):
    """Computes Pairwise Distances between two embedding matrices

    Args:
        mat1: first embedding matrix (n_nodes,d_dimensions)
        mat2: second embedding matrix (n_nodes,d_dimensions)
        distance: distance metric to be used. Defaults to 'cosine'.

    Raises:
        Exception: _description_

    Returns:
        returns Lists of nodes names and distance values sorted by similarity in decreasing order

    """

    if distance == "cosine":
        dists = [scipy.spatial.distance.cosine(row1, row2)
                 for row1, row2 in zip(mat1, mat2)]
    elif distance == "euclidean":
        dists = [scipy.spatial.distance.euclidean(row1, row2)
                 for row1, row2 in zip(mat1, mat2)]
    elif distance == "correlation":
        dists = [scipy.spatial.distance.correlation(row1, row2)
                 for row1, row2 in zip(mat1, mat2)]
    else:
        raise NotImplementedError("Unsupported metric")

    return dists