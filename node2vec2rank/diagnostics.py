"""Diagnostics for n2v2r rankings: agreement between parameter choices and
dependence on node degree."""

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _column_ranks(frame: pd.DataFrame) -> np.ndarray:
    values = frame.to_numpy(dtype=np.float64)
    # NaN distances are ranked last, as in the Borda aggregation
    values = np.where(np.isnan(values), -np.inf, values)
    return rankdata(values, axis=0)


def ranking_agreement(rankings: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between every pair of rankings (columns).

    High agreement across embedding dimensions and distance metrics means the
    aggregated ranking is stable to these choices; blocks of low agreement show
    which choices drive the result.
    """
    ranks = _column_ranks(rankings)
    return pd.DataFrame(np.corrcoef(ranks, rowvar=False),
                        index=rankings.columns, columns=rankings.columns)


def degree_bias(rankings, degree) -> pd.Series:
    """Spearman correlation of every ranking (column) with node degree.

    Values far from 0 mean the ranking partly re-ranks nodes by degree. This
    is expected when degree itself changes, but under little change it shows
    a bias of the distance metric: Euclidean distances tend to favour hubs,
    cosine distances low-degree nodes.
    """
    if isinstance(rankings, pd.Series):
        rankings = rankings.to_frame()
    ranks = _column_ranks(rankings)
    degree_ranks = rankdata(np.asarray(degree, dtype=np.float64))
    correlations = [np.corrcoef(column, degree_ranks)[0, 1] for column in ranks.T]
    return pd.Series(correlations, index=rankings.columns, name="spearman_with_degree")


def top_k_stability(rankings: pd.DataFrame, top_fraction=0.05) -> pd.Series:
    """Fraction of rankings (columns) in which every node is in the top.

    Args:
        rankings: DataFrame with one ranking per column (larger is more differential).
        top_fraction: size of the top as a fraction of all nodes.

    Returns:
        pd.Series in [0, 1] per node; 1 means the node is in the top of every ranking.
    """
    ranks = _column_ranks(rankings)
    num_nodes = ranks.shape[0]
    top_size = max(1, int(round(top_fraction * num_nodes)))
    in_top = ranks > num_nodes - top_size
    return pd.Series(in_top.mean(axis=1), index=rankings.index, name="top_stability")
