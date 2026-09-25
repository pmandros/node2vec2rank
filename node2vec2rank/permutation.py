"""Sample-label permutation test for users who have the data behind the networks.

When the two networks are built from samples (e.g., co-expression networks
from expression profiles), the most direct null is to shuffle the samples
between the two groups, rebuild both networks and re-run n2v2r. Every node then
gets its own null distribution of distances, which accounts for its degree,
its estimation noise and the network construction method, without assuming
that most nodes do not change.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from node2vec2rank.model import N2V2R
from node2vec2rank.model_utils import borda_aggregate
from node2vec2rank.significance import benjamini_hochberg
from node2vec2rank.simulate import coexpression_network


def _standardize(expression):
    centred = expression - expression.mean(axis=0)
    scale = expression.std(axis=0, ddof=1).replace(0, 1.0)
    return centred / scale


def _distances(network_a, network_b, nodes, params):
    model = N2V2R([network_a, network_b], nodes=nodes, **params)
    return model.fit_transform_rank()["1"]


def permutation_test(expression_a, expression_b, build_network=coexpression_network,
                     num_permutations=100, standardize_within_groups=True, random_state=None,
                     **n2v2r_params):
    """Tests every node for a shift between two groups of samples by shuffling
    the sample labels.

    For every embedding dimension and distance metric, a node's observed
    distance is ranked among its own distances over the permutations and
    turned into a normal score. The scores are averaged over the combinations
    and scaled by their spread for that node, and the p-value is the fraction
    of permutation scores of all nodes, computed the same way, that are at
    least as large.

    Args:
        expression_a, expression_b: DataFrames (samples x nodes) with the same columns.
        build_network: function from a samples x nodes DataFrame to a nodes x
            nodes network; defaults to the WGCNA-style |cor|^6 network.
        num_permutations: number of label permutations. The null is pooled
            over nodes, so p-values can go below 1 / (num_permutations + 1),
            but each node's score is still ranked among only
            num_permutations + 1 values; 500 or more is safer for real use.
        standardize_within_groups: z-score every node within each group before
            building any network, observed or permuted. Without it, mean or
            variance differences between the groups (e.g., differential
            expression) become spurious co-expression in the shuffled groups
            and invalidate the null. With a scale-dependent ``build_network``
            (e.g., covariance), this also removes variance differences from
            the observed networks.
        random_state: seed or numpy Generator for the permutations (the
            embeddings use ``seed`` from ``n2v2r_params`` if given).
        **n2v2r_params: parameters for :class:`node2vec2rank.model.N2V2R`,
            e.g. ``embed_dimensions`` or ``distance_metrics``.

    Returns:
        pd.DataFrame indexed by node with columns ``z`` (larger is more
        differential), ``pvalue``, ``qvalue`` (Benjamini-Hochberg) and
        ``borda_ranks`` (the usual n2v2r aggregated ranking of the observed data).
    """
    if list(expression_a.columns) != list(expression_b.columns):
        raise ValueError("Both expression tables must have the same columns (nodes) in the same order")
    if num_permutations < 3:
        raise ValueError("At least 3 permutations are needed")
    rng = np.random.default_rng(random_state)
    params = {"verbose": -1, **n2v2r_params}
    nodes = list(expression_a.columns)

    if standardize_within_groups:
        expression_a, expression_b = (_standardize(expression_a), _standardize(expression_b))
    observed = _distances(build_network(expression_a), build_network(expression_b), nodes, params)

    pooled = pd.concat([expression_a, expression_b], axis=0, ignore_index=True)
    size_a = len(expression_a)
    null = np.empty((num_permutations,) + observed.shape)
    for b in range(num_permutations):
        order = rng.permutation(len(pooled))
        group_a = pooled.iloc[order[:size_a]]
        group_b = pooled.iloc[order[size_a:]]
        null[b] = _distances(build_network(group_a), build_network(group_b), nodes, params).to_numpy()

    # rank-based normal scores: for every node and combination, the observed
    # distance and the permutation distances are ranked together, so all
    # B + 1 scores are exchangeable under the null and bounded (a single
    # unstable dimension, e.g. one cutting through near-equal singular values,
    # cannot dominate)
    stacked = np.concatenate([observed.to_numpy()[None], null], axis=0)
    missing = np.isnan(stacked)
    ranks = rankdata(np.where(missing, -np.inf, stacked), axis=0)
    scores = norm.ppf((ranks - 0.5) / stacked.shape[0])
    scores[missing] = np.nan
    with np.errstate(invalid="ignore"):
        combined_all = np.nanmean(scores, axis=2)  # (B + 1, nodes)
        # nodes differ in how correlated their combinations are, which changes
        # the spread of the average; standardise every node by its own spread
        spread = np.nanstd(combined_all, axis=0, ddof=1)
        combined_all = combined_all / np.where(spread > 0, spread, np.nan)
    combined = combined_all[0]
    combined_null = combined_all[1:].ravel()

    combined_null = np.sort(combined_null[np.isfinite(combined_null)])
    exceed = len(combined_null) - np.searchsorted(combined_null, combined, side="left")
    pvalues = np.where(np.isfinite(combined), (1 + exceed) / (1 + len(combined_null)), np.nan)

    return pd.DataFrame({"z": combined, "pvalue": pvalues, "qvalue": benjamini_hochberg(pvalues),
                         "borda_ranks": borda_aggregate(observed.to_numpy())},
                        index=observed.index)
