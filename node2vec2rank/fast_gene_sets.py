"""A fast gene-set test that needs only a few sample-label permutations.

:func:`node2vec2rank.permutation.gene_set_test` builds its null from hundreds
of permutations, each of which rebuilds and re-embeds both networks. This
module keeps the same competitive statistic but gets its null from a formula,
in the spirit of CAMERA (Wu & Smyth, Nucleic Acids Research 2012):

1. Every node gets the degree-adjusted z-score of :meth:`N2V2R.significance`,
   which needs no permutations.
2. A set's statistic is the mean z of its nodes minus the mean z of all nodes.
3. If the null z-scores of nodes i and j have standard deviations s_i, s_j
   and correlation rho_ij, the null variance of that statistic is a' S G S a,
   with a = 1/m on the set's m nodes minus 1/n everywhere, S = diag(s) and G
   the matrix of rho_ij. Correlated nodes (a co-expression module) make it
   much larger than the variance under independence that gene-permutation
   methods assume.
4. rho_ij is modelled as a smooth function of the absolute expression
   correlation of the two nodes, learnt from a few sample-label permutations
   (10 in the benchmarks) by pooling all node pairs, so each set does not need
   its own permutation distribution. The same permutations give s_i and the
   null mean of every set's statistic, which is not zero: the degree
   adjustment leaves genes of some modules with systematically higher or lower
   z-scores even when nothing changes.

The observed statistic minus its null mean, divided by its standard deviation
(inflated by 1 + 1/B for the estimated mean), gives a normal p-value per set,
which, unlike a permutation p-value, is not bounded below by 1 / (B + 1).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from node2vec2rank.model import N2V2R
from node2vec2rank.permutation import RANKINGS, _standardize_within, _stratified_orders, read_gmt
from node2vec2rank.significance import benjamini_hochberg
from node2vec2rank.simulate import coexpression_network


def _node_zscores(expression_a, expression_b, build_network, params):
    params = dict(params)
    ranking = params.pop("ranking", "n2v2r")
    model = N2V2R([build_network(expression_a), build_network(expression_b)],
                  nodes=list(expression_a.columns), **params)
    if ranking == "degree_difference":
        # normal scores of the absolute degree difference; unlike the n2v2r
        # z-scores they are not degree-adjusted, but the null mean and the
        # correlation model are learnt from the permutations either way
        dedi = model.degree_difference_ranking()["1"]["absDeDi"].to_numpy(dtype=np.float64)
        return norm.ppf((rankdata(dedi) - 0.5) / len(dedi))
    model.fit_transform_rank()
    return np.nan_to_num(model.significance()["1"]["z"].to_numpy(dtype=np.float64))


def _pair_correlations(scaled, first, second, chunk=50000):
    """Correlations of the columns first[k] and second[k] of column-standardised data."""
    out = np.empty(len(first))
    for start in range(0, len(first), chunk):
        stop = start + chunk
        out[start:stop] = np.einsum("ij,ij->j", scaled[:, first[start:stop]], scaled[:, second[start:stop]])
    return out / (scaled.shape[0] - 1)


def _correlation_model(null_z, scaled, num_pairs, bins, rng):
    """The correlation of two nodes' null z-scores as a function of their absolute
    expression correlation, from binned averages over random node pairs."""
    num_nodes = null_z.shape[1]
    first = rng.integers(0, num_nodes, num_pairs)
    second = rng.integers(0, num_nodes, num_pairs)
    keep = first != second
    first, second = first[keep], second[keep]
    spread = null_z.std(axis=0, ddof=1)
    standardized = (null_z - null_z.mean(axis=0)) / np.where(spread > 0, spread, 1.0)
    z_correlation = np.einsum("ij,ij->j", standardized[:, first], standardized[:, second]) / (len(null_z) - 1)
    expression_correlation = np.abs(_pair_correlations(scaled, first, second))
    # finer bins among the most correlated pairs, where the correlation of z rises steeply
    grid = np.r_[np.linspace(0, 0.9, 10), 1 - np.geomspace(0.1, 2e-4, bins - 10), 1.0]
    edges = np.unique(np.quantile(expression_correlation, grid))
    which = np.clip(np.searchsorted(edges, expression_correlation, side="right") - 1, 0, len(edges) - 2)
    centres, values = [], []
    for b in range(len(edges) - 1):
        members = which == b
        if members.any():
            centres.append(expression_correlation[members].mean())
            values.append(z_correlation[members].mean())
    return np.array(centres), np.array(values)


def _quadratic_forms(scaled, members, centres, values, spread, chunk=2000):
    """a' S G S a for every set, with G_ij = g(|cor_ij|) (g interpolated from
    the binned model), G_ii = 1 and S = diag(spread), in chunks of rows."""
    num_nodes = scaled.shape[1]
    row_sums = np.empty(num_nodes)  # (G S 1)_i
    for start in range(0, num_nodes, chunk):
        block = scaled[:, start:start + chunk].T @ scaled / (scaled.shape[0] - 1)
        g = np.interp(np.abs(block), centres, values)
        rows = np.arange(start, min(start + chunk, num_nodes))
        g[rows - start, rows] = 1.0
        row_sums[rows] = g @ spread
    total = spread @ row_sums
    forms = []
    for index in members:
        size = len(index)
        within = np.interp(np.abs(scaled[:, index].T @ scaled[:, index] / (scaled.shape[0] - 1)), centres, values)
        np.fill_diagonal(within, 1.0)
        weights = spread[index]
        forms.append(weights @ within @ weights / size ** 2
                     - 2 * (weights * row_sums[index]).sum() / (size * num_nodes) + total / num_nodes ** 2)
    return np.array(forms)


def fast_gene_set_test(expression_a, expression_b, gene_sets, build_network=coexpression_network,
                       num_permutations=10, min_size=5, max_size=500, standardize_within_groups=True,
                       strata=None, random_state=None, num_pairs=400000, bins=40, ranking="n2v2r",
                       **n2v2r_params):
    """Tests gene sets for differential connectivity with a few label permutations.

    A competitive test: is the mean degree-adjusted z-score of a set's nodes
    larger than that of all nodes? The null variance of the statistic accounts
    for the correlation between nodes, which is learnt from ``num_permutations``
    sample-label permutations (see the module docstring). It asks the same
    question as :meth:`N2V2R.significance` for sets: whether the set's nodes
    changed more than nodes of similar degree. The permutation test of
    :func:`node2vec2rank.permutation.gene_set_test` instead also calls sets
    whose nodes' neighbours changed.

    Args:
        expression_a, expression_b: DataFrames (samples x nodes) with the same columns.
        gene_sets: dict {set name: iterable of nodes} or the path of a .gmt file.
        build_network: function from samples x nodes to a nodes x nodes network.
        num_permutations: label permutations used to learn the correlation
            model; at least 3. More only sharpens the model.
        min_size, max_size: sets with fewer or more nodes in the data are skipped.
        standardize_within_groups: z-score every node within each group (and
            stratum) first, as in :func:`node2vec2rank.permutation.permutation_test`.
        strata: optional pair of label sequences (one per sample of each
            group), e.g. batch; labels are only shuffled within a stratum.
        random_state: seed or numpy Generator.
        num_pairs, bins: node pairs and bins used to fit the correlation model.
        ranking: "n2v2r" (degree-adjusted z-scores of :meth:`N2V2R.significance`)
            or "degree_difference" (normal scores of the absolute degree
            difference, as a baseline).
        **n2v2r_params: passed to :class:`N2V2R` (e.g., ``seed``).

    Returns:
        pd.DataFrame indexed by set, sorted by p-value, with columns ``size``,
        ``score`` (mean z in the set minus the mean over all nodes), ``z``
        (score minus its null mean, divided by its null standard deviation), ``pvalue`` (one-sided),
        ``qvalue`` (Benjamini-Hochberg) and ``leading_nodes``.
    """
    if list(expression_a.columns) != list(expression_b.columns):
        raise ValueError("Both expression tables must have the same columns (nodes) in the same order")
    if num_permutations < 3:
        raise ValueError("At least 3 permutations are needed")
    if isinstance(gene_sets, str):
        gene_sets = read_gmt(gene_sets)
    if ranking not in RANKINGS:
        raise ValueError(f"ranking must be one of {RANKINGS}, got {ranking!r}")
    rng = np.random.default_rng(random_state)
    params = {"verbose": -1, "ranking": ranking, **n2v2r_params}
    nodes = pd.Index(expression_a.columns)
    members = {}
    for name, genes in gene_sets.items():
        index = np.unique(nodes.get_indexer(pd.unique(pd.Series(list(genes), dtype=object))))
        index = index[index >= 0]
        if min_size <= len(index) <= max_size:
            members[name] = index
    if not members:
        raise ValueError("No gene set has between min_size and max_size nodes in the data")

    if strata is None:
        strata_a, strata_b = np.zeros(len(expression_a)), np.zeros(len(expression_b))
    else:
        strata_a, strata_b = (np.asarray(s) for s in strata)
        if len(strata_a) != len(expression_a) or len(strata_b) != len(expression_b):
            raise ValueError("strata must give one label per sample of each group")
    if standardize_within_groups:
        expression_a = _standardize_within(expression_a, strata_a)
        expression_b = _standardize_within(expression_b, strata_b)
    observed = _node_zscores(expression_a, expression_b, build_network, params)

    pooled = pd.concat([expression_a, expression_b], axis=0, ignore_index=True)
    size_a = len(expression_a)
    if strata is None:
        orders = [rng.permutation(len(pooled)) for _ in range(num_permutations)]
    else:
        orders = _stratified_orders(np.r_[strata_a, strata_b], size_a, num_permutations, rng)
    null_z = []
    for order in orders:
        null_z.append(_node_zscores(pooled.iloc[order[:size_a]], pooled.iloc[order[size_a:]],
                                    build_network, params))
    null_z = np.stack(null_z)

    values = pooled.to_numpy(dtype=np.float64)
    spread = values.std(axis=0, ddof=1)
    scaled = (values - values.mean(axis=0)) / np.where(spread > 0, spread, 1.0)
    centres, correlations = _correlation_model(null_z, scaled, num_pairs, bins, rng)
    variance = _quadratic_forms(scaled, members.values(), centres, correlations, null_z.std(axis=0, ddof=1))

    def statistic(z):
        return np.array([z[index].mean() - z.mean() for index in members.values()])

    score = statistic(observed)
    null_mean = np.mean([statistic(z) for z in null_z], axis=0)
    z = (score - null_mean) / np.sqrt(np.maximum(variance * (1 + 1 / num_permutations), np.finfo(float).tiny))
    pvalues = norm.sf(z)
    leading = []
    for index in members.values():
        order = index[np.argsort(-observed[index])]
        leading.append([nodes[i] for i in order if observed[i] > 0])
    return pd.DataFrame({"size": [len(index) for index in members.values()], "score": score, "z": z,
                         "pvalue": pvalues, "qvalue": benjamini_hochberg(pvalues), "leading_nodes": leading},
                        index=pd.Index(list(members), name="gene_set")).sort_values("pvalue")
