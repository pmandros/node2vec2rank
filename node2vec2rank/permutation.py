"""Sample-label permutation tests for users who have the data behind the networks.

When the two networks are built from samples (e.g., co-expression networks
from expression profiles), the most direct null is to shuffle the samples
between the two groups, rebuild both networks and re-run n2v2r. Every node then
gets its own null distribution of distances, which accounts for its degree,
its estimation noise and the network construction method, without assuming
that most nodes do not change.

The same permutations also test gene sets (:func:`gene_set_test`). Because
whole samples are shuffled, the correlation between the genes of a set is kept
in the null. Gene-permutation methods (GSEA prerank, over-representation
tests) treat genes as independent, which does not hold for network rankings:
genes of a co-expression module move together, so co-expressed sets come out
"enriched" even when nothing differs between the conditions.

They test modules as well (:func:`module_permutation_test`): which module of
the network changed how it connects, with a p-value per module. Two networks
alone cannot give that p-value when they are inferred from samples (see
node2vec2rank/modules.py); the samples can.

The null hypothesis is that the two groups do not differ at all. The p-values
are exact under it, but this is not a per-node test when some nodes do change:
rewiring some genes moves every gene a little in the joint embedding, so genes
whose own connections are unchanged are called as well (their partners
changed, or the embedding shifted). Calls are evidence that the networks
differ, and the smallest p-values point at where; they are not a list of
changed nodes with a controlled false discovery rate.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm, rankdata

from node2vec2rank.model import N2V2R
from node2vec2rank.model_utils import borda_aggregate
from node2vec2rank.modules import detect_modules, module_changes
from node2vec2rank.significance import benjamini_hochberg
from node2vec2rank.simulate import coexpression_network


def _standardize(expression):
    centred = expression - expression.mean(axis=0)
    scale = expression.std(axis=0, ddof=1).replace(0, 1.0)
    return centred / scale


def _standardize_within(expression, strata):
    standardized = expression.astype(np.float64)
    for value in pd.unique(strata):
        rows = np.flatnonzero(strata == value)
        standardized.iloc[rows] = _standardize(expression.iloc[rows]).to_numpy()
    return standardized


RANKINGS = ("n2v2r", "degree_difference")


def _distances(network_a, network_b, nodes, params):
    params = dict(params)
    ranking = params.pop("ranking", "n2v2r")
    # the permutations test "the groups do not differ at all"; the total move
    # is the statistic for that, and alignment would be refitted per permutation
    params.setdefault("align", None)
    model = N2V2R([network_a, network_b], nodes=nodes, **params)
    if ranking == "degree_difference":
        return model.degree_difference_ranking()["1"][["absDeDi"]]
    return model.fit_transform_rank()["1"]


_WORKER = {}


def _init_worker(pooled, size_a, build_network, nodes, params):
    _WORKER.update(pooled=pooled, size_a=size_a, build_network=build_network, nodes=nodes, params=params)


def _permuted_distances(order):
    w = _WORKER
    group_a = w["pooled"].iloc[order[:w["size_a"]]]
    group_b = w["pooled"].iloc[order[w["size_a"]:]]
    return _distances(w["build_network"](group_a), w["build_network"](group_b),
                      w["nodes"], w["params"]).to_numpy()


def _stratified_orders(strata, size_a, num_permutations, rng):
    """Permutations of the pooled samples that keep, within every stratum, as
    many samples in the first group as in the observed data."""
    labels = np.r_[np.ones(size_a, dtype=bool), np.zeros(len(strata) - size_a, dtype=bool)]
    groups = [np.flatnonzero(strata == value) for value in pd.unique(strata)]
    orders = []
    for _ in range(num_permutations):
        in_a = np.empty(len(strata), dtype=bool)
        for members in groups:
            in_a[members] = labels[members][rng.permutation(len(members))]
        orders.append(np.r_[np.flatnonzero(in_a), np.flatnonzero(~in_a)])
    return orders


def _prepare(expression_a, expression_b, num_permutations, standardize_within_groups, strata, rng):
    """The two groups (standardised if asked), their pooled samples and the
    permutations of the pooled samples (first group first)."""
    if list(expression_a.columns) != list(expression_b.columns):
        raise ValueError("Both expression tables must have the same columns (nodes) in the same order")
    if num_permutations < 3:
        raise ValueError("At least 3 permutations are needed")
    size_a = len(expression_a)
    if strata is not None:
        strata_a, strata_b = (np.asarray(s) for s in strata)
        if len(strata_a) != len(expression_a) or len(strata_b) != len(expression_b):
            raise ValueError("strata must give one label per sample of each group")
    else:
        strata_a = np.zeros(len(expression_a))
        strata_b = np.zeros(len(expression_b))
    pooled_strata = np.r_[strata_a, strata_b]

    if standardize_within_groups:
        # z-score within every group (and stratum, e.g. batch), so that mean or
        # scale differences do not become co-expression in the shuffled groups
        expression_a = _standardize_within(expression_a, strata_a)
        expression_b = _standardize_within(expression_b, strata_b)
    pooled = pd.concat([expression_a, expression_b], axis=0, ignore_index=True)
    if strata is None:
        orders = [rng.permutation(len(pooled)) for _ in range(num_permutations)]
    else:
        orders = _stratified_orders(pooled_strata, size_a, num_permutations, rng)
    return expression_a, expression_b, pooled, orders


def _node_scores(expression_a, expression_b, build_network, num_permutations,
                 standardize_within_groups, strata, random_state, n_jobs, n2v2r_params):
    """Observed n2v2r distances and, for the observed data (row 0) and every
    permutation, a per-node normal score (larger is more differential)."""
    rng = np.random.default_rng(random_state)
    params = {"verbose": -1, **n2v2r_params}
    if params.get("ranking", "n2v2r") not in RANKINGS:
        raise ValueError(f"ranking must be one of {RANKINGS}, got {params['ranking']!r}")
    nodes = list(expression_a.columns)
    size_a = len(expression_a)

    expression_a, expression_b, pooled, orders = _prepare(expression_a, expression_b, num_permutations,
                                                          standardize_within_groups, strata, rng)
    observed = _distances(build_network(expression_a), build_network(expression_b), nodes, params)
    if n_jobs == 1:
        _init_worker(pooled, size_a, build_network, nodes, params)
        null = [_permuted_distances(order) for order in orders]
    else:
        # spawn rather than fork: forking a process that already runs threads
        # (e.g., after gseapy or a multithreaded BLAS call) can deadlock
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_worker,
                                 initargs=(pooled, size_a, build_network, nodes, params)) as executor:
            null = list(executor.map(_permuted_distances, orders))
    null = np.stack(null)

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
    return observed, combined_all


def _pooled_pvalues(observed, null):
    """Fraction of the pooled null values at least as large as each observed value."""
    null = np.sort(null[np.isfinite(null)])
    exceed = len(null) - np.searchsorted(null, observed, side="left")
    return np.where(np.isfinite(observed), (1 + exceed) / (1 + len(null)), np.nan)


def _node_table(observed, combined_all):
    combined = combined_all[0]
    pvalues = _pooled_pvalues(combined, combined_all[1:].ravel())
    return pd.DataFrame({"z": combined, "pvalue": pvalues, "qvalue": benjamini_hochberg(pvalues),
                         "borda_ranks": borda_aggregate(observed.to_numpy())},
                        index=observed.index)


def permutation_test(expression_a, expression_b, build_network=coexpression_network,
                     num_permutations=100, standardize_within_groups=True, strata=None,
                     random_state=None, n_jobs=1, ranking="n2v2r", **n2v2r_params):
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
            nodes network; defaults to the WGCNA-style |cor|^6 network. For
            single-cell data, see :func:`node2vec2rank.singlecell.metacell_network`.
        num_permutations: number of label permutations. The null is pooled
            over nodes, so p-values can go below 1 / (num_permutations + 1),
            but each node's score is still ranked among only
            num_permutations + 1 values; 500 or more is safer for real use.
        standardize_within_groups: z-score every node within each group (and
            stratum) before building any network, observed or permuted. Without
            it, mean or variance differences between the groups (e.g.,
            differential expression) become spurious co-expression in the
            shuffled groups and invalidate the null. With a scale-dependent
            ``build_network`` (e.g., covariance), this also removes variance
            differences from the observed networks.
        strata: optional pair of label sequences (one label per sample of each
            group), e.g. batch or donor sex. Labels are only shuffled within a
            stratum, keeping each group's composition, and standardisation is
            done per group and stratum. Use it for batches; for samples that are
            not independent (cells of the same donor), permute donors instead.
        random_state: seed or numpy Generator for the permutations (the
            embeddings use ``seed`` from ``n2v2r_params`` if given).
        n_jobs: number of processes for the permutations. With more than one,
            ``build_network`` must be picklable (a module-level function or a
            ``functools.partial`` of one, not a lambda or a function defined in
            a notebook), and scripts need an ``if __name__ == "__main__":``
            guard, since workers are started with "spawn"; limiting every process
            to one BLAS thread (e.g. ``OMP_NUM_THREADS=1``) avoids oversubscription.
        ranking: "n2v2r", or "degree_difference" to test the absolute degree
            difference (DeDi) instead, with the same permutations, e.g. as a baseline.
        **n2v2r_params: parameters for :class:`node2vec2rank.model.N2V2R`,
            e.g. ``embed_dimensions`` or ``distance_metrics``.

    Returns:
        pd.DataFrame indexed by node with columns ``z`` (larger is more
        differential), ``pvalue``, ``qvalue`` (Benjamini-Hochberg) and
        ``borda_ranks`` (the usual n2v2r aggregated ranking of the observed
        data; with ``ranking="degree_difference"``, the ranks of the absolute
        degree difference).
    """
    observed, combined_all = _node_scores(expression_a, expression_b, build_network, num_permutations,
                                          standardize_within_groups, strata, random_state, n_jobs,
                                          {"ranking": ranking, **n2v2r_params})
    return _node_table(observed, combined_all)


def read_gmt(path):
    """Reads a .gmt gene set library into a dict {set name: list of genes}."""
    gene_sets = {}
    with open(path) as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) > 2:
                gene_sets[fields[0]] = [g for g in fields[2:] if g]
    return gene_sets


def gene_set_test(expression_a, expression_b, gene_sets, build_network=coexpression_network,
                  num_permutations=100, min_size=5, max_size=500, standardize_within_groups=True,
                  strata=None, random_state=None, n_jobs=1, return_nodes=False, ranking="n2v2r",
                  **n2v2r_params):
    """Tests gene sets (e.g., pathways) for differential connectivity between
    two groups of samples by shuffling the sample labels.

    Every node gets the permutation score of :func:`permutation_test`. A set's
    statistic is the mean score of its nodes minus the mean score of all nodes
    (a competitive test: is the set more differential than the rest?). It is
    computed on the observed data and on every permutation. The p-value is the
    fraction of the set's own permutation statistics that are at least as
    large (counting the observed one), so it cannot go below
    1 / (num_permutations + 1): with a large library (e.g., all of Reactome),
    use 1,000 or more permutations. ``nes`` is the statistic standardised by
    the set's permutation mean and spread (like GSEA's NES), for ranking sets.

    Shuffling samples keeps the correlation between the genes of a set, so,
    unlike GSEA prerank or over-representation tests on the ranking, sets of
    co-expressed genes are not called just because they are co-expressed.

    Args:
        expression_a, expression_b, build_network, num_permutations,
        standardize_within_groups, strata, random_state, n_jobs, ranking,
        **n2v2r_params: as in :func:`permutation_test`.
        gene_sets: dict {set name: iterable of nodes} or the path of a .gmt file.
            Nodes that are not in the expression tables are ignored.
        min_size, max_size: sets with fewer or more nodes in the data are skipped.
        return_nodes: also return the node table of :func:`permutation_test`
            from the same permutations.

    Returns:
        pd.DataFrame indexed by set, sorted by p-value, with columns ``size``
        (nodes in the data), ``score`` (observed mean node score minus the mean
        over all nodes), ``nes`` (score standardised by its permutations),
        ``pvalue``, ``qvalue`` (Benjamini-Hochberg) and ``leading_nodes`` (the
        set's nodes with a positive score, most differential first). With
        ``return_nodes=True``, a tuple (sets, nodes).
    """
    if isinstance(gene_sets, str):
        gene_sets = read_gmt(gene_sets)
    nodes = pd.Index(expression_a.columns)
    members = {}
    for name, genes in gene_sets.items():
        index = np.unique(nodes.get_indexer(pd.unique(pd.Series(list(genes), dtype=object))))
        index = index[index >= 0]
        if min_size <= len(index) <= max_size:
            members[name] = index
    if not members:
        raise ValueError("No gene set has between min_size and max_size nodes in the data")

    observed, combined_all = _node_scores(expression_a, expression_b, build_network, num_permutations,
                                          standardize_within_groups, strata, random_state, n_jobs,
                                          {"ranking": ranking, **n2v2r_params})
    scores = np.nan_to_num(combined_all, nan=0.0)
    background = scores.mean(axis=1)
    statistic = np.stack([scores[:, index].mean(axis=1) - background for index in members.values()],
                         axis=1)  # (B + 1, sets)
    with np.errstate(invalid="ignore", divide="ignore"):
        centre = statistic[1:].mean(axis=0)
        spread = statistic[1:].std(axis=0, ddof=1)
        standardised = (statistic - centre) / np.where(spread > 0, spread, np.nan)
    nes = standardised[0]
    # every set against its own permutations: set statistics can be skewed
    # (a set inside one module moves with it), so pooling standardised
    # statistics over sets, as for nodes, is not calibrated
    exceed = (statistic[1:] >= statistic[0]).sum(axis=0)
    pvalues = (1 + exceed) / len(statistic)

    observed_scores = scores[0]
    leading = []
    for index in members.values():
        order = index[np.argsort(-observed_scores[index])]
        leading.append([nodes[i] for i in order if observed_scores[i] > 0])
    sets = pd.DataFrame({"size": [len(index) for index in members.values()],
                         "score": statistic[0], "nes": nes, "pvalue": pvalues,
                         "qvalue": benjamini_hochberg(pvalues), "leading_nodes": leading},
                        index=pd.Index(list(members), name="gene_set")).sort_values("pvalue")
    if return_nodes:
        return sets, _node_table(observed, combined_all)
    return sets


MODULE_STATISTICS = ("move", "rotation", "strength_change")


def _dense(network):
    return network if sparse.issparse(network) else np.asarray(network, dtype=np.float64)


def _module_statistics(network_a, network_b, labels, names):
    table = module_changes(_dense(network_a), _dense(network_b), labels).reindex(names)
    return table, np.column_stack([table["move"], table["rotation"], table["strength_change"].abs()])


def _init_module_worker(pooled, size_a, build_network, labels, names):
    _WORKER.update(pooled=pooled, size_a=size_a, build_network=build_network, labels=labels, names=names)


def _permuted_module_statistics(order):
    w = _WORKER
    group_a = w["pooled"].iloc[order[:w["size_a"]]]
    group_b = w["pooled"].iloc[order[w["size_a"]:]]
    return _module_statistics(w["build_network"](group_a), w["build_network"](group_b), w["labels"], w["names"])[1]


def module_permutation_test(expression_a, expression_b, modules=None, build_network=coexpression_network,
                            num_permutations=200, num_modules=None, max_modules=20,
                            standardize_within_groups=True, strata=None, random_state=None, n_jobs=1):
    """Tests which modules changed how they connect between two groups of
    samples, by shuffling the sample labels.

    The modules are found in the network of all samples together
    (:func:`node2vec2rank.modules.detect_modules`), which is the same network
    however the samples are labelled, so choosing the modules does not bias
    the test. The change of every module between the networks of the two
    groups (:func:`node2vec2rank.modules.module_changes`) is then compared
    with its changes between the networks of shuffled groups. Whole samples
    are shuffled, so the noise that the genes of a module share, which is
    what a test on two networks alone cannot know, is in the null.

    As for :func:`permutation_test`, the null hypothesis is that the groups do
    not differ at all, and a module can be called because the module it
    connects to changed.

    Args:
        expression_a, expression_b, build_network, standardize_within_groups,
        strata, random_state, n_jobs: as in :func:`permutation_test`.
        modules: one label per node (a sequence, or a Series indexed by node),
            e.g. known modules or pathways; nodes labelled 0, -1, None or NaN
            are in no module. They must not come from one of the two groups
            alone, or from a comparison of the groups: modules found in one
            group are tighter there than in the other by construction. By
            default, modules are detected in the network of all samples.
        num_permutations: number of label permutations; the smallest possible
            p-value is 1 / (num_permutations + 1).
        num_modules, max_modules: as in :func:`node2vec2rank.modules.detect_modules`.

    Returns:
        (changes, nodes). ``changes`` is the table of
        :func:`node2vec2rank.modules.module_changes` (the first group is the
        reference), sorted by p-value, with ``pvalue`` (the fraction of
        permutations in which the module's ``move`` is at least as large,
        counting the observed one), ``qvalue`` (Benjamini-Hochberg), ``z``
        (the ``move`` standardised by its permutations, for ranking modules
        with the same p-value), and ``pvalue_rotation`` and
        ``pvalue_strength``, the same test for the rotation of the module's
        connection profile and for the size of its change of strength, which
        say what kind of change it is. ``nodes`` has one row per node with its
        ``module`` (and ``fit`` and ``membership`` for detected modules).
    """
    rng = np.random.default_rng(random_state)
    node_names = pd.Index(expression_a.columns)
    size_a = len(expression_a)
    expression_a, expression_b, pooled, orders = _prepare(expression_a, expression_b, num_permutations,
                                                          standardize_within_groups, strata, rng)
    if modules is None:
        nodes = detect_modules([_dense(build_network(pooled))], num_modules=num_modules, max_modules=max_modules,
                               random_state=rng)
        nodes.index = node_names
    else:
        if isinstance(modules, pd.Series):
            modules = modules.reindex(node_names)
        elif len(modules) != len(node_names):
            raise ValueError("modules must give one label per node")
        nodes = pd.DataFrame({"module": pd.Series(list(modules), dtype=object).to_numpy()}, index=node_names)
    labels = nodes["module"].to_numpy()
    in_module = pd.notna(labels) & ~np.isin(labels, [0, -1])
    names = sorted(pd.unique(labels[in_module]), key=lambda name: (str(type(name)), name))

    changes, observed = _module_statistics(build_network(expression_a), build_network(expression_b), labels, names)
    if n_jobs == 1:
        _init_module_worker(pooled, size_a, build_network, labels, names)
        null = [_permuted_module_statistics(order) for order in orders]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_module_worker,
                                 initargs=(pooled, size_a, build_network, labels, names)) as executor:
            null = list(executor.map(_permuted_module_statistics, orders))
    null = np.stack(null)  # (B, modules, statistics)

    with np.errstate(invalid="ignore", divide="ignore"):
        # a statistic that is undefined in a permutation (e.g. the strength of
        # a module without any weight) does not count as exceeding
        exceed = (null >= observed[None]).sum(axis=0)
        pvalues = np.where(np.isfinite(observed), (1 + exceed) / (1 + len(null)), np.nan)
        spread = np.nanstd(null[:, :, 0], axis=0, ddof=1)
        z = (observed[:, 0] - np.nanmean(null[:, :, 0], axis=0)) / np.where(spread > 0, spread, np.nan)
    changes["z"], changes["pvalue"] = z, pvalues[:, 0]
    changes["qvalue"] = benjamini_hochberg(pvalues[:, 0])
    changes["pvalue_rotation"], changes["pvalue_strength"] = pvalues[:, 1], pvalues[:, 2]
    order = np.lexsort((-np.nan_to_num(z, nan=-np.inf), np.nan_to_num(pvalues[:, 0], nan=np.inf)))
    return changes.iloc[order], nodes
