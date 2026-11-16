"""Which modules changed between two graphs.

The rankings of :class:`node2vec2rank.N2V2R` are per node. When a whole
module (a community, a co-expression module) changes how it connects, every
one of its nodes moves a little and in the same way, and the question is
which module that is and what happened to it. This module answers it in two
steps.

1. Modules. The nodes are embedded with the *average* of the two graphs and
   clustered by direction (spherical k-means), which groups nodes with the
   same pattern of connections whatever their degree. The average is used for
   a reason: when nothing changed, the two graphs can be swapped without
   changing their average, so the modules carry no information about the sign
   of any difference between the graphs, and comparing the graphs within the
   modules afterwards is not biased by having chosen the modules on the same
   data. Two communities that merge, or one that splits, still differ in the
   average graph and are found as separate modules. The number of modules is
   the one with the best silhouette, searched up to the elbow of the singular
   values (after the first, which reflects the overall density) plus two.

2. Changes. With ``B_k[a, b]`` the mean edge weight between the nodes of
   modules ``a`` and ``b`` in graph ``k``, row ``a`` of ``B_k`` is the
   connection profile of module ``a``. Its change is reported as

   - ``move``: the size of the change relative to the size of the profile,
     ``|B_2[a] - B_1[a]| / |(B_1[a] + B_2[a]) / 2|`` (norms weighted by the
     module sizes);
   - ``rotation``: the angle between the two profiles, in degrees. It measures
     a change of partners (a merge, a split, new connections) and ignores a
     change of overall strength;
   - ``strength_change``: the log2 ratio of the mean edge weight within the
     module, after over before;
   - ``change``: the largest single change in words (weaker, stronger, merges
     with, splits from, connects more to, connects less to another module).

A p-value per module (:func:`module_changes` with ``test=True``) is available
for graphs with independent edges. For every pair of modules the differences
``A_2[i, j] - A_1[i, j]`` over the node pairs of the block are independent
with mean zero when the block did not change, so their mean divided by its
standard error is standard normal, and the sum of the squares over the blocks
of a module follows a chi-square distribution. It is not valid for networks
inferred from samples (co-expression, PANDA): all entries of such a network
come from the same samples, the entries of a block move together, and their
mean varies far more than independent entries would. Nothing in a pair of
networks says how much; that needs the samples, and
:func:`node2vec2rank.permutation.module_permutation_test` tests the same
changes by shuffling them between the groups. The reported ``inflation`` (the
median squared block statistic over its expectation) is far above 1 for such
networks.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import beta, chi2

from node2vec2rank.embedding import select_dimension, uase
from node2vec2rank.significance import benjamini_hochberg

# modules are searched in at most this many dimensions
MAX_DIMENSION = 24
# a block connects "as within a module" above this share of the within-module weight
COUPLED = 0.75
UNASSIGNED = 0


def _unit(points):
    norms = np.linalg.norm(points, axis=1)
    return points / np.where(norms > 0, norms, 1.0)[:, None], norms


def spherical_kmeans(points, num_clusters, random_state=None, restarts=5, max_iterations=100):
    """k-means on the directions of the rows (cosine similarity), with every
    row weighted by its squared norm, so that rows near the origin, whose
    direction is noise, do not shape the clusters.

    Returns:
        (labels, centres): the cluster of every row and the unit-norm centres.
    """
    rng = np.random.default_rng(random_state)
    unit, norms = _unit(np.asarray(points, dtype=np.float64))
    weights = norms ** 2
    if not weights.sum() > 0:
        raise ValueError("All points are at the origin")
    best = None
    for _ in range(restarts):
        # k-means++ start on the directions
        centres = [unit[rng.choice(len(unit), p=weights / weights.sum())]]
        for _ in range(1, num_clusters):
            distance = np.maximum(1 - np.max(unit @ np.array(centres).T, axis=1), 0)
            probability = weights * distance ** 2
            total = probability.sum()
            centres.append(unit[rng.choice(len(unit), p=probability / total if total > 0 else weights / weights.sum())])
        centres = np.array(centres)
        labels = None
        for _ in range(max_iterations):
            new_labels = (unit @ centres.T).argmax(axis=1)
            if labels is not None and np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for cluster in range(num_clusters):
                members = labels == cluster
                centre = (unit[members] * weights[members, None]).sum(axis=0)
                norm = np.linalg.norm(centre)
                if norm > 0:
                    centres[cluster] = centre / norm
        objective = weights @ (unit @ centres.T)[np.arange(len(unit)), labels]
        if best is None or objective > best[0]:
            best = (objective, labels.copy(), centres.copy())
    return best[1], best[2]


def _silhouette(unit, weights, labels, centres):
    """Weighted mean silhouette with distances to the cluster centres."""
    distance = 1 - unit @ centres.T
    rows = np.arange(len(unit))
    own = distance[rows, labels]
    distance[rows, labels] = np.inf
    other = distance.min(axis=1)
    return weights @ ((other - own) / np.maximum(np.maximum(other, own), 1e-12)) / weights.sum()


def _mean_graph(graphs):
    total = graphs[0]
    for graph in graphs[1:]:
        total = total + graph
    return total / len(graphs)


def detect_modules(graphs, num_modules=None, max_modules=20, min_size=10, min_fit=0.5, min_membership=0.01,
                   tolerance=0.01, random_state=None):
    """Groups the nodes into modules from the average of the graphs.

    Args:
        graphs: list of (n, n) graphs over the same nodes, dense or sparse.
        num_modules: the number of modules; by default the smallest number
            whose silhouette is within ``tolerance`` of the best one.
        max_modules: the largest number of modules considered.
        min_size: the nodes of a module with fewer nodes are left unassigned.
        min_fit: nodes whose direction has a cosine below this with the centre
            of their module are left unassigned (module 0).
        min_membership: so are nodes whose membership (see below) is below
            this fraction of that of the strongest members of their module
            (the 95th percentile): nodes with next to no edges, whose
            direction is noise.
        random_state: seed or numpy Generator.

    Returns:
        pd.DataFrame with one row per node and columns ``module`` (1, 2, ...
        by decreasing size; 0 for unassigned nodes), ``fit`` (the cosine
        between the node and the centre of its module) and ``membership``
        (the length of the node's position along that centre: how strongly
        the node belongs to the module, in the spirit of a module membership
        or kME in WGCNA).
    """
    rng = np.random.default_rng(random_state)
    num_nodes = graphs[0].shape[0]
    if num_modules is not None and not 1 <= num_modules < num_nodes:
        raise ValueError(f"num_modules must be between 1 and the number of nodes, got {num_modules}")
    most = max_modules if num_modules is None else num_modules
    dimension = min(max(most + 2, 3), MAX_DIMENSION if num_modules is None else most + 2, num_nodes - 1)
    embedding, singular_values = uase([_mean_graph(graphs)], dimension, random_state=rng, return_singular_values=True)
    embedding = embedding[0]
    if num_modules is None:
        # the first singular value reflects the overall density; the elbow of
        # the others estimates the number of modules
        rank = 1 + select_dimension(singular_values[1:]) if dimension > 3 else dimension
        dimension = min(dimension, rank + 2)
        candidates = range(2, min(dimension, max_modules) + 1)
    else:
        candidates = [num_modules]
    unit, norms = _unit(embedding[:, :dimension])
    weights = norms ** 2

    results = {}
    for count in candidates:
        labels, centres = spherical_kmeans(embedding[:, :dimension], count, rng)
        results[count] = (_silhouette(unit, weights, labels, centres) if len(results) or len(candidates) > 1 else 0.0,
                          labels, centres)
    best = max(score for score, _, _ in results.values())
    count = min(count for count, (score, _, _) in results.items() if score >= best - tolerance)
    _, labels, centres = results[count]

    fit = (unit @ centres.T)[np.arange(num_nodes), labels]
    membership = fit * norms
    assigned = (fit >= min_fit) & (norms > 0)
    for cluster in range(count):
        members = assigned & (labels == cluster)
        if members.any():
            assigned[members] = membership[members] >= min_membership * np.quantile(membership[members], 0.95)
    sizes = np.bincount(labels[assigned], minlength=count)
    assigned &= sizes[labels] >= min_size
    # number the modules by decreasing size, 0 is unassigned
    sizes = np.bincount(labels[assigned], minlength=count)
    order = np.argsort(-sizes, kind="stable")
    rank_of = np.empty(count, dtype=int)
    rank_of[order] = np.arange(1, count + 1)
    return pd.DataFrame({"module": np.where(assigned, rank_of[labels], UNASSIGNED), "fit": fit,
                         "membership": membership})


def _block_sums(matrix, onehot):
    """Sums of the entries of every block, without the diagonal of the matrix."""
    diagonal = np.asarray(matrix.diagonal()).ravel()
    product = matrix @ onehot
    product = product.toarray() if sparse.issparse(product) else np.asarray(product)
    return onehot.T @ product - np.diag(onehot.T @ diagonal)


def _square(matrix):
    return matrix.multiply(matrix) if sparse.issparse(matrix) else np.asarray(matrix) ** 2


def _describe(delta, before, after, sizes, names):
    """The largest single change of every module, in words."""
    kinds, partners = [], []
    for a in range(len(sizes)):
        contribution = sizes * delta[a] ** 2
        others = contribution.copy()
        others[a] = -np.inf
        partner = int(np.argmax(others)) if len(sizes) > 1 else a
        partners.append(partner)
        largest = int(np.argmax(contribution))
        if largest == a:
            kinds.append("stronger" if delta[a, a] > 0 else "weaker")
            continue

        def coupling(block):
            within = np.sqrt(abs(block[a, a] * block[largest, largest]))
            return block[a, largest] / within if within > 0 else 0.0
        if delta[a, largest] > 0:
            kinds.append(f"merges with {names[largest]}" if coupling(after) > COUPLED
                         else f"connects more to {names[largest]}")
        else:
            kinds.append(f"splits from {names[largest]}" if coupling(before) > COUPLED
                         else f"connects less to {names[largest]}")
    return kinds, partners


def module_changes(reference, target, modules, test=False):
    """Describes how every module's connections differ between two graphs.

    Args:
        reference, target: (n, n) graphs over the same nodes, dense or sparse.
        modules: one label per node. Nodes labelled 0, -1, None or NaN are
            not in any module.
        test: whether to add a chi-square test per module. Only valid for
            graphs with independent edges (see the module docstring).

    Returns:
        pd.DataFrame indexed by module, sorted by decreasing ``move``, with
        columns ``size``, ``change``, ``move``, ``rotation`` (degrees),
        ``strength_change`` (log2), ``within_before``, ``within_after``,
        ``partner`` (the other module whose connection to this one changed
        most), ``partner_before`` and ``partner_after`` (the mean edge weight
        between the two); with ``test`` also ``statistic``, ``pvalue`` and
        ``qvalue`` (Benjamini-Hochberg), and ``attrs["inflation"]``.
    """
    labels = pd.Series(list(modules) if not isinstance(modules, pd.Series) else modules.to_numpy(), dtype=object)
    if len(labels) != reference.shape[0]:
        raise ValueError("modules must give one label per node")
    in_module = labels.notna() & ~labels.isin([UNASSIGNED, -1])
    names = sorted(pd.unique(labels[in_module]), key=lambda name: (str(type(name)), name))
    num_modules = len(names)
    if num_modules == 0:
        raise ValueError("No node is in a module")
    codes = labels.map({name: code for code, name in enumerate(names)})
    onehot = np.zeros((len(labels), num_modules))
    rows = np.flatnonzero(in_module.to_numpy())
    onehot[rows, codes[in_module].to_numpy(dtype=int)] = 1.0
    sizes = onehot.sum(axis=0)
    counts = np.outer(sizes, sizes) - np.diag(sizes)
    with np.errstate(invalid="ignore", divide="ignore"):
        before = np.where(counts > 0, _block_sums(reference, onehot) / counts, 0.0)
        after = np.where(counts > 0, _block_sums(target, onehot) / counts, 0.0)
        delta, mean = after - before, (before + after) / 2

        size_of_profile = np.sqrt((sizes * mean ** 2).sum(axis=1))
        move = np.sqrt((sizes * delta ** 2).sum(axis=1)) / np.where(size_of_profile > 0, size_of_profile, np.nan)
        norms = np.sqrt((sizes * before ** 2).sum(axis=1) * (sizes * after ** 2).sum(axis=1))
        cosine = np.where(norms > 0, (sizes * before * after).sum(axis=1) / np.where(norms > 0, norms, 1.0), np.nan)
        rotation = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        within_before, within_after = np.diag(before), np.diag(after)
        strength = np.where((within_before > 0) & (within_after > 0),
                            np.log2(np.abs(within_after / np.where(within_before != 0, within_before, 1.0))), np.nan)
    kinds, partners = _describe(delta, before, after, sizes, names)
    index = np.arange(num_modules)
    result = pd.DataFrame({
        "size": sizes.astype(int), "change": kinds, "move": move, "rotation": rotation, "strength_change": strength,
        "within_before": within_before, "within_after": within_after,
        "partner": [names[partner] for partner in partners],
        "partner_before": before[index, partners], "partner_after": after[index, partners],
    }, index=pd.Index(names, name="module"))

    if test:
        difference = target - reference
        with np.errstate(invalid="ignore", divide="ignore"):
            block_mean = np.where(counts > 0, _block_sums(difference, onehot) / counts, 0.0)
            variance = np.where(counts > 0, _block_sums(_square(difference), onehot) / counts, 0.0) - block_mean ** 2
            # a block of a module with itself holds every pair of nodes twice
            pairs = np.where(np.eye(num_modules, dtype=bool), counts / 2, counts)
            testable = (variance > 0) & (pairs > 1)
            z = np.where(testable, block_mean / np.sqrt(np.where(testable, variance / np.maximum(pairs, 1), 1.0)), 0.0)
        statistic = (z ** 2).sum(axis=1)
        freedom = testable.sum(axis=1)
        pvalues = np.where(freedom > 0, chi2.sf(statistic, np.maximum(freedom, 1)), np.nan)
        upper = np.triu(testable)
        block_pvalues = np.sort(chi2.sf(z[upper] ** 2, 1))
        inflation = float(np.median(z[upper] ** 2) / chi2.ppf(0.5, 1)) if upper.any() else np.nan
        result["statistic"], result["pvalue"], result["qvalue"] = statistic, pvalues, benjamini_hochberg(pvalues)
        result.attrs["inflation"] = inflation
        # with few blocks the median statistic is itself noisy: warn only when
        # the median block p-value is far below what independent edges give
        middle = len(block_pvalues) // 2 + 1
        if inflation > 3 and beta.cdf(block_pvalues[middle - 1], middle, len(block_pvalues) - middle + 1) < 0.001:
            warnings.warn(
                f"The block statistics are {inflation:.1f} times larger than independent edges without change would "
                "give. Either most modules changed, or the edges are dependent (e.g. a network inferred from "
                "samples), in which case the p-values of the modules are not valid; with the samples, use "
                "node2vec2rank.permutation.module_permutation_test.", stacklevel=2)
    return result.sort_values("move", ascending=False, kind="stable")
