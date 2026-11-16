import functools
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from node2vec2rank import (N2V2R, coexpression_network, detect_modules, module_changes,
                           module_permutation_test)
from node2vec2rank.modules import spherical_kmeans


def block_graphs(rng, block_after=None, num_nodes=600, num_blocks=4, p_in=0.2, p_out=0.03):
    """Two degree-corrected block model graphs with the same communities and,
    optionally, another block matrix in the second graph."""
    blocks = np.repeat(np.arange(num_blocks), num_nodes // num_blocks)
    theta = rng.uniform(0.5, 1.5, num_nodes)
    block_before = np.full((num_blocks, num_blocks), p_out)
    np.fill_diagonal(block_before, p_in)
    graphs = []
    for block in (block_before, block_before if block_after is None else block_after(block_before.copy())):
        probs = np.outer(theta, theta) * block[blocks][:, blocks]
        upper = np.triu(rng.random(probs.shape) < probs, 1).astype(float)
        graphs.append(upper + upper.T)
    return graphs, blocks


def weaken(block):
    block[0, 0] *= 0.6
    return block


def merge(block):
    block[0, 1] = block[1, 0] = block[0, 0]
    return block


def test_spherical_kmeans_groups_by_direction_not_by_norm():
    rng = np.random.default_rng(0)
    directions = np.eye(3)[rng.integers(0, 3, 300)]
    points = directions * rng.uniform(0.1, 10, (300, 1)) + 0.01 * rng.standard_normal((300, 3))
    labels, centres = spherical_kmeans(points, 3, random_state=0)
    assert centres.shape == (3, 3)
    np.testing.assert_allclose(np.linalg.norm(centres, axis=1), 1)
    # one cluster per direction, whatever the norm
    assert len(set(zip(labels, directions.argmax(axis=1)))) == 3


def test_detect_modules_recovers_communities_and_leaves_isolated_nodes_out():
    graphs, blocks = block_graphs(np.random.default_rng(1))
    for graph in graphs:
        graph[:5] = graph[:, :5] = 0
    nodes = detect_modules(graphs, random_state=0)
    assert list(nodes.columns) == ["module", "fit", "membership"]
    assert (nodes["module"][:5] == 0).all()
    assert nodes["module"][5:].nunique() == 4 and (nodes["module"][5:] > 0).all()
    # every community is one module
    assert len(set(zip(nodes["module"][5:], blocks[5:]))) == 4
    # modules are numbered by decreasing size
    sizes = nodes["module"][nodes["module"] > 0].value_counts().sort_index()
    assert list(sizes.index) == [1, 2, 3, 4] and sizes.is_monotonic_decreasing

    assert detect_modules(graphs, num_modules=2, random_state=0)["module"].max() == 2
    # modules below the minimum size are dissolved
    assert (detect_modules(graphs, min_size=1000, random_state=0)["module"] == 0).all()
    with pytest.raises(ValueError):
        detect_modules(graphs, num_modules=0)


def test_weakened_module_is_ranked_first_and_described():
    graphs, blocks = block_graphs(np.random.default_rng(2), weaken)
    nodes = detect_modules(graphs, random_state=0)
    changes = module_changes(graphs[0], graphs[1], nodes["module"], test=True)
    assert list(changes.columns) == ["size", "change", "move", "rotation", "strength_change", "within_before",
                                     "within_after", "partner", "partner_before", "partner_after", "statistic",
                                     "pvalue", "qvalue"]
    assert changes["move"].is_monotonic_decreasing
    top = changes.iloc[0]
    assert set(blocks[nodes["module"] == changes.index[0]]) == {0}
    assert top["change"] == "weaker"
    assert top["strength_change"] == pytest.approx(np.log2(0.6), abs=0.15)
    assert top["within_after"] / top["within_before"] == pytest.approx(0.6, abs=0.06)
    assert top["qvalue"] < 1e-6
    assert (changes["qvalue"].iloc[1:] > 0.05).all()
    assert changes["move"].iloc[0] > 5 * changes["move"].iloc[1]


def test_merging_modules_are_named_with_their_partner():
    graphs, blocks = block_graphs(np.random.default_rng(3), merge)
    labels = np.array(["a", "b", "c", "d"], dtype=object)[blocks]
    labels[:3] = None  # not in any module
    changes = module_changes(graphs[0], graphs[1], labels)
    assert set(changes.index[:2]) == {"a", "b"}
    assert changes.loc["a", "change"] == "merges with b" and changes.loc["b", "change"] == "merges with a"
    assert changes.loc["a", "partner"] == "b"
    assert changes.loc["a", "partner_after"] > 4 * changes.loc["a", "partner_before"]
    assert changes.loc["a", "size"] == (blocks == 0).sum() - 3
    assert changes["rotation"][["a", "b"]].min() > 20 and changes["rotation"][["c", "d"]].max() < 5
    # a merge changes the partners, not the strength within
    assert changes["strength_change"].abs().max() < 0.2
    assert "pvalue" not in changes
    with pytest.raises(ValueError):
        module_changes(graphs[0], graphs[1], labels[:-1])
    with pytest.raises(ValueError):
        module_changes(graphs[0], graphs[1], np.zeros(len(labels)))


@pytest.mark.parametrize("num_modules", [None, 8])
def test_module_test_keeps_its_level_without_change(num_modules):
    # also with twice as many modules as there are: choosing the modules on
    # the average graph does not bias the comparison of the two graphs
    pvalues = []
    for seed in range(12):
        graphs, _ = block_graphs(np.random.default_rng(100 + seed), num_nodes=400)
        nodes = detect_modules(graphs, num_modules=num_modules, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            changes = module_changes(graphs[0], graphs[1], nodes["module"], test=True)
        pvalues.extend(changes["pvalue"])
    assert np.mean(np.array(pvalues) < 0.05) < 0.12


def test_module_test_warns_for_networks_inferred_from_samples():
    rng = np.random.default_rng(4)
    modules = np.repeat(np.arange(4), 50)
    graphs = []
    for _ in range(2):
        expression = 0.7 * rng.standard_normal((60, 4))[:, modules] + rng.standard_normal((60, 200))
        graphs.append(coexpression_network(expression, power=6))
    with pytest.warns(UserWarning, match="dependent"):
        changes = module_changes(graphs[0], graphs[1], modules + 1, test=True)
    assert changes.attrs["inflation"] > 3
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        module_changes(graphs[0], graphs[1], modules + 1)


def test_model_module_changes_with_sparse_graphs_and_own_modules(two_sbm_graphs, tmp_path):
    graphs, memberships = two_sbm_graphs
    names = [f"n{i}" for i in range(len(memberships))]
    model = N2V2R([sparse.csr_matrix(graph) for graph in graphs] + [sparse.csr_matrix(graphs[0])], nodes=names,
                  comp_strategy="one_vs_rest", seed=0, save_dir=str(tmp_path), verbose=-1)
    own = pd.Series(memberships + 1, index=names).sample(frac=1, random_state=0)
    changes = model.module_changes(modules=own, test=True)
    assert list(changes) == ["1", "2", "3"]
    # block 1 (module 2) stops being a community in the second graph only
    assert changes["2"].index[0] == 2 and changes["2"].loc[2, "change"] == "weaker"
    assert changes["2"].loc[2, "qvalue"] < 1e-6
    assert (model.pairwise_modules["2"]["module"].to_numpy() == memberships + 1).all()
    dense = N2V2R(graphs, nodes=names, seed=0, verbose=-1).module_changes(modules=list(memberships + 1))["1"]
    assert dense.loc[2, "change"] == "weaker"

    detected = model.module_changes()
    assert list(model.pairwise_modules["1"].columns) == ["module", "fit", "membership"]
    assert list(model.pairwise_modules["1"].index) == names
    assert detected["2"]["size"].sum() <= len(names)
    import os
    assert {"1_modules.tsv", "1_module_nodes.tsv"} <= set(os.listdir(model.save_dir))
    with pytest.raises(ValueError):
        model.module_changes(modules=[1, 2, 3])


def factor_samples(rng, weaker=None, num_genes=300, num_modules=3, num_samples=80):
    """Expression of two groups from a factor model with equal modules; in the
    second group the genes of module 0 load ``weaker`` times as strongly."""
    labels = np.repeat(np.arange(num_modules), num_genes // num_modules)
    loadings = rng.uniform(0.5, 0.9, num_genes)
    groups = []
    for group in range(2):
        load = loadings * (weaker if weaker is not None and group == 1 else 1.0) ** (labels == 0)
        factors = rng.standard_normal((num_samples, num_modules))
        expression = factors[:, labels] * load + rng.standard_normal((num_samples, num_genes)) * np.sqrt(1 - load ** 2)
        groups.append(pd.DataFrame(expression, columns=[f"g{i}" for i in range(num_genes)]))
    return groups, labels


SIGNED = functools.partial(coexpression_network, power=6, signed=True)


def test_module_permutation_test_finds_the_weakened_module():
    (group_a, group_b), labels = factor_samples(np.random.default_rng(0), weaker=0.6)
    changes, nodes = module_permutation_test(group_a, group_b, build_network=SIGNED, num_permutations=39,
                                             random_state=0)
    assert list(nodes.index) == list(group_a.columns)
    assert set(nodes["module"]) == {1, 2, 3}
    weakened = nodes["module"][labels == 0].mode()[0]
    assert changes.index[0] == weakened
    top = changes.iloc[0]
    assert top["change"] == "weaker" and top["strength_change"] < -1
    assert top["pvalue"] == pytest.approx(1 / 40) and top["pvalue_strength"] == pytest.approx(1 / 40)
    assert (changes["pvalue"].iloc[1:] > 0.05).all()
    assert changes["z"].iloc[0] > 5

    # the same with modules given by name and two processes
    known = pd.Series(np.array(["a", "b", "c"])[labels], index=group_a.columns)[::-1]
    given, given_nodes = module_permutation_test(group_a, group_b, modules=known, build_network=SIGNED,
                                                 num_permutations=39, random_state=0, n_jobs=2)
    assert given.index[0] == "a" and given.loc["a", "pvalue"] == pytest.approx(1 / 40)
    assert list(given_nodes["module"]) == list(np.array(["a", "b", "c"])[labels])
    with pytest.raises(ValueError, match="one label per node"):
        module_permutation_test(group_a, group_b, modules=[1, 2], num_permutations=5)


def test_module_permutation_test_is_calibrated_where_the_chi_square_test_is_not():
    pvalues, chi_square = [], []
    for seed in range(6):
        (group_a, group_b), _ = factor_samples(np.random.default_rng(seed))
        changes, nodes = module_permutation_test(group_a, group_b, build_network=SIGNED, num_permutations=39,
                                                 random_state=seed)
        pvalues.extend(changes[["pvalue", "pvalue_rotation", "pvalue_strength"]].to_numpy().ravel())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chi_square.extend(module_changes(SIGNED(group_a).to_numpy(), SIGNED(group_b).to_numpy(),
                                             nodes["module"], test=True)["pvalue"])
    pvalues = np.array(pvalues)
    assert len(pvalues) == 54
    assert (pvalues <= 0.05).mean() < 0.15
    assert 0.35 < pvalues.mean() < 0.65
    # two networks alone call most of these unchanged modules
    assert (np.array(chi_square) < 0.05).mean() > 0.7
