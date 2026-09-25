import numpy as np
import pandas as pd
import pytest

from node2vec2rank.preprocessing_utils import match_networks, network_transform


def test_match_networks_keeps_first_graph_order_and_symmetry():
    first = pd.DataFrame(np.arange(9).reshape(3, 3), index=list("cab"), columns=list("cab"))
    second = pd.DataFrame(np.ones((3, 3)), index=list("abd"), columns=list("abd"))
    matched = match_networks([first, second])
    for graph in matched:
        assert graph.index.to_list() == ["a", "b"]
        assert graph.columns.to_list() == ["a", "b"]


def test_network_transform_order():
    network = np.array([[0.0, -3.0, 1.0], [-3.0, 0.0, 2.0], [1.0, 2.0, 0.0]])
    out = network_transform(network, absolute=True, threshold=1.5, binarize=True)
    np.testing.assert_array_equal(out, [[0, 1, 0], [1, 0, 1], [0, 1, 0]])


def test_network_transform_top_percent_keep():
    network = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    out = network_transform(network, top_percent_keep=34)
    assert set(np.unique(out)) == {0.0, 3.0}


def test_network_transform_projects_bipartite():
    bipartite = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    out = network_transform(bipartite, project_unipartite_on="columns")
    np.testing.assert_array_equal(out, bipartite.T @ bipartite)
    with pytest.raises(ValueError):
        network_transform(bipartite, project_unipartite_on=None)


def test_network_transform_does_not_modify_input():
    network = pd.DataFrame([[0.0, -1.0], [-1.0, 0.0]])
    network_transform(network, absolute=True, binarize=True)
    assert network.iloc[0, 1] == -1.0
