import numpy as np
import pandas as pd
import pytest
import scipy.spatial.distance

from node2vec2rank.model_utils import (borda_aggregate, borda_aggregate_parallel,
                                       compute_pairwise_distances, signed_transform_single)


@pytest.mark.parametrize("metric", ["euclidean", "cosine", "correlation"])
def test_distances_match_scipy(metric):
    rng = np.random.default_rng(0)
    mat1, mat2 = rng.normal(size=(50, 8)), rng.normal(size=(50, 8))
    expected = [getattr(scipy.spatial.distance, metric)(a, b) for a, b in zip(mat1, mat2)]
    np.testing.assert_allclose(compute_pairwise_distances(mat1, mat2, metric), expected, atol=1e-12)


def test_cosine_of_zero_embedding_is_nan():
    distances = compute_pairwise_distances(np.array([[0.0, 0.0], [1.0, 0.0]]),
                                           np.array([[1.0, 1.0], [0.0, 1.0]]), "cosine")
    assert np.isnan(distances[0])
    assert distances[1] == pytest.approx(1.0)


def _reference_borda(scores, index):
    """The original list-based Borda implementation."""
    rankings = []
    for column in scores.T:
        series = pd.Series(column, index=index).sort_values(ascending=False, kind="stable")
        rankings.append(series.index.to_list())
    points = np.zeros(len(index), dtype=int)
    for ranking in rankings:
        points += np.array([len(index) - ranking.index(node) for node in index])
    return points


def test_borda_matches_reference_implementation():
    rng = np.random.default_rng(0)
    scores = rng.random((200, 7))
    scores[3, 2] = np.nan
    index = [f"n{i}" for i in range(200)]
    np.testing.assert_array_equal(borda_aggregate(scores), _reference_borda(scores, index))


def test_borda_shares_points_between_ties():
    scores = np.array([[1.0], [1.0], [0.0], [np.nan]])
    np.testing.assert_array_equal(borda_aggregate(scores), [3.5, 3.5, 2, 1])


def test_borda_does_not_depend_on_node_order():
    rng = np.random.default_rng(1)
    scores = rng.integers(0, 5, size=(50, 4)).astype(float)
    permutation = rng.permutation(50)
    np.testing.assert_array_equal(borda_aggregate(scores)[permutation],
                                  borda_aggregate(scores[permutation]))


def test_borda_orders_consistent_rankings():
    scores = np.array([[3.0, 30.0], [2.0, 20.0], [1.0, 10.0]])
    np.testing.assert_array_equal(borda_aggregate(scores), [6, 4, 2])


def test_borda_aggregate_parallel_backwards_compatible():
    result = borda_aggregate_parallel([["a", "b", "c"], ["b", "a", "c"]])
    assert result.loc[["a", "b", "c"], "borda_ranks"].tolist() == [5, 5, 2]


def test_signed_transform_single():
    ranks = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    prior = pd.Series([0.5, -1.0], index=["a", "b"])
    signed = signed_transform_single(ranks, prior)
    assert signed.to_dict() == {"a": 1.0, "b": -2.0}
