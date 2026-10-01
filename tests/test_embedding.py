import numpy as np
import pytest
from scipy import sparse

from node2vec2rank.embedding import uase


def test_uase_shape_and_order(two_sbm_graphs):
    graphs, _ = two_sbm_graphs
    embeddings, singular_values = uase(graphs, 6, random_state=0, return_singular_values=True)
    assert embeddings.shape == (2, 400, 6)
    assert np.all(np.diff(singular_values) <= 0)


def test_uase_matches_full_svd(two_sbm_graphs):
    graphs, _ = two_sbm_graphs
    d = 4
    embeddings = uase(graphs, d, random_state=0)

    _, s, vt = np.linalg.svd(np.hstack(graphs), full_matrices=False)
    expected = (vt[:d].T * np.sqrt(s[:d])).reshape(2, 400, d)
    # singular vectors are defined up to sign; the Gram matrix is not
    for t in range(2):
        np.testing.assert_allclose(embeddings[t] @ embeddings[t].T,
                                   expected[t] @ expected[t].T, atol=1e-8)


def test_uase_is_reproducible_and_sparse_equals_dense(two_sbm_graphs):
    graphs, _ = two_sbm_graphs
    dense = uase(graphs, 4, random_state=1)
    np.testing.assert_array_equal(dense, uase(graphs, 4, random_state=1))
    sparse_embeddings = uase([sparse.csr_matrix(g) for g in graphs], 4, random_state=1)
    for t in range(2):
        np.testing.assert_allclose(dense[t] @ dense[t].T,
                                   sparse_embeddings[t] @ sparse_embeddings[t].T, atol=1e-8)


def test_uase_rejects_bad_input():
    with pytest.raises(ValueError):
        uase([np.eye(5), np.eye(4)], 2)
    with pytest.raises(ValueError):
        uase([np.eye(5), np.eye(5)], 5)


def test_dense_fallback_matches_arpack():
    from scipy.sparse.linalg import svds

    from node2vec2rank.embedding import _dense_top_right_singular
    rng = np.random.default_rng(0)
    unfolded = rng.random((30, 60))
    _, s, vt = svds(unfolded, k=5, v0=rng.standard_normal(30))
    order = np.argsort(s)[::-1]
    dense_s, dense_vt = _dense_top_right_singular(unfolded, 5)
    np.testing.assert_allclose(dense_s, s[order], rtol=1e-8)
    np.testing.assert_allclose(np.abs(dense_vt), np.abs(vt[order]), atol=1e-8)


def test_uase_falls_back_when_arpack_does_not_converge(monkeypatch):
    from scipy.sparse.linalg import ArpackNoConvergence

    import node2vec2rank.embedding as embedding

    def failing_svds(*args, **kwargs):
        raise ArpackNoConvergence("no convergence", None, None)

    rng = np.random.default_rng(1)
    graphs = [rng.random((20, 20)) for _ in range(2)]
    graphs = [(g + g.T) / 2 for g in graphs]
    expected = embedding.uase(graphs, 3, random_state=0)
    monkeypatch.setattr(embedding, "svds", failing_svds)
    fallback = embedding.uase(graphs, 3, random_state=0)
    np.testing.assert_allclose(np.abs(fallback), np.abs(expected), atol=1e-8)
