import numpy as np
import pandas as pd
import scipy.sparse as sp

from node2vec2rank.singlecell import highly_variable_genes, metacell_network, metacells, normalize_log1p


def _counts(num_cells=120, num_genes=60, seed=0):
    rng = np.random.default_rng(seed)
    rates = rng.gamma(1.0, 1.0, size=num_genes)
    rates[:10] *= rng.gamma(0.3, 3.0, size=(num_cells, 10)).mean(axis=0)  # a few overdispersed genes
    counts = rng.poisson(rates * rng.gamma(2.0, 1.0, size=(num_cells, 1)))
    counts[:, :5] = rng.poisson(rng.gamma(0.2, 10.0, size=(num_cells, 5)))
    return counts


def test_normalize_log1p_sparse_matches_dense():
    counts = _counts()
    dense = normalize_log1p(counts)
    sparse = normalize_log1p(sp.csr_matrix(counts))
    assert sp.issparse(sparse)
    np.testing.assert_allclose(sparse.toarray(), dense)
    library = counts.sum(axis=1)
    np.testing.assert_allclose(np.expm1(dense).sum(axis=1), np.median(library))


def test_highly_variable_genes_prefers_overdispersed_genes():
    counts = _counts()
    logged = normalize_log1p(sp.csr_matrix(counts))
    top = highly_variable_genes(logged, sp.csr_matrix(counts), num_genes=10, num_bins=5)
    assert len(top) == 10 and len(set(top)) == 10
    assert len(set(top) & set(range(5))) >= 3


def test_metacells_and_metacell_network():
    rng = np.random.default_rng(1)
    expression = pd.DataFrame(rng.normal(size=(200, 30)), columns=[f"g{i}" for i in range(30)])
    aggregated = metacells(expression, k=10, max_shared=3, random_state=0)
    assert list(aggregated.columns) == list(expression.columns)
    assert 5 <= len(aggregated) <= 200
    network = metacell_network(expression, k=10, max_shared=3)
    assert network.shape == (30, 30)
    assert np.allclose(np.diag(network), 0) and np.allclose(network, network.T)
