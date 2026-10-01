import numpy as np
import pytest

from node2vec2rank.fast_gene_sets import fast_gene_set_test
from node2vec2rank.simulate import simulate_expression


def _module_and_rewired_sets(sim, rng, size=15):
    modules = sim.modules.condition_1.to_numpy()
    rewired = sim.rewired.to_numpy()
    genes = np.asarray(sim.expression[0].columns)
    sets = {f"module{k}": genes[rng.choice(np.flatnonzero(modules == k), size, replace=False)]
            for k in np.unique(modules[modules >= 0])}
    sets["rewired"] = genes[np.r_[rng.choice(np.flatnonzero(rewired), size // 2, replace=False),
                                  rng.choice(np.flatnonzero(~rewired), size - size // 2, replace=False)]]
    return sets


def test_fast_gene_set_test_calls_rewired_set_but_not_modules():
    rng = np.random.default_rng(0)
    sim = simulate_expression(num_genes=400, num_samples=(80, 80), random_state=0)
    sets = _module_and_rewired_sets(sim, rng)
    result = fast_gene_set_test(*sim.expression, sets, num_permutations=5, random_state=0, seed=0)
    assert list(result.columns) == ["size", "score", "z", "pvalue", "qvalue", "leading_nodes"]
    assert result.pvalue.is_monotonic_increasing
    assert result.loc["rewired", "pvalue"] < 0.01


def test_fast_gene_set_test_makes_no_call_under_a_shuffled_null():
    rng = np.random.default_rng(1)
    sim = simulate_expression(num_genes=400, num_samples=(160, 160), random_state=1)
    expression = sim.expression[0]
    order = rng.permutation(len(expression))
    sets = _module_and_rewired_sets(sim, rng)
    result = fast_gene_set_test(expression.iloc[order[:80]], expression.iloc[order[80:]], sets,
                                num_permutations=5, random_state=1, seed=1)
    assert result.qvalue.min() > 0.1


def test_fast_gene_set_test_rejects_bad_arguments():
    sim = simulate_expression(num_genes=100, num_samples=(20, 20), random_state=2)
    a, b = sim.expression
    with pytest.raises(ValueError, match="permutations"):
        fast_gene_set_test(a, b, {"s": a.columns[:10]}, num_permutations=2)
    with pytest.raises(ValueError, match="min_size"):
        fast_gene_set_test(a, b, {"s": a.columns[:2]})
    with pytest.raises(ValueError, match="columns"):
        fast_gene_set_test(a, b[b.columns[::-1]], {"s": a.columns[:10]})
