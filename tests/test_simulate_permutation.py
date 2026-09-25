import numpy as np
import pandas as pd
import pytest

from node2vec2rank.permutation import permutation_test
from node2vec2rank.simulate import coexpression_network, simulate_expression


def test_simulate_expression_ground_truth():
    sim = simulate_expression(num_genes=300, num_samples=(40, 50), frac_rewired=0.1, random_state=0)
    assert sim.expression[0].shape == (40, 300)
    assert sim.expression[1].shape == (50, 300)
    assert sim.rewired.sum() == 30
    assert set(sim.rewiring_type[sim.rewired]) == {"switch", "loss", "gain"}
    changed_module = sim.modules.condition_1 != sim.modules.condition_2
    pd.testing.assert_series_equal(changed_module, sim.rewired, check_names=False)
    assert not (sim.differentially_expressed & sim.rewired).any()
    losses = sim.rewiring_type == "loss"
    assert (sim.modules.condition_2[losses] == -1).all()


def test_simulated_modules_are_coexpressed():
    sim = simulate_expression(num_genes=200, num_samples=200, frac_rewired=0.0,
                              frac_differentially_expressed=0.0, random_state=1)
    network = coexpression_network(sim.expression[0], power=1)
    modules = sim.modules.condition_1.to_numpy()
    same = (modules[:, None] == modules[None, :]) & (modules[:, None] >= 0)
    np.fill_diagonal(same, False)
    different = ~same
    np.fill_diagonal(different, False)
    assert network.to_numpy()[same].mean() > 5 * network.to_numpy()[different].mean()


def test_coexpression_network():
    expression = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 4, 6, 8], "c": [4.0, 3, 2, 1],
                               "d": [1.0, 1, 1, 1]})
    unsigned = coexpression_network(expression, power=2)
    assert unsigned.loc["a", "b"] == pytest.approx(1.0)
    assert unsigned.loc["a", "c"] == pytest.approx(1.0)
    assert unsigned.loc["a", "d"] == 0.0
    assert unsigned.loc["a", "a"] == 0.0
    signed = coexpression_network(expression, power=1, signed=True)
    assert signed.loc["a", "c"] == pytest.approx(0.0)


def test_permutation_test_finds_rewired_genes():
    sim = simulate_expression(num_genes=200, num_samples=80, frac_rewired=0.1, random_state=2)
    result = permutation_test(*sim.expression, num_permutations=20, random_state=0,
                              seed=0, embed_dimensions=[4, 8])
    assert list(result.columns) == ["z", "pvalue", "qvalue", "borda_ranks"]
    rewired = sim.rewired.to_numpy()
    assert np.nanmedian(result["z"][rewired]) > np.nanmedian(result["z"][~rewired]) + 1
    assert (result["qvalue"][rewired] < 0.1).mean() > 0.3
    assert result["pvalue"].between(0, 1).all()


def test_permutation_test_is_calibrated_under_shuffled_labels():
    sim = simulate_expression(num_genes=200, num_samples=80, random_state=3)
    pooled = pd.concat(sim.expression)
    order = np.random.default_rng(0).permutation(len(pooled))
    result = permutation_test(pooled.iloc[order[:80]], pooled.iloc[order[80:]], num_permutations=20,
                              random_state=1, seed=0, embed_dimensions=[4, 8])
    assert 0.0 < (result["pvalue"] < 0.05).mean() < 0.1


def test_permutation_test_rejects_mismatched_columns():
    a = pd.DataFrame(np.zeros((5, 3)), columns=list("abc"))
    with pytest.raises(ValueError):
        permutation_test(a, a[["b", "a", "c"]])
