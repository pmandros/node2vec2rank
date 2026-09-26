import numpy as np
import pandas as pd
import pytest

from node2vec2rank.permutation import gene_set_test, permutation_test
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


def test_permutation_test_is_calibrated_with_scale_dependent_networks():
    # the observed and permuted networks must be built from identically
    # standardised data, otherwise a builder that is not scale-invariant
    # (here covariance) compares networks on different scales
    def covariance_network(expression):
        network = np.abs(np.cov(expression.to_numpy(), rowvar=False))
        np.fill_diagonal(network, 0)
        return network

    sim = simulate_expression(num_genes=200, num_samples=80, frac_rewired=0.0,
                              frac_differentially_expressed=0.0, random_state=4)
    scale = np.random.default_rng(0).uniform(1, 5, 200)
    group_a, group_b = (expression * scale for expression in sim.expression)
    result = permutation_test(group_a, group_b, build_network=covariance_network,
                              num_permutations=20, random_state=1, seed=0, embed_dimensions=[4, 8])
    assert (result["pvalue"] < 0.05).mean() < 0.1


def _module_and_random_sets(sim, rng, size=20):
    genes = np.asarray(sim.expression[0].columns)
    modules = sim.modules.condition_1.to_numpy()
    sets = {}
    for k in np.unique(modules[modules >= 0]):
        members = np.flatnonzero(modules == k)
        sets[f"module{k}"] = genes[rng.choice(members, min(size, len(members)), replace=False)]
    sets.update({f"random{j}": genes[rng.choice(len(genes), size, replace=False)] for j in range(5)})
    rewired = np.flatnonzero(sim.rewired.to_numpy())
    sets["rewired"] = genes[rng.choice(rewired, min(size, len(rewired)), replace=False)]
    return sets


def test_gene_set_test_finds_rewired_set_and_returns_nodes():
    sim = simulate_expression(num_genes=200, num_samples=80, frac_rewired=0.1, random_state=2)
    sets = _module_and_random_sets(sim, np.random.default_rng(0))
    sets["too small"] = list(sim.expression[0].columns[:3])
    sets["unknown genes"] = [f"not a gene {i}" for i in range(10)]
    result, nodes = gene_set_test(*sim.expression, sets, num_permutations=20, random_state=0, seed=0,
                                  embed_dimensions=[4, 8], return_nodes=True)
    assert list(result.columns) == ["size", "score", "nes", "pvalue", "qvalue", "leading_nodes"]
    assert "too small" not in result.index and "unknown genes" not in result.index
    # every set is compared with its own permutations: the smallest p-value is 1 / 21
    assert result.loc["rewired", "pvalue"] == pytest.approx(1 / 21)
    assert result["nes"].idxmax() == "rewired"
    assert result["leading_nodes"]["rewired"][0] in set(sets["rewired"])
    assert result["pvalue"].between(0, 1).all()
    assert list(nodes.columns) == ["z", "pvalue", "qvalue", "borda_ranks"]


def test_gene_set_test_does_not_call_coexpressed_sets_under_shuffled_labels():
    # gene-permutation methods call module gene sets here; label permutations do not
    sim = simulate_expression(num_genes=200, num_samples=80, random_state=3)
    pooled = pd.concat(sim.expression)
    order = np.random.default_rng(0).permutation(len(pooled))
    sets = _module_and_random_sets(sim, np.random.default_rng(1))
    result = gene_set_test(pooled.iloc[order[:80]], pooled.iloc[order[80:]], sets, num_permutations=20,
                           random_state=1, seed=0, embed_dimensions=[4, 8])
    assert (result["qvalue"] < 0.1).sum() == 0


def test_permutation_test_with_strata_keeps_group_composition_and_degree_difference():
    sim = simulate_expression(num_genes=100, num_samples=60, frac_rewired=0.1, random_state=4)
    a, b = sim.expression
    strata = (np.repeat(["x", "y"], 30), np.repeat(["x", "y"], 30))
    result = permutation_test(a, b, num_permutations=10, strata=strata, random_state=0, seed=0,
                              embed_dimensions=[4], ranking="degree_difference")
    assert result["pvalue"].between(0, 1).all()
    with pytest.raises(ValueError):
        permutation_test(a, b, num_permutations=10, strata=(strata[0][:5], strata[1]))
    with pytest.raises(ValueError):
        permutation_test(a, b, num_permutations=10, ranking="nonsense")


def test_stratified_orders_keep_counts_per_stratum():
    from node2vec2rank.permutation import _stratified_orders
    strata = np.array(list("xxxxyyyyyy"))
    orders = _stratified_orders(strata, 5, 20, np.random.default_rng(0))
    for order in orders:
        assert sorted(order) == list(range(10))
        first = strata[order[:5]]
        assert (first == "x").sum() == (strata[:5] == "x").sum()


def test_read_gmt(tmp_path):
    from node2vec2rank.permutation import read_gmt
    path = tmp_path / "sets.gmt"
    path.write_text("A\turl\tg1\tg2\nB\tdesc\tg3\n")
    assert read_gmt(str(path)) == {"A": ["g1", "g2"], "B": ["g3"]}


def test_parallel_permutations_match_serial():
    sim = simulate_expression(num_genes=60, num_samples=40, frac_rewired=0.1, random_state=5)
    kwargs = dict(num_permutations=6, random_state=0, seed=0, embed_dimensions=[4])
    serial = permutation_test(*sim.expression, n_jobs=1, **kwargs)
    parallel = permutation_test(*sim.expression, n_jobs=2, **kwargs)
    pd.testing.assert_frame_equal(serial, parallel)
