import numpy as np
import pandas as pd
import pytest

from node2vec2rank import N2V2R, select_dimension, ulse
from node2vec2rank.diagnostics import degree_bias, ranking_agreement, top_k_stability
from node2vec2rank.significance import (benjamini_hochberg, covariate_adjusted_zscores,
                                        empirical_null_test)


def degree_corrected_sbm_pair(rng, num_nodes=600, frac_changed=0.1, change=True):
    """Two degree-corrected SBM graphs; the changed nodes move to the next block."""
    blocks = rng.integers(0, 4, num_nodes)
    blocks_after = blocks.copy()
    changed = np.zeros(num_nodes, bool)
    if change:
        moved = rng.choice(num_nodes, int(frac_changed * num_nodes), replace=False)
        blocks_after[moved] = (blocks[moved] + 1) % 4
        changed[moved] = True
    theta = rng.pareto(2.5, num_nodes) + 1
    theta /= theta.mean()
    block_probs = np.full((4, 4), 0.03) + np.eye(4) * 0.17
    graphs = []
    for z in (blocks, blocks_after):
        probs = np.clip(np.outer(theta, theta) * block_probs[z][:, z], 0, 1)
        upper = np.triu(rng.random(probs.shape) < probs, 1)
        graphs.append((upper + upper.T).astype(float))
    return graphs, changed


def test_benjamini_hochberg_matches_definition():
    p = np.array([0.01, 0.04, 0.03, 0.5, np.nan])
    q = benjamini_hochberg(p)
    np.testing.assert_allclose(q[:4], [0.04, 0.16 / 3, 0.16 / 3, 0.5])
    assert np.isnan(q[4])


def test_covariate_adjusted_zscores_remove_covariate_trend():
    rng = np.random.default_rng(0)
    covariate = rng.uniform(1, 100, 2000)
    statistic = covariate / 50 + rng.normal(scale=0.1 + covariate / 100, size=2000)
    z = covariate_adjusted_zscores(statistic, covariate)
    assert abs(np.corrcoef(z, covariate)[0, 1]) < 0.05
    assert 0.9 < np.std(z) < 1.1
    # the spread is adjusted too: low and high covariate nodes have similar spread
    assert 0.8 < np.std(z[covariate < 30]) / np.std(z[covariate > 70]) < 1.25


def test_empirical_null_is_calibrated_under_no_change():
    rng = np.random.default_rng(1)
    distances = np.exp(rng.normal(size=(5000, 3)))
    _, p, q = empirical_null_test(distances, rng.uniform(size=5000))
    assert 0.03 < np.mean(p < 0.05) < 0.07
    assert np.mean(q < 0.05) < 0.01


def test_changed_group_sharing_a_degree_keeps_its_signal():
    rng = np.random.default_rng(5)
    degree = np.repeat(np.arange(10), 100).astype(float)
    statistic = rng.normal(size=1000)
    statistic[degree == 3] += 3
    z = covariate_adjusted_zscores(statistic, degree)
    assert np.median(z[degree == 3]) > 2


def test_select_dimension_finds_elbow():
    assert select_dimension([100, 90, 80, 10, 9, 8.5, 8, 7.5, 7]) == 3
    assert select_dimension([100, 10, 9, 8], min_dimension=2) == 2


def test_ulse_shape_and_order(two_sbm_graphs):
    graphs, _ = two_sbm_graphs
    embeddings, singular_values = ulse(graphs, 5, random_state=0, return_singular_values=True)
    assert embeddings.shape == (2, 400, 5)
    assert np.all(np.diff(singular_values) <= 0)


def test_auto_dimension_selects_true_rank():
    rng = np.random.default_rng(2)
    graphs, _ = degree_corrected_sbm_pair(rng)
    model = N2V2R(graphs, embed_dimensions="auto", seed=0, verbose=-1)
    ranks = model.fit_transform_rank()["1"]
    assert model.selected_dimension == 4
    assert list(ranks.columns) == ["dim-4_distance-euclidean", "dim-4_distance-cosine"]


def test_significance_detects_planted_change_and_controls_fdr():
    rng = np.random.default_rng(3)
    graphs, changed = degree_corrected_sbm_pair(rng)
    model = N2V2R(graphs, seed=0, verbose=-1)
    model.fit_transform_rank()
    result = model.significance(dimensions="elbow")["1"]
    assert list(result.columns) == ["z", "pvalue", "qvalue", "degree"]
    called = result["qvalue"].to_numpy() < 0.1
    assert called[changed].mean() > 0.8
    false_discovery_proportion = (called & ~changed).sum() / max(called.sum(), 1)
    assert false_discovery_proportion < 0.1


def test_significance_default_combines_configured_dimensions():
    rng = np.random.default_rng(3)
    graphs, changed = degree_corrected_sbm_pair(rng)
    model = N2V2R(graphs, seed=0, verbose=-1)
    model.fit_transform_rank()
    result = model.significance()["1"]
    # combining noisy high dimensions costs power, but the ranking stays good
    z = result["z"].fillna(-np.inf).to_numpy()
    top = np.argsort(-z)[:changed.sum()]
    assert changed[top].mean() > 0.7
    called = result["qvalue"].to_numpy() < 0.1
    assert (called & ~changed).sum() <= max(1, 0.1 * called.sum())


def test_significance_under_no_change_is_not_degree_biased():
    rng = np.random.default_rng(4)
    graphs, _ = degree_corrected_sbm_pair(rng, change=False)
    model = N2V2R(graphs, seed=0, verbose=-1)
    model.fit_transform_rank()
    result = model.significance()["1"]
    assert (result["pvalue"] < 0.05).mean() < 0.08
    assert abs(degree_bias(result["z"], result["degree"]).iloc[0]) < 0.1


def test_significance_rejects_unfitted_model_and_large_dimension(two_sbm_graphs):
    graphs, _ = two_sbm_graphs
    model = N2V2R(graphs, embed_dimensions=[4], verbose=-1)
    with pytest.raises(ValueError):
        model.significance()
    model.fit_transform_rank()
    with pytest.raises(ValueError):
        model.significance(dimensions=[8])


def test_diagnostics():
    rankings = pd.DataFrame({"a": [3.0, 2.0, 1.0, 0.0], "b": [3.0, 2.0, 1.0, 0.0],
                             "c": [0.0, 1.0, 2.0, 3.0]}, index=list("wxyz"))
    agreement = ranking_agreement(rankings)
    assert agreement.loc["a", "b"] == pytest.approx(1.0)
    assert agreement.loc["a", "c"] == pytest.approx(-1.0)
    bias = degree_bias(rankings, [4, 3, 2, 1])
    assert bias["a"] == pytest.approx(1.0)
    stability = top_k_stability(rankings, top_fraction=0.25)
    assert stability.to_dict() == pytest.approx({"w": 2 / 3, "x": 0, "y": 0, "z": 1 / 3})


def test_plots_render(two_sbm_graphs):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from node2vec2rank import plotting

    graphs, _ = two_sbm_graphs
    model = N2V2R(graphs, embed_dimensions=[4, 6], seed=0, verbose=-1)
    ranks = model.fit_transform_rank()["1"]
    significance = model.significance()["1"]
    plotting.plot_scree(model.singular_values, model.selected_dimension)
    plotting.plot_ranking_agreement(ranking_agreement(ranks))
    plotting.plot_degree_bias(ranks.iloc[:, 0], model.degrees("1"))
    plotting.plot_significance(significance)


def test_empirical_null_leaves_nodes_without_distance_untested():
    rng = np.random.default_rng(3)
    degree = rng.uniform(1, 100, 500)
    distances = np.exp(rng.standard_normal((500, 4)))
    distances[:10] = 0.0  # e.g. isolated in both graphs: zero euclidean distance
    distances[:10, 1::2] = np.nan  # and undefined cosine distance
    degree[:10] = 0.0
    z, pvalues, qvalues = empirical_null_test(distances, degree)
    assert np.all(np.isnan(z[:10])) and np.all(np.isnan(pvalues[:10])) and np.all(np.isnan(qvalues[:10]))
    assert np.all(np.isfinite(z[10:]))
