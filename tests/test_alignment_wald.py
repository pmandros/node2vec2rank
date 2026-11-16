import numpy as np
import pytest
from scipy.stats import ortho_group

from node2vec2rank import N2V2R, robust_linear_alignment, uase
from node2vec2rank.model_utils import compute_pairwise_distances
from node2vec2rank.wald import node_covariances, wald_statistics, wald_test


def block_graphs(rng, num_nodes=600, frac_changed=0.0, weighted=False, heterogeneous=False):
    """Two graphs from the same 4-block model; changed nodes move to the next block."""
    blocks = rng.integers(0, 4, num_nodes)
    changed = rng.random(num_nodes) < frac_changed
    after = np.where(changed, (blocks + 1) % 4, blocks)
    theta = rng.uniform(0.5, 1.5, num_nodes) if heterogeneous else np.ones(num_nodes)
    probs = np.full((4, 4), 0.05) + np.eye(4) * 0.25
    graphs = []
    for z in (blocks, after):
        mean = np.outer(theta, theta) * probs[z][:, z]
        upper = np.triu(rng.poisson(3 * mean) / 3 if weighted else rng.random(mean.shape) < mean, 1)
        graphs.append((upper + upper.T).astype(float))
    return graphs, changed


def test_alignment_recovers_the_map_of_the_majority():
    rng = np.random.default_rng(0)
    source = rng.standard_normal((300, 5))
    transform = rng.standard_normal((5, 5))
    target = source @ transform
    changed = np.arange(300) < 120
    target[changed] += rng.standard_normal((120, 5)) * 3
    fitted = robust_linear_alignment(source, target)
    np.testing.assert_allclose(fitted, transform, atol=1e-8)
    # ordinary least squares is pulled by the changed rows
    assert not np.allclose(np.linalg.lstsq(source, target, rcond=None)[0], transform, atol=1e-2)


def test_alignment_is_identity_without_change():
    source = np.random.default_rng(1).standard_normal((100, 3))
    np.testing.assert_allclose(robust_linear_alignment(source, source), np.eye(3), atol=1e-10)
    with pytest.raises(ValueError):
        robust_linear_alignment(source, source[:, :2])


def test_population_distance_of_unchanged_nodes_is_zero_after_alignment():
    """Noise-free: unchanged nodes move between the two embeddings (spill-over),
    and alignment puts them back exactly."""
    rng = np.random.default_rng(2)
    blocks = rng.integers(0, 4, 400)
    changed = rng.random(400) < 0.3
    after = np.where(changed, (blocks + 1) % 4, blocks)
    theta = rng.uniform(0.5, 1.5, 400)
    probs = np.full((4, 4), 0.05) + np.eye(4) * 0.25
    means = [np.outer(theta, theta) * probs[z][:, z] for z in (blocks, after)]
    one, two = uase(means, 4, random_state=0)
    raw = compute_pairwise_distances(one, two, "euclidean")
    aligned = compute_pairwise_distances(one @ robust_linear_alignment(one, two), two, "euclidean")
    assert raw[~changed].max() > 1e-3
    assert aligned[~changed].max() < 1e-8
    assert aligned[changed].min() > 1e-3
    # in any basis: more than half of the nodes fit exactly, and whichever of
    # them the trimmed fit keeps must determine the whole map
    for seed in range(20):
        rotation = np.linalg.qr(np.random.default_rng(seed).standard_normal((4, 4)))[0]
        source, target = one @ rotation, two @ rotation
        moved = np.linalg.norm(source @ robust_linear_alignment(source, target) - target, axis=1)
        assert moved[~changed].max() < 1e-8
        assert moved[changed].min() > 1e-3


def test_align_option_ranks_changed_nodes_better_when_many_change():
    graphs, changed = block_graphs(np.random.default_rng(3), num_nodes=800, frac_changed=0.3, heterogeneous=True)

    def top_precision(**options):
        model = N2V2R(graphs, seed=0, embed_dimensions=[4, 6, 8], verbose=-1, **options)
        model.fit_transform_rank()
        borda = model.aggregate_transform()["1"]["borda_ranks"].to_numpy()
        return changed[np.argsort(-borda)[:changed.sum()]].mean()

    assert top_precision(align="robust") > 0.95
    assert top_precision(align="robust") > top_precision(align=None) + 0.1
    # the default ranks on the total and on the own move, and keeps the gain
    assert top_precision() > 0.9
    with pytest.raises(ValueError):
        N2V2R(graphs, align="procrustes", verbose=-1)


def test_radial_metric_is_the_radial_part_of_euclidean():
    rng = np.random.default_rng(4)
    one, two = rng.standard_normal((50, 6)), rng.standard_normal((50, 6))
    radial = compute_pairwise_distances(one, two, "radial")
    cosine = compute_pairwise_distances(one, two, "cosine")
    euclidean = compute_pairwise_distances(one, two, "euclidean")
    norms = np.linalg.norm(one, axis=1) * np.linalg.norm(two, axis=1)
    np.testing.assert_allclose(euclidean ** 2, radial ** 2 + 2 * norms * cosine)
    # a pure change of scale is all radial, a rotation is all angular
    np.testing.assert_allclose(compute_pairwise_distances(one, 2 * one, "cosine"), 0, atol=1e-12)
    rotation = ortho_group.rvs(6, random_state=0)
    np.testing.assert_allclose(compute_pairwise_distances(one, one @ rotation, "radial"), 0, atol=1e-12)


def test_embedding_is_a_weighted_sum_of_the_edges():
    graphs, _ = block_graphs(np.random.default_rng(5), num_nodes=200)
    embeddings, singular_values, left = uase(graphs, 4, random_state=0, return_left=True)
    weights = left / np.sqrt(singular_values)
    for graph, embedding in zip(graphs, embeddings):
        np.testing.assert_allclose(graph.T @ weights, embedding, atol=1e-8)


def test_wald_statistic_is_invariant_to_linear_maps():
    rng = np.random.default_rng(6)
    differences = rng.standard_normal((40, 3))
    roots = rng.standard_normal((40, 3, 3))
    covariances = roots @ roots.transpose(0, 2, 1) + np.eye(3)
    transform = rng.standard_normal((3, 3))
    np.testing.assert_allclose(
        wald_statistics(differences @ transform, transform.T @ covariances @ transform),
        wald_statistics(differences, covariances), rtol=1e-6)
    assert np.isnan(wald_statistics(np.ones((1, 3)), np.zeros((1, 3, 3)))).all()


@pytest.mark.parametrize("weighted", [False, True])
def test_wald_test_is_calibrated_without_change(weighted):
    rejected = []
    for seed in range(4):
        graphs, _ = block_graphs(np.random.default_rng(seed), weighted=weighted)
        model = N2V2R(graphs, seed=0, embed_dimensions=[4], verbose=-1)
        model.fit_transform_rank()
        result = model.wald_test(dimension=4)["1"]
        assert result.attrs["variance"] == ("robust" if weighted else "bernoulli")
        assert (result["qvalue"] < 0.1).sum() <= 2
        rejected.append((result["pvalue"] < 0.05).mean())
    assert 0.02 < np.mean(rejected) < 0.08


def test_wald_test_finds_changed_nodes_when_many_change():
    graphs, changed = block_graphs(np.random.default_rng(7), num_nodes=800, frac_changed=0.3, heterogeneous=True)
    model = N2V2R(graphs, seed=0, embed_dimensions=[4], verbose=-1)
    model.fit_transform_rank()
    called = model.wald_test(dimension=4)["1"]["qvalue"].to_numpy() < 0.1
    assert called[changed].mean() > 0.9
    assert called[~changed].sum() <= 0.15 * called.sum()
    # the degree-adjusted empirical null needs most nodes to keep their distance
    assert (model.significance()["1"]["qvalue"] < 0.1).sum() < called.sum()
    # without alignment the unchanged nodes are called as well: their partners changed
    unaligned = model.wald_test(dimension=4, align=False, genomic_control=False)["1"]["qvalue"].to_numpy() < 0.1
    assert unaligned[~changed].mean() > 0.3


def test_wald_test_flags_dependent_edges_and_checks_arguments():
    from node2vec2rank import coexpression_network, simulate_expression
    simulated = simulate_expression(num_genes=300, num_samples=60, frac_rewired=0, random_state=0)
    graphs = [coexpression_network(expression).to_numpy() for expression in simulated.expression]
    model = N2V2R(graphs, seed=0, embed_dimensions=[6], verbose=-1)
    model.fit_transform_rank()
    result = model.wald_test()["1"]
    assert result.attrs["inflation"] > 3
    with pytest.raises(ValueError):
        model.wald_test(dimension=7)
    with pytest.raises(ValueError):
        model.wald_test(variance="poisson")
    ulse_model = N2V2R(graphs, seed=0, embed_dimensions=[6], embedding_method="ulse", verbose=-1)
    ulse_model.fit_transform_rank()
    with pytest.raises(ValueError):
        ulse_model.wald_test()


def test_wald_handles_sparse_graphs_and_several_references():
    from scipy import sparse
    graphs, _ = block_graphs(np.random.default_rng(8), num_nodes=300)
    graphs = graphs + [graphs[0]]
    dense = N2V2R(graphs, seed=0, embed_dimensions=[4], comp_strategy="one_vs_rest", verbose=-1)
    dense.fit_transform_rank()
    sparse_model = N2V2R([sparse.csr_matrix(g) for g in graphs], seed=0, embed_dimensions=[4],
                         comp_strategy="one_vs_rest", verbose=-1)
    sparse_model.fit_transform_rank()
    for variance in ("bernoulli", "robust"):
        expected = dense.wald_test(dimension=4, variance=variance)
        result = sparse_model.wald_test(dimension=4, variance=variance)
        assert list(result) == ["1", "2", "3"]
        for key in result:
            np.testing.assert_allclose(result[key]["statistic"], expected[key]["statistic"], rtol=1e-4)


def test_node_covariances_match_the_spread_over_replicates():
    """The estimated covariance of a node's position matches its actual
    spread over independent graphs, with the projection held fixed."""
    rng = np.random.default_rng(9)
    blocks = np.repeat(np.arange(4), 75)
    probs = (np.full((4, 4), 0.05) + np.eye(4) * 0.25)[blocks][:, blocks]

    def sample():
        upper = np.triu(rng.random(probs.shape) < probs, 1)
        return (upper + upper.T).astype(float)

    embeddings, singular_values, left = uase([probs, probs], 4, random_state=0, return_left=True)
    weights = left / np.sqrt(singular_values)
    positions = np.stack([sample().T @ weights for _ in range(300)])
    empirical = np.einsum("rni,rnj->nij", positions - positions.mean(0), positions - positions.mean(0)) / 299
    graph = sample()
    for variance in ("bernoulli", "robust"):
        estimated = node_covariances(graph, left, singular_values, graph.T @ weights, variance)
        ratio = np.trace(estimated, axis1=1, axis2=2) / np.trace(empirical, axis1=1, axis2=2)
        assert 0.8 < np.median(ratio) < 1.2
