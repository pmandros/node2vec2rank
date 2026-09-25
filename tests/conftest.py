import numpy as np
import pytest


def sample_sbm(block_probs, memberships, rng):
    """Samples a symmetric binary adjacency matrix from a stochastic block model."""
    probs = block_probs[memberships][:, memberships]
    upper = np.triu(rng.random(probs.shape) < probs, 1)
    return (upper + upper.T).astype(np.float64)


@pytest.fixture
def two_sbm_graphs():
    """Two 4-block SBM graphs where only the nodes of block 1 change their
    connectivity pattern."""
    rng = np.random.default_rng(0)
    num_nodes = 400
    memberships = np.repeat(np.arange(4), num_nodes // 4)
    before = np.array([
        [0.30, 0.05, 0.05, 0.05],
        [0.05, 0.30, 0.05, 0.05],
        [0.05, 0.05, 0.30, 0.05],
        [0.05, 0.05, 0.05, 0.30]])
    after = before.copy()
    # block 1 stops being a community and connects to block 2 instead
    after[1, :] = after[:, 1] = [0.05, 0.05, 0.30, 0.05]
    graphs = [sample_sbm(before, memberships, rng), sample_sbm(after, memberships, rng)]
    return graphs, memberships
