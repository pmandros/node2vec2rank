"""node2vec2rank: graph differential analysis via multi-layer node embeddings and ranking."""

__version__ = "0.2.0"

from node2vec2rank.config import DEFAULT_CONFIG, resolve_config
from node2vec2rank.dataloader import DataLoader
from node2vec2rank.embedding import select_dimension, uase, ulse
from node2vec2rank.model import N2V2R
from node2vec2rank.permutation import permutation_test
from node2vec2rank.simulate import coexpression_network, simulate_expression

__all__ = ["N2V2R", "DataLoader", "permutation_test", "simulate_expression", "coexpression_network", "uase", "ulse", "select_dimension", "resolve_config", "DEFAULT_CONFIG", "__version__"]
