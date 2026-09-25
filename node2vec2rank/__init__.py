"""node2vec2rank: graph differential analysis via multi-layer node embeddings and ranking."""

__version__ = "0.1.0"

from node2vec2rank.config import DEFAULT_CONFIG, resolve_config
from node2vec2rank.dataloader import DataLoader
from node2vec2rank.embedding import uase
from node2vec2rank.model import N2V2R

__all__ = ["N2V2R", "DataLoader", "uase", "resolve_config", "DEFAULT_CONFIG", "__version__"]
