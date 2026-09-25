"""Configuration defaults, loading and validation."""

import copy
import json

COMPARISON_STRATEGIES = ("sequential", "one_vs_before", "one_vs_rest")
DISTANCE_METRICS = ("euclidean", "cosine", "correlation")
EMBEDDING_METHODS = ("uase", "ulse")
# largest dimension considered when embed_dimensions is "auto"
AUTO_MAX_DIMENSION = 50

DEFAULT_CONFIG = {
    # data_io
    "save_dir": None,
    "data_dir": ".",
    "graph_filenames": None,
    "separator": "\t",
    "is_edge_list": False,
    "transpose": False,
    # data_preprocessing
    "project_unipartite_on": None,
    "threshold": None,
    "top_percent_keep": 100,
    "binarize": False,
    "absolute": False,
    # fitting_ranking
    # a list of dimensions, or "auto" for the elbow of the singular values
    "embed_dimensions": [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
    "embedding_method": "uase",
    "distance_metrics": ["euclidean", "cosine"],
    "comp_strategy": "sequential",
    "seed": None,
    "verbose": 1,
}

# common misspellings kept working for existing config files
_ALIASES = {"seperator": "separator"}


def flatten_config(config: dict) -> dict:
    """Flattens a config grouped in sections (data_io, ...) into one dict."""
    flat = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def resolve_config(config=None, **overrides) -> dict:
    """Returns a complete, validated flat config.

    Args:
        config: a config dict (flat or grouped in sections), a path to a JSON
            config file, or None for the defaults.
        **overrides: parameters that take precedence over ``config``.
    """
    if config is None:
        config = {}
    elif isinstance(config, str):
        with open(config, "r", encoding="utf-8") as file:
            config = json.load(file)

    user_config = flatten_config(config)
    user_config.update(overrides)
    for alias, key in _ALIASES.items():
        if alias in user_config:
            user_config.setdefault(key, user_config.pop(alias))

    unknown = set(user_config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(
            f"Unknown config parameter(s): {sorted(unknown)}. "
            f"Valid parameters: {sorted(DEFAULT_CONFIG)}")

    resolved = copy.deepcopy(DEFAULT_CONFIG)
    resolved.update(user_config)
    _validate(resolved)
    return resolved


def _validate(config: dict):
    dims = config["embed_dimensions"]
    if isinstance(dims, str):
        if dims.casefold() != "auto":
            raise ValueError(f'embed_dimensions must be a list of integers or "auto", got {dims!r}')
        dims = config["embed_dimensions"] = "auto"
    elif isinstance(dims, int):
        dims = config["embed_dimensions"] = [dims]
    if dims != "auto" and (not dims or any(int(d) != d or d < 1 for d in dims)):
        raise ValueError(
            f"embed_dimensions must be a non-empty list of positive integers, got {dims}")

    metrics = config["distance_metrics"]
    if isinstance(metrics, str):
        metrics = config["distance_metrics"] = [metrics]
    metrics = config["distance_metrics"] = [m.casefold() for m in metrics]
    bad_metrics = [m for m in metrics if m not in DISTANCE_METRICS]
    if not metrics or bad_metrics:
        raise ValueError(
            f"distance_metrics must be chosen from {DISTANCE_METRICS}, got {metrics}")

    method = config["embedding_method"] = config["embedding_method"].casefold()
    if method not in EMBEDDING_METHODS:
        raise ValueError(f"embedding_method must be one of {EMBEDDING_METHODS}, got {method!r}")

    if config["comp_strategy"] not in COMPARISON_STRATEGIES:
        raise ValueError(
            f"comp_strategy must be one of {COMPARISON_STRATEGIES}, got {config['comp_strategy']!r}")

    projection = config["project_unipartite_on"]
    if projection is not None and projection.casefold() not in ("rows", "columns"):
        raise ValueError(
            f"project_unipartite_on must be 'rows', 'columns' or null, got {projection!r}")

    top = config["top_percent_keep"]
    if not 0 < top <= 100:
        raise ValueError(f"top_percent_keep must be in (0, 100], got {top}")
