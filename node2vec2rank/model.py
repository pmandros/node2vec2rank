from datetime import datetime
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import sparse

from node2vec2rank.config import AUTO_MAX_DIMENSION, resolve_config
from node2vec2rank.embedding import embed, select_dimension
from node2vec2rank.model_utils import borda_aggregate, compute_pairwise_distances, signed_transform_single
from node2vec2rank.significance import empirical_null_test


class N2V2R:
    """node2vec2rank: ranks nodes by how much their representation changes
    between graphs in a joint (UASE) embedding space.

    Args:
        graphs: list of K square adjacency matrices (numpy arrays, scipy
            sparse matrices or DataFrames) over the same, identically ordered
            nodes, e.g., as returned by :class:`node2vec2rank.dataloader.DataLoader`.
        nodes: the node names, in the order of the rows/columns of the graphs.
            Defaults to the DataFrame columns if the graphs are DataFrames,
            otherwise to 0..n-1.
        config: a config dict (flat or grouped in sections) or a path to a JSON
            config file. Missing parameters take their default values.
        **overrides: config parameters that take precedence over ``config``,
            e.g. ``N2V2R(graphs, nodes, embed_dimensions=[8], seed=0)``.
    """

    def __init__(self, graphs: list, nodes: list = None, config=None, **overrides):
        self.config = resolve_config(config, **overrides)

        if len(graphs) < 2:
            raise ValueError("At least two graphs are needed for a comparison")

        if nodes is None:
            nodes = graphs[0].columns if isinstance(graphs[0], pd.DataFrame) \
                else np.arange(graphs[0].shape[0])
        self.node_names = list(nodes)

        self.graphs = [g if sparse.issparse(g) else np.asarray(g, dtype=np.float64) for g in graphs]
        num_nodes = len(self.node_names)
        for i, graph in enumerate(self.graphs):
            if graph.shape != (num_nodes, num_nodes):
                raise ValueError(
                    f"Graph {i} has shape {graph.shape} but there are {num_nodes} nodes; "
                    "graphs must be square and match the node list")

        self.num_graphs = len(self.graphs)
        self.auto_dimensions = self.config['embed_dimensions'] == "auto"
        if self.auto_dimensions:
            # resolved to the elbow of the singular values when fitting
            self.embed_dimensions = None
            self.max_embed_dim = min(AUTO_MAX_DIMENSION, num_nodes - 1)
        else:
            self.embed_dimensions = list(self.config['embed_dimensions'])
            self.max_embed_dim = max(self.embed_dimensions)
        if self.max_embed_dim >= num_nodes:
            raise ValueError(
                f"Largest embedding dimension ({self.max_embed_dim}) must be smaller "
                f"than the number of nodes ({num_nodes})")
        self.distance_metrics = self.config['distance_metrics']
        self.embedding_method = self.config['embedding_method']
        self.comp_strategy = self.config['comp_strategy']
        self.seed = self.config['seed']
        self.save_dir = None

        self.node_embeddings = None
        self.singular_values = None
        self.selected_dimension = None
        self.pairwise_significance = None
        self.pairwise_ranks = None
        self.pairwise_signed_ranks = None
        self.pairwise_aggregate_ranks = None
        self.pairwise_signed_aggregate_ranks = None
        self.prior_signed_ranks = None

        if self.config["save_dir"]:
            now = datetime.now().strftime(r"%m_%d_%Y_%H_%M_%S")
            self.save_dir = os.path.join(self.config["save_dir"], now)
            os.makedirs(self.save_dir, exist_ok=True)
            self.__log(self.save_dir)
            with open(os.path.join(self.save_dir, "config.json"), 'w', encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)

    # kept for backwards compatibility with the previous (misspelled) name
    @property
    def prior_singed_ranks(self):
        return self.prior_signed_ranks

    def __log(self, message, level=0):
        if self.config["verbose"] >= level:
            print(message)

    def comparisons(self):
        """Returns the comparisons performed under the configured strategy as
        a list of ``(key, reference_graph_indices, target_graph_index)``.

        Rankings, aggregated rankings, degree differences and signed rankings
        all use these keys.
        """
        graph_indices = np.arange(self.num_graphs)
        if self.comp_strategy == 'sequential':
            return [(str(i), [i - 1], i) for i in range(1, self.num_graphs)]
        if self.comp_strategy == 'one_vs_before':
            return [(str(i), list(range(i)), i) for i in range(1, self.num_graphs)]
        # one_vs_rest
        return [(str(i + 1), list(graph_indices[graph_indices != i]), i)
                for i in range(self.num_graphs)]

    def __fit(self):
        self.node_embeddings, self.singular_values = embed(
            self.graphs, self.max_embed_dim, method=self.embedding_method,
            random_state=self.seed, return_singular_values=True)
        # at least 2 dimensions so that angle-based distances are meaningful
        self.selected_dimension = select_dimension(
            self.singular_values, min_dimension=min(2, self.max_embed_dim))
        if self.auto_dimensions:
            self.embed_dimensions = [self.selected_dimension]
            self.__log(f"\tSelected embedding dimension {self.selected_dimension} "
                       "at the elbow of the singular values", level=1)

    def degrees(self, key=None):
        """Node degrees (sum of absolute edge weights).

        Args:
            key: a comparison key; if given, the mean degree over the graphs of
                that comparison, otherwise the mean over all graphs.

        Returns:
            pd.Series of degrees indexed by node.
        """
        if key is None:
            indices = range(self.num_graphs)
        else:
            comparisons = {k: reference + [target] for k, reference, target in self.comparisons()}
            if key not in comparisons:
                raise KeyError(f"Unknown comparison {key!r}, available: {list(comparisons)}")
            indices = comparisons[key]
        degrees = [np.asarray(abs(self.graphs[i]).sum(axis=0)).ravel() for i in indices]
        return pd.Series(np.mean(degrees, axis=0), index=self.node_names, name="degree")

    def __distances(self, dimensions, distance_metrics):
        pairwise_distances = {}
        for key, reference, target in self.comparisons():
            columns = {}
            # go over all provided choices for number of latent dimensions
            for dim in dimensions:
                embed_one = np.mean(self.node_embeddings[reference, :, :dim], axis=0)
                embed_two = self.node_embeddings[target, :, :dim]

                # go over all provided choices for distance metrics
                for distance_metric in distance_metrics:
                    # angles are meaningless in one dimension
                    if distance_metric in ('cosine', 'correlation') and dim == 1:
                        continue
                    columns[f"dim-{dim}_distance-{distance_metric}"] = compute_pairwise_distances(
                        embed_one, embed_two, distance_metric)
            pairwise_distances[key] = pd.DataFrame(columns, index=self.node_names)
        return pairwise_distances

    def fit_transform_rank(self):
        """
        Computes the differential ranks of nodes for the graphs, for every
        combination of embedding dimension and distance metric.

        Returns:
            dict: one DataFrame per comparison (keyed as in :meth:`comparisons`)
            with the distance of every node (rows) for every parameter
            combination (columns). Larger distances mean more differential.
        """
        self.__log(
            f"\nRunning n2v2r with dimensions {self.config['embed_dimensions']} and distance metrics {self.distance_metrics} ...")
        tic_n2v2r = time.time()

        tic_uase = time.time()
        self.__fit()
        self.__log(f"\tMulti-layer embedding in {round(time.time() - tic_uase, 2)} seconds", level=1)

        self.pairwise_ranks = self.__distances(self.embed_dimensions, self.distance_metrics)

        num_rankings = sum(len(ranks.columns) for ranks in self.pairwise_ranks.values())
        self.__log(
            f"n2v2r computed {num_rankings} rankings for {len(self.pairwise_ranks)} comparison(s) "
            f"in {round(time.time() - tic_n2v2r, 2)} seconds")

        if self.save_dir:
            for key, rank in self.pairwise_ranks.items():
                rank.to_csv(os.path.join(
                    self.save_dir, key + ".tsv"), sep='\t', index=True)

        return self.pairwise_ranks

    def aggregate_transform(self, method='Borda'):
        """
        Aggregates, per comparison, the rankings of all parameter combinations
        into one.

        Args:
            method (str, optional): the method to use for aggregation (currently only Borda). Defaults to 'Borda'.

        Returns:
            dict: one DataFrame per comparison with a ``borda_ranks`` column
            (higher is more differential), indexed by node.
        """
        if not self.pairwise_ranks:
            raise ValueError("No n2v2r rankings found, run fit_transform_rank first")
        if method.casefold() != 'borda':
            raise NotImplementedError(
                'Aggregation method not found. Available methods: Borda')

        start_time = time.time()
        self.__log("\nRank aggregation with Borda ...")

        self.pairwise_aggregate_ranks = {
            key: pd.DataFrame({'borda_ranks': borda_aggregate(ranks.to_numpy())},
                              index=ranks.index)
            for key, ranks in self.pairwise_ranks.items()}

        self.__log(f"\tFinished aggregation in {round(time.time() - start_time, 2)} seconds", level=1)

        if self.save_dir:
            for k, rank in self.pairwise_aggregate_ranks.items():
                rank.to_csv(os.path.join(
                    self.save_dir, k + "_agg.tsv"), sep='\t', index=True)

        return self.pairwise_aggregate_ranks

    def signed_ranks_transform(self, prior_signed_ranks=None):
        """
        Gives every rank the sign of the node in a prior signed ranking, e.g.,
        the degree difference.

        Args:
            prior_signed_ranks: list (one per comparison, in order) or dict
                (keyed by comparison) of pd.Series with the prior signed
                ranking. Defaults to the degree difference computed by
                :meth:`degree_difference_ranking`.

        Returns:
            dict: one DataFrame per comparison with all signed ranks for all
            combinations of parameters
        """
        if prior_signed_ranks is None:
            if self.prior_signed_ranks is None:
                raise ValueError("Prior signed ranks needed, run degree_difference_ranking beforehand "
                                 "or provide them in arguments.")
            prior_signed_ranks = self.prior_signed_ranks
        if not self.pairwise_ranks:
            raise ValueError("No n2v2r rankings found, run fit_transform_rank first")

        keys = list(self.pairwise_ranks)
        if not isinstance(prior_signed_ranks, dict):
            prior_signed_ranks = dict(zip(keys, prior_signed_ranks))
        if len(prior_signed_ranks) != len(keys):
            raise ValueError(
                f"Expected {len(keys)} prior signed rankings (one per comparison), "
                f"got {len(prior_signed_ranks)}")

        self.__log("\nSigned ranks transformation ...")
        start_time = time.time()

        self.pairwise_signed_ranks = {}
        for key in keys:
            ranks = self.pairwise_ranks[key]
            self.pairwise_signed_ranks[key] = pd.DataFrame(
                {column: signed_transform_single(ranks[column], prior_signed_ranks[key])
                 for column in ranks.columns})

        if self.pairwise_aggregate_ranks:
            self.pairwise_signed_aggregate_ranks = {
                key: signed_transform_single(
                    self.pairwise_aggregate_ranks[key].iloc[:, 0], prior_signed_ranks[key]
                ).to_frame("signed_agg_ranks")
                for key in keys}

        self.__log(f"\tFinished signed transformation in {round(time.time() - start_time, 2)} seconds", level=1)

        if self.save_dir:
            for k, rank in self.pairwise_signed_ranks.items():
                rank.to_csv(os.path.join(
                    self.save_dir, k + "_signed.tsv"), sep='\t', index=True)
            if self.pairwise_signed_aggregate_ranks:
                for k, rank in self.pairwise_signed_aggregate_ranks.items():
                    rank.to_csv(os.path.join(
                        self.save_dir, k + "_agg_signed.tsv"), sep='\t', index=True)

        return self.pairwise_signed_ranks

    def significance(self, dimensions=None, distance_metrics=None):
        """
        Tests every node for a larger shift between the graphs than nodes of
        similar degree, with a degree-adjusted empirical null (see
        :mod:`node2vec2rank.significance`).

        By default the test combines the same embedding dimensions and
        distance metrics as the rankings. ``dimensions="elbow"`` uses only the
        dimension at the elbow of the singular values instead: when the change
        lives in the dominant structure this is much more powerful, but when
        the elbow is too low it misses the change entirely (see the benchmarks).

        Args:
            dimensions: list of embedding dimensions to combine, or "elbow";
                defaults to the configured dimensions.
            distance_metrics: list of distance metrics to combine; defaults to
                the configured metrics.

        Returns:
            dict: one DataFrame per comparison with columns ``z`` (larger is
            more differential), ``pvalue`` (one-sided), ``qvalue``
            (Benjamini-Hochberg) and ``degree``, indexed by node.
        """
        if self.node_embeddings is None:
            raise ValueError("No n2v2r embeddings found, run fit_transform_rank first")
        if dimensions is None:
            dimensions = self.embed_dimensions
        elif isinstance(dimensions, str):
            if dimensions.casefold() != "elbow":
                raise ValueError(f'dimensions must be a list of integers or "elbow", got {dimensions!r}')
            dimensions = [self.selected_dimension]
        dimensions = list(dimensions)
        if max(dimensions) > self.max_embed_dim:
            raise ValueError(
                f"Dimensions up to {self.max_embed_dim} were embedded, got {dimensions}")
        distance_metrics = self.distance_metrics if distance_metrics is None else list(distance_metrics)

        self.__log(f"\nSignificance with dimensions {dimensions} and distance metrics {distance_metrics} ...")
        self.pairwise_significance = {}
        for key, distances in self.__distances(dimensions, distance_metrics).items():
            degree = self.degrees(key)
            z, pvalues, qvalues = empirical_null_test(distances.to_numpy(), degree.to_numpy())
            self.pairwise_significance[key] = pd.DataFrame(
                {"z": z, "pvalue": pvalues, "qvalue": qvalues, "degree": degree.to_numpy()},
                index=self.node_names)
            self.__log(f"\tComparison {key}: {int(np.sum(qvalues < 0.05))} nodes with q < 0.05", level=1)

        if self.save_dir:
            for k, frame in self.pairwise_significance.items():
                frame.to_csv(os.path.join(
                    self.save_dir, k + "_significance.tsv"), sep='\t', index=True)

        return self.pairwise_significance

    def degree_difference_ranking(self):
        """
        Computes the degree difference (DeDi) ranking for every comparison of
        the configured strategy, as the (mean) degree of the reference graph(s)
        minus the degree of the target graph.

        Returns:
            dict: one DataFrame per comparison with ``DeDi`` and ``absDeDi`` columns
        """
        degrees = [np.asarray(graph.sum(axis=0)).ravel() for graph in self.graphs]

        pairwise_DeDi_ranking = {}
        for key, reference, target in self.comparisons():
            DeDi = np.mean([degrees[i] for i in reference], axis=0) - degrees[target]
            pairwise_DeDi_ranking[key] = pd.DataFrame(
                {"DeDi": DeDi, "absDeDi": np.abs(DeDi)}, index=self.node_names)

        self.prior_signed_ranks = {k: v["DeDi"] for k, v in pairwise_DeDi_ranking.items()}

        if self.save_dir:
            for k, rank in pairwise_DeDi_ranking.items():
                rank.to_csv(os.path.join(
                    self.save_dir, k + "_degDif.tsv"), sep='\t', index=True)

        return pairwise_DeDi_ranking
