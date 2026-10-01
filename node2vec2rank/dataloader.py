import os
import time

import networkx as nx
import numpy as np
import pandas as pd

from node2vec2rank.config import resolve_config
from node2vec2rank.preprocessing_utils import match_networks, network_transform


class DataLoader():
    """Loads graphs from disk, restricts them to their common nodes and
    applies the configured preprocessing.

    Args:
        config: a config dict (flat or grouped in sections) or a path to a
            JSON config file. Needs at least ``graph_filenames``; see
            :data:`node2vec2rank.config.DEFAULT_CONFIG` for the other options.
        **overrides: config parameters that take precedence over ``config``.
    """

    def __init__(self, config=None, **overrides):
        self.config = resolve_config(config, **overrides)
        if not self.config["graph_filenames"]:
            raise ValueError("graph_filenames must list at least one graph file")
        self.graphs = []
        self.interest_nodes = []
        self.__graph_filenames = self.config["graph_filenames"]
        self.__load_graphs()

    def get_graphs(self):
        return self.graphs

    def get_nodes(self):
        return self.interest_nodes

    def __log(self, message):
        if self.config["verbose"] >= 0:
            print(message)

    def __load_graphs(self):
        tic_loading = time.time()

        self.__log('Loading graphs in memory ...')

        for i, graph_filename in enumerate(self.__graph_filenames):
            graph_pd = self.__load_graph(
                graph_filename=graph_filename, graph_index=i)
            self.graphs.append(graph_pd)

        self.graphs = match_networks(self.graphs)

        row_nodes = self.graphs[0].index.to_numpy()
        col_nodes = self.graphs[0].columns.to_numpy()
        projection = self.config["project_unipartite_on"]

        # get the eventual node IDs after the planned transformations
        if np.size(row_nodes) != np.size(col_nodes):
            if projection is None:
                raise ValueError(
                    "The graphs are not square (bipartite); set project_unipartite_on to "
                    "'rows' or 'columns' to project them to unipartite graphs")
            if projection.casefold() == 'rows':
                self.interest_nodes = row_nodes
                self.__log('\tGraphs are non-square and will be projected on row nodes')
            else:
                self.interest_nodes = col_nodes
                self.__log('\tGraphs are non-square and will be projected on column nodes')
        else:
            self.interest_nodes = col_nodes

        num_nodes = np.size(self.interest_nodes)
        self.__log(
            f"\tThere are {num_nodes} common nodes and resulting networks will have size "
            f"{num_nodes} by {num_nodes}")

        transformations = [
            ("absoluting", self.config['absolute']),
            ("thresholding", self.config['threshold'] is not None),
            ('to_unipartite', np.size(row_nodes) != np.size(col_nodes)),
            ('sparsifying', self.config['top_percent_keep'] != 100),
            ("binarize", self.config['binarize'])]
        if any(value for _, value in transformations):
            self.__log('Transforming graphs ...')
            for name, value in transformations:
                if value:
                    self.__log(f"\t {name}")

        self.graphs = [network_transform(graph,
                                         binarize=self.config['binarize'],
                                         threshold=self.config['threshold'],
                                         absolute=self.config['absolute'],
                                         top_percent_keep=self.config['top_percent_keep'],
                                         project_unipartite_on=projection)
                       for graph in self.graphs]

        toc_loading = time.time()
        self.__log(f"Finished loading in {round(toc_loading - tic_loading, 2)} seconds \n")

    def __load_graph(self, graph_filename, graph_index):
        path = os.path.join(self.config["data_dir"], graph_filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Graph file not found: {path}")

        if not self.config["is_edge_list"]:
            if graph_filename.endswith(".h5"):
                graph_pd = pd.read_hdf(path)
            else:
                graph_pd = pd.read_csv(
                    path, index_col=0, header=0, sep=self.config["separator"])
            # adjacency files with integer node IDs have string column headers
            # but integer row labels; align them so the nodes match
            graph_pd.index = graph_pd.index.astype(str)
            graph_pd.columns = graph_pd.columns.astype(str)
        else:
            edge_list_graph = nx.read_weighted_edgelist(
                path, delimiter=self.config["separator"], nodetype=str)
            adj_matrix = nx.to_numpy_array(edge_list_graph)
            graph_pd = pd.DataFrame(
                adj_matrix, index=list(edge_list_graph.nodes), columns=list(edge_list_graph.nodes))

        # transpose if rectangular (e.g., if bipartite to bring row nodes to column)
        graph_pd = graph_pd.T if self.config["transpose"] else graph_pd

        num_rows, num_cols = graph_pd.shape
        self.__log(
            f"\tThere are {num_rows} row nodes and {num_cols} column nodes in graph {graph_index+1}")
        return graph_pd
