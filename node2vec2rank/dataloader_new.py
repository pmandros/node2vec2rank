import pandas as pd
import numpy as np
import networkx as nx
import os

from node2vec2rank.pre_utils import match_networks

class DataLoader():
    def __init__(self, config):
        self.config = config
        self.graphs = []
        self.interest_nodes = []

        self.__graph_filenames = config["graph_filenames"]
        self.__load_graphs()

    def graphs(self):
        return self.graphs

    def interest_nodes(self):
        return self.interest_nodes

    def __load_graphs(self):
        # go through every graph
        for i, graph_filename in enumerate(self.__graph_filenames):
            # check if in h5 format
            if ('h5' not in graph_filename) and (not self.config["is_edge_list"]):
                graph_pd = pd.read_csv(
                    os.path.join(
                        self.config["data_dir"], graph_filename),
                    index_col=0,
                    header=0,
                    sep=self.config["seperator"])
            elif ('h5' in graph_filename) and (not self.config["is_edge_list"]):
                graph_pd = pd.read_hdf(os.path.join(
                    self.config["data_dir"], graph_filename))
            elif self.config["is_edge_list"]:
                raise ValueError(
                    "TODO: turn the edge list into a rectangular dataframe ")

            # transpose (e.g., to bring regulators in rows)
            if self.config['transpose']:
                graph_pd = graph_pd.T

            row_nodes = graph_pd.index.to_numpy()
            col_nodes = graph_pd.columns.to_numpy()

            num_rows, num_cols = np.size(row_nodes), np.size(col_nodes)
            print(
                f"There are {num_rows} row nodes and {num_cols} column nodes in graph {i+1} ")

            self.graphs.append(graph_pd)

        self.graphs = match_networks(self.graphs)

        row_nodes = self.graphs[0].index.to_numpy()
        col_nodes = self.graphs[0].columns.to_numpy()

        # get the eventual node IDs after the planned transformations
        if np.size(row_nodes) != np.size(col_nodes):
            print('Graphs are rectangular')
            if self.config["project_unipartite_on"].casefold() == 'rows':
                self.interest_nodes = row_nodes
            elif self.config["project_unipartite_on"].casefold() == 'columns':
                self.interest_nodes = col_nodes
            else:
                raise Exception("Impossible transformation")
        else:
            self.interest_nodes = col_nodes


