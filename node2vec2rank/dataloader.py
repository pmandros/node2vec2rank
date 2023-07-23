import pandas as pd
import numpy as np
import networkx as nx
import os

from pre_utils import match_networks


class DataLoader():
    def __init__(self, config):
        self.config = config
        self.graphs = []
        self.interest_nodes = []
        self.__graph_filenames = config["graph_filenames"]
        self.__load_graphs()

    def get_graphs(self):
        return self.graphs

    def get_interest_nodes(self):
        return self.interest_nodes

    def __load_graphs(self):
        # go through every graph
        for i, graph_filename in enumerate(self.__graph_filenames):
            graph_pd = self.__load_graph(
                graph_filename=graph_filename, graph_index=i)
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
                raise ValueError("Impossible transformation")
        else:
            self.interest_nodes = col_nodes

    def __load_graph(self, graph_filename, graph_index):
        # check if in h5 format
        extension_type = graph_filename.split(".")[-1]
        if not self.config["is_edge_list"]:
            if extension_type != "h5":
                graph_pd = pd.read_csv(
                    os.path.join(
                        self.config["data_dir"], graph_filename),
                    index_col=0,
                    header=0,
                    sep=self.config["seperator"])
            else:
                graph_pd = pd.read_hdf(os.path.join(
                    self.config["data_dir"], graph_filename))
        else:
            edge_list_graph = nx.read_weighted_edgelist(os.path.join(
                self.config["data_dir"], graph_filename), nodetype=str)
            adj_matrix = nx.to_numpy_array(edge_list_graph)
            graph_pd = pd.DataFrame(
                adj_matrix, index=edge_list_graph.nodes, columns=edge_list_graph.nodes)
            print(graph_pd)

        # transpose (e.g., to bring regulators in rows)
        graph_pd = graph_pd.T if self.config["transpose"] else graph_pd

        row_nodes, col_nodes = graph_pd.index.to_numpy(), graph_pd.columns.to_numpy()
        num_rows, num_cols = np.size(row_nodes), np.size(col_nodes)
        print(
            f"""There are {num_rows} row nodes and {num_cols} column nodes in graph {graph_index+1}""")
        return graph_pd
