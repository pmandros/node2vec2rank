import pandas as pd
import numpy as np
import networkx as nx
import os
import csrgraph as cg

from node2vec2rank.pre_utils import network_transform


class DataLoader():
    def __init__(self, config):
        self.config = config
        self.graphs = []
        self.id2node = {}
        self.__graph_names = [
            key for key in config.keys() if "graph_name" in key]
        self.__load_graphs()

        self.graphs_size = 0

    def get_graphs(self):
        return self.graphs

    def get_id2node(self):
        return self.id2node

    def size(self):
        return self.graphs_size

    def __load_graphs(self):
        if not self.config["bipartite"]:
            self.__process_unipartite_graph()
        else:
            self.__process_bipartite_graph()

    def __process_unipartite_graph(self):
        for i, graph_name in enumerate(self.__graph_names):
            graph_pd = pd.read_csv(
                os.path.join(self.config["data_dir"], self.config[graph_name]),
                index_col=0,
                header=0,
                sep=self.config["seperator"]).sort_index(ascending=True)

            row_nodes = graph_pd.index.to_numpy()
            col_nodes = graph_pd.columns.to_numpy()
            total_nodes = row_nodes.copy()
            num_rows, num_cols, num_total = np.size(
                row_nodes), np.size(col_nodes), np.size(total_nodes)
            print(
                f"There are {num_rows} row genes, {num_cols} column genes, and {num_total} total genes in graph no. {i} ")
            self.graphs.append(cg.csrgraph(graph_pd.values))
            self.id2node = total_nodes.copy()
            self.graphs_size = np.size(total_nodes)

    def __process_bipartite_graph(self):
        if not self.config["is_edge_list"]:
            for i, graph_name in enumerate(self.__graph_names):
                graph_pd = pd.read_csv(
                    os.path.join(self.config["data_dir"],
                                 self.config[graph_name]),
                    index_col=0,
                    header=0,
                    sep=self.config["seperator"]).sort_index(ascending=True)
                graph_pd.index = 'row_' + graph_pd.index.astype(str)
                graph_pd.columns = 'col_' + graph_pd.columns.astype(str)

                row_nodes = graph_pd.index.to_numpy()
                col_nodes = graph_pd.columns.to_numpy()
                total_nodes = np.append(row_nodes, col_nodes)

                num_rows, num_cols, num_total = np.size(
                    row_nodes), np.size(col_nodes), np.size(total_nodes)
                print(
                    f"There are {num_rows} row genes, {num_cols} column genes, and {num_total} total genes in graph no. {i} ")
                adj_matrix = network_transform(graph_pd,
                                               threshold=self.config["threshold"],
                                               percentile_to_keep=self.config["percentile_to_keep"],
                                               binarize=self.config["binarize"],
                                               symmetrize=self.config["symmetrize"],
                                               absolute=self.config["absolute"])

                self.graphs.append(cg.csrgraph(adj_matrix.values))
            self.id2node = total_nodes.copy()
            self.graphs_size = np.size(total_nodes)

        else:
            for i, graph_name in enumerate(self.__graph_names):
                grn = pd.read_csv(os.path.join(self.config["data_dir"], self.config[graph_name]),
                                  sep=self.config["seperator"], index_col=False, header=None,
                                  names=['source', 'target'])
                total_nodes = np.union1d(
                    grn['source'].unique(), grn['target'].unique())
                num_nodes = np.size(total_nodes)
                print(f"There are {num_nodes} total genes in graph no. {i} ")
                self.graphs.append(cg.read_edgelist(os.path.join(self.config["data_dir"], self.config[graph_name]),
                                                    sep=self.config["seperator"]))
            self.id2node = total_nodes.copy()
            self.graphs_size = np.size(total_nodes)
