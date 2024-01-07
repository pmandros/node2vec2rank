import pandas as pd
import numpy as np
import networkx as nx
import os
from scipy.sparse import csc_matrix
import time


from preprocessing_utils import match_networks
from preprocessing_utils import network_transform



class DataLoader():
    def __init__(self, config):
        self.config = config
        self.graphs = []
        self.interest_nodes = []
        self.__graph_filenames = config["graph_filenames"]
        self.__load_graphs()

    def get_graphs(self):
        return self.graphs

    def get_nodes(self):
        return self.interest_nodes

    def __load_graphs(self):
        # go through every graph
        tic_loading = time.time()

        print('Loading graphs in memory ...')

        for i, graph_filename in enumerate(self.__graph_filenames):
            graph_pd = self.__load_graph(
                graph_filename=graph_filename, graph_index=i)
            self.graphs.append(graph_pd)

        self.graphs = match_networks(self.graphs)

        row_nodes = self.graphs[0].index.to_numpy()
        col_nodes = self.graphs[0].columns.to_numpy()

        # get the eventual node IDs after the planned transformations
        if np.size(row_nodes) != np.size(col_nodes):
            if self.config["project_unipartite_on"].casefold() == 'rows':
                self.interest_nodes = row_nodes
                print('\tGraphs are non-square and will be projected on row nodes')
            elif self.config["project_unipartite_on"].casefold() == 'columns':
                self.interest_nodes = col_nodes
                print('\tGraphs are non-square and will be projected on column nodes')
            else:
                raise ValueError("Impossible transformation")
        else:
            self.interest_nodes = col_nodes
            
        print(
            f"""\tThere are {np.size(self.interest_nodes)} common nodes and resulting networks will have size {np.size(self.interest_nodes)} by {np.size(self.interest_nodes)}""")
        
        
        
        if any([self.config['binarize'], self.config['threshold'] is not None, self.config['absolute'], int(self.config['top_percent_keep'])!=100, self.config['project_unipartite_on'] is not None]):
            print('Transforming graphs ...')
            
        variables = [("absoluting", self.config['absolute']), ("thresholding", self.config['threshold'] is not None),  ('to_unipartite',self.config['project_unipartite_on'] is not None),   ('sparcifying',int(self.config['top_percent_keep'])!=100),("binarize", self.config['binarize'])]
        for var_name, var_value in variables:
            if var_value:
                print(f"\t {var_name}")
        
        self.graphs=[network_transform(graph,
                                    binarize=self.config['binarize'],
                                    threshold=self.config['threshold'],
                                    absolute=self.config['absolute'],
                                    top_percent_keep=self.config['top_percent_keep'],
                                    project_unipartite_on=self.config['project_unipartite_on'])
                                    for graph in self.graphs]
        
        toc_loading = time.time()
        print(f"""Finished loading in {round(toc_loading - tic_loading, 2)} seconds \n""")

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
                    sep=self.config["separator"])
            else:
                graph_pd = pd.read_hdf(os.path.join(
                    self.config["data_dir"], graph_filename))
        else:
            edge_list_graph = nx.read_weighted_edgelist(os.path.join(
                self.config["data_dir"], graph_filename), delimiter=self.config["separator"], nodetype=str)
            adj_matrix = nx.to_numpy_array(edge_list_graph)
            graph_pd = pd.DataFrame(
                adj_matrix, index=edge_list_graph.nodes, columns=edge_list_graph.nodes)

        # transpose if rectangular (e.g., if bipartite to bring row nodes to column)
        graph_pd = graph_pd.T if self.config["transpose"] else graph_pd

        row_nodes, col_nodes = graph_pd.index.to_numpy(), graph_pd.columns.to_numpy()
        num_rows, num_cols = np.size(row_nodes), np.size(col_nodes)
        print(
            f"""\tThere are {num_rows} row nodes and {num_cols} column nodes in graph {graph_index+1}""")
        return graph_pd
