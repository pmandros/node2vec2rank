import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
import scipy.spatial.distance

import os
import random
import time

from joblib import Parallel, delayed
import spectral_embedding as se

from datetime import datetime

from node2vec2rank.pre_utils import network_transform


class n2v2r():
    def __init__(self, graphs: list, node_names: list, config: dict):
        self.config = config
        self.node_names = node_names
        self.graphs = graphs
        self.num_graphs = len(graphs)
        self.embed_dim = self.config['embed_dim']
        self.distance_metrics = self.config['distance_metrics']
        self.num_reps = self.config['reps']

        self.save_dir = None

        self.node_embeddings = None
        self.pairwise_ranks = None
        self.pairwise_signed_ranks = None
        self.pairwise_aggregate_ranks = None
        self.pairwise_signed_aggregate_ranks = None

        if self.config["seed"]:
            random.seed(self.config["seed"])
            np.random.seed(self.config["seed"])

    """
    Computes the differential ranks of nodes for a given sequence of graphs.
    returns A list of dataframes (one per comparison) with all computed ranks for all combinations of parameters
    """

    def fit_transform_rank(self):
        pairwise_ranks_dict = {}

        now = datetime.now().strftime(r"%m_%d_%Y_%H_%M_%S")

        if self.config["save_dir"]:
            self.save_dir = os.path.join(
                self.config["save_dir"], now)
            os.makedirs(self.save_dir)

        print(
            f"\nRunning n2v2r with dimensions {self.embed_dim} and distance metrics {self.distance_metrics} ...")
        start_time = time.time()

        for top_percent in self.config['top_percent_keep']:
            for bin in self.config['binarize']:
                # network transformation
                grns_transformed = []
                for graph in self.graphs:
                    grns_transformed.append(csc_matrix(network_transform(graph,
                                                                         binarize=bin,
                                                                         threshold=self.config['threshold'],
                                                                         absolute=self.config['absolute'],
                                                                         top_percent_keep=top_percent,
                                                                         project_unipartite=self.config['project_unipartite'])))

                # fitting UASE
                start_time_uase = time.time()
                _, self.node_embeddings = se.UASE(
                    grns_transformed, self.embed_dim)
                exec_time_embed = round(time.time() - start_time_uase, 2)

                del grns_transformed

                if self.config["verbose"] == 1:
                    print(
                        f"\tUASE embedding in {exec_time_embed} seconds for bin={bin} and keep_top={top_percent}%")

                start_time_ranking = time.time()

                # for every pair of consecutive graphs
                for i in range(1, self.num_graphs):
                    graph_comparison_key = str(i) + "vs" + str(i+1)
                    per_graph_comp_and_prepro_combo_ranks_pd = pd.DataFrame(
                        index=self.node_names)

                    # go over all provided choices for number of latent dimensions
                    subsets = sample_noreplace(
                        np.arange(self.embed_dim), self.num_reps, 4)
                    for index, subset in enumerate(subsets):
                        embed_one = np.transpose(
                            self.node_embeddings[i-1, :, subset])
                        embed_two = np.transpose(
                            self.node_embeddings[i, :, subset])
                        for distance_metric in self.distance_metrics:
                            col_name = "bin-" + \
                                str(bin)+"_top-"+str(top_percent)+"_rep-" + \
                                str(index)+"_distance-"+distance_metric
                            distances = compute_pairwise_distances(
                                embed_one, embed_two, distance_metric)
                            per_graph_comp_and_prepro_combo_ranks_pd[col_name] = distances

                    if self.config["save_dir"]:
                        per_graph_comp_and_prepro_combo_ranks_pd.to_csv(os.path.join(
                            self.save_dir, graph_comparison_key+".tsv"), sep='\t', index=True)

                    if graph_comparison_key in pairwise_ranks_dict:
                        pairwise_ranks_dict[graph_comparison_key].append(
                            per_graph_comp_and_prepro_combo_ranks_pd)
                    else:
                        pairwise_ranks_dict[graph_comparison_key] = []
                        pairwise_ranks_dict[graph_comparison_key].append(
                            per_graph_comp_and_prepro_combo_ranks_pd)

                exec_time_ranking = round(time.time() - start_time_ranking, 2)
                if self.config["verbose"] == 1:
                    print(f"\tRanking in {exec_time_ranking} seconds")

        self.pairwise_ranks = dict([(key, pd.concat(
            pairwise_ranks_dict[key], axis=1)) for key in pairwise_ranks_dict])
        assert (len(pairwise_ranks_dict) == len(self.graphs),
                'Number of comparisons should be the same as number of graphs')

        print(
            f"n2v2r computed {len(self.pairwise_ranks)*self.num_reps*len(self.config['binarize'])*len(self.config['top_percent_keep'])*len(self.distance_metrics)} rankings for {len(self.pairwise_ranks)} comparison(s) in {round(time.time() - start_time, 2)} seconds")
        return self.pairwise_ranks

    """
    Computes the aggregation of ranks of nodes for a given sequence of graphs.
    method: the method to use for aggregation (currently only Borda)
    returns A list of dataframes (one per comparison) with aggregated ranks 
    """

    def aggregate_transform(self, method='Borda'):
        # if ranks have been computed already
        if self.pairwise_ranks:
            pairwise_aggregate_ranks_dict = {}

            start_time = time.time()
            print("\nRank aggregation with Borda ...")

            for comparison_key in self.pairwise_ranks:

                ranks_list = []
                # collect the columns containing the different combo rankings
                for (_, column_data) in self.pairwise_ranks[comparison_key].items():
                    # sort according to rank value and get the index
                    rank_series = pd.Series(column_data, index=self.node_names)
                    rank_series.sort_values(ascending=False, inplace=True)
                    ranks_list.append(rank_series.index.to_list())

                # aggregate the rankings
                # TODO currently it works only with Borda and rankaggregator package
                if method.casefold() == 'borda'.casefold():

                    # agg = ra.RankAggregator()
                    # borda_ranking = agg.borda(ranks_list)
                    # borda_node_names = [x[0] for x in borda_ranking]
                    # borda_ranks = [x[1] for x in borda_ranking]
                    # aggregate_ranking_pd = pd.DataFrame(
                    #     borda_ranks, index=borda_node_names, columns=["borda_ranks"])
                    aggregate_ranking_pd = borda_aggregate_parallel(ranks_list)

                    if self.config["save_dir"]:
                        aggregate_ranking_pd.to_csv(os.path.join(
                            self.save_dir, comparison_key+"_agg.tsv"), sep='\t', index=True)

                else:
                    print('Aggregation method not found. Available methods: Borda')
                    return None

                pairwise_aggregate_ranks_dict[comparison_key] = aggregate_ranking_pd

            self.pairwise_aggregate_ranks = pairwise_aggregate_ranks_dict
            exec_time_agg = round(time.time() - start_time, 2)
            print(f"\tFinished aggregation in {exec_time_agg} seconds")
        else:
            print("No n2v2r embeddings found")

        return self.pairwise_aggregate_ranks

    """
    Computes the sign transofrmation of ranks of nodes for a given sequence of 
    graphs and a prior signed ranking.
    prior_signed_ranks: the prior singed ranking to use
    returns A list of dataframes (one per comparison) with all computed singed ranks for all combinations of parameters
    """

    def signed_ranks_transform(self, prior_signed_ranks: pd.Series):
        if self.pairwise_ranks:
            print("\nSigned ranks transformation ...")
            start_time = time.time()

            pairwise_signed_ranks_dict = {}

            # sign the aggregate if already computed
            if self.pairwise_aggregate_ranks:
                pairwise_signed_aggregate_ranks_dict = {}

            for comparison_key in self.pairwise_ranks:
                singed_ranks_pd = pd.DataFrame()

                # sign every column and add to the dataframe
                for column_combo_index in range(self.pairwise_ranks[comparison_key].shape[1]):
                    combo_rank_s = self.pairwise_ranks[comparison_key].iloc[:,
                                                                            column_combo_index]
                    combo_signed_ranks_s = signed_transform_single(
                        combo_rank_s, prior_signed_ranks)
                    singed_ranks_pd[self.pairwise_ranks[comparison_key]
                                    .columns[column_combo_index]] = combo_signed_ranks_s.values

                singed_ranks_pd.index = combo_signed_ranks_s.index
                if self.config["save_dir"]:
                    singed_ranks_pd.to_csv(os.path.join(
                        self.save_dir, comparison_key+"_signed.tsv"), sep='\t', index=True)

                pairwise_signed_ranks_dict[comparison_key] = singed_ranks_pd

                # sign the aggregate
                if self.pairwise_aggregate_ranks:
                    combo_agg_rank_s = self.pairwise_ranks[comparison_key].iloc[:, 0]
                    combo_signed_agg_ranks_s = signed_transform_single(
                        combo_agg_rank_s, prior_signed_ranks)
                    combo_signed_agg_ranks_pd = pd.DataFrame(
                        combo_signed_agg_ranks_s.values, index=combo_signed_agg_ranks_s.index, columns=["signed_agg_ranks"])
                    if self.config["save_dir"]:
                        combo_signed_agg_ranks_pd.to_csv(os.path.join(
                            self.save_dir, comparison_key+"_agg_signed.tsv"), sep='\t', index=True)
                    pairwise_signed_aggregate_ranks_dict[comparison_key] = pd.DataFrame(
                        combo_signed_agg_ranks_s.values, index=combo_signed_agg_ranks_s.index, columns=["signed_agg_ranks"])

        self.pairwise_signed_ranks = pairwise_signed_ranks_dict

        if self.pairwise_aggregate_ranks:
            self.pairwise_signed_aggregate_ranks = pairwise_signed_aggregate_ranks_dict

        exec_time_signed = round(time.time() - start_time, 2)
        print(
            f"\tFinished signed transformation in {exec_time_signed} seconds")

        return self.pairwise_signed_ranks


def signed_transform_single(ranks: pd.Series, prior_signed_ranks: pd.Series):
    node_names_list = []
    ranks_list = []
    for index, rank in ranks.items():
        if index in prior_signed_ranks.index:
            node_names_list.append(index)
            value = prior_signed_ranks.loc[index]
            if value > 0:
                ranks_list.append(rank)
            else:
                ranks_list.append(-rank)

    return pd.Series(ranks_list, index=node_names_list)


# def borda_aggregate(rankings: list):
#     index = rankings[0]
#     borda_ranking = pd.DataFrame(index=index)
#     num_candidates = len(index)

#     for i in range(len(rankings)):
#         ranking = rankings[i]

#         ranks = [num_candidates-ranking.index(node) for node in index]

#         borda_ranking[str(i)] = ranks

#     ranks = borda_ranking.sum(axis=1)

#     to_return = pd.DataFrame(ranks, index=index, columns=['borda_ranks'])
#     to_return.sort_values(by='borda_ranks', ascending=False,inplace=True)
#     return to_return

def get_ranking(ranking: list, index: list):
    num_candidates = len(index)

    return [num_candidates-ranking.index(node) for node in index]


def borda_aggregate_parallel(rankings: list):
    index = rankings[0]

    results = np.asarray(Parallel(
        n_jobs=-2)(delayed(get_ranking)(ranking, index) for ranking in rankings))
    borda_ranks = np.sum(results, axis=0)
    to_return = pd.DataFrame(borda_ranks, index=index, columns=['borda_ranks'])
    to_return.sort_values(by='borda_ranks', ascending=False, inplace=True)

    return to_return


"""
Computes Pairwise Distances between two embedding matrices
nodes: names of nodes in the graph
mat1: first embedding matrix (n_nodes,d_dimensions)
mat1: second embedding matrix (n_nodes,d_dimensions)
distance: distance metric to be used
returns Lists of nodes names and distance values sorted by similarity in decreasing order
"""


def compute_pairwise_distances(mat1, mat2, distance='cosine'):
    mat1 = mat1 - mat1.mean(axis=0, keepdims=True)
    mat2 = mat2 - mat2.mean(axis=0, keepdims=True)

    if distance == "cosine":
        dists = [scipy.spatial.distance.cosine(row1, row2)
                 for row1, row2 in zip(mat1, mat2)]
    elif distance == "euclidean":
        dists = [scipy.spatial.distance.euclidean(row1, row2)
                 for row1, row2 in zip(mat1, mat2)]
    elif distance == "cityblock":
        dists = [scipy.spatial.distance.cityblock(row1, row2)
                 for row1, row2 in zip(mat1, mat2)]
    elif distance == "chebyshev":
        dists = [scipy.spatial.distance.chebyshev(row1, row2)
                 for row1, row2 in zip(mat1, mat2)]
    elif distance == "correlation":
        dists = [scipy.spatial.distance.correlation(row1, row2, centered=False)
                 for row1, row2 in zip(mat1, mat2)]
        dists = [0 if np.isnan(x) else x for x in dists]
    else:
        raise Exception("Distance metric unknown")

    return dists


def sample_noreplace(arr, n, k):
    assert k <= len(arr)
    idx = np.random.randint(len(arr) - np.arange(k), size=[n, k])
    for i in range(k-1, 0, -1):
        idx[:, i:] += idx[:, i:] >= idx[:, i-1, None]
    return np.array(arr)[idx]
