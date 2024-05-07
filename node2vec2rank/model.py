from datetime import datetime
import gc
import json
import os
import random
import numpy as np
import pandas as pd
import time
from scipy.sparse import csc_matrix
import spectral_embedding as se


from node2vec2rank.preprocessing_utils import network_transform
from node2vec2rank.model_utils import borda_aggregate_parallel, compute_pairwise_distances, signed_transform_single


class N2V2R:
    def __init__(self, graphs: list, nodes: list, config: dict):
        self.config = config
        self.node_names = nodes
        self.graphs = graphs
        self.num_graphs = len(graphs)
        self.embed_dimensions = self.config['embed_dimensions']
        self.max_embed_dim = max(self.embed_dimensions)
        self.distance_metrics = self.config['distance_metrics']
        self.save_dir = None
        self.comp_strategy = self.config['comp_strategy']

        self.node_embeddings = None
        self.pairwise_ranks = None
        self.pairwise_signed_ranks = None
        self.pairwise_aggregate_ranks = None
        self.pairwise_signed_aggregate_ranks = None
        self.prior_singed_ranks = None

        if self.config["seed"]:
            random.seed(self.config["seed"])
            np.random.seed(self.config["seed"])
            
        now = datetime.now().strftime(r"%m_%d_%Y_%H_%M_%S")

        if self.config["save_dir"]:
            self.save_dir = os.path.join(
                self.config["save_dir"], now)
            print(self.save_dir)
            os.makedirs(self.save_dir)
            with open(os.path.join(self.save_dir, "config.json"), 'w', encoding="utf-8") as f:
                json.dump(self.config, f)


    def __fit(self, graphs):
        # fitting UASE
        sparce_graphs = [csc_matrix(graph) for graph in graphs]
        _, self.node_embeddings = se.UASE(
            sparce_graphs, self.max_embed_dim)

    def __rank(self, pairwise_ranks_dict, comp_strategy):
        # for every pair of consecutive graphs
        for i in range(0, self.num_graphs):
            if i==0 and comp_strategy!='one_vs_rest':
                continue
            
            if comp_strategy!='one_vs_rest':
                graph_comparison_key = str(i)
            else:
                graph_comparison_key = str(i+1)  
  
   
            per_graph_comp_and_prepro_combo_ranks_pd = pd.DataFrame(
                index=self.node_names)

            # go over all provided choices for number of latent dimensions
            for dim in self.embed_dimensions:
                if comp_strategy=='sequential':
                    embed_one = self.node_embeddings[i-1, :, :dim]
                    embed_two = self.node_embeddings[i, :, :dim]

                elif comp_strategy=='one_vs_before':
                    embed_one = np.mean(self.node_embeddings[:i, :, :dim],axis=0)
                    embed_two = self.node_embeddings[i, :, :dim]

                elif comp_strategy=='one_vs_rest':
                    embed_one = np.mean(self.node_embeddings[np.arange(self.node_embeddings.shape[0]) != i, :, :dim],axis=0)
                    embed_two = self.node_embeddings[i, :, :dim]

                # go over all provided choices for distance metrics
                for distance_metric in self.distance_metrics:
                    if distance_metric=='cosine' and dim==1:
                        continue
                    col_name = f"dim-{dim}_distance-{distance_metric}"
                    distances = compute_pairwise_distances(
                        embed_one, embed_two, distance_metric)
                    per_graph_comp_and_prepro_combo_ranks_pd[col_name] = distances

            pairwise_ranks_dict.setdefault(graph_comparison_key, []).append(
                per_graph_comp_and_prepro_combo_ranks_pd)

    def fit_transform_rank(self):
        """
        Computes the differential ranks of nodes for a given sequence of graphs.
        returns A list of dataframes (one per comparison) with all computed ranks for all combinations of parameters

        Returns:
            pd.DatFrame: pairwise ranks of all nodes    
        """

        if self.config["verbose"] >= 0:
            print(
                f"\nRunning n2v2r with dimensions {self.embed_dimensions} and distance metrics {self.distance_metrics} ...")
        tic_n2v2r = time.time()

        # go over all pairwise comparisons 
        pairwise_ranks_dict = {}
        # fit UASE model
        tic_uase = time.time()
        self.__fit(graphs=self.graphs)
        toc_uase = time.time()
        if self.config["verbose"] == 1:
            print(f"""\tMulti-layer embedding in {
                round(toc_uase-tic_uase,2)} seconds""")
        toc_uase = time.time()

        # rank the nodes
        tic_rank = time.time()
        self.__rank(pairwise_ranks_dict=pairwise_ranks_dict, comp_strategy=self.comp_strategy)
        toc_rank = time.time()
        # if self.config["verbose"] == 1:
        #     print(f"\t\tRanking in {round(toc_rank-tic_rank,2)} seconds")

        self.pairwise_ranks = {key: pd.concat(
            pairwise_ranks_dict[key], axis=1) for key in pairwise_ranks_dict}
        # assert len(pairwise_ranks_dict) == len(
        #     self.graphs)-1, "Number of comparisons should be the same as number of graphs"
        
        num_rankings = sum([len(self.pairwise_ranks[key].columns) for key in self.pairwise_ranks])

        toc_n2v2r = time.time()
        if self.config["verbose"] >= 0:
            print(f"""n2v2r computed {num_rankings} rankings for {
                    len(self.pairwise_ranks)} comparison(s) in {round(toc_n2v2r - tic_n2v2r, 2)} seconds""")

        if self.config["save_dir"]:
            for key, rank in self.pairwise_ranks.items():
                rank.to_csv(os.path.join(
                    self.save_dir, key + ".tsv"), sep='\t', index=True)

        return self.pairwise_ranks

    def aggregate_transform(self, method='Borda'):
        """
        Computes the aggregation of ranks of nodes for a given sequence of graphs. 

        Args:
            method (str, optional): the method to use for aggregation (currently only Borda). Defaults to 'Borda'.

        Returns:
            List: A list of dataframes (one per comparison) with aggregated ranks
        """
        # if ranks have been computed already
        if self.pairwise_ranks:
            pairwise_aggregate_ranks_dict = {}

            start_time = time.time()
            if self.config["verbose"] >= 0:
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
                # TODO currently it works only with Borda
                if method.casefold() == 'borda':
                    aggregate_ranking_pd = borda_aggregate_parallel(ranks_list)
                else:
                    raise NotImplementedError(
                        'Aggregation method not found. Available methods: Borda')

                aggregate_ranking_pd = aggregate_ranking_pd.loc[self.node_names, :]
                pairwise_aggregate_ranks_dict[comparison_key] = aggregate_ranking_pd

            self.pairwise_aggregate_ranks = pairwise_aggregate_ranks_dict
            exec_time_agg = round(time.time() - start_time, 2)
            if self.config["verbose"] == 1:
                print(f"\tFinished aggregation in {exec_time_agg} seconds")

            if self.config["save_dir"]:
                for k, rank in self.pairwise_aggregate_ranks.items():
                    rank.to_csv(os.path.join(
                        self.save_dir, k + "_agg.tsv"), sep='\t', index=True)

        else:
            raise ValueError("No n2v2r embeddings found")

        return self.pairwise_aggregate_ranks

    def signed_ranks_transform(self, prior_signed_ranks: pd.Series = None):
        """ 
        Computes the sign transofrmation of ranks of nodes for a given sequence of 
        graphs and a prior signed ranking.

        Args:
            prior_signed_ranks (pd.Series): the prior singed ranking to use

        Returns:
            List(pd.DataFrame):  A list of dataframes (one per comparison) 
            with all computed singed ranks for all combinations of parameters
        """
        if prior_signed_ranks is None:
            if self.prior_singed_ranks:
                prior_signed_ranks = self.prior_singed_ranks
            else:
                raise ValueError("""Prior signed ranks needed, run degree_difference_ranking beforehand
                                 or provide them in arguments.""")

        if self.pairwise_ranks:
            print("\nSigned ranks transformation ...")
            start_time = time.time()

            pairwise_signed_ranks_dict = {}

            # sign the aggregate if already computed
            if self.pairwise_aggregate_ranks:
                pairwise_signed_aggregate_ranks_dict = {}

            for index, comparison_key in enumerate(self.pairwise_ranks):
                singed_ranks_pd = pd.DataFrame()

                # sign every column and add to the dataframe
                for column_combo_index in range(self.pairwise_ranks[comparison_key].shape[1]):
                    combo_rank_s = self.pairwise_ranks[comparison_key].iloc[:,
                                                                            column_combo_index]

                    combo_signed_ranks_s = signed_transform_single(
                        combo_rank_s, prior_signed_ranks[index])
                    singed_ranks_pd[self.pairwise_ranks[comparison_key]
                                    .columns[column_combo_index]] = combo_signed_ranks_s.values

                singed_ranks_pd.index = combo_signed_ranks_s.index

                pairwise_signed_ranks_dict[comparison_key] = singed_ranks_pd

                # sign the aggregate
                if self.pairwise_aggregate_ranks:
                    combo_agg_rank_s = self.pairwise_aggregate_ranks[comparison_key].iloc[:, 0]
                    combo_signed_agg_ranks_s = signed_transform_single(
                        combo_agg_rank_s, prior_signed_ranks[index])
                    combo_signed_agg_ranks_pd = pd.DataFrame(
                        combo_signed_agg_ranks_s.values, index=combo_signed_agg_ranks_s.index, columns=["signed_agg_ranks"])

                    pairwise_signed_aggregate_ranks_dict[comparison_key] = combo_signed_agg_ranks_pd

        self.pairwise_signed_ranks = pairwise_signed_ranks_dict

        if self.pairwise_aggregate_ranks:
            self.pairwise_signed_aggregate_ranks = pairwise_signed_aggregate_ranks_dict

        exec_time_signed = round(time.time() - start_time, 2)
        if self.config["verbose"] == 1:
            print(
                f"\tFinished signed transformation in {exec_time_signed} seconds")

        if self.config["save_dir"]:
            for (i, k) in enumerate(self.pairwise_signed_ranks.keys()):
                self.pairwise_signed_ranks[k].to_csv(os.path.join(
                    self.save_dir, k + "_signed.tsv"), sep='\t', index=True)

            # sign the aggregate
            if self.pairwise_aggregate_ranks:
                for (i, k) in enumerate(self.pairwise_signed_aggregate_ranks.keys()):
                    self.pairwise_signed_aggregate_ranks[k].to_csv(os.path.join(
                        self.save_dir, str(k) + "_agg_signed.tsv"), sep='\t', index=True)

        return self.pairwise_signed_ranks

    def degree_difference_ranking(self):
        """
        Computes the degree difference ranking. 

        Returns:
            list: pairwise DeDi ranking
        """
        pairwise_DeDi_ranking = {}
        
        for i in range(1, len(self.graphs)):
            graph_comparison_key = str(i)
            DeDi = np.sum(
                self.graphs[i-1], axis=0) - np.sum(self.graphs[i], axis=0)
            absDeDi = np.abs(DeDi)
            DeDi_data_dict = {"DeDi": DeDi, "absDeDi": absDeDi}

            ranking = pd.DataFrame.from_dict(DeDi_data_dict)
            ranking.index = self.node_names

            pairwise_DeDi_ranking[graph_comparison_key] = ranking

        self.prior_singed_ranks = [v.iloc[:, 0]
                                   for k, v in pairwise_DeDi_ranking.items()]

        if self.config["save_dir"]:
            for k, rank in pairwise_DeDi_ranking.items():
                rank.to_csv(os.path.join(
                    self.save_dir, k + "_degDif.tsv"), sep='\t', index=True)
        
        return pairwise_DeDi_ranking
