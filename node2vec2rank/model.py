import pandas as pd
import numpy as np
import os
import random
import time

import umap
from sklearn.manifold import Isomap, TSNE
from sklearn.decomposition import PCA

from node2vec2rank.utils import compute_pairwise_distances
import spectral_embedding as se
import rankaggregation as ra

from datetime import datetime

class n2v2r():
    def __init__(self, graphs: list, node_names: list, config: dict):
        self.config = config
        self.node_names = node_names
        self.graphs = graphs
        self.num_graphs = len(graphs)
        self.embed_dimensions = self.config['embed_dimensions']
        self.distance_metrics = self.config['distance_metrics']
        self.save_dir = None

        self.node_embeddings = None
        self.ranks_sequence = None
        self.signed_ranks_sequence = None
        self.aggregate_ranks_sequence = None
        self.aggregate_signed_ranks_sequence = None

        if self.config["seed"]:
            random.seed(self.config["seed"])
            np.random.seed(self.config["seed"])

    """
    Computes the differential ranks of nodes for a given sequence of graphs.
    returns A list of dataframes (one per comparison) with all computed ranks for all combinations of parameters
    """
    def fit_transform_rank(self):
        ranks_sequence_list = []

        now = datetime.now().strftime(r"%m_%d_%Y_%H_%M_%S")

        if self.config["save_dir"]:
            self.save_dir = os.path.join(
                self.config["save_dir"], now) 
            os.makedirs(self.save_dir)

        max_embed_dim = max(self.embed_dimensions)

        print(f"\nRunning n2v2r with dimensions {self.embed_dimensions} and distance metrics {self.distance_metrics} ...")
        start_time = time.time()

        #fitting UASE
        _, self.node_embeddings = se.UASE(self.graphs, max_embed_dim)

        exec_time_embed = round(time.time() - start_time, 2)

        if self.config["verbose"] == 1:
            print(f"    Finished UASE embedding in {exec_time_embed} seconds")

        if self.config["verbose"] == 1:
            print("Ranking ...")
        start_time = time.time()

        #for every pair of consecutive graphs
        for i in range(1,self.num_graphs):
            rank_pd = pd.DataFrame(index = self.node_names)

            #go over all provided choices for number of latent dimensions
            for d in self.embed_dimensions:
                embed_one = self.node_embeddings[i-1, :, :d+1]
                embed_two = self.node_embeddings[i, :, :d+1]

                for distance_metric in self.distance_metrics:
                    distances = compute_pairwise_distances(embed_one, embed_two, distance_metric)
                    rank_pd["dim-"+str(d)+"_distance-"+distance_metric] = distances

            if self.config["save_dir"]:
                rank_pd.to_csv(os.path.join(self.save_dir,str(i)+"vs"+str(i+1)+".tsv"), sep='\t', index = True)

            ranks_sequence_list.append(rank_pd)
            
        exec_time_ranking = round(time.time() - start_time, 2)
        if self.config["verbose"] == 1:
            print(f"    Finished ranking in {exec_time_ranking} seconds")

        self.ranks_sequence = ranks_sequence_list
     
        print(f"    n2v2r took {exec_time_embed+exec_time_ranking} seconds")
        return ranks_sequence_list

    """
    Computes the aggregation of ranks of nodes for a given sequence of graphs.
    method: the method to use for aggregation (currently only Borda)
    returns A list of dataframes (one per comparison) with aggregated ranks 
    """
    def aggregate_transform(self, method = 'Borda'):
        #if ranks have been computed already
        if self.ranks_sequence:
            aggregate_ranks_sequence_list = []

            start_time = time.time()
       
            i=0
            for ranks in self.ranks_sequence:
                ranks_list = []

                #collect the columns containing the different combo rankings
                for (_, column_data) in ranks.iteritems():
                    #sort according to rank value and get the index
                    rank_series = pd.Series(column_data, index=self.node_names)
                    rank_series.sort_values(ascending=False,inplace=True)
                    ranks_list.append(rank_series.index.to_list())

                #aggregate the rankings
                ##TODO currently it works only with Borda and rankaggregator package
                if method.casefold() == 'borda'.casefold():
                    print("\nRank aggregation with Borda ...")

                    agg = ra.RankAggregator()
                    borda_ranking = agg.borda(ranks_list)
                    borda_node_names = [x[0] for x in borda_ranking]
                    borda_ranks = [x[1] for x in borda_ranking]
                    aggregate_ranking_pd = pd.DataFrame(borda_ranks, index=borda_node_names, columns=["borda_ranks"])
                    if self.config["save_dir"]:
                        aggregate_ranking_pd.to_csv(os.path.join(self.save_dir,str(i+1)+"VS"+str(i+2)+"_agg.tsv"), sep='\t', index = True)
                    aggregate_ranks_sequence_list.append(aggregate_ranking_pd)
                else:
                    print('Aggregation method not found. Available methods: Borda')
                    return None
                
                i+=1

            exec_time_agg = round(time.time() - start_time, 2)
            print(f"    Finished aggregation in {exec_time_agg} seconds")


            self.aggregate_ranks_sequence = aggregate_ranks_sequence_list
        else:
            print("No n2v2r embeddings found")

        return self.aggregate_ranks_sequence
    
    """
    Computes the sign transofrmation of ranks of nodes for a given sequence of 
    graphs and a prior signed ranking.
    prior_signed_ranks: the prior singed ranking to use
    returns A list of dataframes (one per comparison) with all computed singed ranks for all combinations of parameters
    """
    def signed_transform(self, prior_signed_ranks: pd.Series):
        if self.ranks_sequence:
            print("\nSigned transformation ...")
            start_time = time.time()

            signed_ranks_sequence_list = []

            #sign the aggregate if already computed 
            if self.aggregate_ranks_sequence:
                aggregate_signed_ranks_sequence_list = []

            i = 0
            for ranks_pd in self.ranks_sequence:
                singed_ranks_pd = pd.DataFrame()

                #sign every column and add to the dataframe
                for column_combo_index in range(ranks_pd.shape[1]):
                    combo_rank_s = ranks_pd.iloc[:, column_combo_index]
                    combo_signed_ranks_s = self._signed_transform_single(combo_rank_s, prior_signed_ranks)
                    singed_ranks_pd[ranks_pd.columns[column_combo_index]] = combo_signed_ranks_s.values

                singed_ranks_pd.index = combo_signed_ranks_s.index
                if self.config["save_dir"]:
                        singed_ranks_pd.to_csv(os.path.join(self.save_dir,str(i+1)+"vs"+str(i+2)+"_signed.tsv"), sep='\t', index = True)
                signed_ranks_sequence_list.append(singed_ranks_pd)

                #sign the aggregate
                if self.aggregate_ranks_sequence:
                    combo_agg_rank_s = self.aggregate_ranks_sequence[i].iloc[:,0]
                    combo_signed_agg_ranks_s = self._signed_transform_single(combo_agg_rank_s, prior_signed_ranks)
                    combo_signed_agg_ranks_pd = pd.DataFrame(combo_signed_agg_ranks_s.values,index=combo_signed_agg_ranks_s.index,columns=["signed_agg_ranks"])
                    if self.config["save_dir"]:
                        combo_signed_agg_ranks_pd.to_csv(os.path.join(self.save_dir,str(i+1)+"vs"+str(i+2)+"_agg_signed.tsv"), sep='\t', index = True)
                    aggregate_signed_ranks_sequence_list.append(pd.DataFrame(combo_signed_agg_ranks_s.values,index=combo_signed_agg_ranks_s.index,columns=["signed_agg_ranks"]))

                i += 1                

        self.aggregate_signed_ranks_sequence = signed_ranks_sequence_list

        if self.aggregate_ranks_sequence:
            self.aggregate_signed_ranks_sequence = aggregate_signed_ranks_sequence_list

        exec_time_signed = round(time.time() - start_time, 2)
        print(f"    Finished signed transformation in {exec_time_signed} seconds")

        return signed_ranks_sequence_list
    
    def _signed_transform_single(self,ranks: pd.Series, prior_signed_ranks: pd.Series):
        node_names_list = []
        ranks_list = []
        for index, rank in ranks.iteritems():
            if index in prior_signed_ranks.index:
                node_names_list.append(index)
                value = prior_signed_ranks.loc[index]
                if value>0:
                    ranks_list.append(rank)
                else:
                    ranks_list.append(-rank)
        
        return pd.Series(ranks_list, index=node_names_list)
    
   