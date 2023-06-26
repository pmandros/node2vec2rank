
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.sparse import csc_matrix
import json
import os


import matplotlib as mpl
from collections import defaultdict
import gseapy
from itertools import chain
import plotly.express as px
import plotly.io as pio
from math import ceil


"""
Reads gmt files into dictionaries
"""


def read_gmt(gmt_fn, read_descriptor=False):
    genesets_dict = {}
    with open(gmt_fn) as genesets:
        for line in genesets:
            entry = line.strip().split("\t")
            name = entry[0]
            descriptor = entry[1]
            genes = [g.split(",")[0] for g in entry[2:]]
            if read_descriptor:
                genesets_dict[name] = {
                    'descriptor': descriptor, 'genes': genes}
            else:
                genesets_dict[name] = genes
    return genesets_dict


"""
Computes the discounted cumulative gain for a given relevance vector
"""


def discounted_cumulative_gain(relevance_vector, denominator=None):
    relevance_vector = np.array(relevance_vector)
    numerator = (2**relevance_vector-1)
    if denominator is None:
        denominator = np.log2(np.arange(relevance_vector.shape[0]) + 2)
    return np.sum(np.divide(numerator, denominator))


"""
Computes the normalized discounted cumulative gain for a given relevance vector
"""


def normalized_discounted_cumulative_gain(relevance_vector, binary_relevance=False, denominator=None, ideal_numerator=None):
    relevance_vector = np.array(relevance_vector)
    if np.sum(relevance_vector) == 0:
        return 0
    else:
        if denominator is None:
            denominator = np.log2(np.arange(len(relevance_vector)) + 2)

        dcg = discounted_cumulative_gain(relevance_vector, denominator)
        if binary_relevance:
            num_relevant = np.sum(relevance_vector)
            if ideal_numerator is None:
                ideal_numerator = (2**np.ones(num_relevant)-1)
                ideal_numerator = np.pad(ideal_numerator,
                                         (0, relevance_vector.shape[0]-num_relevant), mode='constant')
            idcg = np.sum(np.divide(ideal_numerator, denominator))
        else:
            idcg = discounted_cumulative_gain(
                sorted(relevance_vector, reverse=True), denominator)

        return dcg/idcg


def relevance_and_ndcg(ranked_items, relevant_items, binary_relevance=True, denominator=None, ideal_numerator=None):
    relevance_vector = [1 if x in relevant_items else 0 for x in ranked_items]
    if sum(relevance_vector) == 0:
        return 0
    else:
        return normalized_discounted_cumulative_gain(relevance_vector, binary_relevance=binary_relevance,
                                                     denominator=denominator, ideal_numerator=ideal_numerator)


"""
Computes the empirical p-value performing permutation tests in parallel
"""


def parallel_empirical_p_ndcg(ndcg, permutations, relevant_items, denominator=None, ideal_numerator=None):
    results = Parallel(n_jobs=-2)(delayed(relevance_and_ndcg)(i, relevant_items,
                                                              denominator=denominator, ideal_numerator=ideal_numerator) for i in permutations)
    return (np.array(results) > ndcg).sum()/len(permutations)


# def generate_two_DSBM_graphs_random_changes(num_nodes, num_com, multiplicity=1):
#     first_true_memb_matrix = np.random.uniform(0.01, 0.2, (num_com, num_com))
#     second_true_memb_matrix = np.random.uniform(0.01, 0.2, (num_com, num_com))

#     stacked_true_memb_matrix = np.stack(
#         (first_true_memb_matrix, second_true_memb_matrix))

#     self_membership_probs = np.repeat(1/num_com, num_com)

#     Z = np.random.choice(
#         range(num_com), p=self_membership_probs, size=num_nodes)

#     list_of_pairs_of_graphs = []
#     for _ in range(multiplicity):
#         As = np.zeros((2, num_nodes, num_nodes))
#         for t in range(2):
#             temp = stats.bernoulli.rvs(stacked_true_memb_matrix[t][Z, :][:, Z])
#             As[t] = np.tril(temp, -1) + np.tril(temp, -1).T
#         list_of_pairs_of_graphs.append(As)
#     return list_of_pairs_of_graphs, first_true_memb_matrix, second_true_memb_matrix


# def generate_two_DSBM_graphs_fixed_four(num_nodes, num_com=4,  multiplicity=1):
#     if num_com != 4:
#         num_com = 4

#     first_true_memb_matrix = np.array([
#         [0.10, 0.10, 0.02, 0.06],
#         [0.02, 0.20, 0.04, 0.10],
#         [0.18, 0.04, 0.02, 0.02],
#         [0.08, 0.02, 0.18, 0.10]])

#     second_true_memb_matrix = np.array([
#         [0.10, 0.10, 0.02, 0.06],
#         [0.16, 0.16, 0.04, 0.10],
#         [0.04, 0.04, 0.09, 0.02],
#         [0.16, 0.16, 0.04, 0.10]])

#     stacked_true_memb_matrix = np.stack(
#         (first_true_memb_matrix, second_true_memb_matrix))

#     self_membership_probs = np.repeat(1/num_com, num_com)

#     Z = np.random.choice(
#         range(num_com), p=self_membership_probs, size=num_nodes)
#     colours = np.array(list(mpl.colors.TABLEAU_COLORS.keys())[0:num_com])
#     Zcol = colours[Z]
#     # need to have integers to plot the color label
#     le = preprocessing.LabelEncoder()
#     assignments = le.fit_transform(Zcol)

#     list_of_pairs_of_graphs = []
#     for i in range(multiplicity):
#         As = np.zeros((2, num_nodes, num_nodes))
#         for t in range(2):
#             temp = stats.bernoulli.rvs(stacked_true_memb_matrix[t][Z, :][:, Z])
#             As[t] = np.tril(temp, -1) + np.tril(temp, -1).T
#         list_of_pairs_of_graphs.append(As)
#     return list_of_pairs_of_graphs, stacked_true_memb_matrix[0], stacked_true_memb_matrix[1], assignments


"""
Given a ranking (node integer IDs) and the true community membership matrices, it outputs the total and per group ndcg
"""
# def evaluate_DBSM_ranking(first_true_memb_matrix, second_true_memb_matrix, ranking):
#     ranking = np.int16(ranking)
#     num_communities = first_true_memb_matrix.shape[1]
#     n = len(ranking)

#     # assumes uniform group membership
#     if not (n/num_communities).is_integer():
#         raise Exception("n should be divided by num comms")
#     step = int(n/num_communities)

#     Ds = np.linalg.norm(np.abs(first_true_memb_matrix -
#                         second_true_memb_matrix), axis=1)
#     sort_comm_difs = np.flip(np.argsort(Ds))
#     print(f"Sorted Communities: {sort_comm_difs}")

#     rel_per_com = num_communities - sort_comm_difs
#     vector_of_relevances = np.array(
#         [element for element in rel_per_com for i in range(step)])

#     # get the ndcg for the entire ranking
#     total_ranking_relevance = vector_of_relevances[ranking]
#     total_score = normalized_discounted_cumulative_gain(
#         total_ranking_relevance, binary_relevance=False)

#     scores = []
#     scores.append(total_score)
#     # get the ndcg community
#     for i in range(num_communities):
#         group_relevance_vector = total_ranking_relevance[(
#             i*step):((i+1)*step)].copy()
#         group_relevance_vector[group_relevance_vector != rel_per_com[i]] = 0
#         group_relevance_vector[group_relevance_vector != 0] = 1
#         group_score = normalized_discounted_cumulative_gain(
#             group_relevance_vector, binary_relevance=True)
#         scores.append(group_score)

#     return scores, total_ranking_relevance


# def versus_DeDi_comparison(data_generator, num_com, num_nodes, multiplicity):
#     list_of_pairs_of_graphs, first_true_memb_matrix, second_true_memb_matrix = data_generator(num_com=num_com, num_nodes=num_nodes, multiplicity=multiplicity)

#     all_DeDi_scores = []
#     all_n2v2r_scores = []
#     all_n2v2r_scores_borda = []

#     for random_graphs in list_of_pairs_of_graphs:
#         graphs=[]

#          # Opening JSON file
#         f = open('../config_run_dbsm_comp_fixed.json')
#         # returns JSON object as
#         # a dictionary
#         data = json.load(f)

#         config = {param: value for section, params in data.items()
#                 for param, value in params.items()}
#         node_dict = dict(zip(np.arange(num_nodes), np.arange(num_nodes)))

#         # print(data['method'] )
#         if data['simulation_params']['method'] == 'UASE':
#             graphs.append(csc_matrix(random_graphs[0]))
#             graphs.append(csc_matrix(random_graphs[1]))
#         else:
#             graphs.append(cg.csrgraph(random_graphs[0]))
#             graphs.append(cg.csrgraph(random_graphs[1]))

#         # model = Model2Rank(node_dict=node_dict, config=data, graphs=graphs)
#         # rankings = model.walk_fit_rank()

#         # check if there are multiple values for any given parameter
#         multi_params = [key for key in config.keys() if isinstance(
#             config[key], list) and len(config[key]) > 1]

#         if not multi_params:
#             print("Model2Rank")
#             rankings = Model2Rank(graphs=graphs, config=config,
#                     node_dict=node_dict).walk_fit_rank()
#         else:
#             print('MultiModel2Rank')
#             rankings = MultiModel2Rank(graphs=graphs, config=config,
#                             node_dict=node_dict).walk_fit_rank()

#         shutil.rmtree("../output/Word2Vec/")

#         ##

#         # ranking_distances_per_combo = rankings.filter(like="dist", axis=1)
#         # avg_distances = ranking_distances_per_combo.mean(axis=1)
#         # n2v_ranked_nodes = avg_distances.sort_values(ascending = False, inplace=False).index.to_list()

#         #### try to aggregate rel_per_com according to rank aggregation methods

#         #create one ranking per combo

#         n2v2r_distances_per_combo = rankings.filter(like="dist", axis=1)
#         avg_distances = n2v2r_distances_per_combo.mean(axis=1)
#         n2v_ranked_nodes = avg_distances.sort_values(ascending = False, inplace=False).index.to_list()


#         index = rankings.index.to_list()

#         ranked_lists = []
#         for (columnName, columnData) in n2v2r_distances_per_combo.iteritems():
#             temp_df = pd.DataFrame(columnData.values, columns=['col'], index = index)
#             temp_df.sort_values(by='col', ascending=False, inplace=True)
#             ranked_lists.append(temp_df.index.to_list())

#         agg = ra.RankAggregator()
#         n2v_ranked_nodes_borda = agg.borda(ranked_lists)
#         n2v_ranked_nodes_borda = [x[0] for x in n2v_ranked_nodes_borda]


#         degs_one = np.sum(random_graphs[0], axis = 0)
#         degs_two = np.sum(random_graphs[1], axis = 0)
#         DeDi = np.abs(degs_two - degs_one)
#         DeDi_ranked_nodes = np.flip(np.argsort(DeDi))


#         n2v2r_scores, _ = evaluate_DBSM_ranking(first_true_memb_matrix, second_true_memb_matrix, n2v_ranked_nodes)
#         DeDi_scores, _ = evaluate_DBSM_ranking(first_true_memb_matrix, second_true_memb_matrix, DeDi_ranked_nodes)
#         n2v2r_scores_borda, _ = evaluate_DBSM_ranking(first_true_memb_matrix, second_true_memb_matrix, n2v_ranked_nodes_borda)


#         all_DeDi_scores.append(DeDi_scores)
#         all_n2v2r_scores.append(n2v2r_scores)
#         all_n2v2r_scores_borda.append(n2v2r_scores_borda)


# def versus_DeDi_comparison_multi(data_generator, num_com, num_nodes, multiplicity_model, multiplicity_sampling):

#     all_DeDi_scores = []
#     all_n2v2r_scores = []
#     all_n2v2r_scores_borda = []

#     for _ in range(multiplicity_model):
#         list_of_pairs_of_graphs, first_true_memb_matrix, second_true_memb_matrix = data_generator(num_com=num_com, num_nodes=num_nodes, multiplicity=multiplicity_sampling)
#         for random_graphs in list_of_pairs_of_graphs:
#             graphs=[]

#             # Opening JSON file
#             f = open('../config_run_dbsm_comp_fixed.json')
#             # returns JSON object as
#             # a dictionary
#             data = json.load(f)

#             config = {param: value for section, params in data.items()
#                     for param, value in params.items()}
#             node_dict = dict(zip(np.arange(num_nodes), np.arange(num_nodes)))

#             # print(data['method'] )
#             if data['simulation_params']['method'] == 'UASE':
#                 graphs.append(csc_matrix(random_graphs[0]))
#                 graphs.append(csc_matrix(random_graphs[1]))
#             else:
#                 graphs.append(cg.csrgraph(random_graphs[0]))
#                 graphs.append(cg.csrgraph(random_graphs[1]))

#             # model = Model2Rank(node_dict=node_dict, config=data, graphs=graphs)
#             # rankings = model.walk_fit_rank()

#             # check if there are multiple values for any given parameter
#             multi_params = [key for key in config.keys() if isinstance(
#                 config[key], list) and len(config[key]) > 1]

#             if not multi_params:
#                 print("Model2Rank")
#                 rankings = Model2Rank(graphs=graphs, config=config,
#                         node_dict=node_dict).walk_fit_rank()
#             else:
#                 print('MultiModel2Rank')
#                 rankings = MultiModel2Rank(graphs=graphs, config=config,
#                                 node_dict=node_dict).walk_fit_rank()

#             shutil.rmtree("../output/Word2Vec/")

#             ##

#             # ranking_distances_per_combo = rankings.filter(like="dist", axis=1)
#             # avg_distances = ranking_distances_per_combo.mean(axis=1)
#             # n2v_ranked_nodes = avg_distances.sort_values(ascending = False, inplace=False).index.to_list()

#             #### try to aggregate rel_per_com according to rank aggregation methods

#             #create one ranking per combo

#             n2v2r_distances_per_combo = rankings.filter(like="dist", axis=1)
#             avg_distances = n2v2r_distances_per_combo.mean(axis=1)
#             n2v_ranked_nodes = avg_distances.sort_values(ascending = False, inplace=False).index.to_list()


#             index = rankings.index.to_list()

#             ranked_lists = []
#             for (columnName, columnData) in n2v2r_distances_per_combo.iteritems():
#                 temp_df = pd.DataFrame(columnData.values, columns=['col'], index = index)
#                 temp_df.sort_values(by='col', ascending=False, inplace=True)
#                 ranked_lists.append(temp_df.index.to_list())

#             agg = ra.RankAggregator()
#             n2v_ranked_nodes_borda = agg.borda(ranked_lists)
#             n2v_ranked_nodes_borda = [x[0] for x in n2v_ranked_nodes_borda]

#             degs_one = np.sum(random_graphs[0], axis = 0)
#             degs_two = np.sum(random_graphs[1], axis = 0)
#             DeDi = np.abs(degs_two - degs_one)
#             DeDi_ranked_nodes = np.flip(np.argsort(DeDi))


#             n2v2r_scores, _ = evaluate_DBSM_ranking(first_true_memb_matrix, second_true_memb_matrix, n2v_ranked_nodes)
#             DeDi_scores, _ = evaluate_DBSM_ranking(first_true_memb_matrix, second_true_memb_matrix, DeDi_ranked_nodes)
#             n2v2r_scores_borda, _ = evaluate_DBSM_ranking(first_true_memb_matrix, second_true_memb_matrix, n2v_ranked_nodes_borda)

#             all_DeDi_scores.append(DeDi_scores)
#             all_n2v2r_scores.append(n2v2r_scores)
#             all_n2v2r_scores_borda.append(n2v2r_scores_borda)

#     return all_DeDi_scores, all_n2v2r_scores, all_n2v2r_scores_borda


# def prerank_gsea(ranking_pd, library_fn, one_sided=True, prerank_padj_cutoff=0.25, prerank_min_path_size=5, prerank_max_path_size=1500, prerank_num_perms=1000, prerank_weight=0, num_threads=4):
#     aggregate_count_dict = defaultdict(int)
#     aggregate_padj_dict = defaultdict(float)
#     aggregate_NES_dict = defaultdict(float)

#     results_found = 0
#     for (column_name, column_data) in ranking_pd.iteritems():
#         ranking_pd = pd.DataFrame(
#             column_data, index=ranking_pd.index.to_list())

#         pre_res = gseapy.prerank(rnk=ranking_pd,  # or rnk = rnk,
#                                  gene_sets=library_fn,
#                                  threads=num_threads,
#                                  min_size=prerank_min_path_size,
#                                  max_size=prerank_max_path_size,
#                                  permutation_num=prerank_num_perms,  # reduce number to speed up testing
#                                  outdir=None,  # don't write to disk
#                                  seed=6,
#                                  verbose=False,  # see what's going on behind the scenes
#                                  weighted_score_type=prerank_weight
#                                  )

#         if one_sided:
#             filtered_prerank = pre_res.res2d.loc[(
#                 pre_res.res2d['FDR q-val'] <= prerank_padj_cutoff) & (pre_res.res2d["NES"] > 0)][['Term', 'NES', 'FDR q-val']]
#         else:
#             filtered_prerank = pre_res.res2d.loc[(
#                 pre_res.res2d['FDR q-val'] <= prerank_padj_cutoff)][['Term', 'NES', 'FDR q-val']]

#         if not filtered_prerank.empty:
#             print(
#                 f'combo: {column_name} with {len(filtered_prerank.index)} found')
#             results_found += 1

#         for _, row in filtered_prerank.iterrows():
#             term = row['Term']
#             padj = row['FDR q-val']
#             nes = row['NES']
#             aggregate_count_dict[term] += 1
#             aggregate_padj_dict[term] += padj
#             aggregate_NES_dict[term] += nes

#     aggregate_prerank_pd = pd.DataFrame(
#         aggregate_count_dict.items(), columns=['pathway', 'freq'])
#     aggregate_prerank_pd['mean_padj'] = [v/aggregate_count_dict[k]
#                                          for (k, v) in aggregate_padj_dict.items()]
#     aggregate_prerank_pd['mean_NES'] = [v/aggregate_count_dict[k]
#                                         for (k, v) in aggregate_NES_dict.items()]
#     aggregate_prerank_pd['freq'] = [aggregate_prerank_pd.iloc[i, 1] /
#                                     results_found for i in range(len(aggregate_prerank_pd.index))]

#     return aggregate_prerank_pd


# def enrich_gsea(ranking_pd, library_fn, background, enrich_padj_cutoff=0.1, enrich_quantile_cutoff=0.9, organism='human'):
#     aggregate_count_dict = defaultdict(int)
#     aggregate_sum_padj_dict = defaultdict(float)

#     results_found = 0
#     for (column_name, column_data) in ranking_pd.iteritems():
#         ranking_pd = pd.DataFrame(
#             column_data, index=ranking_pd.index.to_list())
#         top_cutoff = ranking_pd[column_name].quantile(enrich_quantile_cutoff)

#         ind_keep = np.where(column_data >= top_cutoff)[0]
#         top_genes = ranking_pd.iloc[ind_keep, :].index.to_list()

#         enr = gseapy.enrichr(gene_list=top_genes,
#                              gene_sets=library_fn,
#                              background=background,
#                              organism=organism
#                              )

#         filtered_enr = enr.res2d.loc[(
#             enr.res2d['Adjusted P-value'] <= enrich_padj_cutoff)][['Term', 'Adjusted P-value']]

#         if not filtered_enr.empty:
#             print(f'combo: {column_name} with {len(filtered_enr.index)} found')
#             results_found += 1

#         for _, row in filtered_enr.iterrows():
#             term = row['Term']
#             padj = row['Adjusted P-value']
#             aggregate_count_dict[term] += 1
#             aggregate_sum_padj_dict[term] += padj

#     aggregate_enr_pd = pd.DataFrame(
#         aggregate_count_dict.items(), columns=['pathway', 'freq'])
#     aggregate_enr_pd['mean_padj'] = [v/aggregate_count_dict[k]
#                                      for (k, v) in aggregate_sum_padj_dict.items()]
#     aggregate_enr_pd['stability'] = [aggregate_enr_pd.iloc[i, 1] /
#                                 results_found for i in range(len(aggregate_enr_pd.index))]

#     return aggregate_enr_pd

def prerank_gseapy(ranking_pd, library_fn, one_sided=True, padj_cutoff=0.25, prerank_min_path_size=5, prerank_max_path_size=1500, prerank_num_perms=1000, prerank_weight=0, num_threads=4):
    aggregate_count_dict = defaultdict(int)
    aggregate_padj_dict = defaultdict(float)
    aggregate_NES_dict = defaultdict(float)
    aggregate_overlap = defaultdict(float)

    results_found = 0
    for (column_name, column_data) in ranking_pd.iteritems():
        column_pd = pd.DataFrame(
            column_data, index=ranking_pd.index.to_list())

        pre_res = gseapy.prerank(rnk=column_pd,  # or rnk = rnk,
                                 gene_sets=library_fn,
                                 threads=num_threads,
                                 min_size=prerank_min_path_size,
                                 max_size=prerank_max_path_size,
                                 permutation_num=prerank_num_perms,  # reduce number to speed up testing
                                 outdir=None,  # don't write to disk
                                 seed=6,
                                 verbose=False,  # see what's going on behind the scenes
                                 weighted_score_type=prerank_weight,
                                 no_plot=True
                                 )

        if one_sided:
            filtered_prerank = pre_res.res2d.loc[(pre_res.res2d["NES"] > 0) & (pre_res.res2d['FDR q-val'] <= padj_cutoff)][[
                'Term', 'NES', 'FDR q-val', 'Gene %']]
        else:
            filtered_prerank = pre_res.res2d.loc[pre_res.res2d['FDR q-val'] <= padj_cutoff][[
                'Term', 'NES', 'FDR q-val', 'Gene %']]

        if not filtered_prerank.empty:
            results_found += 1

            for _, row in filtered_prerank.iterrows():
                term = row['Term']
                padj = row['FDR q-val']
                nes = row['NES']
                overlap = row['Gene %'][:-1]
                aggregate_count_dict[term] += 1
                aggregate_padj_dict[term] += padj
                aggregate_NES_dict[term] += nes
                aggregate_overlap[term] += float(overlap)

    aggregate_prerank_pd = pd.DataFrame(
        aggregate_count_dict.items(), columns=['pathway', 'freq'], index=aggregate_count_dict.keys())
    aggregate_prerank_pd['padj'] = [v/aggregate_count_dict[k]
                                    for (k, v) in aggregate_padj_dict.items()]
    aggregate_prerank_pd['NES'] = [v/aggregate_count_dict[k]
                                   for (k, v) in aggregate_NES_dict.items()]
    aggregate_prerank_pd['stability'] = [aggregate_prerank_pd.iloc[i, 1] /
                                         len(ranking_pd.columns) for i in range(len(aggregate_prerank_pd.index))]

    aggregate_prerank_pd['overlap'] = [v/aggregate_count_dict[k]
                                       for (k, v) in aggregate_overlap.items()]

    aggregate_prerank_pd.sort_values(
        by=['padj', 'stability'], ascending=False, inplace=True)

    return aggregate_prerank_pd


def enrichr_gseapy(ranking_pd, library_fn, background, padj_cutoff=0.1, enrich_quantile_cutoff=0.9, organism='human'):
    aggregate_count_dict = defaultdict(int)
    aggregate_padj_dict = defaultdict(float)
    aggregate_overlap = defaultdict(float)

    results_found = 0
    for (column_name, column_data) in ranking_pd.iteritems():
        column_pd = pd.DataFrame(
            np.abs(column_data), index=ranking_pd.index.to_list())
        top_cutoff = column_pd[column_name].quantile(enrich_quantile_cutoff)

        ind_keep = np.where(column_data >= top_cutoff)[0]
        top_genes = column_pd.iloc[ind_keep, :].index.to_list()

        enr = gseapy.enrichr(gene_list=top_genes,
                             gene_sets=library_fn,
                             background=background,
                             organism=organism,
                             no_plot=True
                             )
        filtered_enr = enr.res2d.loc[enr.res2d['Adjusted P-value']
                                     <= padj_cutoff][['Term', 'Adjusted P-value', 'Overlap']]

        if not filtered_enr.empty:
            results_found += 1

            for _, row in filtered_enr.iterrows():
                term = row['Term']
                padj = row['Adjusted P-value']
                overlap = row['Overlap']
                aggregate_count_dict[term] += 1
                aggregate_padj_dict[term] += padj
                aggregate_overlap[term] += float(overlap.split("/")[0]) / \
                    float(overlap.split("/")[1])

    aggregate_enr_pd = pd.DataFrame(
        aggregate_count_dict.items(), columns=['pathway', 'freq'], index=aggregate_count_dict.keys())

    aggregate_enr_pd['padj'] = [v/aggregate_count_dict[k]
                                for (k, v) in aggregate_padj_dict.items()]
    aggregate_enr_pd['stability'] = [aggregate_enr_pd.iloc[i, 1] /
                                     float(len(ranking_pd.columns)) for i in range(len(aggregate_enr_pd.index))]
    aggregate_enr_pd['overlap'] = [v/aggregate_count_dict[k]
                                   for (k, v) in aggregate_overlap.items()]

    aggregate_enr_pd.sort_values(
        by=['padj'], ascending=True, inplace=True)
    return aggregate_enr_pd


def plot_gseapy_enrich(ranking, title='enrichr', topk=25, padj_cutoff=0.1, stability_cutoff=0.5, has_stability=False, characters_trim=50, trim_first_num_characters=0, output_dir=None):
    ranking_copy = ranking.copy()

    ranking_copy['pathway'] = ranking_copy['pathway'].str[trim_first_num_characters:]

    ranking_copy['pathway'] = ranking_copy['pathway'].str[:characters_trim]

    ranking_copy = ranking_copy.loc[(ranking_copy['padj'] <= padj_cutoff) & (
        ranking_copy['stability'] >= stability_cutoff)]

    num_results = len(ranking_copy.index)

    if num_results == 0:
        print(f"No results found for {title}")
        return

    ranking_copy['-log padj'] = - \
        np.log10(ranking_copy['padj'].to_numpy()+np.finfo(float).eps)
    ranking_copy.sort_values(by=['-log padj'],
                             ascending=True, inplace=True)

    if topk > num_results:
        topk = num_results

    if has_stability:
        fig = px.scatter(ranking_copy.iloc[-topk:, :], x="-log padj", y="pathway", color='stability', size='overlap',
                         title=title, color_continuous_scale='BuGn', range_color=[ranking_copy['stability'].min(), ranking_copy['stability'].max()]
                         )
    else:
        fig = px.scatter(ranking_copy.iloc[-topk:, :], x="-log padj", y="pathway", size='overlap',
                         title=title, color_discrete_sequence=['green']
                         )

    fig.update_layout(
        autosize=True,
        width=800,
        height=800,)

    fig.show()

    if output_dir:
        filename = '-'.join(title.split(" "))
        pio.write_image(fig, os.path.join(output_dir, filename+".pdf"))


def plot_gseapy_prerank(ranking, title='prerank', one_sided=True, topk=25, padj_cutoff=0.25, stability_cutoff=0, has_stability=False, characters_trim=50, trim_first_num_characters=0, output_dir=None):
    ranking_copy = ranking.copy()

    if one_sided:
        ranking_copy = ranking_copy.loc[ranking_copy['NES'] >= 0]

    ranking_copy['pathway'] = ranking_copy['pathway'].str[trim_first_num_characters:]

    ranking_copy['pathway'] = ranking_copy['pathway'].str[:characters_trim]

    ranking_copy = ranking_copy.loc[(ranking_copy['padj'] <= padj_cutoff) & (
        ranking_copy['stability'] >= stability_cutoff)]

    num_results = len(ranking_copy.index)

    if num_results == 0:
        print(f"No results found for {title}")
        return

    if topk > num_results:
        topk = num_results

    ranking_copy['-log padj'] = - \
        np.log10(ranking_copy['padj'].to_numpy()+np.finfo(float).eps)
    ranking_copy.sort_values(by=['-log padj'],
                             ascending=True, inplace=True)

    ranking_copy['-log padj'] = ranking_copy['-log padj'].round(2)

    if has_stability:
        if one_sided:
            fig = px.scatter(ranking_copy.iloc[-topk:, :], x='NES', y="pathway", color='stability', size='overlap',
                             title=title, color_continuous_scale='BuGn', range_color=[ranking_copy['stability'].min(), ranking_copy['stability'].max()]
                             )
            fig.add_annotation(
                x=ranking_copy.loc[ranking_copy.index[-1],
                                   'NES'], y=ranking_copy.loc[ranking_copy.index[-1], 'pathway'],
                text='-log padj ' +
                str(round(
                    ranking_copy.loc[ranking_copy.index[-1], '-log padj'], 2)),
                showarrow=True,
                yshift=10
            )
            fig.add_annotation(
                x=ranking_copy.loc[ranking_copy.index[-topk],
                                   'NES'], y=ranking_copy.loc[ranking_copy.index[-topk], 'pathway'],
                text='-log padj ' +
                str(round(
                    ranking_copy.loc[ranking_copy.index[-topk], '-log padj'], 2)),
                showarrow=True,
                yshift=10
            )
        else:
            biggest_nes_value = ceil(np.max(
                np.abs(ranking_copy.iloc[-topk:, :]['NES'])))
            fig = px.scatter(ranking_copy.iloc[-topk:, :], x='NES', y="pathway", color='stability', size='overlap',
                             title=title, color_continuous_scale='BuGn', range_x=[-biggest_nes_value-0.2, biggest_nes_value+0.2], range_color=[ranking_copy['stability'].min(), ranking_copy['stability'].max()]
                             )
            fig.add_annotation(
                x=ranking_copy.loc[ranking_copy.index[-1],
                                   'NES'], y=ranking_copy.loc[ranking_copy.index[-1], 'pathway'],
                text='-log padj ' +
                str(round(
                    ranking_copy.loc[ranking_copy.index[-1], '-log padj'], 2)),
                showarrow=True,
                yshift=10
            )
            fig.add_annotation(
                x=ranking_copy.loc[ranking_copy.index[-topk],
                                   'NES'], y=ranking_copy.loc[ranking_copy.index[-topk], 'pathway'],
                text='-log padj ' +
                str(round(
                    ranking_copy.loc[ranking_copy.index[-topk], '-log padj'], 2)),
                showarrow=True,
                yshift=10
            )
    else:
        if one_sided:
            fig = px.scatter(ranking_copy.iloc[-topk:, :], x='NES', y="pathway", color='-log padj', size='overlap',
                             title=title, color_continuous_scale='BuGn'
                             )
        else:
            biggest_nes_value = ceil(np.max(
                np.abs(ranking_copy.iloc[-topk:, :]['NES'])))
            fig = px.scatter(ranking_copy.iloc[-topk:, :], x='NES', y="pathway", color='-log padj', size='overlap',
                             title=title, color_continuous_scale='BuGn', range_x=[-biggest_nes_value-0.2, biggest_nes_value+0.2]
                             )

    fig.update_layout(
        autosize=True,
        width=800,
        height=800,)

    fig.show()

    if output_dir:
        filename = '-'.join(title.split(" "))
        pio.write_image(fig, os.path.join(output_dir, filename+".pdf"))
