

import sys
import os


sys.path.insert(1, os.path.realpath(os.path.pardir))
sys.path.append("../node2vec2rank/")
import itertools
import scipy
import json
import argparse
from model import N2V2R
import spectral_embedding as se
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib as mpl
import random
import csv


from post_utils import discounted_cumulative_gain, normalized_discounted_cumulative_gain



def replace_with_bernoulli(matrix,  elements, value_to_check, probability):
    if elements.size == 0:
        return matrix
    # Create a boolean mask for cells that meet the condition
    condition_mask = (matrix[elements[:, 0], elements[:, 1]] == value_to_check)
    # print(condition_mask)

    # Generate random Bernoulli-distributed replacement values for the selected cells
    num_replacements_needed = np.sum(condition_mask)
    # print(f'inside replace using probs {1-probability} and {probability}')
    random_replacements = np.random.choice([0, 1], size=num_replacements_needed, p=[
                                           1 - probability, probability])
                                           
    # print(random_replacements)
    # Create an array of default values (original matrix elements) with the same shape as random_replacements
    default_values = matrix[elements[:, 0], elements[:, 1]].copy()

    # Modify the selected cells with random replacement values using the mask and random replacements
    default_values[condition_mask] = random_replacements
    matrix[elements[:, 0], elements[:, 1]] = default_values
    matrix[elements[:, 1], elements[:, 0]] = default_values

    return matrix

def update_adjacency_matrix(A, assignments, first_block, second_block):
    A_second = A.copy()
    num_comms = first_block.shape[0]
    for i in range(0, num_comms):
        for j in range(i, num_comms):
            where_one_comm = np.where(assignments == i)[0]
            where_other_comm = np.where(assignments == j)[0]

            elements = []
            for k in range(len(where_one_comm)):
                if i!=j:
                    for q in range(len(where_other_comm)):
                        elements.append([where_one_comm[k],where_other_comm[q]])
                else:
                    for q in range(k,len(where_other_comm)):
                        elements.append([where_one_comm[k],where_other_comm[q]])

            elements = [row for row in elements if len(set(row)) > 1]
            elements = np.asarray(elements)

            # elements = np.asarray(list(combinations_with_replacement(where_one_comm + where_other_comm, 2)))       
            # if probability higher in first graph
            if (first_block[i, j] > second_block[i, j]):
                # if there is an edge already, we reject
                replace_with_bernoulli(
                    A_second, elements, 1, second_block[i, j]/first_block[i, j])

            if (first_block[i, j] < second_block[i, j]):
                # if higher in second, we sample with second prob
                replace_with_bernoulli(
                    A_second, elements, 0, 1-(1-second_block[i, j])/(1-first_block[i, j]))
    return A_second

def create_symmetric_mixed_noise_matrix(matrix_size, mean=0, std=0.05, prior_binomial_p=0.5):
    symmetric_matrix = np.zeros((matrix_size, matrix_size))

    for i in range(1, matrix_size):
        for j in range(i, matrix_size):
            random_value = np.random.choice([0, np.random.normal(mean, std)], p=[
                                            1-prior_binomial_p, prior_binomial_p])
            if i == j:
                symmetric_matrix[i, j] = random_value
            else:
                symmetric_matrix[i, j] = random_value
                symmetric_matrix[j, i] = random_value
    return symmetric_matrix


def borda_sort(lists):
    scores = {}
    for l in lists:
        for idx, elem in enumerate(reversed(l)):
            if not elem in scores:
                scores[elem] = 0
            scores[elem] += idx
    return sorted(scores.keys(), key=lambda elem: scores[elem], reverse=True)


def vector_to_symmetric_matrix(vector, n):
    matrix = np.zeros((n, n))
    triu_indices = np.triu_indices(n)
    # Fill the upper triangular part
    matrix[triu_indices] = vector

    # Copy the upper triangular part to the lower triangular part to make the matrix symmetric
    matrix.T[triu_indices] = vector

    return matrix


def generate_two_DSBM_graphs_random_changes(num_nodes, num_com, min_comm_prob=0, max_comm_prob=0.5, uniform_node_comm_assignment=False, noise_gaussian_std=None, degree_naive=False, multiplicity_model=1):
    list_of_pairs_of_graphs = []
    list_of_assignments = []
    list_of_first_true_membs = []
    list_of_second_true_membs = []

    first_block_prob_matrix = vector_to_symmetric_matrix(np.random.uniform(
        min_comm_prob, max_comm_prob, int(num_com*(num_com+1)/2)), num_com)

    if noise_gaussian_std is None:
        noise_matrix = create_symmetric_mixed_noise_matrix(
            num_com, mean=0, std=(max_comm_prob-min_comm_prob)/4)
    else:
        noise_matrix = create_symmetric_mixed_noise_matrix(
            num_com, mean=0, std=noise_gaussian_std)

    if degree_naive is True:
        second_block_prob_matrix = first_block_prob_matrix.copy()
        second_block_prob_matrix[0, 1:] = second_block_prob_matrix[0, 1:][::-1]
        second_block_prob_matrix[1:, 0] = second_block_prob_matrix[1:, 0][::-1]
    else:
        second_block_prob_matrix = first_block_prob_matrix.copy()
        second_block_prob_matrix[0, :] = np.random.uniform(min_comm_prob, max_comm_prob, num_com)
        second_block_prob_matrix[:, 0] = second_block_prob_matrix[0, :]

    second_block_prob_matrix = np.add(second_block_prob_matrix, noise_matrix)
    second_block_prob_matrix = np.clip(
        second_block_prob_matrix, min_comm_prob, max_comm_prob)

    stacked_true_memb_matrix = np.stack(
        (first_block_prob_matrix, second_block_prob_matrix))

    # whether to sample node assignments from a uniform or a randomly generated multinomial
    if uniform_node_comm_assignment:
        pi = np.repeat(1/num_com, num_com)
    else:
        pi = np.random.random(num_com)
        pi /= pi.sum()

    for _ in range(multiplicity_model):
        # using the spectral embedding package to generate a graph and return assignments
        As, Z = se.generate_SBM_dynamic(
            num_nodes, stacked_true_memb_matrix, pi)

        colours = np.array(list(mpl.colors.TABLEAU_COLORS.keys())[0:num_com])
        Zcol = colours[Z]

        le = preprocessing.LabelEncoder()
        assignments = le.fit_transform(Zcol)

        # enforce at least one node per community
        if not set(np.arange(num_com)).issubset(assignments):
            continue

        As[1] = update_adjacency_matrix(
            As[0], assignments, first_block_prob_matrix, second_block_prob_matrix)

        list_of_pairs_of_graphs.append(As)
        list_of_assignments.append(assignments)
        list_of_first_true_membs.append(first_block_prob_matrix)
        list_of_second_true_membs.append(second_block_prob_matrix)

    return list_of_pairs_of_graphs, list_of_first_true_membs, list_of_second_true_membs, list_of_assignments


def evaluate_DBSM_ranking(first_true_memb_matrix, second_true_memb_matrix, ranking, assignments, degree_ground_truth=False):
    ranking = np.int16(ranking)
    assignments = np.int16(assignments) + 1
    num_communities = first_true_memb_matrix.shape[1]

    # community assignment for each node in the ranking
    ranking_com_mapped = assignments[ranking].copy()

    # rank the communities according to how much they changed
    Ds_l2 = [scipy.spatial.distance.euclidean(row1, row2)
             for row1, row2 in zip(first_true_memb_matrix, second_true_memb_matrix)]

    Ds_cheb = [scipy.spatial.distance.chebyshev(row1, row2)
               for row1, row2 in zip(first_true_memb_matrix, second_true_memb_matrix)]

    Ds_l1 = [scipy.spatial.distance.cityblock(row1, row2)
             for row1, row2 in zip(first_true_memb_matrix, second_true_memb_matrix)]

    Ds_cosine = [scipy.spatial.distance.cosine(row1, row2)
                 for row1, row2 in zip(first_true_memb_matrix, second_true_memb_matrix)]

    Ds_dedi = np.abs(
        np.sum(np.subtract(first_true_memb_matrix, second_true_memb_matrix), axis=1))

    ordered_communities_by_l2 = np.flip(np.argsort(Ds_l2)) + 1
    ordered_communities_by_dedi = np.flip(np.argsort(Ds_dedi)) + 1
    ordered_communities_by_cos = np.flip(np.argsort(Ds_cosine)) + 1
    ordered_communities_by_l1 = np.flip(np.argsort(Ds_l1)) + 1
    ordered_communities_by_cheb = np.flip(np.argsort(Ds_cheb)) + 1

    borda_ranking = borda_sort([ordered_communities_by_l2, ordered_communities_by_cos,
                               ordered_communities_by_l1, ordered_communities_by_cheb])
    ordered_communities_by_borda = borda_ranking

    if degree_ground_truth:
        ground_truth_ranking = ordered_communities_by_dedi
    else:
        ground_truth_ranking = ordered_communities_by_borda

    # assign relevance k-i to the i-th ordered community out of k total
    rel_dict = {}
    for i in range(num_communities):
        rel_dict[ground_truth_ranking[i]] = num_communities - i

    ranking_mapped_to_relevances = []
    for i in range(ranking_com_mapped.size):
        relevance_of_node = rel_dict[ranking_com_mapped[i]]
        ranking_mapped_to_relevances.append(relevance_of_node)

    # get the dcg for the entire ranking (non binary relevance)
    ranking_mapped_to_relevances = np.int16(ranking_mapped_to_relevances)
    total_ndcg_score = discounted_cumulative_gain(ranking_mapped_to_relevances)

    binary_scores = []
    non_binary_scores = []
    non_binary_scores.append(total_ndcg_score)
    recalls = []

    # get the dcgs per ranked community
    prev_com_num_nodes = 0
    for i in range(num_communities):
        nodes_in_com = np.where(ranking_com_mapped ==
                                ground_truth_ranking[i])[0]
        num_nodes_in_com = nodes_in_com.shape[0]
        com_relevance = num_communities-i

        # chunk the relevance vector to where ideally they should be in order for the non binary case with dcg
        group_relevance_non_binary = ranking_mapped_to_relevances[prev_com_num_nodes:(
            prev_com_num_nodes+num_nodes_in_com)].copy()
        group_non_binary_score = discounted_cumulative_gain(
            group_relevance_non_binary)
        non_binary_scores.append(group_non_binary_score)

        # chunk the relevance vector to where ideally they should be in order for the binary case with ndcg
        group_relevance_binary = ranking_mapped_to_relevances[prev_com_num_nodes:(
            prev_com_num_nodes+num_nodes_in_com)].copy()
        group_relevance_binary[group_relevance_binary != com_relevance] = 0
        group_relevance_binary[group_relevance_binary != 0] = 1
        group_binary_score = normalized_discounted_cumulative_gain(
            group_relevance_binary, binary_relevance=True, denom_all_rel=True)
        binary_scores.append(group_binary_score)

        recalls.append(np.count_nonzero(
            group_relevance_binary)/len(group_relevance_binary))

        prev_com_num_nodes = prev_com_num_nodes + num_nodes_in_com

    return ranking_mapped_to_relevances, non_binary_scores, binary_scores, recalls, ground_truth_ranking[0]


def dbsm_experiment(num_nodes, num_com, num_samples, config_fn, min_comm_prob=0.0, max_comm_prob=0.5, uniform_node_comm_assignment=False, degree_naive=False, degree_ground_truth=False,   noise_gaussian_std=None, multiplicity_model=1, save_dir=None):
    config = json.load(open(config_fn, 'r'))

    config = {param: value for section, params in config.items()
              for param, value in params.items()}

    node_names = np.arange(num_nodes)

    all_n2v2r_non_binary_scores = []
    all_n2v2r_borda_non_binary_scores = []
    all_absDeDi_non_binary_scores = []

    all_n2v2r_binary_scores = []
    all_n2v2r_borda_binary_scores = []
    all_absDeDi_binary_scores = []

    all_n2v2r_recalls = []
    all_n2v2r_borda_recalls = []
    all_absDeDi_recalls = []

    while len(all_n2v2r_non_binary_scores) < multiplicity_model*num_samples:
        list_of_pairs_of_graphs, list_of_first_true_membs, list_of_second_true_membs, list_of_assignments = generate_two_DSBM_graphs_random_changes(
            num_nodes=num_nodes,  num_com=num_com, min_comm_prob=min_comm_prob, max_comm_prob=max_comm_prob, uniform_node_comm_assignment=uniform_node_comm_assignment, degree_naive=degree_naive,   noise_gaussian_std=noise_gaussian_std,  multiplicity_model=multiplicity_model)

        for j in range(len(list_of_pairs_of_graphs)):
            random_graphs = list_of_pairs_of_graphs[j]
            assignments = list_of_assignments[j]
            first_block_prob_matrix = list_of_first_true_membs[j]
            second_block_prob_matrix = list_of_second_true_membs[j]

            model = N2V2R(graphs=random_graphs, config=config,
                          node_names=node_names)
            rankings = model.fit_transform_rank()['1']
            n2v2r_ranking = rankings.iloc[:, 0]
            dedi_ranking = model.degree_difference_ranking()['1']

            model.pairwise_ranks['1'] = model.pairwise_ranks['1'].drop(model.pairwise_ranks['1'].columns[0], axis=1)

            borda_ranking = model.aggregate_transform()['1']

            n2v2r_ranked_nodes = n2v2r_ranking.sort_values(
                ascending=False, inplace=False).index.to_list()
            n2v2r_borda_ranking = borda_ranking.sort_values(
                by='borda_ranks', ascending=False, inplace=False).index.to_list()

            absDeDi_ranking = dedi_ranking.sort_values(
                by='absDeDi', ascending=False, inplace=False).index.to_list()

            _, n2v2r_non_binary_scores, n2v2r_binary_scores, n2v2r_recalls,_ = evaluate_DBSM_ranking(
                first_block_prob_matrix, second_block_prob_matrix, n2v2r_ranked_nodes, assignments, degree_ground_truth=degree_ground_truth)

            _, n2v2r_borda_non_binary_scores, n2v2r_borda_binary_scores, n2v2r_borda_recalls,_ = evaluate_DBSM_ranking(
                first_block_prob_matrix, second_block_prob_matrix, n2v2r_borda_ranking, assignments, degree_ground_truth=degree_ground_truth)

            _, absDeDi_non_binary_scores, absDeDi_binary_scores, absDeDi_recalls, most_changing = evaluate_DBSM_ranking(
                first_block_prob_matrix, second_block_prob_matrix, absDeDi_ranking, assignments, degree_ground_truth=degree_ground_truth)

            if most_changing!=1:
                continue

            all_n2v2r_non_binary_scores.append(n2v2r_non_binary_scores)

            all_n2v2r_borda_non_binary_scores.append(
                n2v2r_borda_non_binary_scores)
            all_absDeDi_non_binary_scores.append(absDeDi_non_binary_scores)

            all_n2v2r_binary_scores.append(n2v2r_binary_scores)
            all_n2v2r_borda_binary_scores.append(n2v2r_borda_binary_scores)
            all_absDeDi_binary_scores.append(absDeDi_binary_scores)

            all_n2v2r_recalls.append(n2v2r_recalls)
            all_n2v2r_borda_recalls.append(n2v2r_borda_recalls)
            all_absDeDi_recalls.append(absDeDi_recalls)

    num_samples = len(all_absDeDi_recalls)

    all_n2v2r_non_binary_scores_np = np.asarray(all_n2v2r_non_binary_scores)
    all_n2v2r_borda_non_binary_scores_np = np.asarray(
        all_n2v2r_borda_non_binary_scores)
    all_absDeDi_non_binary_scores_np = np.asarray(
        all_absDeDi_non_binary_scores)

    all_n2v2r_binary_scores_np = np.asarray(all_n2v2r_binary_scores)
    all_n2v2r_borda_binary_scores_np = np.asarray(
        all_n2v2r_borda_binary_scores)
    all_absDeDi_binary_scores_np = np.asarray(all_absDeDi_binary_scores)

    all_n2v2r_recalls_np = np.asarray(all_n2v2r_recalls)
    all_n2v2r_borda_recalls_np = np.asarray(all_n2v2r_borda_recalls)
    all_absDeDi_recalls_np = np.asarray(all_absDeDi_recalls)

    if save_dir:
        par_configuration = str(num_nodes) + "_" + str(num_com) + "_" + str(num_samples) + "_" + str(min_comm_prob) + "_" + str(max_comm_prob) + \
            "_" + "sampleuni-"+str(uniform_node_comm_assignment) + "_" + "_degn-" + str(
                degree_naive) + "_noisestd-"+str(noise_gaussian_std)
        save_dir = os.path.join(
            save_dir, par_configuration)
        os.makedirs(save_dir)

        method = 'n2v2r'
        np.savetxt(save_dir+"/"+method + "_non_binary_scores.csv",
                   all_n2v2r_non_binary_scores_np)
        np.savetxt(save_dir+"/"+method + "_binary_scores.csv",
                   all_n2v2r_binary_scores_np)
        np.savetxt(save_dir+"/"+method + "_recalls.csv", all_n2v2r_recalls_np)

        method = 'n2v2r_borda'
        np.savetxt(save_dir+"/"+method + "_non_binary_scores.csv",
                   all_n2v2r_borda_non_binary_scores_np)
        np.savetxt(save_dir+"/"+method + "_binary_scores.csv",
                   all_n2v2r_borda_binary_scores_np)
        np.savetxt(save_dir+"/"+method + "_recalls.csv",
                   all_n2v2r_borda_recalls_np)

        method = 'absDeDi'
        np.savetxt(save_dir+"/"+method + "_non_binary_scores.csv",
                   all_absDeDi_non_binary_scores_np)
        np.savetxt(save_dir+"/"+method + "_binary_scores.csv",
                   all_absDeDi_binary_scores_np)
        np.savetxt(save_dir+"/"+method + "_recalls.csv",
                   all_absDeDi_recalls_np)


def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError  # evil ValueError that doesn't tell you what the wrong value was


def main():
    parser = argparse.ArgumentParser(description='asdas')

    # Add command line arguments here
    parser.add_argument('-num_nodes', help='Number of nodes', required=True)
    parser.add_argument(
        '-num_com', help='Number of communities', required=True)
    parser.add_argument(
        '-num_samples', help='Number of samples', required=True)
    parser.add_argument(
        '-n2v2r_config', help='Config file for n2v2r', required=True)
    parser.add_argument(
        '-degree_naive', help='Whether the simulation is degree-naive or not', required=True)
    parser.add_argument(
        '-sample_uni', help='Whether uniform community assignment or not', required=True)
    parser.add_argument(
        '-noise_denominator', help='Regulates the std for the Gaussian noise of 0 mean by dividing the [0, 0.5] span', default=None)

    parser.add_argument(
        '-degree_ground_truth', help='Whether ground truth is based on absolute degree difference or an aggregate of distance metrics', default='False')

    args = parser.parse_args()
    # Access the values of the arguments
    num_nodes = int(args.num_nodes)
    num_com = int(args.num_com)
    n2v2r_config = args.n2v2r_config
    degree_naive = str_to_bool(args.degree_naive)
    num_samples = int(args.num_samples)
    sample_uni = str_to_bool(args.sample_uni)
    noise_std = 0.5/float(args.noise_denominator)

    degree_ground_truth = str_to_bool(args.degree_ground_truth)

    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    multiplicity_model = 1

    save_dir = "../results/vs_dedi_new_paper_params_cor_emb"

    dbsm_experiment(num_nodes, num_com, num_samples=num_samples, config_fn=n2v2r_config, min_comm_prob=0.0, max_comm_prob=0.5, uniform_node_comm_assignment=sample_uni,
                    degree_naive=degree_naive, degree_ground_truth=degree_ground_truth,   noise_gaussian_std=noise_std, multiplicity_model=multiplicity_model, save_dir=save_dir)


if __name__ == "__main__":
    main()
