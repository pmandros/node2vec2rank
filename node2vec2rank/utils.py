from distutils.log import error
import numpy as np
import pandas as pd
import pickle
import os
import scipy.spatial.distance
from joblib import Parallel, delayed
import plotly.graph_objects as go

"""
Computes the projections W^TW if on_right is True, or WW^T otherwise 
for some mXn bipartite matrix W, e.g., a TFxGENE adjacency matrix.
on_right: Whether to project right W^TW or left WW^T
returns A symmetric matrix corresponding to the projection
"""


def bipartite_to_unipartite_projection(grn, on_right=True):
    [r, c] = grn.shape

    assert (r != c, 'Graph is not bipartite')

    if (on_right):
        to_return = np.matmul(np.transpose(grn), grn)
    else:
        to_return = np.matmul(grn, np.transpose(grn))

    [r, c] = np.shape(to_return)
    assert (r == c, 'Graph should be square')

    return to_return


"""
Transforms a mXn bipartite matrix into a (m+n)X(m+n) symmetric matrix by zero-padding 
for representing bipartite matrices as symmetric adjacency matrices.
matrix: An mXn bipartite matrix to symmetrize
returns A symmetric matrix 
"""


def symmetrize_matrix(matrix):
    [r, c] = np.shape(matrix)
    if (r == c):
        return matrix
    else:
        print(f"Size of matrix to be symmetrized {r}x{c}")

        if isinstance(matrix, (np.ndarray, np.generic)):
            number_of_nodes = r+c
            sym_matrix = np.zeros((number_of_nodes, number_of_nodes))
            sym_matrix[:r, r:] = matrix
            transposed = np.transpose(matrix)
            sym_matrix[r:, :r] = transposed
            # return (sym_matrix)
        # elif isinstance(matrix, pd.DataFrame):
        #     row_index_ls = list(matrix.index.values)
        #     col_index_ls = list(matrix.columns.values)
        #     row_index_ls.extend(col_index_ls)

        #     matrix_np = matrix.to_numpy()
        #     sym_matrix = symmetrize_matrix(matrix_np)
        #     sym_matrix = pd.DataFrame(
        #         sym_matrix, index=row_index_ls, columns=row_index_ls)
        #     sym_matrix.reindex(sym_matrix.columns)
        return (sym_matrix)


"""
Computes various matrix transformations for pre-processing in a certain sequence. Use with caution.
threshold: below this everything is set to 0 (second operation to be applied)
top_percent_keep: keep the top percent of edges according to their weisght (fourth operation to be applied)
binarize: turn all non-zero elements into 1 (last operation to be applied)
symmetrize: represent bipartite graph as symmetric adjacency matrix by zero padding (third oprtation to be applied)
absolute: compute the absolute of the matrix (first operation to be applied)
project_unipartite: project the mXn matrix into a mXm (left) or nXn (right) matrix (third oprtation to be applied)
on_right: whether to project an mXn matrix into a mXm (left) or nXn (right) matrix
returns A transformed and preprocessed symmetric matrix 
"""


def network_transform(network, threshold=None, top_percent_keep=100, binarize=False, symmetrize=True, absolute=False, project_unipartite=False, on_right=True):
    [r, c] = np.shape(network)

    if absolute:
        network = np.abs(network)

    if threshold is not None:
        network[network < threshold] = 0

    if (r != c) and symmetrize and (not project_unipartite):
        network = symmetrize_matrix(network)
    elif (r != c) and project_unipartite:
        network = bipartite_to_unipartite_projection(
            network, on_right=on_right)

    cut_off = np.percentile(network, 100-top_percent_keep)
    network[network < cut_off] = 0

    # binarize
    if binarize:
        network[network != 0] = 1

    return np.float32(network)

# # remove nodes that have no edges
# # if two networks are passed, then their intersection of genes is used
# def get_non_zero_degree_graph(network, second_network=None):
#     [r,c] = np.shape(network)
#     if(r==c):
#         print("Graphs symmetric")
#     else:
#         error("Graphs not symmetric. This should not happen.")

#     diagonal = np.diagonal(network).copy()
#     row_sum = np.sum(network,axis=0)

#     idx_sum_row_eq_diagonal = np.where(row_sum==diagonal)[0]

#     if(second_network is not None):

#         second_diagonal = np.diagonal(second_network).copy()
#         second_row_sum = np.sum(second_network, axis=0)

#         second_idx_sum_row_eq_diagonal = np.where(second_row_sum==second_diagonal)[0]

#         idx_sum_row_eq_diagonal = np.intersect1d(idx_sum_row_eq_diagonal, second_idx_sum_row_eq_diagonal)
#         second_network=np.delete(second_network,idx_sum_row_eq_diagonal, 0)
#         second_network=np.delete(second_network,idx_sum_row_eq_diagonal, 1)

#     print(f'There are {np.size(idx_sum_row_eq_diagonal)} genes with 0 degree')

#     network=np.delete(network, idx_sum_row_eq_diagonal,0)
#     network=np.delete(network, idx_sum_row_eq_diagonal,1)

#     if(second_network is not None):
#         return network, second_network, idx_sum_row_eq_diagonal
#     else:
#         return network, idx_sum_row_eq_diagonal

# # removes disconnected nodes
# # if two networks, it finds the intersection of common connected genes
# def trim_networks(network, second_network=None):
#     diagonal = np.diagonal(network).copy()
#     row_sum = np.sum(network,axis=0)

#     idx_sum_row_eq_diagonal = np.where(row_sum==diagonal)[0]

#     if(second_network is not None):
#         second_diagonal = np.diagonal(second_network).copy()
#         second_row_sum = np.sum(second_network, axis=0)

#         second_idx_sum_row_eq_diagonal = np.where(second_row_sum==second_diagonal)[0]

#         idx_sum_row_eq_diagonal = np.intersect1d(idx_sum_row_eq_diagonal, second_idx_sum_row_eq_diagonal)
#         second_network=np.delete(second_network,idx_sum_row_eq_diagonal, 0)
#         second_network=np.delete(second_network,idx_sum_row_eq_diagonal, 1)

#     print(f'There are {np.size(idx_sum_row_eq_diagonal)} nodes with 0 degree')

#     network=np.delete(network, idx_sum_row_eq_diagonal,0)
#     network=np.delete(network, idx_sum_row_eq_diagonal,1)

#     if(second_network is not None):
#         return network, second_network, idx_sum_row_eq_diagonal
#     else:
#         return network, idx_sum_row_eq_diagonal


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


"""
Join all rankings and compute average pairwise distances for all nodes across trials
rankings: rankings of all nodes across several trials 
save_dir: path to save the average rankings (Optional)
agg: aggregation operations to be done on the data (Default: average and standard deviation)
returns Dataframe of nodes names and average distance values sorted by similarity in decreasing order
"""


def join_aggregate_ranking(rankings, agg=None, save_dir=None):
    joined = pd.DataFrame()
    for i, df in enumerate(rankings):
        if i == 0:
            joined = df.copy()
            continue
        joined = joined.set_index("Node").join(df.set_index(
            "Node"), lsuffix=f"_{i-1}", rsuffix=f"_{i}").reset_index()

    dist_columns = [col for col in joined.columns if "Dist" in col]

    if "avg" in agg:
        joined["Avg"] = joined[dist_columns].mean(axis=1)
        joined = joined.sort_values(by="Avg").copy()
    if "std" in agg:
        joined["Std"] = joined[dist_columns].std(axis=1)
    if "sum" in agg:
        joined["Sum"] = joined[dist_columns].sum(axis=1)

    if save_dir:
        joined.to_csv(os.path.join(save_dir, "average_rankings.csv"))
    return joined


"""
save experiment data as a pickle
"""


def save_experiment(data, path):
    file = open(path, 'wb')
    pickle.dump(data, file)
    file.close()


"""
load experiement data from pickle
"""


def load_experiment(path):
    file = open(path, 'rb')
    data = pickle.load(file)
    file.close()
    return data


"""
Add a message to a log file, also prettify any dictionary into user-friendly format when writing it to the log file
"""


def logger(message, save_dir, selected_keys=None):
    with open(os.path.join(save_dir, "simulation_results.log"), "a") as file:
        if isinstance(message, str):
            print(message)
            file.write(message+"\n")
        elif isinstance(message, dict):
            if selected_keys is None:
                message = {key: val for key, val in message.items(
                ) if isinstance(val, int) or isinstance(val, float)}
            else:
                message = {key: val for key,
                           val in message.items() if key in selected_keys}
            key_width = max(len(key) for key in message.keys())
            keys_row = "\t".join("{:<{width}}".format(
                key, width=key_width) for key in message.keys())
            values_row = "\t".join("{:<{width}}".format(
                value, width=key_width) for value in message.values())

            print(keys_row)
            print(values_row)
            file.write(keys_row+"\n")
            file.write(values_row+"\n")
