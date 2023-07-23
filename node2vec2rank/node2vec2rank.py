import json
import os
import argparse
from dataloader import DataLoader
from model import n2v2r
from model import degree_difference_ranking

# Create the parser
parser = argparse.ArgumentParser(description='Script arguments')
parser.add_argument("--config", default=None, help='Configuration file path')
# Add data_loading arguments
data_loading_group = parser.add_argument_group('data_io')
data_loading_group.add_argument(
    '--save_dir', type=str, help='Save directory')
data_loading_group.add_argument(
    '--graph_filenames', nargs='+', type=str, help='Graph filenames')
data_loading_group.add_argument(
    '--data_dir', type=str, help='Data Directory')
data_loading_group.add_argument(
    '--seperator', default='\t', type=str, help='Separator')
data_loading_group.add_argument(
    '--is_edge_list', action='store_true', help='Whether the input is an edge list')
data_loading_group.add_argument(
    '--transpose', action='store_true', help='whether to transpose the adjacency matrix or not')

# Add data_preprocessing arguments
data_preprocessing_group = parser.add_argument_group('data_preprocessing')
data_preprocessing_group.add_argument(
    '--project_unipartite_on', default='columns', type=str, help='Project unipartite')
data_preprocessing_group.add_argument(
    '--threshold', type=float, default=0, help='Threshold value')
data_preprocessing_group.add_argument(
    '--top_percent_keep', nargs='+', type=int, default=[100, 75], help='Top percentage to keep')
data_preprocessing_group.add_argument(
    '--binarize', nargs='+', type=bool, default=[False, True], help='Whether to binarize the data')
data_preprocessing_group.add_argument(
    '--absolute', action='store_true', help='Take the absolute value')

# Add fitting_ranking arguments
fitting_ranking_group = parser.add_argument_group('fitting_ranking')
fitting_ranking_group.add_argument(
    '--embed_dimensions', nargs='+', type=int, default=[2, 4, 8, 16], help='Embed dimensions')
fitting_ranking_group.add_argument(
    '--distance_metrics', nargs='+', default=["euclidean", "cosine"], help='Distance metrics')
fitting_ranking_group.add_argument(
    '--seed', type=int, default=None, help='Random seed')
fitting_ranking_group.add_argument(
    '--verbose', type=int, default=0, help='Verbose level')

# Parse the arguments from the command line
config = parser.parse_args()

# user should provide path of config file (all other args will be ignored and will be extracted from the file)
if config.config is not None:
    with open(config.config, 'r', encoding='utf-8') as file:
        config = json.load(file)
        config = {param: value for _, params in config.items()
                  for param, value in params.items()}
# if no config file, these args are required
elif any([config.data_dir is None, config.graph_filenames is None,
          config.save_dir is None]):
    print("The following arguments are required: --save_dir, --graph_filenames, --data_dir")
    parser.print_help()
    exit(1)


# create dataloader and load the graphs in memory
dataloader = DataLoader(config=config)
graphs = dataloader.get_graphs()
interest_nodes = dataloader.get_interest_nodes()
print(interest_nodes)

# compute DeDi ranking
DeDi_ranking = degree_difference_ranking(
    graphs=graphs, node_names=interest_nodes)
prior_singed_ranks = [v.iloc[:, 0] for k, v in DeDi_ranking.items()]

# define and train the Node2Vec2Rank
model = n2v2r(graphs=graphs, config=config, node_names=interest_nodes)
rankings = model.fit_transform_rank()

# generate ranking based on borda ranking
borda_rankings = model.aggregate_transform()

# get signed ranking based on dedi ranks
signed_rankings = model.signed_ranks_transform(
    prior_signed_ranks=prior_singed_ranks)
