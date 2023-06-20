import json
import os
import argparse
from node2vec2rank.model_old import Model2Rank
from node2vec2rank.multimodel import MultiModel2Rank
from node2vec2rank.dataloader import DataLoader


# Create the parser
parser = argparse.ArgumentParser(description='Script arguments')

# Add data_loading arguments
data_loading_group = parser.add_argument_group('data_loading')
data_loading_group.add_argument(
    '--save_dir', default=None, help='Save directory')
data_loading_group.add_argument(
    '--network_filenames', nargs='+', required=True, help='Network filenames')
data_loading_group.add_argument('--seperator', default=',', help='Separator')
data_loading_group.add_argument(
    '--is_edge_list', action='store_true', help='Whether the input is an edge list')

# Add data_preprocessing arguments
data_preprocessing_group = parser.add_argument_group('data_preprocessing')
data_preprocessing_group.add_argument(
    '--project_unipartite', action='store_true', help='Project unipartite')
data_preprocessing_group.add_argument(
    '--threshold', type=float, default=0, help='Threshold value')
data_preprocessing_group.add_argument(
    '--top_percent_keep', nargs='+', type=int, default=[100], help='Top percentage to keep')
data_preprocessing_group.add_argument(
    '--binarize', nargs='+', type=bool, default=[False], help='Whether to binarize the data')
data_preprocessing_group.add_argument(
    '--absolute', action='store_true', help='Take the absolute value')

# Add fitting_ranking arguments
fitting_ranking_group = parser.add_argument_group('fitting_ranking')
fitting_ranking_group.add_argument(
    '--embed_dimensions', nargs='+', type=int, default=[2, 4, 8, 16], help='Embed dimensions')
fitting_ranking_group.add_argument('--distance_metrics', nargs='+', default=[
                                   'euclidean', 'cosine', 'chebyshev'], help='Distance metrics')
fitting_ranking_group.add_argument(
    '--seed', type=int, default=None, help='Random seed')
fitting_ranking_group.add_argument(
    '--verbose', type=int, default=0, help='Verbose level')

# Parse the arguments
config = parser.parse_args()
if all(value is None for value in vars(config).values()):
    # read the config file and create output file it doesn't exist
    config = json.load(open('config.json', 'r'))

if config is None:
    print("No arguments provided. Please provide the required command line arguments.")
    parser.print_help()
    exit(1)

# collapse arguments in one dictionary
config = {param: value for section, params in config.items()
          for param, value in params.items()}

os.makedirs(config["main_save_dir"], exist_ok=True)

# create dataloader and load the graphs in memory
dataloader = DataLoader(config=config)
graphs = dataloader.get_graphs()
node_dict = dataloader.get_id2node()

# check if there are multiple values for any given parameter
multi_params = [key for key in config.keys() if isinstance(
    config[key], list) and len(config[key]) > 1]
if not multi_params:
    print("Model2Rank")
    Model2Rank(graphs=graphs, config=config,
               node_names=node_dict).walk_fit_rank()
else:
    print('MultiModel2Rank')
    MultiModel2Rank(graphs=graphs, config=config,
                    node_dict=node_dict).walk_fit_rank()
