import json
import argparse
import sys
from dataloader import DataLoader
from model import N2V2R


# Create the parser
parser = argparse.ArgumentParser(description='Script arguments')
parser.add_argument("--config", required=True, help='Configuration file path')

# # Add data_loading arguments
# data_loading_group = parser.add_argument_group('data_io')
# data_loading_group.add_argument(
#     '--save_dir', type=str, help='Save directory')
# data_loading_group.add_argument(
#     '--graph_filenames', nargs='+', type=str, help='Graph filenames')
# data_loading_group.add_argument(
#     '--data_dir', type=str, help='Data Directory')
# data_loading_group.add_argument(
#     '--separator', default='\t', type=str, help='Separator used in the graph files')
# data_loading_group.add_argument(
#     '--is_edge_list', action='store_true', help='Whether the input is an edge list or tabular')
# data_loading_group.add_argument(
#     '--transpose', action='store_true', help='Whether to transpose the graph adjacency matrices or not if bipartite')

# # Add data_preprocessing arguments
# data_preprocessing_group = parser.add_argument_group('data_preprocessing')
# data_preprocessing_group.add_argument(
#     '--project_unipartite_on', default='columns', type=str, help='If the graph adjacency matrices are non-square (i.e., bipartite), it will make them square by projecting into column or row space')
# data_preprocessing_group.add_argument(
#     '--threshold', type=float, default=None, help='Everything below this value will be 0')
# data_preprocessing_group.add_argument(
#     '--top_percent_keep',  type=int, default=100, help='Keeps the top percentage of edges, turning the rest to 0')
# data_preprocessing_group.add_argument(
#     '--binarize',  action='store_true', help='Whether to binarize the graphs, turning everything above 0 to 1')
# data_preprocessing_group.add_argument(
#     '--absolute', action='store_true', help='Absolute the graphs, i.e., turn negative values into positive')

# # Add fitting_ranking arguments
# fitting_ranking_group = parser.add_argument_group('fitting_ranking')
# fitting_ranking_group.add_argument(
#     '--embed_dimensions', nargs='+', type=int, default=[4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24], help='Embedding dimensions')
# fitting_ranking_group.add_argument(
#     '--distance_metrics', nargs='+', default=["euclidean", "cosine"], help='Distance metrics')
# fitting_ranking_group.add_argument(
#     '--seed', type=int, default=None, help='Random seed')
# fitting_ranking_group.add_argument(
#     '--verbose', type=int, default=1, help='Verbose level')

# Parse the arguments from the command line
args = parser.parse_args()

# user should provide path of config file
# all other args will be ignored and will be extracted from the file
if args.config is not None:
    with open(args.config, 'r', encoding='utf-8') as file:
        args = json.load(file)
        args = {param: value for _, params in args.items()
                  for param, value in params.items()}
else:
    print("The following argument is required: --config")
    parser.print_help()
    sys.exit(1)

# create dataloader and load the graphs in memory
dataloader = DataLoader(config=args)
graphs = dataloader.get_graphs()
interest_nodes = dataloader.get_nodes()

# define Node2Vec2Rank model
model = N2V2R(graphs=graphs, config=args, nodes=interest_nodes)

# train Node2Vec2Rank and generate rankings
rankings = model.fit_transform_rank()

# generate ranking based on borda ranking
borda_rankings = model.aggregate_transform()

# compute DeDi ranking
DeDi_ranking = model.degree_difference_ranking()
