from node2vec2rank.model import Model2Rank
from node2vec2rank.multimodel import MultiModel2Rank
from node2vec2rank.dataloader import DataLoader
import json
import os


#read the config file and create output file it doesn't exist
config = json.load(open('config.json', 'r'))

config = {param: value for section, params in config.items()
          for param, value in params.items()}

os.makedirs(config["main_save_dir"], exist_ok=True)

#create dataloader and load the graphs in memory
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
