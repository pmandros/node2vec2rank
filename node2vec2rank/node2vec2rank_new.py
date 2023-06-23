from node2vec2rank.model import n2v2r
from node2vec2rank.dataloader_new import DataLoader
import json
import os
from node2vec2rank.model import degree_difference_ranking



#read the config file and create output file it doesn't exist
config = json.load(open('config.json', 'r'))

config = {param: value for section, params in config.items()
          for param, value in params.items()}

os.makedirs(config["main_save_dir"], exist_ok=True)

#create dataloader and load the graphs in memory
dataloader = DataLoader(config=config)
graphs = dataloader.get_graphs()
interest_nodes = dataloader.interest_nodes()


DeDi_ranking = degree_difference_ranking(dataloader.graphs, dataloader.interest_nodes)

# check if there are multiple values for any given parameter
multi_params = [key for key in config.keys() if isinstance(
    config[key], list) and len(config[key]) > 1]

rankings = n2v2r.fit_transform_rank()

borda_rankings = n2v2r.aggregate_transform()

signed_rankings = n2v2r.signed_ranks_transform([v.iloc[:,0] for k,v in DeDi_ranking.items()])