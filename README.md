
<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT Liscence][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
 <!-- <img src="" alt="logo" align="center"> -->
  <h3 align="center"> node2vec2rank: Large Scale and Stable Graph Differential Analysis via Node Embeddings and Ranking</h3>

  <p align="center">
    <br />
    <br />
    <a href="https://github.com/pmandros/n2v2r/issues">Report Bug</a>
    ·
    <a href="https://github.com/pmandros/n2v2r/pulls">Add Feature</a>
  </p>
</p>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
      <li><a href="#practicalities">Practicalities</a></li>
    <li><a href="#contributing">Contributing</a></li>
   <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
Computational methods in biology can infer large molecular interaction networks from multiple data modalities and resolutions, creating unprecedented opportunities to better understand complex biological phenomena. Such graphs can be built from different conditions and get contrasted to uncover graph-level differences, e.g., a case-control study utilizing gene regulatory networks. <br> <br>
Towards this end, we introduce **node2vec2rank**, a method for graph differential analysis that ranks nodes  according to the disparities of their representations in joint latent embedding spaces. Unlike previous bag-of-features approaches, we take advantage of recent advances in machine learning and statistics to compare graphs in higher-order structures and in a data-driven manner. Employing [UASE](https://github.com/iggallagher/Spectral-Embedding), a multi-layer spectral embedding technique, n2v2r is computationally efficient and can provably identify the correct ranking of differences. Furthermore, we incorporate stability into n2v2r for an overall procedure that adheres to veridical data science principles by running it multiple times and aggregating the results. See figure below for a demonstration simulating a case-control study. <br>

Note, that in our case, node2vec in the title does not correspond to the algorithm by Grover and Leskovec, but rather to any algorithm that can produce (multi-layer) node embeddings. An earlier version of node2vec2rank was using node2vec with transfer learning, but it was unstable and with many paramaters. UASE is based on SVD and is more stable, more efficient, has practically no parameters, and we were able to provide theoretical guarantees about the correct ranking.  <br>

While the method is motivated and validated with biological applications, it can be used in any other domain with similar objectives. <br>

This repository provides the method, source code, and example notebooks. In particular, we provide the notebooks corresponding to the biological applications used in the paper, as well as a demo notebook for the general usage. 

![alt text](https://github.com/pmandros/n2v2r/blob/main/n2v2r.png)


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow steps below.

### Prerequisites

It is recommended to create a new virtual enviroment with [conda](https://www.anaconda.com/).

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/pmandros/n2v2r
   ```
2. Change to the project repositry
   ```sh
   cd n2v2r
   ```
3. Run the following command to create the environment with the needed packages in conda. Please note that it might take a few minutes for the environment to be created
   ```sh
   conda env create --file environment.yaml
   ```
4. Activate the environment
   ```sh
   conda activate n2v2r
   ```

<!-- USAGE EXAMPLES -->
## Usage

### Running node2vec2rank model in command line
To run the node2vec2rank algorithm in command line, run the following command with a configuration file as input
   ```sh
   python node2vec2rank/node2vec2rank.py --config config.json
   ```
The configuration file template is as follows
   ```json
{
    "data_io": {
        "save_dir": "output",
        "data_dir": "data/networks",
        "graph_filenames": ["network_control.tsv","network_case.tsv"],
        "seperator": "\t",
        "is_edge_list": false,
        "transpose": false
    },
    "data_preprocessing": {
        "project_unipartite_on": null,
        "threshold": null,
        "top_percent_keep": 100,
        "binarize": false,
        "absolute": false
    },
    "fitting_ranking": {
        "embed_dimensions": [4,6,8,10,12,14,16,18,20,22,24],
        "distance_metrics": ["euclidean","cosine"],
        "seed": null,
        "verbose": 1
    }
}
   ```
The parameters have the following funcitonality
```sh
data_io:
  --save_dir SAVE_DIR   Save directory
  --graph_filenames [GRAPH_FILENAMES ...]
                        Graph filenames
  --data_dir DATA_DIR   Data Directory
  --separator SEPARATOR
                        Separator used in the graph files
  --is_edge_list        Whether the input is an edge list or tabular
  --transpose           Whether to transpose the graph adjacency matrices or not if bipartite

data_preprocessing:
  --project_unipartite_on PROJECT_UNIPARTITE_ON
                        If the graph adjacency matrices are non-square (i.e., bipartite), it will make them square by projecting into column or row space
  --threshold THRESHOLD
                        Everything below this value will be 0
  --top_percent_keep TOP_PERCENT_KEEP
                        Keeps the top percentage of edges, turning the rest to 0
  --binarize            Whether to binarize the graphs, turning everything above 0 to 1
  --absolute            Absolute the graphs, i.e., turn negative values into positive

fitting_ranking:
  --embed_dimensions [EMBED_DIMENSIONS ...]
                        A list of all the embedding dimensions to use in n2v2r
  --distance_metrics [DISTANCE_METRICS ...]
                        A list of all the distance metrics to use in n2v2r (currently supporting "cosine" and/or "Euclidean")
  --seed SEED           Random seed
  --verbose VERBOSE     Verbose level
```
### Running in a Jupyter Notebook Environment
You can also run the code in jupyter notebook. Details about setting up your own workflow in jupyter notebook can be found in the notebooks provided. Check the demo notebook for general usage.  

<!-- Practicalities -->
## Practicalities

The input files can be either in adjacency format with index and header, or a weighted edge list (three columns of source target and edge weight) without header (the latter supported by networkx). At the moment, n2v2r accepts a list of symmetric dense numpy matrices as input, so the above input files will be converted by the Dataloader accordingly automatically. If your graphs are bipartite and in adjacency format (i.e., non-square matrices), they will be projected to unipartite with multiplication depending on the PROJECT_UNIPARTITE_ON parameter. As an example, if the graphs are in adjacency format with regulators in rows and genes in the columns, PROJECT_UNIPARTITE_ON = 'columns' will create symmetric networks by projecting the bipartite networks to gene space. If you want to work directly with bipartite graphs, they should be represented in edge list format and not adjacency. Note that internally the graphs are converted to scipy sparse format prior to the SVD. <br>

The output (i.e., node rankings) and config file are written to disk in the folder specified in the config file with a timestamp attached. The node rankings are all dataframes but not sorted, rather the index follows the original node order as returned by the Dataloader. <br>

Regarding the parameters embed_dimensions and distance_metrics, node2vec2rank runs multiple times for every parameter combination, and then all the rankings are aggregated into one using the Borda scheme. The default parameter settings have been tested thoroughly. The data_preprocessing parameters are more involved. In a nutshell, they perform graph preprocessing such as binarization, sparsifying by keeping top edges, absoluting, thresholding, and projecting bipartite graphs to unipartite. We highly recommend performing your own preprocessing and use the resulting networks with n2v2r. Otherwise, check the network_transform function and the order of network transformations. <br>

When the input is more than 2 graphs, there exists three different strategies to compare the graphs: one-vs-rest, sequential, one-vs-before. The last two imply some notion of ordering, e.g., one-vs-before implies strictly ordered graphs (e.g., longitudinal data). Node2vec2rank will perform pairwise comparisons depending on the strategies. The one-vs-rest will compare each network node embedding with the mean of the remaining network embeddings, producing for K graphs K rankings. The sequential strategy compares each graph node embedding with the next graph node embedding in the order, producing for K graphs K-1 rankings (we use this strategy in the cell cycle notebook). Lastly, the one-vs-before will compare each graph node embedding with the mean embedding of all previous graphs in the order, producing for K graphs K-1 rankings. Note that all strategies are equivalent for two graphs, but we recommend using the sequential strategy for two graphs (to better access the resulting rankings). So far we have not tested graphs with the one-vs-before strategy. <br>

The post_utils class contains functions to perform over-representation analysis (ORA) and gene set enrichment analysis (GSEA) using [GSEApy](https://gseapy.readthedocs.io/en/latest/), as well as plotting the results using bubbleplots that are saved as pdfs and are publication ready. We also include gene set libraries such as KEGG and GOBP from [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/) for your convenience. 

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!--LICENSE -->
## License

Distributed under the GPL-3 License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

[Panagiotis Mandros](https://linkedin.com/in/pmandros) - pmandros[at]hsph[dot]harvard[dot]edu <br>
[Anis Ismail](https://linkedin.com/in/anisdimail) - anis[dot]ismail[at]student[dot]kuleuven[dot]be







<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/pmandros/n2v2r.svg?style=for-the-badge
[contributors-url]: https://github.com/pmandros/n2v2r/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/pmandros/n2v2r.svg?style=for-the-badge
[forks-url]: https://github.com/pmandros/n2v2r/network/members
[stars-shield]: https://img.shields.io/github/stars/pmandros/n2v2r.svg?style=for-the-badge
[stars-url]: https://github.com/pmandros/n2v2r/stargazers
[issues-shield]: https://img.shields.io/github/issues/pmandros/n2v2r.svg?style=for-the-badge
[issues-url]: https://github.com/pmandros/n2v2r/issues
[license-shield]: https://img.shields.io/badge/license-GPL--3.0--only-green?style=for-the-badge
[license-url]: https://github.com/pmandros/n2v2r/LICENSE
