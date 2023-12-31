
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
Towards this end, we introduce **node2vec2rank**, a method for graph differential analysis that ranks nodes  according to the disparities of their representations in joint latent embedding spaces. Unlike previous bag-of-features approaches, we take advantage of recent advances in machine learning and statistics to compare graphs in higher-order structures and in a data-driven manner. Employing [UASE](https://github.com/iggallagher/Spectral-Embedding), a multi-layer spectral embedding technique, n2v2r is computationally efficient and can provably identify the correct ranking of differences. Furthermore, we incorporate stability into n2v2r for an overall procedure that adheres to veridical data science principles. <br>

Note, that in our case, node2vec in the title does not correspond to the algorithm by Grover and Leskovec, but rather to any algorithm that can produce (multi-layer) node embeddings. An earlier version of node2vec2rank was using node2vec with transfer learning, but it was unstable and with many paramaters. <br>

While the method is motivated and validated with biological applications, it can be used in any other domain with similar objectives. <br>

This repository provides the method, source code, and example notebooks. In particular, we provide the notebooks corresponding to the biological applications used in the paper, as well as a demo notebook for the general usage. 

![alt text](https://github.com/pmandros/n2v2r/blob/main/n2v2r.png)


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps:

### Prerequisites

* It is recommended to create a new virtual enviroment with [conda](https://www.anaconda.com/)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/pmandros/n2v2r
   ```
2. Change to the project repositry:
   ```sh
   cd n2v2r

   ```
3. Run the following command to create the environment with the needed packages in conda. Please note that it might take around 10-20 mins for the environment to be created:
```sh
conda env create --file environment.yaml
```
4. Activate the environment:
```sh
conda activate n2v2r
```

<!-- USAGE EXAMPLES -->
## Usage

1. Running a single node2vec2rank model:
To run the node2vec2rank algorithm, run the following command:
   ```sh
   python node2vec2rank/node2vec2rank.py --config config.json
   ```
You can modify the experiment parameters in the config.json file:
   ```json
{
    "data_io": {
        "save_dir": "../output",
        "data_dir": "../data/networks",
        "graph_filenames": ["network_control.tsv","network_case.tsv"],
        "seperator": "\t",
        "is_edge_list": false,
        "transpose": true
    },
    "data_preprocessing": {
        "project_unipartite_on": null,
        "threshold": 0,
        "top_percent_keep": [100],
        "binarize": [false],
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
2. You can alternatively run the script from the command line and it will parse the command line arguments based on the given parameters:
```sh
python node2vec2rank.py --save_dir ../output --data_dir ../data/networks --graph_filenames network_control.tsv network_case.tsv --seperator "\t" --is_edge_list false --transpose true --project_unipartite_on null --threshold 0 --top_percent_keep 100 --binarize false --absolute false --embed_dimensions 4 6 8 10 12 14 16 18 20 22 24 --distance_metrics "euclidean" "cosine" --verbose 1

```
Please note that the following arguments are **required**: **--save_dir**, **--graph_filenames**, **--data_dir**
3. You can also check all the possible parameters with their corresponding description using the following command:
```sh
python node2vec2rank/node2vec2rank.py --help
```
It will generate the following output:
```sh
usage: node2vec2rank.py [-h] [--config CONFIG] [--save_dir SAVE_DIR] [--graph_filenames GRAPH_FILENAMES [GRAPH_FILENAMES ...]]
                        [--data_dir DATA_DIR] [--seperator SEPERATOR] [--is_edge_list] [--transpose]
                        [--project_unipartite_on PROJECT_UNIPARTITE_ON] [--threshold THRESHOLD]
                        [--top_percent_keep TOP_PERCENT_KEEP [TOP_PERCENT_KEEP ...]] [--binarize BINARIZE [BINARIZE ...]]
                        [--absolute] [--embed_dimensions EMBED_DIMENSIONS [EMBED_DIMENSIONS ...]]
                        [--distance_metrics DISTANCE_METRICS [DISTANCE_METRICS ...]] [--seed SEED] [--verbose VERBOSE]

Script arguments

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Configuration file path

data_io:
  --save_dir SAVE_DIR   Save directory
  --graph_filenames GRAPH_FILENAMES [GRAPH_FILENAMES ...]
                        Graph filenames
  --data_dir DATA_DIR   Data Directory
  --seperator SEPERATOR
                        Separator
  --is_edge_list        Whether the input is an edge list or tabular
  --transpose           Wether to transpose the graph adjacency matrices or not, e.g., bringing the row genes to the column

data_preprocessing:
  --project_unipartite_on PROJECT_UNIPARTITE_ON
                        If the graphs are rectangular, it will project them into column or row space
  --threshold THRESHOLD
                        Everything below this value will be 0
  --top_percent_keep TOP_PERCENT_KEEP [TOP_PERCENT_KEEP ...]
                        Keeps the top percentage of edges, turning the rest to 0
  --binarize BINARIZE [BINARIZE ...]
                        Whether to binarize the graphs, turning everything above 0 to 1
  --absolute            Absolute the graphs, i.e., turn negative values into positive

fitting_ranking:
  --embed_dimensions EMBED_DIMENSIONS [EMBED_DIMENSIONS ...]
                        Embedding dimensions
  --distance_metrics DISTANCE_METRICS [DISTANCE_METRICS ...]
                        Distance metrics
  --seed SEED           Random seed
  --verbose VERBOSE     Verbose level
```
4. Running in a Jupyter Notebook Environment:
You can also run the code in jupyter notebook. Details about setting up your own workflow in jupyter notebook can be found in the notebooks provided. 

<!-- Practicalities -->
## Practicalities

The input can be either in adjacency format with index and header, or a weigted edge list without header (supported by networkx). Regarding the parameters embed_dimensions and distance_metrics, node2vec2rank runs multiple times for every parameter combination, and then all the rankings are aggregated into one using the Borda scheme. The default settings have been tested thoroughly. The data_preprocessing parameteres are more involved. In a nutshell, they perform graph preprocessing such as binarization, sparsifying by keeping top edges, absoluting. We highly recommend to perform your own preprocessing and use the resulting networks with n2v2r.

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
