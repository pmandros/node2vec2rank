
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
  <h3 align="center"> Node2vec2rank: the coolest kid in the block for extracting changes in networks</h3>

  <p align="center">
    <br />
    <a href="https://github.com/pmandros/n2v2r"><strong>Explore the docs</strong></a>
    <br />
    <br />
    <a href="https://github.com/pmandros/n2v2r">View Demo</a>
    ·
    <a href="https://github.com/pmandros/n2v2r/issues">Report Bug</a>
    ·
    <a href="https://github.com/pmandros/n2v2r/pulls">Request Feature</a>
  </p>
</p>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
   <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
Graphs are ubiquitous in science as they can flexibly represent the interactions driving the arbitrarily complex
phenomena under investigation. Surprisingly, we find that the fundamental question of “how two graphs
differ”, e.g., a case-control study utilizing gene regulatory networks, lacks the attention and success of
its tabular data counterpart. <br> <br>
We introduce **node2vec2rank**, a framework for graph differential analysis that
ranks the nodes according to their drift in latent embedding spaces—provided the latent space is shared.
Unlike previous approaches that are primarily based on the bag of features approach comparing individual
edges with handcrafted criteria, we take advantage of the recent advances in machine learning and statistics
to compare graphs in their higher-order structure and in a data-driven manner. Coupled with a **statistical model** for **tractability**, **interpretability**, and **reproducibility**, our procedure further incorporates perturbations for **stability** and consensus-based downstream applications.
### Built With

* [NumPy](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Pandas](https://pandas.pydata.org/docs)
* [NetworkX](https://networkx.org/)
* [Spectral Embedding](https://github.com/iggallagher/Spectral-Embedding)

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps:

### Prerequisites

* It is recommended to create a new virtual enviroment with [conda](https://www.anaconda.com/): <br>
`conda create --name node2vec2rank python=3.9`

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/pmandros/n2v2r
   ```
2. Change to the project repositry:
   ```sh
   cd N2V2R

   ```
3. Run the following command to create the environment with the needed packages in conda. Please note that it might take around 10-20 mins for the environment to be created:
```sh
conda env create --file environment.yaml
```
4. Activate the environment:
```sh
conda activate bio_embed_env
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
        "data_dir": "data/networks/inferelator",
        "graph_filenames": [
            "signed_network.tsv",
            "CSTARVE_signed_network.tsv"
        ],
        "seperator": "\t",
        "is_edge_list": false,
        "transpose": true
    },
    "data_preprocessing": {
        "project_unipartite_on": "columns",
        "threshold": 0,
        "top_percent_keep": [
            100,
            75
        ],
        "binarize": [
            false,
            true
        ],
        "absolute": true
    },
    "fitting_ranking": {
        "embed_dimensions": [
            2,
            4,
            8,
            16
        ],
        "distance_metrics": [
            "euclidean",
            "cosine"
        ],
        "seed": null,
        "verbose": 1
    }
}
   ```
2. You can alternatively run the script from the command line and it will parse the command line arguments based on the given parameters:
```sh
python node2vec2rank.py --save_dir ../output --data_dir data/networks/inferelator --graph_filenames signed_network.tsv CSTARVE_signed_network.tsv --seperator "\t" --is_edge_list false --transpose true --project_unipartite_on columns --threshold 0 --top_percent_keep 100 75 --binarize false true --absolute true --embed_dimensions 2 4 8 16 --distance_metrics "euclidean" "cosine" --verbose 1

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
  --is_edge_list        Whether the input is an edge list
  --transpose           whether to transpose the adjacency matrix or not

data_preprocessing:
  --project_unipartite_on PROJECT_UNIPARTITE_ON
                        Project unipartite
  --threshold THRESHOLD
                        Threshold value
  --top_percent_keep TOP_PERCENT_KEEP [TOP_PERCENT_KEEP ...]
                        Top percentage to keep
  --binarize BINARIZE [BINARIZE ...]
                        Whether to binarize the data
  --absolute            Take the absolute value

fitting_ranking:
  --embed_dimensions EMBED_DIMENSIONS [EMBED_DIMENSIONS ...]
                        Embed dimensions
  --distance_metrics DISTANCE_METRICS [DISTANCE_METRICS ...]
                        Distance metrics
  --seed SEED           Random seed
  --verbose VERBOSE     Verbose level
```
4. Running in a Jupyter Notebook Environment:
You can also run the code in jupyter notebook. Details about setting up your own workflow in jupyter notebook can be found in [node2vec2rank_demo.ipynb]((https://github.com/pmandros/n2v2r/notebooks/node2vec2rank_demo.ipynb). 
<!-- ROADMAP -->
## Roadmap


See the [open issues](https://github.com/pmandros/n2v2r/issues) for a list of proposed features (and known issues).



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



<!-- ACKNOWLEDGEMENTS 
## Acknowledgements

* []()
-->



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
