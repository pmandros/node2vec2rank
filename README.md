
<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT Liscence][liscence-shield]][liscence-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
 <!-- <img src="" alt="logo" align="center"> -->
  <h3 align="center"> Node2vec2rank: the coolest kid in the block for extracting changes in networks</h3>

  <p align="center">
    <br />
    <a href=""><strong>Explore the docs</strong></a>
    <br />
    <br />
    <a href="">View Demo</a>
    ·
    <a href="">Report Bug</a>
    ·
    <a href="">Request Feature</a>
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

### Built With

* [Gensim](https://pytorch.org/)
* [NumPy](https://numpy.org/)
* [matplotlib](https://matplotlib.org/)
* [Pandas]()
* [CSRGraph]()

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

* It is recommended to create a new virtual enviroment with conda: <br>
`conda create --name node2vec2rank python=3.9`

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/pmandros/NODE2VEC2RANK
   ```
2. Change to the project repositry:
   ```sh
   cd NODE2VEC2RANK

   ```

3. Install required packages
   ```sh
   pip install -r requirements.txt
   ```
4. Run the following bash script to install the needed packages in conda:
```sh
conda_install.sh
```

<!-- USAGE EXAMPLES -->
## Usage

1. Running a single node2vec2rank model:
To run the node2vec2rank algorithm, run the following command:
   ```sh
   python main.py
   ```
You can modify the experiment parameters in the config.json file:
   ```json
{
    "data_loading": {
        "main_save_dir": "../output",
        "data_dir": "../input",
        "graph_name_1": "sbm1.txt",
        "graph_name_2": "sbm2.txt",
        "seperator": ",",
        "bipartite": false,
        "threshold": null,
        "percentile_to_keep": 100,
        "binarize": false,
        "symmetrize": true,
        "absolute": false,
        "is_edge_list": false,
        "directed": false
    },
    "training": {
        "repetitions": 2,
        "run_random_walks": true,
        "visualize": false,
        "method": "Word2Vec",
        "embed_dim": 8,
        "distance": "euclidean",
        "max_iter_first": 10,
        "max_iter_second": 10,
        "seed": 48,
        "rank_aggregation_method": "mean"
    },
    "random_walk_params": {
        "walklen": 50,
        "epochs": 10,
        "return_weight": 1,
        "neighbor_weight": 1
    },
    "visualization_params": {
        "embed_algorithm": "pca",
        "n_components_manifold": 2
    },
    "Word2Vec_params": {
        "window": 5,
        "negative": 5
    },
    "Mittens_params": {}
}
   ```
2. Running node2vec2model with multiple values for parameters:
3. Running in a Jupyter Notebook Environment
You can also run the code in jupyter notebook in [__.ipynb](). 
<!-- ROADMAP -->
## Roadmap


See the [open issues](https://github.com/github_username/repo_name/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



< LICENSE 
## License

Distributed under the MIT License. See `LICENSE` for more information.
>


<!-- CONTACT -->
## Contact

[Panagiotis Mandros]() - 
[Anis Ismail](https://linkedin.com/in/anisdimail) - anis[dot]ismail[at]lau[dot]edu




<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* []()




<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/_/_.svg?style=for-the-badge
[contributors-url]: https://github.com/_/_/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/_/_.svg?style=for-the-badge
[forks-url]: https://github.com/_/_/network/members
[stars-shield]: https://img.shields.io/github/stars/_/_.svg?style=for-the-badge
[stars-url]: https://github.com/_/_/stargazers
[issues-shield]: https://img.shields.io/github/issues/_/_.svg?style=for-the-badge
[issues-url]: https://github.com/_/_/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: 
[license-shield]: https://img.shields.io/github/license/_/_.svg?style=for-the-badge
[license-url]: https://github.com/_/_/blob/master/LICENSE.txt 
