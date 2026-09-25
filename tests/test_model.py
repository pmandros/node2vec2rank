import json
import os

import numpy as np
import pandas as pd
import pytest

from node2vec2rank import N2V2R, DataLoader, resolve_config
from node2vec2rank.node2vec2rank import run

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_planted_change_is_ranked_first(two_sbm_graphs):
    graphs, memberships = two_sbm_graphs
    model = N2V2R(graphs, seed=0, embed_dimensions=[4, 6], verbose=-1)
    model.fit_transform_rank()
    borda = model.aggregate_transform()["1"]["borda_ranks"].to_numpy()

    top = np.argsort(-borda)[:100]
    # the 100 changed nodes of block 1 should dominate the top of the ranking
    assert np.mean(memberships[top] == 1) > 0.9


def test_reproducible_with_seed(two_sbm_graphs):
    graphs, _ = two_sbm_graphs
    runs = []
    for _ in range(2):
        model = N2V2R(graphs, seed=3, embed_dimensions=[4], verbose=-1)
        runs.append(model.fit_transform_rank()["1"])
    pd.testing.assert_frame_equal(*runs)


@pytest.mark.parametrize("strategy,keys", [
    ("sequential", ["1", "2"]),
    ("one_vs_before", ["1", "2"]),
    ("one_vs_rest", ["1", "2", "3"]),
])
def test_comparison_keys_are_consistent(two_sbm_graphs, strategy, keys):
    graphs, _ = two_sbm_graphs
    graphs = graphs + [graphs[0]]
    model = N2V2R(graphs, comp_strategy=strategy, embed_dimensions=[4], seed=0, verbose=-1)
    assert list(model.fit_transform_rank()) == keys
    assert list(model.aggregate_transform()) == keys
    assert list(model.degree_difference_ranking()) == keys
    assert list(model.signed_ranks_transform()) == keys
    assert list(model.pairwise_signed_aggregate_ranks) == keys


def test_degree_difference_sign_convention():
    before = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    after = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    model = N2V2R([before, after], nodes=["a", "b", "c"], embed_dimensions=[1],
                  distance_metrics=["euclidean"], verbose=-1)
    dedi = model.degree_difference_ranking()["1"]
    assert dedi["DeDi"].to_dict() == {"a": 1.0, "b": -1.0, "c": 0.0}


def test_config_validation():
    with pytest.raises(ValueError):
        resolve_config(comp_strategy="pairwise")
    with pytest.raises(ValueError):
        resolve_config({"data_io": {"unknown_option": 1}})
    assert resolve_config({"data_io": {"seperator": ","}})["separator"] == ","


def test_dimension_larger_than_graph_is_rejected():
    with pytest.raises(ValueError):
        N2V2R([np.eye(5), np.eye(5)], embed_dimensions=[5], verbose=-1)


def test_demo_edge_list_and_adjacency_give_same_ranking():
    common = {"data_dir": os.path.join(REPO, "data", "networks", "demo"), "separator": ",",
              "embed_dimensions": [4, 8], "seed": 0, "verbose": -1}
    adj = DataLoader(graph_filenames=["adj_matrix_1.csv", "adj_matrix_2.csv"], **common)
    edge = DataLoader(graph_filenames=["edge_list_1.csv", "edge_list_2.csv"],
                      is_edge_list=True, **common)

    rankings = []
    for loader in (adj, edge):
        model = N2V2R(loader.get_graphs(), loader.get_nodes(), config=loader.config)
        model.fit_transform_rank()
        rankings.append(model.aggregate_transform()["1"]["borda_ranks"])

    # edge lists omit isolated nodes, so compare on the shared nodes
    shared = rankings[1].index.intersection(rankings[0].index)
    assert len(shared) > 900
    corr = np.corrcoef(rankings[0].loc[shared].rank(), rankings[1].loc[shared].rank())[0, 1]
    assert corr > 0.99


def test_cli_writes_outputs(tmp_path):
    with open(os.path.join(REPO, "configs", "config_demo_adj_CLI.json"), encoding="utf-8") as f:
        config = json.load(f)
    config["data_io"]["data_dir"] = os.path.join(REPO, "data", "networks", "demo")
    config["fitting_ranking"]["embed_dimensions"] = [4, 8]
    config["fitting_ranking"]["verbose"] = -1
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    model = run(["--config", str(config_path), "--save_dir", str(tmp_path / "out"), "--signed"])
    written = set(os.listdir(model.save_dir))
    assert {"config.json", "1.tsv", "1_agg.tsv", "1_degDif.tsv", "1_signed.tsv",
            "1_agg_signed.tsv"} <= written
