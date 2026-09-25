"""Command line interface: ``n2v2r --config config.json`` or
``python -m node2vec2rank --config config.json``."""

import argparse

from node2vec2rank import __version__
from node2vec2rank.config import resolve_config
from node2vec2rank.dataloader import DataLoader
from node2vec2rank.model import N2V2R


def build_parser():
    parser = argparse.ArgumentParser(
        prog="n2v2r",
        description="node2vec2rank: graph differential analysis via multi-layer "
                    "spectral embedding and ranking. Parameters come from the JSON "
                    "config file; the options below override it.")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument("--save_dir", help="Output directory (overrides the config)")
    parser.add_argument("--seed", type=int, help="Random seed (overrides the config)")
    parser.add_argument("--signed", action="store_true",
                        help="Also write the rankings signed by the degree difference")
    parser.add_argument("--significance", action="store_true",
                        help="Also write per-node p- and q-values from the degree-adjusted empirical null")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def run(argv=None):
    """Runs the command line workflow and returns the fitted model."""
    args = build_parser().parse_args(argv)

    overrides = {key: value for key, value in
                 (("save_dir", args.save_dir), ("seed", args.seed)) if value is not None}
    config = resolve_config(args.config, **overrides)

    # create dataloader and load the graphs in memory
    dataloader = DataLoader(config=config)

    model = N2V2R(graphs=dataloader.get_graphs(), nodes=dataloader.get_nodes(), config=config)

    # compute the rankings for every parameter combination, then aggregate them
    model.fit_transform_rank()
    model.aggregate_transform()

    # compute the degree difference ranking (also the prior for signing)
    model.degree_difference_ranking()
    if args.signed:
        model.signed_ranks_transform()
    if args.significance:
        model.significance()

    if model.save_dir:
        print(f"\nResults written to {model.save_dir}")
    return model


def main(argv=None):
    run(argv)
    return 0


if __name__ == "__main__":
    main()
