"""The default (total and own move, ``align="both"``), its two halves, the
radial metric and the chi-square test per node (``N2V2R.wald_test``).

Two kinds of change are simulated, because the total move of a node and its
own move after robust alignment answer two questions (see
node2vec2rank/alignment.py):

node-level     some nodes change their own position (switch community or
               module, change hub status); the truth is those nodes.
               dcsbm, sbm, weighted and coexpression are the scenarios of
               simulations.py; "coexpression signed" is the same factor model
               built as ((1 + cor) / 2)^10 (hdWGCNA); "mixed" is the two-truths
               design of two_truths.py (5% of nodes change hub status, 5%
               community, 5% both).
community-level  a whole community changes how it connects, and no node
               changes relative to its community: a community splits, two
               merge (revision_sims.py), one row of the block matrix is
               re-drawn (the paper's simulation design, paper_sims.py), or one
               co-expression module weakens. The truth is every node of the
               affected communities.

Methods: Borda and the degree-adjusted test (``significance()``) with the
default, with the total move only (``align=None``, the paper's method) and
with the own move only (``align="robust"``); the chi-square test at the elbow
dimension without and with alignment (not for co-expression, whose entries are
dependent; its inflation is still recorded); a radial-only and a cosine-only
Borda; DeDi. ``own_share`` is recorded for changed and unchanged nodes.

A last part runs the chi-square test without any change at dimensions up to
24, to show that it is only valid up to the rank of the signal.

    OMP_NUM_THREADS=1 python benchmarks/alignment_wald.py            # about 25 minutes on 4 cores

Results: results/alignment_wald.csv (one row per graph pair, method and truth),
results/alignment_wald_summary.csv and results/alignment_wald_dimension.csv.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import simulations as sims  # noqa: E402
from node2vec2rank import N2V2R, coexpression_network  # noqa: E402

RESULTS = os.path.join(HERE, "results")
DIMS = sims.DEFAULT_DIMS


# ---------------------------------------------------------------- simulators

def coexpression_signed(rng, frac):
    """The factor model of simulations.simulate_coexpression as a signed network."""
    num_genes, num_modules, num_samples = 1000, 5, 150
    in_module = rng.random(num_genes) < 0.7
    modules = np.where(in_module, rng.integers(0, num_modules, num_genes), -1)
    after, changed = modules.copy(), np.zeros(num_genes, bool)
    moved = rng.choice(np.flatnonzero(in_module), int(round(frac * num_genes)), replace=False)
    after[moved] = (modules[moved] + 1) % num_modules
    changed[moved] = True
    loadings = rng.uniform(0.3, 0.9, num_genes)
    graphs = []
    for z in (modules, after):
        factors = rng.standard_normal((num_samples, num_modules))
        expression = rng.standard_normal((num_samples, num_genes))
        member = z >= 0
        expression[:, member] = (loadings[member] * factors[:, z[member]]
                                 + np.sqrt(1 - loadings[member] ** 2) * expression[:, member])
        graphs.append(coexpression_network(expression, power=10, signed=True))
    return graphs, changed


def module_weakens(rng, power, signed):
    """Every gene of module 0 loads 0.6 times as strongly in the second condition."""
    num_genes, num_modules, num_samples = 1000, 5, 150
    modules = np.where(rng.random(num_genes) < 0.7, rng.integers(0, num_modules, num_genes), -1)
    loadings = rng.uniform(0.3, 0.9, num_genes) * (modules >= 0)
    graphs = []
    for condition in range(2):
        scaled = loadings * np.where((modules == 0) & (condition == 1), 0.6, 1.0)
        factors = rng.standard_normal((num_samples, num_modules))
        expression = (np.where(modules >= 0, factors[:, np.maximum(modules, 0)] * scaled, 0.0)
                      + rng.standard_normal((num_samples, num_genes)) * np.sqrt(1 - scaled ** 2))
        graphs.append(coexpression_network(expression, power=power, signed=signed))
    return graphs, modules == 0


def _node_level(name):
    def simulate(rng, frac):
        graphs, changed = (coexpression_signed(rng, frac) if name == "coexpression signed"
                           else sims.SCENARIOS[name](rng, frac))
        return graphs, {"changed": changed}, name.startswith("coexpression")
    return simulate


def _mixed(rng, _):
    import two_truths
    graphs, kind = two_truths.mixed_graphs(2000, rng, "binary")
    return graphs, {truth: np.where(np.isin(kind, [truth, "none"]), kind == truth, np.nan)
                    for truth in ("hub", "community", "both")}, False


def _split_merge(model, kind):
    def simulate(rng, _):
        import revision_sims
        pair = revision_sims.sbm_pair if model == "block model" else revision_sims.coexpression_pair
        graphs, changed, _ = pair(rng, kind)
        return graphs, {"changed": changed}, model != "block model"
    return simulate


def _paper(regime):
    def simulate(rng, _):
        import paper_sims
        graphs, changed = paper_sims.sample_graphs(rng, 500, 6, regime)
        return graphs, {"changed": changed}, False
    return simulate


def _weakens(power, signed):
    def simulate(rng, _):
        graphs, changed = module_weakens(rng, power, signed)
        return graphs, {"changed": changed}, True
    return simulate


# (kind of change, scenario, fraction changed or None, simulator)
SETTINGS = [("node-level", name, frac, _node_level(name))
            for name in ("dcsbm", "sbm", "weighted", "coexpression", "coexpression signed")
            for frac in (0.0, 0.1, 0.3)]
SETTINGS += [("node-level", "mixed hub and community changes", None, _mixed)]
SETTINGS += [("community-level", f"{model}: {kind}", None, _split_merge(model, kind))
             for model in ("block model", "co-expression") for kind in ("split", "merge")]
SETTINGS += [("community-level", f"paper design: {regime}", None, _paper(regime)) for regime in ("naive", "random")]
SETTINGS += [("community-level", "co-expression signed: one module weakens", None, _weakens(10, True)),
             ("community-level", "co-expression: one module weakens", None, _weakens(6, False))]


# ---------------------------------------------------------------- methods

def auroc(labels, scores):
    return sims.auroc(labels, np.nan_to_num(scores, nan=-np.inf))


TAGS = {"both": "", None: ", total move", "robust": ", own move"}


def methods(graphs, seed, dependent_edges):
    """{method: (score, pvalues or None)} and extra columns."""
    out, extra = {}, {}
    for align, tag in TAGS.items():
        model = N2V2R(graphs, seed=seed, align=align, verbose=-1)
        model.fit_transform_rank()
        out["Borda" + tag] = (model.aggregate_transform()["1"]["borda_ranks"].to_numpy(), None)
        test = model.significance()["1"]
        out["degree-adjusted test" + tag] = (test["z"].to_numpy(), test["pvalue"].to_numpy())
        if align == "both":
            extra["own_share"] = test["own_share"].to_numpy()
            continue
        for metric in ("radial", "cosine"):
            single = N2V2R(graphs, seed=seed, align=align, distance_metrics=[metric], verbose=-1)
            single.fit_transform_rank()
            out[f"Borda, {metric} only" + tag] = (single.aggregate_transform()["1"]["borda_ranks"].to_numpy(), None)
        wald = model.wald_test(align=bool(align))["1"]
        extra["elbow"] = wald.attrs["dimension"]
        extra["inflation" + tag] = wald.attrs["inflation"]
        if not dependent_edges:
            out["chi-square test" + tag] = (wald["statistic"].to_numpy(), wald["pvalue"].to_numpy())
    out["DeDi"] = (model.degree_difference_ranking()["1"]["absDeDi"].to_numpy(), None)
    return out, extra


def run_one(task):
    index, replicate = task
    kind, scenario, frac, simulate = SETTINGS[index]
    rng = np.random.default_rng([replicate, index])
    graphs, truths, dependent_edges = simulate(rng, frac)
    scores, extra = methods(graphs, replicate, dependent_edges)
    own_share = extra.pop("own_share")
    rows = []
    for truth, labels in truths.items():
        keep = ~np.isnan(labels.astype(float))
        positive = labels.astype(float) == 1
        # the share of the move that is the node's own, for changed and unchanged nodes
        extra["own_share_changed"] = np.nanmedian(own_share[positive]) if positive.any() else np.nan
        extra["own_share_unchanged"] = np.nanmedian(own_share[keep & ~positive])
        for method, (score, pvalues) in scores.items():
            row = dict(kind=kind, scenario=scenario, frac_changed=frac, replicate=replicate, truth=truth,
                       method=method, **extra)
            if positive.any():
                row["auroc"] = auroc(positive[keep], score[keep])
                top = np.argsort(-np.nan_to_num(score[keep], nan=-np.inf), kind="stable")[:positive.sum()]
                row["precision_at_k"] = positive[keep][top].mean()
            if pvalues is not None:
                from node2vec2rank.significance import benjamini_hochberg
                called = benjamini_hochberg(pvalues) < 0.1
                # nodes with another kind of change are neither true nor false calls
                unchanged = keep & ~positive
                row["frac_p_below_0.05"] = np.nanmean(pvalues[unchanged] < 0.05)
                row["true_calls"] = int((called & positive).sum())
                row["false_calls"] = int((called & unchanged).sum())
                row["num_changed"] = int(positive.sum())
            rows.append(row)
    return rows


def dimension_check(replicate):
    """Without any change: nodes with p < 0.05 by dimension of the chi-square test."""
    rows = []
    for scenario in ("sbm", "dcsbm", "weighted"):
        graphs, _ = sims.SCENARIOS[scenario](np.random.default_rng([replicate, 99]), 0.0)
        model = N2V2R(graphs, seed=replicate, verbose=-1)
        model.fit_transform_rank()
        for dimension in (2, 3, 4, 6, 8, 16, 24):
            for control in (False, True):
                result = model.wald_test(dimension=dimension, genomic_control=control)["1"]
                rows.append(dict(scenario=scenario, replicate=replicate, dimension=dimension,
                                 genomic_control=control, inflation=result.attrs["inflation"],
                                 frac_p_below_05=(result["pvalue"] < 0.05).mean(),
                                 calls=int((result["qvalue"] < 0.1).sum())))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()
    tasks = [(index, replicate) for index in range(len(SETTINGS)) for replicate in range(args.replicates)]
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = [row for result in pool.map(run_one, tasks, chunksize=1) for row in result]
        dimension_rows = [row for result in pool.map(dimension_check, range(args.replicates)) for row in result]
    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(RESULTS, "alignment_wald.csv"), index=False)
    summary = (results.groupby(["kind", "scenario", "frac_changed", "truth", "method"], sort=False, dropna=False)
               [["auroc", "precision_at_k", "frac_p_below_0.05", "true_calls", "false_calls", "num_changed",
                 "own_share_changed", "own_share_unchanged", "elbow", "inflation, total move",
                 "inflation, own move"]].mean().round(3).reset_index())
    summary.to_csv(os.path.join(RESULTS, "alignment_wald_summary.csv"), index=False)
    dimension = (pd.DataFrame(dimension_rows).groupby(["scenario", "dimension", "genomic_control"])
                 [["inflation", "frac_p_below_05", "calls"]].mean().round(3).reset_index())
    dimension.to_csv(os.path.join(RESULTS, "alignment_wald_dimension.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 1000)
    print(summary.to_string(index=False))
    print(dimension.to_string(index=False))


if __name__ == "__main__":
    main()
