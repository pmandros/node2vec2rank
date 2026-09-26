"""Multi-network benchmark: is UASE the right joint embedding for node-level differential ranking?

Compares, on sequences of K graphs over the same nodes, the embedding that
node2vec2rank uses (UASE) against the alternatives a reviewer would ask about:

- UASE        unfolded adjacency spectral embedding (node2vec2rank's default), with
              the default Borda over dimensions 4..24 x {euclidean, cosine}, the
              elbow dimension, and the degree-adjusted z of ``significance()``.
- OMNI        the omnibus embedding (Levin et al., 2017): the ASE of the Kn x Kn
              matrix whose (s, t) block is (A_s + A_t) / 2, compared exactly like UASE.
- ASE+Proc    a separate ASE of every graph, aligned to the previous graph with an
              orthogonal Procrustes rotation (the "why embed jointly?" baseline).
- raw rows    each node's adjacency row compared directly, with no embedding
              (the "why embed at all?" baseline), Borda of euclidean and cosine,
              and the same degree-adjusted z.
- DeDi        absolute degree difference.

Every method ranks the nodes for each sequential comparison (graph t-1 vs t), as
``comp_strategy="sequential"`` does, and is scored against the nodes that changed
at that transition. Run from the repository root (about 20 minutes on 4 cores):

    python benchmarks/multilayer.py              # -> results/multilayer*.csv, *.png

Scenarios (1,000 nodes; ``null`` variants have no change)
---------
switch        degree-corrected SBM, 4 blocks; at every transition a fresh 5% of the
              nodes switch block. K = 2, 4 and 8 graphs.
density       as switch with K = 4, but every graph also has its own overall density
              (x1, x1.5, x0.7, x1.2), as when conditions have different cell numbers.
cohesion      degree-corrected SBM, 8 blocks; at every transition one community
              becomes more cohesive or dissolves (within-block probability x2 or x0.3)
              while every node keeps its block. This is the change for which UASE
              is proven longitudinally stable (Gallagher et al., 2021).
weighted      switch with Gaussian edge weights, K = 4.
coexpression  |cor|^6 networks of 150 samples per condition from a latent module
              model; at every transition 5% of the genes switch module. K = 4.
coexp_unequal as coexpression with 200, 60, 150 and 80 samples per condition, so
              the graphs differ in noise level.
"""

import argparse
import itertools
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.linalg import orthogonal_procrustes  # noqa: E402
from scipy.sparse.linalg import eigsh  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "benchmarks"))

from node2vec2rank.embedding import select_dimension, uase  # noqa: E402
from node2vec2rank.model_utils import borda_aggregate, compute_pairwise_distances  # noqa: E402
from node2vec2rank.significance import empirical_null_test  # noqa: E402
from simulations import auroc, precision_at_k, spearman  # noqa: E402

DEFAULT_DIMS = list(range(4, 25, 2))
METRICS = ("euclidean", "cosine")
RESULTS_DIR = os.path.join(REPO, "benchmarks", "results")
DENSITY = (1.0, 1.5, 0.7, 1.2)
SAMPLES_UNEQUAL = (200, 60, 150, 80)


# ---------------------------------------------------------------- embeddings

def omnibus(graphs, d, random_state=None, return_eigenvalues=False):
    """Omnibus embedding (Levin et al., 2017): (K, n, d) per-graph node embeddings."""
    num_graphs, n = len(graphs), graphs[0].shape[0]
    if all(sparse.issparse(g) for g in graphs):
        matrix = sparse.bmat([[(graphs[s] + graphs[t]) / 2 for t in range(num_graphs)]
                              for s in range(num_graphs)], format="csr")
    else:
        rows = np.vstack([np.hstack([np.asarray(g)] * num_graphs) for g in graphs])
        matrix = (rows + rows.T) / 2
    rng = np.random.default_rng(random_state)
    eigenvalues, vectors = eigsh(matrix, k=d, which="LM", v0=rng.standard_normal(num_graphs * n))
    order = np.argsort(-np.abs(eigenvalues))
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    embeddings = (vectors * np.sqrt(np.abs(eigenvalues))).reshape(num_graphs, n, d)
    return (embeddings, np.abs(eigenvalues)) if return_eigenvalues else embeddings


def separate_ase(graphs, d, random_state=None):
    """Adjacency spectral embedding of every graph on its own: (K, n, d)."""
    rng = np.random.default_rng(random_state)
    out = []
    for graph in graphs:
        eigenvalues, vectors = eigsh(graph, k=d, which="LM", v0=rng.standard_normal(graph.shape[0]))
        order = np.argsort(-np.abs(eigenvalues))
        out.append(vectors[:, order] * np.sqrt(np.abs(eigenvalues[order])))
    return np.stack(out)


# ---------------------------------------------------------------- simulators

def _theta(rng, n):
    theta = rng.pareto(2.5, n) + 1
    return theta / theta.mean()


def _sample(probs, rng, weighted, weight_noise=0.1):
    if weighted:
        upper = np.triu(probs + weight_noise * rng.standard_normal(probs.shape), 1)
    else:
        upper = np.triu(rng.random(probs.shape) < np.clip(probs, 0, 1), 1).astype(float)
    return upper + upper.T


def _switch_labels(rng, labels, num_blocks, num_graphs, frac, eligible=None):
    """Label sequences in which a fresh frac of the (eligible) nodes switch at every transition."""
    n = len(labels)
    sequence, changed = [labels], []
    available = np.ones(n, bool) if eligible is None else eligible.copy()
    for _ in range(1, num_graphs):
        current = sequence[-1].copy()
        moved = np.zeros(n, bool)
        count = int(round(frac * n))
        if count:
            chosen = rng.choice(np.flatnonzero(available), count, replace=False)
            current[chosen] = (current[chosen] + rng.integers(1, num_blocks, count)) % num_blocks
            moved[chosen] = True
            available[chosen] = False
        sequence.append(current)
        changed.append(moved)
    return sequence, changed


def sim_switch(rng, num_graphs, frac, n=1000, weighted=False, density=None,
               num_blocks=4, p_in=0.1, p_out=0.04, population=False):
    blocks = rng.integers(0, num_blocks, n)
    theta = _theta(rng, n)
    sequence, changed = _switch_labels(rng, blocks, num_blocks, num_graphs, frac)
    block_probs = np.full((num_blocks, num_blocks), p_out) + np.eye(num_blocks) * (p_in - p_out)
    graphs = []
    for t, z in enumerate(sequence):
        probs = np.outer(theta, theta) * block_probs[z][:, z] * (1 if density is None else density[t])
        graphs.append(probs if population else _sample(probs, rng, weighted))
    return graphs, changed


def sim_cohesion(rng, num_graphs, frac, n=1000, num_blocks=8, p_in=0.1, p_out=0.04, population=False):
    blocks = rng.integers(0, num_blocks, n)
    theta = _theta(rng, n)
    block_probs = np.full((num_blocks, num_blocks), p_out) + np.eye(num_blocks) * (p_in - p_out)
    graphs, changed = [], []
    for t in range(num_graphs):
        if t > 0:
            moved = np.zeros(n, bool)
            if frac > 0:
                block_probs = block_probs.copy()
                community = rng.integers(num_blocks)
                block_probs[community, community] *= 2.0 if block_probs[community, community] <= p_in else 0.3
                moved = blocks == community
            changed.append(moved)
        probs = np.outer(theta, theta) * block_probs[blocks][:, blocks]
        graphs.append(probs if population else _sample(probs, rng, False))
    return graphs, changed


def sim_coexpression(rng, num_graphs, frac, samples=150, n=1000, num_modules=5, frac_in_modules=0.7,
                     soft_power=6):
    in_module = rng.random(n) < frac_in_modules
    modules = np.where(in_module, rng.integers(0, num_modules, n), -1)
    sequence, changed = _switch_labels(rng, modules, num_modules, num_graphs, frac, eligible=in_module)
    sequence = [np.where(in_module, z, -1) for z in sequence]
    loadings = rng.uniform(0.3, 0.9, n)
    sizes = [samples] * num_graphs if np.isscalar(samples) else list(samples)
    graphs = []
    for z, size in zip(sequence, sizes):
        factors = rng.standard_normal((size, num_modules))
        expression = rng.standard_normal((size, n))
        member = z >= 0
        expression[:, member] = (loadings[member] * factors[:, z[member]]
                                 + np.sqrt(1 - loadings[member] ** 2) * expression[:, member])
        adjacency = np.abs(np.corrcoef(expression, rowvar=False)) ** soft_power
        np.fill_diagonal(adjacency, 0)
        graphs.append(adjacency)
    return graphs, changed


SCENARIOS = {
    ("switch", 2): lambda rng, frac: sim_switch(rng, 2, frac),
    ("switch", 4): lambda rng, frac: sim_switch(rng, 4, frac),
    ("switch", 8): lambda rng, frac: sim_switch(rng, 8, frac),
    ("density", 4): lambda rng, frac: sim_switch(rng, 4, frac, density=DENSITY),
    ("cohesion", 4): lambda rng, frac: sim_cohesion(rng, 4, frac),
    ("weighted", 4): lambda rng, frac: sim_switch(rng, 4, frac, weighted=True),
    ("coexpression", 4): lambda rng, frac: sim_coexpression(rng, 4, frac),
    ("coexp_unequal", 4): lambda rng, frac: sim_coexpression(rng, 4, frac, samples=SAMPLES_UNEQUAL),
}


# ---------------------------------------------------------------- methods

def _embedding_distances(embeddings, s, t, dims, metrics=METRICS):
    return np.column_stack([compute_pairwise_distances(embeddings[s][:, :d], embeddings[t][:, :d], m)
                            for d in dims for m in metrics])


def _procrustes_distances(embeddings, s, t, dims, metrics=METRICS):
    columns = []
    for d in dims:
        rotation, _ = orthogonal_procrustes(embeddings[t][:, :d], embeddings[s][:, :d])
        aligned = embeddings[t][:, :d] @ rotation
        columns.extend(compute_pairwise_distances(embeddings[s][:, :d], aligned, m) for m in metrics)
    return np.column_stack(columns)


EMBEDDED = ("UASE", "OMNI")
TESTED = ("UASE degree-adjusted z", "OMNI degree-adjusted z", "raw rows degree-adjusted z")


def score_methods(graphs, seed=0):
    """Returns ({comparison t: {method: score}}, {t: degree}, {t: {method: (p, q)}}, timings, elbows)."""
    dense = [np.asarray(g, dtype=float) for g in graphs]
    max_dim = max(DEFAULT_DIMS)
    timings, elbows, embeddings = {}, {}, {}

    tic = time.perf_counter()
    embeddings["UASE"], values = uase(dense, max_dim, random_state=seed, return_singular_values=True)
    timings["UASE"] = time.perf_counter() - tic
    elbows["UASE"] = select_dimension(values, min_dimension=2)

    tic = time.perf_counter()
    embeddings["OMNI"], values = omnibus(dense, max_dim, random_state=seed, return_eigenvalues=True)
    timings["OMNI"] = time.perf_counter() - tic
    elbows["OMNI"] = select_dimension(values, min_dimension=2)

    tic = time.perf_counter()
    ase = separate_ase(dense, max_dim, random_state=seed)
    timings["ASE+Procrustes"] = time.perf_counter() - tic

    scores, degrees, tests = {}, {}, {}
    for t in range(1, len(dense)):
        s = t - 1
        degree = (np.abs(dense[s]).sum(axis=0) + np.abs(dense[t]).sum(axis=0)) / 2
        comparison, comparison_tests = {"DeDi": np.abs(dense[s].sum(axis=0) - dense[t].sum(axis=0))}, {}
        for name in EMBEDDED:
            all_distances = _embedding_distances(embeddings[name], s, t, DEFAULT_DIMS)
            comparison[f"{name} Borda (default)"] = borda_aggregate(all_distances)
            comparison[f"{name} elbow"] = borda_aggregate(
                _embedding_distances(embeddings[name], s, t, [elbows[name]]))
            z, p, q = empirical_null_test(all_distances, degree)
            comparison[f"{name} degree-adjusted z"] = np.nan_to_num(z, nan=-np.inf)
            comparison_tests[f"{name} degree-adjusted z"] = (p, q)
        comparison["ASE+Procrustes Borda"] = borda_aggregate(_procrustes_distances(ase, s, t, DEFAULT_DIMS))
        raw = np.column_stack([compute_pairwise_distances(dense[s], dense[t], m) for m in METRICS])
        comparison["raw rows Borda"] = borda_aggregate(raw)
        z, p, q = empirical_null_test(raw, degree)
        comparison["raw rows degree-adjusted z"] = np.nan_to_num(z, nan=-np.inf)
        comparison_tests["raw rows degree-adjusted z"] = (p, q)
        scores[t], degrees[t], tests[t] = comparison, degree, comparison_tests
    return scores, degrees, tests, timings, elbows


# ---------------------------------------------------------------- runner

def replicate(task):
    (scenario, num_graphs), frac, rep = task
    rng = np.random.default_rng([rep, int(frac * 100), num_graphs, sorted({s for s, _ in SCENARIOS}).index(scenario)])
    graphs, changed = SCENARIOS[(scenario, num_graphs)](rng, frac)
    scores, degrees, tests, timings, elbows = score_methods(graphs, seed=rep)
    records = []
    for t, comparison in scores.items():
        truth = changed[t - 1]
        for method, score in comparison.items():
            record = {"scenario": scenario, "num_graphs": num_graphs, "frac_changed": frac,
                      "replicate": rep, "comparison": t, "method": method,
                      "spearman_with_degree": spearman(score[~truth], degrees[t][~truth])}
            if truth.any():
                record["auroc"] = auroc(truth, score)
                record["precision_at_k"] = precision_at_k(truth, score)
            if method in tests[t]:
                pvalues, qvalues = tests[t][method]
                valid = ~np.isnan(pvalues)
                called = qvalues < 0.1
                record["frac_p_below_0.05_unchanged"] = np.mean(pvalues[valid & ~truth] < 0.05)
                record["num_false_q_0.1"] = int(np.sum(called & ~truth))
                if truth.any():
                    record["power_q_0.1"] = called[truth].mean()
                    record["fdp_q_0.1"] = (called & ~truth).sum() / max(called.sum(), 1)
            records.append(record)
    for name, seconds in timings.items():
        records.append({"scenario": scenario, "num_graphs": num_graphs, "frac_changed": frac,
                        "replicate": rep, "method": f"time {name}", "seconds": seconds,
                        "elbow": elbows.get(name)})
    print(f"{scenario:13s} K={num_graphs} frac={frac:.2f} rep={rep} "
          + " ".join(f"{k}={v:.1f}s" for k, v in timings.items()), flush=True)
    return records


def fractions(scenario):
    # the cohesion scenario changes one whole community per transition
    return (0.0, 1.0) if scenario == "cohesion" else (0.0, 0.05)


def run(num_replicates, processes):
    tasks = [(key, frac, rep) for key in SCENARIOS for frac in fractions(key[0])
             for rep in range(num_replicates)]
    with Pool(processes) as pool:
        records = list(itertools.chain.from_iterable(pool.imap_unordered(replicate, tasks)))
    return pd.DataFrame.from_records(records)


def population_drift(seed=0):
    """Distances on the expected (noise-free) graphs: how far unchanged nodes move."""
    records = []
    settings = {"switch": lambda rng: sim_switch(rng, 4, 0.05, population=True),
                "density": lambda rng: sim_switch(rng, 4, 0.05, density=DENSITY, population=True),
                "cohesion": lambda rng: sim_cohesion(rng, 4, 1.0, population=True)}
    for scenario, simulate in settings.items():
        graphs, changed = simulate(np.random.default_rng(seed))
        for dim in (4, 8, 24):
            embeddings = {"UASE": uase(graphs, dim, random_state=seed),
                          "OMNI": omnibus(graphs, dim, random_state=seed)}
            for name, embedding in embeddings.items():
                for t in range(1, len(graphs)):
                    distance = np.linalg.norm(embedding[t] - embedding[t - 1], axis=1)
                    truth = changed[t - 1]
                    records.append({"scenario": scenario, "dimension": dim, "embedding": name,
                                    "comparison": t,
                                    "median_unchanged": np.median(distance[~truth]),
                                    "median_changed": np.median(distance[truth]),
                                    "auroc": auroc(truth, distance)})
    return pd.DataFrame.from_records(records)


def run_runtime(sizes=(1000, 2000, 4000), num_graphs_list=(2, 4, 8), seed=0):
    """Embedding time at dimension 24 on dense weighted graphs (like co-expression networks)."""
    records = []
    for n, num_graphs in itertools.product(sizes, num_graphs_list):
        if n * num_graphs > 16000:
            continue  # the dense omnibus matrix would need well over 2 GB
        rng = np.random.default_rng(seed)
        graphs = sim_switch(rng, num_graphs, 0.05, n=n, weighted=True)[0]
        for name, function in (("UASE", uase), ("OMNI", omnibus)):
            tic = time.perf_counter()
            function(graphs, 24, random_state=seed)
            records.append({"num_nodes": n, "num_graphs": num_graphs, "embedding": name,
                            "seconds": time.perf_counter() - tic,
                            "matrix_gb": 8 * n * n * num_graphs * (num_graphs if name == "OMNI" else 1) / 1e9})
        print(f"runtime n={n} K={num_graphs}", flush=True)
    return pd.DataFrame.from_records(records)


def run_demo(seed=0):
    """Recall of the demo network's rewired community (two graphs)."""
    from node2vec2rank.dataloader import DataLoader
    demo_dir = os.path.join(REPO, "data", "networks", "demo")
    loader = DataLoader(data_dir=demo_dir, graph_filenames=["adj_matrix_1.csv", "adj_matrix_2.csv"],
                        separator=",", verbose=-1)
    nodes = pd.Index(loader.get_nodes())
    communities = pd.read_csv(os.path.join(demo_dir, "comm_asiggnments.csv"), index_col=0)
    communities.index = communities.index.astype(str)
    changed = (communities.loc[nodes, "0"] == 0).to_numpy()
    scores, _, _, _, _ = score_methods(loader.get_graphs(), seed=seed)
    return pd.DataFrame.from_records(
        {"method": method, "recall_at_k": precision_at_k(changed, score), "auroc": auroc(changed, score)}
        for method, score in scores[1].items())


def run_real_spike_ins(num_replicates=10):
    """Degree-matched and random neighbourhood swaps on the locCSN ASD network (see real_data.py)."""
    import real_data
    _, graphs = real_data.load_networks()
    records = []
    for rep, degree_matched in itertools.product(range(num_replicates), (True, False)):
        rng = np.random.default_rng([rep, int(degree_matched), 7])
        spiked, changed = real_data.spike_in(graphs[1], rng, degree_matched=degree_matched)
        scores = score_methods([graphs[0], spiked], seed=rep)[0][1]
        records.extend({"replicate": rep, "swaps": "degree-matched" if degree_matched else "random",
                        "method": method, "auroc": auroc(changed, score)}
                       for method, score in scores.items())
        print(f"locCSN spike-in rep={rep} degree_matched={degree_matched}", flush=True)
    return pd.DataFrame.from_records(records)


METHODS = ["DeDi", "raw rows Borda", "raw rows degree-adjusted z", "ASE+Procrustes Borda",
           "OMNI Borda (default)", "OMNI elbow", "OMNI degree-adjusted z",
           "UASE Borda (default)", "UASE elbow", "UASE degree-adjusted z"]
TITLES = {("switch", 2): "Switch, K = 2", ("switch", 4): "Switch, K = 4", ("switch", 8): "Switch, K = 8",
          ("density", 4): "Switch + density, K = 4", ("cohesion", 4): "Cohesion, K = 4",
          ("weighted", 4): "Weighted switch, K = 4", ("coexpression", 4): "Co-expression, K = 4",
          ("coexp_unequal", 4): "Co-expression, unequal n, K = 4"}


def figures(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figures import BLUE, GRID, MUTED_INK, ORANGE  # noqa: F401  (also sets the rcParams)

    y = np.arange(len(METHODS))[::-1]
    colors = {"UASE": BLUE, "OMNI": ORANGE}
    changed = results[results.auroc.notna()]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharex=True, sharey=True)
    for ax, (key, title) in zip(axes.ravel(), TITLES.items()):
        subset = changed[(changed.scenario == key[0]) & (changed.num_graphs == key[1])]
        per_rep = subset.groupby(["method", "replicate"]).auroc.mean()
        summary = per_rep.groupby("method").agg(["mean", "std"]).reindex(METHODS)
        ax.hlines(y, 0.3, summary["mean"], color=GRID, linewidth=2)
        ax.errorbar(summary["mean"], y, xerr=summary["std"], fmt="none", ecolor=MUTED_INK, elinewidth=1)
        ax.scatter(summary["mean"], y, s=22, zorder=3,
                   color=[colors.get(m.split()[0], MUTED_INK) for m in METHODS])
        ax.set_yticks(y)
        ax.set_yticklabels(METHODS)
        ax.set_xlim(0.3, 1.01)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("AUROC, mean over transitions")
    fig.suptitle("Multi-network ranking accuracy (mean and sd over 10 replicates; blue UASE, orange OMNI)",
                 x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "multilayer_auroc.png"), dpi=150)

    null = results[(results.frac_changed == 0) & results.method.isin(METHODS)]
    fig, axes = plt.subplots(1, 8, figsize=(15, 3.6), sharey=True, sharex=True)
    for ax, (key, title) in zip(axes, TITLES.items()):
        subset = null[(null.scenario == key[0]) & (null.num_graphs == key[1])]
        summary = subset.groupby("method").spearman_with_degree.mean().reindex(METHODS)
        ax.barh(y, summary, color=[ORANGE if abs(v) > 0.2 else BLUE for v in summary], height=0.6)
        ax.axvline(0, color=MUTED_INK, linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(METHODS)
        ax.set_xlim(-1, 1)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_title(title.replace(", ", "\n"), fontsize=8)
        ax.set_xlabel("Spearman with degree")
    fig.suptitle("Degree bias when nothing changes (orange: |rho| > 0.2)", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "multilayer_degree_bias.png"), dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--figures-only", action="store_true", help="redraw the figures from the saved results")
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.figures_only:
        figures(pd.read_csv(os.path.join(RESULTS_DIR, "multilayer.csv")))
        sys.exit()

    drift = population_drift()
    drift.to_csv(os.path.join(RESULTS_DIR, "multilayer_population_drift.csv"), index=False)
    print(drift.groupby(["scenario", "dimension", "embedding"])[["median_unchanged", "median_changed", "auroc"]]
          .mean().round(3).to_string())

    demo = run_demo()
    demo.to_csv(os.path.join(RESULTS_DIR, "multilayer_demo.csv"), index=False)
    print(demo.round(3).to_string(index=False))

    spikes = run_real_spike_ins()
    spikes.to_csv(os.path.join(RESULTS_DIR, "multilayer_locscn_spike_in.csv"), index=False)

    results = run(args.replicates, args.processes)
    results.to_csv(os.path.join(RESULTS_DIR, "multilayer.csv"), index=False)
    figures(results)

    runtime = run_runtime()
    runtime.to_csv(os.path.join(RESULTS_DIR, "multilayer_runtime.csv"), index=False)
    print(runtime.round(2).to_string(index=False))
