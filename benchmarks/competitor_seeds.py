"""Run-to-run variation of competitor rankings, compared with n2v2r.

Random-walk and neural-network embeddings give a different ranking every time
they are run with a new random seed. This benchmark measures how much, on the
same pair of networks, and compares it with n2v2r, whose only random choice is
the starting vector of the truncated SVD.

Methods (every ranking: higher score = more changed):

- n2v2r: the paper's default (UASE, dims 4-24, euclidean + cosine, Borda).
  The seed sets ARPACK's starting vector.
- node2vec (p = 1, q = 0.5) and DeepWalk (node2vec with p = q = 1), via
  pecanpy: each network is embedded separately (32 dims, 10 walks of length
  80 per node, window 10), the second embedding is rotated onto the first with
  an orthogonal Procrustes fit, and nodes are ranked by the Borda of their
  euclidean and cosine distances. DeepWalk is also the base embedding of iHerd
  (Duan et al. 2023).
- PLEX.I (Yousefi et al. 2023), ported from its R source (CRAN package PLEXI
  1.0, plexi_embedding_2layer + plexi_node_detection_2layer, with demo = FALSE
  so the network is actually trained): weighted random walks (100 per node, 5
  steps) give each node's visit probabilities; a shared encoder-decoder (input
  = adjacency row, 2-unit ReLU code, sigmoid output, MSE, Adam, batch 5, 10
  epochs) is trained on both networks and on two edge-weight-permuted null
  networks; nodes are ranked by the rank sum of their cosine distances over 50
  trainings (the package default). "PLEX.I, 1 training" uses only the first
  of those trainings.
- Deterministic references: degree difference (DeDi) and, where the expression
  is available, a DGCA-style score (McKenzie et al. 2016): the mean absolute
  z-difference of Fisher-transformed correlations with all other genes.

Data:
- dcsbm: degree-corrected SBM, 1,000 nodes, 4 blocks, 10% of nodes switch
  block (benchmarks/simulations.py).
- coexpression: 1,000 genes, 5 module factors, 150 samples per condition,
  |cor|^6, 10% of genes switch module (as in benchmarks/simulations.py, but
  keeping the expression for the DGCA-style score).
- hela: the HeLa G1 -> S hdWGCNA networks of benchmarks/cell_cycle.py (2,000
  genes). There is no gene-level truth; the AUROC column uses the genes of
  Reactome cell-cycle pathways only as context.

Run from the repository root, one dataset per process (sims: about 20 minutes
each; hela: about 40 minutes):

    OMP_NUM_THREADS=1 python benchmarks/competitor_seeds.py --dataset dcsbm
    OMP_NUM_THREADS=1 python benchmarks/competitor_seeds.py --dataset hela --revelio-dir path/to/Revelio/data
    python benchmarks/competitor_seeds.py --summarise
"""

import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from node2vec2rank.embedding import embed  # noqa: E402
from node2vec2rank.model_utils import borda_aggregate, compute_pairwise_distances  # noqa: E402

RESULTS_DIR = os.path.join(HERE, "results")
DIMS = list(range(4, 25, 2))
TOP_K = 100


# ---------------------------------------------------------------- data

def simulate_coexpression(rng, num_genes=1000, frac_changed=0.1, num_modules=5, num_samples=150,
                          frac_in_modules=0.7, soft_power=6):
    """benchmarks/simulations.py's co-expression simulator, also returning the
    expression of both conditions."""
    in_module = rng.random(num_genes) < frac_in_modules
    modules = np.where(in_module, rng.integers(0, num_modules, num_genes), -1)
    after = modules.copy()
    changed = np.zeros(num_genes, bool)
    moved = rng.choice(np.flatnonzero(in_module), int(round(frac_changed * num_genes)), replace=False)
    after[moved] = (modules[moved] + 1) % num_modules
    changed[moved] = True
    loadings = rng.uniform(0.3, 0.9, num_genes)
    graphs, expressions = [], []
    for z in (modules, after):
        factors = rng.standard_normal((num_samples, num_modules))
        expression = rng.standard_normal((num_samples, num_genes))
        member = z >= 0
        expression[:, member] = (loadings[member] * factors[:, z[member]]
                                 + np.sqrt(1 - loadings[member] ** 2) * expression[:, member])
        adjacency = np.abs(np.corrcoef(expression, rowvar=False)) ** soft_power
        np.fill_diagonal(adjacency, 0)
        graphs.append(adjacency)
        expressions.append(expression)
    return graphs, changed, expressions


def load(dataset, revelio_dir=None):
    """Returns (graphs, labels, expressions or None)."""
    if dataset == "dcsbm":
        from simulations import simulate_sbm
        graphs, changed = simulate_sbm(np.random.default_rng(1), frac_changed=0.1)
        return graphs, changed, None
    if dataset == "coexpression":
        return simulate_coexpression(np.random.default_rng(1))
    if dataset == "hela":
        import cell_cycle as cc
        from node2vec2rank.permutation import read_gmt
        from node2vec2rank.simulate import coexpression_network
        from node2vec2rank.singlecell import metacells
        cells_table, groups, batches, build = cc.prepare(revelio_dir)
        power = build.keywords["power"]
        expressions = [metacells(groups[phase]) for phase in ("G1", "S")]
        graphs = [np.asarray(coexpression_network(e, power=power, signed=True)) for e in expressions]
        genes = groups["G1"].columns
        reactome = read_gmt(cc.REACTOME)
        cell_cycle = set().union(*(m for name, m in reactome.items() if cc.CELL_CYCLE.search(name)))
        labels = np.asarray(genes.isin(list(cell_cycle)))
        return graphs, labels, [np.asarray(e) for e in expressions]
    raise ValueError(dataset)


# ---------------------------------------------------------------- methods

def n2v2r(graphs, seed):
    embeddings = embed(graphs, max(DIMS), "uase", random_state=seed)
    columns = [compute_pairwise_distances(embeddings[0][:, :d], embeddings[1][:, :d], m)
               for d in DIMS for m in ("euclidean", "cosine")]
    return borda_aggregate(np.column_stack(columns))


def degree_difference(graphs):
    return np.abs(graphs[0].sum(axis=0) - graphs[1].sum(axis=0))


def dgca_score(expressions):
    """Mean |z-difference| of Fisher-transformed correlations, per gene."""
    zs, ns = [], []
    for e in expressions:
        r = np.clip(np.corrcoef(e, rowvar=False), -0.999999, 0.999999)
        zs.append(np.arctanh(r))
        ns.append(len(e))
    zdiff = np.abs(zs[0] - zs[1]) / np.sqrt(1 / (ns[0] - 3) + 1 / (ns[1] - 3))
    np.fill_diagonal(zdiff, 0)
    return zdiff.sum(axis=1) / (len(zdiff) - 1)


def random_walk_embedding(graph, seed, p, q, dim=32):
    from pecanpy.pecanpy import DenseOTF
    nodes = [str(i) for i in range(len(graph))]
    model = DenseOTF.from_mat(np.asarray(graph, dtype=np.float32), nodes, p=p, q=q, workers=1,
                              random_state=seed)
    return np.asarray(model.embed(dim=dim, num_walks=10, walk_length=80, window_size=10), dtype=np.float64)


def walk_then_rank(graphs, seed, p, q):
    first = random_walk_embedding(graphs[0], seed, p, q)
    second = random_walk_embedding(graphs[1], seed + 10_000, p, q)
    rotation, _ = orthogonal_procrustes(second, first)
    second = second @ rotation
    columns = [compute_pairwise_distances(first, second, m) for m in ("euclidean", "cosine")]
    return borda_aggregate(np.column_stack(columns))


# PLEX.I port -------------------------------------------------------------

def plexi_walk_probabilities(adjacency, rng, walk_rep=100, n_steps=5, chunk=20_000):
    """rep_random_walk(): fraction of walks from each node that visit each
    node; the diagonal is set to 1."""
    n = len(adjacency)
    row_sums = adjacency.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cumulative = np.cumsum(adjacency / np.where(row_sums > 0, row_sums, 1)[:, None], axis=1)
    starts = np.repeat(np.arange(n), walk_rep)
    counts = np.zeros((n, n))
    for begin in range(0, len(starts), chunk):
        start = starts[begin:begin + chunk]
        position = start.copy()
        visits = np.empty((len(start), n_steps), dtype=np.int64)
        for step in range(n_steps):
            u = rng.random(len(position))
            following = np.minimum((cumulative[position] < u[:, None]).sum(axis=1), n - 1)
            stuck = row_sums[position] == 0  # isolated node: the walk stays put
            position = np.where(stuck, position, following)
            visits[:, step] = position
        walk = np.repeat(np.arange(len(start)), n_steps)
        pairs = np.unique(walk * n + visits.ravel())
        np.add.at(counts, (start[pairs // n], pairs % n), 1)
    probabilities = np.minimum(counts / walk_rep, 1)
    np.fill_diagonal(probabilities, 1)
    return probabilities


def plexi_train(x, y, rng, embedding_size=2, epochs=10, batch_size=5, lr=1e-3):
    """ednn(): Dense(2, relu) encoder, Dense(n, sigmoid) decoder, MSE, Adam
    (Keras defaults: Glorot-uniform weights, zero biases, beta 0.9 / 0.999,
    epsilon 1e-7), shuffled batches. Returns the codes of every row of x."""
    n_in, n_out = x.shape[1], y.shape[1]
    limit1, limit2 = np.sqrt(6 / (n_in + embedding_size)), np.sqrt(6 / (embedding_size + n_out))
    params = [rng.uniform(-limit1, limit1, (n_in, embedding_size)), np.zeros(embedding_size),
              rng.uniform(-limit2, limit2, (embedding_size, n_out)), np.zeros(n_out)]
    moments = [np.zeros_like(w) for w in params]
    velocities = [np.zeros_like(w) for w in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for begin in range(0, len(x), batch_size):
            batch = order[begin:begin + batch_size]
            xb, yb = x[batch], y[batch]
            pre = xb @ params[0] + params[1]
            code = np.maximum(pre, 0)
            out = 1 / (1 + np.exp(-(code @ params[2] + params[3])))
            d_out = 2 * (out - yb) / yb.size * out * (1 - out)
            d_code = (d_out @ params[2].T) * (pre > 0)
            grads = [xb.T @ d_code, d_code.sum(axis=0), code.T @ d_out, d_out.sum(axis=0)]
            t += 1
            for i, g in enumerate(grads):
                moments[i] = 0.9 * moments[i] + 0.1 * g
                velocities[i] = 0.999 * velocities[i] + 0.001 * g * g
                m_hat = moments[i] / (1 - 0.9 ** t)
                v_hat = velocities[i] / (1 - 0.999 ** t)
                params[i] -= lr * m_hat / (np.sqrt(v_hat) + 1e-7)
    return np.maximum(x @ params[0] + params[1], 0)


def plexi(graphs, seed, train_rep=50):
    """Returns (rank sum over train_rep trainings, ranking from the first
    training only)."""
    rng = np.random.default_rng(seed)
    n = len(graphs[0])
    first, second = (np.asarray(g, dtype=np.float64) for g in graphs)
    # null.perm = TRUE: two more networks whose edge weights are resampled
    # with replacement over the edge list (the union of edges of both graphs)
    iu = np.triu_indices(n, 1)
    edges = (first[iu] != 0) | (second[iu] != 0)
    layers = [first, second]
    for graph in (first, second):
        weights = graph[iu][edges]
        permuted = np.zeros((n, n))
        rows, cols = iu[0][edges], iu[1][edges]
        permuted[rows, cols] = rng.choice(weights, len(weights), replace=True)
        layers.append(permuted + permuted.T)
    x = np.vstack(layers)
    y = np.vstack([plexi_walk_probabilities(layer, rng) for layer in layers])
    rank_sum = np.zeros(n)
    single = None
    for rep in range(train_rep):
        codes = plexi_train(x, y, rng)
        a, b = codes[:n], codes[n:2 * n]
        cosine = 1 - (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1) + 1e-9)
        ranks = rankdata(cosine)
        rank_sum += ranks
        if rep == 0:
            single = ranks
    return rank_sum, single


# ---------------------------------------------------------------- metrics

def auroc(labels, scores):
    ranks = rankdata(scores)
    positives = labels.sum()
    return (ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * (len(labels) - positives))


def top_overlap(a, b, k=TOP_K):
    return len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k])) / k


def agreement(runs):
    """Mean and min over all pairs of runs of Spearman and top-k overlap."""
    rhos, overlaps = [], []
    for i, j in itertools.combinations(range(len(runs)), 2):
        rhos.append(spearmanr(runs[i], runs[j])[0])
        overlaps.append(top_overlap(runs[i], runs[j]))
    return np.mean(rhos), np.min(rhos), np.mean(overlaps), np.min(overlaps)


def averaging_curve(runs, sizes=(1, 2, 5, 10)):
    """Agreement between the mean ranks of two disjoint sets of m runs."""
    ranks = np.array([rankdata(r) for r in runs])
    rows = []
    for m in sizes:
        if 2 * m > len(ranks):
            break
        a, b = ranks[:m].mean(axis=0), ranks[m:2 * m].mean(axis=0)
        rows.append(dict(runs_averaged=m, spearman=spearmanr(a, b)[0], top_overlap=top_overlap(a, b)))
    return rows


# ---------------------------------------------------------------- main

def run(dataset, revelio_dir, num_runs, plexi_runs):
    graphs, labels, expressions = load(dataset, revelio_dir)
    print(f"{dataset}: {len(labels)} nodes, {labels.sum()} labelled", flush=True)
    stochastic = {
        "n2v2r (default)": lambda s: n2v2r(graphs, s),
        "node2vec + Procrustes": lambda s: walk_then_rank(graphs, s, p=1, q=0.5),
        "DeepWalk + Procrustes": lambda s: walk_then_rank(graphs, s, p=1, q=1),
    }
    deterministic = {"DeDi": lambda: degree_difference(graphs)}
    if expressions is not None:
        deterministic["DGCA-style"] = lambda: dgca_score(expressions)

    scores, times = {}, {}
    for name, method in stochastic.items():
        scores[name], times[name] = [], []
        for seed in range(num_runs):
            begin = time.perf_counter()
            scores[name].append(method(seed))
            times[name].append(time.perf_counter() - begin)
        print(f"  {name}: {np.mean(times[name]):.1f} s per run", flush=True)
    scores["PLEX.I (50 trainings)"], scores["PLEX.I (1 training)"] = [], []
    times["PLEX.I (50 trainings)"], times["PLEX.I (1 training)"] = [], []
    for seed in range(plexi_runs):
        begin = time.perf_counter()
        full, single = plexi(graphs, seed)
        elapsed = time.perf_counter() - begin
        scores["PLEX.I (50 trainings)"].append(full)
        scores["PLEX.I (1 training)"].append(single)
        times["PLEX.I (50 trainings)"].append(elapsed)
        times["PLEX.I (1 training)"].append(elapsed / 50)
        print(f"  PLEX.I run {seed}: {elapsed:.0f} s", flush=True)
    for name, method in deterministic.items():
        begin = time.perf_counter()
        scores[name] = [method(), method()]
        times[name] = [(time.perf_counter() - begin) / 2]

    # same seed twice must give the same ranking (reproducible, but seed-dependent)
    same_seed = {
        "n2v2r (default)": np.abs(n2v2r(graphs, 0) - scores["n2v2r (default)"][0]).max(),
        "node2vec + Procrustes": np.abs(walk_then_rank(graphs, 0, 1, 0.5)
                                        - scores["node2vec + Procrustes"][0]).max(),
    }

    summary, curves = [], []
    for name, runs in scores.items():
        rho_mean, rho_min, ov_mean, ov_min = agreement(runs)
        aurocs = [auroc(labels, r) for r in runs]
        summary.append(dict(dataset=dataset, method=name, runs=len(runs), spearman_mean=rho_mean,
                            spearman_min=rho_min, top100_overlap_mean=ov_mean, top100_overlap_min=ov_min,
                            auroc_mean=np.mean(aurocs), auroc_min=np.min(aurocs), auroc_max=np.max(aurocs),
                            labelled_in_top100_min=min(int(labels[np.argsort(-r)[:TOP_K]].sum()) for r in runs),
                            labelled_in_top100_max=max(int(labels[np.argsort(-r)[:TOP_K]].sum()) for r in runs),
                            seconds_per_run=np.mean(times[name]),
                            same_seed_max_difference=same_seed.get(name, np.nan)))
        for row in averaging_curve(runs):
            curves.append(dict(dataset=dataset, method=name, **row))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(summary).to_csv(os.path.join(RESULTS_DIR, f"competitor_seeds_{dataset}.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(RESULTS_DIR, f"competitor_seeds_{dataset}_averaging.csv"),
                                index=False)
    print(pd.DataFrame(summary).round(3).to_string(), flush=True)
    print(pd.DataFrame(curves).round(3).to_string(), flush=True)


def summarise():
    frames = [pd.read_csv(os.path.join(RESULTS_DIR, f)) for f in sorted(os.listdir(RESULTS_DIR))
              if f.startswith("competitor_seeds_") and not f.endswith("_averaging.csv")]
    print(pd.concat(frames).round(3).to_string())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["dcsbm", "coexpression", "hela"])
    parser.add_argument("--revelio-dir")
    parser.add_argument("--runs", type=int, default=20, help="seeds per random method")
    parser.add_argument("--plexi-runs", type=int, default=10, help="PLEX.I calls (50 trainings each)")
    parser.add_argument("--summarise", action="store_true")
    args = parser.parse_args(argv)
    if args.summarise:
        summarise()
    else:
        run(args.dataset, args.revelio_dir, args.runs, args.plexi_runs)


if __name__ == "__main__":
    main()
