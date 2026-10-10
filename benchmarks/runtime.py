"""Runtime and peak memory of n2v2r (paper default: joint UASE at dimension 24,
dimensions 4..24 x euclidean and cosine, Borda) and of degree difference.

Networks are block-model graphs (8 blocks, mean degree about 100) with
n in {2,000, 5,000, 10,000, 20,000} nodes and K in {2, 4} networks, stored as
sparse matrices, and as dense matrices up to 10,000 nodes (four dense
20,000-node networks alone take 12.8 GB, about the memory of this machine).
Every setting runs in a fresh process; the peak memory is that process's peak
resident set size, input networks included. For K = 4, n2v2r ranks the three
consecutive pairs from one joint embedding.

    python benchmarks/runtime.py           # writes benchmarks/results/runtime.csv
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

SIZES = (2_000, 5_000, 10_000, 20_000)
NUM_GRAPHS = (2, 4)
MAX_DENSE = 10_000
DIMS = list(range(4, 25, 2))


def block_graphs(n, num_graphs, rng, num_blocks=8, mean_degree=100, frac_in=0.6, frac_changed=0.1):
    blocks = rng.integers(0, num_blocks, n)
    members = [np.flatnonzero(blocks == b) for b in range(num_blocks)]
    graphs = []
    for _ in range(num_graphs):
        moved = rng.choice(n, int(frac_changed * n), replace=False)
        current = blocks.copy()
        current[moved] = (current[moved] + 1) % num_blocks
        members = [np.flatnonzero(current == b) for b in range(num_blocks)]
        num_edges = n * mean_degree // 2
        sources = rng.integers(0, n, num_edges)
        inside = rng.random(num_edges) < frac_in
        targets = rng.integers(0, n, num_edges)
        for b in range(num_blocks):
            pick = inside & (current[sources] == b)
            targets[pick] = rng.choice(members[b], pick.sum())
        keep = sources != targets
        upper = sparse.coo_matrix((np.ones(keep.sum()), (sources[keep], targets[keep])), shape=(n, n)).tocsr()
        graph = upper + upper.T
        graph.data[:] = 1.0
        graphs.append(graph)
    return graphs


def one(n, num_graphs, storage, method):
    from node2vec2rank.embedding import embed
    from node2vec2rank.model_utils import borda_aggregate, compute_pairwise_distances
    graphs = block_graphs(n, num_graphs, np.random.default_rng(0))
    if storage == "dense":
        graphs = [g.toarray() for g in graphs]
    begin = time.perf_counter()
    if method == "n2v2r":
        embeddings = embed(graphs, max(DIMS), "uase", random_state=42)
        for i in range(num_graphs - 1):
            borda_aggregate(np.column_stack([compute_pairwise_distances(embeddings[i][:, :d],
                                                                        embeddings[i + 1][:, :d], m)
                                             for d in DIMS for m in ("euclidean", "cosine")]))
    else:
        for i in range(num_graphs - 1):
            np.abs(np.asarray(graphs[i].sum(axis=0)).ravel() - np.asarray(graphs[i + 1].sum(axis=0)).ravel())
    seconds = time.perf_counter() - begin
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2  # kB -> GB
    print(json.dumps(dict(nodes=n, networks=num_graphs, storage=storage, method=method, seconds=seconds,
                          peak_memory_gb=peak)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--one", nargs=4, metavar=("N", "K", "STORAGE", "METHOD"))
    args = parser.parse_args(argv)
    if args.one:
        n, k, storage, method = args.one
        one(int(n), int(k), storage, method)
        return
    rows = []
    for n in SIZES:
        for k in NUM_GRAPHS:
            for storage in ("sparse", "dense"):
                if storage == "dense" and n > MAX_DENSE:
                    continue
                for method in ("n2v2r", "DeDi"):
                    out = subprocess.run([sys.executable, __file__, "--one", str(n), str(k), storage, method],
                                         capture_output=True, text=True)
                    if out.returncode != 0:
                        print(out.stderr[-2000:], flush=True)
                        continue
                    rows.append(json.loads(out.stdout.strip().splitlines()[-1]))
                    print(rows[-1], flush=True)
    result = pd.DataFrame(rows)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    result.to_csv(os.path.join(HERE, "results", "runtime.csv"), index=False)
    print(result.round(2).to_string())


if __name__ == "__main__":
    main()
