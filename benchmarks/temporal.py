"""Ordered conditions (time, phases): what the sequential comparison can and cannot tell.

Simulates co-expression networks over K ordered conditions in which genes follow
known trajectories between modules, and asks four questions:

1. per-step detection: does a state-space smoother (random-walk prior on the
   UASE positions, Kalman/RTS) find the genes that change at a step better than
   the paper's sequential pairwise distance?
2. direction: can the direction of a change (which module a gene leaves, which
   it joins) be read off the joint embedding, and how does that compare with
   the raw networks and with the sign of the degree difference?
3. trajectory shape: can a persistent change be told apart from a transient one
   (a gene that leaves and comes back) from the sequential distances alone?
4. a noisy condition (fewer samples): what happens to the steps next to it?

Run from the repository root:

    python benchmarks/temporal.py        # writes benchmarks/results/temporal_*.csv

Model
-----
Every gene belongs to one of ``num_modules`` modules or to none; module genes
have expression ``loading * factor + noise`` as in
:func:`node2vec2rank.simulate.simulate_expression`. Along the K conditions a
gene is

- stable: same module throughout (or no module),
- persistent: moves from module a to module b at a random step and stays,
- transient: is in module b at one interior condition only, a elsewhere,
- gradual: mixes a and b with weight t / (K - 1) on b at condition t, so it
  changes a little at every step.

Networks are |cor|^6 (WGCNA-style).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from node2vec2rank.model_utils import borda_aggregate, compute_pairwise_distances  # noqa: E402
from node2vec2rank.significance import empirical_null_test  # noqa: E402
from node2vec2rank.simulate import coexpression_network  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DIMS = [4, 6, 8, 10, 12]
METRICS = ("euclidean", "cosine")
TYPES = ("stable", "persistent", "transient", "gradual")


def auroc(score, label):
    label = np.asarray(label, bool)
    if label.all() or not label.any():
        return np.nan
    ranks = rankdata(score)
    n_pos = label.sum()
    return (ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (~label).sum())


# ---------------------------------------------------------------- simulation

def simulate_trajectories(num_conditions, num_genes=600, num_modules=5, samples=100,
                          frac_in_modules=0.7, frac_each=0.05, loading_range=(0.4, 0.9), rng=None):
    """Returns (expression list, weights array K x n x (M+1), gene type, (a, b) modules, step)."""
    K, M = num_conditions, num_modules
    samples = [samples] * K if np.isscalar(samples) else list(samples)
    in_module = rng.random(num_genes) < frac_in_modules
    home = np.where(in_module, rng.integers(0, M, num_genes), M)  # M = no module
    kind = np.full(num_genes, "stable", dtype=object)
    target = home.copy()
    step = np.full(num_genes, -1)
    pool = np.flatnonzero(in_module)
    rng.shuffle(pool)
    count = int(frac_each * num_genes)
    for j, name in enumerate(("persistent", "transient", "gradual")):
        chosen = pool[j * count:(j + 1) * count]
        kind[chosen] = name
        target[chosen] = (home[chosen] + rng.integers(1, M, count)) % M
    # weight on the target module per condition
    weight = np.zeros((K, num_genes))
    for i in np.flatnonzero(kind == "persistent"):
        step[i] = rng.integers(1, K)          # changes between step-1 and step
        weight[step[i]:, i] = 1
    for i in np.flatnonzero(kind == "transient"):
        step[i] = rng.integers(1, K - 1)      # in b only at condition `step`
        weight[step[i], i] = 1
    weight[:, kind == "gradual"] = (np.arange(K) / (K - 1))[:, None]

    loadings = rng.uniform(*loading_range, num_genes)
    expression = []
    for t in range(K):
        factors = rng.standard_normal((samples[t], M + 1))
        factors[:, M] = 0  # "no module"
        noise = rng.standard_normal((samples[t], num_genes))
        signal = (np.sqrt(1 - weight[t]) * factors[:, home] + np.sqrt(weight[t]) * factors[:, target])
        values = loadings * signal + np.sqrt(1 - loadings ** 2) * noise
        values[:, home == M] = noise[:, home == M]  # genes without a module are pure noise
        expression.append(values)
    return expression, weight, kind, home, target, step, loadings


def changed_at_step(weight, kind, K):
    """Boolean (K-1) x n: did the gene's membership change between t and t+1."""
    return np.abs(np.diff(weight, axis=0)) > 0


# ---------------------------------------------------------------- embedding

def uase_both(graphs, d):
    """Left (shared, n x d) and right (K x n x d) UASE embeddings."""
    K, n = len(graphs), graphs[0].shape[0]
    unfolded = np.hstack(graphs)
    u, s, vt = np.linalg.svd(unfolded, full_matrices=False)
    left = u[:, :d] * np.sqrt(s[:d])
    right = (vt[:d].T * np.sqrt(s[:d])).reshape(K, n, d)
    return left, right


def sequential_borda(right, t):
    """The paper's per-step ranking: Borda over dims x metrics of distance between t and t+1."""
    columns = []
    for dim in DIMS:
        for metric in METRICS:
            columns.append(compute_pairwise_distances(right[t, :, :dim], right[t + 1, :, :dim], metric))
    return borda_aggregate(np.column_stack(columns)), np.column_stack(columns)


def rts_smooth(obs, r, q):
    """Random-walk Kalman filter + RTS smoother, independently per gene and dim.

    obs: K x n x d observations; r: observation noise variance; q: process variance.
    """
    K = obs.shape[0]
    m = np.empty_like(obs)
    p = np.empty(K)
    m[0], p[0] = obs[0], r
    pred_p = np.empty(K)
    for t in range(1, K):
        pred_p[t] = p[t - 1] + q
        gain = pred_p[t] / (pred_p[t] + r)
        m[t] = m[t - 1] + gain * (obs[t] - m[t - 1])
        p[t] = (1 - gain) * pred_p[t]
    s = m.copy()
    for t in range(K - 2, -1, -1):
        c = p[t] / pred_p[t + 1]
        s[t] = m[t] + c * (s[t + 1] - m[t])
    return s


def moment_noise(right):
    """Method-of-moments (r, q) for a random walk observed with noise, from
    median lag-1 and lag-2 squared differences per dimension."""
    K, _, d = right.shape
    lag1 = np.median(np.sum(np.diff(right, axis=0) ** 2, axis=2)) / d
    if K < 3:
        return lag1 / 2, 0.0
    lag2 = np.median(np.sum((right[2:] - right[:-2]) ** 2, axis=2)) / d
    q = max(lag2 - lag1, 0.0)
    r = max((2 * lag1 - lag2) / 2, 1e-12)
    return r, q


# ---------------------------------------------------------------- experiments

def run_replicate(K, samples, seed, d=12):
    rng = np.random.default_rng(seed)
    expression, weight, kind, home, target, step, loadings = simulate_trajectories(
        K, samples=samples, rng=rng)
    graphs = [coexpression_network(x) for x in expression]
    left, right = uase_both(graphs, d)
    strong = loadings >= 0.6  # weaker genes barely co-express under |cor|^6
    changed = changed_at_step(weight, kind, K)
    rows = []

    # 1. per-step detection: sequential (paper) vs smoothed positions
    r_mm, q_mm = moment_noise(right)
    smoothed = {"moments": rts_smooth(right, r_mm, q_mm)}
    for ratio in (0.1, 1.0, 10.0):  # q / r grid: the smoother's best case
        smoothed[f"q/r={ratio}"] = rts_smooth(right, r_mm, ratio * r_mm)
    for t in range(K - 1):
        borda, _ = sequential_borda(right, t)
        label = changed[t]
        row = dict(K=K, samples=str(samples), seed=seed, step=t + 1,
                   sequential=auroc(borda, label))
        # the same comparison with only the two conditions embedded
        _, pair = uase_both(graphs[t:t + 2], d)
        row["sequential, pair-only UASE"] = auroc(sequential_borda(pair, 0)[0], label)
        for name, sm in smoothed.items():
            row[f"smoothed {name}"] = auroc(sequential_borda(sm, t)[0], label)
        for gene_type in ("persistent", "transient", "gradual"):
            sel = label & (kind == gene_type)
            if sel.any():
                row[f"sequential auroc {gene_type}"] = auroc(borda[sel | ~label], sel[sel | ~label])
        rows.append(row)
    detection = pd.DataFrame(rows)

    # 2. direction of change: module affinity change for persistent switches at their step
    modules = [np.flatnonzero((home == m) & (kind == "stable")) for m in range(home.max())]
    direction_rows = []
    degree = [np.abs(g).sum(axis=0) for g in graphs]
    for i in np.flatnonzero((kind == "persistent") & strong):
        t0, t1 = step[i] - 1, step[i]
        delta = right[t1, i] - right[t0, i]
        emb = np.array([left[m].mean(axis=0) @ delta for m in modules])
        raw = np.array([graphs[t1][i, m].mean() - graphs[t0][i, m].mean() for m in modules])
        direction_rows.append(dict(
            K=K, samples=str(samples), seed=seed,
            embedding_joined=np.argmax(emb) == target[i], embedding_left=np.argmin(emb) == home[i],
            raw_joined=np.argmax(raw) == target[i], raw_left=np.argmin(raw) == home[i],
            degree_difference_sign=np.sign(degree[t1][i] - degree[t0][i])))
    direction = pd.DataFrame(direction_rows)

    # 3. trajectory shape: persistent vs transient from the sequential distances
    dist = np.stack([sequential_borda(right, t)[1][:, DIMS.index(d) * len(METRICS)]
                     for t in range(K - 1)])  # euclidean at the largest dim, (K-1) x n
    path = dist.sum(axis=0)
    net = np.linalg.norm(right[-1] - right[0], axis=1)
    sel = np.isin(kind, ("persistent", "transient")) & strong
    persistent = (kind == "persistent") & strong
    # a change point: the step with the largest move
    located = np.argmax(dist, axis=0) + 1
    shape = dict(K=K, samples=str(samples), seed=seed,
                 straightness_auroc=auroc((net / path)[sel], kind[sel] == "persistent"),
                 net_displacement_auroc=auroc(net[sel], kind[sel] == "persistent"),
                 changepoint_accuracy=np.mean(located[persistent] == step[persistent]))
    return detection, direction, pd.DataFrame([shape])


def noisy_condition(K=4, seeds=range(10), low=30):
    """Condition 3 of K has few samples: stable genes' distances next to it,
    against a control in which every condition has 100 samples."""
    rows = []
    for seed, setting in [(s, x) for s in seeds for x in ("control", f"condition 3 has {low} samples")]:
        samples = [100] * K
        if setting != "control":
            samples[2] = low
        rng = np.random.default_rng(seed)
        expression, weight, kind, home, target, step, _ = simulate_trajectories(
            K, samples=samples, rng=rng)
        graphs = [coexpression_network(x) for x in expression]
        _, right = uase_both(graphs, 12)
        stable = kind == "stable"
        changed = changed_at_step(weight, kind, K)
        degree_mean = np.mean([np.abs(g).sum(axis=0) for g in graphs], axis=0)
        for t in range(K - 1):
            _, columns = sequential_borda(right, t)
            z, _, q = empirical_null_test(columns, degree_mean)
            rows.append(dict(seed=seed, setting=setting, step=f"{t + 1}->{t + 2}",
                             median_stable_distance=np.median(columns[stable, 0]),
                             false_calls=int(np.sum((q < 0.1) & ~changed[t])),
                             true_calls=int(np.sum((q < 0.1) & changed[t]))))
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=10)
    args = parser.parse_args(argv)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    detection, direction, shape = [], [], []
    for K in (4, 8):
        for samples in (25, 50, 100):
            for seed in range(args.replicates):
                a, b, c = run_replicate(K, samples, seed)
                detection.append(a)
                direction.append(b)
                shape.append(c)
    detection = pd.concat(detection)
    direction = pd.concat(direction)
    shape = pd.concat(shape)
    noisy = noisy_condition(seeds=range(args.replicates))

    detection.to_csv(os.path.join(RESULTS_DIR, "temporal_detection.csv"), index=False)
    direction.to_csv(os.path.join(RESULTS_DIR, "temporal_direction.csv"), index=False)
    shape.to_csv(os.path.join(RESULTS_DIR, "temporal_shape.csv"), index=False)
    noisy.to_csv(os.path.join(RESULTS_DIR, "temporal_noisy_condition.csv"), index=False)

    pd.set_option("display.width", 200)
    print("Per-step detection AUROC (mean over steps and replicates)")
    print(detection.drop(columns=["seed", "step"]).groupby(["K", "samples"]).mean().round(3).T)
    print("\nDirection of persistent switches, loading >= 0.6 (fraction correct)")
    summary = direction.groupby(["K", "samples"]).agg(
        embedding_joined=("embedding_joined", "mean"), embedding_left=("embedding_left", "mean"),
        raw_joined=("raw_joined", "mean"), raw_left=("raw_left", "mean"),
        degree_up=("degree_difference_sign", lambda s: np.mean(s > 0)))
    print(summary.round(3))
    print("\nTrajectory shape (genes with loading >= 0.6)")
    print(shape.drop(columns="seed").groupby(["K", "samples"]).mean().round(3))
    print("\nNoisy condition 3 (30 samples, others 100)")
    print(noisy.drop(columns="seed").groupby(["setting", "step"]).mean().round(3))


if __name__ == "__main__":
    main()
